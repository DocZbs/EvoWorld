"""Strict TOML configuration loading for reproducible EvoWorld runs."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Mapping, cast

from evoworld.contracts import ExpertSpec, GroundingMode, JsonValue

_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "seed",
        "run_root",
        "mode",
        "grounding_mode",
        "grounding_temperature",
        "router_weight",
        "reliability_threshold",
        "rollout_horizon",
        "executor",
        "experts",
    }
)
_EXPERT_KEYS = frozenset(
    {"expert_id", "backend", "task_families", "checkpoint", "action_schema", "enabled"}
)


class ConfigError(ValueError):
    """Raised when a configuration cannot describe a safe, reproducible run."""


@dataclass(frozen=True, slots=True)
class EvoWorldConfig:
    """All behavior settings required by the lightweight EvoWorld core."""

    schema_version: int
    seed: int
    run_root: Path
    mode: str
    grounding_mode: GroundingMode
    grounding_temperature: float
    router_weight: float
    reliability_threshold: float
    rollout_horizon: int
    executor: str | None
    experts: tuple[ExpertSpec, ...]


def load_config(path: str | Path) -> EvoWorldConfig:
    """Load and validate TOML, resolving every configured path from its directory."""
    config_path = Path(path).resolve(strict=True)
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"invalid TOML in {config_path}: {error}") from error
    config = _parse_config(raw, config_path.parent)
    validate_config(config)
    return config


def config_from_resolved_manifest(manifest: Mapping[str, object]) -> EvoWorldConfig:
    """Restore a validated config from the durable, already-resolved run manifest."""
    expected = {
        "schema_version",
        "seed",
        "run_root",
        "mode",
        "grounding_mode",
        "grounding_temperature",
        "router_weight",
        "reliability_threshold",
        "rollout_horizon",
        "executor",
        "experts",
        "settings",
    }
    if set(manifest) != expected:
        raise ConfigError("resolved manifest fields do not match schema version 1")
    experts_raw = manifest.get("experts")
    if not isinstance(experts_raw, list):
        raise ConfigError("resolved manifest experts must be an array")
    experts = tuple(_expert_from_manifest(item) for item in experts_raw)
    try:
        grounding_mode = GroundingMode(_required_text(manifest, "grounding_mode"))
    except ValueError as error:
        raise ConfigError("resolved manifest grounding_mode is unsupported") from error
    run_root = _absolute_manifest_path(manifest.get("run_root"), "run_root")
    executor = _optional_text(manifest, "executor")
    settings = manifest.get("settings")
    if not isinstance(settings, dict):
        raise ConfigError("resolved manifest settings must be an object")
    config = EvoWorldConfig(
        schema_version=_required_int(manifest, "schema_version"),
        seed=_required_int(manifest, "seed"),
        run_root=run_root,
        mode=_required_text(manifest, "mode"),
        grounding_mode=grounding_mode,
        grounding_temperature=_required_float(manifest, "grounding_temperature"),
        router_weight=_required_float(manifest, "router_weight"),
        reliability_threshold=_required_float(manifest, "reliability_threshold"),
        rollout_horizon=_required_int(manifest, "rollout_horizon"),
        executor=executor,
        experts=experts,
    )
    validate_config(config)
    return config


def validate_config(config: EvoWorldConfig) -> None:
    """Reject incomplete settings before a run can write artifacts or call a backend."""
    if config.schema_version != 1:
        raise ConfigError("schema_version must be 1")
    if config.mode not in {"dry_run", "real"}:
        raise ConfigError("mode must be 'dry_run' or 'real'")
    if config.mode == "real" and not config.executor:
        raise ConfigError("real mode requires an explicit executor")
    for value, name in (
        (config.grounding_temperature, "grounding_temperature"),
        (config.router_weight, "router_weight"),
        (config.reliability_threshold, "reliability_threshold"),
    ):
        _validate_finite_float(value, name)
    if config.grounding_temperature <= 0:
        raise ConfigError("grounding_temperature must be positive")
    for value, name in (
        (config.router_weight, "router_weight"),
        (config.reliability_threshold, "reliability_threshold"),
    ):
        if not 0.0 <= value <= 1.0:
            raise ConfigError(f"{name} must be between 0 and 1")
    if config.rollout_horizon <= 0:
        raise ConfigError("rollout_horizon must be positive")
    if not config.experts:
        raise ConfigError("at least one expert must be enabled")
    expert_ids = [expert.expert_id for expert in config.experts]
    if len(expert_ids) != len(set(expert_ids)):
        raise ConfigError("enabled expert_id values must be unique")
    action_schemas = {expert.action_schema for expert in config.experts}
    if len(action_schemas) != 1:
        raise ConfigError("enabled experts must use one compatible action_schema")


def resolved_manifest(
    config: EvoWorldConfig, settings: Mapping[str, JsonValue] | None = None
) -> dict[str, JsonValue]:
    """Return an artifact-safe, JSON-compatible record of resolved behavior settings."""
    expert_records: list[JsonValue] = []
    for expert in config.experts:
        record = asdict(expert)
        checkpoint = record["checkpoint"]
        record["checkpoint"] = str(checkpoint) if checkpoint is not None else None
        expert_records.append(cast(JsonValue, record))
    return {
        "schema_version": config.schema_version,
        "seed": config.seed,
        "run_root": str(config.run_root),
        "mode": config.mode,
        "grounding_mode": config.grounding_mode.value,
        "grounding_temperature": config.grounding_temperature,
        "router_weight": config.router_weight,
        "reliability_threshold": config.reliability_threshold,
        "rollout_horizon": config.rollout_horizon,
        "executor": config.executor,
        "experts": expert_records,
        "settings": _redact_settings(settings or {}),
    }


def config_digest(config: EvoWorldConfig, settings: Mapping[str, JsonValue] | None = None) -> str:
    """Hash the canonical, secret-redacted configuration manifest."""
    validate_config(config)
    try:
        payload = json.dumps(
            resolved_manifest(config, settings),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    except (TypeError, ValueError) as error:
        raise ConfigError("resolved configuration must contain finite canonical JSON") from error
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_config(raw: Mapping[str, object], base_dir: Path) -> EvoWorldConfig:
    _reject_unknown_fields(raw, _ROOT_KEYS, "root")
    experts_raw = raw.get("experts", [])
    if not isinstance(experts_raw, list):
        raise ConfigError("experts must be an array of tables")
    experts = tuple(_parse_expert(row, base_dir) for row in experts_raw if _enabled_expert(row))
    executor = _optional_text(raw, "executor")
    try:
        grounding_mode = GroundingMode(_required_text(raw, "grounding_mode"))
    except ValueError as error:
        raise ConfigError("grounding_mode must be 'nearest' or 'softmax'") from error
    return EvoWorldConfig(
        schema_version=_required_int(raw, "schema_version"),
        seed=_required_int(raw, "seed"),
        run_root=_resolve_path(_required_text(raw, "run_root"), base_dir),
        mode=_required_text(raw, "mode"),
        grounding_mode=grounding_mode,
        grounding_temperature=_required_float(raw, "grounding_temperature"),
        router_weight=_required_float(raw, "router_weight"),
        reliability_threshold=_required_float(raw, "reliability_threshold"),
        rollout_horizon=_required_int(raw, "rollout_horizon"),
        executor=executor,
        experts=experts,
    )


def _enabled_expert(row: object) -> bool:
    if not isinstance(row, dict):
        raise ConfigError("each expert must be a table")
    _reject_unknown_fields(row, _EXPERT_KEYS, "expert")
    enabled = row.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError("expert enabled must be boolean")
    return enabled


def _parse_expert(row: object, base_dir: Path) -> ExpertSpec:
    if not isinstance(row, dict):
        raise ConfigError("each expert must be a table")
    task_families = row.get("task_families")
    if not isinstance(task_families, list) or not all(
        isinstance(item, str) and item for item in task_families
    ):
        raise ConfigError("expert task_families must be a non-empty array of strings")
    checkpoint = _optional_text(row, "checkpoint")
    try:
        return ExpertSpec(
            expert_id=_required_text(row, "expert_id"),
            backend=_required_text(row, "backend"),
            task_families=tuple(task_families),
            checkpoint=_resolve_path(checkpoint, base_dir) if checkpoint else None,
            action_schema=_required_text(row, "action_schema"),
        )
    except ValueError as error:
        raise ConfigError(str(error)) from error


def _expert_from_manifest(row: object) -> ExpertSpec:
    if not isinstance(row, dict):
        raise ConfigError("resolved manifest expert must be an object")
    expected = {"expert_id", "backend", "task_families", "checkpoint", "action_schema"}
    if set(row) != expected:
        raise ConfigError("resolved manifest expert fields do not match schema version 1")
    task_families = row.get("task_families")
    if not isinstance(task_families, list) or not all(
        isinstance(item, str) and item for item in task_families
    ):
        raise ConfigError("resolved manifest task_families must be non-empty strings")
    checkpoint_value = row.get("checkpoint")
    checkpoint = (
        None
        if checkpoint_value is None
        else _absolute_manifest_path(checkpoint_value, "checkpoint")
    )
    try:
        return ExpertSpec(
            expert_id=_required_text(row, "expert_id"),
            backend=_required_text(row, "backend"),
            task_families=tuple(task_families),
            checkpoint=checkpoint,
            action_schema=_required_text(row, "action_schema"),
        )
    except ValueError as error:
        raise ConfigError(str(error)) from error


def _reject_unknown_fields(raw: Mapping[str, object], allowed: frozenset[str], scope: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"unknown {scope} field(s): {', '.join(unknown)}")


def _required_text(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be a non-empty string")
    return value


def _optional_text(raw: Mapping[str, object], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{key} must be a string")
    return value.strip() or None


def _required_int(raw: Mapping[str, object], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{key} must be an integer")
    return value


def _required_float(raw: Mapping[str, object], key: str) -> float:
    value = raw.get(key)
    _validate_finite_float(value, key)
    return float(cast(int | float, value))


def _validate_finite_float(value: object, key: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value):
        raise ConfigError(f"{key} must be a finite number")


def _resolve_path(value: str, base_dir: Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (base_dir / candidate).resolve()


def _absolute_manifest_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"resolved manifest {name} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        raise ConfigError(f"resolved manifest {name} must be absolute")
    return path.resolve()


def _redact_settings(value: JsonValue | Mapping[str, JsonValue]) -> JsonValue:
    if isinstance(value, Mapping):
        return {
            key: "<redacted>" if _is_secret_key(key) else _redact_settings(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_settings(item) for item in value]
    return value


def _is_secret_key(key: str) -> bool:
    parts = {part.lower() for part in key.replace("-", "_").split("_")}
    return bool(parts & {"secret", "token", "password"}) or key.lower().endswith("key")

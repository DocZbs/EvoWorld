"""Fail-closed composition of the real EvoWorld process adapters.

The public repository owns the audited backend boundaries.  Deployment-specific
task, robot, scoring, training, memory, and storage ports stay behind one explicit
importable provider named by a strict runtime manifest.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import cast

from evoworld.adapters.ctrl_world import (
    CtrlWorldAdapter,
    CtrlWorldAssets,
    build_ctrl_world_process_bridge,
)
from evoworld.adapters.content import (
    combined_manifest,
    directory_manifest,
    file_manifest,
)
from evoworld.adapters.groot import GrootPolicyAdapter, build_groot_process_bridge
from evoworld.adapters.openpi import OpenPiPolicyAdapter, build_openpi_process_bridge
from evoworld.adapters.projection import (
    AFFINE_IMPLEMENTATION,
    AffineActionProjector,
    PROJECTOR_SOURCE_PATH,
    ProjectedExpertPolicy,
    ProjectionError,
    ProjectorIdentity,
    load_affine_projector,
    projector_content_digest,
)
from evoworld.adapters.qwen_vl import (
    DEFAULT_QWEN_MODEL,
    QwenVlEvaluator,
    build_qwen_process_bridge,
)
from evoworld.codec import canonical_json, decode_json_object, input_digest
from evoworld.config import (
    EvoWorldConfig,
    config_digest,
    resolved_manifest,
    validate_config,
)
from evoworld.contracts import JsonValue
from evoworld.pipeline import PipelineAdapters
from evoworld.ports import (
    Executor,
    ExperienceMemory,
    ExpertPolicy,
    RolloutScorer,
    Router,
    RunStore,
    SceneAnchor,
    TaskSource,
    Trainer,
    TrajectoryEvaluator,
    WorldModel,
)

RUNTIME_MANIFEST_ENV = "EVOWORLD_RUNTIME_MANIFEST"
_IMPORT_ID = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$"
)
_GROOT_ACTION_SCHEMA = re.compile(r"^groot:[A-Za-z0-9_.-]+(?:\+[A-Za-z0-9_.-]+)*$")
_ROOT_FIELDS = {
    "schema_version",
    "project_root",
    "media_root",
    "asset_root",
    "pipeline_provider",
    "action_projectors",
    "openpi",
    "groot",
    "ctrl_world",
    "qwen_vl",
}
_PROJECTOR_FIELDS = {
    "identity",
    "implementation",
    "source_schema",
    "source_width",
    "target_schema",
    "config_path",
    "config_digest",
    "source_digest",
    "weights_path",
    "weights_digest",
    "content_digest",
}
_OPENPI_FIELDS = {
    "expert_id",
    "python",
    "model",
    "host",
    "port",
    "timeout_seconds",
    "checkpoint_identity",
    "checkpoint_digest",
}
_GROOT_FIELDS = {
    "expert_id",
    "python",
    "host",
    "port",
    "timeout_seconds",
    "checkpoint_identity",
    "checkpoint_digest",
}
_CTRL_FIELDS = {
    "python",
    "model_root",
    "text_encoder_root",
    "dataset_root",
    "checkpoint_path",
    "stats_path",
    "timeout_seconds",
    "fps",
}
_QWEN_FIELDS = {
    "python",
    "model_id",
    "model_digest",
    "model_revision",
    "adapter_checkpoint",
    "checkpoint_root",
    "device",
    "dtype",
    "max_new_tokens",
    "timeout_seconds",
}


class RuntimeManifestError(RuntimeError):
    """Real composition is incomplete, unsafe, or not explicitly configured."""


@dataclass(frozen=True, slots=True)
class RuntimeExpertMetadata:
    """Pure-value identity disclosed to the deployment-private provider."""

    expert_id: str
    backend: str
    checkpoint_digest: str
    projector_content_digest: str


@dataclass(frozen=True, slots=True)
class RuntimeBackendMetadata:
    """Capability-free runtime summary safe to pass across the provider boundary."""

    schema_version: int
    runtime_manifest_digest: str
    experts: tuple[RuntimeExpertMetadata, ...]


@dataclass(frozen=True, slots=True)
class RealAdapterSet:
    """Complete validated adapter set returned to the core pipeline."""

    task_source: TaskSource
    scene_anchor: SceneAnchor
    router: Router
    policies: Mapping[str, ExpertPolicy]
    world_model: WorldModel
    rollout_scorer: RolloutScorer
    executor: Executor
    evaluator: TrajectoryEvaluator
    trainer: Trainer
    memory: ExperienceMemory
    store: RunStore
    runtime_manifest_digest: str


@dataclass(frozen=True, slots=True)
class CoreRuntimePorts:
    """Only deployment-private core ports; public backends cannot be replaced."""

    task_source: TaskSource
    scene_anchor: SceneAnchor
    router: Router
    rollout_scorer: RolloutScorer
    executor: Executor
    trainer: Trainer
    memory: ExperienceMemory
    store: RunStore


Provider = Callable[[EvoWorldConfig, Path, RuntimeBackendMetadata], CoreRuntimePorts]


def build_real_adapters(config: EvoWorldConfig, run_dir: Path) -> PipelineAdapters:
    """Build every public real backend and require a private provider for remaining ports."""
    trusted_openpi_bridge_builder = build_openpi_process_bridge
    trusted_groot_bridge_builder = build_groot_process_bridge
    trusted_ctrl_bridge_builder = build_ctrl_world_process_bridge
    trusted_qwen_bridge_builder = build_qwen_process_bridge
    trusted_openpi_adapter = OpenPiPolicyAdapter
    trusted_groot_adapter = GrootPolicyAdapter
    trusted_projected_policy = ProjectedExpertPolicy
    trusted_ctrl_adapter = CtrlWorldAdapter
    trusted_qwen_evaluator = QwenVlEvaluator
    trusted_validator = _validated_adapter_set
    validate_config(config)
    if config.mode != "real":
        raise RuntimeManifestError("real composition requires mode='real'")
    _, raw = _load_runtime_manifest()
    project_root = _directory(raw, "project_root")
    media_root = _directory(raw, "media_root")
    asset_root = _directory(raw, "asset_root")
    provider_identity = _text(raw, "pipeline_provider")
    resolved_run = _prepare_run_dir(run_dir)
    artifact_root = _real_directory(resolved_run / "backends")

    openpi_raw = _object(raw, "openpi", _OPENPI_FIELDS)
    groot_raw = _object(raw, "groot", _GROOT_FIELDS)
    ctrl_raw = _object(raw, "ctrl_world", _CTRL_FIELDS)
    qwen_raw = _object(raw, "qwen_vl", _QWEN_FIELDS)
    openpi_id = _text(openpi_raw, "expert_id")
    groot_id = _text(groot_raw, "expert_id")
    _require_configured_experts(config, openpi_id, groot_id)
    projectors = _build_projectors(raw, project_root, asset_root, openpi_id, groot_id)
    digest = _runtime_digest(raw, project_root, asset_root, ctrl_raw, qwen_raw)
    provider = _load_provider(provider_identity)
    metadata = RuntimeBackendMetadata(
        2,
        digest,
        (
            RuntimeExpertMetadata(
                groot_id,
                "groot",
                _digest(groot_raw, "checkpoint_digest"),
                projectors[groot_id].identity_record.content_digest,
            ),
            RuntimeExpertMetadata(
                openpi_id,
                "openpi",
                _digest(openpi_raw, "checkpoint_digest"),
                projectors[openpi_id].identity_record.content_digest,
            ),
        ),
    )
    openpi_artifacts = _real_directory(artifact_root / "openpi")
    groot_artifacts = _real_directory(artifact_root / "groot")
    ctrl_artifacts = _real_directory(artifact_root / "ctrl_world")
    qwen_artifacts = _real_directory(artifact_root / "qwen_vl")

    openpi_bridge = trusted_openpi_bridge_builder(
        project_root / "third_party" / "openpi",
        openpi_artifacts,
        media_root,
        backend_python=_file(openpi_raw, "python", allow_symlink=True),
        project_root=project_root,
    )
    groot_bridge = trusted_groot_bridge_builder(
        project_root / "third_party" / "isaac_groot",
        groot_artifacts,
        media_root,
        backend_python=_file(groot_raw, "python", allow_symlink=True),
        project_root=project_root,
    )
    ctrl_bridge = trusted_ctrl_bridge_builder(
        project_root / "third_party" / "ctrl_world",
        ctrl_artifacts,
        asset_root,
        media_root,
        backend_python=_file(ctrl_raw, "python", allow_symlink=True),
        project_root=project_root,
    )
    adapter_checkpoint = _optional_directory(qwen_raw, "adapter_checkpoint")
    checkpoint_root = _optional_directory(qwen_raw, "checkpoint_root")
    qwen_bridge = trusted_qwen_bridge_builder(
        qwen_artifacts,
        media_root,
        backend_python=_file(qwen_raw, "python", allow_symlink=True),
        project_root=project_root,
        model_id=_text(qwen_raw, "model_id"),
        model_digest=_digest(qwen_raw, "model_digest"),
        model_revision=_revision(qwen_raw, "model_revision"),
        adapter_checkpoint=adapter_checkpoint,
        checkpoint_root=checkpoint_root,
        device=_text(qwen_raw, "device"),
        dtype=_text(qwen_raw, "dtype"),
        max_new_tokens=_integer(qwen_raw, "max_new_tokens", minimum=16),
    )

    raw_policies: Mapping[str, ExpertPolicy] = MappingProxyType(
        {
            openpi_id: trusted_openpi_adapter(
                openpi_bridge,
                openpi_id,
                _text(openpi_raw, "model"),
                artifact_dir=openpi_artifacts,
                allowed_media_root=media_root,
                checkpoint_identity=_text(openpi_raw, "checkpoint_identity"),
                checkpoint_digest=_digest(openpi_raw, "checkpoint_digest"),
                timeout_seconds=_number(openpi_raw, "timeout_seconds"),
                host=_text(openpi_raw, "host"),
                port=_integer(openpi_raw, "port", minimum=1, maximum=65535),
            ),
            groot_id: trusted_groot_adapter(
                groot_bridge,
                groot_id,
                artifact_dir=groot_artifacts,
                allowed_media_root=media_root,
                checkpoint_identity=_text(groot_raw, "checkpoint_identity"),
                checkpoint_digest=_digest(groot_raw, "checkpoint_digest"),
                timeout_seconds=_number(groot_raw, "timeout_seconds"),
                host=_text(groot_raw, "host"),
                port=_integer(groot_raw, "port", minimum=1, maximum=65535),
            ),
        }
    )
    policies: Mapping[str, ExpertPolicy] = MappingProxyType(
        {
            expert_id: trusted_projected_policy(
                raw_policies[expert_id], projector, projector.identity_record
            )
            for expert_id, projector in projectors.items()
        }
    )
    world_model = trusted_ctrl_adapter(
        ctrl_bridge,
        CtrlWorldAssets(
            repo_root=project_root / "third_party" / "ctrl_world",
            config_path=project_root / "third_party" / "ctrl_world" / "config.py",
            model_root=_directory(ctrl_raw, "model_root"),
            text_encoder_root=_directory(ctrl_raw, "text_encoder_root"),
            dataset_root=_directory(ctrl_raw, "dataset_root"),
            checkpoint_path=_file(ctrl_raw, "checkpoint_path"),
            stats_path=_file(ctrl_raw, "stats_path"),
        ),
        artifact_dir=ctrl_artifacts,
        artifact_root=artifact_root,
        asset_root=asset_root,
        media_root=media_root,
        timeout_seconds=_number(ctrl_raw, "timeout_seconds"),
        fps=_number(ctrl_raw, "fps"),
    )
    evaluator = trusted_qwen_evaluator(
        qwen_bridge,
        artifact_dir=qwen_artifacts,
        artifact_root=artifact_root,
        media_root=media_root,
        model_id=_text(qwen_raw, "model_id") or DEFAULT_QWEN_MODEL,
        model_digest=_digest(qwen_raw, "model_digest"),
        model_revision=_revision(qwen_raw, "model_revision"),
        adapter_checkpoint=adapter_checkpoint,
        timeout_seconds=_number(qwen_raw, "timeout_seconds"),
    )
    candidate = provider(config, resolved_run, metadata)
    return trusted_validator(
        candidate,
        policies,
        world_model,
        evaluator,
        digest,
        resolved_run,
        config,
    )


def _load_runtime_manifest() -> tuple[Path, dict[str, object]]:
    configured = os.environ.get(RUNTIME_MANIFEST_ENV)
    if configured is None or not configured.strip():
        raise RuntimeManifestError(f"{RUNTIME_MANIFEST_ENV} is required")
    path = Path(configured)
    if not path.is_absolute() or path.is_symlink():
        raise RuntimeManifestError("runtime manifest must be an absolute regular file")
    try:
        resolved = path.resolve(strict=True)
        payload = resolved.read_bytes()
        raw = decode_json_object(payload.decode("utf-8"), source="runtime manifest")
    except (OSError, UnicodeError, ValueError) as error:
        raise RuntimeManifestError("runtime manifest cannot be read as strict JSON") from error
    if raw.get("schema_version") == 1:
        raise RuntimeManifestError(
            "runtime manifest schema version 1 is unsupported; migrate projectors to schema 2"
        )
    if not resolved.is_file() or set(raw) != _ROOT_FIELDS or raw.get("schema_version") != 2:
        raise RuntimeManifestError("runtime manifest fields do not match schema version 2")
    return resolved, cast(dict[str, object], raw)


def _load_provider(identity: str) -> Provider:
    if _IMPORT_ID.fullmatch(identity) is None:
        raise RuntimeManifestError("pipeline_provider must use module:callable syntax")
    module_name, callable_name = identity.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        raise RuntimeManifestError("private pipeline provider cannot be imported") from error
    provider = getattr(module, callable_name, None)
    if provider is None or not callable(provider):
        raise RuntimeManifestError("private pipeline provider callable is unavailable")
    return cast(Provider, provider)


def _prepare_run_dir(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise RuntimeManifestError("run_dir must be an absolute non-symlink path")
    path.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve(strict=True)
    if resolved != path or not resolved.is_dir():
        raise RuntimeManifestError("run_dir must not traverse symlinked directories")
    return resolved


def _real_directory(path: Path) -> Path:
    path.mkdir(mode=0o750, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise RuntimeManifestError("backend artifact path must be a real directory")
    return path.resolve(strict=True)


def _require_configured_experts(config: EvoWorldConfig, openpi_id: str, groot_id: str) -> None:
    if openpi_id == groot_id:
        raise RuntimeManifestError("OpenPi and GR00T expert IDs must be distinct")
    configured = {expert.expert_id for expert in config.experts}
    required = {openpi_id, groot_id}
    if configured != required:
        raise RuntimeManifestError("real config experts must exactly match OpenPi and GR00T")
    by_id = {expert.expert_id: expert for expert in config.experts}
    if by_id[openpi_id].backend != "openpi" or by_id[groot_id].backend != "groot":
        raise RuntimeManifestError(
            "real config expert backends do not match their official adapters"
        )
    if any(expert.action_schema != "cartesian_pose_7d" for expert in config.experts):
        raise RuntimeManifestError("real config experts must use cartesian_pose_7d")


def _build_projectors(
    raw: Mapping[str, object],
    project_root: Path,
    asset_root: Path,
    openpi_id: str,
    groot_id: str,
) -> Mapping[str, AffineActionProjector]:
    value = raw.get("action_projectors")
    if not isinstance(value, dict) or set(value) != {openpi_id, groot_id}:
        raise RuntimeManifestError("action projector IDs must exactly match configured experts")
    source_path = (project_root / "src" / "evoworld" / "adapters" / "projection.py").resolve()
    if source_path != PROJECTOR_SOURCE_PATH:
        raise RuntimeManifestError("project_root does not contain the active projector source")
    result: dict[str, AffineActionProjector] = {}
    for expert_id in sorted(value):
        entry = _object(cast(Mapping[str, object], value), expert_id, _PROJECTOR_FIELDS)
        identity_name = _text(entry, "identity")
        implementation = _text(entry, "implementation")
        source_schema = _text(entry, "source_schema")
        source_width = _integer(entry, "source_width", minimum=1)
        target_schema = _text(entry, "target_schema")
        if implementation != AFFINE_IMPLEMENTATION:
            raise RuntimeManifestError("projector implementation must be affine_v1")
        if expert_id == openpi_id and source_schema != "joint_velocity":
            raise RuntimeManifestError("OpenPi projector source_schema must be joint_velocity")
        if expert_id == openpi_id and source_width != 8:
            raise RuntimeManifestError("OpenPi projector source_width must be 8")
        if expert_id == groot_id and _GROOT_ACTION_SCHEMA.fullmatch(source_schema) is None:
            raise RuntimeManifestError("GR00T projector source_schema must name exact action keys")
        config_digest = _digest(entry, "config_digest")
        source_digest = _digest(entry, "source_digest")
        weights_digest = _digest(entry, "weights_digest")
        claimed_content = _digest(entry, "content_digest")
        actual_content = projector_content_digest(
            expert_id=expert_id,
            identity=identity_name,
            implementation=implementation,
            source_schema=source_schema,
            source_width=source_width,
            target_schema=target_schema,
            config_digest=config_digest,
            source_digest=source_digest,
            weights_digest=weights_digest,
        )
        if claimed_content != actual_content:
            raise RuntimeManifestError("projector content_digest does not match its identity")
        try:
            identity = ProjectorIdentity(
                expert_id,
                identity_name,
                implementation,
                source_schema,
                source_width,
                target_schema,
                config_digest,
                source_digest,
                weights_digest,
                claimed_content,
            )
            result[expert_id] = load_affine_projector(
                identity,
                config_path=_path(entry, "config_path"),
                weights_path=_path(entry, "weights_path"),
                asset_root=asset_root,
                source_path=source_path,
                source_root=project_root,
            )
        except ProjectionError as error:
            raise RuntimeManifestError(str(error)) from error
    return MappingProxyType(result)


def _runtime_digest(
    raw: Mapping[str, object],
    project_root: Path,
    asset_root: Path,
    ctrl_raw: Mapping[str, object],
    qwen_raw: Mapping[str, object],
) -> str:
    adapter_checkpoint = _optional_directory(qwen_raw, "adapter_checkpoint")
    checkpoint_root = _optional_directory(qwen_raw, "checkpoint_root")
    if adapter_checkpoint is not None and checkpoint_root is None:
        raise RuntimeManifestError("checkpoint_root is required for a Qwen adapter")
    static_assets = combined_manifest(
        [
            file_manifest(
                project_root / "third_party" / "ctrl_world" / "config.py",
                project_root,
                "Ctrl-World config",
            ),
            directory_manifest(
                _directory(ctrl_raw, "model_root"), asset_root, "Ctrl-World model_root"
            ),
            directory_manifest(
                _directory(ctrl_raw, "text_encoder_root"),
                asset_root,
                "Ctrl-World text_encoder_root",
            ),
            file_manifest(_file(ctrl_raw, "checkpoint_path"), asset_root, "Ctrl-World checkpoint"),
            file_manifest(_file(ctrl_raw, "stats_path"), asset_root, "Ctrl-World stats"),
            *(
                []
                if adapter_checkpoint is None or checkpoint_root is None
                else [
                    directory_manifest(
                        adapter_checkpoint, checkpoint_root, "Qwen adapter checkpoint"
                    )
                ]
            ),
            file_manifest(
                project_root / "src" / "evoworld" / "adapters" / "projection.py",
                project_root,
                "projector source",
            ),
        ]
    )
    return input_digest(
        {
            "runtime_manifest": cast(JsonValue, dict(raw)),
            "static_assets": static_assets,
        }
    )


def _validated_adapter_set(
    candidate: CoreRuntimePorts,
    policies: Mapping[str, ExpertPolicy],
    world_model: WorldModel,
    evaluator: TrajectoryEvaluator,
    runtime_manifest_digest: str,
    run_dir: Path,
    config: EvoWorldConfig,
) -> PipelineAdapters:
    if type(candidate) is not CoreRuntimePorts:
        raise RuntimeManifestError("private provider must return CoreRuntimePorts exactly")
    values = (
        (candidate.task_source, TaskSource, "task_source"),
        (candidate.scene_anchor, SceneAnchor, "scene_anchor"),
        (candidate.router, Router, "router"),
        (candidate.rollout_scorer, RolloutScorer, "rollout_scorer"),
        (candidate.executor, Executor, "executor"),
        (candidate.trainer, Trainer, "trainer"),
        (candidate.memory, ExperienceMemory, "memory"),
        (candidate.store, RunStore, "store"),
    )
    for value, protocol, field in values:
        if not isinstance(value, protocol):
            raise RuntimeManifestError(f"private provider returned invalid {field}")
    try:
        store_run = candidate.store.run_dir.resolve(strict=True)
    except OSError as error:
        raise RuntimeManifestError("private provider store has no durable run directory") from error
    if store_run != run_dir:
        raise RuntimeManifestError("private provider store is bound to a different run directory")
    _validate_store_manifest(candidate.store, config)
    return RealAdapterSet(
        candidate.task_source,
        candidate.scene_anchor,
        candidate.router,
        policies,
        world_model,
        candidate.rollout_scorer,
        candidate.executor,
        evaluator,
        candidate.trainer,
        candidate.memory,
        candidate.store,
        runtime_manifest_digest,
    )


def _validate_store_manifest(store: RunStore, config: EvoWorldConfig) -> None:
    try:
        wrapper = decode_json_object(
            (store.run_dir / "manifest.json").read_text(encoding="utf-8"),
            source="private provider run manifest",
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise RuntimeManifestError(
            "private provider did not create a valid run manifest"
        ) from error
    if set(wrapper) not in (
        {"config_digest", "resolved_config"},
        {"config_digest", "resolved_config", "composition"},
    ):
        raise RuntimeManifestError("private provider run manifest fields are invalid")
    if wrapper.get("config_digest") != config_digest(config) or canonical_json(
        wrapper.get("resolved_config")
    ) != canonical_json(resolved_manifest(config)):
        raise RuntimeManifestError("private provider run manifest does not match real config")


def _object(raw: Mapping[str, object], field: str, expected: set[str]) -> dict[str, object]:
    value = raw.get(field)
    if not isinstance(value, dict) or set(value) != expected:
        raise RuntimeManifestError(f"runtime manifest {field} fields are invalid")
    return cast(dict[str, object], value)


def _text(raw: Mapping[str, object], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeManifestError(f"runtime manifest {field} must be nonblank text")
    return value


def _digest(raw: Mapping[str, object], field: str) -> str:
    value = _text(raw, field)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeManifestError(f"runtime manifest {field} must be a lowercase SHA-256")
    return value


def _revision(raw: Mapping[str, object], field: str) -> str:
    value = _text(raw, field)
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeManifestError(f"runtime manifest {field} must be a lowercase Git commit")
    return value


def _integer(
    raw: Mapping[str, object], field: str, *, minimum: int, maximum: int | None = None
) -> int:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeManifestError(f"runtime manifest {field} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise RuntimeManifestError(f"runtime manifest {field} is out of range")
    return value


def _number(raw: Mapping[str, object], field: str) -> float:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeManifestError(f"runtime manifest {field} must be numeric")
    number = float(value)
    if not number > 0 or not number < float("inf"):
        raise RuntimeManifestError(f"runtime manifest {field} must be finite and positive")
    return number


def _path(raw: Mapping[str, object], field: str) -> Path:
    value = _text(raw, field)
    path = Path(value)
    if not path.is_absolute():
        raise RuntimeManifestError(f"runtime manifest {field} must be absolute")
    return path


def _directory(raw: Mapping[str, object], field: str) -> Path:
    path = _path(raw, field)
    if path.is_symlink():
        raise RuntimeManifestError(f"runtime manifest {field} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeManifestError(f"runtime manifest {field} directory is missing") from error
    if not resolved.is_dir():
        raise RuntimeManifestError(f"runtime manifest {field} must be a directory")
    return resolved


def _file(raw: Mapping[str, object], field: str, *, allow_symlink: bool = False) -> Path:
    path = _path(raw, field)
    if path.is_symlink() and not allow_symlink:
        raise RuntimeManifestError(f"runtime manifest {field} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeManifestError(f"runtime manifest {field} file is missing") from error
    if not resolved.is_file():
        raise RuntimeManifestError(f"runtime manifest {field} must be a file")
    return path if allow_symlink else resolved


def _optional_file(raw: Mapping[str, object], field: str) -> Path | None:
    return None if raw.get(field) is None else _file(raw, field)


def _optional_directory(raw: Mapping[str, object], field: str) -> Path | None:
    return None if raw.get(field) is None else _directory(raw, field)

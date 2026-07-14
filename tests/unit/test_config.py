"""Tests for deterministic EvoWorld configuration loading."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from evoworld.config import (
    ConfigError,
    GroundingMode,
    config_digest,
    load_config,
    resolved_manifest,
)
from evoworld.contracts import GroundingMode as ContractGroundingMode


def write_config(
    directory: Path,
    *,
    run_root: str = "artifacts/runs",
    mode: str = "dry_run",
    executor: str = "",
    expert_rows: str | None = None,
) -> Path:
    """Write a valid configuration, with focused overrides for each test."""
    executor_line = f'executor = "{executor}"\n' if executor else ""
    experts = (
        expert_rows
        or """
[[experts]]
expert_id = "pi05"
backend = "openpi"
task_families = ["manipulation"]
checkpoint = "checkpoints/pi05"
action_schema = "joint_velocity"
enabled = true
"""
    )
    path = directory / "evoworld.toml"
    path.write_text(
        f"""schema_version = 1
seed = 7
run_root = "{run_root}"
mode = "{mode}"
grounding_mode = "nearest"
grounding_temperature = 0.5
router_weight = 0.25
reliability_threshold = 0.7
rollout_horizon = 10
{executor_line}
{experts}
""",
        encoding="utf-8",
    )
    return path


def test_paths_resolve_relative_to_config(tmp_path: Path) -> None:
    path = write_config(tmp_path, run_root="artifacts/runs")

    config = load_config(path)

    assert config.run_root == (tmp_path / "artifacts/runs").resolve()
    assert config.experts[0].checkpoint == (tmp_path / "checkpoints/pi05").resolve()


def test_real_mode_requires_explicit_executor(tmp_path: Path) -> None:
    path = write_config(tmp_path, mode="real", executor="")

    with pytest.raises(ConfigError, match="executor"):
        load_config(path)


def test_config_rejects_unknown_root_field(tmp_path: Path) -> None:
    path = write_config(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace("seed = 7\n", "seed = 7\nmystery = true\n"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="unknown root field.*mystery"):
        load_config(path)


def test_config_rejects_unknown_expert_field(tmp_path: Path) -> None:
    path = write_config(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + "surprise = true\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown expert field.*surprise"):
        load_config(path)


def test_config_rejects_incompatible_enabled_action_schemas(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        expert_rows="""
[[experts]]
expert_id = "pi05"
backend = "openpi"
task_families = ["manipulation"]
action_schema = "joint_velocity"
enabled = true

[[experts]]
expert_id = "groot"
backend = "groot"
task_families = ["manipulation"]
action_schema = "cartesian_pose"
enabled = true
""",
    )

    with pytest.raises(ConfigError, match="action_schema"):
        load_config(path)


def test_resolved_manifest_is_stable_and_redacts_secret_values(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))

    manifest = resolved_manifest(config, {"api_key": "not-for-artifacts", "region": "local"})

    assert manifest["settings"]["api_key"] == "<redacted>"
    assert manifest["experts"][0]["checkpoint"] == str((tmp_path / "checkpoints/pi05").resolve())
    assert config_digest(
        config, {"api_key": "not-for-artifacts", "region": "local"}
    ) == config_digest(config, {"api_key": "another-secret", "region": "local"})
    assert config.grounding_mode is GroundingMode.NEAREST
    assert GroundingMode is ContractGroundingMode


@pytest.mark.parametrize(
    ("field", "toml_value"),
    (
        (field, toml_value)
        for field in (
            "grounding_temperature",
            "router_weight",
            "reliability_threshold",
        )
        for toml_value in ("nan", "+inf", "-inf")
    ),
)
def test_toml_nonfinite_float_is_rejected(
    tmp_path: Path,
    field: str,
    toml_value: str,
) -> None:
    path = write_config(tmp_path)
    text = path.read_text(encoding="utf-8")
    prefix = f"{field} = "
    path.write_text(
        "\n".join(
            f"{prefix}{toml_value}" if line.startswith(prefix) else line
            for line in text.splitlines()
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match=field):
        load_config(path)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_config_digest_rejects_nonfinite_programmatic_config(
    tmp_path: Path,
    value: float,
) -> None:
    config = replace(load_config(write_config(tmp_path)), grounding_temperature=value)

    with pytest.raises(ConfigError, match="finite"):
        config_digest(config)


def test_config_digest_rejects_nonfinite_settings_json(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))

    with pytest.raises(ConfigError, match="finite canonical JSON"):
        config_digest(config, {"temperature": float("nan")})

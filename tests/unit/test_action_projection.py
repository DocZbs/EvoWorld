"""Contracts for the fixed, declarative action projector."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path

import pytest

from evoworld.adapters.projection import (
    AffineActionProjector,
    PROJECTOR_SOURCE_PATH,
    ProjectedExpertPolicy,
    ProjectionError,
    ProjectorIdentity,
    load_affine_projector,
    projector_content_digest,
)
from evoworld.contracts import ActionChunk, SceneRecord, TaskSpec


class _RawPolicy:
    def __init__(self, result: object) -> None:
        self.result = result
        self.operations: list[str] = []

    def act(self, task: TaskSpec, observation: SceneRecord, *, operation_id: str) -> ActionChunk:
        del task, observation
        self.operations.append(operation_id)
        return self.result  # type: ignore[return-value]


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _assets(
    tmp_path: Path,
    *,
    expert_id: str = "pi05",
    source_schema: str = "joint_velocity",
    source_width: int = 8,
) -> tuple[ProjectorIdentity, Path, Path, Path]:
    root = tmp_path / expert_id
    root.mkdir()
    config = root / "config.json"
    weights = root / "weights.json"
    source = root / "projection.py"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "implementation": "affine_v1",
                "source_width": source_width,
                "target_width": 7,
            }
        ),
        encoding="utf-8",
    )
    matrix = [[1.0 if column == row else 0.0 for column in range(source_width)] for row in range(7)]
    weights.write_text(
        json.dumps({"schema_version": 1, "matrix": matrix, "bias": [0.5] * 7}),
        encoding="utf-8",
    )
    source.write_bytes(PROJECTOR_SOURCE_PATH.read_bytes())
    identity_name = f"{expert_id}-affine-v1"
    config_digest = _digest(config)
    source_digest = _digest(source)
    weights_digest = _digest(weights)
    identity = ProjectorIdentity(
        expert_id,
        identity_name,
        "affine_v1",
        source_schema,
        source_width,
        "cartesian_pose_7d",
        config_digest,
        source_digest,
        weights_digest,
        projector_content_digest(
            expert_id=expert_id,
            identity=identity_name,
            implementation="affine_v1",
            source_schema=source_schema,
            source_width=source_width,
            target_schema="cartesian_pose_7d",
            config_digest=config_digest,
            source_digest=source_digest,
            weights_digest=weights_digest,
        ),
    )
    return identity, config, weights, source


def _load(
    tmp_path: Path,
    *,
    expert_id: str = "pi05",
    source_schema: str = "joint_velocity",
    source_width: int = 8,
) -> tuple[AffineActionProjector, ProjectorIdentity, Path, Path, Path]:
    identity, config, weights, source = _assets(
        tmp_path,
        expert_id=expert_id,
        source_schema=source_schema,
        source_width=source_width,
    )
    return (
        load_affine_projector(
            identity,
            config_path=config,
            weights_path=weights,
            asset_root=tmp_path,
            source_path=source,
            source_root=tmp_path,
        ),
        identity,
        config,
        weights,
        source,
    )


def _rebind_content(identity: ProjectorIdentity) -> ProjectorIdentity:
    return replace(
        identity,
        content_digest=projector_content_digest(
            expert_id=identity.expert_id,
            identity=identity.identity,
            implementation=identity.implementation,
            source_schema=identity.source_schema,
            source_width=identity.source_width,
            target_schema=identity.target_schema,
            config_digest=identity.config_digest,
            source_digest=identity.source_digest,
            weights_digest=identity.weights_digest,
        ),
    )


def test_fixed_affine_projector_maps_real_width_to_cartesian_7d(tmp_path: Path) -> None:
    projector, _, _, _, _ = _load(tmp_path)
    raw = ActionChunk("pi05", ((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),), (0.0,), "joint_velocity")

    result = projector.project(
        TaskSpec("task", "pick"), SceneRecord("scene"), raw, operation_id="op:project"
    )

    assert result == ActionChunk(
        "pi05", ((1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5),), (0.0,), "cartesian_pose_7d"
    )


def test_wrapper_derives_raw_and_project_operation_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    projector, identity, _, _, _ = _load(tmp_path)
    raw_chunk = ActionChunk("pi05", ((1.0,) * 8,), (0.0,), "joint_velocity")
    raw = _RawPolicy(raw_chunk)
    policy = ProjectedExpertPolicy(raw, projector, identity)
    project_operations: list[str] = []
    original_project = AffineActionProjector.project

    def record_project(
        self: AffineActionProjector,
        task: TaskSpec,
        observation: SceneRecord,
        actions: ActionChunk,
        *,
        operation_id: str,
    ) -> ActionChunk:
        project_operations.append(operation_id)
        return original_project(self, task, observation, actions, operation_id=operation_id)

    monkeypatch.setattr(AffineActionProjector, "project", record_project)

    result = policy.act(
        TaskSpec("task", "pick"), SceneRecord("scene"), operation_id="round:act:pi05"
    )

    assert raw.operations == ["round:act:pi05:raw"]
    assert project_operations == ["round:act:pi05:project"]
    assert result.schema_name == "cartesian_pose_7d"
    assert len(result.actions[0]) == 7


@pytest.mark.parametrize("timestamps", [(-0.1,), (-0.2, -0.1), (-0.1, 0.0), (0.1, 0.0)])
def test_negative_or_non_increasing_timestamps_fail_before_projection(
    tmp_path: Path, timestamps: tuple[float, ...]
) -> None:
    projector, identity, _, _, _ = _load(tmp_path)
    raw_chunk = ActionChunk(
        "pi05", tuple((1.0,) * 8 for _ in timestamps), timestamps, "joint_velocity"
    )
    raw = _RawPolicy(raw_chunk)
    policy = ProjectedExpertPolicy(raw, projector, identity)

    with pytest.raises(ProjectionError, match="timestamps"):
        policy.act(TaskSpec("task", "pick"), SceneRecord("scene"), operation_id="act")


@pytest.mark.parametrize(
    "raw",
    [
        ActionChunk("other", ((1.0,) * 8,), (0.0,), "joint_velocity"),
        ActionChunk("pi05", ((1.0,) * 8,), (0.0,), "other"),
        ActionChunk("pi05", ((1.0,) * 7,), (0.0,), "joint_velocity"),
    ],
)
def test_wrong_expert_schema_or_source_width_is_rejected(tmp_path: Path, raw: ActionChunk) -> None:
    projector, identity, _, _, _ = _load(tmp_path)
    policy = ProjectedExpertPolicy(_RawPolicy(raw), projector, identity)

    with pytest.raises(ProjectionError):
        policy.act(TaskSpec("task", "pick"), SceneRecord("scene"), operation_id="act")


@pytest.mark.parametrize(
    ("file_name", "payload", "message"),
    [
        (
            "config",
            {"schema_version": 1, "implementation": "affine_v1", "source_width": 8},
            "config fields",
        ),
        (
            "config",
            {
                "schema_version": 1,
                "implementation": "affine_v1",
                "source_width": 8,
                "target_width": 7,
                "factory": "malicious.module:build",
            },
            "config fields",
        ),
        (
            "config",
            {
                "schema_version": 1,
                "implementation": "dynamic_python",
                "source_width": 8,
                "target_width": 7,
            },
            "implementation",
        ),
        (
            "config",
            {
                "schema_version": True,
                "implementation": "affine_v1",
                "source_width": 8,
                "target_width": 7,
            },
            "schema_version",
        ),
        (
            "config",
            {
                "schema_version": 1,
                "implementation": "affine_v1",
                "source_width": 8.0,
                "target_width": 7,
            },
            "source_width",
        ),
        (
            "config",
            {
                "schema_version": 1,
                "implementation": "affine_v1",
                "source_width": 8,
                "target_width": 7.0,
            },
            "target_width",
        ),
        (
            "weights",
            {"schema_version": True, "matrix": [[0.0] * 8] * 7, "bias": [0.0] * 7},
            "schema_version",
        ),
        (
            "weights",
            {"schema_version": 1, "matrix": [[0.0] * 8] * 6, "bias": [0.0] * 7},
            "matrix",
        ),
        (
            "weights",
            {"schema_version": 1, "matrix": [[0.0] * 8] * 7, "bias": [0.0] * 8},
            "bias",
        ),
        (
            "weights",
            {"schema_version": 1, "matrix": [[0.0] * 7] * 7, "bias": [0.0] * 7},
            "matrix",
        ),
    ],
)
def test_calibration_json_is_strict_and_shape_checked(
    tmp_path: Path, file_name: str, payload: dict[str, object], message: str
) -> None:
    identity, config, weights, source = _assets(tmp_path)
    target = config if file_name == "config" else weights
    target.write_text(json.dumps(payload), encoding="utf-8")
    identity = _rebind_content(
        replace(identity, config_digest=_digest(target))
        if file_name == "config"
        else replace(identity, weights_digest=_digest(target))
    )

    with pytest.raises(ProjectionError, match=message):
        load_affine_projector(
            identity,
            config_path=config,
            weights_path=weights,
            asset_root=tmp_path,
            source_path=source,
            source_root=tmp_path,
        )


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema_version":1,"matrix":NaN,"bias":[]}',
        b'{"schema_version":1,"schema_version":1,"matrix":[],"bias":[]}',
        b"[]",
    ],
)
def test_malicious_or_non_object_json_is_rejected(tmp_path: Path, payload: bytes) -> None:
    identity, config, weights, source = _assets(tmp_path)
    weights.write_bytes(payload)
    identity = _rebind_content(replace(identity, weights_digest=_digest(weights)))

    with pytest.raises(ProjectionError, match="weights JSON"):
        load_affine_projector(
            identity,
            config_path=config,
            weights_path=weights,
            asset_root=tmp_path,
            source_path=source,
            source_root=tmp_path,
        )


def test_huge_json_integer_is_translated_to_projection_error(tmp_path: Path) -> None:
    identity, config, weights, source = _assets(tmp_path)
    payload = weights.read_text(encoding="utf-8").replace("1.0", "9" * 4_000, 1)
    weights.write_text(payload, encoding="utf-8")
    identity = _rebind_content(replace(identity, weights_digest=_digest(weights)))

    with pytest.raises(ProjectionError, match="finite"):
        load_affine_projector(
            identity,
            config_path=config,
            weights_path=weights,
            asset_root=tmp_path,
            source_path=source,
            source_root=tmp_path,
        )


def test_snapshot_rejects_symlink_path_replacement_and_source_drift(tmp_path: Path) -> None:
    identity, config, weights, source = _assets(tmp_path)
    original = weights.with_name("original.json")
    weights.rename(original)
    weights.symlink_to(original)
    with pytest.raises(ProjectionError, match="safely"):
        load_affine_projector(
            identity,
            config_path=config,
            weights_path=weights,
            asset_root=tmp_path,
            source_path=source,
            source_root=tmp_path,
        )

    weights.unlink()
    original.rename(weights)
    source.write_text("# changed implementation\n", encoding="utf-8")
    identity = _rebind_content(replace(identity, source_digest=_digest(source)))
    with pytest.raises(ProjectionError, match="changed after its implementation was loaded"):
        load_affine_projector(
            identity,
            config_path=config,
            weights_path=weights,
            asset_root=tmp_path,
            source_path=source,
            source_root=tmp_path,
        )


def test_snapshot_rejects_parent_directory_symlink_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, config, weights, source = _assets(tmp_path)
    directory = config.parent
    saved = tmp_path / "saved-projector"
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    original_open = os.open
    mutated = False

    def racing_open(
        path: str | bytes | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal mutated
        if path == config.name and dir_fd is not None and not mutated:
            mutated = True
            directory.rename(saved)
            directory.symlink_to(attacker, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", racing_open)
    try:
        with pytest.raises(ProjectionError, match="parent changed during snapshot"):
            load_affine_projector(
                identity,
                config_path=config,
                weights_path=weights,
                asset_root=tmp_path,
                source_path=source,
                source_root=tmp_path,
            )
    finally:
        directory.unlink(missing_ok=True)
        saved.rename(directory)


def test_snapshot_rejects_lexical_parent_traversal(tmp_path: Path) -> None:
    identity, config, weights, source = _assets(tmp_path)
    traversing = config.parent / "nested" / ".." / config.name

    with pytest.raises(ProjectionError, match="child file"):
        load_affine_projector(
            identity,
            config_path=traversing,
            weights_path=weights,
            asset_root=tmp_path,
            source_path=source,
            source_root=tmp_path,
        )


def test_loaded_projector_uses_immutable_snapshot_not_mutable_paths(tmp_path: Path) -> None:
    projector, _, config, weights, source = _load(tmp_path)
    config.unlink()
    weights.write_text("malicious", encoding="utf-8")
    source.write_text("malicious", encoding="utf-8")
    raw = ActionChunk("pi05", ((1.0,) * 8,), (0.0,), "joint_velocity")

    result = projector.project(
        TaskSpec("task", "pick"), SceneRecord("scene"), raw, operation_id="project"
    )

    assert result.actions == ((1.5,) * 7,)

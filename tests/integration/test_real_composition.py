"""Fail-closed checks for the checked-in real composition factory."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, cast

import imageio_ffmpeg  # type: ignore[import-untyped]
import numpy as np
from PIL import Image
import pytest
import evoworld.real_composition as real_composition

from evoworld.adapters.content import combined_manifest, file_manifest
from evoworld.adapters.ctrl_world import CTRL_WORLD_PIN, CtrlWorldAdapter
from evoworld.adapters.deterministic import build_deterministic_adapters
from evoworld.adapters.groot import GrootPolicyAdapter
from evoworld.adapters.openpi import OpenPiPolicyAdapter
from evoworld.adapters.process import BridgeRequest, BridgeResponse
from evoworld.adapters.projection import (
    AffineActionProjector,
    PROJECTOR_SOURCE_PATH,
    ProjectedExpertPolicy,
    projector_content_digest,
)
from evoworld.adapters.qwen_vl import QwenVlEvaluator
from evoworld.config import EvoWorldConfig, config_digest, resolved_manifest
from evoworld.contracts import (
    ActionChunk,
    CandidateScore,
    ExpertSpec,
    GroundingMode,
    JsonObject,
    Rollout,
    SceneRecord,
    TaskSpec,
    Trajectory,
)
from evoworld.pipeline import EvoWorldPipeline
from evoworld.ports import ExperienceMemory
from evoworld.real_composition import (
    RUNTIME_MANIFEST_ENV,
    CoreRuntimePorts,
    RuntimeBackendMetadata,
    RuntimeManifestError,
    _load_runtime_manifest,
    _runtime_digest,
    _validate_store_manifest,
    build_real_adapters,
)
from evoworld.storage import atomic_write_json

PROJECT_ROOT = Path(__file__).parents[2].resolve()


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _config(tmp_path: Path) -> EvoWorldConfig:
    return EvoWorldConfig(
        schema_version=1,
        seed=7,
        run_root=(tmp_path / "runs").resolve(),
        mode="real",
        grounding_mode=GroundingMode.NEAREST,
        grounding_temperature=0.5,
        router_weight=0.25,
        reliability_threshold=0.7,
        rollout_horizon=2,
        executor="private-robot-cell",
        experts=(
            ExpertSpec("groot", "groot", ("manipulation",), None, "cartesian_pose_7d"),
            ExpertSpec("pi05", "openpi", ("manipulation",), None, "cartesian_pose_7d"),
        ),
    )


def _projector_entry(
    assets: Path, expert_id: str, source_schema: str, source_width: int = 8
) -> dict[str, object]:
    directory = assets / "projectors" / expert_id
    directory.mkdir(parents=True)
    config = directory / "config.json"
    weights = directory / "weights.json"
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
    identity = f"{expert_id}-affine-v1"
    config_digest = _sha(config)
    source_digest = _sha(PROJECTOR_SOURCE_PATH)
    weights_digest = _sha(weights)
    return {
        "identity": identity,
        "implementation": "affine_v1",
        "source_schema": source_schema,
        "source_width": source_width,
        "target_schema": "cartesian_pose_7d",
        "config_digest": config_digest,
        "source_digest": source_digest,
        "weights_digest": weights_digest,
        "config_path": str(config.resolve()),
        "weights_path": str(weights.resolve()),
        "content_digest": projector_content_digest(
            expert_id=expert_id,
            identity=identity,
            implementation="affine_v1",
            source_schema=source_schema,
            source_width=source_width,
            target_schema="cartesian_pose_7d",
            config_digest=config_digest,
            source_digest=source_digest,
            weights_digest=weights_digest,
        ),
    }


def _runtime_manifest(tmp_path: Path, *, schema_version: int = 2) -> Path:
    media = tmp_path / "media"
    assets = tmp_path / "assets"
    for directory in (
        media / "annotation" / "val",
        assets / "svd",
        assets / "clip",
        assets / "qwen-adapter",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    checkpoint = assets / "ctrl-world.ckpt"
    stats = assets / "stats.json"
    checkpoint.write_bytes(b"checkpoint")
    stats.write_text("{}", encoding="utf-8")
    manifest = {
        "schema_version": schema_version,
        "project_root": str(PROJECT_ROOT),
        "media_root": str(media.resolve()),
        "asset_root": str(assets.resolve()),
        "pipeline_provider": "private_runtime:build",
        "action_projectors": {
            "pi05": _projector_entry(assets, "pi05", "joint_velocity"),
            "groot": _projector_entry(assets, "groot", "groot:arm"),
        },
        "openpi": {
            "expert_id": "pi05",
            "python": sys.executable,
            "model": "pi05",
            "host": "127.0.0.1",
            "port": 8000,
            "timeout_seconds": 5.0,
            "checkpoint_identity": "pi05-runtime",
            "checkpoint_digest": "a" * 64,
        },
        "groot": {
            "expert_id": "groot",
            "python": sys.executable,
            "host": "127.0.0.1",
            "port": 5555,
            "timeout_seconds": 20.0,
            "checkpoint_identity": "groot-runtime",
            "checkpoint_digest": "b" * 64,
        },
        "ctrl_world": {
            "python": sys.executable,
            "model_root": str((assets / "svd").resolve()),
            "text_encoder_root": str((assets / "clip").resolve()),
            "dataset_root": str(media.resolve()),
            "checkpoint_path": str(checkpoint.resolve()),
            "stats_path": str(stats.resolve()),
            "timeout_seconds": 300.0,
            "fps": 4.0,
        },
        "qwen_vl": {
            "python": sys.executable,
            "model_id": "Qwen/Qwen3-VL-4B-Instruct",
            "model_digest": "c" * 64,
            "model_revision": "d" * 40,
            "adapter_checkpoint": str((assets / "qwen-adapter").resolve()),
            "checkpoint_root": str(assets.resolve()),
            "device": "cpu",
            "dtype": "float32",
            "max_new_tokens": 128,
            "timeout_seconds": 300.0,
        },
    }
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class _UnusedRequester:
    def request(self, request: BridgeRequest, timeout: float) -> BridgeResponse:
        del request, timeout
        raise AssertionError("unused backend transport was called")


class _PolicyRequester:
    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.requests: list[BridgeRequest] = []

    def request(self, request: BridgeRequest, timeout: float) -> BridgeResponse:
        del timeout
        self.requests.append(request)
        if self.backend == "openpi":
            result: dict[str, object] = {"actions": [[float(index)] * 8 for index in range(15)]}
        else:
            result = {
                "dtype": "float32",
                "actions": [[[1.0] * 8, [2.0] * 8]],
                "delta_indices": [0.0, 0.25],
                "action_keys": ["arm"],
            }
        return BridgeResponse(request.request_id, "ok", cast(JsonObject, result))


def _write_mp4(path: Path, frame_count: int, fps: float) -> None:
    frames = np.stack([np.full((2, 4, 3), index, dtype=np.uint8) for index in range(frame_count)])
    writer = imageio_ffmpeg.write_frames(
        str(path), (4, 2), fps=fps, macro_block_size=1, ffmpeg_log_level="error"
    )
    writer.send(None)
    try:
        for frame in frames:
            writer.send(frame.tobytes())
    finally:
        writer.close()


class _CtrlRequester:
    def __init__(self) -> None:
        self.requests: list[BridgeRequest] = []

    def request(self, request: BridgeRequest, timeout: float) -> BridgeResponse:
        del timeout
        self.requests.append(request)
        payload = request.payload
        actions = cast(dict[str, object], payload["actions"])
        values = cast(list[list[float]], actions["values"])
        rollout_dir = request.artifact_dir / f"rollout-{len(self.requests)}"
        rollout_dir.mkdir()
        frames: list[Path] = []
        for index in range(len(values)):
            frame = rollout_dir / f"frame_{index:06d}.png"
            Image.new("RGB", (4, 2), (index, index, index)).save(frame)
            frames.append(frame)
        video = rollout_dir / "rollout.mp4"
        fps = float(cast(float, payload["fps"]))
        _write_mp4(video, len(values), fps)
        result: dict[str, object] = {
            "rollout_id": request.request_id,
            "task_id": payload["task_id"],
            "scene_id": payload["scene_id"],
            "expert_id": payload["expert_id"],
            "frame_paths": [str(frame.resolve()) for frame in frames],
            "video_path": str(video.resolve()),
            "actions": actions,
            "metrics": {"fps": fps, "frame_count": len(values)},
            "provenance": {
                "backend": "ctrl-world",
                "revision": CTRL_WORLD_PIN,
                "trajectory_id": payload["trajectory_id"],
                "start_index": payload["start_index"],
            },
            "output_manifest": combined_manifest(
                [
                    *(
                        file_manifest(frame, request.artifact_dir, "rollout frame")
                        for frame in frames
                    ),
                    file_manifest(video, request.artifact_dir, "rollout video"),
                ]
            ),
        }
        return BridgeResponse(request.request_id, "ok", cast(JsonObject, result))


class _QwenRequester:
    def __init__(self) -> None:
        self.requests: list[BridgeRequest] = []

    def request(self, request: BridgeRequest, timeout: float) -> BridgeResponse:
        del timeout
        self.requests.append(request)
        payload = request.payload
        provenance = {
            key: payload[key]
            for key in (
                "task_id",
                "scene_id",
                "trajectory_id",
                "checkpoint",
                "checkpoint_digest",
            )
        }
        return BridgeResponse(
            request.request_id,
            "ok",
            {
                "status": "ok",
                "text": json.dumps(
                    {"score": 0.95, "label": "NONE", "rationale": "motion succeeded"}
                ),
                "provenance": provenance,
            },
        )


class _TaskSource:
    def __init__(self, task: TaskSpec) -> None:
        self._task = task

    def next_task(self, *, operation_id: str) -> TaskSpec:
        assert operation_id
        return self._task


class _SceneAnchor:
    def __init__(self, scene: SceneRecord) -> None:
        self._scene = scene

    def anchor(self, task: TaskSpec, *, operation_id: str) -> SceneRecord:
        del task
        assert operation_id
        return self._scene


class _Router:
    def __init__(self, experts: tuple[ExpertSpec, ...]) -> None:
        self._experts = experts

    def shortlist(
        self,
        task: TaskSpec,
        scene: SceneRecord,
        memory: ExperienceMemory,
        *,
        operation_id: str,
    ) -> tuple[ExpertSpec, ...]:
        del task, scene, memory
        assert operation_id
        return self._experts


class _Scorer:
    def __init__(self, rollouts: dict[str, str]) -> None:
        self._rollouts = rollouts

    def score(self, task: TaskSpec, rollout: Rollout, *, operation_id: str) -> CandidateScore:
        del task
        assert operation_id
        self._rollouts[rollout.expert_id] = rollout.rollout_id
        world_score = 0.9 if rollout.expert_id == "pi05" else 0.8
        return CandidateScore.from_components(
            rollout.expert_id, rollout.rollout_id, 0.8, world_score, 0.25
        )


class _RobotExecutor:
    def __init__(self, media_root: Path, rollouts: dict[str, str]) -> None:
        self._media_root = media_root
        self._rollouts = rollouts

    def execute(
        self,
        task: TaskSpec,
        scene: SceneRecord,
        expert: ExpertSpec,
        actions: ActionChunk,
        *,
        operation_id: str,
    ) -> Trajectory:
        del task, scene
        assert operation_id
        frame = self._media_root / f"trajectory-{expert.expert_id}.png"
        Image.new("RGB", (4, 2), (20, 30, 40)).save(frame)
        return Trajectory(
            f"trajectory-{expert.expert_id}",
            self._rollouts[expert.expert_id],
            (frame,),
            actions,
            {"executor": "private-robot-port"},
        )


def _private_provider(
    captured: list[RuntimeBackendMetadata],
    config: EvoWorldConfig,
    task: TaskSpec,
    scene: SceneRecord,
    media_root: Path,
) -> Callable[[EvoWorldConfig, Path, RuntimeBackendMetadata], CoreRuntimePorts]:
    rollouts: dict[str, str] = {}

    def provider(
        requested: EvoWorldConfig, requested_run: Path, metadata: RuntimeBackendMetadata
    ) -> CoreRuntimePorts:
        captured.append(metadata)
        assert not hasattr(metadata, "policies")
        assert not hasattr(metadata, "projectors")
        assert not hasattr(metadata, "world_model")
        assert not hasattr(metadata, "evaluator")
        with pytest.raises(FrozenInstanceError):
            metadata.runtime_manifest_digest = "0" * 64  # type: ignore[misc]
        deterministic = build_deterministic_adapters(
            replace(requested, mode="dry_run", executor=None), requested_run
        )
        atomic_write_json(
            deterministic.store.run_dir / "manifest.json",
            {
                "config_digest": config_digest(config),
                "resolved_config": resolved_manifest(config),
            },
        )
        return CoreRuntimePorts(
            _TaskSource(task),
            _SceneAnchor(scene),
            _Router(config.experts),
            _Scorer(rollouts),
            _RobotExecutor(media_root, rollouts),
            deterministic.trainer,
            deterministic.memory,
            deterministic.store,
        )

    return provider


def _prepare_media(manifest: Path) -> tuple[TaskSpec, SceneRecord]:
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    media = Path(raw["media_root"])
    exterior = media / "exterior.png"
    wrist = media / "wrist.png"
    front = media / "front.png"
    for path in (exterior, wrist, front):
        Image.new("RGB", (4, 2), (10, 20, 30)).save(path)
    conditioning = media / "conditioning.mp4"
    _write_mp4(conditioning, 2, 4.0)
    annotation = {
        "texts": ["move block"],
        "states": [[0.0] * 7 for _ in range(15)],
        "joints": [[0.0] * 8 for _ in range(15)],
        "video_length": 15,
        "videos": [{"video_path": conditioning.name}],
    }
    (media / "annotation" / "val" / "0001.json").write_text(
        json.dumps(annotation), encoding="utf-8"
    )
    task = TaskSpec("task-1", "move block")
    scene = SceneRecord(
        "scene-union",
        (conditioning, exterior, wrist, front),
        {
            "exterior_image_1_left": str(exterior),
            "wrist_image_left": str(wrist),
            "joint_position": [0.0] * 7,
            "gripper_position": [0.0],
            "video": {"front": [str(front)]},
            "state": {"arm": [[0.0] * 8]},
            "ctrl_world_trajectory_id": "0001",
            "ctrl_world_start_index": 0,
        },
    )
    return task, scene


@pytest.mark.backend
def test_real_composition_builds_public_adapters_and_requires_private_ports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    manifest = _runtime_manifest(tmp_path)
    task, scene = _prepare_media(manifest)
    openpi = _PolicyRequester("openpi")
    groot = _PolicyRequester("groot")
    ctrl = _CtrlRequester()
    qwen = _QwenRequester()
    metadata: list[RuntimeBackendMetadata] = []
    project_operations: list[str] = []
    original_project = AffineActionProjector.project

    def record_project(
        self: AffineActionProjector,
        task_value: TaskSpec,
        scene: SceneRecord,
        actions: ActionChunk,
        *,
        operation_id: str,
    ) -> ActionChunk:
        project_operations.append(operation_id)
        return original_project(self, task_value, scene, actions, operation_id=operation_id)

    base_provider = _private_provider(metadata, config, task, scene, tmp_path / "media")

    def provider(
        requested: EvoWorldConfig,
        requested_run: Path,
        backend_metadata: RuntimeBackendMetadata,
    ) -> CoreRuntimePorts:
        ports = base_provider(requested, requested_run, backend_metadata)

        def replacement(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise AssertionError("provider replaced a public backend capability")

        monkeypatch.setattr(real_composition, "OpenPiPolicyAdapter", replacement)
        monkeypatch.setattr(real_composition, "_validated_adapter_set", replacement)
        return ports

    monkeypatch.setenv(RUNTIME_MANIFEST_ENV, str(manifest))
    monkeypatch.setattr("evoworld.real_composition._load_provider", lambda _: provider)
    monkeypatch.setattr(
        "evoworld.real_composition.build_openpi_process_bridge", lambda *a, **k: openpi
    )
    monkeypatch.setattr(
        "evoworld.real_composition.build_groot_process_bridge", lambda *a, **k: groot
    )
    monkeypatch.setattr(
        "evoworld.real_composition.build_ctrl_world_process_bridge", lambda *a, **k: ctrl
    )
    monkeypatch.setattr(
        "evoworld.real_composition.build_qwen_process_bridge",
        lambda *a, **k: qwen,
    )
    monkeypatch.setattr(AffineActionProjector, "project", record_project)

    adapters = build_real_adapters(config, (tmp_path / "runs" / "real").resolve())

    assert type(adapters.policies["pi05"]) is ProjectedExpertPolicy
    assert type(adapters.policies["groot"]) is ProjectedExpertPolicy
    assert type(adapters.policies["pi05"].raw_policy) is OpenPiPolicyAdapter
    assert type(adapters.policies["groot"].raw_policy) is GrootPolicyAdapter
    assert type(adapters.policies["pi05"].projector) is AffineActionProjector
    assert type(adapters.policies["groot"].projector) is AffineActionProjector
    assert type(adapters.world_model) is CtrlWorldAdapter
    assert type(adapters.evaluator) is QwenVlEvaluator
    assert metadata[0].schema_version == 2
    assert {item.expert_id for item in metadata[0].experts} == {"pi05", "groot"}

    result = EvoWorldPipeline(config, adapters).run(1)[0]

    assert result.round_id == "round-000000"
    assert len(openpi.requests) == 1
    assert len(groot.requests) == 1
    assert openpi.requests[0].request_id.endswith(":act:pi05:raw")
    assert groot.requests[0].request_id.endswith(":act:groot:raw")
    assert any(value.endswith(":act:pi05:project") for value in project_operations)
    assert any(value.endswith(":act:groot:project") for value in project_operations)
    assert len(ctrl.requests) == 2
    for request in ctrl.requests:
        sent_actions = cast(dict[str, object], request.payload["actions"])
        values = cast(list[list[float]], sent_actions["values"])
        assert sent_actions["schema_name"] == "cartesian_pose_7d"
        assert all(len(row) == 7 for row in values)
    state = adapters.store.load_state("round-000000")
    assert state is not None
    assert {candidate.expert.expert_id for candidate in state.candidates} == {"pi05", "groot"}
    assert all(candidate.actions is not None for candidate in state.candidates)
    assert all(candidate.rollout is not None for candidate in state.candidates)
    assert qwen.requests


def test_runtime_schema_one_has_explicit_migration_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _runtime_manifest(tmp_path, schema_version=1)
    monkeypatch.setenv(RUNTIME_MANIFEST_ENV, str(manifest))

    with pytest.raises(RuntimeManifestError, match="schema version 1.*migrate"):
        _load_runtime_manifest()


def test_runtime_manifest_rejects_symlink_unknown_and_incomplete_projectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _runtime_manifest(tmp_path)
    alias = tmp_path / "runtime-link.json"
    alias.symlink_to(manifest)
    monkeypatch.setenv(RUNTIME_MANIFEST_ENV, str(alias))
    with pytest.raises(RuntimeManifestError, match="regular file"):
        _load_runtime_manifest()

    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["unexpected"] = True
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setenv(RUNTIME_MANIFEST_ENV, str(manifest))
    with pytest.raises(RuntimeManifestError, match="fields"):
        _load_runtime_manifest()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw["action_projectors"].pop("groot"), "exactly match"),
        (
            lambda raw: raw["action_projectors"]["pi05"].__setitem__("content_digest", "0" * 64),
            "content_digest",
        ),
        (
            lambda raw: raw["action_projectors"]["pi05"].__setitem__("source_digest", "0" * 64),
            "content_digest",
        ),
    ],
)
def test_projector_manifest_fails_before_transport_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], object],
    message: str,
) -> None:
    manifest = _runtime_manifest(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    mutation(raw)
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setenv(RUNTIME_MANIFEST_ENV, str(manifest))
    monkeypatch.setattr(
        "evoworld.real_composition.build_openpi_process_bridge",
        lambda *a, **k: pytest.fail("transport must not build"),
    )

    with pytest.raises(RuntimeManifestError, match=message):
        build_real_adapters(_config(tmp_path), (tmp_path / "run-invalid").resolve())


def test_projector_asset_path_is_bound_into_runtime_digest(tmp_path: Path) -> None:
    manifest = _runtime_manifest(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    project_root = tmp_path / "project"
    ctrl_world_config = project_root / "third_party/ctrl_world/config.py"
    projector_source = project_root / "src/evoworld/adapters/projection.py"
    ctrl_world_config.parent.mkdir(parents=True)
    projector_source.parent.mkdir(parents=True)
    ctrl_world_config.write_text("MODEL_CONFIG = {}\n", encoding="utf-8")
    projector_source.write_bytes(PROJECTOR_SOURCE_PATH.read_bytes())
    original = _runtime_digest(
        raw,
        project_root,
        Path(raw["asset_root"]),
        raw["ctrl_world"],
        raw["qwen_vl"],
    )
    config = Path(raw["action_projectors"]["pi05"]["config_path"])
    replacement = config.with_name("replacement.json")
    replacement.write_bytes(config.read_bytes())
    raw["action_projectors"]["pi05"]["config_path"] = str(replacement)

    changed = _runtime_digest(
        raw,
        project_root,
        Path(raw["asset_root"]),
        raw["ctrl_world"],
        raw["qwen_vl"],
    )

    assert changed != original


def test_private_store_manifest_must_match_requested_real_config(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dry = build_deterministic_adapters(
        replace(config, mode="dry_run", executor=None),
        (tmp_path / "runs" / "wrong-config").resolve(),
    )

    with pytest.raises(RuntimeManifestError, match="does not match real config"):
        _validate_store_manifest(dry.store, config)

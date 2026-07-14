"""Integration coverage for normally constructed public process adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import imageio_ffmpeg
import numpy as np
from PIL import Image

from evoworld.adapters.content import combined_manifest, file_manifest
from evoworld.adapters.ctrl_world import CTRL_WORLD_PIN, CtrlWorldAdapter, CtrlWorldAssets
from evoworld.adapters.groot import GrootPolicyAdapter
from evoworld.adapters.openpi import OpenPiPolicyAdapter
from evoworld.adapters.process import BridgeRequest, BridgeResponse
from evoworld.adapters.qwen_vl import QwenVlEvaluator
from evoworld.contracts import (
    ActionChunk,
    ExperienceRecord,
    ExpertSpec,
    JsonObject,
    JsonValue,
    SceneRecord,
    TaskSpec,
    Trajectory,
)
from evoworld.ports import ExpertPolicy, TrajectoryEvaluator, WorldModel

CHECKPOINT_DIGEST = "c" * 64
MODEL_DIGEST = "d" * 64
MODEL_REVISION = "e" * 40


class _Requester:
    def __init__(self, result: JsonObject) -> None:
        self.result = result
        self.requests: list[BridgeRequest] = []

    def request(self, request: BridgeRequest, timeout: float) -> BridgeResponse:
        del timeout
        self.requests.append(request)
        result = dict(self.result)
        if request.operation == "imagine":
            result["rollout_id"] = request.request_id
        if request.operation == "evaluate":
            result["provenance"] = {
                key: request.payload[key]
                for key in (
                    "task_id",
                    "scene_id",
                    "trajectory_id",
                    "checkpoint",
                    "checkpoint_digest",
                )
            }
        return BridgeResponse(request.request_id, "ok", result)


class _Memory:
    def lookup(
        self, task: TaskSpec, scene: SceneRecord, *, operation_id: str
    ) -> tuple[ExperienceRecord, ...]:
        del task, scene, operation_id
        return ()

    def append(self, record: ExperienceRecord, *, operation_id: str) -> None:
        raise AssertionError("evaluation must not append memory")


def _write_mp4(path: Path, frames: np.ndarray, fps: float = 4.0) -> None:
    height, width = frames.shape[1:3]
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        fps=fps,
        macro_block_size=1,
        ffmpeg_log_level="error",
    )
    writer.send(None)
    try:
        for frame in frames:
            writer.send(frame.tobytes())
    finally:
        writer.close()


def _ctrl_fixture(tmp_path: Path) -> tuple[CtrlWorldAssets, Path, Path, Path, JsonObject]:
    repo = tmp_path / "source" / "third_party" / "ctrl_world"
    assets = tmp_path / "assets"
    media = tmp_path / "media"
    artifact_root = tmp_path / "artifacts"
    artifact_dir = artifact_root / "ctrl_world"
    for directory in (
        repo / "scripts",
        repo / "models",
        assets / "svd",
        assets / "clip",
        media / "annotation" / "val",
        media / "videos" / "val" / "0001",
        artifact_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    (repo / "config.py").write_text("class wm_args: pass\n", encoding="utf-8")
    (repo / "scripts" / "rollout_replay_traj.py").write_text(
        "class agent: pass\n", encoding="utf-8"
    )
    (repo / "models" / "ctrl_world.py").write_text("class CrtlWorld: pass\n", encoding="utf-8")
    checkpoint = assets / "checkpoint.pt"
    stats = assets / "stats.json"
    checkpoint.write_bytes(b"checkpoint")
    stats.write_text("{}", encoding="utf-8")
    conditioning = media / "videos" / "val" / "0001" / "0.mp4"
    conditioning.write_bytes(b"conditioning")
    annotation = {
        "texts": ["do thing"],
        "states": [[0.0] * 7 for _ in range(2)],
        "joints": [[0.0] * 8 for _ in range(2)],
        "videos": [{"video_path": "videos/val/0001/0.mp4"}],
        "video_length": 2,
    }
    (media / "annotation" / "val" / "0001.json").write_text(
        json.dumps(annotation), encoding="utf-8"
    )
    rollout_dir = artifact_dir / "rollout-output"
    rollout_dir.mkdir()
    frame_paths: list[str] = []
    for index in range(2):
        frame = rollout_dir / f"frame_{index:06d}.png"
        Image.new("RGB", (4, 2), (index, index, index)).save(frame)
        frame_paths.append(str(frame.resolve()))
    video = rollout_dir / "rollout.mp4"
    _write_mp4(video, np.zeros((2, 2, 4, 3), dtype=np.uint8))
    actions = ActionChunk("pi05", ((0.0,) * 7, (1.0,) * 7), (0.0, 0.25), "cartesian_pose_7d")
    response: JsonObject = {
        "rollout_id": "replaced-by-requester",
        "task_id": "task-1",
        "scene_id": "scene-1",
        "expert_id": "pi05",
        "frame_paths": cast(JsonValue, frame_paths),
        "video_path": str(video.resolve()),
        "actions": {
            "expert_id": actions.expert_id,
            "schema_name": actions.schema_name,
            "values": [list(row) for row in actions.actions],
            "timestamps": list(actions.timestamps),
        },
        "metrics": {"fps": 4.0, "frame_count": 2},
        "provenance": {
            "backend": "ctrl-world",
            "revision": CTRL_WORLD_PIN,
            "trajectory_id": "0001",
            "start_index": 0,
        },
        "output_manifest": combined_manifest(
            [
                *(file_manifest(Path(path), artifact_dir, "rollout frame") for path in frame_paths),
                file_manifest(video, artifact_dir, "rollout video"),
            ]
        ),
    }
    return (
        CtrlWorldAssets(
            repo, repo / "config.py", assets / "svd", assets / "clip", media, checkpoint, stats
        ),
        artifact_root,
        artifact_dir,
        conditioning,
        response,
    )


def test_all_public_adapters_construct_and_replay_verified_receipts(tmp_path: Path) -> None:
    media = tmp_path / "policy-media"
    openpi_artifacts = tmp_path / "openpi-artifacts"
    groot_artifacts = tmp_path / "groot-artifacts"
    media.mkdir()
    openpi_artifacts.mkdir()
    groot_artifacts.mkdir()
    exterior = media / "exterior.png"
    wrist = media / "wrist.png"
    Image.new("RGB", (4, 3)).save(exterior)
    Image.new("RGB", (4, 3)).save(wrist)
    task = TaskSpec("task-1", "do thing")

    openpi_requester = _Requester({"actions": [[0.0] * 8 for _ in range(15)]})
    openpi: ExpertPolicy = OpenPiPolicyAdapter(
        openpi_requester,
        "pi05",
        "pi05",
        artifact_dir=openpi_artifacts,
        allowed_media_root=media,
        checkpoint_identity="pi05-integration",
        checkpoint_digest=CHECKPOINT_DIGEST,
    )
    openpi_scene = SceneRecord(
        "openpi-scene",
        context={
            "exterior_image_1_left": str(exterior),
            "wrist_image_left": str(wrist),
            "joint_position": [0.0] * 7,
            "gripper_position": [0.0],
        },
    )
    first_openpi = openpi.act(task, openpi_scene, operation_id="round-1:openpi")
    assert openpi.act(task, openpi_scene, operation_id="round-1:openpi") == first_openpi
    assert len(openpi_requester.requests) == 1

    groot_requester = _Requester(
        {
            "dtype": "float32",
            "actions": [[[0.0, 0.0], [1.0, 1.0]]],
            "delta_indices": [0, 1],
            "action_keys": ["arm"],
        }
    )
    groot: ExpertPolicy = GrootPolicyAdapter(
        groot_requester,
        "groot",
        artifact_dir=groot_artifacts,
        allowed_media_root=media,
        checkpoint_identity="groot-integration",
        checkpoint_digest=CHECKPOINT_DIGEST,
    )
    groot_scene = SceneRecord(
        "groot-scene",
        context={"video": {"front": [str(exterior)]}, "state": {"arm": [[0.0, 0.0]]}},
    )
    first_groot = groot.act(task, groot_scene, operation_id="round-1:groot")
    assert groot.act(task, groot_scene, operation_id="round-1:groot") == first_groot
    assert len(groot_requester.requests) == 1

    ctrl_assets, artifact_root, ctrl_dir, conditioning, ctrl_response = _ctrl_fixture(tmp_path)
    ctrl_requester = _Requester(ctrl_response)
    world_model: WorldModel = CtrlWorldAdapter(
        ctrl_requester,
        ctrl_assets,
        artifact_dir=ctrl_dir,
        artifact_root=artifact_root,
        asset_root=tmp_path / "assets",
        media_root=tmp_path / "media",
    )
    ctrl_scene = SceneRecord(
        "scene-1",
        (conditioning,),
        {"ctrl_world_trajectory_id": "0001", "ctrl_world_start_index": 0},
    )
    ctrl_expert = ExpertSpec("pi05", "openpi", (), None, "cartesian_pose_7d")
    ctrl_actions = ActionChunk("pi05", ((0.0,) * 7, (1.0,) * 7), (0.0, 0.25), "cartesian_pose_7d")
    rollout = world_model.imagine(
        task, ctrl_scene, ctrl_expert, ctrl_actions, operation_id="round-1:ctrl"
    )
    assert (
        world_model.imagine(
            task, ctrl_scene, ctrl_expert, ctrl_actions, operation_id="round-1:ctrl"
        )
        == rollout
    )
    assert len(ctrl_requester.requests) == 1

    qwen_dir = artifact_root / "qwen_vl"
    qwen_dir.mkdir()
    qwen_requester = _Requester(
        {"status": "ok", "text": '{"score":0.9,"label":"NONE","rationale":"complete"}'}
    )
    evaluator: TrajectoryEvaluator = QwenVlEvaluator(
        qwen_requester,
        artifact_dir=qwen_dir,
        artifact_root=artifact_root,
        media_root=tmp_path,
        model_digest=MODEL_DIGEST,
        model_revision=MODEL_REVISION,
    )
    trajectory = Trajectory(
        "trajectory-1", rollout.rollout_id, rollout.frame_paths, rollout.actions
    )
    diagnosis = evaluator.evaluate(
        task, ctrl_scene, trajectory, _Memory(), operation_id="round-1:qwen"
    )
    assert (
        evaluator.evaluate(task, ctrl_scene, trajectory, _Memory(), operation_id="round-1:qwen")
        == diagnosis
    )
    assert len(qwen_requester.requests) == 1

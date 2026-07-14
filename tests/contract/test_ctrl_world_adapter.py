"""Contracts for the pinned Ctrl-World rollout boundary."""

from __future__ import annotations

import ast
from collections.abc import Sequence
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import cast

import imageio_ffmpeg
import numpy as np
from PIL import Image
import pytest

from evoworld.adapters.content import combined_manifest, file_manifest
from evoworld.adapters.ctrl_world import (
    CTRL_WORLD_PIN,
    AdapterSchemaError,
    CtrlWorldAdapter,
    CtrlWorldAssets,
    build_ctrl_world_process_bridge,
    build_ctrl_world_rollout_command,
    map_ctrl_response,
    validate_ctrl_actions,
)
from evoworld.adapters.process import (
    BridgeRemoteError,
    BridgeRequest,
    BridgeResponse,
    BridgeTimeoutError,
    ProcessBridge,
)
from evoworld.adapters.revisions import (
    EXPECTED_REVISIONS,
    GitRunner,
    SubmoduleDirtyError,
    SubmoduleRevisionError,
)
from evoworld.contracts import ActionChunk, ExpertSpec, JsonObject, Rollout, SceneRecord, TaskSpec
from evoworld.ports import OperationConflictError

PROJECT_ROOT = Path(__file__).parents[2]
_BRIDGE_PATH = PROJECT_ROOT / "integrations" / "ctrl_world" / "rollout_bridge.py"
_BRIDGE_SPEC = importlib.util.spec_from_file_location("ctrl_world_rollout_bridge", _BRIDGE_PATH)
assert _BRIDGE_SPEC is not None and _BRIDGE_SPEC.loader is not None
_BRIDGE_MODULE = importlib.util.module_from_spec(_BRIDGE_SPEC)
_BRIDGE_SPEC.loader.exec_module(_BRIDGE_MODULE)
handle_request = _BRIDGE_MODULE.handle_request


class _Requester:
    def __init__(self, result: JsonObject, *, response_id: str | None = None) -> None:
        self.result = result
        self.response_id = response_id
        self.requests: list[BridgeRequest] = []

    def request(self, request: BridgeRequest, timeout: float) -> BridgeResponse:
        del timeout
        self.requests.append(request)
        result = dict(self.result)
        result["rollout_id"] = request.request_id
        return BridgeResponse(
            request_id=request.request_id if self.response_id is None else self.response_id,
            status="ok",
            result=result,
        )


@pytest.mark.backend
def test_pinned_replay_agent_seam_matches_the_bridge() -> None:
    repo = PROJECT_ROOT / "third_party" / "ctrl_world"
    rollout_source = (repo / "scripts" / "rollout_replay_traj.py").read_text(encoding="utf-8")
    tree = ast.parse(rollout_source)
    agent_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "agent"
    )
    methods = {
        node.name: [argument.arg for argument in node.args.args]
        for node in agent_class.body
        if isinstance(node, ast.FunctionDef)
    }

    assert methods["__init__"] == ["self", "args"]
    assert methods["get_traj_info"] == ["self", "id", "start_idx", "steps"]
    assert methods["forward_wm"] == [
        "self",
        "action_cond",
        "video_latent_true",
        "video_latent_cond",
        "his_cond",
        "text",
    ]
    assert "args.val_model_path = args.ckpt_path" in rollout_source
    assert "self.model = CrtlWorld(args)" in rollout_source
    assert "CtrlWorldDiffusionPipeline.__call__(" in rollout_source
    config_source = (repo / "config.py").read_text(encoding="utf-8")
    for field in ("svd_model_path", "clip_model_path", "ckpt_path", "pi_ckpt"):
        assert field in config_source


def _png(path: Path, color: int = 0) -> None:
    Image.new("RGB", (4, 2), (color, color, color)).save(path)


def _mp4(path: Path, frames: np.ndarray, fps: float = 4.0) -> None:
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


def _response_fixture(
    tmp_path: Path, frame_count: int = 2
) -> tuple[dict[str, object], ActionChunk]:
    rollout_dir = tmp_path / "ctrl-world-fixture"
    rollout_dir.mkdir(parents=True)
    frame_paths = []
    for index in range(frame_count):
        frame = rollout_dir / f"frame_{index:06d}.png"
        _png(frame, index)
        frame_paths.append(str(frame.resolve()))
    video = rollout_dir / "rollout.mp4"
    video_frames = np.stack(
        [np.full((2, 4, 3), index, dtype=np.uint8) for index in range(frame_count)]
    )
    _mp4(video, video_frames)
    actions = ActionChunk(
        "pi05",
        tuple((float(index),) * 7 for index in range(frame_count)),
        tuple(index / 4.0 for index in range(frame_count)),
        "cartesian_pose_7d",
    )
    response: dict[str, object] = {
        "rollout_id": "rollout-1",
        "task_id": "task-1",
        "scene_id": "scene-1",
        "expert_id": "pi05",
        "frame_paths": frame_paths,
        "video_path": str(video.resolve()),
        "actions": {
            "expert_id": actions.expert_id,
            "schema_name": actions.schema_name,
            "values": [list(row) for row in actions.actions],
            "timestamps": list(actions.timestamps),
        },
        "metrics": {"fps": 4.0, "frame_count": frame_count},
        "provenance": {
            "backend": "ctrl-world",
            "revision": CTRL_WORLD_PIN,
            "trajectory_id": "0001",
            "start_index": 0,
        },
        "output_manifest": combined_manifest(
            [
                *(file_manifest(Path(path), tmp_path, "rollout frame") for path in frame_paths),
                file_manifest(video, tmp_path, "rollout video"),
            ]
        ),
    }
    return response, actions


def _map_fixture(response: dict[str, object], tmp_path: Path, actions: ActionChunk) -> Rollout:
    return map_ctrl_response(
        response,
        artifact_dir=tmp_path,
        expected_request_id="rollout-1",
        expected_task_id="task-1",
        expected_scene_id="scene-1",
        expected_expert_id="pi05",
        expected_actions=actions,
        expected_trajectory_id="0001",
        expected_start_index=0,
    )


def test_ctrl_world_requires_finite_seven_dimensional_cartesian_actions() -> None:
    with pytest.raises(AdapterSchemaError, match="7-D"):
        validate_ctrl_actions(ActionChunk("pi05", ((0.0,) * 8,), (0.0,), "joint"))
    with pytest.raises(ValueError, match="finite"):
        ActionChunk(
            "pi05",
            ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("nan")),),
            (0.0,),
            "cartesian_pose_7d",
        )
    with pytest.raises(AdapterSchemaError, match="project"):
        validate_ctrl_actions(ActionChunk("pi05", ((0.0,) * 7,), (0.0,), "joint_velocity"))
    with pytest.raises(AdapterSchemaError, match="strictly increasing"):
        validate_ctrl_actions(
            ActionChunk("pi05", ((0.0,) * 7, (1.0,) * 7), (0.0, 0.0), "cartesian_pose_7d")
        )
    with pytest.raises(AdapterSchemaError, match="non-negative"):
        validate_ctrl_actions(ActionChunk("pi05", ((0.0,) * 7,), (-0.1,), "cartesian_pose_7d"))


def test_rollout_response_returns_absolute_decodable_ordered_frames(tmp_path: Path) -> None:
    response, actions = _response_fixture(tmp_path, frame_count=16)

    rollout = _map_fixture(response, tmp_path, actions)

    assert len(rollout.frame_paths) == 16
    assert all(path.is_absolute() for path in rollout.frame_paths)
    assert rollout.metrics["provenance"] == {
        "backend": "ctrl-world",
        "revision": CTRL_WORLD_PIN,
        "trajectory_id": "0001",
        "start_index": 0,
    }


@pytest.mark.parametrize(
    "failure",
    ["unknown", "identity", "rollout", "provenance", "revision", "frame-count", "action"],
)
def test_response_rejects_unknown_fields_and_identity_mismatches(
    tmp_path: Path, failure: str
) -> None:
    response, actions = _response_fixture(tmp_path)
    if failure == "unknown":
        response["extra"] = True
    elif failure == "identity":
        response["scene_id"] = "other"
    elif failure == "rollout":
        response["rollout_id"] = "other"
    elif failure == "provenance":
        provenance = cast(dict[str, object], response["provenance"])
        provenance["trajectory_id"] = "other"
    elif failure == "revision":
        provenance = cast(dict[str, object], response["provenance"])
        provenance["revision"] = "0" * 40
    elif failure == "frame-count":
        response["metrics"] = {"fps": 4.0, "frame_count": 99}
    else:
        action = cast(dict[str, object], response["actions"])
        action["timestamps"] = [0.0, 8.0]

    with pytest.raises(AdapterSchemaError):
        _map_fixture(response, tmp_path, actions)


@pytest.mark.parametrize("failure", ["symlink", "order", "decode", "video"])
def test_response_rejects_unsafe_or_invalid_media(tmp_path: Path, failure: str) -> None:
    response, actions = _response_fixture(tmp_path)
    frames = cast(list[str], response["frame_paths"])
    if failure == "symlink":
        outside = tmp_path.parent / f"outside-{tmp_path.name}.png"
        _png(outside)
        Path(frames[0]).unlink()
        Path(frames[0]).symlink_to(outside)
    elif failure == "order":
        frames.reverse()
    elif failure == "decode":
        Path(frames[0]).write_bytes(b"not-an-image")
    else:
        Path(cast(str, response["video_path"])).write_bytes(b"\x00\x00\x00\x18ftypisom")

    with pytest.raises(AdapterSchemaError):
        _map_fixture(response, tmp_path, actions)


def test_response_rejects_symlinked_or_nested_rollout_directories(tmp_path: Path) -> None:
    response, actions = _response_fixture(tmp_path)
    outside = tmp_path.parent / f"outside-rollout-{tmp_path.name}"
    outside.mkdir()
    outside_frames = []
    for index in range(2):
        frame = outside / f"frame_{index:06d}.png"
        _png(frame, index)
        outside_frames.append(str(frame))
    outside_video = outside / "rollout.mp4"
    _mp4(outside_video, np.zeros((2, 2, 4, 3), dtype=np.uint8))
    alias = tmp_path / "rollout-alias"
    alias.symlink_to(outside, target_is_directory=True)
    response["frame_paths"] = [str(alias / Path(path).name) for path in outside_frames]
    response["video_path"] = str(alias / "rollout.mp4")

    with pytest.raises(AdapterSchemaError, match="symlink"):
        _map_fixture(response, tmp_path, actions)

    response, actions = _response_fixture(tmp_path / "nested")
    nested_root = tmp_path / "nested"
    original = Path(cast(list[str], response["frame_paths"])[0]).parent
    deeper = original / "deeper"
    deeper.mkdir()
    for path in list(original.glob("frame_*.png")) + [original / "rollout.mp4"]:
        path.rename(deeper / path.name)
    response["frame_paths"] = [str(deeper / f"frame_{index:06d}.png") for index in range(2)]
    response["video_path"] = str(deeper / "rollout.mp4")
    with pytest.raises(AdapterSchemaError, match="direct"):
        _map_fixture(response, nested_root, actions)


@pytest.mark.parametrize("failure", ["count", "dimensions", "fps"])
def test_response_decodes_video_and_matches_frames_and_metrics(
    tmp_path: Path, failure: str
) -> None:
    response, actions = _response_fixture(tmp_path)
    video = Path(cast(str, response["video_path"]))
    video.unlink()
    if failure == "count":
        frames = np.zeros((1, 2, 4, 3), dtype=np.uint8)
        fps = 4.0
    elif failure == "dimensions":
        frames = np.zeros((2, 2, 6, 3), dtype=np.uint8)
        fps = 4.0
    else:
        frames = np.zeros((2, 2, 4, 3), dtype=np.uint8)
        fps = 8.0
    _mp4(video, frames, fps)

    with pytest.raises(AdapterSchemaError, match="video media"):
        _map_fixture(response, tmp_path, actions)


def _fake_workspace(tmp_path: Path, instruction: str = "do thing") -> dict[str, Path]:
    source = tmp_path / "source"
    repo = source / "third_party" / "ctrl_world"
    assets = tmp_path / "assets"
    media = tmp_path / "media"
    output = tmp_path / "output"
    artifact = output / "request"
    for path in (
        repo / "scripts",
        repo / "models",
        source / "third_party" / "openpi",
        source / "third_party" / "isaac_groot",
        assets / "svd",
        assets / "clip",
        media / "annotation" / "val",
        media / "videos" / "val" / "0001",
        artifact,
    ):
        path.mkdir(parents=True, exist_ok=True)
    (repo / "models" / "ctrl_world.py").write_text("class CrtlWorld: pass\n", encoding="utf-8")
    (assets / "checkpoint.pt").write_bytes(b"checkpoint")
    (assets / "stats.json").write_text(
        json.dumps({"state_01": [0.0] * 7, "state_99": [1.0] * 7}), encoding="utf-8"
    )
    conditioning = media / "videos" / "val" / "0001" / "0.mp4"
    _mp4(conditioning, np.zeros((8, 2, 4, 3), dtype=np.uint8), fps=15.0)
    annotation = {
        "texts": [instruction],
        "states": [[0.0] * 7 for _ in range(8)],
        "joints": [[0.0] * 8 for _ in range(8)],
        "videos": [{"video_path": "videos/val/0001/0.mp4"}],
        "video_length": 8,
    }
    (media / "annotation" / "val" / "0001.json").write_text(
        json.dumps(annotation), encoding="utf-8"
    )
    (repo / "config.py").write_text(
        textwrap.dedent(
            """
            from pathlib import Path
            Path(__file__).with_name("import-order.txt").write_text("config\\n")

            class wm_args:
                svd_model_path = "/cephfs/default-svd"
                clip_model_path = "/cephfs/default-clip"
                ckpt_path = "/cephfs/default-checkpoint"
                pi_ckpt = "/cephfs/default-policy"
                action_adapter = "/cephfs/default-adapter"
                num_history = 2
                text_cond = True

                def __init__(self, task_type):
                    self.task_type = task_type
            """
        ),
        encoding="utf-8",
    )
    (repo / "torch.py").write_text(
        textwrap.dedent(
            """
            class Tensor:
                def __getitem__(self, key):
                    return self

                def unsqueeze(self, dimension):
                    return self

            def cat(values, dim=0):
                assert values
                return Tensor()
            """
        ),
        encoding="utf-8",
    )
    (repo / "mediapy.py").write_text(
        "import imageio_ffmpeg\n"
        "def write_video(path, frames, fps):\n"
        "    height, width = frames.shape[1:3]\n"
        "    writer = imageio_ffmpeg.write_frames(path, (width, height), fps=fps, macro_block_size=1, ffmpeg_log_level='error')\n"
        "    writer.send(None)\n"
        "    try:\n"
        "        for frame in frames: writer.send(frame.tobytes())\n"
        "    finally: writer.close()\n",
        encoding="utf-8",
    )
    (repo / "scripts" / "rollout_replay_traj.py").write_text(
        textwrap.dedent(
            """
            from pathlib import Path
            import time
            import numpy as np
            import torch

            marker = Path(__file__).parents[1] / "import-order.txt"
            assert marker.read_text() == "config\\n"
            marker.write_text("config\\nrollout\\n")

            class agent:
                def __init__(self, args):
                    self.args = args
                    values = [getattr(args, name) for name in dir(args) if not name.startswith("_")]
                    assert not any(isinstance(value, str) and value.startswith("/cephfs") for value in values)
                    marker.write_text(marker.read_text() + "agent\\n")

                def get_traj_info(self, trajectory_id, start_idx=0, steps=8):
                    assert trajectory_id == "0001" and start_idx == 0
                    states = np.zeros((steps, 7), dtype=np.float32)
                    joints = np.zeros((steps, 8), dtype=np.float32)
                    return states, joints, [], [torch.Tensor()], """
            + repr(instruction)
            + """

                def forward_wm(self, action_cond, video_latent_true, video_latent_cond, his_cond=None, text=None):
                    assert action_cond.shape == (self.args.num_history + self.args.num_frames, 7)
                    assert len(video_latent_true) == 1 and his_cond is not None and text == """
            + repr(instruction)
            + """
                    if text == "sleep":
                        time.sleep(0.2)
                    marker.write_text(marker.read_text() + "forward_wm\\n")
                    composite = np.full((self.args.num_frames, 2, 6, 3), 11, dtype=np.uint8)
                    ground_truth = np.full((2, self.args.num_frames, 2, 3, 3), 22, dtype=np.uint8)
                    predicted = np.stack([
                        np.full((self.args.num_frames, 2, 3, 3), 31, dtype=np.uint8),
                        np.full((self.args.num_frames, 2, 3, 3), 32, dtype=np.uint8),
                    ])
                    return composite, ground_truth, predicted, object()
            """
        ),
        encoding="utf-8",
    )
    return {
        "source": source,
        "repo": repo,
        "assets": assets,
        "media": media,
        "output": output,
        "artifact": artifact,
        "conditioning": conditioning,
    }


def _fake_revision_runner(source: Path) -> GitRunner:
    def run(command: Sequence[str]) -> str:
        operation = tuple(command[3:])
        if operation == ("submodule", "status", "--recursive"):
            return "".join(f" {revision} {path}\n" for path, revision in EXPECTED_REVISIONS.items())
        checkout = Path(command[2])
        relative = str(checkout.relative_to(source))
        if operation == ("rev-parse", "HEAD"):
            return EXPECTED_REVISIONS[relative] + "\n"
        if operation == ("status", "--porcelain"):
            return ""
        raise AssertionError(f"unexpected git operation: {command}")

    return run


def _adapter(
    paths: dict[str, Path],
    *,
    timeout: float = 2.0,
    requester: ProcessBridge | _Requester | None = None,
) -> CtrlWorldAdapter:
    bridge = (
        build_ctrl_world_process_bridge(
            paths["repo"],
            paths["output"],
            paths["assets"],
            paths["media"],
            backend_python=Path(sys.executable),
            project_root=PROJECT_ROOT,
            revision_runner=_fake_revision_runner(paths["source"]),
        )
        if requester is None
        else requester
    )
    assets = CtrlWorldAssets(
        repo_root=paths["repo"],
        config_path=paths["repo"] / "config.py",
        model_root=paths["assets"] / "svd",
        text_encoder_root=paths["assets"] / "clip",
        dataset_root=paths["media"],
        checkpoint_path=paths["assets"] / "checkpoint.pt",
        stats_path=paths["assets"] / "stats.json",
    )
    return CtrlWorldAdapter(
        bridge,
        assets,
        artifact_dir=paths["artifact"],
        artifact_root=paths["output"],
        asset_root=paths["assets"],
        media_root=paths["media"],
        timeout_seconds=timeout,
    )


def _domain_request(
    paths: dict[str, Path], instruction: str = "do thing"
) -> tuple[TaskSpec, SceneRecord, ExpertSpec, ActionChunk]:
    return (
        TaskSpec("task-1", instruction),
        SceneRecord(
            "scene-1",
            (paths["conditioning"],),
            {"ctrl_world_trajectory_id": "0001", "ctrl_world_start_index": 0},
        ),
        ExpertSpec("pi05", "openpi", (), None, "cartesian_pose_7d"),
        ActionChunk(
            "pi05",
            ((0.0,) * 7, (1.0,) * 7),
            (0.0, 0.25),
            "cartesian_pose_7d",
        ),
    )


def test_real_process_bridge_calls_pinned_agent_forward_in_order(tmp_path: Path) -> None:
    paths = _fake_workspace(tmp_path)
    task, scene, expert, actions = _domain_request(paths)

    rollout = _adapter(paths).imagine(
        task, scene, expert, actions, operation_id="round-1:imagine:expert-1"
    )

    assert rollout.actions == actions
    assert len(rollout.frame_paths) == 2
    with Image.open(rollout.frame_paths[0]) as image:
        pixels = np.asarray(image.convert("RGB"))
    assert pixels.shape == (2, 6, 3)
    assert np.all(pixels[:, :3] == 31)
    assert np.all(pixels[:, 3:] == 32)
    assert not np.any(pixels == 11)
    assert not np.any(pixels == 22)
    assert (paths["repo"] / "import-order.txt").read_text() == (
        "config\nrollout\nagent\nforward_wm\n"
    )
    log_dir = next((paths["artifact"] / ".process").iterdir())
    process_stdout = (log_dir / "process.stdout.log").read_text(encoding="utf-8")
    assert process_stdout.count("\n") == 1
    assert json.loads(process_stdout)["status"] == "ok"


def test_adapter_rejects_blank_operation_id(tmp_path: Path) -> None:
    paths = _fake_workspace(tmp_path)
    task, scene, expert, actions = _domain_request(paths)

    with pytest.raises(AdapterSchemaError, match="operation_id"):
        _adapter(paths).imagine(task, scene, expert, actions, operation_id="\t")


def test_adapter_correlates_exact_id_retries_and_rejects_mismatch(tmp_path: Path) -> None:
    paths = _fake_workspace(tmp_path)
    task, scene, expert, actions = _domain_request(paths)
    response, response_actions = _response_fixture(paths["artifact"])
    assert response_actions == actions
    requester = _Requester(cast(JsonObject, response))
    adapter = _adapter(paths, requester=requester)

    adapter.imagine(task, scene, expert, actions, operation_id="round-1:imagine")
    adapter.imagine(task, scene, expert, actions, operation_id="round-1:imagine")
    adapter.imagine(task, scene, expert, actions, operation_id="round-2:imagine")
    assert [request.request_id for request in requester.requests] == [
        "round-1:imagine",
        "round-2:imagine",
    ]
    changed_actions = ActionChunk(
        actions.expert_id,
        tuple(tuple(value + 1.0 for value in row) for row in actions.actions),
        actions.timestamps,
        actions.schema_name,
    )
    with pytest.raises(OperationConflictError):
        adapter.imagine(
            task,
            scene,
            expert,
            changed_actions,
            operation_id="round-1:imagine",
        )

    mismatched = _adapter(
        paths,
        requester=_Requester(cast(JsonObject, response), response_id="other"),
    )
    with pytest.raises(AdapterSchemaError, match="correlation"):
        mismatched.imagine(task, scene, expert, actions, operation_id="round-3:imagine")


def test_completed_retry_rehashes_ctrl_world_output_receipt(tmp_path: Path) -> None:
    paths = _fake_workspace(tmp_path)
    task, scene, expert, actions = _domain_request(paths)
    response, _ = _response_fixture(paths["artifact"])
    requester = _Requester(cast(JsonObject, response))
    adapter = _adapter(paths, requester=requester)

    rollout = adapter.imagine(task, scene, expert, actions, operation_id="round-1:imagine")
    _png(rollout.frame_paths[0], color=200)

    with pytest.raises(OperationConflictError, match="content changed"):
        adapter.imagine(task, scene, expert, actions, operation_id="round-1:imagine")
    assert len(requester.requests) == 1


def test_operation_id_cannot_escape_the_artifact_directory(tmp_path: Path) -> None:
    paths = _fake_workspace(tmp_path)
    task, scene, expert, actions = _domain_request(paths)

    rollout = _adapter(paths).imagine(
        task,
        scene,
        expert,
        actions,
        operation_id="../../outside/round-1:imagine",
    )

    assert all(path.is_relative_to(paths["artifact"]) for path in rollout.frame_paths)
    assert not (tmp_path / "outside").exists()


def test_core_import_does_not_load_ctrl_world_torch_or_numpy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import evoworld.adapters.ctrl_world; "
                "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules; "
                "assert 'models.ctrl_world' not in sys.modules"
            ),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_command_is_absolute_fixed_and_rejects_upstream_default(tmp_path: Path) -> None:
    paths = _fake_workspace(tmp_path)
    command = build_ctrl_world_rollout_command(
        paths["repo"],
        paths["output"],
        paths["assets"],
        paths["media"],
        backend_python=Path(sys.executable),
        project_root=PROJECT_ROOT,
        revision_runner=_fake_revision_runner(paths["source"]),
    )
    assert Path(command[0]).is_absolute()
    assert Path(command[1]).is_absolute()
    assert command[2:] == [
        "--repo-root",
        str(paths["repo"].resolve()),
        "--asset-root",
        str(paths["assets"].resolve()),
        "--media-root",
        str(paths["media"].resolve()),
        "--output-root",
        str(paths["output"].resolve()),
        "--revision",
        CTRL_WORLD_PIN,
    ]
    with pytest.raises(AdapterSchemaError, match="/cephfs"):
        build_ctrl_world_rollout_command(
            Path("/cephfs/default-repo"),
            paths["output"],
            paths["assets"],
            paths["media"],
            backend_python=Path(sys.executable),
            project_root=PROJECT_ROOT,
            revision_runner=_fake_revision_runner(paths["source"]),
        )


@pytest.mark.parametrize("failure", ["wrong", "dirty"])
def test_revision_gate_rejects_unpinned_or_dirty_checkout_before_launch(
    tmp_path: Path, failure: str
) -> None:
    paths = _fake_workspace(tmp_path)
    valid = _fake_revision_runner(paths["source"])

    def failing(command: Sequence[str]) -> str:
        if command[3:] == ("rev-parse", "HEAD") and command[2].endswith("third_party/ctrl_world"):
            return "0" * 40 + "\n" if failure == "wrong" else CTRL_WORLD_PIN + "\n"
        if (
            failure == "dirty"
            and command[3:] == ("status", "--porcelain")
            and command[2].endswith("third_party/ctrl_world")
        ):
            return " M models/ctrl_world.py\n"
        return valid(command)

    error_type = SubmoduleRevisionError if failure == "wrong" else SubmoduleDirtyError
    with pytest.raises(error_type):
        build_ctrl_world_process_bridge(
            paths["repo"],
            paths["output"],
            paths["assets"],
            paths["media"],
            backend_python=Path(sys.executable),
            project_root=PROJECT_ROOT,
            revision_runner=failing,
        )
    assert not (paths["repo"] / "import-order.txt").exists()


def test_adapter_rejects_dataset_symlink_escape_and_expert_mismatch(tmp_path: Path) -> None:
    paths = _fake_workspace(tmp_path)
    task, scene, expert, actions = _domain_request(paths)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"\x00\x00\x00\x18ftypisom")
    paths["conditioning"].unlink()
    paths["conditioning"].symlink_to(outside)

    with pytest.raises(AdapterSchemaError, match="contained"):
        _adapter(paths).imagine(task, scene, expert, actions, operation_id="round-1:imagine:pi05")

    paths = _fake_workspace(tmp_path / "mismatch")
    task, scene, expert, actions = _domain_request(paths)
    mismatched = ActionChunk("other", actions.actions, actions.timestamps, actions.schema_name)
    with pytest.raises(AdapterSchemaError, match="identities"):
        _adapter(paths).imagine(
            task, scene, expert, mismatched, operation_id="round-1:imagine:pi05"
        )


def test_adapter_rejects_missing_assets_before_launch(tmp_path: Path) -> None:
    paths = _fake_workspace(tmp_path)
    (paths["assets"] / "checkpoint.pt").unlink()

    with pytest.raises(AdapterSchemaError, match="checkpoint_path is missing"):
        _adapter(paths)


@pytest.mark.parametrize(
    "failure",
    [
        "missing-length",
        "zero-length",
        "short-states",
        "state-width",
        "joint-length",
        "texts",
        "instruction",
        "range",
    ],
)
def test_droid_preflight_fails_before_model_initialization(tmp_path: Path, failure: str) -> None:
    paths = _fake_workspace(tmp_path)
    annotation_path = paths["media"] / "annotation" / "val" / "0001.json"
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    if failure == "missing-length":
        annotation.pop("video_length")
    elif failure == "zero-length":
        annotation["video_length"] = 0
    elif failure == "short-states":
        annotation["states"].pop()
    elif failure == "state-width":
        annotation["states"][0] = [0.0] * 6
    elif failure == "joint-length":
        annotation["joints"].pop()
    elif failure == "texts":
        annotation["texts"] = []
    elif failure == "instruction":
        annotation["texts"] = ["different task"]
    else:
        annotation["video_length"] = 1
        annotation["states"] = annotation["states"][:1]
        annotation["joints"] = annotation["joints"][:1]
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    task, scene, expert, actions = _domain_request(paths)

    with pytest.raises(AdapterSchemaError):
        _adapter(paths).imagine(task, scene, expert, actions, operation_id="round-1:imagine:pi05")

    assert not (paths["repo"] / "import-order.txt").exists()


def test_droid_preflight_accepts_action_length_instead_of_video_length(tmp_path: Path) -> None:
    paths = _fake_workspace(tmp_path)
    annotation_path = paths["media"] / "annotation" / "val" / "0001.json"
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotation.pop("video_length")
    annotation["action"] = [[0.0] * 7 for _ in range(8)]
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    task, scene, expert, actions = _domain_request(paths)

    rollout = _adapter(paths).imagine(
        task, scene, expert, actions, operation_id="round-1:imagine:pi05"
    )

    assert len(rollout.frame_paths) == len(actions.actions)


def test_backend_returns_structured_error_for_unhandled_upstream_default(tmp_path: Path) -> None:
    paths = _fake_workspace(tmp_path)
    config = paths["repo"] / "config.py"
    config.write_text(
        config.read_text(encoding="utf-8")
        + "\nwm_args.unpublished_default = '/cephfs/must-not-leak'\n",
        encoding="utf-8",
    )
    task, scene, expert, actions = _domain_request(paths)

    with pytest.raises(BridgeRemoteError):
        _adapter(paths).imagine(task, scene, expert, actions, operation_id="round-1:imagine:pi05")

    log_dir = next((paths["artifact"] / ".process").iterdir())
    response = json.loads((log_dir / "process.stdout.log").read_text(encoding="utf-8"))
    assert response["status"] == "error"
    assert response["error"] == {"type": "ValueError"}
    assert (paths["repo"] / "import-order.txt").read_text() == "config\nrollout\n"


def test_adapter_propagates_process_timeout(tmp_path: Path) -> None:
    paths = _fake_workspace(tmp_path, instruction="sleep")
    task, scene, expert, actions = _domain_request(paths, instruction="sleep")

    with pytest.raises(BridgeTimeoutError):
        _adapter(paths, timeout=0.02).imagine(
            task, scene, expert, actions, operation_id="round-1:imagine:pi05"
        )

    log_dir = next((paths["artifact"] / ".process").iterdir())
    assert (log_dir / "process.error.json").is_file()


def test_backend_rejects_unknown_payload_before_importing_source(tmp_path: Path) -> None:
    paths = _fake_workspace(tmp_path)
    payload: JsonObject = {"unexpected": True}
    request = importlib.import_module("evoworld.adapters.process").BridgeRequest(
        "request-1", "imagine", paths["artifact"], payload
    )

    with pytest.raises(ValueError, match="unknown or missing"):
        handle_request(
            request,
            repo_root=paths["repo"].resolve(),
            asset_root=paths["assets"].resolve(),
            media_root=paths["media"].resolve(),
            output_root=paths["output"].resolve(),
            verified_revision=CTRL_WORLD_PIN,
        )
    assert not (paths["repo"] / "import-order.txt").exists()

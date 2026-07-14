"""Isolated JSONL bridge for the pinned Ctrl-World replay agent."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType
from typing import cast

from evoworld.adapters.content import (
    combined_manifest,
    directory_manifest,
    file_manifest,
    verify_manifest,
)
from evoworld.adapters.ctrl_world import (
    CTRL_ACTION_SCHEMA,
    CTRL_WORLD_PIN,
    validate_ctrl_actions,
)
from evoworld.adapters.process import BridgeRequest
from evoworld.contracts import ActionChunk, JsonObject, JsonValue

_PAYLOAD_FIELDS = {
    "task_id",
    "instruction",
    "scene_id",
    "expert_id",
    "revision",
    "trajectory_id",
    "start_index",
    "fps",
    "actions",
    "repo_root",
    "config_path",
    "model_root",
    "text_encoder_root",
    "dataset_root",
    "checkpoint_path",
    "stats_path",
    "conditioning_media_path",
    "asset_manifest",
    "output_dir",
}


def _string(value: JsonValue | object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _object(value: JsonValue | object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be an object")
    return cast(Mapping[str, object], value)


def _configured_root(path: Path, field_name: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be absolute")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{field_name} must be a directory")
    return resolved


def _path(
    value: JsonValue | object,
    root: Path,
    field_name: str,
    *,
    directory: bool,
) -> Path:
    path = value if isinstance(value, Path) else Path(_string(value, field_name))
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be absolute")
    if len(path.parts) > 1 and path.parts[1] == "cephfs":
        raise ValueError(f"{field_name} must not use an upstream /cephfs default")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{field_name} is missing") from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field_name} must be contained by its configured root") from error
    expected = resolved.is_dir() if directory else resolved.is_file()
    if not expected:
        raise ValueError(f"{field_name} has the wrong path type")
    return resolved


def _action_chunk(value: JsonValue | object, expected_expert_id: str) -> ActionChunk:
    action = _object(value, "actions")
    if set(action) != {"expert_id", "schema_name", "values", "timestamps"}:
        raise ValueError("actions fields do not match the bridge contract")
    expert_id = _string(action["expert_id"], "actions.expert_id")
    if expert_id != expected_expert_id:
        raise ValueError("action expert identity does not match")
    if action["schema_name"] != CTRL_ACTION_SCHEMA:
        raise ValueError("actions must use cartesian_pose_7d")
    rows = action["values"]
    timestamps = action["timestamps"]
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("actions.values must be an array")
    if isinstance(timestamps, (str, bytes)) or not isinstance(timestamps, Sequence):
        raise ValueError("actions.timestamps must be an array")
    try:
        chunk = ActionChunk(
            expert_id,
            tuple(tuple(cast(Sequence[float], row)) for row in rows),
            tuple(cast(Sequence[float], timestamps)),
            CTRL_ACTION_SCHEMA,
        )
        return validate_ctrl_actions(chunk)
    except (TypeError, ValueError) as error:
        raise ValueError("actions must be a finite timestamped 7-D chunk") from error


def _load_source_module(path: Path, module_name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load pinned source module {path.name}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _assert_no_upstream_defaults(args: object) -> None:
    leaked: list[str] = []
    for name in dir(args):
        if name.startswith("_"):
            continue
        value = getattr(args, name)
        if isinstance(value, str) and value.startswith("/cephfs"):
            leaked.append(name)
    if leaked:
        raise ValueError(f"upstream /cephfs defaults remain configured: {', '.join(leaked)}")


def _configure_official_args(
    config_module: ModuleType,
    *,
    model_root: Path,
    text_encoder_root: Path,
    checkpoint_path: Path,
    dataset_root: Path,
    stats_path: Path,
    output_dir: Path,
    frame_count: int,
) -> object:
    factory = getattr(config_module, "wm_args", None)
    if factory is None or not callable(factory):
        raise RuntimeError("pinned config.py must expose callable wm_args")
    args = factory(task_type="replay")
    explicit = {
        "svd_model_path": str(model_root),
        "clip_model_path": str(text_encoder_root),
        "ckpt_path": str(checkpoint_path),
        "val_model_path": str(checkpoint_path),
        "dataset_root_path": str(dataset_root),
        "val_dataset_dir": str(dataset_root),
        "dataset_meta_info_path": str(stats_path.parent),
        "data_stat_path": str(stats_path),
        "save_dir": str(output_dir),
        "pi_ckpt": None,
        "action_adapter": None,
        "task_type": "replay",
        "action_dim": 7,
        "num_frames": frame_count,
    }
    for name, value in explicit.items():
        setattr(args, name, value)
    _assert_no_upstream_defaults(args)
    return args


def _droid_episode(
    dataset_root: Path,
    trajectory_id: str,
    conditioning_media: Path,
    start_index: int,
    frame_count: int,
    instruction: str,
) -> None:
    annotation_path = dataset_root / "annotation" / "val" / f"{trajectory_id}.json"
    annotation_path = _path(str(annotation_path), dataset_root, "DROID annotation", directory=False)
    try:
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("DROID annotation is not valid JSON") from error
    if not isinstance(annotation, dict) or not {"texts", "states", "joints", "videos"} <= set(
        annotation
    ):
        raise ValueError("DROID annotation is missing rollout fields")
    action = annotation.get("action")
    video_length = annotation.get("video_length")
    if action is not None:
        if not isinstance(action, list) or not action:
            raise ValueError("DROID action must be a nonempty array")
        length = len(action)
        if video_length is not None:
            if (
                isinstance(video_length, bool)
                or not isinstance(video_length, int)
                or video_length <= 0
            ):
                raise ValueError("DROID video_length must be a positive integer")
            if video_length != length:
                raise ValueError("DROID action and video_length must agree")
    elif isinstance(video_length, bool) or not isinstance(video_length, int) or video_length <= 0:
        raise ValueError("DROID annotation requires positive action or video_length")
    else:
        length = video_length
    texts = annotation["texts"]
    states = annotation["states"]
    joints = annotation["joints"]
    videos = annotation["videos"]
    if (
        not isinstance(texts, list)
        or not texts
        or any(not isinstance(text, str) or not text.strip() for text in texts)
    ):
        raise ValueError("DROID annotation texts must contain an instruction")
    if texts[0].strip() != instruction.strip():
        raise ValueError("task instruction does not match the selected DROID episode")
    if start_index + frame_count > length:
        raise ValueError("DROID episode is too short for the requested action steps")
    for field_name, rows, width in (("states", states, 7), ("joints", joints, 8)):
        if not isinstance(rows, list) or len(rows) != length:
            raise ValueError(f"DROID {field_name} length must match the episode")
        for row in rows:
            if not isinstance(row, list) or len(row) != width:
                raise ValueError(f"DROID {field_name} rows must be {width}-D")
            for item in row:
                if (
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                ):
                    raise ValueError(f"DROID {field_name} must contain finite numbers")
    if not isinstance(videos, list) or not videos:
        raise ValueError("DROID videos must be a nonempty array")
    resolved: list[Path] = []
    for item in videos:
        if not isinstance(item, dict) or set(item) != {"video_path"}:
            raise ValueError("DROID video entry must contain only video_path")
        relative = item["video_path"]
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("DROID video_path must be relative")
        resolved.append(
            _path(dataset_root / relative, dataset_root, "DROID video_path", directory=False)
        )
    if conditioning_media not in resolved:
        raise ValueError("conditioning media does not belong to the DROID episode")


def _official_forward(
    *,
    repo_root: Path,
    config_path: Path,
    model_root: Path,
    text_encoder_root: Path,
    checkpoint_path: Path,
    dataset_root: Path,
    stats_path: Path,
    output_dir: Path,
    trajectory_id: str,
    start_index: int,
    instruction: str,
    actions: ActionChunk,
) -> object:
    """Construct the pinned replay agent and call its real forward_wm signature."""
    config_module = _load_source_module(config_path, "evoworld_ctrl_world_config")
    rollout_module = _load_source_module(
        repo_root / "scripts" / "rollout_replay_traj.py",
        "evoworld_ctrl_world_rollout_replay_traj",
    )
    agent_factory = getattr(rollout_module, "agent", None)
    if agent_factory is None or not callable(agent_factory):
        raise RuntimeError("pinned rollout_replay_traj.py must expose callable agent")
    args = _configure_official_args(
        config_module,
        model_root=model_root,
        text_encoder_root=text_encoder_root,
        checkpoint_path=checkpoint_path,
        dataset_root=dataset_root,
        stats_path=stats_path,
        output_dir=output_dir,
        frame_count=len(actions.actions),
    )
    official_agent = agent_factory(args)
    get_traj_info = getattr(official_agent, "get_traj_info", None)
    forward_wm = getattr(official_agent, "forward_wm", None)
    if not callable(get_traj_info) or not callable(forward_wm):
        raise RuntimeError("pinned agent must expose get_traj_info and forward_wm")

    numpy = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    trajectory = get_traj_info(
        trajectory_id,
        start_idx=start_index,
        steps=len(actions.actions),
    )
    if not isinstance(trajectory, tuple) or len(trajectory) != 5:
        raise RuntimeError("pinned get_traj_info return signature changed")
    recorded_actions, _joint_positions, _video_frames, video_latents, dataset_instruction = (
        trajectory
    )
    if (
        not isinstance(dataset_instruction, str)
        or dataset_instruction.strip() != instruction.strip()
    ):
        raise ValueError("task instruction does not match the selected DROID episode")
    if not isinstance(video_latents, list) or not video_latents:
        raise RuntimeError("pinned get_traj_info returned no video latents")
    history_count = getattr(args, "num_history", None)
    if isinstance(history_count, bool) or not isinstance(history_count, int) or history_count <= 0:
        raise RuntimeError("pinned wm_args.num_history must be a positive integer")

    first_latent = torch.cat([latent[0] for latent in video_latents], dim=1).unsqueeze(0)
    history_latents = torch.cat([first_latent] * history_count, dim=0).unsqueeze(0)
    recorded_array = numpy.asarray(recorded_actions)
    if recorded_array.ndim != 2 or recorded_array.shape[1] != 7:
        raise RuntimeError("pinned DROID states must be a two-dimensional 7-D array")
    history_actions = numpy.repeat(recorded_array[0:1], history_count, axis=0)
    action_array = numpy.asarray(actions.actions, dtype=numpy.float32)
    action_condition = numpy.concatenate((history_actions, action_array), axis=0)
    true_future_latents = [latent[: len(actions.actions)] for latent in video_latents]
    forward_result = forward_wm(
        action_condition,
        true_future_latents,
        first_latent,
        his_cond=history_latents,
        text=instruction if bool(getattr(args, "text_cond", True)) else None,
    )
    if not isinstance(forward_result, tuple) or len(forward_result) != 4:
        raise RuntimeError("pinned forward_wm return signature changed")
    predicted = numpy.asarray(forward_result[2])
    if (
        predicted.ndim != 5
        or predicted.shape[0] <= 0
        or predicted.shape[1] != len(actions.actions)
        or predicted.shape[-1] != 3
        or predicted.dtype != numpy.uint8
    ):
        raise RuntimeError("pinned predicted videos must be uint8 (V,F,H,W,3)")
    views, frames, height, width, channels = predicted.shape
    return predicted.transpose(1, 2, 0, 3, 4).reshape(frames, height, views * width, channels)


def _atomic_media(
    frames: object,
    output_dir: Path,
    request_id: str,
    fps: float,
) -> tuple[list[str], str]:
    numpy = importlib.import_module("numpy")
    image_module = importlib.import_module("PIL.Image")
    media_module = importlib.import_module("mediapy")
    array = numpy.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3 or array.shape[0] <= 0:
        raise RuntimeError("official forward_wm frames must be shaped (F,H,W,3)")
    if array.dtype != numpy.uint8:
        raise RuntimeError("official forward_wm frames must use uint8 pixels")
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:16]
    final_dir = output_dir / f"ctrl-world-{digest}"
    if final_dir.exists():
        raise FileExistsError("rollout artifact directory already exists")
    staging = Path(tempfile.mkdtemp(prefix=f".ctrl-world-{digest}-", dir=output_dir))
    try:
        for index, frame in enumerate(array):
            image_module.fromarray(frame, mode="RGB").save(staging / f"frame_{index:06d}.png")
        video_path = staging / "rollout.mp4"
        media_module.write_video(str(video_path), array, fps=fps)
        if not video_path.is_file():
            raise RuntimeError("media writer did not create rollout.mp4")
        os.replace(staging, final_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    paths = [str(final_dir / f"frame_{index:06d}.png") for index in range(array.shape[0])]
    return paths, str(final_dir / "rollout.mp4")


def _finite_positive(value: JsonValue | object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return number


def handle_request(
    request: BridgeRequest,
    *,
    repo_root: Path,
    asset_root: Path,
    media_root: Path,
    output_root: Path,
    verified_revision: str,
) -> JsonObject:
    """Validate one request, call the official path, and publish atomic media."""
    if request.operation != "imagine":
        raise ValueError("Ctrl-World bridge supports only imagine")
    payload = request.payload
    if set(payload) != _PAYLOAD_FIELDS:
        raise ValueError("payload contains unknown or missing fields")
    if verified_revision != CTRL_WORLD_PIN or payload["revision"] != verified_revision:
        raise ValueError("verified Ctrl-World revision does not match the request")
    payload_repo = _path(payload["repo_root"], repo_root, "repo_root", directory=True)
    if payload_repo != repo_root:
        raise ValueError("payload repo_root does not match the configured checkout")
    config_path = _path(payload["config_path"], repo_root, "config_path", directory=False)
    if config_path != (repo_root / "config.py").resolve():
        raise ValueError("config_path must select the pinned config.py")
    model_root = _path(payload["model_root"], asset_root, "model_root", directory=True)
    text_encoder_root = _path(
        payload["text_encoder_root"], asset_root, "text_encoder_root", directory=True
    )
    checkpoint_path = _path(
        payload["checkpoint_path"], asset_root, "checkpoint_path", directory=False
    )
    stats_path = _path(payload["stats_path"], asset_root, "stats_path", directory=False)
    dataset_root = _path(payload["dataset_root"], media_root, "dataset_root", directory=True)
    conditioning_media = _path(
        payload["conditioning_media_path"],
        media_root,
        "conditioning_media_path",
        directory=False,
    )
    output_dir = _path(payload["output_dir"], output_root, "output_dir", directory=True)
    if output_dir != request.artifact_dir.resolve(strict=True):
        raise ValueError("output_dir must match request.artifact_dir")

    task_id = _string(payload["task_id"], "task_id")
    instruction = _string(payload["instruction"], "instruction")
    scene_id = _string(payload["scene_id"], "scene_id")
    expert_id = _string(payload["expert_id"], "expert_id")
    trajectory_id = _string(payload["trajectory_id"], "trajectory_id")
    start_index = payload["start_index"]
    if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 0:
        raise ValueError("start_index must be a non-negative integer")
    fps = _finite_positive(payload["fps"], "fps")
    actions = _action_chunk(payload["actions"], expert_id)
    annotation_path = dataset_root / "annotation" / "val" / f"{trajectory_id}.json"
    asset_manifest_value = payload["asset_manifest"]
    if not isinstance(asset_manifest_value, dict):
        raise ValueError("asset_manifest must be an object")
    asset_manifest: JsonObject = asset_manifest_value
    expected_asset_manifest = combined_manifest(
        [
            file_manifest(config_path, repo_root, "config_path"),
            directory_manifest(model_root, asset_root, "model_root"),
            directory_manifest(text_encoder_root, asset_root, "text_encoder_root"),
            file_manifest(checkpoint_path, asset_root, "checkpoint_path"),
            file_manifest(stats_path, asset_root, "stats_path"),
            file_manifest(annotation_path, dataset_root, "DROID annotation"),
            file_manifest(conditioning_media, media_root, "conditioning media"),
        ]
    )
    if asset_manifest != expected_asset_manifest:
        raise ValueError("Ctrl-World asset content manifest does not match")
    verify_manifest(asset_manifest, "Ctrl-World input assets")
    _droid_episode(
        dataset_root,
        trajectory_id,
        conditioning_media,
        start_index,
        len(actions.actions),
        instruction,
    )
    frames = _official_forward(
        repo_root=payload_repo,
        config_path=config_path,
        model_root=model_root,
        text_encoder_root=text_encoder_root,
        checkpoint_path=checkpoint_path,
        dataset_root=dataset_root,
        stats_path=stats_path,
        output_dir=output_dir,
        trajectory_id=trajectory_id,
        start_index=start_index,
        instruction=instruction,
        actions=actions,
    )
    frame_paths, video_path = _atomic_media(frames, output_dir, request.request_id, fps)
    verify_manifest(asset_manifest, "Ctrl-World input assets")
    output_manifest = combined_manifest(
        [
            *(file_manifest(Path(path), output_dir, "rollout frame") for path in frame_paths),
            file_manifest(Path(video_path), output_dir, "rollout video"),
        ]
    )
    action_json: JsonObject = {
        "expert_id": actions.expert_id,
        "schema_name": actions.schema_name,
        "values": [list(row) for row in actions.actions],
        "timestamps": list(actions.timestamps),
    }
    return {
        "rollout_id": request.request_id,
        "task_id": task_id,
        "scene_id": scene_id,
        "expert_id": expert_id,
        "frame_paths": cast(list[JsonValue], frame_paths),
        "video_path": video_path,
        "actions": action_json,
        "metrics": {"fps": fps, "frame_count": len(frame_paths)},
        "provenance": {
            "backend": "ctrl-world",
            "revision": verified_revision,
            "trajectory_id": trajectory_id,
            "start_index": start_index,
        },
        "output_manifest": output_manifest,
    }


def main() -> None:
    """Serve one request while keeping official logging off protocol stdout."""
    from integrations.common.jsonl_bridge import serve

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    arguments = parser.parse_args()
    repo_root = _configured_root(arguments.repo_root, "repo_root")
    asset_root = _configured_root(arguments.asset_root, "asset_root")
    media_root = _configured_root(arguments.media_root, "media_root")
    output_root = _configured_root(arguments.output_root, "output_root")
    if arguments.revision != CTRL_WORLD_PIN:
        raise ValueError("Ctrl-World revision must match the pinned checkout")
    serve(
        lambda request: handle_request(
            request,
            repo_root=repo_root,
            asset_root=asset_root,
            media_root=media_root,
            output_root=output_root,
            verified_revision=arguments.revision,
        )
    )


if __name__ == "__main__":
    main()

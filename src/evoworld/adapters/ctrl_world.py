"""Backend-neutral contracts for pinned Ctrl-World imagined rollouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Protocol, cast

from evoworld.adapters.content import (
    combined_manifest,
    directory_manifest,
    file_manifest,
    verify_manifest,
)
from evoworld.adapters.process import BridgeRequest, BridgeResponse, ProcessBridge
from evoworld.adapters.operations import OperationLedger
from evoworld.adapters.process import request_to_json
from evoworld.adapters.revisions import GitRunner, inspect_submodules
from evoworld.codec import rollout_from_json, rollout_to_json
from evoworld.contracts import (
    ActionChunk,
    ExpertSpec,
    JsonObject,
    JsonValue,
    Rollout,
    SceneRecord,
    TaskSpec,
)

CTRL_WORLD_PIN = "99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d"
CTRL_ACTION_SCHEMA = "cartesian_pose_7d"
_RESPONSE_FIELDS = {
    "rollout_id",
    "task_id",
    "scene_id",
    "expert_id",
    "frame_paths",
    "video_path",
    "actions",
    "metrics",
    "provenance",
    "output_manifest",
}


class AdapterSchemaError(ValueError):
    """A Ctrl-World value violates the stable EvoWorld boundary."""


class ProcessRequester(Protocol):
    """The narrow process surface used by the rollout adapter."""

    def request(self, request: BridgeRequest, timeout: float) -> BridgeResponse: ...


def _require_operation_id(operation_id: str) -> str:
    if not operation_id.strip():
        raise AdapterSchemaError("operation_id must not be blank")
    return operation_id


@dataclass(frozen=True, slots=True)
class CtrlWorldAssets:
    """Every heavyweight path required by the official replay agent."""

    repo_root: Path
    config_path: Path
    model_root: Path
    text_encoder_root: Path
    dataset_root: Path
    checkpoint_path: Path
    stats_path: Path


def _reject_upstream_default(path: Path, field_name: str) -> None:
    if len(path.parts) > 1 and path.parts[1] == "cephfs":
        raise AdapterSchemaError(f"{field_name} must not use an upstream /cephfs default")


def _existing_absolute(path: Path, field_name: str, *, directory: bool) -> Path:
    if not path.is_absolute():
        raise AdapterSchemaError(f"{field_name} must be absolute")
    _reject_upstream_default(path, field_name)
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise AdapterSchemaError(f"{field_name} is missing") from error
    expected = resolved.is_dir() if directory else resolved.is_file()
    if not expected:
        kind = "directory" if directory else "file"
        raise AdapterSchemaError(f"{field_name} must be a {kind}")
    return resolved


def _contained_path(
    path: Path,
    root: Path,
    field_name: str,
    *,
    directory: bool,
) -> Path:
    resolved = _existing_absolute(path, field_name, directory=directory)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise AdapterSchemaError(
            f"{field_name} must be contained by its configured root"
        ) from error
    return resolved


def _repo_root(path: Path) -> Path:
    repo = _existing_absolute(path, "repo_root", directory=True)
    required = (
        repo / "config.py",
        repo / "scripts" / "rollout_replay_traj.py",
        repo / "models" / "ctrl_world.py",
    )
    if not all(item.is_file() for item in required):
        raise AdapterSchemaError("repo_root does not contain the pinned Ctrl-World rollout source")
    return repo


def _python_executable(path: Path) -> Path:
    """Validate without dereferencing a virtual-environment launcher symlink."""
    if not path.is_absolute() or not path.is_file():
        raise AdapterSchemaError("backend_python must be an existing absolute file")
    _reject_upstream_default(path, "backend_python")
    return path


def _verified_revision(repo: Path, runner: GitRunner | None) -> str:
    source_root = repo.parent.parent
    expected_checkout = (source_root / "third_party" / "ctrl_world").resolve()
    if repo != expected_checkout:
        raise AdapterSchemaError("repo_root must be the audited third_party/ctrl_world checkout")
    revisions = (
        inspect_submodules(source_root)
        if runner is None
        else inspect_submodules(source_root, runner=runner)
    )
    revision = revisions.get("third_party/ctrl_world")
    if revision != CTRL_WORLD_PIN:
        raise AdapterSchemaError("audited Ctrl-World revision does not match the pin")
    return revision


def build_ctrl_world_rollout_command(
    repo_root: Path,
    artifact_root: Path,
    asset_root: Path,
    media_root: Path,
    *,
    backend_python: Path,
    project_root: Path,
    revision_runner: GitRunner | None = None,
) -> list[str]:
    """Build a shell-free command for the isolated rollout bridge."""
    repo = _repo_root(repo_root)
    output = _existing_absolute(artifact_root, "artifact_root", directory=True)
    assets = _existing_absolute(asset_root, "asset_root", directory=True)
    media = _existing_absolute(media_root, "media_root", directory=True)
    project = _existing_absolute(project_root, "project_root", directory=True)
    python = _python_executable(backend_python)
    bridge = (project / "integrations" / "ctrl_world" / "rollout_bridge.py").resolve()
    if not bridge.is_file():
        raise AdapterSchemaError("project_root is missing the Ctrl-World rollout bridge")
    revision = _verified_revision(repo, revision_runner)
    return [
        str(python),
        str(bridge),
        "--repo-root",
        str(repo),
        "--asset-root",
        str(assets),
        "--media-root",
        str(media),
        "--output-root",
        str(output),
        "--revision",
        revision,
    ]


def build_ctrl_world_process_bridge(
    repo_root: Path,
    artifact_root: Path,
    asset_root: Path,
    media_root: Path,
    *,
    backend_python: Path,
    project_root: Path,
    revision_runner: GitRunner | None = None,
) -> ProcessBridge:
    """Bind the real ProcessBridge to absolute Ctrl-World paths and a fixed cwd."""
    repo = _repo_root(repo_root)
    output = _existing_absolute(artifact_root, "artifact_root", directory=True)
    project = _existing_absolute(project_root, "project_root", directory=True)
    command = build_ctrl_world_rollout_command(
        repo,
        output,
        asset_root,
        media_root,
        backend_python=backend_python,
        project_root=project,
        revision_runner=revision_runner,
    )
    python_path = os.pathsep.join((str(project / "src"), str(project), str(repo)))
    return ProcessBridge(
        command,
        cwd=repo,
        artifact_root=output,
        env={"PYTHONIOENCODING": "utf-8", "PYTHONPATH": python_path},
    )


def validate_ctrl_actions(actions: ActionChunk) -> ActionChunk:
    """Require explicit finite 7-D Cartesian poses; never truncate policy actions."""
    if not actions.actions:
        raise AdapterSchemaError("Ctrl-World actions must not be empty")
    normalized: list[tuple[float, ...]] = []
    for row_index, row in enumerate(actions.actions):
        if len(row) != 7:
            raise AdapterSchemaError(f"Ctrl-World actions[{row_index}] must be 7-D")
        values: list[float] = []
        for column_index, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise AdapterSchemaError(
                    f"Ctrl-World actions[{row_index}][{column_index}] must be numeric"
                )
            number = float(value)
            if not math.isfinite(number):
                raise AdapterSchemaError(
                    f"Ctrl-World actions[{row_index}][{column_index}] must be finite"
                )
            values.append(number)
        normalized.append(tuple(values))
    if actions.schema_name != CTRL_ACTION_SCHEMA:
        raise AdapterSchemaError(
            "Ctrl-World requires cartesian_pose_7d; project policy-native actions explicitly"
        )
    timestamps: list[float] = []
    for index, value in enumerate(actions.timestamps):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AdapterSchemaError(f"timestamps[{index}] must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise AdapterSchemaError(f"timestamps[{index}] must be finite")
        if number < 0:
            raise AdapterSchemaError(f"timestamps[{index}] must be non-negative")
        if timestamps and number <= timestamps[-1]:
            raise AdapterSchemaError("timestamps must be strictly increasing")
        timestamps.append(number)
    return ActionChunk(
        actions.expert_id,
        tuple(normalized),
        tuple(timestamps),
        actions.schema_name,
    )


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise AdapterSchemaError(f"{field_name} must be an object")
    return cast(Mapping[str, object], value)


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdapterSchemaError(f"{field_name} must be a non-empty string")
    return value


def _response_actions(value: object) -> ActionChunk:
    action = _mapping(value, "actions")
    if set(action) != {"expert_id", "schema_name", "values", "timestamps"}:
        raise AdapterSchemaError("actions fields do not match the Ctrl-World contract")
    rows = action["values"]
    timestamps = action["timestamps"]
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise AdapterSchemaError("actions.values must be an array")
    if isinstance(timestamps, (str, bytes)) or not isinstance(timestamps, Sequence):
        raise AdapterSchemaError("actions.timestamps must be an array")
    try:
        chunk = ActionChunk(
            _string(action["expert_id"], "actions.expert_id"),
            tuple(tuple(cast(Sequence[float], row)) for row in rows),
            tuple(cast(Sequence[float], timestamps)),
            _string(action["schema_name"], "actions.schema_name"),
        )
    except (TypeError, ValueError) as error:
        raise AdapterSchemaError("actions are not a rectangular numeric chunk") from error
    return validate_ctrl_actions(chunk)


def _artifact_file(value: object, root: Path, field_name: str) -> Path:
    path = Path(_string(value, field_name))
    if not path.is_absolute():
        raise AdapterSchemaError(f"{field_name} must be absolute")
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise AdapterSchemaError(f"{field_name} must be contained by artifact_dir") from error
    if len(relative.parts) != 2 or any(part in {".", ".."} for part in relative.parts):
        raise AdapterSchemaError(
            f"{field_name} must be in one direct rollout directory under artifact_dir"
        )
    current = root
    for index, component in enumerate(relative.parts):
        current = current / component
        try:
            metadata = current.lstat()
        except OSError as error:
            raise AdapterSchemaError(f"{field_name} is missing") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise AdapterSchemaError(f"{field_name} must not contain a symlink component")
        if index == 0 and not stat.S_ISDIR(metadata.st_mode):
            raise AdapterSchemaError(f"{field_name} rollout parent must be a real directory")
    if not stat.S_ISREG(metadata.st_mode):
        raise AdapterSchemaError(f"{field_name} must be a regular file")
    return current


def _decode_frame(path: Path) -> tuple[int, int]:
    try:
        image_module = importlib.import_module("PIL.Image")
        with image_module.open(path) as image:
            image.verify()
        with image_module.open(path) as image:
            rgb = image.convert("RGB")
            rgb.load()
            size = cast(tuple[int, int], rgb.size)
    except Exception as error:
        raise AdapterSchemaError(f"frame media cannot be decoded: {path.name}") from error
    if size[0] <= 0 or size[1] <= 0:
        raise AdapterSchemaError("frame media dimensions must be positive")
    return size


def _decode_video(
    path: Path,
    *,
    expected_count: int,
    expected_size: tuple[int, int],
    expected_fps: float,
) -> None:
    if path.suffix.lower() != ".mp4":
        raise AdapterSchemaError("video_path must be an MP4 artifact")
    try:
        ffmpeg = importlib.import_module("imageio_ffmpeg")
        reader = ffmpeg.read_frames(str(path), pix_fmt="rgb24")
        try:
            metadata = next(reader)
            if not isinstance(metadata, Mapping):
                raise ValueError("video metadata is unavailable")
            raw_size = metadata.get("size")
            raw_fps = metadata.get("fps")
            if (
                not isinstance(raw_size, (tuple, list))
                or len(raw_size) != 2
                or tuple(raw_size) != expected_size
            ):
                raise ValueError("video dimensions do not match rollout frames")
            if (
                isinstance(raw_fps, bool)
                or not isinstance(raw_fps, (int, float))
                or not math.isclose(float(raw_fps), expected_fps, abs_tol=0.01)
            ):
                raise ValueError("video fps does not match rollout metrics")
            expected_bytes = expected_size[0] * expected_size[1] * 3
            decoded_count = 0
            for frame in reader:
                if not isinstance(frame, bytes) or len(frame) != expected_bytes:
                    raise ValueError("decoded video frame has the wrong size")
                decoded_count += 1
            if decoded_count != expected_count:
                raise ValueError("decoded video frame count does not match rollout frames")
        finally:
            reader.close()
    except AdapterSchemaError:
        raise
    except Exception as error:
        raise AdapterSchemaError("video media cannot be decoded consistently") from error


def map_ctrl_response(
    response: Mapping[str, object],
    *,
    artifact_dir: Path,
    expected_request_id: str,
    expected_task_id: str,
    expected_scene_id: str,
    expected_expert_id: str,
    expected_actions: ActionChunk,
    expected_trajectory_id: str,
    expected_start_index: int,
) -> Rollout:
    """Validate identity, action provenance, and contained ordered rollout artifacts."""
    if set(response) != _RESPONSE_FIELDS:
        raise AdapterSchemaError("Ctrl-World response contains unknown or missing fields")
    root = _existing_absolute(artifact_dir, "artifact_dir", directory=True)
    rollout_id = _string(response["rollout_id"], "rollout_id")
    if rollout_id != expected_request_id:
        raise AdapterSchemaError("rollout_id does not match the request")
    task_id = _string(response["task_id"], "task_id")
    scene_id = _string(response["scene_id"], "scene_id")
    expert_id = _string(response["expert_id"], "expert_id")
    if (task_id, scene_id, expert_id) != (
        expected_task_id,
        expected_scene_id,
        expected_expert_id,
    ):
        raise AdapterSchemaError("rollout task, scene, or expert identity does not match")
    returned_actions = _response_actions(response["actions"])
    if returned_actions != validate_ctrl_actions(expected_actions):
        raise AdapterSchemaError("rollout action identity does not match the request")

    frames_value = response["frame_paths"]
    if isinstance(frames_value, (str, bytes)) or not isinstance(frames_value, Sequence):
        raise AdapterSchemaError("frame_paths must be an array")
    frame_paths = tuple(
        _artifact_file(value, root, f"frame_paths[{index}]")
        for index, value in enumerate(frames_value)
    )
    if not frame_paths:
        raise AdapterSchemaError("frame_paths must not be empty")
    if len(frame_paths) != len(expected_actions.actions):
        raise AdapterSchemaError("frame_paths count must match the requested action steps")
    frame_parent = frame_paths[0].parent
    frame_size: tuple[int, int] | None = None
    for index, path in enumerate(frame_paths):
        if path.parent != frame_parent or path.name != f"frame_{index:06d}.png":
            raise AdapterSchemaError("frame paths must be ordered sequential PNG artifacts")
        decoded_size = _decode_frame(path)
        if frame_size is not None and decoded_size != frame_size:
            raise AdapterSchemaError("all rollout frames must have identical dimensions")
        frame_size = decoded_size

    metrics = _mapping(response["metrics"], "metrics")
    if set(metrics) != {"fps", "frame_count"}:
        raise AdapterSchemaError("metrics fields do not match the Ctrl-World contract")
    frame_count = metrics["frame_count"]
    fps = metrics["fps"]
    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise AdapterSchemaError("metrics.frame_count must be an integer")
    if frame_count != len(frame_paths):
        raise AdapterSchemaError("metrics.frame_count does not match frame_paths")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(float(fps)):
        raise AdapterSchemaError("metrics.fps must be finite")
    if float(fps) <= 0:
        raise AdapterSchemaError("metrics.fps must be positive")

    if frame_size is None:  # Defensive for static analysis; empty frames were rejected above.
        raise AdapterSchemaError("frame media dimensions are unavailable")
    video_path = _artifact_file(response["video_path"], root, "video_path")
    if video_path.parent != frame_parent:
        raise AdapterSchemaError("video_path must share the atomic rollout directory")
    _decode_video(
        video_path,
        expected_count=frame_count,
        expected_size=frame_size,
        expected_fps=float(fps),
    )
    output_manifest_value = response["output_manifest"]
    if not isinstance(output_manifest_value, dict):
        raise AdapterSchemaError("output_manifest must be an object")
    output_manifest: JsonObject = output_manifest_value
    expected_output_manifest = combined_manifest(
        [
            *(file_manifest(path, root, "rollout frame") for path in frame_paths),
            file_manifest(video_path, root, "rollout video"),
        ]
    )
    if output_manifest != expected_output_manifest:
        raise AdapterSchemaError("rollout output content manifest does not match")

    provenance = _mapping(response["provenance"], "provenance")
    if set(provenance) != {"backend", "revision", "trajectory_id", "start_index"}:
        raise AdapterSchemaError("provenance fields do not match the Ctrl-World contract")
    if provenance["backend"] != "ctrl-world" or provenance["revision"] != CTRL_WORLD_PIN:
        raise AdapterSchemaError("Ctrl-World provenance revision does not match")
    if not isinstance(provenance["start_index"], int) or isinstance(
        provenance["start_index"], bool
    ):
        raise AdapterSchemaError("provenance.start_index must be an integer")
    trajectory_id = _string(provenance["trajectory_id"], "provenance.trajectory_id")
    if (trajectory_id, provenance["start_index"]) != (
        expected_trajectory_id,
        expected_start_index,
    ):
        raise AdapterSchemaError("rollout trajectory provenance does not match the request")

    metrics_json: JsonObject = {
        "fps": float(fps),
        "frame_count": frame_count,
        "video_path": str(video_path),
        "provenance": cast(JsonValue, dict(provenance)),
        "output_manifest": output_manifest,
    }
    return Rollout(
        rollout_id=rollout_id,
        expert_id=expert_id,
        scene_id=scene_id,
        frame_paths=frame_paths,
        actions=returned_actions,
        metrics=metrics_json,
    )


def _annotation_length(record: Mapping[str, object]) -> int:
    action = record.get("action")
    video_length = record.get("video_length")
    if action is not None:
        if not isinstance(action, list) or not action:
            raise AdapterSchemaError("DROID action must be a nonempty array")
        length = len(action)
        if video_length is not None:
            if (
                isinstance(video_length, bool)
                or not isinstance(video_length, int)
                or video_length <= 0
            ):
                raise AdapterSchemaError("DROID video_length must be a positive integer")
            if video_length != length:
                raise AdapterSchemaError("DROID action and video_length must agree")
        return length
    if isinstance(video_length, bool) or not isinstance(video_length, int) or video_length <= 0:
        raise AdapterSchemaError("DROID annotation requires positive action or video_length")
    return video_length


def _finite_rows(value: object, length: int, width: int, field_name: str) -> None:
    if not isinstance(value, list) or len(value) != length:
        raise AdapterSchemaError(f"DROID {field_name} length must match the episode")
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != width:
            raise AdapterSchemaError(f"DROID {field_name}[{row_index}] must be {width}-D")
        for item in row:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise AdapterSchemaError(f"DROID {field_name} must be numeric")
            if not math.isfinite(float(item)):
                raise AdapterSchemaError(f"DROID {field_name} must be finite")


def _scene_episode(
    scene: SceneRecord,
    dataset_root: Path,
    media_root: Path,
    instruction: str,
    action_count: int,
) -> tuple[str, int, Path, Path]:
    trajectory_id = _string(scene.context.get("ctrl_world_trajectory_id"), "trajectory_id")
    start_index = scene.context.get("ctrl_world_start_index")
    if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 0:
        raise AdapterSchemaError("ctrl_world_start_index must be a non-negative integer")
    if not scene.observation_paths:
        raise AdapterSchemaError("scene must provide a conditioning media path")
    conditioning = _contained_path(
        scene.observation_paths[0], media_root, "conditioning_media_path", directory=False
    )
    annotation = dataset_root / "annotation" / "val" / f"{trajectory_id}.json"
    annotation = _contained_path(annotation, dataset_root, "DROID annotation", directory=False)
    try:
        record = json.loads(annotation.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdapterSchemaError("DROID annotation is not valid JSON") from error
    if not isinstance(record, dict) or not {"texts", "states", "joints", "videos"} <= set(record):
        raise AdapterSchemaError("DROID annotation is missing rollout fields")
    length = _annotation_length(record)
    if start_index + action_count > length:
        raise AdapterSchemaError("DROID episode is too short for the requested action steps")
    _finite_rows(record["states"], length, 7, "states")
    _finite_rows(record["joints"], length, 8, "joints")
    texts = record["texts"]
    if (
        not isinstance(texts, list)
        or not texts
        or any(not isinstance(text, str) or not text.strip() for text in texts)
    ):
        raise AdapterSchemaError("DROID texts must contain non-empty instructions")
    if texts[0].strip() != instruction.strip():
        raise AdapterSchemaError("task instruction must match the selected DROID episode")
    videos = record["videos"]
    if not isinstance(videos, list) or not videos:
        raise AdapterSchemaError("DROID annotation videos must be a nonempty array")
    resolved_media: list[Path] = []
    for item in videos:
        if not isinstance(item, dict) or set(item) != {"video_path"}:
            raise AdapterSchemaError("DROID video entry must contain only video_path")
        relative = item["video_path"]
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise AdapterSchemaError("DROID video_path must be relative to dataset_root")
        resolved_media.append(
            _contained_path(
                dataset_root / relative,
                dataset_root,
                "DROID video_path",
                directory=False,
            )
        )
    if conditioning not in resolved_media:
        raise AdapterSchemaError("conditioning media must belong to the selected DROID episode")
    return trajectory_id, start_index, conditioning, annotation


def _verify_rollout_content(rollout: Rollout) -> None:
    manifest = rollout.metrics.get("output_manifest")
    if not isinstance(manifest, dict):
        raise AdapterSchemaError("rollout receipt is missing its output content manifest")
    verify_manifest(manifest, "Ctrl-World rollout artifacts")


class CtrlWorldAdapter:
    """WorldModel implementation that invokes one pinned replay-agent process."""

    def __init__(
        self,
        bridge: ProcessRequester,
        assets: CtrlWorldAssets,
        *,
        artifact_dir: Path,
        artifact_root: Path,
        asset_root: Path,
        media_root: Path,
        timeout_seconds: float = 300.0,
        fps: float = 4.0,
    ) -> None:
        self._bridge = bridge
        self._artifact_root = _existing_absolute(artifact_root, "artifact_root", directory=True)
        self._artifact_dir = _contained_path(
            artifact_dir, self._artifact_root, "artifact_dir", directory=True
        )
        self._asset_root = _existing_absolute(asset_root, "asset_root", directory=True)
        self._media_root = _existing_absolute(media_root, "media_root", directory=True)
        repo = _repo_root(assets.repo_root)
        config = _contained_path(assets.config_path, repo, "config_path", directory=False)
        if config != (repo / "config.py").resolve():
            raise AdapterSchemaError("config_path must be the pinned repo config.py")
        self._assets = CtrlWorldAssets(
            repo,
            config,
            _contained_path(assets.model_root, self._asset_root, "model_root", directory=True),
            _contained_path(
                assets.text_encoder_root,
                self._asset_root,
                "text_encoder_root",
                directory=True,
            ),
            _contained_path(assets.dataset_root, self._media_root, "dataset_root", directory=True),
            _contained_path(
                assets.checkpoint_path,
                self._asset_root,
                "checkpoint_path",
                directory=False,
            ),
            _contained_path(assets.stats_path, self._asset_root, "stats_path", directory=False),
        )
        if timeout_seconds <= 0:
            raise AdapterSchemaError("timeout_seconds must be positive")
        if not math.isfinite(fps) or fps <= 0:
            raise AdapterSchemaError("fps must be finite and positive")
        self._timeout_seconds = timeout_seconds
        self._fps = fps
        self._operations = OperationLedger(self._artifact_dir)

    def imagine(
        self,
        task: TaskSpec,
        scene: SceneRecord,
        expert: ExpertSpec,
        actions: ActionChunk,
        *,
        operation_id: str,
    ) -> Rollout:
        """Generate and validate a rollout for exactly one expert/action identity."""
        request_id = _require_operation_id(operation_id)
        normalized = validate_ctrl_actions(actions)
        if expert.expert_id != normalized.expert_id:
            raise AdapterSchemaError("expert and action identities must match")
        if expert.action_schema != CTRL_ACTION_SCHEMA:
            raise AdapterSchemaError("expert action schema must be cartesian_pose_7d")
        trajectory_id, start_index, conditioning, annotation = _scene_episode(
            scene,
            self._assets.dataset_root,
            self._media_root,
            task.instruction,
            len(normalized.actions),
        )
        action_payload: JsonObject = {
            "expert_id": normalized.expert_id,
            "schema_name": normalized.schema_name,
            "values": [list(row) for row in normalized.actions],
            "timestamps": list(normalized.timestamps),
        }
        asset_manifest = combined_manifest(
            [
                file_manifest(self._assets.config_path, self._assets.repo_root, "config_path"),
                directory_manifest(self._assets.model_root, self._asset_root, "model_root"),
                directory_manifest(
                    self._assets.text_encoder_root,
                    self._asset_root,
                    "text_encoder_root",
                ),
                file_manifest(self._assets.checkpoint_path, self._asset_root, "checkpoint_path"),
                file_manifest(self._assets.stats_path, self._asset_root, "stats_path"),
                file_manifest(annotation, self._assets.dataset_root, "DROID annotation"),
                file_manifest(conditioning, self._media_root, "conditioning media"),
            ]
        )
        payload: JsonObject = {
            "task_id": task.task_id,
            "instruction": task.instruction,
            "scene_id": scene.scene_id,
            "expert_id": expert.expert_id,
            "revision": CTRL_WORLD_PIN,
            "trajectory_id": trajectory_id,
            "start_index": start_index,
            "fps": self._fps,
            "actions": action_payload,
            "repo_root": str(self._assets.repo_root),
            "config_path": str(self._assets.config_path),
            "model_root": str(self._assets.model_root),
            "text_encoder_root": str(self._assets.text_encoder_root),
            "dataset_root": str(self._assets.dataset_root),
            "checkpoint_path": str(self._assets.checkpoint_path),
            "stats_path": str(self._assets.stats_path),
            "conditioning_media_path": str(conditioning),
            "asset_manifest": asset_manifest,
            "output_dir": str(self._artifact_dir),
        }
        request = BridgeRequest(request_id, "imagine", self._artifact_dir, payload)
        inputs: JsonObject = {"revision": CTRL_WORLD_PIN, "request": request_to_json(request)}

        def execute() -> Rollout:
            response = self._bridge.request(request, self._timeout_seconds)
            verify_manifest(asset_manifest, "Ctrl-World input assets")
            if response.request_id != request_id or not isinstance(response.result, dict):
                raise AdapterSchemaError("Ctrl-World response correlation or result is invalid")
            return map_ctrl_response(
                response.result,
                artifact_dir=self._artifact_dir,
                expected_request_id=request_id,
                expected_task_id=task.task_id,
                expected_scene_id=scene.scene_id,
                expected_expert_id=expert.expert_id,
                expected_actions=normalized,
                expected_trajectory_id=trajectory_id,
                expected_start_index=start_index,
            )

        return self._operations.perform(
            "ctrl_world.imagine",
            request_id,
            inputs,
            execute,
            rollout_to_json,
            rollout_from_json,
            _verify_rollout_content,
        )

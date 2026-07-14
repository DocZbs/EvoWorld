"""Backend-neutral contracts for the pinned GR00T-N1.6 policy service."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
import os
from pathlib import Path
import re
from typing import Protocol

from evoworld.adapters.content import combined_manifest, file_manifest, verify_manifest
from evoworld.adapters.operations import OperationLedger
from evoworld.adapters.process import (
    BridgeRequest,
    BridgeResponse,
    ProcessBridge,
    request_to_json,
)
from evoworld.codec import action_from_json, action_to_json
from evoworld.adapters.revisions import GitRunner, verify_backend_checkout
from evoworld.contracts import ActionChunk, JsonObject, JsonValue, SceneRecord, TaskSpec

GROOT_N16_PIN = "ead52833afbbf4243f8cd5e7664f48a94de03b19"
_EMBODIMENT_TAG = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AdapterSchemaError(ValueError):
    """A GR00T value does not satisfy the stable EvoWorld boundary."""


class ProcessRequester(Protocol):
    """The narrow process surface used by the policy adapter."""

    def request(self, request: BridgeRequest, timeout: float) -> BridgeResponse: ...


def _require_operation_id(operation_id: str) -> str:
    if not operation_id.strip():
        raise AdapterSchemaError("operation_id must not be blank")
    return operation_id


def _checkpoint(value: str, digest: str) -> tuple[str, str]:
    if not value.strip():
        raise AdapterSchemaError("checkpoint_identity must not be blank")
    if _SHA256.fullmatch(digest) is None:
        raise AdapterSchemaError("checkpoint_digest must be a lowercase SHA-256")
    return value, digest


def _require_directory(path: Path, field_name: str) -> Path:
    if not path.is_absolute() or not path.is_dir():
        raise AdapterSchemaError(f"{field_name} must be an existing absolute directory")
    return path.resolve()


def build_groot_server_command(
    repo_root: Path,
    checkpoint_root: Path,
    embodiment_tag: str,
    port: int,
    *,
    project_root: Path,
    revision_runner: GitRunner | None = None,
) -> list[str]:
    """Build safe argv for the pinned official server plus identity endpoint."""
    repo = _verified_repo(repo_root, project_root, revision_runner)
    checkpoint = _require_directory(checkpoint_root, "checkpoint_root")
    server_script = (project_root / "integrations" / "groot" / "server_wrapper.py").resolve()
    if not server_script.is_file():
        raise AdapterSchemaError("project_root is missing the GR00T server wrapper")
    if not _EMBODIMENT_TAG.fullmatch(embodiment_tag):
        raise AdapterSchemaError("embodiment_tag contains unsupported characters")
    if type(port) is not int or not 1 <= port <= 65535:
        raise AdapterSchemaError("port must be between 1 and 65535")
    return [
        "uv",
        "run",
        "--project",
        str(repo),
        "python",
        str(server_script),
        "--repo-root",
        str(repo),
        "--checkpoint-path",
        str(checkpoint),
        "--expected-revision",
        GROOT_N16_PIN,
        "--embodiment-tag",
        embodiment_tag,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--strict",
        "True",
    ]


def build_groot_process_bridge(
    repo_root: Path,
    artifact_root: Path,
    allowed_media_root: Path,
    *,
    backend_python: Path,
    project_root: Path,
    revision_runner: GitRunner | None = None,
) -> ProcessBridge:
    """Build the fixed JSONL subprocess for one isolated GR00T environment."""
    repo = _verified_repo(repo_root, project_root, revision_runner)
    artifacts = _require_directory(artifact_root, "artifact_root")
    media = _require_directory(allowed_media_root, "allowed_media_root")
    project = _require_directory(project_root, "project_root")
    if not backend_python.is_absolute() or not backend_python.is_file():
        raise AdapterSchemaError("backend_python must be an existing absolute file")
    bridge_script = (project / "integrations" / "groot" / "policy_bridge.py").resolve()
    if not bridge_script.is_file():
        raise AdapterSchemaError("project_root is missing the GR00T policy bridge")
    if not (repo / "gr00t" / "policy" / "server_client.py").is_file():
        raise AdapterSchemaError("repo_root is missing the official PolicyClient")
    python_path = os.pathsep.join((str(project / "src"), str(project), str(repo)))

    def preflight() -> None:
        _verified_repo(repo, project, revision_runner)

    return ProcessBridge(
        (str(backend_python), str(bridge_script), "--allowed-media-root", str(media)),
        cwd=repo,
        artifact_root=artifacts,
        env={"PYTHONIOENCODING": "utf-8", "PYTHONPATH": python_path},
        preflight=preflight,
    )


def _verified_repo(repo_root: Path, project_root: Path, runner: GitRunner | None) -> Path:
    if project_root.is_symlink() or repo_root.is_symlink():
        raise AdapterSchemaError("repo_root must not use symlinked checkout paths")
    project = _require_directory(project_root, "project_root")
    repo = _require_directory(repo_root, "repo_root")
    try:
        if runner is None:
            verify_backend_checkout(project, repo, "third_party/isaac_groot")
        else:
            verify_backend_checkout(
                project,
                repo,
                "third_party/isaac_groot",
                runner=runner,
            )
    except RuntimeError as error:
        raise AdapterSchemaError("repo_root is not the clean pinned GR00T gitlink") from error
    return repo


def _mapping(value: JsonValue | None, field_name: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict) or not value:
        raise AdapterSchemaError(f"{field_name} must be a nonempty object")
    return value


def _contained_file(value: JsonValue, root: Path, field_name: str) -> str:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise AdapterSchemaError(f"{field_name} must be an absolute media path")
    try:
        path = Path(value).resolve(strict=True)
    except FileNotFoundError as error:
        raise AdapterSchemaError(f"{field_name} media file is missing") from error
    try:
        path.relative_to(root)
    except ValueError as error:
        raise AdapterSchemaError(f"{field_name} must be contained by allowed_media_root") from error
    if not path.is_file():
        raise AdapterSchemaError(f"{field_name} must resolve to a media file")
    return str(path)


def _media_stream(value: JsonValue, root: Path, field_name: str) -> list[JsonValue]:
    if not isinstance(value, list) or not value:
        raise AdapterSchemaError(f"{field_name} must contain one or more media paths")
    return [_contained_file(item, root, field_name) for item in value]


def _state_stream(value: JsonValue, field_name: str) -> list[JsonValue]:
    if not isinstance(value, list) or not value:
        raise AdapterSchemaError(f"{field_name} must contain one or more state rows")
    result: list[JsonValue] = []
    width: int | None = None
    for row_index, raw_row in enumerate(value):
        if not isinstance(raw_row, list) or not raw_row:
            raise AdapterSchemaError(f"{field_name}[{row_index}] must be a nonempty state row")
        if width is None:
            width = len(raw_row)
        if len(raw_row) != width:
            raise AdapterSchemaError(f"{field_name} state rows must have equal width")
        row: list[JsonValue] = []
        for item in raw_row:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise AdapterSchemaError(f"{field_name} must contain numeric values")
            if not math.isfinite(float(item)):
                raise AdapterSchemaError(f"{field_name} must contain finite values")
            row.append(item)
        result.append(row)
    return result


def map_groot_observation(
    task: TaskSpec, scene: SceneRecord, allowed_media_root: Path
) -> JsonObject:
    """Map domain context to a JSON-only nested GR00T observation description."""
    root = _require_directory(allowed_media_root, "allowed_media_root")
    if not {"video", "state"} <= set(scene.context):
        raise AdapterSchemaError("GR00T scene context must contain video and state")
    videos = _mapping(scene.context.get("video"), "video")
    states = _mapping(scene.context.get("state"), "state")
    return {
        "video": {key: _media_stream(value, root, f"video.{key}") for key, value in videos.items()},
        "state": {key: _state_stream(value, f"state.{key}") for key, value in states.items()},
        "language": task.instruction,
    }


def _sequence(value: object, field_name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AdapterSchemaError(f"{field_name} must be a sequence")
    return value


def map_groot_response(expert_id: str, response: Mapping[str, object]) -> ActionChunk:
    """Validate a real `(B,T,D)` float32 result and select batch zero."""
    required = {"dtype", "actions", "delta_indices", "action_keys"}
    if set(response) != required:
        missing = required - set(response)
        name = sorted(missing)[0] if missing else "unsupported fields"
        raise AdapterSchemaError(f"GR00T response is missing or contains {name}")
    if response["dtype"] != "float32":
        raise AdapterSchemaError("GR00T actions must have float32 dtype")
    batches = _sequence(response["actions"], "actions")
    if not batches:
        raise AdapterSchemaError("GR00T actions must have nonempty B, T, and D dimensions")
    expected_horizon: int | None = None
    expected_width: int | None = None
    normalized: list[tuple[tuple[float, ...], ...]] = []
    for batch_index, raw_batch in enumerate(batches):
        batch = _sequence(raw_batch, "actions must have shape (B, T, D)")
        if not batch:
            raise AdapterSchemaError("GR00T actions must have nonempty B, T, and D dimensions")
        if expected_horizon is None:
            expected_horizon = len(batch)
        if len(batch) != expected_horizon:
            raise AdapterSchemaError("GR00T actions must have rectangular shape (B, T, D)")
        rows: list[tuple[float, ...]] = []
        for time_index, raw_row in enumerate(batch):
            row = _sequence(raw_row, "actions must have shape (B, T, D)")
            if not row:
                raise AdapterSchemaError("GR00T actions must have nonempty B, T, and D dimensions")
            if expected_width is None:
                expected_width = len(row)
            if len(row) != expected_width:
                raise AdapterSchemaError("GR00T actions must have rectangular shape (B, T, D)")
            values: list[float] = []
            for column_index, item in enumerate(row):
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    raise AdapterSchemaError(
                        f"actions[{batch_index}][{time_index}][{column_index}] must be numeric"
                    )
                number = float(item)
                if not math.isfinite(number):
                    raise AdapterSchemaError("GR00T actions must contain finite values")
                values.append(number)
            rows.append(tuple(values))
        normalized.append(tuple(rows))
    delta_indices = _sequence(response["delta_indices"], "delta_indices")
    if len(delta_indices) != expected_horizon:
        raise AdapterSchemaError("delta_indices must match action horizon")
    timestamps: list[float] = []
    for item in delta_indices:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise AdapterSchemaError("delta_indices must be numeric")
        timestamp = float(item)
        if not math.isfinite(timestamp):
            raise AdapterSchemaError("delta_indices must be finite")
        timestamps.append(timestamp)
    keys = _sequence(response["action_keys"], "action_keys")
    if not keys or any(not isinstance(key, str) or not key for key in keys):
        raise AdapterSchemaError("action_keys must contain nonempty strings")
    return ActionChunk(
        expert_id,
        normalized[0],
        tuple(timestamps),
        "groot:" + "+".join(str(key) for key in keys),
    )


class GrootPolicyAdapter:
    """Core-side adapter for an externally managed GR00T-N1.6 service."""

    _bridge: ProcessRequester
    _expert_id: str
    _artifact_dir: Path
    _media_root: Path
    _timeout: float
    _host: str
    _port: int
    _checkpoint_identity: str
    _checkpoint_digest: str
    _operations: OperationLedger

    def __init__(
        self,
        bridge: ProcessRequester,
        expert_id: str,
        *,
        artifact_dir: Path,
        allowed_media_root: Path,
        checkpoint_identity: str,
        checkpoint_digest: str,
        timeout_seconds: float = 20.0,
        host: str = "127.0.0.1",
        port: int = 5555,
    ) -> None:
        if not expert_id.strip():
            raise AdapterSchemaError("expert_id must not be blank")
        if not artifact_dir.is_absolute():
            raise AdapterSchemaError("artifact_dir must be absolute")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, Real)
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise AdapterSchemaError("timeout_seconds must be a positive finite real number")
        if not host.strip():
            raise AdapterSchemaError("host must not be blank")
        if type(port) is not int or not 1 <= port <= 65535:
            raise AdapterSchemaError("port must be between 1 and 65535")
        self._bridge = bridge
        self._expert_id = expert_id
        self._artifact_dir = artifact_dir
        self._operations = OperationLedger(artifact_dir)
        self._media_root = _require_directory(allowed_media_root, "allowed_media_root")
        self._checkpoint_identity, self._checkpoint_digest = _checkpoint(
            checkpoint_identity, checkpoint_digest
        )
        self._timeout = float(timeout_seconds)
        self._host = host
        self._port = port

    def act(self, task: TaskSpec, observation: SceneRecord, *, operation_id: str) -> ActionChunk:
        """Request and validate one action chunk."""
        request_id = _require_operation_id(operation_id)
        observation_payload = map_groot_observation(task, observation, self._media_root)
        videos = observation_payload["video"]
        if not isinstance(videos, dict):  # map_groot_observation guarantees this.
            raise AdapterSchemaError("GR00T video observation is invalid")
        media_manifest = combined_manifest(
            file_manifest(Path(str(path)), self._media_root, f"video.{key}")
            for key, paths in sorted(videos.items())
            if isinstance(paths, list)
            for path in paths
        )
        request = BridgeRequest(
            request_id,
            "act",
            self._artifact_dir,
            {
                "host": self._host,
                "port": self._port,
                "checkpoint_identity": self._checkpoint_identity,
                "checkpoint_digest": self._checkpoint_digest,
                "allowed_media_root": str(self._media_root),
                "media_manifest": media_manifest,
                "observation": observation_payload,
            },
        )
        inputs: JsonObject = {
            "revision": GROOT_N16_PIN,
            "request": request_to_json(request),
        }

        def execute() -> ActionChunk:
            response = self._bridge.request(request, self._timeout)
            verify_manifest(media_manifest, "GR00T input media")
            if response.request_id != request_id:
                raise AdapterSchemaError("GR00T response correlation does not match request")
            if not isinstance(response.result, dict):
                raise AdapterSchemaError("GR00T response result must be an object")
            return map_groot_response(self._expert_id, response.result)

        return self._operations.perform(
            "groot.act",
            request_id,
            inputs,
            execute,
            action_to_json,
            action_from_json,
        )

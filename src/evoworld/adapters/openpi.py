"""Backend-neutral contracts for invoking a pinned openpi policy service."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import math
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

_DROID_FIELDS = (
    "exterior_image_1_left",
    "wrist_image_left",
    "joint_position",
    "gripper_position",
)
_POLICY_CONFIG = re.compile(r"^[A-Za-z0-9_-]+$")
_CONTROL_FREQUENCY_HZ = 15.0
OPENPI_PIN = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AdapterSchemaError(ValueError):
    """A backend payload does not satisfy the stable EvoWorld adapter contract."""


class ProcessRequester(Protocol):
    """The narrow ProcessBridge surface required by this adapter."""

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


def build_openpi_server_command(
    repo_root: Path,
    policy_config: str,
    checkpoint_root: Path,
    port: int,
    *,
    project_root: Path,
    revision_runner: GitRunner | None = None,
) -> list[str]:
    """Build the verified OpenPi server wrapper argv with explicit immutable inputs."""
    repo = _verified_repo(repo_root, project_root, revision_runner)
    if checkpoint_root.is_symlink() or checkpoint_root.resolve() != checkpoint_root:
        raise AdapterSchemaError("checkpoint_root must be a canonical real directory")
    checkpoint = _require_directory(checkpoint_root, "checkpoint_root")
    if not (repo / "scripts" / "serve_policy.py").is_file():
        raise AdapterSchemaError("repo_root must contain scripts/serve_policy.py")
    wrapper = (project_root / "integrations" / "openpi" / "policy_server.py").resolve()
    if not wrapper.is_file():
        raise AdapterSchemaError("project_root is missing the OpenPi policy server wrapper")
    if not _POLICY_CONFIG.fullmatch(policy_config):
        raise AdapterSchemaError("policy_config contains unsupported characters")
    if type(port) is not int or not 1 <= port <= 65535:
        raise AdapterSchemaError("port must be between 1 and 65535")
    return [
        "uv",
        "run",
        "--project",
        str(repo),
        "python",
        str(wrapper),
        f"--repo-root={repo}",
        f"--expected-pin={OPENPI_PIN}",
        f"--policy-config={policy_config}",
        f"--checkpoint={checkpoint}",
        f"--port={port}",
    ]


def build_openpi_serve_command(
    repo_root: Path,
    policy_config: str,
    checkpoint_root: Path,
    port: int,
    *,
    project_root: Path,
    revision_runner: GitRunner | None = None,
) -> list[str]:
    """Compatibility alias for :func:`build_openpi_server_command`."""
    return build_openpi_server_command(
        repo_root,
        policy_config,
        checkpoint_root,
        port,
        project_root=project_root,
        revision_runner=revision_runner,
    )


def build_openpi_process_bridge(
    repo_root: Path,
    artifact_root: Path,
    allowed_media_root: Path,
    *,
    backend_python: Path,
    project_root: Path,
    revision_runner: GitRunner | None = None,
) -> ProcessBridge:
    """Build the fixed JSONL subprocess boundary for one openpi environment."""
    repo = _verified_repo(repo_root, project_root, revision_runner)
    artifacts = _require_directory(artifact_root, "artifact_root")
    media = _require_directory(allowed_media_root, "allowed_media_root")
    project = _require_directory(project_root, "project_root")
    if not backend_python.is_absolute() or not backend_python.is_file():
        raise AdapterSchemaError("backend_python must be an existing absolute file")
    bridge_script = (project / "integrations" / "openpi" / "policy_bridge.py").resolve()
    if not bridge_script.is_file():
        raise AdapterSchemaError("project_root is missing the openpi policy bridge")
    client_source = (repo / "packages" / "openpi-client" / "src").resolve()
    if not client_source.is_dir():
        raise AdapterSchemaError("repo_root is missing the official openpi client source")
    python_path = os.pathsep.join((str(project / "src"), str(project), str(client_source)))

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
            verify_backend_checkout(project, repo, "third_party/openpi")
        else:
            verify_backend_checkout(
                project,
                repo,
                "third_party/openpi",
                runner=runner,
            )
    except RuntimeError as error:
        raise AdapterSchemaError("repo_root is not the clean pinned openpi gitlink") from error
    return repo


def _contained_media_file(value: JsonValue, root: Path, field_name: str) -> str:
    if not isinstance(value, str):
        raise AdapterSchemaError(f"{field_name} must be an absolute media path")
    path = Path(value)
    if not path.is_absolute():
        raise AdapterSchemaError(f"{field_name} must be an absolute media path")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise AdapterSchemaError(f"{field_name} media file is missing") from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise AdapterSchemaError(f"{field_name} must be contained by allowed_media_root") from error
    if not resolved.is_file():
        raise AdapterSchemaError(f"{field_name} must resolve to a media file")
    return str(resolved)


def _state_vector(value: JsonValue, width: int, field_name: str) -> list[JsonValue]:
    if isinstance(value, (str, bytes)) or not isinstance(value, list) or len(value) != width:
        raise AdapterSchemaError(f"{field_name} must contain exactly {width} values")
    result: list[JsonValue] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise AdapterSchemaError(f"{field_name} must contain numeric values")
        if not math.isfinite(float(item)):
            raise AdapterSchemaError(f"{field_name} must contain finite values")
        result.append(item)
    return result


def map_droid_observation(
    task: TaskSpec, scene: SceneRecord, allowed_media_root: Path
) -> JsonObject:
    """Translate a grounded scene to the exact DROID policy field names."""
    missing = [field for field in _DROID_FIELDS if scene.context.get(field) is None]
    if missing:
        raise AdapterSchemaError(f"DROID observation is missing {missing[0]}")
    root = _require_directory(allowed_media_root, "allowed_media_root")
    return {
        "observation/exterior_image_1_left": _contained_media_file(
            scene.context["exterior_image_1_left"], root, "exterior_image_1_left"
        ),
        "observation/wrist_image_left": _contained_media_file(
            scene.context["wrist_image_left"], root, "wrist_image_left"
        ),
        "observation/joint_position": _state_vector(
            scene.context["joint_position"], 7, "joint_position"
        ),
        "observation/gripper_position": _state_vector(
            scene.context["gripper_position"], 1, "gripper_position"
        ),
        "prompt": task.instruction,
    }


def _shape_for_model(model: str) -> tuple[int, int]:
    normalized = model.lower().replace("_", "-")
    if normalized in {"pi05", "pi0.5"}:
        return (15, 8)
    if normalized == "pi0":
        return (10, 8)
    if normalized in {"pi0-fast", "pi0fast"}:
        return (10, 8)
    raise AdapterSchemaError(f"unsupported openpi model {model!r}")


def _numeric_row(value: object, expected_width: int, row_index: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AdapterSchemaError(f"actions[{row_index}] must be a numeric row")
    if len(value) != expected_width:
        raise AdapterSchemaError(f"actions[{row_index}] has wrong width")
    row: list[float] = []
    for column_index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise AdapterSchemaError(f"actions[{row_index}][{column_index}] must be numeric")
        numeric = float(item)
        if not math.isfinite(numeric):
            raise AdapterSchemaError(f"actions[{row_index}][{column_index}] must be finite")
        row.append(numeric)
    return tuple(row)


def map_openpi_response(
    model: str,
    response: Mapping[str, object],
    *,
    expert_id: str = "openpi",
) -> ActionChunk:
    """Validate a service response and expose immutable joint-velocity actions."""
    if set(response) != {"actions"}:
        raise AdapterSchemaError("openpi response contains unsupported fields")
    action_values = response["actions"]
    if isinstance(action_values, (str, bytes)) or not isinstance(action_values, Sequence):
        raise AdapterSchemaError("actions must be a rectangular numeric sequence")
    horizon, width = _shape_for_model(model)
    if len(action_values) != horizon:
        raise AdapterSchemaError(f"actions must have horizon {horizon}")
    actions = tuple(_numeric_row(row, width, index) for index, row in enumerate(action_values))
    timestamps = tuple(index / _CONTROL_FREQUENCY_HZ for index in range(horizon))
    return ActionChunk(expert_id, actions, timestamps, "joint_velocity")


class OpenPiPolicyAdapter:
    """Core-side policy adapter that delegates transport to an isolated process."""

    def __init__(
        self,
        bridge: ProcessRequester,
        expert_id: str,
        model: str,
        *,
        artifact_dir: Path,
        allowed_media_root: Path,
        checkpoint_identity: str,
        checkpoint_digest: str,
        timeout_seconds: float = 5.0,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        if not expert_id.strip():
            raise AdapterSchemaError("expert_id must not be blank")
        if not artifact_dir.is_absolute():
            raise AdapterSchemaError("artifact_dir must be absolute")
        if timeout_seconds <= 0:
            raise AdapterSchemaError("timeout_seconds must be positive")
        if not host.strip():
            raise AdapterSchemaError("host must not be blank")
        if not 1 <= port <= 65535:
            raise AdapterSchemaError("port must be between 1 and 65535")
        _shape_for_model(model)
        self._bridge = bridge
        self._expert_id = expert_id
        self._model = model
        self._artifact_dir = artifact_dir
        self._operations = OperationLedger(artifact_dir)
        self._timeout_seconds = timeout_seconds
        self._allowed_media_root = _require_directory(allowed_media_root, "allowed_media_root")
        self._checkpoint_identity, self._checkpoint_digest = _checkpoint(
            checkpoint_identity, checkpoint_digest
        )
        self._host = host
        self._port = port

    def act(self, task: TaskSpec, observation: SceneRecord, *, operation_id: str) -> ActionChunk:
        """Request one policy chunk under the caller's stable operation identity."""
        request_id = _require_operation_id(operation_id)
        observation_payload = map_droid_observation(task, observation, self._allowed_media_root)
        media_manifest = combined_manifest(
            file_manifest(
                Path(str(observation_payload[field])),
                self._allowed_media_root,
                field,
            )
            for field in (
                "observation/exterior_image_1_left",
                "observation/wrist_image_left",
            )
        )
        request = BridgeRequest(
            request_id,
            "act",
            self._artifact_dir,
            {
                "model": self._model,
                "host": self._host,
                "port": self._port,
                "checkpoint_identity": self._checkpoint_identity,
                "checkpoint_digest": self._checkpoint_digest,
                "allowed_media_root": str(self._allowed_media_root),
                "media_manifest": media_manifest,
                "observation": observation_payload,
            },
        )
        inputs: JsonObject = {
            "revision": OPENPI_PIN,
            "request": request_to_json(request),
        }

        def execute() -> ActionChunk:
            response = self._bridge.request(request, self._timeout_seconds)
            verify_manifest(media_manifest, "OpenPi input media")
            if response.request_id != request_id:
                raise AdapterSchemaError("openpi response correlation does not match request")
            if not isinstance(response.result, dict):
                raise AdapterSchemaError("openpi response result must be an object")
            return map_openpi_response(
                self._model,
                response.result,
                expert_id=self._expert_id,
            )

        return self._operations.perform(
            "openpi.act",
            request_id,
            inputs,
            execute,
            action_to_json,
            action_from_json,
        )

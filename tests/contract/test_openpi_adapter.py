"""Offline contracts for the pinned openpi policy service boundary."""

from __future__ import annotations

from collections.abc import Mapping
import importlib.util
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
from threading import Thread
import time
from typing import Any

import numpy as np
from PIL import Image as PillowImage
import pytest

from evoworld.adapters.openpi import (
    AdapterSchemaError,
    OpenPiPolicyAdapter,
    build_openpi_process_bridge,
    build_openpi_server_command,
    build_openpi_serve_command,
    map_droid_observation,
    map_openpi_response,
)
from evoworld.adapters.content import combined_manifest, file_manifest
from evoworld.adapters.process import BridgeRequest, BridgeResponse, ProcessBridge
from evoworld.contracts import SceneRecord, TaskSpec
from evoworld.ports import OperationConflictError, OperationInDoubtError

_BRIDGE_PATH = Path(__file__).parents[2] / "integrations" / "openpi" / "policy_bridge.py"
PROJECT_ROOT = Path(__file__).parents[2].resolve()
_BRIDGE_SPEC = importlib.util.spec_from_file_location("openpi_policy_bridge", _BRIDGE_PATH)
assert _BRIDGE_SPEC is not None and _BRIDGE_SPEC.loader is not None
_BRIDGE_MODULE = importlib.util.module_from_spec(_BRIDGE_SPEC)
sys.modules[_BRIDGE_SPEC.name] = _BRIDGE_MODULE
_BRIDGE_SPEC.loader.exec_module(_BRIDGE_MODULE)
PolicyHealthError = _BRIDGE_MODULE.PolicyHealthError
check_healthz = _BRIDGE_MODULE.check_healthz
handle_request = _BRIDGE_MODULE.handle_request
load_droid_observation = _BRIDGE_MODULE.load_droid_observation
CHECKPOINT_IDENTITY = "pi05-test-checkpoint"
CHECKPOINT_DIGEST = "a" * 64


class FakeProcessBridge:
    """Captures one process request without launching a child."""

    def __init__(self, response: Mapping[str, object], request_id: str | None = None) -> None:
        self.response = response
        self.request_id = request_id
        self.captured_request: BridgeRequest | None = None
        self.requests: list[BridgeRequest] = []

    def request(self, request: BridgeRequest, timeout: float) -> BridgeResponse:
        del timeout
        self.captured_request = request
        self.requests.append(request)
        response_id = request.request_id if self.request_id is None else self.request_id
        return BridgeResponse(request_id=response_id, status="ok", result=dict(self.response))


class FakeClient:
    def __init__(
        self,
        response: Mapping[str, object],
        metadata: Mapping[str, object] | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.response = response
        self.metadata = metadata or {
            "checkpoint_identity": CHECKPOINT_IDENTITY,
            "checkpoint_digest": CHECKPOINT_DIGEST,
        }
        self.events = events
        self.observation: Mapping[str, object] | None = None

    def get_server_metadata(self) -> Mapping[str, object]:
        if self.events is not None:
            self.events.append("metadata")
        return self.metadata

    def infer(self, observation: Mapping[str, object]) -> Mapping[str, object]:
        if self.events is not None:
            self.events.append("infer")
        self.observation = observation
        return self.response


class HealthHandler(BaseHTTPRequestHandler):
    paths: list[str] = []

    def do_GET(self) -> None:  # noqa: N802 - HTTPServer requires this spelling.
        type(self).paths.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK\n")

    def log_message(self, format: str, *args: object) -> None:
        del format, args


class SlowHealthHandler(HealthHandler):
    def do_GET(self) -> None:  # noqa: N802 - HTTPServer requires this spelling.
        time.sleep(0.2)
        try:
            super().do_GET()
        except BrokenPipeError:
            pass


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "openpi"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "serve_policy.py").touch()
    return repo


def make_fake_backend_repo(tmp_path: Path) -> Path:
    repo = make_repo(tmp_path)
    source = repo / "packages" / "openpi-client" / "src"
    client_package = source / "openpi_client"
    pil_package = source / "PIL"
    client_package.mkdir(parents=True)
    pil_package.mkdir()
    (client_package / "__init__.py").write_text("", encoding="utf-8")
    (client_package / "image_tools.py").write_text(
        "import numpy as np\n"
        "def resize_with_pad(image, height, width):\n"
        "    assert image.shape == (2, 2, 3)\n"
        "    return np.zeros((height, width, 3), dtype=np.uint8)\n",
        encoding="utf-8",
    )
    (client_package / "websocket_client_policy.py").write_text(
        "class WebsocketClientPolicy:\n"
        "    def __init__(self, host, port):\n"
        "        self.host, self.port = host, port\n"
        "    def get_server_metadata(self):\n"
        f"        return {{'checkpoint_identity': {CHECKPOINT_IDENTITY!r}, "
        f"'checkpoint_digest': {CHECKPOINT_DIGEST!r}}}\n"
        "    def infer(self, observation):\n"
        "        assert observation['observation/exterior_image_1_left'].shape == (224, 224, 3)\n"
        "        assert observation['observation/wrist_image_left'].shape == (224, 224, 3)\n"
        "        assert len(observation['observation/joint_position']) == 7\n"
        "        assert len(observation['observation/gripper_position']) == 1\n"
        "        return {'actions': [[0.0] * 8 for _ in range(10)], "
        "'policy_timing': {'infer_ms': 0.5}, "
        "'server_timing': {'infer_ms': 1.0}}\n",
        encoding="utf-8",
    )
    (pil_package / "__init__.py").write_text("", encoding="utf-8")
    (pil_package / "Image.py").write_text(
        "from pathlib import Path\n"
        "import numpy as np\n"
        "class _Opened:\n"
        "    def __enter__(self):\n"
        "        return self\n"
        "    def __exit__(self, exc_type, exc, traceback):\n"
        "        return False\n"
        "    def convert(self, mode):\n"
        "        assert mode == 'RGB'\n"
        "        return self\n"
        "    def __array__(self, dtype=None, copy=None):\n"
        "        return np.zeros((2, 2, 3), dtype=np.uint8)\n"
        "def open(path):\n"
        "    Path(path).read_bytes()\n"
        "    return _Opened()\n",
        encoding="utf-8",
    )
    return repo


def fake_process_bridge(repo: Path, artifacts: Path, media: Path) -> ProcessBridge:
    client_source = repo / "packages" / "openpi-client" / "src"
    return ProcessBridge(
        (
            sys.executable,
            str(PROJECT_ROOT / "integrations" / "openpi" / "policy_bridge.py"),
            "--allowed-media-root",
            str(media.resolve()),
        ),
        cwd=repo.resolve(),
        artifact_root=artifacts.resolve(),
        env={
            "PYTHONIOENCODING": "utf-8",
            "PYTHONPATH": os.pathsep.join(
                (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT), str(client_source))
            ),
        },
    )


def make_media_scene(media_root: Path) -> SceneRecord:
    media_root.mkdir(parents=True, exist_ok=True)
    exterior = media_root / "exterior.png"
    wrist = media_root / "wrist.png"
    exterior.write_bytes(b"fixture-image")
    wrist.write_bytes(b"fixture-image")
    return SceneRecord(
        "scene-1",
        context={
            "exterior_image_1_left": str(exterior),
            "wrist_image_left": str(wrist),
            "joint_position": [0.0] * 7,
            "gripper_position": [0.5],
        },
    )


def make_png_scene(media_root: Path) -> SceneRecord:
    media_root.mkdir(parents=True, exist_ok=True)
    exterior = media_root / "exterior.png"
    wrist = media_root / "wrist.png"
    PillowImage.new("RGB", (4, 2), color=(255, 0, 0)).save(exterior)
    PillowImage.new("RGB", (3, 5), color=(0, 255, 0)).save(wrist)
    return SceneRecord(
        "scene-1",
        context={
            "exterior_image_1_left": str(exterior),
            "wrist_image_left": str(wrist),
            "joint_position": [0.0] * 7,
            "gripper_position": [0.5],
        },
    )


def media_manifest(observation: Mapping[str, object], media_root: Path) -> dict[str, object]:
    return combined_manifest(
        file_manifest(Path(str(observation[field])), media_root, field)
        for field in (
            "observation/exterior_image_1_left",
            "observation/wrist_image_left",
        )
    )


@pytest.mark.backend
def test_openpi_serve_command_is_argument_safe(tmp_path: Path) -> None:
    repo = PROJECT_ROOT / "third_party" / "openpi"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    command = build_openpi_server_command(
        repo, "pi05_droid", checkpoint, 8000, project_root=PROJECT_ROOT
    )

    assert command == [
        "uv",
        "run",
        "--project",
        str(repo),
        "python",
        str(PROJECT_ROOT / "integrations" / "openpi" / "policy_server.py"),
        f"--repo-root={repo}",
        "--expected-pin=15a9616a00943ada6c20a0f158e3adb39df2ccac",
        "--policy-config=pi05_droid",
        f"--checkpoint={checkpoint.resolve()}",
        "--port=8000",
    ]
    assert (
        build_openpi_serve_command(repo, "pi05_droid", checkpoint, 8000, project_root=PROJECT_ROOT)
        == command
    )


@pytest.mark.backend
@pytest.mark.parametrize("port", [False, True, 8000.0, 0, 65536])
def test_openpi_serve_command_rejects_invalid_port(tmp_path: Path, port: int) -> None:
    repo = PROJECT_ROOT / "third_party" / "openpi"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    with pytest.raises(AdapterSchemaError, match="port"):
        build_openpi_serve_command(repo, "pi05_droid", checkpoint, port, project_root=PROJECT_ROOT)


@pytest.mark.backend
def test_openpi_server_command_rejects_symlinked_checkpoint(tmp_path: Path) -> None:
    repo = PROJECT_ROOT / "third_party" / "openpi"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    alias = tmp_path / "checkpoint-alias"
    alias.symlink_to(checkpoint, target_is_directory=True)

    with pytest.raises(AdapterSchemaError, match="canonical"):
        build_openpi_server_command(repo, "pi05_droid", alias, 8000, project_root=PROJECT_ROOT)


@pytest.mark.backend
def test_openpi_process_bridge_has_absolute_fixed_boundary(tmp_path: Path) -> None:
    repo = PROJECT_ROOT / "third_party" / "openpi"
    artifacts = tmp_path / "artifacts"
    media = tmp_path / "media"
    artifacts.mkdir()
    media.mkdir()
    bridge = build_openpi_process_bridge(
        repo,
        artifacts,
        media,
        backend_python=Path(sys.executable),
        project_root=PROJECT_ROOT,
    )

    assert Path(bridge.argv[0]).is_absolute()
    assert Path(bridge.argv[1]).is_absolute()
    assert bridge.argv[2:] == ("--allowed-media-root", str(media.resolve()))
    assert bridge.cwd == repo.resolve()
    assert bridge.artifact_root == artifacts.resolve()
    assert set(bridge.env) == {"PYTHONIOENCODING", "PYTHONPATH"}
    assert str(repo / "packages" / "openpi-client" / "src") in bridge.env["PYTHONPATH"]


def test_droid_mapping_uses_controlled_absolute_media_and_official_keys(tmp_path: Path) -> None:
    task = TaskSpec("task-1", "put the cup in the drawer")
    media_root = tmp_path / "media"
    scene = make_media_scene(media_root)

    mapped = map_droid_observation(task, scene, media_root)

    assert mapped == {
        "observation/exterior_image_1_left": str((media_root / "exterior.png").resolve()),
        "observation/wrist_image_left": str((media_root / "wrist.png").resolve()),
        "observation/joint_position": [0.0] * 7,
        "observation/gripper_position": [0.5],
        "prompt": "put the cup in the drawer",
    }


def test_droid_mapping_rejects_missing_required_input(tmp_path: Path) -> None:
    task = TaskSpec("task-1", "put the cup in the drawer")
    scene = SceneRecord(
        "scene-1",
        context={
            "exterior_image_1_left": "exterior.png",
            "joint_position": [0.0] * 7,
            "gripper_position": [0.0],
        },
    )

    media_root = tmp_path / "media"
    media_root.mkdir()
    with pytest.raises(AdapterSchemaError, match="wrist_image_left"):
        map_droid_observation(task, scene, media_root)


def test_droid_mapping_rejects_media_symlink_escape_and_nonfinite_state(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    (media_root / "escape.png").symlink_to(outside)
    wrist = media_root / "wrist.png"
    wrist.write_bytes(b"image")
    scene = SceneRecord(
        "scene-1",
        context={
            "exterior_image_1_left": str(media_root / "escape.png"),
            "wrist_image_left": str(wrist),
            "joint_position": [0.0] * 7,
            "gripper_position": [0.0],
        },
    )

    with pytest.raises(AdapterSchemaError, match="contained"):
        map_droid_observation(TaskSpec("task-1", "move"), scene, media_root)
    scene.context["exterior_image_1_left"] = str(wrist)
    scene.context["joint_position"] = [0.0] * 6 + [float("inf")]
    with pytest.raises(AdapterSchemaError, match="finite"):
        map_droid_observation(TaskSpec("task-1", "move"), scene, media_root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("joint_position", [0.0] * 6, "7"),
        ("joint_position", [0.0] * 6 + [float("nan")], "finite"),
        ("gripper_position", [0.0, 1.0], "1"),
        ("gripper_position", [float("inf")], "finite"),
    ],
)
def test_droid_mapping_validates_joint_and_gripper_state(
    tmp_path: Path, field: str, value: list[float], message: str
) -> None:
    media_root = tmp_path / "media"
    scene = make_media_scene(media_root)
    scene.context[field] = value

    with pytest.raises(AdapterSchemaError, match=message):
        map_droid_observation(TaskSpec("task-1", "move"), scene, media_root)


@pytest.mark.parametrize(
    ("model", "shape"),
    [("pi05", (15, 8)), ("pi0", (10, 8)), ("pi0-fast", (10, 8))],
)
def test_openpi_actions_map_to_domain_chunk(model: str, shape: tuple[int, int]) -> None:
    rows, columns = shape
    chunk = map_openpi_response(
        model,
        {"actions": [[0.0] * columns for _ in range(rows)]},
        expert_id="expert-1",
    )

    assert (len(chunk.actions), len(chunk.actions[0])) == shape
    assert chunk.expert_id == "expert-1"
    assert chunk.timestamps == tuple(index / 15.0 for index in range(rows))


@pytest.mark.parametrize(
    "response",
    [
        {"actions": [[0.0] * 8] * 14},
        {"actions": [[0.0] * 8, [0.0] * 7] + [[0.0] * 8] * 13},
        {"actions": [[float("nan")] * 8 for _ in range(15)]},
        {"actions": [[0.0] * 8 for _ in range(15)], "log": "not protocol"},
    ],
)
def test_openpi_actions_reject_malformed_or_contaminated_response(
    response: Mapping[str, object],
) -> None:
    with pytest.raises(AdapterSchemaError):
        map_openpi_response("pi05", response, expert_id="expert-1")


@pytest.mark.parametrize("timing_field", ["policy_timing", "server_timing"])
def test_openpi_core_response_rejects_timing_fields(timing_field: str) -> None:
    with pytest.raises(AdapterSchemaError, match="unsupported fields"):
        map_openpi_response(
            "pi05",
            {
                "actions": [[0.0] * 8 for _ in range(15)],
                timing_field: {"infer_ms": 1.0},
            },
        )


def test_adapter_preserves_request_correlation_and_uses_contained_artifacts(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_dir = artifact_root / "run"
    artifact_dir.mkdir(parents=True)
    media_root = tmp_path / "media"
    scene = make_media_scene(media_root)
    process = FakeProcessBridge({"actions": [[0.0] * 8 for _ in range(15)]})
    adapter = OpenPiPolicyAdapter(
        process,
        "expert-1",
        "pi05",
        artifact_dir=artifact_dir,
        allowed_media_root=media_root,
        checkpoint_identity=CHECKPOINT_IDENTITY,
        checkpoint_digest=CHECKPOINT_DIGEST,
        timeout_seconds=0.5,
    )
    task = TaskSpec("task-1", "open drawer")

    result = adapter.act(task, scene, operation_id="round-1:act:expert-1")

    assert result.expert_id == "expert-1"
    assert process.captured_request is not None
    assert process.captured_request.request_id == "round-1:act:expert-1"
    assert process.captured_request.artifact_dir == artifact_dir
    assert process.captured_request.payload["model"] == "pi05"
    assert process.captured_request.payload["host"] == "127.0.0.1"
    assert process.captured_request.payload["port"] == 8000
    assert process.captured_request.payload["allowed_media_root"] == str(media_root.resolve())


def test_adapter_rejects_mismatched_process_correlation(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "run"
    artifact_dir.mkdir(parents=True)
    media_root = tmp_path / "media"
    scene = make_media_scene(media_root)
    adapter = OpenPiPolicyAdapter(
        FakeProcessBridge({"actions": [[0.0] * 8 for _ in range(15)]}, request_id="other"),
        "expert-1",
        "pi05",
        artifact_dir=artifact_dir,
        allowed_media_root=media_root,
        checkpoint_identity=CHECKPOINT_IDENTITY,
        checkpoint_digest=CHECKPOINT_DIGEST,
    )

    with pytest.raises(AdapterSchemaError, match="correlation"):
        adapter.act(
            TaskSpec("task-1", "open drawer"),
            scene,
            operation_id="round-1:act:expert-1",
        )


def test_adapter_retry_reuses_operation_id_and_distinct_operations_do_not_collide(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts" / "run"
    artifact_dir.mkdir(parents=True)
    media_root = tmp_path / "media"
    scene = make_media_scene(media_root)
    process = FakeProcessBridge({"actions": [[0.0] * 8 for _ in range(15)]})
    adapter = OpenPiPolicyAdapter(
        process,
        "expert-1",
        "pi05",
        artifact_dir=artifact_dir,
        allowed_media_root=media_root,
        checkpoint_identity=CHECKPOINT_IDENTITY,
        checkpoint_digest=CHECKPOINT_DIGEST,
    )

    adapter.act(TaskSpec("task-1", "open drawer"), scene, operation_id="round-1:act")
    adapter.act(TaskSpec("task-1", "open drawer"), scene, operation_id="round-1:act")
    adapter.act(TaskSpec("task-1", "open drawer"), scene, operation_id="round-2:act")

    assert [request.request_id for request in process.requests] == [
        "round-1:act",
        "round-2:act",
    ]
    with pytest.raises(OperationConflictError):
        adapter.act(
            TaskSpec("task-1", "different instruction"),
            scene,
            operation_id="round-1:act",
        )


def test_adapter_binds_input_bytes_and_sends_expected_checkpoint_identity(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    media_root = tmp_path / "media"
    scene = make_media_scene(media_root)
    process = FakeProcessBridge({"actions": [[0.0] * 8 for _ in range(15)]})
    adapter = OpenPiPolicyAdapter(
        process,
        "expert-1",
        "pi05",
        artifact_dir=artifact_dir,
        allowed_media_root=media_root,
        checkpoint_identity=CHECKPOINT_IDENTITY,
        checkpoint_digest=CHECKPOINT_DIGEST,
    )
    task = TaskSpec("task-1", "open drawer")

    adapter.act(task, scene, operation_id="round-1:act")
    assert process.captured_request is not None
    assert process.captured_request.payload["checkpoint_identity"] == CHECKPOINT_IDENTITY
    assert process.captured_request.payload["checkpoint_digest"] == CHECKPOINT_DIGEST
    (media_root / "exterior.png").write_bytes(b"changed-image-bytes")

    with pytest.raises(OperationConflictError, match="different canonical input"):
        adapter.act(task, scene, operation_id="round-1:act")
    assert len(process.requests) == 1


def test_adapter_detects_input_drift_after_backend_and_leaves_no_receipt(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    media_root = tmp_path / "media"
    scene = make_media_scene(media_root)
    exterior = media_root / "exterior.png"

    class MutatingBridge(FakeProcessBridge):
        def request(self, request: BridgeRequest, timeout: float) -> BridgeResponse:
            response = super().request(request, timeout)
            exterior.write_bytes(b"changed-during-backend")
            return response

    process = MutatingBridge({"actions": [[0.0] * 8 for _ in range(15)]})
    adapter = OpenPiPolicyAdapter(
        process,
        "expert-1",
        "pi05",
        artifact_dir=artifact_dir,
        allowed_media_root=media_root,
        checkpoint_identity=CHECKPOINT_IDENTITY,
        checkpoint_digest=CHECKPOINT_DIGEST,
    )
    task = TaskSpec("task-1", "open drawer")

    with pytest.raises(OperationConflictError, match="content changed"):
        adapter.act(task, scene, operation_id="round-1:act")
    assert list(artifact_dir.rglob("receipt.json")) == []
    exterior.write_bytes(b"fixture-image")
    with pytest.raises(OperationInDoubtError):
        adapter.act(task, scene, operation_id="round-1:act")
    assert len(process.requests) == 1


def test_adapter_rejects_blank_operation_id(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "run"
    artifact_dir.mkdir(parents=True)
    media_root = tmp_path / "media"
    scene = make_media_scene(media_root)
    adapter = OpenPiPolicyAdapter(
        FakeProcessBridge({"actions": [[0.0] * 8 for _ in range(15)]}),
        "expert-1",
        "pi05",
        artifact_dir=artifact_dir,
        allowed_media_root=media_root,
        checkpoint_identity=CHECKPOINT_IDENTITY,
        checkpoint_digest=CHECKPOINT_DIGEST,
    )

    with pytest.raises(AdapterSchemaError, match="operation_id"):
        adapter.act(TaskSpec("task-1", "open drawer"), scene, operation_id="  ")


@pytest.mark.backend
def test_real_pillow_decode_closes_images_and_calls_pinned_resize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_root = tmp_path / "media"
    scene = make_png_scene(media_root)
    observation = map_droid_observation(TaskSpec("task-1", "move cup"), scene, media_root)
    client_source = (
        Path(__file__).parents[2] / "third_party" / "openpi" / "packages" / "openpi-client" / "src"
    )
    monkeypatch.syspath_prepend(str(client_source))
    image_module = importlib.import_module("PIL.Image")
    image_tools = importlib.import_module("openpi_client.image_tools")
    original_open = image_module.open
    original_resize = image_tools.resize_with_pad
    opened: list[Any] = []
    resize_calls: list[tuple[int, int]] = []

    def tracked_open(path: str) -> Any:
        image = original_open(path)
        opened.append(image)
        return image

    def tracked_resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
        resize_calls.append((height, width))
        return original_resize(image, height, width)

    monkeypatch.setattr(image_module, "open", tracked_open)
    monkeypatch.setattr(image_tools, "resize_with_pad", tracked_resize)

    loaded = load_droid_observation(observation, media_root.resolve())

    for field in (
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
    ):
        array = loaded[field]
        assert isinstance(array, np.ndarray)
        assert array.shape == (224, 224, 3)
        assert array.dtype == np.uint8
    assert resize_calls == [(224, 224), (224, 224)]
    assert len(opened) == 2
    assert all(image.fp is None for image in opened)


def test_service_bridge_checks_health_before_client_and_accepts_timing(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    scene = make_media_scene(media_root)
    observation = map_droid_observation(TaskSpec("task-1", "move cup"), scene, media_root)
    events: list[str] = []
    client = FakeClient(
        {
            "actions": [[0.0] * 8 for _ in range(10)],
            "policy_timing": {"infer_ms": 3.0},
            "server_timing": {"infer_ms": 4.0, "prev_total_ms": 5.0},
        },
        events=events,
    )
    request = BridgeRequest(
        "request-1",
        "act",
        Path("/tmp/artifacts"),
        {
            "model": "pi0-fast",
            "host": "127.0.0.1",
            "port": 8000,
            "checkpoint_identity": CHECKPOINT_IDENTITY,
            "checkpoint_digest": CHECKPOINT_DIGEST,
            "allowed_media_root": str(media_root.resolve()),
            "media_manifest": media_manifest(observation, media_root),
            "observation": observation,
        },
    )

    def factory(host: str, port: int) -> FakeClient:
        del host, port
        events.append("client")
        return client

    def health(host: str, port: int, timeout: float) -> None:
        del host, port, timeout
        events.append("health")

    result = handle_request(
        request,
        allowed_media_root=media_root,
        client_factory=factory,
        health_check=health,
        image_loader=lambda value, root: value,
    )

    assert result == {"actions": [[0.0] * 8 for _ in range(10)]}
    assert client.observation == request.payload["observation"]
    assert events == ["health", "client", "metadata", "infer"]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            {
                "actions": [[0.0] * 8 for _ in range(10)],
                "policy_timing": [],
                "server_timing": {"infer_ms": 1.0},
            },
            "policy_timing",
        ),
        (
            {
                "actions": [[0.0] * 8 for _ in range(10)],
                "policy_timing": {"infer_ms": float("inf")},
                "server_timing": {"infer_ms": 1.0},
            },
            "policy_timing",
        ),
        (
            {
                "actions": [[0.0] * 8 for _ in range(10)],
                "policy_timing": {1: 0.5},
                "server_timing": {"infer_ms": 1.0},
            },
            "policy_timing",
        ),
        (
            {
                "actions": [[0.0] * 8 for _ in range(10)],
                "policy_timing": {"infer_ms": 0.5},
                "server_timing": {"infer_ms": float("nan")},
            },
            "server_timing",
        ),
        (
            {
                "actions": [[0.0] * 8 for _ in range(10)],
                "policy_timing": {"infer_ms": 0.5},
                "server_timing": {"infer_ms": 1.0, "unknown_ms": 2.0},
            },
            "server_timing",
        ),
        (
            {
                "actions": [[0.0] * 8 for _ in range(10)],
                "policy_timing": {"infer_ms": 0.5},
                "server_timing": {"infer_ms": 1.0},
                "unknown": {},
            },
            "unsupported fields",
        ),
    ],
)
def test_service_bridge_rejects_invalid_or_unknown_timing_response(
    tmp_path: Path, response: Mapping[str, object], message: str
) -> None:
    media_root = tmp_path / "media"
    scene = make_media_scene(media_root)
    observation = map_droid_observation(TaskSpec("task-1", "move cup"), scene, media_root)
    request = BridgeRequest(
        "request-1",
        "act",
        Path("/tmp/artifacts"),
        {
            "model": "pi0-fast",
            "host": "127.0.0.1",
            "port": 8000,
            "checkpoint_identity": CHECKPOINT_IDENTITY,
            "checkpoint_digest": CHECKPOINT_DIGEST,
            "allowed_media_root": str(media_root.resolve()),
            "media_manifest": media_manifest(observation, media_root),
            "observation": observation,
        },
    )

    with pytest.raises(ValueError, match=message):
        handle_request(
            request,
            allowed_media_root=media_root,
            client_factory=lambda host, port: FakeClient(response),
            health_check=lambda host, port, timeout: None,
            image_loader=lambda value, root: value,
        )


def test_service_bridge_rejects_server_metadata_before_inference(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    scene = make_media_scene(media_root)
    observation = map_droid_observation(TaskSpec("task-1", "move cup"), scene, media_root)
    events: list[str] = []
    client = FakeClient(
        {"actions": [[0.0] * 8 for _ in range(10)]},
        metadata={
            "checkpoint_identity": "different-checkpoint",
            "checkpoint_digest": CHECKPOINT_DIGEST,
        },
        events=events,
    )
    request = BridgeRequest(
        "request-1",
        "act",
        Path("/tmp/artifacts"),
        {
            "model": "pi0-fast",
            "host": "127.0.0.1",
            "port": 8000,
            "checkpoint_identity": CHECKPOINT_IDENTITY,
            "checkpoint_digest": CHECKPOINT_DIGEST,
            "allowed_media_root": str(media_root.resolve()),
            "media_manifest": media_manifest(observation, media_root),
            "observation": observation,
        },
    )

    with pytest.raises(ValueError, match="checkpoint identity"):
        handle_request(
            request,
            allowed_media_root=media_root,
            client_factory=lambda host, port: client,
            health_check=lambda host, port, timeout: None,
            image_loader=lambda value, root: value,
        )

    assert events == ["metadata"]
    assert client.observation is None


def test_health_check_uses_bounded_healthz_endpoint() -> None:
    HealthHandler.paths = []
    server = HTTPServer(("127.0.0.1", 0), HealthHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        check_healthz("127.0.0.1", server.server_port, 0.2)
    finally:
        server.shutdown()
        thread.join(timeout=1)
        server.server_close()

    assert HealthHandler.paths == ["/healthz"]


def test_health_check_has_a_real_bounded_timeout() -> None:
    server = HTTPServer(("127.0.0.1", 0), SlowHealthHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        with pytest.raises(PolicyHealthError, match="healthz"):
            check_healthz("127.0.0.1", server.server_port, 0.03)
    finally:
        server.shutdown()
        thread.join(timeout=1)
        server.server_close()

    assert time.monotonic() - started < 1.0


def test_service_bridge_surfaces_health_failure_before_client_creation(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    scene = make_media_scene(media_root)
    request = BridgeRequest(
        "request-1",
        "act",
        Path("/tmp/artifacts"),
        {
            "model": "pi05",
            "host": "127.0.0.1",
            "port": 8000,
            "checkpoint_identity": CHECKPOINT_IDENTITY,
            "checkpoint_digest": CHECKPOINT_DIGEST,
            "allowed_media_root": str(media_root.resolve()),
            "media_manifest": media_manifest(
                map_droid_observation(TaskSpec("task-1", "x"), scene, media_root),
                media_root,
            ),
            "observation": map_droid_observation(TaskSpec("task-1", "x"), scene, media_root),
        },
    )
    created = False

    def factory(host: str, port: int) -> Any:
        nonlocal created
        del host, port
        created = True
        return FakeClient({"actions": []})

    def unavailable(host: str, port: int, timeout: float) -> None:
        del host, port, timeout
        raise PolicyHealthError("/healthz timed out")

    with pytest.raises(PolicyHealthError, match="healthz"):
        handle_request(
            request,
            allowed_media_root=media_root,
            client_factory=factory,
            health_check=unavailable,
            image_loader=lambda value, root: value,
        )

    assert not created


def test_backend_rejects_media_escape_before_health(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    scene = make_media_scene(media_root)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    observation = map_droid_observation(TaskSpec("task-1", "x"), scene, media_root)
    manifest = media_manifest(observation, media_root)
    observation["observation/exterior_image_1_left"] = str(outside)
    request = BridgeRequest(
        "request-1",
        "act",
        Path("/tmp/artifacts"),
        {
            "model": "pi05",
            "host": "127.0.0.1",
            "port": 8000,
            "checkpoint_identity": CHECKPOINT_IDENTITY,
            "checkpoint_digest": CHECKPOINT_DIGEST,
            "allowed_media_root": str(media_root.resolve()),
            "media_manifest": manifest,
            "observation": observation,
        },
    )
    health_called = False

    def health(host: str, port: int, timeout: float) -> None:
        nonlocal health_called
        del host, port, timeout
        health_called = True

    with pytest.raises(ValueError, match="contained"):
        handle_request(
            request,
            allowed_media_root=media_root,
            client_factory=lambda host, port: FakeClient({"actions": []}),
            health_check=health,
            image_loader=lambda value, root: value,
        )

    assert not health_called


def test_real_process_bridge_runs_jsonl_with_fake_official_client(tmp_path: Path) -> None:
    repo = make_fake_backend_repo(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifact_dir = artifacts / "run"
    media_root = tmp_path / "media"
    artifact_dir.mkdir(parents=True)
    scene = make_media_scene(media_root)
    bridge = fake_process_bridge(repo, artifacts, media_root)
    server = HTTPServer(("127.0.0.1", 0), HealthHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        adapter = OpenPiPolicyAdapter(
            bridge,
            "expert-1",
            "pi0-fast",
            artifact_dir=artifact_dir,
            allowed_media_root=media_root,
            checkpoint_identity=CHECKPOINT_IDENTITY,
            checkpoint_digest=CHECKPOINT_DIGEST,
            host="127.0.0.1",
            port=server.server_port,
            timeout_seconds=3.0,
        )
        chunk = adapter.act(
            TaskSpec("task-1", "move cup"), scene, operation_id="round-1:act:expert-1"
        )
    finally:
        server.shutdown()
        thread.join(timeout=1)
        server.server_close()

    assert (len(chunk.actions), len(chunk.actions[0])) == (10, 8)
    log_dir = next((artifact_dir / ".process").iterdir())
    assert (log_dir / "process.stdout.log").is_file()
    assert (log_dir / "process.stderr.log").read_text(encoding="utf-8") == ""

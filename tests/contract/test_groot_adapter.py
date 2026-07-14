from __future__ import annotations

from collections.abc import Callable, Sequence
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from evoworld.adapters.groot import (
    AdapterSchemaError,
    GrootPolicyAdapter,
    build_groot_process_bridge,
    build_groot_server_command,
    map_groot_observation,
    map_groot_response,
)
from evoworld.adapters.process import (
    BridgeRemoteError,
    BridgeRequest,
    BridgeResponse,
    BridgeTimeoutError,
    ProcessBridge,
)
from evoworld.contracts import SceneRecord, TaskSpec
from evoworld.ports import OperationConflictError


PIN = "ead52833afbbf4243f8cd5e7664f48a94de03b19"
CHECKPOINT_IDENTITY = "groot-test-checkpoint"
CHECKPOINT_DIGEST = "b" * 64
PROJECT_ROOT = Path(__file__).parents[2].resolve()
_SERVER_PATH = PROJECT_ROOT / "integrations" / "groot" / "server_wrapper.py"
_SERVER_SPEC = importlib.util.spec_from_file_location("groot_server_wrapper", _SERVER_PATH)
assert _SERVER_SPEC is not None and _SERVER_SPEC.loader is not None
_SERVER_MODULE = importlib.util.module_from_spec(_SERVER_SPEC)
sys.modules[_SERVER_SPEC.name] = _SERVER_MODULE
_SERVER_SPEC.loader.exec_module(_SERVER_MODULE)
create_server = _SERVER_MODULE.create_server


class _NeverRequester:
    def request(self, request: BridgeRequest, timeout: float) -> BridgeResponse:
        raise AssertionError("blank operation ID must fail before the process request")


class _Requester:
    def __init__(self, *, response_id: str | None = None) -> None:
        self.response_id = response_id
        self.requests: list[BridgeRequest] = []

    def request(self, request: BridgeRequest, timeout: float) -> BridgeResponse:
        del timeout
        self.requests.append(request)
        return BridgeResponse(
            request_id=request.request_id if self.response_id is None else self.response_id,
            status="ok",
            result={
                "dtype": "float32",
                "actions": [[[1.0, 1.0, 1.0, 0.0]]],
                "delta_indices": [0],
                "action_keys": ["arm", "gripper"],
            },
        )


def _make_server_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "isaac-groot"
    script = repo / "gr00t" / "eval" / "run_gr00t_server.py"
    script.parent.mkdir(parents=True)
    script.touch()
    return repo


def _make_fake_backend_repo(tmp_path: Path) -> Path:
    repo = _make_server_repo(tmp_path)
    policy = repo / "gr00t" / "policy"
    policy.mkdir(parents=True, exist_ok=True)
    (repo / "gr00t" / "__init__.py").write_text("", encoding="utf-8")
    (policy / "__init__.py").write_text("", encoding="utf-8")
    (policy / "server_client.py").write_text(
        "from dataclasses import dataclass\n"
        "import json\n"
        "from pathlib import Path\n"
        "import time\n"
        "import numpy as np\n"
        "TRACE = Path(__file__).parents[2] / 'calls.json'\n"
        "def record(value):\n"
        "    calls = json.loads(TRACE.read_text()) if TRACE.exists() else []\n"
        "    calls.append(value)\n"
        "    TRACE.write_text(json.dumps(calls))\n"
        "@dataclass\n"
        "class Config:\n"
        "    modality_keys: list[str]\n"
        "    delta_indices: list[int]\n"
        "class PolicyClient:\n"
        "    def __init__(self, host='localhost', port=5555, timeout_ms=15000, strict=False):\n"
        "        record(['init', host, port, timeout_ms, strict])\n"
        "    def ping(self):\n"
        "        record('ping')\n"
        "        return True\n"
        "    def call_endpoint(self, endpoint, data=None, requires_input=True):\n"
        "        record(['endpoint', endpoint, requires_input])\n"
        "        assert endpoint == 'evoworld_identity' and requires_input is False\n"
        "        return {'checkpoint_identity': 'groot-test-checkpoint', "
        "'checkpoint_digest': '" + "b" * 64 + "'}\n"
        "    def get_modality_config(self):\n"
        "        record('modality')\n"
        "        return {\n"
        "            'video': Config(['front'], [0]),\n"
        "            'state': Config(['arm'], [0]),\n"
        "            'action': Config(['arm', 'gripper'], [0, 1]),\n"
        "            'language': Config(['annotation.human.task_description'], [0]),\n"
        "        }\n"
        "    def get_action(self, observation, options=None):\n"
        "        record('action')\n"
        "        assert observation['video']['front'].shape == (1, 1, 3, 4, 3)\n"
        "        assert observation['video']['front'].dtype == np.uint8\n"
        "        assert observation['state']['arm'].shape == (1, 1, 3)\n"
        "        assert observation['state']['arm'].dtype == np.float32\n"
        "        language = observation['language']['annotation.human.task_description']\n"
        "        assert language[0][0] in {\n"
        "            'move safely', 'slow', 'explode', 'wrong dtype', 'wrong rank',\n"
        "            'empty action', 'nonfinite action'}\n"
        "        if language[0][0] == 'slow': time.sleep(0.2)\n"
        "        if language[0][0] == 'explode': raise RuntimeError('boom')\n"
        "        if language[0][0] == 'wrong dtype':\n"
        "            return ({'arm': np.ones((1, 2, 3), dtype=np.float64), "
        "'gripper': np.zeros((1, 2, 1), dtype=np.float32)}, {})\n"
        "        if language[0][0] == 'wrong rank':\n"
        "            return ({'arm': np.ones((2, 3), dtype=np.float32), "
        "'gripper': np.zeros((1, 2, 1), dtype=np.float32)}, {})\n"
        "        if language[0][0] == 'empty action':\n"
        "            return ({'arm': np.ones((1, 0, 3), dtype=np.float32), "
        "'gripper': np.zeros((1, 0, 1), dtype=np.float32)}, {})\n"
        "        if language[0][0] == 'nonfinite action':\n"
        "            return ({'arm': np.full((1, 2, 3), np.inf, dtype=np.float32), "
        "'gripper': np.zeros((1, 2, 1), dtype=np.float32)}, {})\n"
        "        return ({\n"
        "            'arm': np.ones((1, 2, 3), dtype=np.float32),\n"
        "            'gripper': np.zeros((1, 2, 1), dtype=np.float32),\n"
        "        }, {})\n",
        encoding="utf-8",
    )
    return repo


def _fake_process_bridge(repo: Path, artifacts: Path, media: Path) -> ProcessBridge:
    return ProcessBridge(
        (
            sys.executable,
            str(PROJECT_ROOT / "integrations" / "groot" / "policy_bridge.py"),
            "--allowed-media-root",
            str(media.resolve()),
        ),
        cwd=repo.resolve(),
        artifact_root=artifacts.resolve(),
        env={
            "PYTHONIOENCODING": "utf-8",
            "PYTHONPATH": os.pathsep.join(
                (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT), str(repo.resolve()))
            ),
        },
    )


def _make_scene(media_root: Path, instruction: str = "move safely") -> tuple[TaskSpec, SceneRecord]:
    media_root.mkdir(parents=True, exist_ok=True)
    image = media_root / "front.png"
    Image.new("RGB", (4, 3), (20, 40, 60)).save(image)
    return (
        TaskSpec("task-1", instruction),
        SceneRecord(
            "scene-1",
            context={"video": {"front": [str(image)]}, "state": {"arm": [[0.1, 0.2, 0.3]]}},
        ),
    )


@pytest.mark.backend
def test_groot_server_command_uses_pinned_n16_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = PROJECT_ROOT / "third_party" / "isaac_groot"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    unrelated = tmp_path / "unrelated-cwd"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    command = build_groot_server_command(
        repo, checkpoint, "NEW_EMBODIMENT", 5555, project_root=PROJECT_ROOT
    )

    assert command == [
        "uv",
        "run",
        "--project",
        str(repo.resolve()),
        "python",
        str((PROJECT_ROOT / "integrations" / "groot" / "server_wrapper.py").resolve()),
        "--repo-root",
        str(repo.resolve()),
        "--checkpoint-path",
        str(checkpoint.resolve()),
        "--expected-revision",
        PIN,
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--host",
        "127.0.0.1",
        "--port",
        "5555",
        "--strict",
        "True",
    ]


def test_groot_server_wrapper_registers_independent_loaded_checkpoint_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    checkpoint = tmp_path / "checkpoint"
    repo.mkdir()
    checkpoint.mkdir()
    weights = checkpoint / "weights.bin"
    weights.write_bytes(b"loaded-checkpoint")
    embodiment = object()
    calls: list[object] = []

    class FakePolicy:
        def __init__(
            self,
            *,
            embodiment_tag: object,
            model_path: str,
            device: str,
            strict: bool,
        ) -> None:
            calls.append(("policy", embodiment_tag, model_path, device, strict))

    class FakeServer:
        def __init__(self, *, policy: object, host: str, port: int) -> None:
            calls.append(("server", policy, host, port))
            self.endpoint: Callable[[], dict[str, object]] | None = None
            self.requires_input: bool | None = None

        def register_endpoint(
            self,
            name: str,
            handler: Callable[[], dict[str, object]],
            requires_input: bool = True,
        ) -> None:
            assert name == "evoworld_identity"
            self.endpoint = handler
            self.requires_input = requires_input

    class Tags:
        NEW_EMBODIMENT = embodiment

        @classmethod
        def __class_getitem__(cls, name: str) -> object:
            return getattr(cls, name)

    monkeypatch.setattr(
        _SERVER_MODULE.importlib,
        "import_module",
        lambda name: (
            type("Module", (), {"EmbodimentTag": Tags})
            if name == "gr00t.data.embodiment_tags"
            else pytest.fail(f"unexpected import: {name}")
        ),
    )

    def runner(command: Sequence[str]) -> str:
        rendered = tuple(command)
        return PIN + "\n" if rendered[-2:] == ("rev-parse", "HEAD") else ""

    server = create_server(
        repo_root=repo.resolve(),
        checkpoint_path=checkpoint.resolve(),
        expected_revision=PIN,
        embodiment_tag="NEW_EMBODIMENT",
        device="cpu",
        host="127.0.0.1",
        port=5555,
        strict=True,
        runner=runner,
        policy_factory=FakePolicy,
        server_factory=FakeServer,
    )

    assert isinstance(server, FakeServer)
    assert calls[0] == (
        "policy",
        embodiment,
        str(checkpoint.resolve()),
        "cpu",
        True,
    )
    assert calls[1][0] == "server"
    assert calls[1][2:] == ("127.0.0.1", 5555)
    assert server.requires_input is False
    assert server.endpoint is not None
    identity = server.endpoint()
    assert identity["checkpoint_identity"] == str(checkpoint.resolve())
    assert len(identity["checkpoint_digest"]) == 64
    weights.write_bytes(b"changed")
    with pytest.raises(OperationConflictError, match="content changed"):
        server.endpoint()


@pytest.mark.backend
@pytest.mark.parametrize("port", [False, True, 5555.0, 0, 65536])
def test_groot_server_command_requires_strict_valid_integer_port(tmp_path: Path, port: int) -> None:
    repo = PROJECT_ROOT / "third_party" / "isaac_groot"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    with pytest.raises(AdapterSchemaError, match="port"):
        build_groot_server_command(
            repo, checkpoint, "NEW_EMBODIMENT", port, project_root=PROJECT_ROOT
        )


@pytest.mark.backend
@pytest.mark.parametrize("embodiment", ["", "NEW EMBODIMENT", "x; touch /tmp/injected"])
def test_groot_server_command_rejects_command_injection(tmp_path: Path, embodiment: str) -> None:
    repo = PROJECT_ROOT / "third_party" / "isaac_groot"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    with pytest.raises(AdapterSchemaError, match="embodiment"):
        build_groot_server_command(repo, checkpoint, embodiment, 5555, project_root=PROJECT_ROOT)


@pytest.mark.backend
def test_groot_process_bridge_has_fixed_absolute_boundary(tmp_path: Path) -> None:
    repo = PROJECT_ROOT / "third_party" / "isaac_groot"
    artifacts = tmp_path / "artifacts"
    media = tmp_path / "media"
    artifacts.mkdir()
    media.mkdir()
    bridge = build_groot_process_bridge(
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
    assert str(repo.resolve()) in bridge.env["PYTHONPATH"]


@pytest.mark.parametrize("port", [False, True, 5555.0, 0, 65536])
def test_groot_adapter_requires_strict_valid_integer_port(tmp_path: Path, port: int) -> None:
    media = tmp_path / "media"
    artifacts = tmp_path / "artifacts"
    media.mkdir()
    artifacts.mkdir()
    bridge = ProcessBridge(("/bin/true",), cwd=tmp_path, artifact_root=artifacts)
    with pytest.raises(AdapterSchemaError, match="port"):
        GrootPolicyAdapter(
            bridge,
            "groot",
            artifact_dir=artifacts,
            allowed_media_root=media,
            checkpoint_identity=CHECKPOINT_IDENTITY,
            checkpoint_digest=CHECKPOINT_DIGEST,
            port=port,
        )


@pytest.mark.parametrize(
    "timeout", [False, True, 0.0, -1.0, float("nan"), float("inf"), float("-inf")]
)
def test_groot_adapter_requires_positive_finite_real_timeout(
    tmp_path: Path, timeout: float
) -> None:
    media = tmp_path / "media"
    artifacts = tmp_path / "artifacts"
    media.mkdir()
    artifacts.mkdir()
    bridge = ProcessBridge(("/bin/true",), cwd=tmp_path, artifact_root=artifacts)
    with pytest.raises(AdapterSchemaError, match="timeout"):
        GrootPolicyAdapter(
            bridge,
            "groot",
            artifact_dir=artifacts,
            allowed_media_root=media,
            checkpoint_identity=CHECKPOINT_IDENTITY,
            checkpoint_digest=CHECKPOINT_DIGEST,
            timeout_seconds=timeout,
        )


def test_groot_observation_maps_nested_modalities(tmp_path: Path) -> None:
    media = tmp_path / "media"
    task, scene = _make_scene(media)

    result = map_groot_observation(task, scene, media)

    assert result == {
        "video": {"front": [str((media / "front.png").resolve())]},
        "state": {"arm": [[0.1, 0.2, 0.3]]},
        "language": "move safely",
    }


def test_groot_observation_rejects_symlink_escape_and_nonfinite_state(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    escape = media / "escape.png"
    escape.symlink_to(outside)
    task = TaskSpec("task-1", "move")
    scene = SceneRecord(
        "scene-1",
        context={"video": {"front": [str(escape)]}, "state": {"arm": [[0.0]]}},
    )

    with pytest.raises(AdapterSchemaError, match="contained"):
        map_groot_observation(task, scene, media)

    valid = media / "front.png"
    valid.write_bytes(b"image")
    scene.context["video"] = {"front": [str(valid)]}
    scene.context["state"] = {"arm": [[float("nan")]]}
    with pytest.raises(AdapterSchemaError, match="finite"):
        map_groot_observation(task, scene, media)


def test_groot_observation_rejects_nested_missing_keys(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    scene = SceneRecord("scene-1", context={"video": {}})
    with pytest.raises(AdapterSchemaError, match="contain video and state"):
        map_groot_observation(TaskSpec("task-1", "move"), scene, media)


def test_groot_observation_accepts_union_context_without_forwarding_extras(
    tmp_path: Path,
) -> None:
    media = tmp_path / "media"
    media.mkdir()
    frame = media / "front.png"
    frame.write_bytes(b"image")
    scene = SceneRecord(
        "scene-1",
        context={
            "video": {"front": [str(frame)]},
            "state": {"arm": [[0.1, 0.2, 0.3]]},
            "ctrl_world_trajectory_id": "0001",
            "joint_position": [0.0] * 7,
        },
    )

    result = map_groot_observation(TaskSpec("task-1", "move"), scene, media)

    assert result == {
        "video": {"front": [str(frame.resolve())]},
        "state": {"arm": [[0.1, 0.2, 0.3]]},
        "language": "move",
    }


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            {
                "dtype": "float64",
                "actions": [[[0.0]]],
                "delta_indices": [0],
                "action_keys": ["arm"],
            },
            "float32",
        ),
        (
            {"dtype": "float32", "actions": [[0.0]], "delta_indices": [0], "action_keys": ["arm"]},
            "B, T, D",
        ),
        (
            {"dtype": "float32", "actions": [], "delta_indices": [], "action_keys": ["arm"]},
            "nonempty",
        ),
        (
            {
                "dtype": "float32",
                "actions": [[[float("inf")]]],
                "delta_indices": [0],
                "action_keys": ["arm"],
            },
            "finite",
        ),
        ({"dtype": "float32", "actions": [[[0.0]]], "delta_indices": [0]}, "action_keys"),
    ],
)
def test_groot_response_rejects_invalid_arrays(response: dict[str, object], message: str) -> None:
    with pytest.raises(AdapterSchemaError, match=message):
        map_groot_response("groot", response)


def test_groot_response_selects_batch_zero_and_maps_schema() -> None:
    chunk = map_groot_response(
        "expert-1",
        {
            "dtype": "float32",
            "actions": [
                [[1.0, 2.0], [3.0, 4.0]],
                [[9.0, 9.0], [9.0, 9.0]],
            ],
            "delta_indices": [0, 1],
            "action_keys": ["arm", "gripper"],
        },
    )

    assert chunk.actions == ((1.0, 2.0), (3.0, 4.0))
    assert chunk.timestamps == (0.0, 1.0)
    assert chunk.schema_name == "groot:arm+gripper"


def test_groot_adapter_runs_real_jsonl_process_and_official_order(tmp_path: Path) -> None:
    repo = _make_fake_backend_repo(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifact_dir = artifacts / "request"
    media = tmp_path / "media"
    artifacts.mkdir()
    artifact_dir.mkdir()
    task, scene = _make_scene(media)
    bridge = _fake_process_bridge(repo, artifacts, media)
    adapter = GrootPolicyAdapter(
        bridge,
        "groot-n16",
        artifact_dir=artifact_dir,
        allowed_media_root=media,
        checkpoint_identity=CHECKPOINT_IDENTITY,
        checkpoint_digest=CHECKPOINT_DIGEST,
    )

    chunk = adapter.act(task, scene, operation_id="round-1:act:groot-n16")

    assert chunk.actions == ((1.0, 1.0, 1.0, 0.0), (1.0, 1.0, 1.0, 0.0))
    assert json.loads((repo / "calls.json").read_text()) == [
        ["init", "127.0.0.1", 5555, 15000, False],
        "ping",
        ["endpoint", "evoworld_identity", False],
        "modality",
        "action",
    ]


def test_groot_bridge_rejects_server_identity_before_inference(tmp_path: Path) -> None:
    repo = _make_fake_backend_repo(tmp_path)
    client = repo / "gr00t" / "policy" / "server_client.py"
    client.write_text(
        client.read_text(encoding="utf-8").replace(
            "'checkpoint_identity': 'groot-test-checkpoint'",
            "'checkpoint_identity': 'unexpected-checkpoint'",
        ),
        encoding="utf-8",
    )
    artifacts = tmp_path / "artifacts"
    artifact_dir = artifacts / "request"
    media = tmp_path / "media"
    artifacts.mkdir()
    artifact_dir.mkdir()
    task, scene = _make_scene(media)
    adapter = GrootPolicyAdapter(
        _fake_process_bridge(repo, artifacts, media),
        "groot-n16",
        artifact_dir=artifact_dir,
        allowed_media_root=media,
        checkpoint_identity=CHECKPOINT_IDENTITY,
        checkpoint_digest=CHECKPOINT_DIGEST,
    )

    with pytest.raises(BridgeRemoteError):
        adapter.act(task, scene, operation_id="round-1:act:groot-n16")

    assert json.loads((repo / "calls.json").read_text()) == [
        ["init", "127.0.0.1", 5555, 15000, False],
        "ping",
        ["endpoint", "evoworld_identity", False],
    ]


def test_groot_process_propagates_remote_error_and_timeout(tmp_path: Path) -> None:
    repo = _make_fake_backend_repo(tmp_path)
    artifacts = tmp_path / "artifacts"
    media = tmp_path / "media"
    artifacts.mkdir()
    media.mkdir()
    bridge = _fake_process_bridge(repo, artifacts, media)
    for instruction, timeout, error in (
        ("explode", 1.0, BridgeRemoteError),
        ("wrong dtype", 1.0, BridgeRemoteError),
        ("wrong rank", 1.0, BridgeRemoteError),
        ("empty action", 1.0, BridgeRemoteError),
        ("nonfinite action", 1.0, BridgeRemoteError),
        ("slow", 0.05, BridgeTimeoutError),
    ):
        artifact_dir = artifacts / instruction
        artifact_dir.mkdir()
        task, scene = _make_scene(media, instruction)
        adapter = GrootPolicyAdapter(
            bridge,
            "groot-n16",
            artifact_dir=artifact_dir,
            allowed_media_root=media,
            checkpoint_identity=CHECKPOINT_IDENTITY,
            checkpoint_digest=CHECKPOINT_DIGEST,
            timeout_seconds=timeout,
        )
        with pytest.raises(error):
            adapter.act(task, scene, operation_id="round-1:act:groot-n16")


def test_groot_adapter_rejects_blank_operation_id(tmp_path: Path) -> None:
    media = tmp_path / "media"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    task, scene = _make_scene(media, "move")
    adapter = GrootPolicyAdapter(
        _NeverRequester(),
        "groot-n16",
        artifact_dir=artifacts,
        allowed_media_root=media,
        checkpoint_identity=CHECKPOINT_IDENTITY,
        checkpoint_digest=CHECKPOINT_DIGEST,
    )

    with pytest.raises(AdapterSchemaError, match="operation_id"):
        adapter.act(task, scene, operation_id="")


def test_groot_adapter_correlates_exact_id_retries_and_rejects_mismatch(
    tmp_path: Path,
) -> None:
    media = tmp_path / "media"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    task, scene = _make_scene(media, "move")
    requester = _Requester()
    adapter = GrootPolicyAdapter(
        requester,
        "groot-n16",
        artifact_dir=artifacts,
        allowed_media_root=media,
        checkpoint_identity=CHECKPOINT_IDENTITY,
        checkpoint_digest=CHECKPOINT_DIGEST,
    )

    adapter.act(task, scene, operation_id="round-1:act")
    adapter.act(task, scene, operation_id="round-1:act")
    adapter.act(task, scene, operation_id="round-2:act")
    assert [request.request_id for request in requester.requests] == [
        "round-1:act",
        "round-2:act",
    ]
    with pytest.raises(OperationConflictError):
        adapter.act(TaskSpec(task.task_id, "different"), scene, operation_id="round-1:act")

    mismatched = GrootPolicyAdapter(
        _Requester(response_id="other"),
        "groot-n16",
        artifact_dir=artifacts,
        allowed_media_root=media,
        checkpoint_identity=CHECKPOINT_IDENTITY,
        checkpoint_digest=CHECKPOINT_DIGEST,
    )
    with pytest.raises(AdapterSchemaError, match="correlation"):
        mismatched.act(task, scene, operation_id="round-3:act")


def test_groot_core_adapter_delays_heavy_imports() -> None:
    code = (
        "import sys; import evoworld.adapters.groot; "
        "assert 'gr00t' not in sys.modules; assert 'zmq' not in sys.modules; "
        "assert 'msgpack' not in sys.modules; assert 'numpy' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.backend
def test_fake_layout_represents_exact_pinned_source() -> None:
    source = Path(__file__).parents[2] / "third_party" / "isaac_groot"
    assert (
        subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        == PIN
    )
    client = (source / "gr00t" / "policy" / "server_client.py").read_text()
    server = (source / "gr00t" / "eval" / "run_gr00t_server.py").read_text()
    assert all(
        name in client for name in ("def ping", "def get_modality_config", "def _get_action")
    )
    assert "class ServerConfig" in server and "model_path" in server

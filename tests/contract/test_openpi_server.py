"""Offline contracts for the checked-in OpenPi server wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest

from evoworld.ports import OperationConflictError

_SERVER_PATH = Path(__file__).parents[2] / "integrations" / "openpi" / "policy_server.py"
_SERVER_SPEC = importlib.util.spec_from_file_location("openpi_policy_server", _SERVER_PATH)
assert _SERVER_SPEC is not None and _SERVER_SPEC.loader is not None
policy_server = importlib.util.module_from_spec(_SERVER_SPEC)
sys.modules[_SERVER_SPEC.name] = policy_server
_SERVER_SPEC.loader.exec_module(policy_server)


def _checkpoint(tmp_path: Path) -> Path:
    checkpoint = (tmp_path / "checkpoint").resolve()
    (checkpoint / "assets" / "droid").mkdir(parents=True)
    (checkpoint / "params").mkdir()
    (checkpoint / "assets" / "droid" / "norm_stats.json").write_text(
        '{"mean":[0.0]}', encoding="utf-8"
    )
    (checkpoint / "params" / "weights").write_bytes(b"pinned-weights")
    return checkpoint


def test_checkpoint_identity_binds_canonical_path_and_all_content(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)

    identity, digest, manifest = policy_server.checkpoint_identity(checkpoint)

    assert identity == str(checkpoint)
    assert digest == manifest["sha256"]
    assert len(digest) == 64
    (checkpoint / "params" / "weights").write_bytes(b"different-weights")
    _, changed_digest, _ = policy_server.checkpoint_identity(checkpoint)
    assert changed_digest != digest


def test_checkpoint_identity_rejects_noncanonical_symlink(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    alias = tmp_path / "checkpoint-alias"
    alias.symlink_to(checkpoint, target_is_directory=True)

    with pytest.raises(ValueError, match="real directory"):
        policy_server.checkpoint_identity(alias)


def test_server_uses_pinned_loader_and_server_signatures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _checkpoint(tmp_path)
    events: list[object] = []

    class FakePolicy:
        metadata = {
            "reset_pose": [0.0, 1.0],
            "checkpoint_identity": "untrusted-training-value",
            "checkpoint_digest": "untrusted-training-value",
        }

    class FakeServer:
        def __init__(
            self,
            policy: FakePolicy,
            host: str = "0.0.0.0",
            port: int | None = None,
            metadata: dict[str, object] | None = None,
        ) -> None:
            events.append(("server", policy, host, port, metadata))

        def serve_forever(self) -> None:
            events.append("serve_forever")

    train_config = object()
    config_module = ModuleType("openpi.training.config")
    policy_module = ModuleType("openpi.policies.policy_config")
    server_module = ModuleType("openpi.serving.websocket_policy_server")

    def get_config(name: str) -> object:
        events.append(("get_config", name))
        return train_config

    def create_trained_policy(
        config: object,
        checkpoint_dir: str,
        *,
        default_prompt: str | None = None,
    ) -> FakePolicy:
        events.append(("load", config, checkpoint_dir, default_prompt))
        return FakePolicy()

    config_module.get_config = get_config  # type: ignore[attr-defined]
    policy_module.create_trained_policy = create_trained_policy  # type: ignore[attr-defined]
    server_module.WebsocketPolicyServer = FakeServer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openpi.training.config", config_module)
    monkeypatch.setitem(sys.modules, "openpi.policies.policy_config", policy_module)
    monkeypatch.setitem(sys.modules, "openpi.serving.websocket_policy_server", server_module)
    monkeypatch.setattr(
        policy_server,
        "_verify_repo",
        lambda repo_root, expected_pin: repo_root,
    )

    args = policy_server.ServerArgs(
        repo_root=tmp_path,
        expected_pin=policy_server.OPENPI_PIN,
        policy_config="pi05_droid",
        checkpoint=checkpoint,
        port=8123,
        default_prompt="move the cup",
    )
    policy_server.serve(args)

    identity, digest, _ = policy_server.checkpoint_identity(checkpoint)
    assert events[:2] == [
        ("get_config", "pi05_droid"),
        ("load", train_config, str(checkpoint), "move the cup"),
    ]
    server_event = events[2]
    assert isinstance(server_event, tuple)
    assert server_event[:4] == ("server", server_event[1], "0.0.0.0", 8123)
    metadata = server_event[4]
    assert metadata == {
        "reset_pose": [0.0, 1.0],
        "checkpoint_identity": identity,
        "checkpoint_digest": digest,
    }
    assert events[3] == "serve_forever"


def test_server_detects_checkpoint_drift_during_policy_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _checkpoint(tmp_path)
    server_created = False

    class FakePolicy:
        metadata: dict[str, object] = {}

    def mutate_while_loading(
        policy_config: str, checkpoint_path: Path, default_prompt: str | None
    ) -> FakePolicy:
        del policy_config, default_prompt
        (checkpoint_path / "params" / "weights").write_bytes(b"mutated-during-load")
        return FakePolicy()

    def create_server(policy: Any, port: int, metadata: dict[str, object]) -> Any:
        nonlocal server_created
        del policy, port, metadata
        server_created = True
        raise AssertionError("server must not be created after checkpoint drift")

    monkeypatch.setattr(policy_server, "_verify_repo", lambda repo_root, expected_pin: repo_root)
    monkeypatch.setattr(policy_server, "_load_policy", mutate_while_loading)
    monkeypatch.setattr(policy_server, "_create_server", create_server)
    args = policy_server.ServerArgs(
        tmp_path,
        policy_server.OPENPI_PIN,
        "pi05_droid",
        checkpoint,
        8000,
    )

    with pytest.raises(OperationConflictError, match="content changed"):
        policy_server.serve(args)
    assert not server_created


def test_server_rejects_wrong_expected_pin_before_checkout_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspected = False

    def inspect(project_root: Path, repo_root: Path, relative_path: str) -> str:
        nonlocal inspected
        del project_root, repo_root, relative_path
        inspected = True
        return policy_server.OPENPI_PIN

    monkeypatch.setattr(policy_server, "verify_backend_checkout", inspect)

    with pytest.raises(ValueError, match="expected_pin"):
        policy_server._verify_repo(tmp_path, "0" * 40)
    assert not inspected

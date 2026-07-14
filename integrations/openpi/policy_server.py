"""Verified server entry point for the pinned OpenPi policy implementation."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib
from pathlib import Path
import re
import sys
from typing import Protocol, cast

PROJECT_ROOT = Path(__file__).parents[2].resolve()
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from evoworld.adapters.content import (  # noqa: E402 - bootstrap project source path first.
    combined_manifest,
    directory_manifest,
    verify_manifest,
)
from evoworld.adapters.revisions import (  # noqa: E402 - bootstrap project source path first.
    verify_backend_checkout,
)
from evoworld.contracts import JsonObject  # noqa: E402 - bootstrap project source path first.

OPENPI_PIN = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
_POLICY_CONFIG = re.compile(r"^[A-Za-z0-9_-]+$")


class Policy(Protocol):
    """The official policy surface used by the pinned server."""

    @property
    def metadata(self) -> Mapping[str, object]: ...


class Server(Protocol):
    """The blocking official websocket server surface."""

    def serve_forever(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ServerArgs:
    """Explicit inputs needed to reproduce one served checkpoint."""

    repo_root: Path
    expected_pin: str
    policy_config: str
    checkpoint: Path
    port: int
    default_prompt: str | None = None


def _checkpoint_path(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("checkpoint must be an absolute real directory")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError("checkpoint is missing") from error
    if resolved != path or not resolved.is_dir():
        raise ValueError("checkpoint must be a canonical real directory")
    return resolved


def checkpoint_identity(path: Path) -> tuple[str, str, JsonObject]:
    """Bind a canonical checkpoint path to the complete bytes below it."""
    checkpoint = _checkpoint_path(path)
    manifest = combined_manifest(
        (directory_manifest(checkpoint, checkpoint.parent, "OpenPi checkpoint"),)
    )
    digest = manifest.get("sha256")
    if not isinstance(digest, str):
        raise RuntimeError("OpenPi checkpoint manifest has no digest")
    return str(checkpoint), digest, manifest


def _verify_repo(repo_root: Path, expected_pin: str) -> Path:
    if expected_pin != OPENPI_PIN:
        raise ValueError("expected_pin does not match the supported OpenPi pin")
    revision = verify_backend_checkout(PROJECT_ROOT, repo_root, "third_party/openpi")
    if revision != expected_pin:
        raise RuntimeError("OpenPi checkout does not match expected_pin")
    return repo_root.resolve(strict=True)


def _load_policy(policy_config: str, checkpoint: Path, default_prompt: str | None) -> Policy:
    """Call the exact loader used by the pinned upstream serve_policy script."""
    config_module = importlib.import_module("openpi.training.config")
    policy_module = importlib.import_module("openpi.policies.policy_config")
    train_config = getattr(config_module, "get_config")(policy_config)
    loader = getattr(policy_module, "create_trained_policy")
    return cast(
        Policy,
        loader(train_config, str(checkpoint), default_prompt=default_prompt),
    )


def _create_server(policy: Policy, port: int, metadata: dict[str, object]) -> Server:
    server_module = importlib.import_module("openpi.serving.websocket_policy_server")
    factory = getattr(server_module, "WebsocketPolicyServer")
    return cast(
        Server,
        factory(policy=policy, host="0.0.0.0", port=port, metadata=metadata),
    )


def serve(args: ServerArgs) -> None:
    """Verify, load, re-verify, and serve one immutable checkpoint identity."""
    _verify_repo(args.repo_root, args.expected_pin)
    if _POLICY_CONFIG.fullmatch(args.policy_config) is None:
        raise ValueError("policy_config contains unsupported characters")
    if type(args.port) is not int or not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    identity, digest, manifest = checkpoint_identity(args.checkpoint)
    policy = _load_policy(args.policy_config, args.checkpoint, args.default_prompt)
    verify_manifest(manifest, "OpenPi checkpoint")
    _verify_repo(args.repo_root, args.expected_pin)
    policy_metadata = policy.metadata
    if not isinstance(policy_metadata, Mapping) or not all(
        isinstance(key, str) for key in policy_metadata
    ):
        raise TypeError("OpenPi policy metadata must be a string-keyed mapping")
    metadata = dict(policy_metadata)
    metadata.update(
        {
            "checkpoint_identity": identity,
            "checkpoint_digest": digest,
        }
    )
    _create_server(policy, args.port, metadata).serve_forever()


def _parse_args(argv: Sequence[str] | None = None) -> ServerArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-pin", required=True)
    parser.add_argument("--policy-config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--default-prompt")
    parsed = parser.parse_args(argv)
    return ServerArgs(
        repo_root=parsed.repo_root,
        expected_pin=parsed.expected_pin,
        policy_config=parsed.policy_config,
        checkpoint=parsed.checkpoint,
        port=parsed.port,
        default_prompt=parsed.default_prompt,
    )


def main() -> None:
    """Run the checked-in wrapper without modifying the pinned submodule."""
    serve(_parse_args())


if __name__ == "__main__":
    main()

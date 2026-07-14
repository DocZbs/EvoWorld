"""Pinned GR00T server with an independently queryable checkpoint identity."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import importlib
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evoworld.adapters.content import (  # noqa: E402
    combined_manifest,
    directory_manifest,
    verify_manifest,
)
from evoworld.contracts import JsonObject  # noqa: E402


Runner = Callable[[Sequence[str]], str]


def _run(command: Sequence[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError("pinned GR00T checkout verification failed")
    return completed.stdout


def _real_directory(path: Path, field_name: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{field_name} must be an absolute non-symlink directory")
    resolved = path.resolve(strict=True)
    if resolved != path or not resolved.is_dir():
        raise ValueError(f"{field_name} must not traverse symlinks")
    return resolved


def _verify_checkout(repo: Path, expected_revision: str, runner: Runner) -> None:
    if len(expected_revision) != 40 or any(
        character not in "0123456789abcdef" for character in expected_revision
    ):
        raise ValueError("expected_revision must be a lowercase Git commit")
    revision = runner(("git", "-C", str(repo), "rev-parse", "HEAD")).strip()
    status = runner(("git", "-C", str(repo), "status", "--porcelain"))
    if revision != expected_revision or status:
        raise RuntimeError("GR00T checkout does not match the expected clean pin")


def _checkpoint_manifest(checkpoint: Path) -> JsonObject:
    return combined_manifest(
        [directory_manifest(checkpoint, checkpoint.parent, "GR00T checkpoint")]
    )


def create_server(
    *,
    repo_root: Path,
    checkpoint_path: Path,
    expected_revision: str,
    embodiment_tag: str,
    device: str,
    host: str,
    port: int,
    strict: bool,
    runner: Runner = _run,
    policy_factory: Callable[..., object] | None = None,
    server_factory: Callable[..., object] | None = None,
) -> object:
    """Load the official policy and expose identity over a separate endpoint."""
    repo = _real_directory(repo_root, "repo_root")
    checkpoint = _real_directory(checkpoint_path, "checkpoint_path")
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    _verify_checkout(repo, expected_revision, runner)
    manifest = _checkpoint_manifest(checkpoint)

    if policy_factory is None:
        module = importlib.import_module("gr00t.policy.gr00t_policy")
        policy_factory = getattr(module, "Gr00tPolicy")
    if server_factory is None:
        module = importlib.import_module("gr00t.policy.server_client")
        server_factory = getattr(module, "PolicyServer")
    embodiment_module = importlib.import_module("gr00t.data.embodiment_tags")
    embodiment_enum = getattr(embodiment_module, "EmbodimentTag")
    try:
        embodiment = embodiment_enum[embodiment_tag]
    except KeyError as error:
        raise ValueError("embodiment_tag is not defined by the pinned GR00T API") from error

    policy = policy_factory(
        embodiment_tag=embodiment,
        model_path=str(checkpoint),
        device=device,
        strict=strict,
    )
    verify_manifest(manifest, "GR00T checkpoint")
    _verify_checkout(repo, expected_revision, runner)
    server = server_factory(policy=policy, host=host, port=port)
    register_endpoint = getattr(server, "register_endpoint")

    def identity() -> Mapping[str, object]:
        verify_manifest(manifest, "GR00T checkpoint")
        return {
            "checkpoint_identity": str(checkpoint),
            "checkpoint_digest": manifest["sha256"],
        }

    register_endpoint("evoworld_identity", identity, requires_input=False)
    return server


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--embodiment-tag", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--strict", choices=("True", "False"), default="True")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    server = create_server(
        repo_root=args.repo_root,
        checkpoint_path=args.checkpoint_path,
        expected_revision=args.expected_revision,
        embodiment_tag=args.embodiment_tag,
        device=args.device,
        host=args.host,
        port=args.port,
        strict=args.strict == "True",
    )
    try:
        getattr(server, "run")()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

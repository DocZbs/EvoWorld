"""Run the complete lightweight EvoWorld release gate locally."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Protocol, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OWNED_FORMAT_PATHS = ("src", "training", "integrations", "scripts", "tests")
RELEASE_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("uv", "run", "python", "scripts/verify_submodules.py"),
    ("uv", "run", "pytest"),
    ("uv", "run", "ruff", "check", "."),
    ("uv", "run", "ruff", "format", "--check", *OWNED_FORMAT_PATHS),
    ("uv", "run", "mypy", "src", "training", "integrations"),
    ("uv", "run", "evoworld", "validate", "--config", "configs/dry_run.toml"),
    (
        "uv",
        "run",
        "evoworld",
        "run",
        "--config",
        "configs/dry_run.toml",
        "--rounds",
        "1",
    ),
)


class CommandRunner(Protocol):
    """Execution boundary used by the release verifier."""

    def run(self, argv: Sequence[str], *, cwd: Path) -> int:
        """Run one argv array in ``cwd`` and return its process status."""


class SubprocessRunner:
    """Production runner with no shell expansion."""

    def run(self, argv: Sequence[str], *, cwd: Path) -> int:
        completed = subprocess.run(tuple(argv), cwd=cwd, check=False)
        return completed.returncode


def verify_release(runner: CommandRunner | None = None) -> int:
    """Run each release command in order and return the first failure."""

    command_runner = SubprocessRunner() if runner is None else runner
    for command in RELEASE_COMMANDS:
        returncode = command_runner.run(command, cwd=REPOSITORY_ROOT)
        if returncode != 0:
            return returncode
    return 0


def main(runner: CommandRunner | None = None) -> int:
    """CLI entry point."""

    return verify_release(runner)


if __name__ == "__main__":
    raise SystemExit(main())

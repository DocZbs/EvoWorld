"""Verification for the immutable upstream backend revisions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
import re
import subprocess

EXPECTED_REVISIONS = {
    "third_party/ctrl_world": "99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d",
    "third_party/openpi": "15a9616a00943ada6c20a0f158e3adb39df2ccac",
    "third_party/isaac_groot": "ead52833afbbf4243f8cd5e7664f48a94de03b19",
}

GitRunner = Callable[[Sequence[str]], str]
_STATUS_LINE = re.compile(r"^(?P<state>[ +-U])(?P<revision>[0-9a-f]{40}) (?P<path>.+)$")


class SubmoduleAuditError(RuntimeError):
    """Base class for a checkout that cannot reproduce the pinned backends."""


class SubmoduleMissingError(SubmoduleAuditError):
    """A required submodule directory or git checkout is unavailable."""


class SubmoduleRevisionError(SubmoduleAuditError):
    """A required top-level submodule is not at its pinned commit."""


class SubmoduleDirtyError(SubmoduleAuditError):
    """A submodule has local changes or differs from its recorded gitlink."""


class SubmoduleUninitializedError(SubmoduleAuditError):
    """A recursive submodule has not been initialized."""


class SubmoduleConflictError(SubmoduleAuditError):
    """A recursive submodule is in a merge conflict state."""


def _run_git(command: Sequence[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "git command failed"
        raise SubmoduleMissingError(f"{detail}: {' '.join(command)}")
    return completed.stdout


def _registered_path(repo_root: Path, rendered_path: str) -> str:
    """Separate an optional git describe suffix without corrupting a real path."""
    if (repo_root / rendered_path).is_dir():
        return rendered_path
    suffix_start = rendered_path.rfind(" (")
    while suffix_start >= 0:
        candidate = rendered_path[:suffix_start]
        if rendered_path.endswith(")") and (repo_root / candidate).is_dir():
            return candidate
        suffix_start = rendered_path.rfind(" (", 0, suffix_start)
    return rendered_path


def _recursive_entries(repo_root: Path, output: str) -> tuple[tuple[str, str], ...]:
    entries: list[tuple[str, str]] = []
    for line in output.splitlines():
        match = _STATUS_LINE.match(line)
        if match is None:
            raise SubmoduleRevisionError(f"unreadable recursive submodule status: {line!r}")
        state = match.group("state")
        path = _registered_path(repo_root, match.group("path"))
        if state == "-":
            raise SubmoduleUninitializedError(f"submodule is uninitialized: {path}")
        if state == "U":
            raise SubmoduleConflictError(f"submodule has a conflict: {path}")
        if state == "+":
            raise SubmoduleDirtyError(f"submodule differs from its recorded revision: {path}")
        entries.append((path, match.group("revision")))
    return tuple(entries)


def _ensure_clean(path: Path, relative_path: str, runner: GitRunner) -> str:
    try:
        revision = runner(("git", "-C", str(path), "rev-parse", "HEAD")).strip()
        status = runner(("git", "-C", str(path), "status", "--porcelain"))
    except SubmoduleAuditError:
        raise
    except Exception as error:
        raise SubmoduleMissingError(f"unable to inspect submodule: {relative_path}") from error
    if status:
        raise SubmoduleDirtyError(f"submodule has local changes: {relative_path}")
    return revision


def inspect_submodules(repo_root: Path, runner: GitRunner = _run_git) -> dict[str, str]:
    """Return pinned revisions after rejecting every unusable nested checkout."""
    repo_root = repo_root.resolve()
    try:
        recursive_status = runner(
            ("git", "-C", str(repo_root), "submodule", "status", "--recursive")
        )
    except SubmoduleAuditError:
        raise
    except Exception as error:
        raise SubmoduleMissingError("unable to inspect recursive submodule status") from error

    recursive_entries = _recursive_entries(repo_root, recursive_status)
    registered_paths = {path for path, _ in recursive_entries}
    for relative_path in EXPECTED_REVISIONS:
        if relative_path not in registered_paths:
            raise SubmoduleMissingError(f"required submodule is not registered: {relative_path}")

    resolved: dict[str, str] = {}
    for relative_path, expected_revision in EXPECTED_REVISIONS.items():
        path = repo_root / relative_path
        if not path.is_dir():
            raise SubmoduleMissingError(f"required submodule is missing: {relative_path}")
        revision = _ensure_clean(path, relative_path, runner)
        if revision != expected_revision:
            raise SubmoduleRevisionError(
                f"submodule revision mismatch for {relative_path}: "
                f"expected {expected_revision}, got {revision}"
            )
        resolved[relative_path] = revision

    for relative_path, _ in recursive_entries:
        if relative_path in EXPECTED_REVISIONS:
            continue
        path = repo_root / relative_path
        if not path.is_dir():
            raise SubmoduleMissingError(f"required submodule is missing: {relative_path}")
        _ensure_clean(path, relative_path, runner)
    return resolved


def verify_backend_checkout(
    project_root: Path,
    repo_root: Path,
    relative_path: str,
    *,
    runner: GitRunner = _run_git,
) -> str:
    """Verify one backend is the exact audited gitlink under this project root."""
    if relative_path not in EXPECTED_REVISIONS:
        raise SubmoduleRevisionError("backend path is not an audited gitlink")
    if project_root.is_symlink() or repo_root.is_symlink():
        raise SubmoduleDirtyError("backend checkout paths must not be symlinks")
    try:
        project = project_root.resolve(strict=True)
        repo = repo_root.resolve(strict=True)
    except FileNotFoundError as error:
        raise SubmoduleMissingError("backend checkout path is missing") from error
    expected = (project / relative_path).resolve(strict=True)
    if repo != expected:
        raise SubmoduleMissingError("backend checkout is not the audited project gitlink")
    revisions = inspect_submodules(project, runner=runner)
    revision = revisions.get(relative_path)
    if revision != EXPECTED_REVISIONS[relative_path]:
        raise SubmoduleRevisionError("backend checkout revision does not match its pin")
    return revision

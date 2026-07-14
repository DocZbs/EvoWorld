"""Contract tests for the pinned upstream revision gate."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from evoworld.adapters.revisions import (
    EXPECTED_REVISIONS,
    SubmoduleConflictError,
    SubmoduleDirtyError,
    SubmoduleRevisionError,
    SubmoduleUninitializedError,
    inspect_submodules,
)


class FakeGitRunner:
    """Small deterministic model of the git calls issued by the auditor."""

    def __init__(
        self,
        repo_root: Path,
        heads: dict[str, str],
        dirty: set[str] | None = None,
        recursive_status: tuple[str, ...] | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.heads = heads
        self.dirty = dirty or set()
        self.recursive_status = recursive_status
        for path in heads:
            (repo_root / path).mkdir(parents=True, exist_ok=True)

    def __call__(self, command: Sequence[str]) -> str:
        command = tuple(command)
        assert command[:2] == ("git", "-C")
        path = Path(command[2]).relative_to(self.repo_root).as_posix()
        operation = command[3:]
        if operation == ("rev-parse", "HEAD"):
            return self.heads[path] + "\n"
        if operation == ("status", "--porcelain"):
            return " M tracked-file\n" if path in self.dirty else ""
        if operation == ("submodule", "status", "--recursive"):
            if self.recursive_status is not None:
                return "\n".join(self.recursive_status) + "\n"
            return (
                "\n".join(
                    f" {revision} {submodule_path}"
                    for submodule_path, revision in self.heads.items()
                )
                + "\n"
            )
        raise AssertionError(f"unexpected git command: {command!r}")


def test_revision_audit_returns_expected_resolved_revisions(tmp_path: Path) -> None:
    runner = FakeGitRunner(tmp_path, heads=dict(EXPECTED_REVISIONS))

    assert inspect_submodules(tmp_path, runner=runner) == EXPECTED_REVISIONS


def test_revision_audit_rejects_wrong_head(tmp_path: Path) -> None:
    heads = dict(EXPECTED_REVISIONS, **{"third_party/openpi": "wrong"})
    runner = FakeGitRunner(tmp_path, heads=heads)

    with pytest.raises(SubmoduleRevisionError, match="openpi"):
        inspect_submodules(tmp_path, runner=runner)


def test_revision_audit_rejects_dirty_nested_module(tmp_path: Path) -> None:
    nested_path = "third_party/openpi/third_party/libero"
    runner = FakeGitRunner(
        tmp_path,
        heads=dict(EXPECTED_REVISIONS),
        recursive_status=(
            *(f" {revision} {path}" for path, revision in EXPECTED_REVISIONS.items()),
            f"+f78abd68ee283de9f9be3c8f7e2a9ad60246e95c {nested_path}",
        ),
    )

    with pytest.raises(SubmoduleDirtyError, match="libero"):
        inspect_submodules(tmp_path, runner=runner)


@pytest.mark.parametrize(
    ("path", "rendered_path"),
    [
        (
            "third_party/openpi/third_party/path (with parens)",
            "third_party/openpi/third_party/path (with parens)",
        ),
        (
            "third_party/openpi/third_party/libero",
            "third_party/openpi/third_party/libero (heads/main)",
        ),
    ],
)
def test_revision_audit_distinguishes_parentheses_in_paths_from_git_descriptions(
    tmp_path: Path, path: str, rendered_path: str
) -> None:
    nested_revision = "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"
    heads = {**EXPECTED_REVISIONS, path: nested_revision}
    runner = FakeGitRunner(
        tmp_path,
        heads=heads,
        recursive_status=(
            *(
                f" {revision} {submodule_path}"
                for submodule_path, revision in heads.items()
                if submodule_path != path
            ),
            f" {nested_revision} {rendered_path}",
        ),
    )

    assert inspect_submodules(tmp_path, runner=runner) == EXPECTED_REVISIONS


def test_revision_audit_classifies_uninitialized_top_level_before_missing(
    tmp_path: Path,
) -> None:
    runner = FakeGitRunner(
        tmp_path,
        heads=dict(EXPECTED_REVISIONS),
        recursive_status=tuple(
            f"{'-' if path == 'third_party/openpi' else ' '}{revision} {path}"
            for path, revision in EXPECTED_REVISIONS.items()
        ),
    )
    (tmp_path / "third_party/openpi").rmdir()

    with pytest.raises(SubmoduleUninitializedError, match="openpi"):
        inspect_submodules(tmp_path, runner=runner)


@pytest.mark.parametrize(
    ("marker", "error_type"),
    [("-", SubmoduleUninitializedError), ("U", SubmoduleConflictError)],
)
def test_revision_audit_rejects_unusable_recursive_module(
    tmp_path: Path, marker: str, error_type: type[Exception]
) -> None:
    runner = FakeGitRunner(
        tmp_path,
        heads=dict(EXPECTED_REVISIONS),
        recursive_status=(
            *(f" {revision} {path}" for path, revision in EXPECTED_REVISIONS.items()),
            f"{marker}f78abd68ee283de9f9be3c8f7e2a9ad60246e95c third_party/openpi/third_party/libero",
        ),
    )

    with pytest.raises(error_type, match="libero"):
        inspect_submodules(tmp_path, runner=runner)

"""Release-verifier and lightweight CI contract tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from typing import Sequence

import pytest

PROJECT_ROOT = Path(__file__).parents[2].resolve()
FORMAT_PATHS = ("src", "training", "integrations", "scripts", "tests")
BACKEND_NODEIDS = frozenset(
    {
        "tests/contract/test_backend_revision_gates.py::test_builders_reject_fake_symlink_dirty_and_wrong_revision[third_party/openpi-build_openpi_process_bridge-AdapterSchemaError]",
        "tests/contract/test_backend_revision_gates.py::test_builders_reject_fake_symlink_dirty_and_wrong_revision[third_party/isaac_groot-build_groot_process_bridge-AdapterSchemaError]",
        "tests/contract/test_backend_revision_gates.py::test_process_launch_rechecks_source_after_builder_validation[third_party/openpi-build_openpi_process_bridge-AdapterSchemaError]",
        "tests/contract/test_backend_revision_gates.py::test_process_launch_rechecks_source_after_builder_validation[third_party/isaac_groot-build_groot_process_bridge-AdapterSchemaError]",
        "tests/contract/test_ctrl_world_adapter.py::test_pinned_replay_agent_seam_matches_the_bridge",
        "tests/contract/test_groot_adapter.py::test_groot_server_command_uses_pinned_n16_shape",
        "tests/contract/test_groot_adapter.py::test_groot_server_command_requires_strict_valid_integer_port[False]",
        "tests/contract/test_groot_adapter.py::test_groot_server_command_requires_strict_valid_integer_port[True]",
        "tests/contract/test_groot_adapter.py::test_groot_server_command_requires_strict_valid_integer_port[5555.0]",
        "tests/contract/test_groot_adapter.py::test_groot_server_command_requires_strict_valid_integer_port[0]",
        "tests/contract/test_groot_adapter.py::test_groot_server_command_requires_strict_valid_integer_port[65536]",
        "tests/contract/test_groot_adapter.py::test_groot_server_command_rejects_command_injection[]",
        "tests/contract/test_groot_adapter.py::test_groot_server_command_rejects_command_injection[NEW EMBODIMENT]",
        "tests/contract/test_groot_adapter.py::test_groot_server_command_rejects_command_injection[x; touch /tmp/injected]",
        "tests/contract/test_groot_adapter.py::test_groot_process_bridge_has_fixed_absolute_boundary",
        "tests/contract/test_groot_adapter.py::test_fake_layout_represents_exact_pinned_source",
        "tests/contract/test_openpi_adapter.py::test_openpi_serve_command_is_argument_safe",
        "tests/contract/test_openpi_adapter.py::test_openpi_serve_command_rejects_invalid_port[False]",
        "tests/contract/test_openpi_adapter.py::test_openpi_serve_command_rejects_invalid_port[True]",
        "tests/contract/test_openpi_adapter.py::test_openpi_serve_command_rejects_invalid_port[8000.0]",
        "tests/contract/test_openpi_adapter.py::test_openpi_serve_command_rejects_invalid_port[0]",
        "tests/contract/test_openpi_adapter.py::test_openpi_serve_command_rejects_invalid_port[65536]",
        "tests/contract/test_openpi_adapter.py::test_openpi_server_command_rejects_symlinked_checkpoint",
        "tests/contract/test_openpi_adapter.py::test_openpi_process_bridge_has_absolute_fixed_boundary",
        "tests/contract/test_openpi_adapter.py::test_real_pillow_decode_closes_images_and_calls_pinned_resize",
        "tests/integration/test_real_composition.py::test_real_composition_builds_public_adapters_and_requires_private_ports",
    }
)
_SPEC = importlib.util.spec_from_file_location(
    "verify_release", PROJECT_ROOT / "scripts/verify_release.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_VERIFY_RELEASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_VERIFY_RELEASE)
main = _VERIFY_RELEASE.main
verify_release = _VERIFY_RELEASE.verify_release


class RecordingRunner:
    def __init__(self, results: Sequence[int] = ()) -> None:
        self.results = iter(results)
        self.calls: list[tuple[tuple[str, ...], Path]] = []

    def run(self, argv: Sequence[str], *, cwd: Path) -> int:
        self.calls.append((tuple(argv), cwd))
        return next(self.results, 0)


def test_release_verifier_runs_every_local_gate_from_repository_root() -> None:
    runner = RecordingRunner()

    assert verify_release(runner) == 0

    assert runner.calls == [
        (("uv", "run", "python", "scripts/verify_submodules.py"), PROJECT_ROOT),
        (("uv", "run", "pytest"), PROJECT_ROOT),
        (("uv", "run", "ruff", "check", "."), PROJECT_ROOT),
        (("uv", "run", "ruff", "format", "--check", *FORMAT_PATHS), PROJECT_ROOT),
        (("uv", "run", "mypy", "src", "training", "integrations"), PROJECT_ROOT),
        (
            (
                "uv",
                "run",
                "evoworld",
                "validate",
                "--config",
                "configs/dry_run.toml",
            ),
            PROJECT_ROOT,
        ),
        (
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
            PROJECT_ROOT,
        ),
    ]


def test_release_verifier_stops_on_the_first_test_failure() -> None:
    runner = RecordingRunner((0, 17, 0))

    assert verify_release(runner) == 17
    assert [call[0] for call in runner.calls] == [
        ("uv", "run", "python", "scripts/verify_submodules.py"),
        ("uv", "run", "pytest"),
    ]


def test_release_verifier_stops_on_submodule_audit_failure() -> None:
    runner = RecordingRunner((19, 0))

    assert verify_release(runner) == 19
    assert [call[0] for call in runner.calls] == [
        ("uv", "run", "python", "scripts/verify_submodules.py"),
    ]


def test_release_verifier_main_propagates_the_first_failure() -> None:
    runner = RecordingRunner((23,))

    assert main(runner) == 23


def test_subprocess_runner_uses_an_argv_array_without_a_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], Path, bool]] = []

    def run(argv: tuple[str, ...], *, cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        calls.append((argv, cwd, check))
        return subprocess.CompletedProcess(argv, 31)

    monkeypatch.setattr(_VERIFY_RELEASE.subprocess, "run", run)

    result = _VERIFY_RELEASE.SubprocessRunner().run(("tool", "arg with spaces"), cwd=PROJECT_ROOT)

    assert result == 31
    assert calls == [(("tool", "arg with spaces"), PROJECT_ROOT, False)]


def test_ci_is_lightweight_locked_and_explicit_about_submodule_coverage() -> None:
    workflow = (PROJECT_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "permissions:\n  contents: read" in workflow
    assert 'python-version: "3.11"' in workflow
    assert "astral-sh/setup-uv@" in workflow
    assert "uv sync --locked --extra dev" in workflow
    assert "submodules: false" in workflow
    assert "submodules: recursive" not in workflow
    assert "third_party" not in _run_commands(workflow)
    assert 'extend-exclude = ["third_party"]' in (PROJECT_ROOT / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert "uv run pytest -m 'not backend'" in workflow
    assert "workflow_dispatch" not in workflow
    assert "pull_request:" in workflow

    commands = _run_commands(workflow)
    for command in (
        "uv run ruff check .",
        "uv run ruff format --check src training integrations scripts tests",
        "uv run mypy src training integrations",
        "uv run evoworld validate --config configs/dry_run.toml",
        "uv run evoworld run --config configs/dry_run.toml --rounds 1",
    ):
        assert command in commands
    assert "curl " not in commands
    assert "wget " not in commands
    assert "git submodule" not in commands
    assert "cuda" not in commands.lower()


def test_ci_workflow_is_compact() -> None:
    workflow = (PROJECT_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert len(workflow.splitlines()) <= 32
    assert "concurrency:" not in workflow


def test_only_checkout_dependent_tests_are_marked_backend() -> None:
    assert _collect_backend_nodeids(PROJECT_ROOT, ("tests",)) == BACKEND_NODEIDS


def test_backend_collection_detects_function_class_and_module_markers(tmp_path: Path) -> None:
    (tmp_path / "test_function_marker.py").write_text(
        "import pytest\n@pytest.mark.backend\ndef test_function_marker(): pass\n",
        encoding="utf-8",
    )
    (tmp_path / "test_class_marker.py").write_text(
        "import pytest\n"
        "@pytest.mark.backend\n"
        "class TestBackend:\n"
        "    def test_class_marker(self): pass\n",
        encoding="utf-8",
    )
    (tmp_path / "test_module_marker.py").write_text(
        "import pytest\npytestmark = pytest.mark.backend\ndef test_module_marker(): pass\n",
        encoding="utf-8",
    )

    assert _collect_backend_nodeids(tmp_path, (".",)) == {
        "test_class_marker.py::TestBackend::test_class_marker",
        "test_function_marker.py::test_function_marker",
        "test_module_marker.py::test_module_marker",
    }


def _run_commands(workflow: str) -> str:
    return "\n".join(
        line.strip().removeprefix("run: ")
        for line in workflow.splitlines()
        if line.strip().startswith("run:") or line.startswith("          ")
    )


def _collect_backend_nodeids(root: Path, paths: Sequence[str]) -> frozenset[str]:
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-m",
            "backend",
            "-q",
            *paths,
        ),
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return frozenset(line for line in completed.stdout.splitlines() if "::" in line)

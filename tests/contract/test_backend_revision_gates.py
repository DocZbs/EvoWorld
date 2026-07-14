"""Exact-source gates for the OpenPi and GR00T launch boundaries."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
import subprocess
import sys

import pytest

from evoworld.adapters.groot import (
    AdapterSchemaError as GrootSchemaError,
    build_groot_process_bridge,
)
from evoworld.adapters.openpi import (
    AdapterSchemaError as OpenPiSchemaError,
    build_openpi_process_bridge,
)
from evoworld.adapters.process import BridgeRequest

PROJECT_ROOT = Path(__file__).parents[2].resolve()


def _runner(
    target: Path,
    *,
    dirty: Callable[[], bool] = lambda: False,
    wrong_revision: bool = False,
) -> Callable[[Sequence[str]], str]:
    def run(command: Sequence[str]) -> str:
        operation = tuple(command[3:])
        checkout = Path(command[2]).resolve() if len(command) > 2 else PROJECT_ROOT
        if checkout == target.resolve() and operation == ("status", "--porcelain") and dirty():
            return " M tracked.py\n"
        if checkout == target.resolve() and operation == ("rev-parse", "HEAD") and wrong_revision:
            return "0000000000000000000000000000000000000000\n"
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr)
        return completed.stdout

    return run


@pytest.mark.backend
@pytest.mark.parametrize(
    ("relative", "builder", "error_type"),
    [
        ("third_party/openpi", build_openpi_process_bridge, OpenPiSchemaError),
        ("third_party/isaac_groot", build_groot_process_bridge, GrootSchemaError),
    ],
)
def test_builders_reject_fake_symlink_dirty_and_wrong_revision(
    tmp_path: Path,
    relative: str,
    builder: Callable[..., object],
    error_type: type[Exception],
) -> None:
    artifacts = tmp_path / "artifacts"
    media = tmp_path / "media"
    fake = tmp_path / "fake"
    artifacts.mkdir()
    media.mkdir()
    fake.mkdir()
    real = PROJECT_ROOT / relative

    with pytest.raises(error_type):
        builder(
            fake, artifacts, media, backend_python=Path(sys.executable), project_root=PROJECT_ROOT
        )

    link = tmp_path / "checkout-link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(error_type, match="symlink"):
        builder(
            link, artifacts, media, backend_python=Path(sys.executable), project_root=PROJECT_ROOT
        )

    with pytest.raises(error_type, match="clean pinned"):
        builder(
            real,
            artifacts,
            media,
            backend_python=Path(sys.executable),
            project_root=PROJECT_ROOT,
            revision_runner=_runner(real, dirty=lambda: True),
        )
    with pytest.raises(error_type, match="clean pinned"):
        builder(
            real,
            artifacts,
            media,
            backend_python=Path(sys.executable),
            project_root=PROJECT_ROOT,
            revision_runner=_runner(real, wrong_revision=True),
        )


@pytest.mark.backend
@pytest.mark.parametrize(
    ("relative", "builder", "error_type"),
    [
        ("third_party/openpi", build_openpi_process_bridge, OpenPiSchemaError),
        ("third_party/isaac_groot", build_groot_process_bridge, GrootSchemaError),
    ],
)
def test_process_launch_rechecks_source_after_builder_validation(
    tmp_path: Path,
    relative: str,
    builder: Callable[..., object],
    error_type: type[Exception],
) -> None:
    artifacts = tmp_path / "artifacts"
    request_artifacts = artifacts / "request"
    media = tmp_path / "media"
    request_artifacts.mkdir(parents=True)
    media.mkdir()
    real = PROJECT_ROOT / relative
    changed = False
    bridge = builder(
        real,
        artifacts,
        media,
        backend_python=Path(sys.executable),
        project_root=PROJECT_ROOT,
        revision_runner=_runner(real, dirty=lambda: changed),
    )
    changed = True

    with pytest.raises(error_type, match="clean pinned"):
        bridge.request(BridgeRequest("operation-1", "act", request_artifacts, {}), 1.0)
    assert not (request_artifacts / ".process").exists()

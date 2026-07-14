"""Contracts for the request-correlated backend process boundary."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from evoworld.adapters.process import (
    BridgeArtifactError,
    BridgeCorrelationError,
    BridgeLaunchError,
    BridgeMalformedResponseError,
    BridgeProcessError,
    BridgeRequest,
    BridgeRemoteError,
    BridgeStdoutError,
    BridgeTimeoutError,
    ProcessBridge,
    process_log_dir,
)


def fixture_bridge(tmp_path: Path, mode: str) -> Path:
    """Create a tiny stdlib-only child process with one deliberate behavior."""
    script = tmp_path / f"bridge_{mode}.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import json
            import sys
            import time

            request = json.loads(sys.stdin.readline())
            mode = {mode!r}
            if mode == "malformed":
                print("{{not-json")
            elif mode == "wrong-id":
                print(json.dumps({{"protocol_version": 1, "request_id": "other", "status": "ok", "result": {{}}}}))
            elif mode == "nonzero":
                print("child diagnostics", file=sys.stderr)
                sys.exit(7)
            elif mode == "timeout":
                time.sleep(1)
            elif mode == "contaminated":
                print("child log")
                print(json.dumps({{"protocol_version": 1, "request_id": request["request_id"], "status": "ok", "result": {{}}}}))
            elif mode == "invalid-status":
                print(json.dumps({{"protocol_version": 1, "request_id": request["request_id"], "status": "wat", "result": {{}}}}))
            elif mode == "non-json-value":
                print('{{"protocol_version": 1, "request_id": "' + request["request_id"] + '", "status": "ok", "result": NaN}}')
            elif mode == "remote-error":
                print(json.dumps({{"protocol_version": 1, "request_id": request["request_id"], "status": "error", "error": {{"message": "bad input"}}}}))
            else:
                print("child diagnostics", file=sys.stderr)
                print(json.dumps({{"protocol_version": 1, "request_id": request["request_id"], "status": "ok", "result": {{"text": "你好"}}}}, ensure_ascii=False))
            """
        ),
        encoding="utf-8",
    )
    return script


def request(artifact_dir: Path, request_id: str = "req-1") -> BridgeRequest:
    return BridgeRequest(request_id, "act", artifact_dir, {"nested": [1, {"ok": True}]})


def bridge(tmp_path: Path, mode: str, artifact_root: Path | None = None) -> ProcessBridge:
    return ProcessBridge(
        (sys.executable, str(fixture_bridge(tmp_path, mode))),
        cwd=tmp_path,
        artifact_root=artifact_root,
        env={"PYTHONIOENCODING": "utf-8"},
    )


def log_dir(artifact_dir: Path, request_id: str = "req-1") -> Path:
    return process_log_dir(request(artifact_dir, request_id))


def test_process_bridge_correlates_response_id_and_preserves_utf8(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "run"
    artifact_dir.mkdir(parents=True)

    response = bridge(tmp_path, "success", tmp_path / "artifacts").request(
        request(artifact_dir), timeout=2.0
    )

    assert response.request_id == "req-1"
    assert response.status == "ok"
    assert response.result == {"text": "你好"}
    logs = log_dir(artifact_dir)
    assert (logs / "process.stdout.log").read_text(encoding="utf-8").endswith("\n")
    assert (logs / "process.stderr.log").read_text(encoding="utf-8") == "child diagnostics\n"
    assert json.loads((logs / "process.error.json").read_text(encoding="utf-8")) == {
        "error": None,
        "outcome": "ok",
        "returncode": 0,
    }


@pytest.mark.parametrize(
    ("mode", "error_type"),
    [
        ("malformed", BridgeMalformedResponseError),
        ("wrong-id", BridgeCorrelationError),
        ("nonzero", BridgeProcessError),
        ("timeout", BridgeTimeoutError),
        ("contaminated", BridgeStdoutError),
        ("invalid-status", BridgeMalformedResponseError),
        ("non-json-value", BridgeMalformedResponseError),
        ("remote-error", BridgeRemoteError),
    ],
)
def test_process_bridge_reports_structured_failures(
    tmp_path: Path, mode: str, error_type: type[Exception]
) -> None:
    artifact_dir = tmp_path / "artifacts" / mode
    artifact_dir.mkdir(parents=True)

    with pytest.raises(error_type):
        bridge(tmp_path, mode, tmp_path / "artifacts").request(request(artifact_dir), timeout=0.05)

    logs = log_dir(artifact_dir)
    assert (logs / "process.stdout.log").is_file()
    assert (logs / "process.stderr.log").is_file()
    error_metadata = json.loads((logs / "process.error.json").read_text(encoding="utf-8"))
    assert error_metadata["error"]
    assert error_metadata["outcome"] == "error"
    assert error_metadata["returncode"] == (
        7 if mode == "nonzero" else None if mode == "timeout" else 0
    )


def test_process_bridge_types_launch_failure_and_persists_metadata(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "run"
    artifact_dir.mkdir(parents=True)
    process = ProcessBridge(
        (str(tmp_path / "does-not-exist"),),
        cwd=tmp_path,
        artifact_root=tmp_path / "artifacts",
        env={},
    )

    with pytest.raises(BridgeLaunchError, match="failed to start"):
        process.request(request(artifact_dir), timeout=1.0)

    logs = log_dir(artifact_dir)
    assert (logs / "process.stdout.log").read_text(encoding="utf-8") == ""
    assert (logs / "process.stderr.log").read_text(encoding="utf-8") == ""
    metadata = json.loads((logs / "process.error.json").read_text(encoding="utf-8"))
    assert metadata == {
        "error": {"message": "backend process failed to start", "type": "BridgeLaunchError"},
        "outcome": "error",
        "returncode": None,
    }


def test_process_bridge_rejects_missing_or_escaping_artifact_directories(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(BridgeArtifactError, match="contained"):
        bridge(tmp_path, "success", root).request(request(outside), timeout=1.0)
    with pytest.raises(BridgeArtifactError, match="missing"):
        bridge(tmp_path, "success", root).request(request(root / "missing"), timeout=1.0)


@pytest.mark.parametrize("target", ["root", "operation"])
def test_process_bridge_rejects_symlinked_log_directories_before_launch(
    tmp_path: Path, target: str
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_dir = artifact_root / "run"
    outside = tmp_path / "outside"
    artifact_dir.mkdir(parents=True)
    outside.mkdir()
    process_logs = artifact_dir / ".process"
    if target == "root":
        process_logs.symlink_to(outside, target_is_directory=True)
    else:
        process_logs.mkdir()
        log_dir(artifact_dir).symlink_to(outside, target_is_directory=True)

    with pytest.raises(BridgeArtifactError, match="log directory"):
        bridge(tmp_path, "success", artifact_root).request(request(artifact_dir), timeout=1.0)

    assert list(outside.iterdir()) == []


def test_bridge_uses_explicit_environment_without_parent_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_dir = tmp_path / "artifacts" / "run"
    artifact_dir.mkdir(parents=True)
    script = tmp_path / "environment.py"
    script.write_text(
        "import json, os, sys\n"
        "request = json.loads(sys.stdin.readline())\n"
        "print(json.dumps({'protocol_version': 1, 'request_id': request['request_id'], "
        "'status': 'ok', 'result': {'secret': os.environ.get('PARENT_SECRET')}}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PARENT_SECRET", "must-not-leak")

    response = ProcessBridge(
        (sys.executable, str(script)), cwd=tmp_path, artifact_root=tmp_path / "artifacts", env={}
    ).request(request(artifact_dir), timeout=1.0)

    assert response.result == {"secret": None}


def test_serve_emits_one_json_line_and_routes_logs_to_stderr(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "run"
    artifact_dir.mkdir(parents=True)
    script = tmp_path / "serve_child.py"
    bridge_module = Path(__file__).parents[2] / "integrations" / "common" / "jsonl_bridge.py"
    script.write_text(
        "import runpy\n"
        f"namespace = runpy.run_path({str(bridge_module)!r})\n"
        "def handler(request):\n"
        "    print('handler diagnostics')\n"
        "    return {'operation': request.operation}\n"
        "namespace['serve'](handler)\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        input=json.dumps(
            {
                "protocol_version": 1,
                "request_id": "req-1",
                "operation": "act",
                "artifact_dir": str(artifact_dir),
                "payload": {},
            }
        )
        + "\n",
        text=True,
        capture_output=True,
        check=False,
        env={"PYTHONPATH": str(Path(__file__).parents[2] / "src")},
    )

    assert completed.returncode == 0
    assert completed.stderr == "handler diagnostics\n"
    assert completed.stdout.count("\n") == 1
    assert json.loads(completed.stdout)["result"] == {"operation": "act"}


def test_serve_returns_structured_error_without_handler_stdout_contamination(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts" / "run"
    artifact_dir.mkdir(parents=True)
    script = tmp_path / "serve_error_child.py"
    bridge_module = Path(__file__).parents[2] / "integrations" / "common" / "jsonl_bridge.py"
    script.write_text(
        "import runpy\n"
        f"namespace = runpy.run_path({str(bridge_module)!r})\n"
        "def handler(request):\n"
        "    print('before failure')\n"
        "    raise RuntimeError('private detail')\n"
        "namespace['serve'](handler)\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        input=json.dumps(
            {
                "protocol_version": 1,
                "request_id": "req-err",
                "operation": "act",
                "artifact_dir": str(artifact_dir),
                "payload": {},
            }
        )
        + "\n",
        text=True,
        capture_output=True,
        check=False,
        env={"PYTHONPATH": str(Path(__file__).parents[2] / "src")},
    )

    response = json.loads(completed.stdout)
    assert completed.returncode == 0
    assert completed.stdout.count("\n") == 1
    assert response == {
        "protocol_version": 1,
        "request_id": "req-err",
        "status": "error",
        "error": {"type": "RuntimeError"},
    }
    assert completed.stderr == "before failure\nbridge handler failed: RuntimeError\n"

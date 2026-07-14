"""Backend-side helpers for the EvoWorld version-one JSONL protocol."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

from evoworld.adapters.process import (
    PROTOCOL_VERSION,
    BridgeArtifactError,
    BridgeRequest,
    BridgeValidationError,
    _is_json_value,
)
from evoworld.contracts import JsonValue

Handler = Callable[[BridgeRequest], JsonValue]


def _decode_request(raw: str) -> BridgeRequest:
    if raw.count("\n") != 1 or not raw.endswith("\n"):
        raise BridgeValidationError(
            "stdin must contain exactly one newline-terminated JSONL request"
        )
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise BridgeValidationError("request is not valid JSON") from error
    if not isinstance(decoded, dict):
        raise BridgeValidationError("request must be a JSON object")
    payload = decoded.get("payload")
    if not isinstance(payload, dict) or not _is_json_value(payload):
        raise BridgeValidationError("payload must be a JSON object")
    artifact_dir = decoded.get("artifact_dir")
    if not isinstance(artifact_dir, str):
        raise BridgeArtifactError("artifact_dir must be a path string")
    request_id = decoded.get("request_id")
    operation = decoded.get("operation")
    protocol_version = decoded.get("protocol_version")
    if (
        not isinstance(request_id, str)
        or not isinstance(operation, str)
        or not isinstance(protocol_version, int)
    ):
        raise BridgeValidationError("request fields have invalid types")
    return BridgeRequest(
        request_id=request_id,
        operation=operation,
        artifact_dir=Path(artifact_dir),
        payload=payload,
        protocol_version=protocol_version,
    )


def _response(request_id: str, status: str, body: JsonValue, key: str) -> str:
    rendered: Mapping[str, JsonValue] = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "status": status,
        key: body,
    }
    return json.dumps(rendered, ensure_ascii=False, allow_nan=False) + "\n"


def serve(handler: Handler) -> None:
    """Read one request and emit exactly one response; backend logs use stderr."""
    raw = sys.stdin.read()
    protocol_stdout = sys.stdout
    request_id = ""
    try:
        request = _decode_request(raw)
        request_id = request.request_id
        with redirect_stdout(sys.stderr):
            result = handler(request)
        if not _is_json_value(result):
            raise BridgeValidationError("handler result must contain only JSON values")
        protocol_stdout.write(_response(request_id, "ok", result, "result"))
    except (BridgeValidationError, BridgeArtifactError) as error:
        print(f"bridge request rejected: {type(error).__name__}", file=sys.stderr)
        protocol_stdout.write(
            _response(request_id, "error", {"type": type(error).__name__}, "error")
        )
    except Exception as error:  # pragma: no cover - backend failures are integration-specific.
        print(f"bridge handler failed: {type(error).__name__}", file=sys.stderr)
        protocol_stdout.write(
            _response(request_id, "error", {"type": type(error).__name__}, "error")
        )

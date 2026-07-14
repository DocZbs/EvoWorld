"""Strict JSONL transport for isolated backend processes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from secrets import token_hex
import stat
import subprocess
from typing import Final, TypeAlias

from evoworld.contracts import JsonValue

PROTOCOL_VERSION: Final = 1
JsonMapping: TypeAlias = Mapping[str, JsonValue]


class BridgeError(RuntimeError):
    """Base error for a process boundary violation."""


class BridgeValidationError(BridgeError):
    """A caller supplied an invalid protocol value."""


class BridgeArtifactError(BridgeError):
    """The artifact location cannot safely retain protocol logs."""


class BridgeArtifactMissingError(BridgeArtifactError):
    """The requested artifact directory does not exist."""


class BridgeArtifactContainmentError(BridgeArtifactError):
    """The requested artifact directory escapes the configured root."""


class BridgeTimeoutError(BridgeError):
    """The child did not return within the caller's timeout."""


class BridgeLaunchError(BridgeError):
    """The operating system could not start the configured child process."""


class BridgeProcessError(BridgeError):
    """The child process exited unsuccessfully."""


class BridgeStdoutError(BridgeError):
    """Child stdout was not exactly one protocol JSONL response."""


class BridgeMalformedResponseError(BridgeError):
    """The JSONL response did not conform to protocol version one."""


class BridgeCorrelationError(BridgeError):
    """The response request identifier or protocol version was unexpected."""


class BridgeRemoteError(BridgeError):
    """The backend returned an explicit error response."""


@dataclass(frozen=True, slots=True)
class BridgeRequest:
    """A single backend operation and its contained output location."""

    request_id: str
    operation: str
    artifact_dir: Path
    payload: JsonMapping
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise BridgeValidationError("request_id must not be blank")
        if not self.operation.strip():
            raise BridgeValidationError("operation must not be blank")
        if self.protocol_version != PROTOCOL_VERSION:
            raise BridgeValidationError("unsupported request protocol version")
        if not self.artifact_dir.is_absolute():
            raise BridgeArtifactError("artifact directory must be absolute")
        if not _is_json_value(dict(self.payload)):
            raise BridgeValidationError("payload must contain only JSON values")


@dataclass(frozen=True, slots=True)
class BridgeResponse:
    """A validated response to exactly one correlated request."""

    request_id: str
    status: str
    result: JsonValue | None = None
    error: JsonValue | None = None
    protocol_version: int = PROTOCOL_VERSION


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


def _as_json_mapping(value: object, field_name: str) -> JsonMapping:
    if not isinstance(value, dict) or not _is_json_value(value):
        raise BridgeMalformedResponseError(f"{field_name} must be a JSON object")
    return value


def request_to_json(request: BridgeRequest) -> dict[str, JsonValue]:
    """Return the complete canonical input bound to a process request."""
    return {
        "protocol_version": request.protocol_version,
        "request_id": request.request_id,
        "operation": request.operation,
        "artifact_dir": str(request.artifact_dir),
        "payload": dict(request.payload),
    }


def encode_request(request: BridgeRequest) -> str:
    """Render one newline-terminated JSON protocol request."""
    return json.dumps(request_to_json(request), ensure_ascii=False, allow_nan=False) + "\n"


def process_log_dir(request: BridgeRequest) -> Path:
    """Map a request identity to a digest-only log directory."""
    identity = json.dumps(
        {"operation": request.operation, "request_id": request.request_id},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return request.artifact_dir / ".process" / sha256(identity.encode("utf-8")).hexdigest()


def _decode_response(stdout: str, expected_id: str) -> BridgeResponse:
    if stdout.count("\n") != 1 or not stdout.endswith("\n"):
        raise BridgeStdoutError("stdout must contain exactly one newline-terminated JSONL response")
    try:
        decoded = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise BridgeMalformedResponseError("stdout response is not valid JSON") from error
    response = _as_json_mapping(decoded, "response")
    version = response.get("protocol_version")
    if version != PROTOCOL_VERSION:
        raise BridgeCorrelationError("response protocol version does not match")
    request_id = response.get("request_id")
    if not isinstance(request_id, str) or request_id != expected_id:
        raise BridgeCorrelationError("response request_id does not match")
    status = response.get("status")
    if not isinstance(status, str) or status not in {"ok", "error"}:
        raise BridgeMalformedResponseError("response status must be 'ok' or 'error'")
    if status == "ok":
        if (
            "error" in response
            or "result" not in response
            or not _is_json_value(response["result"])
        ):
            raise BridgeMalformedResponseError("ok response must contain one JSON result")
        return BridgeResponse(request_id=request_id, status=status, result=response["result"])
    if "result" in response or "error" not in response or not _is_json_value(response["error"]):
        raise BridgeMalformedResponseError("error response must contain one JSON error")
    return BridgeResponse(request_id=request_id, status=status, error=response["error"])


def _safe_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _atomic_write(path: Path, content: str) -> None:
    if path.is_symlink():
        raise BridgeArtifactContainmentError("process log file must not be a symlink")
    if path.exists() and not path.is_file():
        raise BridgeArtifactError("process log path must be a regular file")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{token_hex(8)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as error:
        raise BridgeArtifactError("process log could not be written atomically") from error
    finally:
        if temporary.exists():
            temporary.unlink()


class ProcessBridge:
    """Execute an argv-only child process with a deliberately minimal environment."""

    def __init__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        artifact_root: Path | None = None,
        env: Mapping[str, str] | None = None,
        preflight: Callable[[], None] | None = None,
    ) -> None:
        if not argv or any(not item for item in argv):
            raise BridgeValidationError("argv must contain non-empty entries")
        if not cwd.is_absolute():
            raise BridgeValidationError("cwd must be absolute")
        self.argv = tuple(argv)
        self.cwd = cwd
        self.artifact_root = artifact_root if artifact_root is not None else cwd
        if not self.artifact_root.is_absolute():
            raise BridgeValidationError("artifact_root must be absolute")
        self.env = dict(env) if env is not None else {}
        self.preflight = preflight

    def _artifact_dir(self, request: BridgeRequest) -> Path:
        if self.artifact_root.is_symlink() or request.artifact_dir.is_symlink():
            raise BridgeArtifactContainmentError("artifact paths must not be symlinks")
        try:
            root = self.artifact_root.resolve(strict=True)
        except FileNotFoundError as error:
            raise BridgeArtifactMissingError("configured artifact root is missing") from error
        try:
            artifact_dir = request.artifact_dir.resolve(strict=True)
        except FileNotFoundError as error:
            raise BridgeArtifactMissingError("artifact directory is missing") from error
        if not artifact_dir.is_dir():
            raise BridgeArtifactMissingError("artifact directory is missing")
        try:
            artifact_dir.relative_to(root)
        except ValueError as error:
            raise BridgeArtifactContainmentError(
                "artifact directory is not contained by artifact root"
            ) from error
        return artifact_dir

    @staticmethod
    def _safe_log_directory(path: Path, artifact_dir: Path) -> Path:
        if path.is_symlink():
            raise BridgeArtifactContainmentError("process log directory must not be a symlink")
        try:
            path.mkdir(mode=0o700, exist_ok=True)
            metadata = path.lstat()
            resolved = path.resolve(strict=True)
            resolved.relative_to(artifact_dir)
        except (OSError, ValueError) as error:
            raise BridgeArtifactContainmentError(
                "process log directory must be a contained real directory"
            ) from error
        if not stat.S_ISDIR(metadata.st_mode):
            raise BridgeArtifactError("process log path must be a directory")
        return resolved

    @classmethod
    def _prepare_log_dir(cls, request: BridgeRequest) -> Path:
        artifact_dir = request.artifact_dir.resolve(strict=True)
        base = cls._safe_log_directory(request.artifact_dir / ".process", artifact_dir)
        return cls._safe_log_directory(process_log_dir(request), base)

    @staticmethod
    def _persist(
        request: BridgeRequest,
        stdout: str,
        stderr: str,
        error: BridgeError | None,
        returncode: int | None,
    ) -> None:
        log_dir = ProcessBridge._prepare_log_dir(request)
        _atomic_write(log_dir / "process.stdout.log", stdout)
        _atomic_write(log_dir / "process.stderr.log", stderr)
        metadata: dict[str, JsonValue] = {
            "error": None,
            "outcome": "ok",
            "returncode": returncode,
        }
        if error is not None:
            metadata = {
                "error": {"type": type(error).__name__, "message": str(error)},
                "outcome": "error",
                "returncode": returncode,
            }
        _atomic_write(
            log_dir / "process.error.json",
            json.dumps(metadata, ensure_ascii=False, allow_nan=False) + "\n",
        )

    def request(self, request: BridgeRequest, timeout: float) -> BridgeResponse:
        """Submit one request, retain complete logs, and validate its response."""
        if timeout <= 0:
            raise BridgeValidationError("timeout must be positive")
        self._artifact_dir(request)
        if self.preflight is not None:
            self.preflight()
        self._prepare_log_dir(request)
        stdout = ""
        stderr = ""
        try:
            completed = subprocess.run(
                self.argv,
                input=encode_request(request),
                text=True,
                capture_output=True,
                cwd=self.cwd,
                timeout=timeout,
                check=False,
                env=self.env,
                shell=False,
            )
            stdout, stderr = completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as error:
            stdout, stderr = _safe_text(error.stdout), _safe_text(error.stderr)
            timeout_error = BridgeTimeoutError("backend process timed out")
            self._persist(request, stdout, stderr, timeout_error, None)
            raise timeout_error from error
        except OSError as error:
            launch_error = BridgeLaunchError("backend process failed to start")
            self._persist(request, stdout, stderr, launch_error, None)
            raise launch_error from error
        if completed.returncode != 0:
            bridge_error = BridgeProcessError(
                f"backend process exited with status {completed.returncode}"
            )
            self._persist(request, stdout, stderr, bridge_error, completed.returncode)
            raise bridge_error
        try:
            response = _decode_response(stdout, request.request_id)
            if response.status == "error":
                raise BridgeRemoteError("backend returned an error response")
        except BridgeError as error:
            self._persist(request, stdout, stderr, error, completed.returncode)
            raise
        self._persist(request, stdout, stderr, None, completed.returncode)
        return response

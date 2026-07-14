"""Crash-safe receipts for process-backed adapter operations."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
from secrets import token_hex
import stat
import threading
from typing import Iterator, TypeVar

from evoworld.contracts import JsonObject
from evoworld.ports import OperationConflictError, OperationInDoubtError

T = TypeVar("T")
_SCHEMA_VERSION = 1


class OperationReceiptError(RuntimeError):
    """A durable operation record is unsafe, corrupt, or semantically invalid."""


def _canonical_bytes(value: JsonObject, field_name: str) -> bytes:
    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise OperationReceiptError(f"{field_name} must contain finite JSON values") from error
    try:
        decoded = json.loads(rendered)
    except json.JSONDecodeError as error:  # pragma: no cover - json.dumps produced it
        raise OperationReceiptError(f"{field_name} is not canonical JSON") from error
    if not isinstance(decoded, dict):
        raise OperationReceiptError(f"{field_name} must be a JSON object")
    return rendered.encode("utf-8")


def _digest(value: JsonObject, field_name: str) -> str:
    return sha256(_canonical_bytes(value, field_name)).hexdigest()


class OperationLedger:
    """Bind one operation identity to canonical input and one durable result."""

    def __init__(self, artifact_root: Path) -> None:
        if not artifact_root.is_absolute():
            raise OperationReceiptError("artifact root must be absolute")
        if artifact_root.is_symlink():
            raise OperationReceiptError("artifact root must not be a symlink")
        try:
            self._root = artifact_root.resolve(strict=True)
        except FileNotFoundError as error:
            raise OperationReceiptError("artifact root is missing") from error
        if not self._root.is_dir():
            raise OperationReceiptError("artifact root must be a directory")
        operations = self._safe_directory(self._root / ".operations", create=True)
        self._entries = self._safe_directory(operations / "entries", create=True)
        self._locks = self._safe_directory(operations / "locks", create=True)

    def entry_dir(self, kind: str, operation_id: str) -> Path:
        """Return the contained digest-only directory for one operation key."""
        self._validate_identity(kind, operation_id)
        identity: JsonObject = {"kind": kind, "operation_id": operation_id}
        return self._entries / _digest(identity, "operation identity")

    def perform(
        self,
        kind: str,
        operation_id: str,
        inputs: JsonObject,
        execute: Callable[[], T],
        encode: Callable[[T], JsonObject],
        decode: Callable[[JsonObject], T],
        verify: Callable[[T], None] | None = None,
    ) -> T:
        """Return a verified receipt or execute exactly one newly claimed effect."""
        self._validate_identity(kind, operation_id)
        input_digest = _digest(inputs, "operation inputs")
        entry = self.entry_dir(kind, operation_id)
        key = entry.name
        with self._lock(key):
            entry = self._safe_directory(entry, create=not entry.exists())
            intent_path = entry / "intent.json"
            receipt_path = entry / "receipt.json"
            expected_intent: JsonObject = {
                "schema_version": _SCHEMA_VERSION,
                "kind": kind,
                "operation_id": operation_id,
                "input_digest": input_digest,
                "inputs": inputs,
            }
            if intent_path.exists() or intent_path.is_symlink():
                persisted_intent = self._read_json(intent_path, "operation intent")
                if persisted_intent != expected_intent:
                    raise OperationConflictError(
                        "operation_id is already bound to different canonical input"
                    )
                if not (receipt_path.exists() or receipt_path.is_symlink()):
                    raise OperationInDoubtError(
                        "operation was claimed but no completion receipt exists"
                    )
                return self._decode_receipt(
                    receipt_path,
                    kind,
                    operation_id,
                    input_digest,
                    decode,
                    verify,
                )
            if receipt_path.exists() or receipt_path.is_symlink():
                raise OperationReceiptError("operation receipt exists without its intent")
            self._atomic_json(intent_path, expected_intent)
            result = execute()
            if verify is not None:
                verify(result)
            encoded = encode(result)
            result_digest = _digest(encoded, "operation result")
            receipt: JsonObject = {
                "schema_version": _SCHEMA_VERSION,
                "kind": kind,
                "operation_id": operation_id,
                "input_digest": input_digest,
                "result_digest": result_digest,
                "result": encoded,
            }
            self._atomic_json(receipt_path, receipt)
            return result

    def _decode_receipt(
        self,
        path: Path,
        kind: str,
        operation_id: str,
        input_digest: str,
        decode: Callable[[JsonObject], T],
        verify: Callable[[T], None] | None,
    ) -> T:
        receipt = self._read_json(path, "operation receipt")
        expected_fields = {
            "schema_version",
            "kind",
            "operation_id",
            "input_digest",
            "result_digest",
            "result",
        }
        if set(receipt) != expected_fields:
            raise OperationReceiptError("operation receipt fields do not match schema 1")
        if (
            receipt.get("schema_version") != _SCHEMA_VERSION
            or receipt.get("kind") != kind
            or receipt.get("operation_id") != operation_id
            or receipt.get("input_digest") != input_digest
        ):
            raise OperationReceiptError("operation receipt identity does not match intent")
        result = receipt.get("result")
        if not isinstance(result, dict):
            raise OperationReceiptError("operation receipt result must be an object")
        typed_result: JsonObject = result
        result_digest = receipt.get("result_digest")
        if not isinstance(result_digest, str) or result_digest != _digest(
            typed_result, "operation result"
        ):
            raise OperationReceiptError("operation receipt result digest does not match")
        try:
            decoded = decode(typed_result)
        except (TypeError, ValueError) as error:
            raise OperationReceiptError("operation receipt result cannot be decoded") from error
        if verify is not None:
            verify(decoded)
        return decoded

    @staticmethod
    def _validate_identity(kind: str, operation_id: str) -> None:
        if not kind.strip():
            raise OperationReceiptError("operation kind must not be blank")
        if not operation_id.strip():
            raise OperationReceiptError("operation_id must not be blank")

    def _safe_directory(self, path: Path, *, create: bool) -> Path:
        if path.is_symlink():
            raise OperationReceiptError("operation directory must not be a symlink")
        if create:
            try:
                path.mkdir(mode=0o700)
            except FileExistsError:
                pass
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as error:
            raise OperationReceiptError("operation directory is missing") from error
        try:
            resolved.relative_to(self._root)
        except ValueError as error:
            raise OperationReceiptError("operation directory escapes artifact root") from error
        if not resolved.is_dir():
            raise OperationReceiptError("operation path must be a directory")
        return resolved

    @contextmanager
    def _lock(self, key: str) -> Iterator[None]:
        path = self._locks / f"{key}.lock"
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as error:
            raise OperationReceiptError("operation lock cannot be opened safely") from error
        try:
            with os.fdopen(descriptor, "a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                yield
        finally:
            # fdopen owns the descriptor once it succeeds.
            pass

    def _read_json(self, path: Path, field_name: str) -> JsonObject:
        try:
            metadata = path.lstat()
        except FileNotFoundError as error:
            raise OperationReceiptError(f"{field_name} is missing") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise OperationReceiptError(f"{field_name} must not be a symlink")
        if not stat.S_ISREG(metadata.st_mode):
            raise OperationReceiptError(f"{field_name} must be a regular file")
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(self._root)
            decoded = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise OperationReceiptError(f"{field_name} is not valid contained JSON") from error
        if not isinstance(decoded, dict):
            raise OperationReceiptError(f"{field_name} must be a JSON object")
        typed: JsonObject = decoded
        _canonical_bytes(typed, field_name)
        return typed

    def _atomic_json(self, path: Path, value: JsonObject) -> None:
        if path.exists() or path.is_symlink():
            raise OperationReceiptError("operation record already exists")
        content = _canonical_bytes(value, "operation record") + b"\n"
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.{token_hex(8)}.tmp"
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(temporary, flags, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
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
            raise OperationReceiptError(
                "operation record could not be written atomically"
            ) from error
        finally:
            if temporary.exists():
                temporary.unlink()

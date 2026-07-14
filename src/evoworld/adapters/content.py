"""Immutable byte manifests for files consumed or produced by real adapters."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from typing import Iterable, cast

from evoworld.contracts import JsonObject, JsonValue
from evoworld.ports import OperationConflictError


class ContentManifestError(ValueError):
    """An external file tree is unsafe, unreadable, or changed from its manifest."""


_HASH_CHUNK_BYTES = 1024 * 1024


def _canonical(value: JsonObject) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _real_root(root: Path, field_name: str) -> Path:
    if not root.is_absolute() or root.is_symlink():
        raise ContentManifestError(f"{field_name} root must be an absolute real directory")
    try:
        resolved = root.resolve(strict=True)
    except OSError as error:
        raise ContentManifestError(f"{field_name} root is missing") from error
    if not resolved.is_dir():
        raise ContentManifestError(f"{field_name} root must be a directory")
    return resolved


def _contained(path: Path, root: Path, field_name: str) -> tuple[Path, Path]:
    if not path.is_absolute():
        raise ContentManifestError(f"{field_name} must be absolute")
    base = _real_root(root, field_name)
    try:
        relative = path.relative_to(base)
    except ValueError as error:
        raise ContentManifestError(f"{field_name} must be contained by its root") from error
    if not relative.parts or any(component in {".", ".."} for component in relative.parts):
        raise ContentManifestError(f"{field_name} must be a contained child path")
    current = base
    for component in relative.parts:
        current = current / component
        try:
            metadata = current.lstat()
        except OSError as error:
            raise ContentManifestError(f"{field_name} is missing") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ContentManifestError(f"{field_name} must not contain symlinks")
    return current.resolve(strict=True), base


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns


def _hash_file(path: Path, field_name: str) -> tuple[int, str]:
    descriptor = -1
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ContentManifestError(f"{field_name} must be a regular file")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _identity(metadata) != _identity(opened):
            raise ContentManifestError(f"{field_name} changed before it was hashed")
        digest = sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while chunk := stream.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
        after = os.fstat(descriptor)
        current = path.lstat()
    except OSError as error:
        raise ContentManifestError(f"{field_name} cannot be read safely") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if _identity(opened) != _identity(after) or _identity(opened) != _identity(current):
        raise ContentManifestError(f"{field_name} changed while it was hashed")
    return opened.st_size, digest.hexdigest()


def _file_record(path: Path, root: Path, field_name: str) -> JsonObject:
    resolved, base = _contained(path, root, field_name)
    size, digest = _hash_file(resolved, field_name)
    return {
        "kind": "file",
        "root": str(base),
        "path": str(resolved),
        "size": size,
        "sha256": digest,
    }


def file_manifest(path: Path, root: Path, field_name: str) -> JsonObject:
    """Hash one contained regular file and bind its canonical path and size."""
    return _file_record(path, root, field_name)


def directory_manifest(path: Path, root: Path, field_name: str) -> JsonObject:
    """Recursively inventory one contained real directory without following links."""
    resolved, base = _contained(path, root, field_name)
    if not resolved.is_dir():
        raise ContentManifestError(f"{field_name} must be a directory")
    records: list[JsonValue] = []
    try:
        children = sorted(
            resolved.rglob("*"), key=lambda item: item.relative_to(resolved).as_posix()
        )
    except OSError as error:
        raise ContentManifestError(f"{field_name} cannot be inventoried") from error
    for child in children:
        relative = child.relative_to(resolved).as_posix()
        try:
            metadata = child.lstat()
        except OSError as error:
            raise ContentManifestError(f"{field_name} entry is unreadable") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ContentManifestError(f"{field_name} must not contain symlinks")
        if stat.S_ISDIR(metadata.st_mode):
            records.append({"kind": "directory", "path": relative})
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ContentManifestError(f"{field_name} contains a non-regular entry")
        size, digest = _hash_file(child, field_name)
        records.append(
            {
                "kind": "file",
                "path": relative,
                "size": size,
                "sha256": digest,
            }
        )
    inventory: JsonObject = {
        "kind": "directory",
        "root": str(base),
        "path": str(resolved),
        "entries": records,
    }
    inventory["sha256"] = sha256(_canonical(inventory)).hexdigest()
    return inventory


def combined_manifest(entries: Iterable[JsonObject]) -> JsonObject:
    """Bind an ordered collection of file/directory manifests to one digest."""
    items = list(entries)
    manifest: JsonObject = {"schema_version": 1, "entries": cast(list[JsonValue], items)}
    manifest["sha256"] = sha256(_canonical(manifest)).hexdigest()
    return manifest


def verify_manifest(manifest: JsonObject, field_name: str) -> None:
    """Re-hash every path and raise a typed conflict on any byte or path drift."""
    if set(manifest) != {"schema_version", "entries", "sha256"}:
        raise ContentManifestError(f"{field_name} fields are invalid")
    entries = manifest.get("entries")
    if manifest.get("schema_version") != 1 or not isinstance(entries, list):
        raise ContentManifestError(f"{field_name} schema is invalid")
    rebuilt: list[JsonObject] = []
    try:
        for index, raw in enumerate(entries):
            if not isinstance(raw, dict):
                raise ContentManifestError(f"{field_name}[{index}] must be an object")
            kind = raw.get("kind")
            path = raw.get("path")
            root = raw.get("root")
            if not isinstance(path, str) or not isinstance(root, str):
                raise ContentManifestError(f"{field_name}[{index}] paths are invalid")
            if kind == "file":
                rebuilt.append(file_manifest(Path(path), Path(root), f"{field_name}[{index}]"))
            elif kind == "directory":
                rebuilt.append(directory_manifest(Path(path), Path(root), f"{field_name}[{index}]"))
            else:
                raise ContentManifestError(f"{field_name}[{index}] kind is invalid")
    except ContentManifestError as error:
        raise OperationConflictError(f"{field_name} is no longer readable or safe") from error
    current = combined_manifest(rebuilt)
    if current != manifest:
        raise OperationConflictError(f"{field_name} content changed")

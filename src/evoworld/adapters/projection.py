"""Library-owned, declarative action projection for Ctrl-World."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import re
import stat
from typing import cast

from evoworld.codec import input_digest
from evoworld.contracts import ActionChunk, JsonObject, SceneRecord, TaskSpec
from evoworld.ports import ExpertPolicy

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPERT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024
AFFINE_IMPLEMENTATION = "affine_v1"
CTRL_WORLD_ACTION_SCHEMA = "cartesian_pose_7d"
CTRL_WORLD_ACTION_WIDTH = 7
PROJECTOR_SOURCE_PATH = Path(__file__).resolve()
_LOADED_SOURCE_DIGEST = sha256(PROJECTOR_SOURCE_PATH.read_bytes()).hexdigest()


class ProjectionError(ValueError):
    """A calibration snapshot or projected action violated the fixed contract."""


@dataclass(frozen=True, slots=True)
class ProjectorIdentity:
    """Immutable identity for one fixed projector and its private calibration."""

    expert_id: str
    identity: str
    implementation: str
    source_schema: str
    source_width: int
    target_schema: str
    config_digest: str
    source_digest: str
    weights_digest: str
    content_digest: str

    def __post_init__(self) -> None:
        if _EXPERT_ID.fullmatch(self.expert_id) is None:
            raise ProjectionError("projector expert_id is invalid")
        if not self.identity.strip():
            raise ProjectionError("projector identity must not be blank")
        if self.implementation != AFFINE_IMPLEMENTATION:
            raise ProjectionError("projector implementation must be affine_v1")
        if not self.source_schema.strip():
            raise ProjectionError("projector source_schema must not be blank")
        if isinstance(self.source_width, bool) or not isinstance(self.source_width, int):
            raise ProjectionError("projector source_width must be an integer")
        if self.source_width < 1:
            raise ProjectionError("projector source_width must be positive")
        if self.target_schema != CTRL_WORLD_ACTION_SCHEMA:
            raise ProjectionError("projector target_schema must be cartesian_pose_7d")
        for field in ("config_digest", "source_digest", "weights_digest", "content_digest"):
            if _SHA256.fullmatch(getattr(self, field)) is None:
                raise ProjectionError(f"projector {field} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    path: Path
    digest: str
    data: bytes


@dataclass(frozen=True, slots=True)
class AffineActionProjector:
    """Fixed 7D affine map loaded from immutable JSON snapshots."""

    identity_record: ProjectorIdentity
    matrix: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]

    @property
    def identity(self) -> str:
        return self.identity_record.identity

    @property
    def content_digest(self) -> str:
        return self.identity_record.content_digest

    @property
    def source_schema(self) -> str:
        return self.identity_record.source_schema

    @property
    def target_schema(self) -> str:
        return self.identity_record.target_schema

    @property
    def source_width(self) -> int:
        return self.identity_record.source_width

    def project(
        self,
        task: TaskSpec,
        observation: SceneRecord,
        actions: ActionChunk,
        *,
        operation_id: str,
    ) -> ActionChunk:
        del task, observation
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise ProjectionError("project operation_id must not be blank")
        if actions.expert_id != self.identity_record.expert_id:
            raise ProjectionError("raw action expert does not match projector")
        if actions.schema_name != self.source_schema:
            raise ProjectionError("raw action schema does not match projector")
        if any(len(row) != self.source_width for row in actions.actions):
            raise ProjectionError("raw action width does not match projector source_width")
        projected = tuple(
            tuple(
                sum(coefficient * value for coefficient, value in zip(weights, row)) + offset
                for weights, offset in zip(self.matrix, self.bias)
            )
            for row in actions.actions
        )
        try:
            result = ActionChunk(
                actions.expert_id,
                projected,
                actions.timestamps,
                CTRL_WORLD_ACTION_SCHEMA,
            )
        except ValueError as error:
            raise ProjectionError("affine projection produced an invalid action") from error
        return result


def projector_content_digest(
    *,
    expert_id: str,
    identity: str,
    implementation: str,
    source_schema: str,
    source_width: int,
    target_schema: str,
    config_digest: str,
    source_digest: str,
    weights_digest: str,
) -> str:
    """Bind the fixed implementation and every private calibration input."""
    payload: JsonObject = {
        "schema_version": 2,
        "expert_id": expert_id,
        "identity": identity,
        "implementation": implementation,
        "source_schema": source_schema,
        "source_width": source_width,
        "target_schema": target_schema,
        "config_digest": config_digest,
        "source_digest": source_digest,
        "weights_digest": weights_digest,
    }
    return input_digest(payload)


def load_affine_projector(
    identity: ProjectorIdentity,
    *,
    config_path: Path,
    weights_path: Path,
    asset_root: Path,
    source_path: Path,
    source_root: Path,
) -> AffineActionProjector:
    """Load verified config and weights from the exact bytes hashed through one fd."""
    config = _snapshot(config_path, asset_root, "projector config")
    weights = _snapshot(weights_path, asset_root, "projector weights")
    source = _snapshot(source_path, source_root, "projector source")
    if source.digest != _LOADED_SOURCE_DIGEST:
        raise ProjectionError("projector source changed after its implementation was loaded")
    for field, actual in (
        ("config digest", config.digest),
        ("weights digest", weights.digest),
        ("source digest", source.digest),
    ):
        expected = getattr(identity, field.replace(" ", "_"))
        if actual != expected:
            raise ProjectionError(f"projector {field} does not match snapshot")
    expected_content = projector_content_digest(
        expert_id=identity.expert_id,
        identity=identity.identity,
        implementation=identity.implementation,
        source_schema=identity.source_schema,
        source_width=identity.source_width,
        target_schema=identity.target_schema,
        config_digest=identity.config_digest,
        source_digest=identity.source_digest,
        weights_digest=identity.weights_digest,
    )
    if identity.content_digest != expected_content:
        raise ProjectionError("projector content_digest does not match identity")
    config_json = _strict_object(config.data, "config")
    if set(config_json) != {
        "schema_version",
        "implementation",
        "source_width",
        "target_width",
    }:
        raise ProjectionError("projector config fields are invalid")
    if type(config_json.get("schema_version")) is not int or config_json.get("schema_version") != 1:
        raise ProjectionError("projector config schema_version must be 1")
    if config_json.get("implementation") != AFFINE_IMPLEMENTATION:
        raise ProjectionError("projector config implementation must be affine_v1")
    if (
        type(config_json.get("source_width")) is not int
        or config_json.get("source_width") != identity.source_width
    ):
        raise ProjectionError("projector config source_width does not match identity")
    if (
        type(config_json.get("target_width")) is not int
        or config_json.get("target_width") != CTRL_WORLD_ACTION_WIDTH
    ):
        raise ProjectionError("projector config target_width must be 7")
    weights_json = _strict_object(weights.data, "weights")
    if set(weights_json) != {"schema_version", "matrix", "bias"}:
        raise ProjectionError("projector weights fields are invalid")
    if (
        type(weights_json.get("schema_version")) is not int
        or weights_json.get("schema_version") != 1
    ):
        raise ProjectionError("projector weights schema_version must be 1")
    matrix = _matrix(weights_json.get("matrix"), identity.source_width)
    bias = _vector(weights_json.get("bias"), CTRL_WORLD_ACTION_WIDTH, "bias")
    return AffineActionProjector(identity, matrix, bias)


class ProjectedExpertPolicy:
    """Trusted wrapper retaining an official raw adapter and fixed projector."""

    def __init__(
        self,
        raw_policy: ExpertPolicy,
        projector: AffineActionProjector,
        identity: ProjectorIdentity,
    ) -> None:
        if not isinstance(raw_policy, ExpertPolicy):
            raise ProjectionError("raw_policy does not implement ExpertPolicy")
        if not isinstance(projector, AffineActionProjector):
            raise ProjectionError("projector must be the library-owned affine implementation")
        if projector.identity_record is not identity:
            raise ProjectionError("projector identity object does not match composition")
        self._raw_policy = raw_policy
        self._projector = projector
        self._identity = identity

    @property
    def raw_policy(self) -> ExpertPolicy:
        return self._raw_policy

    @property
    def projector(self) -> AffineActionProjector:
        return self._projector

    @property
    def identity(self) -> ProjectorIdentity:
        return self._identity

    def act(
        self,
        task: TaskSpec,
        observation: SceneRecord,
        *,
        operation_id: str,
    ) -> ActionChunk:
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise ProjectionError("operation_id must not be blank")
        raw = self._raw_policy.act(task, observation, operation_id=f"{operation_id}:raw")
        self._validate_raw(raw)
        result = self._projector.project(
            task,
            observation,
            raw,
            operation_id=f"{operation_id}:project",
        )
        if (
            result.expert_id != self._identity.expert_id
            or result.schema_name != CTRL_WORLD_ACTION_SCHEMA
            or result.timestamps != raw.timestamps
            or len(result.actions) != len(raw.actions)
            or any(len(row) != CTRL_WORLD_ACTION_WIDTH for row in result.actions)
        ):
            raise ProjectionError("projected action provenance is invalid")
        return result

    def _validate_raw(self, value: object) -> None:
        if not isinstance(value, ActionChunk):
            raise ProjectionError("raw policy must return an ActionChunk")
        if value.expert_id != self._identity.expert_id:
            raise ProjectionError("raw action expert does not match projector")
        if value.schema_name != self._identity.source_schema:
            raise ProjectionError("raw action schema does not match projector")
        if any(len(row) != self._identity.source_width for row in value.actions):
            raise ProjectionError("raw action width does not match projector source_width")
        if value.timestamps[0] < 0 or not all(
            current < following
            for current, following in zip(value.timestamps, value.timestamps[1:])
        ):
            raise ProjectionError("raw action timestamps must start nonnegative and increase")


def _snapshot(path: Path, root: Path, field_name: str) -> _FileSnapshot:
    base, relative = _contained_relative_path(path, root, field_name)
    descriptors: list[int] = []
    directory_snapshots: list[tuple[int, str, os.stat_result]] = []
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    except AttributeError as error:
        raise ProjectionError(
            "safe no-follow snapshots are unsupported on this platform"
        ) from error
    try:
        root_before = base.lstat()
        root_descriptor = os.open(base, directory_flags)
        descriptors.append(root_descriptor)
        root_opened = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_opened.st_mode) or _file_identity(root_before) != _file_identity(
            root_opened
        ):
            raise ProjectionError(f"{field_name} root changed before snapshot")
        parent_descriptor = root_descriptor
        for component in relative.parts[:-1]:
            child_descriptor = os.open(component, directory_flags, dir_fd=parent_descriptor)
            descriptors.append(child_descriptor)
            child_opened = os.fstat(child_descriptor)
            if not stat.S_ISDIR(child_opened.st_mode):
                raise ProjectionError(f"{field_name} parent must be a real directory")
            directory_snapshots.append((parent_descriptor, component, child_opened))
            parent_descriptor = child_descriptor
        leaf = relative.parts[-1]
        descriptor = os.open(leaf, file_flags, dir_fd=parent_descriptor)
        descriptors.append(descriptor)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ProjectionError(f"{field_name} must be a regular file")
        chunks: list[bytes] = []
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            size += len(chunk)
            if size > _MAX_SNAPSHOT_BYTES:
                raise ProjectionError(f"{field_name} exceeds snapshot size limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = os.stat(leaf, dir_fd=parent_descriptor, follow_symlinks=False)
        for ancestor_descriptor, component, expected in directory_snapshots:
            ancestor_current = os.stat(component, dir_fd=ancestor_descriptor, follow_symlinks=False)
            if not stat.S_ISDIR(ancestor_current.st_mode) or _file_identity(
                expected
            ) != _file_identity(ancestor_current):
                raise ProjectionError(f"{field_name} parent changed during snapshot")
        root_current = base.lstat()
    except OSError as error:
        raise ProjectionError(f"{field_name} cannot be snapshotted safely") from error
    finally:
        for opened_descriptor in reversed(descriptors):
            os.close(opened_descriptor)
    if _file_identity(opened) != _file_identity(after) or _file_identity(opened) != _file_identity(
        current
    ):
        raise ProjectionError(f"{field_name} changed during snapshot")
    if _file_identity(root_opened) != _file_identity(root_current):
        raise ProjectionError(f"{field_name} root changed during snapshot")
    data = b"".join(chunks)
    return _FileSnapshot(path, sha256(data).hexdigest(), data)


def _contained_relative_path(path: Path, root: Path, field_name: str) -> tuple[Path, Path]:
    if not root.is_absolute() or root.is_symlink():
        raise ProjectionError(f"{field_name} root must be an absolute real directory")
    try:
        base = root.resolve(strict=True)
    except OSError as error:
        raise ProjectionError(f"{field_name} root is missing") from error
    if not base.is_dir() or not path.is_absolute():
        raise ProjectionError(f"{field_name} must be an absolute contained file")
    try:
        relative = path.relative_to(base)
    except ValueError as error:
        raise ProjectionError(f"{field_name} must be contained by its root") from error
    if not relative.parts or any(component in {".", ".."} for component in relative.parts):
        raise ProjectionError(f"{field_name} must be a child file")
    return base, relative


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _strict_object(data: bytes, label: str) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite constant {value}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key}")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            data.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ProjectionError(f"projector {label} JSON is invalid") from error
    if not isinstance(decoded, dict):
        raise ProjectionError(f"projector {label} JSON must be an object")
    return cast(dict[str, object], decoded)


def _vector(value: object, width: int, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != width:
        raise ProjectionError(f"projector {field_name} must contain {width} values")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ProjectionError(f"projector {field_name} must be numeric")
        try:
            number = float(item)
        except (OverflowError, ValueError) as error:
            raise ProjectionError(f"projector {field_name} must be finite") from error
        if not isfinite(number):
            raise ProjectionError(f"projector {field_name} must be finite")
        result.append(number)
    return tuple(result)


def _matrix(value: object, source_width: int) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list) or len(value) != CTRL_WORLD_ACTION_WIDTH:
        raise ProjectionError("projector matrix must contain 7 rows")
    return tuple(_vector(row, source_width, f"matrix[{index}]") for index, row in enumerate(value))

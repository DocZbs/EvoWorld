"""Concurrent, idempotent local storage for resumable EvoWorld runs."""

from __future__ import annotations

import fcntl
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterator, Mapping

from evoworld.codec import (
    StateDecodeError,
    UnsupportedStateSchemaError,
    canonical_json,
    decode_json_object,
    diagnosis_from_json,
    diagnosis_to_json,
    round_state_from_json,
    round_state_to_json,
    score_from_json,
    score_to_json,
    task_to_json,
    trajectory_to_json,
)
from evoworld.contracts import (
    ExperienceRecord,
    JsonObject,
    JsonValue,
    RoundStage,
    RoundState,
    SceneRecord,
    TaskSpec,
    Trajectory,
)


class StorageConflictError(RuntimeError):
    """A durable value conflicts with an existing claim or revision."""


class ArtifactConflict(StorageConflictError):
    """An artifact path already contains different canonical bytes."""


ArtifactConflictError = ArtifactConflict


class StateConflictError(StorageConflictError):
    """A state write is stale, non-monotonic, or belongs to another round."""


@dataclass(frozen=True, slots=True)
class _StoredRound:
    state: RoundState
    claim_sequence: int


_CLAIM_SCHEMA_VERSION = 2
_CLAIM_COUNTER_SCHEMA_VERSION = 1


def atomic_write_json(path: Path, value: JsonValue) -> None:
    """Atomically replace a JSON file with a canonical, complete document."""
    payload = (canonical_json(value, indent=2) + "\n").encode("utf-8")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(path.with_name(f".{path.name}.lock")):
        _atomic_replace(path, payload)


class JsonRunStore:
    """Store a claimed run with CAS state and immutable canonical artifacts."""

    def __init__(self, run_dir: Path, config_digest: str) -> None:
        self.run_dir = run_dir.resolve()
        self._config_digest = config_digest

    @classmethod
    def create(
        cls,
        run_root: Path,
        run_id: str,
        resolved_config: JsonObject,
        config_digest: str,
    ) -> JsonRunStore:
        _validate_path_component(run_id, "run_id")
        root = run_root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        run_dir = _resolve_contained(root, root / run_id, "run directory")
        run_dir.mkdir(parents=True, exist_ok=True)
        run_dir = _resolve_contained(root, run_dir, "run directory")
        manifest: JsonObject = {
            "config_digest": config_digest,
            "resolved_config": resolved_config,
        }
        path = _resolve_contained(root, run_dir / "manifest.json", "run manifest path")
        payload = (canonical_json(manifest, indent=2) + "\n").encode("utf-8")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(run_dir)
        return cls(run_dir, config_digest)

    @classmethod
    def open(cls, run_dir: Path, config_digest: str) -> JsonRunStore:
        resolved = run_dir.resolve(strict=True)
        manifest = decode_json_object(
            (resolved / "manifest.json").read_text(encoding="utf-8"),
            source=str(resolved / "manifest.json"),
        )
        if _text(manifest.get("config_digest"), "manifest config_digest") != config_digest:
            raise ValueError(
                "run manifest config digest does not match the requested configuration"
            )
        return cls(resolved, config_digest)

    def claim_round(self, state: RoundState) -> None:
        """Atomically claim a fresh round with its complete initial snapshot."""
        if state.stage is not RoundStage.STARTED or state.revision != 0:
            raise ValueError("round claim requires an initial STARTED state")
        round_id = state.round_id
        _validate_path_component(round_id, "round_id")
        with _file_lock(self._lock_path("state")):
            rounds = self.run_dir / "rounds"
            rounds.mkdir(parents=True, exist_ok=True)
            rounds = _resolve_contained(self.run_dir, rounds, "rounds directory")
            round_dir = rounds / round_id
            round_dir.mkdir(parents=True, exist_ok=True)
            round_dir = _resolve_contained(self.run_dir, round_dir, "round directory")
            claim = round_dir / ".claim"
            if claim.exists():
                raise FileExistsError(f"round {round_id} already exists")
            claim_sequence = self._next_claim_sequence_unlocked()
            payload: JsonObject = {
                "claim_schema_version": _CLAIM_SCHEMA_VERSION,
                "claim_sequence": claim_sequence,
                "state": round_state_to_json(state),
            }
            try:
                _atomic_create_exclusive(
                    claim,
                    (canonical_json(payload, indent=2) + "\n").encode("utf-8"),
                )
            except FileExistsError as error:
                raise FileExistsError(f"round {round_id} already exists") from error

    def _next_claim_sequence_unlocked(self) -> int:
        path = _resolve_contained(
            self.run_dir,
            self.run_dir / ".claim-sequence.json",
            "claim sequence path",
        )
        current = 0
        if path.exists():
            value = decode_json_object(path.read_text(encoding="utf-8"), source=str(path))
            schema = value.get("schema_version")
            if schema != _CLAIM_COUNTER_SCHEMA_VERSION:
                raise UnsupportedStateSchemaError(
                    f"claim counter schema {schema!r} is unsupported; start a new run"
                )
            if set(value) != {"schema_version", "claim_sequence"}:
                raise StateDecodeError(f"invalid claim sequence counter at {path}")
            sequence = value.get("claim_sequence")
            if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
                raise StateDecodeError(f"invalid claim sequence counter at {path}")
            current = sequence
        following = current + 1
        payload: JsonObject = {
            "schema_version": _CLAIM_COUNTER_SCHEMA_VERSION,
            "claim_sequence": following,
        }
        _atomic_replace(
            path,
            (canonical_json(payload, indent=2) + "\n").encode("utf-8"),
        )
        return following

    def write_state(self, state: RoundState) -> None:
        self.save_state(state)

    def save_state(self, state: RoundState) -> None:
        """CAS a monotonic round snapshot under a cross-process file lock."""
        path = _resolve_contained(self.run_dir, self.run_dir / "state.json", "state path")
        payload = (canonical_json(round_state_to_json(state), indent=2) + "\n").encode("utf-8")
        with _file_lock(self._lock_path("state")):
            current = self._read_state_unlocked()
            if current == state:
                round_path = _resolve_contained(
                    self.run_dir,
                    self.run_dir / "rounds" / state.round_id / "state.json",
                    "round state path",
                )
                round_path.parent.mkdir(parents=True, exist_ok=True)
                if not round_path.exists():
                    _atomic_replace(round_path, payload)
                if not path.exists():
                    _atomic_replace(path, payload)
                return
            if current is None:
                if state.revision != 0:
                    raise StateConflictError("initial state revision must be zero")
            else:
                if current.round_id != state.round_id:
                    if current.stage is not RoundStage.COMPLETE or state.revision != 0:
                        raise StateConflictError(
                            "cannot replace an incomplete or stale round state"
                        )
                elif state.revision != current.revision + 1:
                    raise StateConflictError("state revision compare-and-swap failed")
                elif tuple(RoundStage).index(state.stage) < tuple(RoundStage).index(current.stage):
                    raise StateConflictError("state stage must advance monotonically")
            round_path = _resolve_contained(
                self.run_dir,
                self.run_dir / "rounds" / state.round_id / "state.json",
                "round state path",
            )
            round_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_replace(round_path, payload)
            _atomic_replace(path, payload)

    def read_state(self) -> RoundState | None:
        with _file_lock(self._lock_path("state")):
            return self._read_state_unlocked()

    def _read_state_unlocked(self) -> RoundState | None:
        current = self._read_global_state_unlocked()
        stored = self._latest_round_unlocked()
        if stored is None:
            return current
        local = stored.state
        if current is None or current.round_id != local.round_id:
            return local
        return _higher_revision_state(current, local)

    def _read_global_state_unlocked(self) -> RoundState | None:
        path = _resolve_contained(self.run_dir, self.run_dir / "state.json", "state path")
        if not path.exists():
            return None
        value = decode_json_object(path.read_text(encoding="utf-8"), source=str(path))
        return round_state_from_json(value)

    def load_state(self, round_id: str) -> RoundState | None:
        _validate_path_component(round_id, "round_id")
        with _file_lock(self._lock_path("state")):
            stored = self._read_round_unlocked(round_id)
            return stored.state if stored is not None else None

    def _latest_round_unlocked(self) -> _StoredRound | None:
        rounds = self.run_dir / "rounds"
        if not rounds.exists():
            return None
        stored_rounds = [
            stored
            for path in rounds.iterdir()
            if path.is_dir()
            for stored in [self._read_round_unlocked(path.name)]
            if stored is not None
        ]
        sequenced = [stored.claim_sequence for stored in stored_rounds if stored.claim_sequence > 0]
        if len(sequenced) != len(set(sequenced)):
            raise StateConflictError("duplicate claim sequence across round directories")
        unsequenced = [stored for stored in stored_rounds if stored.claim_sequence == 0]
        if len(unsequenced) > 1:
            raise StateConflictError("multiple unsequenced round states cannot be ordered")
        return max(
            stored_rounds,
            key=lambda stored: stored.claim_sequence,
            default=None,
        )

    def _read_round_unlocked(self, round_id: str) -> _StoredRound | None:
        round_dir = _resolve_contained(
            self.run_dir, self.run_dir / "rounds" / round_id, "round directory"
        )
        claim_path = round_dir / ".claim"
        claim_state: RoundState | None = None
        claim_sequence = 0
        if claim_path.exists():
            claim_state, claim_sequence = self._decode_claim(claim_path, round_id)
        state_path = round_dir / "state.json"
        if state_path.exists():
            value = decode_json_object(
                state_path.read_text(encoding="utf-8"), source=str(state_path)
            )
            state = round_state_from_json(value)
            _validate_embedded_round_id(state, round_id, state_path)
            if claim_state is not None:
                state = _higher_revision_state(state, claim_state)
            return _StoredRound(state, claim_sequence)
        if claim_state is None:
            return None
        return _StoredRound(claim_state, claim_sequence)

    def _decode_claim(self, claim_path: Path, round_id: str) -> tuple[RoundState, int]:
        claim = decode_json_object(claim_path.read_text(encoding="utf-8"), source=str(claim_path))
        schema = claim.get("claim_schema_version")
        if schema != _CLAIM_SCHEMA_VERSION:
            raise UnsupportedStateSchemaError(
                f"round claim schema {schema!r} is unsupported; start a new run"
            )
        if set(claim) != {"claim_schema_version", "claim_sequence", "state"}:
            raise StateDecodeError(f"invalid round claim wrapper at {claim_path}")
        claim_sequence = claim.get("claim_sequence")
        if (
            not isinstance(claim_sequence, int)
            or isinstance(claim_sequence, bool)
            or claim_sequence <= 0
        ):
            raise StateDecodeError(f"invalid round claim sequence at {claim_path}")
        state_value = claim.get("state")
        if not isinstance(state_value, dict):
            raise StateDecodeError(f"invalid round claim state at {claim_path}")
        state = round_state_from_json(state_value)
        _validate_embedded_round_id(state, round_id, claim_path)
        return state, claim_sequence

    def read_state_required(self) -> RoundState:
        state = self.read_state()
        if state is None:
            raise FileNotFoundError("run has no persisted state")
        return state

    def write_artifact(self, round_id: str, name: str, metadata: JsonObject) -> Path:
        _validate_path_component(round_id, "round_id")
        _validate_path_component(name, "artifact name")
        return self._write_idempotent_json(
            self.run_dir / "rounds" / round_id / f"{name}.json",
            metadata,
            field_name="artifact path",
        )

    def append_task(self, round_id: str, operation_id: str, task: TaskSpec) -> None:
        """Append one deterministic task journal entry without duplication."""
        _validate_path_component(round_id, "round_id")
        if not operation_id.strip():
            raise ValueError("operation_id must not be blank")
        record: JsonObject = {
            "round_id": round_id,
            "operation_id": operation_id,
            "task": task_to_json(task),
        }
        path = _resolve_contained(self.run_dir, self.run_dir / "tasks.jsonl", "task journal")
        with _file_lock(self._lock_path("tasks")):
            for existing in _read_jsonl(path):
                if existing.get("round_id") != round_id:
                    continue
                if existing == record:
                    return
                raise StorageConflictError(f"different task already exists for {round_id}")
            _append_jsonl(path, record)

    def write_verified(self, trajectory: Trajectory, provenance: JsonObject) -> Path:
        """Write a complete verified trajectory and provenance idempotently."""
        _validate_path_component(trajectory.trajectory_id, "trajectory_id")
        root = self.run_dir / "verified" / trajectory.trajectory_id
        self._write_idempotent_json(
            root / "trajectory.json",
            trajectory_to_json(trajectory),
            field_name="verified trajectory path",
        )
        self._write_idempotent_json(
            root / "provenance.json",
            provenance,
            field_name="verified provenance path",
        )
        return _resolve_contained(self.run_dir, root, "verified directory")

    def _write_idempotent_json(self, path: Path, value: JsonObject, *, field_name: str) -> Path:
        path = _create_contained_parent(self.run_dir, path, field_name)
        payload = (canonical_json(value, indent=2) + "\n").encode("utf-8")
        relative = str(path.relative_to(self.run_dir))
        lock = self._lock_path(sha256(relative.encode()).hexdigest())
        with _file_lock(lock):
            if path.exists():
                if path.read_bytes() == payload:
                    return path
                raise ArtifactConflict(f"different artifact already exists at {path}")
            _atomic_replace(path, payload)
        return path

    def _lock_path(self, name: str) -> Path:
        return _resolve_contained(
            self.run_dir, self.run_dir / ".locks" / f"{name}.lock", "lock path"
        )


class JsonlExperienceMemory:
    """Cross-process safe append-only experience records."""

    def __init__(self, path: Path, *, containment_root: Path | None = None) -> None:
        self._containment_root = containment_root.resolve() if containment_root else None
        self.path = path.absolute()

    def append(self, record: ExperienceRecord, *, operation_id: str = "") -> None:
        operation_id = operation_id or f"direct:append:{record.record_id}"
        if not operation_id.strip():
            raise ValueError("operation_id must not be blank")
        serialized = _experience_to_json(record)
        serialized["operation_id"] = operation_id
        path = self._safe_path()
        with _file_lock(self._safe_lock(path)):
            for value in _read_jsonl(path):
                if value.get("operation_id") == operation_id:
                    if value == serialized:
                        return
                    raise ValueError(
                        f"operation_id {operation_id} already appended a different record"
                    )
                existing = _experience_from_json(value)
                if existing.record_id != record.record_id:
                    continue
                if _experience_to_json(existing) == _experience_to_json(record):
                    return
                raise ValueError(
                    f"different record already exists for deterministic ID {record.record_id}"
                )
            _append_jsonl(path, serialized)

    def lookup(
        self, task: TaskSpec, scene: SceneRecord, *, operation_id: str = ""
    ) -> tuple[ExperienceRecord, ...]:
        operation_id = operation_id or f"direct:lookup:{task.task_id}:{scene.scene_id}"
        if not operation_id.strip():
            raise ValueError("operation_id must not be blank")
        path = self._safe_path()
        with _file_lock(self._safe_lock(path)):
            return tuple(
                record
                for record in self._records_unlocked(path)
                if record.task_id == task.task_id and record.scene_id == scene.scene_id
            )

    def _records_unlocked(self, path: Path) -> tuple[ExperienceRecord, ...]:
        records: list[ExperienceRecord] = []
        for line_number, value in enumerate(_read_jsonl(path), start=1):
            try:
                records.append(_experience_from_json(value))
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid experience record at {path}:{line_number}") from error
        return tuple(records)

    def _safe_path(self) -> Path:
        if self._containment_root is None:
            return self.path.resolve()
        return _resolve_contained(self._containment_root, self.path, "experience memory path")

    def _safe_lock(self, path: Path) -> Path:
        lock = path.with_name(f".{path.name}.lock")
        if self._containment_root is None:
            return lock.resolve()
        return _resolve_contained(self._containment_root, lock, "experience memory lock")


def _experience_to_json(record: ExperienceRecord) -> JsonObject:
    return {
        "record_id": record.record_id,
        "task_id": record.task_id,
        "scene_id": record.scene_id,
        "expert_id": record.expert_id,
        "score": score_to_json(record.score),
        "diagnosis": diagnosis_to_json(record.diagnosis),
        "artifact_refs": [str(path.resolve()) for path in record.artifact_refs],
    }


def _experience_from_json(value: JsonObject) -> ExperienceRecord:
    expected = {
        "record_id",
        "task_id",
        "scene_id",
        "expert_id",
        "score",
        "diagnosis",
        "artifact_refs",
        "operation_id",
    }
    if set(value) != expected:
        raise ValueError("experience record contains unknown or missing fields")
    return ExperienceRecord(
        record_id=_text(value.get("record_id"), "record_id"),
        task_id=_text(value.get("task_id"), "task_id"),
        scene_id=_text(value.get("scene_id"), "scene_id"),
        expert_id=_text(value.get("expert_id"), "expert_id"),
        score=score_from_json(_object(value.get("score"), "score")),
        diagnosis=diagnosis_from_json(_object(value.get("diagnosis"), "diagnosis")),
        artifact_refs=tuple(
            Path(_text(item, "artifact reference")).resolve()
            for item in _list(value.get("artifact_refs"), "artifact_refs")
        ),
    )


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path = path.resolve()
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _atomic_replace(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_create_exclusive(path: Path, payload: bytes) -> None:
    """Publish complete bytes atomically without replacing an existing path."""
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    published = False
    try:
        _write_descriptor_and_sync(descriptor, payload)
        os.link(temporary, path)
        _fsync_directory(path.parent)
        published = True
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
    if published:
        _fsync_directory(path.parent)


def _write_descriptor_and_sync(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        try:
            written = os.write(descriptor, remaining)
        except InterruptedError:
            continue
        if written <= 0:
            raise OSError("claim temporary write made no progress")
        remaining = remaining[written:]
    os.fsync(descriptor)


def _append_jsonl(path: Path, value: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write((canonical_json(value) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    if not path.exists():
        return ()
    values: list[JsonObject] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            values.append(decode_json_object(line, source=f"{path}:{number}"))
        except ValueError as error:
            raise ValueError(f"invalid JSONL record at {path}:{number}") from error
    return tuple(values)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_path_component(value: str, field_name: str) -> None:
    candidate = Path(value)
    if (
        not value.strip()
        or value in {".", ".."}
        or candidate.is_absolute()
        or "/" in value
        or "\\" in value
        or len(candidate.parts) != 1
        or candidate.name != value
    ):
        raise ValueError(f"{field_name} must be a safe single path component")


def _validate_embedded_round_id(state: RoundState, round_id: str, source: Path) -> None:
    if state.round_id != round_id:
        raise StateDecodeError(
            f"round state identity does not match directory {round_id!r} at {source}"
        )


def _higher_revision_state(first: RoundState, second: RoundState) -> RoundState:
    if first.round_id != second.round_id:
        raise StateConflictError("cannot reconcile state from different rounds")
    if first.revision == second.revision:
        if first != second:
            raise StateConflictError("same round has equal revision with different durable state")
        return first
    return first if first.revision > second.revision else second


def _resolve_contained(root: Path, candidate: Path, field_name: str) -> Path:
    resolved_root = root.resolve(strict=True)
    resolved = candidate.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{field_name} must remain within {resolved_root}")
    return resolved


def _create_contained_parent(root: Path, path: Path, field_name: str) -> Path:
    """Reject an escaping existing ancestor before creating missing directories."""
    ancestor = path.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    _resolve_contained(root, ancestor, field_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return _resolve_contained(root, path, field_name)


def _object(value: object, field_name: str) -> JsonObject:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be a JSON object")
    return {key: _json_value(item, field_name) for key, item in value.items()}


def _list(value: object, field_name: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return [_json_value(item, field_name) for item in value]


def _json_value(value: object, field_name: str) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_json_value(item, field_name) for item in value]
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return {key: _json_value(item, field_name) for key, item in value.items()}
    raise ValueError(f"{field_name} contains a non-JSON value")


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value

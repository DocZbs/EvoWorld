"""Immutable schemas shared by EvoWorld fine-tuning launchers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from evoworld.contracts import DiagnosisLabel


class TrainingSchemaError(ValueError):
    """Raised when a training manifest or plan is not safe to launch."""


_ALLOWED_ENVIRONMENT = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "HF_HOME",
        "NCCL_DEBUG",
        "OMP_NUM_THREADS",
        "PYTHONPATH",
        "TOKENIZERS_PARALLELISM",
        "WANDB_MODE",
        "WANDB_PROJECT",
    }
)
_SECRET_MARKERS = ("credential", "password", "secret", "token")


def _required(value: str, name: str) -> None:
    if not value.strip():
        raise TrainingSchemaError(f"{name} must not be blank")


def _sha256(value: str, name: str, length: int) -> None:
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise TrainingSchemaError(f"{name} must be a lowercase {length}-character SHA-256 value")


def validate_relative_posix_path(path: Path, name: str) -> None:
    """Require a non-empty relative path using POSIX separators without traversal."""
    value = str(path)
    if path.is_absolute() or "\\" in value or ".." in path.parts:
        raise TrainingSchemaError(f"{name} must be a manifest-relative POSIX path without '..'")
    if path == Path(".") or not path.parts:
        raise TrainingSchemaError(f"{name} must not be empty")


def _safe_identifier(value: str, name: str) -> None:
    _required(value, name)
    if (
        value in {".", ".."}
        or "/" in value
        or "\\" in value
        or any(char.isspace() for char in value)
    ):
        raise TrainingSchemaError(f"{name} must be a safe single path segment")


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """One provenance-complete sample addressed by manifest-relative media paths."""

    schema_version: int
    sample_id: str
    target: str
    task_id: str
    instruction: str
    skill_family: str
    scene_id: str
    trajectory_id: str
    dataset_root: Path
    observation_paths: tuple[Path, ...]
    action_path: Path | None
    action_schema: str | None
    action_dimension: int | None
    action_horizon: int | None
    source_type: str
    source_checkpoint: str
    diagnosis: DiagnosisLabel
    split: str
    content_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation_paths", tuple(self.observation_paths))
        if self.schema_version < 1:
            raise TrainingSchemaError("schema_version must be positive")
        for name in (
            "sample_id",
            "target",
            "task_id",
            "instruction",
            "skill_family",
            "scene_id",
            "trajectory_id",
            "source_type",
            "source_checkpoint",
            "split",
        ):
            _required(getattr(self, name), name)
        validate_relative_posix_path(self.dataset_root, "dataset_root")
        if not self.observation_paths:
            raise TrainingSchemaError("observation_paths must not be empty")
        for path in self.observation_paths:
            validate_relative_posix_path(path, "observation_paths")
        if self.action_path is not None:
            validate_relative_posix_path(self.action_path, "action_path")
        action_values = (
            self.action_path,
            self.action_schema,
            self.action_dimension,
            self.action_horizon,
        )
        if any(value is None for value in action_values) and any(
            value is not None for value in action_values
        ):
            raise TrainingSchemaError("action provenance must be fully specified or fully absent")
        if self.action_dimension is not None and self.action_dimension < 1:
            raise TrainingSchemaError("action_dimension must be positive")
        if self.action_horizon is not None and self.action_horizon < 1:
            raise TrainingSchemaError("action_horizon must be positive")
        if not isinstance(self.diagnosis, DiagnosisLabel):
            raise TrainingSchemaError("diagnosis must be a DiagnosisLabel")
        _sha256(self.content_sha256, "content_sha256", 64)

    def resolve_media_paths(self, manifest_root: Path) -> tuple[Path, ...]:
        """Resolve validated media only when a caller is ready to launch a backend."""
        if not manifest_root.is_absolute():
            raise TrainingSchemaError("manifest_root must be absolute at launch")
        media = (*self.observation_paths, *((self.action_path,) if self.action_path else ()))
        return tuple(manifest_root / self.dataset_root / path for path in media)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation without resolving media paths."""
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "target": self.target,
            "task_id": self.task_id,
            "instruction": self.instruction,
            "skill_family": self.skill_family,
            "scene_id": self.scene_id,
            "trajectory_id": self.trajectory_id,
            "dataset_root": self.dataset_root.as_posix(),
            "observation_paths": [path.as_posix() for path in self.observation_paths],
            "action_path": self.action_path.as_posix() if self.action_path else None,
            "action_schema": self.action_schema,
            "action_dimension": self.action_dimension,
            "action_horizon": self.action_horizon,
            "source_type": self.source_type,
            "source_checkpoint": self.source_checkpoint,
            "diagnosis": self.diagnosis.value,
            "split": self.split,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True, slots=True)
class TrainingManifest:
    """A deterministically hashed collection of immutable training samples."""

    schema_version: int
    manifest_id: str
    samples: tuple[TrainingSample, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", tuple(self.samples))
        if self.schema_version < 1:
            raise TrainingSchemaError("schema_version must be positive")
        _required(self.manifest_id, "manifest_id")
        if not self.samples:
            raise TrainingSchemaError("samples must not be empty")
        if len({sample.sample_id for sample in self.samples}) != len(self.samples):
            raise TrainingSchemaError("sample_id values must be unique")

    @property
    def sha256(self) -> str:
        """Return the stable hash used to reject stale training plans."""
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Return the manifest's canonical JSON payload."""
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "samples": [sample.to_dict() for sample in self.samples],
        }


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    """The complete, non-secret command plan for one fine-tuning job."""

    schema_version: int
    job_id: str
    target: str
    manifest_sha256: str
    upstream_revision: str
    cwd: Path
    argv: tuple[str, ...]
    environment: Mapping[str, str]
    expected_outputs: tuple[Path, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(self, "expected_outputs", tuple(self.expected_outputs))
        if self.schema_version < 1:
            raise TrainingSchemaError("schema_version must be positive")
        _safe_identifier(self.job_id, "job_id")
        _required(self.target, "target")
        _sha256(self.manifest_sha256, "manifest_sha256", 64)
        _sha256(self.upstream_revision, "upstream_revision", 40)
        if not self.cwd.is_absolute():
            raise TrainingSchemaError("cwd must be absolute")
        if not self.argv or any(not argument for argument in self.argv):
            raise TrainingSchemaError("argv must contain non-empty argument values")
        environment = dict(self.environment)
        for key, value in environment.items():
            if key not in _ALLOWED_ENVIRONMENT or any(
                marker in key.lower() for marker in _SECRET_MARKERS
            ):
                raise TrainingSchemaError(f"environment key {key!r} is not allowlisted")
            if not isinstance(value, str):
                raise TrainingSchemaError("environment values must be strings")
        for path in self.expected_outputs:
            validate_relative_posix_path(path, "expected_outputs")
        object.__setattr__(self, "environment", MappingProxyType(environment))

    def to_dict(self) -> dict[str, object]:
        """Return the exact command plan recorded in a job artifact."""
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "target": self.target,
            "manifest_sha256": self.manifest_sha256,
            "upstream_revision": self.upstream_revision,
            "cwd": str(self.cwd),
            "argv": list(self.argv),
            "environment": dict(self.environment),
            "expected_outputs": [path.as_posix() for path in self.expected_outputs],
        }

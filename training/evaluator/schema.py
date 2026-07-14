"""Strict annotations and canonical ms-swift export for the EvoWorld evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, cast

from PIL import Image, UnidentifiedImageError

from evoworld.contracts import DiagnosisLabel
from training.common.schema import TrainingSchemaError, validate_relative_posix_path


# This is the public baseline prompt; the exact experiment prompt is unpublished.
EVALUATOR_SYSTEM_INSTRUCTION = (
    "You are the EvoWorld trajectory reliability evaluator. Assess only the provided "
    "task, grounded scene context, and ordered trajectory frames. Return exactly one JSON "
    "object with keys score, label, and rationale. score is in [0,1]; label is one of "
    "TSM, PF, WMF, or NONE."
)
_SOURCE_TYPES = frozenset({"real", "manually_verified_synthetic"})
_SPLITS = frozenset({"train", "eval"})
_FRAME_FORMATS = frozenset({"JPEG", "PNG", "PPM", "WEBP"})
_FRAME_MODES = frozenset({"L", "RGB", "RGBA"})
_EXAMPLE_FIELDS = frozenset(
    {
        "schema_version",
        "example_id",
        "task_id",
        "instruction",
        "scene_id",
        "trajectory_id",
        "media_paths",
        "media_sha256",
        "source_type",
        "source_checkpoint",
        "provenance",
        "content_sha256",
        "score",
        "label",
        "rationale",
        "annotator",
        "revision",
        "split",
    }
)


class EvaluatorSchemaError(ValueError):
    """Raised when evaluator annotations or media are unsafe or ambiguous."""


def _required(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise EvaluatorSchemaError(f"{name} must not be blank")


def _hash(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvaluatorSchemaError(f"{name} must be a lowercase 64-character SHA-256 value")


def _relative_path(path: Path, name: str) -> None:
    try:
        validate_relative_posix_path(path, name)
    except TrainingSchemaError as error:
        raise EvaluatorSchemaError(str(error)) from error


@dataclass(frozen=True, slots=True)
class EvaluatorExample:
    """One provenance-complete, manually labelled evaluator example."""

    schema_version: int
    example_id: str
    task_id: str
    instruction: str
    scene_id: str
    trajectory_id: str
    media_paths: tuple[Path, ...]
    media_sha256: tuple[str, ...]
    source_type: str
    source_checkpoint: str
    provenance: str
    content_sha256: str
    score: float
    label: DiagnosisLabel
    rationale: str
    annotator: str
    revision: int
    split: str

    def __post_init__(self) -> None:
        if isinstance(self.media_paths, (str, bytes)) or isinstance(
            self.media_sha256, (str, bytes)
        ):
            raise EvaluatorSchemaError("media_paths and media_sha256 must be sequences")
        try:
            object.__setattr__(self, "media_paths", tuple(self.media_paths))
            object.__setattr__(self, "media_sha256", tuple(self.media_sha256))
        except TypeError as error:
            raise EvaluatorSchemaError(
                "media_paths and media_sha256 must be iterable sequences"
            ) from error
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise EvaluatorSchemaError("schema_version must be 1")
        for name in (
            "example_id",
            "task_id",
            "instruction",
            "scene_id",
            "trajectory_id",
            "source_checkpoint",
            "provenance",
            "rationale",
            "annotator",
        ):
            _required(getattr(self, name), name)
        if not self.media_paths:
            raise EvaluatorSchemaError("media_paths must not be empty")
        if len(self.media_paths) != len(self.media_sha256):
            raise EvaluatorSchemaError("media_paths and media_sha256 must have equal length")
        if not all(isinstance(path, Path) for path in self.media_paths):
            raise EvaluatorSchemaError("media_paths must contain Path values")
        for path in self.media_paths:
            _relative_path(path, "media_paths")
        if len(set(self.media_paths)) != len(self.media_paths):
            raise EvaluatorSchemaError("duplicate media_paths within one example are not allowed")
        for digest in self.media_sha256:
            _hash(digest, "media_sha256")
        if len(set(self.media_sha256)) != len(self.media_sha256):
            raise EvaluatorSchemaError("media hash leakage within one example is not allowed")
        if not isinstance(self.source_type, str) or self.source_type not in _SOURCE_TYPES:
            raise EvaluatorSchemaError(
                "source_type must be real or manually_verified_synthetic; it is never inferred"
            )
        _hash(self.content_sha256, "content_sha256")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise EvaluatorSchemaError("score must be numeric")
        if not 0.0 <= float(self.score) <= 1.0:
            raise EvaluatorSchemaError("score must be between 0 and 1")
        if not isinstance(self.label, DiagnosisLabel):
            raise EvaluatorSchemaError("label must be TSM, PF, WMF, or NONE")
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 1
        ):
            raise EvaluatorSchemaError("revision must be a positive integer")
        if not isinstance(self.split, str) or self.split not in _SPLITS:
            raise EvaluatorSchemaError("split must be train or eval")


@dataclass(frozen=True, slots=True)
class EvaluatorExport:
    """A canonical ms-swift dataset bound to annotation revisions and a byte hash."""

    path: Path
    validation_path: Path
    dataset_root: Path
    sha256: str
    train_sha256: str
    validation_sha256: str
    dataset_revision: str
    example_count: int

    def __post_init__(self) -> None:
        if not all(isinstance(path, Path) for path in (self.path, self.validation_path)):
            raise EvaluatorSchemaError("train and validation export paths must be Path values")
        if not isinstance(self.dataset_root, Path):
            raise EvaluatorSchemaError("export dataset_root must be a Path value")
        if not self.path.is_absolute() or not self.validation_path.is_absolute():
            raise EvaluatorSchemaError("train and validation export paths must be absolute")
        if self.path == self.validation_path:
            raise EvaluatorSchemaError("train and validation export paths must be distinct")
        if not self.dataset_root.is_absolute() or not self.dataset_root.is_dir():
            raise EvaluatorSchemaError("export dataset_root must be an existing absolute directory")
        _hash(self.sha256, "export sha256")
        _hash(self.train_sha256, "train export sha256")
        _hash(self.validation_sha256, "validation export sha256")
        _hash(self.dataset_revision, "dataset_revision")
        if (
            isinstance(self.example_count, bool)
            or not isinstance(self.example_count, int)
            or self.example_count < 1
        ):
            raise EvaluatorSchemaError("example_count must be positive")


def validate_example(
    example: EvaluatorExample,
    *,
    reliability_threshold: float,
) -> EvaluatorExample:
    """Enforce the reliability threshold's one unambiguous label partition."""
    if isinstance(reliability_threshold, bool) or not isinstance(
        reliability_threshold, (int, float)
    ):
        raise EvaluatorSchemaError("reliability threshold must be numeric")
    if not 0.0 <= float(reliability_threshold) <= 1.0:
        raise EvaluatorSchemaError("reliability threshold must be between 0 and 1")
    reliable = example.score >= reliability_threshold
    if reliable and example.label is not DiagnosisLabel.NONE:
        raise EvaluatorSchemaError("score at or above the reliability threshold requires NONE")
    if not reliable and example.label is DiagnosisLabel.NONE:
        raise EvaluatorSchemaError("score below the reliability threshold requires TSM, PF, or WMF")
    return example


def _duplicate_safe_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluatorSchemaError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise EvaluatorSchemaError(f"{name} must be a string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluatorSchemaError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluatorSchemaError(f"{name} must be numeric")
    return float(value)


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise EvaluatorSchemaError(f"{name} must be a list of strings")
    return tuple(cast(list[str], value))


def _example_from_payload(payload: object, line_number: int) -> EvaluatorExample:
    if not isinstance(payload, dict):
        raise EvaluatorSchemaError(f"line {line_number} must be a JSON object")
    values = cast(dict[str, object], payload)
    missing = _EXAMPLE_FIELDS - set(values)
    unknown = set(values) - _EXAMPLE_FIELDS
    if missing:
        raise EvaluatorSchemaError(f"line {line_number} missing fields: {sorted(missing)}")
    if unknown:
        raise EvaluatorSchemaError(f"line {line_number} has unknown fields: {sorted(unknown)}")
    try:
        label = DiagnosisLabel(_string(values["label"], "label"))
    except ValueError as error:
        raise EvaluatorSchemaError("label must be TSM, PF, WMF, or NONE") from error
    return EvaluatorExample(
        schema_version=_integer(values["schema_version"], "schema_version"),
        example_id=_string(values["example_id"], "example_id"),
        task_id=_string(values["task_id"], "task_id"),
        instruction=_string(values["instruction"], "instruction"),
        scene_id=_string(values["scene_id"], "scene_id"),
        trajectory_id=_string(values["trajectory_id"], "trajectory_id"),
        media_paths=tuple(
            Path(item) for item in _string_tuple(values["media_paths"], "media_paths")
        ),
        media_sha256=_string_tuple(values["media_sha256"], "media_sha256"),
        source_type=_string(values["source_type"], "source_type"),
        source_checkpoint=_string(values["source_checkpoint"], "source_checkpoint"),
        provenance=_string(values["provenance"], "provenance"),
        content_sha256=_string(values["content_sha256"], "content_sha256"),
        score=_number(values["score"], "score"),
        label=label,
        rationale=_string(values["rationale"], "rationale"),
        annotator=_string(values["annotator"], "annotator"),
        revision=_integer(values["revision"], "revision"),
        split=_string(values["split"], "split"),
    )


def _contained_media(dataset_root: Path, relative: Path) -> Path:
    candidate = (dataset_root / relative).resolve()
    if not candidate.is_relative_to(dataset_root):
        raise EvaluatorSchemaError(f"media path escapes dataset root: {relative.as_posix()}")
    return candidate


def _decode_frame(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise EvaluatorSchemaError(f"media is missing or empty: {path.name}")
    try:
        with Image.open(path) as probe:
            if probe.format not in _FRAME_FORMATS:
                raise EvaluatorSchemaError(
                    f"media must be a supported still frame image: {path.name}"
                )
            if getattr(probe, "is_animated", False):
                raise EvaluatorSchemaError(
                    f"media must be one non-animated still frame: {path.name}"
                )
            probe.verify()
        with Image.open(path) as frame:
            frame.load()
            if frame.mode not in _FRAME_MODES or frame.width < 1 or frame.height < 1:
                raise EvaluatorSchemaError(
                    f"still frame has unsupported mode or dimensions: {path.name}"
                )
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as error:
        raise EvaluatorSchemaError(
            f"media is not a fully decodable still frame image: {path.name}"
        ) from error


def _validate_media(example: EvaluatorExample, dataset_root: Path) -> None:
    for relative, expected_hash in zip(example.media_paths, example.media_sha256, strict=True):
        path = _contained_media(dataset_root, relative)
        _decode_frame(path)
        if sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise EvaluatorSchemaError(f"media hash mismatch: {relative.as_posix()}")


def _validate_no_leakage(examples: tuple[EvaluatorExample, ...]) -> None:
    checks: tuple[tuple[str, tuple[object, ...]], ...] = (
        ("example_id", tuple(example.example_id for example in examples)),
        ("trajectory_id", tuple(example.trajectory_id for example in examples)),
        ("content_sha256", tuple(example.content_sha256 for example in examples)),
        (
            "media_sha256",
            tuple(digest for example in examples for digest in example.media_sha256),
        ),
    )
    for name, values in checks:
        if len(values) != len(set(values)):
            raise EvaluatorSchemaError(f"duplicate {name} leakage within or across train/eval")


def validate_evaluator_dataset(
    dataset_path: Path,
    *,
    dataset_root: Path,
    reliability_threshold: float,
) -> tuple[EvaluatorExample, ...]:
    """Load strict JSONL, decode every frame, and reject all leakage identities."""
    if not dataset_root.is_absolute():
        raise EvaluatorSchemaError("dataset_root must be absolute")
    root = dataset_root.resolve()
    if not root.is_dir():
        raise EvaluatorSchemaError("dataset_root must be an existing directory")
    try:
        text = dataset_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise EvaluatorSchemaError("dataset must be valid UTF-8") from error
    except OSError as error:
        raise EvaluatorSchemaError(f"unable to read evaluator dataset {dataset_path}") from error
    examples: list[EvaluatorExample] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line, object_pairs_hook=_duplicate_safe_object)
        except json.JSONDecodeError as error:
            raise EvaluatorSchemaError(f"invalid JSON on line {line_number}") from error
        example = validate_example(
            _example_from_payload(payload, line_number),
            reliability_threshold=reliability_threshold,
        )
        _validate_media(example, root)
        examples.append(example)
    if not examples:
        raise EvaluatorSchemaError("evaluator dataset must not be empty")
    result = tuple(examples)
    _validate_no_leakage(result)
    return result


def _revision_hash(examples: tuple[EvaluatorExample, ...]) -> str:
    values = [
        {
            "content_sha256": example.content_sha256,
            "example_id": example.example_id,
            "revision": example.revision,
        }
        for example in sorted(examples, key=lambda item: item.example_id)
    ]
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _export_row(example: EvaluatorExample) -> dict[str, Any]:
    # Source: https://github.com/modelscope/ms-swift/blob/v4.3.1/docs/source_en/Customization/Custom-dataset.md#multimodal
    grounded = {
        "instruction": example.instruction,
        "response_schema": {
            "label": ["TSM", "PF", "WMF", "NONE"],
            "rationale": "nonblank string",
            "score": "number in [0,1]",
        },
        "scene_id": example.scene_id,
        "task_id": example.task_id,
        "trajectory_id": example.trajectory_id,
    }
    response = {
        "label": example.label.value,
        "rationale": example.rationale,
        "score": example.score,
    }
    return {
        "messages": [
            {"role": "system", "content": EVALUATOR_SYSTEM_INSTRUCTION},
            {
                "role": "user",
                "content": "<video>\n"
                + json.dumps(grounded, sort_keys=True, separators=(",", ":")),
            },
            {
                "role": "assistant",
                "content": json.dumps(response, sort_keys=True, separators=(",", ":")),
            },
        ],
        "videos": [[path.as_posix() for path in example.media_paths]],
        "evoworld_provenance": {
            "annotator": example.annotator,
            "content_sha256": example.content_sha256,
            "example_id": example.example_id,
            "media_sha256": list(example.media_sha256),
            "provenance": example.provenance,
            "revision": example.revision,
            "source_checkpoint": example.source_checkpoint,
            "source_type": example.source_type,
            "split": example.split,
        },
    }


def export_ms_swift(
    examples: tuple[EvaluatorExample, ...],
    *,
    dataset_root: Path,
    export_path: Path,
) -> EvaluatorExport:
    """Write deterministic ms-swift 4.3.1 JSONL without absolute media paths."""
    if not examples:
        raise EvaluatorSchemaError("examples must not be empty")
    if not dataset_root.is_absolute() or not export_path.is_absolute():
        raise EvaluatorSchemaError("dataset_root and export_path must be absolute")
    root = dataset_root.resolve()
    ordered = tuple(sorted(examples, key=lambda item: item.example_id))
    if {example.split for example in ordered} != _SPLITS:
        raise EvaluatorSchemaError("evaluator export requires nonempty train and eval splits")
    _validate_no_leakage(ordered)
    for example in ordered:
        _validate_media(example, root)
    payloads = {
        split: "".join(
            json.dumps(_export_row(example), sort_keys=True, separators=(",", ":")) + "\n"
            for example in ordered
            if example.split == split
        ).encode("utf-8")
        for split in _SPLITS
    }
    path = export_path.resolve()
    validation_path = path.with_name(f"{path.stem}.eval{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    for destination, payload in ((path, payloads["train"]), (validation_path, payloads["eval"])):
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(destination)
    train_hash = sha256(payloads["train"]).hexdigest()
    validation_hash = sha256(payloads["eval"]).hexdigest()
    dataset_revision = _revision_hash(ordered)
    combined = json.dumps(
        {
            "dataset_revision": dataset_revision,
            "train_sha256": train_hash,
            "validation_sha256": validation_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return EvaluatorExport(
        path=path,
        validation_path=validation_path,
        dataset_root=root,
        sha256=sha256(combined).hexdigest(),
        train_sha256=train_hash,
        validation_sha256=validation_hash,
        dataset_revision=dataset_revision,
        example_count=len(ordered),
    )

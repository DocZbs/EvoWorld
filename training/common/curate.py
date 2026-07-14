"""Diagnosis-aware curation and deterministic, leakage-safe data splits."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Mapping

from evoworld.contracts import DiagnosisLabel

from training.common.schema import TrainingSample, TrainingSchemaError


class DataLeakageError(ValueError):
    """Raised with the leaked provenance key type and values."""

    def __init__(self, key_type: str, scope: str, values: Iterable[str]) -> None:
        self.key_type = key_type
        self.scope = scope
        self.values = tuple(sorted(set(values)))
        super().__init__(f"{key_type} leakage {scope}: {', '.join(self.values)}")


RawTrainingRecord = TrainingSample | Mapping[str, object]


def curate_records(
    records: Iterable[RawTrainingRecord], *, include_verified_evaluator: bool = False
) -> tuple[TrainingSample, ...]:
    """Route diagnosis records to their sole authorized training target.

    The supplied ``target`` field is deliberately ignored. PF records must name the
    selected expert explicitly; no target is inferred from a directory or path.
    """
    curated: list[TrainingSample] = []
    for record in records:
        sample, expert_id = _sample_and_expert(record)
        target = _target_for(sample.diagnosis, expert_id, include_verified_evaluator)
        if target is not None:
            curated.append(replace(sample, target=target))
    return tuple(curated)


def group_split(
    samples: Iterable[TrainingSample], *, seed: int, eval_fraction: float
) -> tuple[TrainingSample, ...]:
    """Split trajectory groups deterministically without relying on input order."""
    if not 0.0 <= eval_fraction < 1.0:
        raise TrainingSchemaError("eval_fraction must be in [0, 1)")

    groups = _connected_groups(samples)
    if not groups:
        return ()

    ordered_groups = sorted(
        groups,
        key=lambda group: _group_rank(group, seed),
    )
    eval_groups = _evaluation_group_count(len(ordered_groups), eval_fraction)
    evaluation_samples = {sample for group in ordered_groups[:eval_groups] for sample in group}
    split_samples = (
        replace(sample, split="eval" if sample in evaluation_samples else "train")
        for group in ordered_groups
        for sample in sorted(group, key=lambda item: item.sample_id)
    )
    return tuple(sorted(split_samples, key=lambda item: item.sample_id))


def validate_no_leakage(
    train: Iterable[TrainingSample], evaluation: Iterable[TrainingSample]
) -> None:
    """Reject sample, trajectory, or exact-content reuse across train and eval."""
    train_samples = tuple(train)
    evaluation_samples = tuple(evaluation)
    provenance_keys = (
        ("sample_id", "sample_id"),
        ("trajectory_id", "trajectory_id"),
        ("content_sha256", "content_sha256"),
    )
    for scope, samples in (("within train", train_samples), ("within eval", evaluation_samples)):
        for key_type, attribute in provenance_keys:
            duplicates = _duplicates(samples, attribute)
            if duplicates:
                raise DataLeakageError(key_type, scope, duplicates)
    for key_type, attribute in provenance_keys:
        overlap = {getattr(sample, attribute) for sample in train_samples} & {
            getattr(sample, attribute) for sample in evaluation_samples
        }
        if overlap:
            raise DataLeakageError(key_type, "between train and eval", overlap)


def _target_for(
    diagnosis: DiagnosisLabel, expert_id: str | None, include_verified_evaluator: bool
) -> str | None:
    if diagnosis is DiagnosisLabel.TSM:
        return None
    if diagnosis is DiagnosisLabel.PF:
        if expert_id is None or not expert_id.strip():
            raise TrainingSchemaError("PF records require an explicit expert_id")
        return expert_id
    if diagnosis is DiagnosisLabel.WMF:
        return "ctrl_world"
    if diagnosis is DiagnosisLabel.NONE and include_verified_evaluator:
        return "evaluator"
    return None


def _sample_and_expert(record: RawTrainingRecord) -> tuple[TrainingSample, str | None]:
    if isinstance(record, TrainingSample):
        return record, record.target if record.diagnosis is DiagnosisLabel.PF else None
    if not isinstance(record, Mapping):
        raise TrainingSchemaError("training records must be TrainingSample values or mappings")

    diagnosis_value = _required_string(record, "diagnosis")
    try:
        diagnosis = DiagnosisLabel(diagnosis_value)
    except ValueError as error:
        raise TrainingSchemaError(f"unknown diagnosis: {diagnosis_value}") from error
    action_path_value = record.get("action_path")
    if action_path_value is not None and not isinstance(action_path_value, str):
        raise TrainingSchemaError("action_path must be a string when supplied")
    action_path = Path(action_path_value) if action_path_value is not None else None
    if action_path is None:
        action_schema = action_dimension = action_horizon = None
    else:
        action_schema = _required_string(record, "action_schema")
        action_dimension = _required_int(record, "action_dimension")
        action_horizon = _required_int(record, "action_horizon")
    sample = TrainingSample(
        schema_version=_required_int(record, "schema_version"),
        sample_id=_required_string(record, "sample_id"),
        target="curation-pending",
        task_id=_required_string(record, "task_id"),
        instruction=_required_string(record, "instruction"),
        skill_family=_required_string(record, "skill_family"),
        scene_id=_required_string(record, "scene_id"),
        trajectory_id=_required_string(record, "trajectory_id"),
        dataset_root=Path(_required_string(record, "dataset_root")),
        observation_paths=_paths(record),
        action_path=action_path,
        action_schema=action_schema,
        action_dimension=action_dimension,
        action_horizon=action_horizon,
        source_type=_required_string(record, "source_type"),
        source_checkpoint=_required_string(record, "source_checkpoint"),
        diagnosis=diagnosis,
        split=_required_string(record, "split"),
        content_sha256=_required_string(record, "content_sha256"),
    )
    expert_id = record.get("expert_id")
    if expert_id is not None and not isinstance(expert_id, str):
        raise TrainingSchemaError("expert_id must be a string when supplied")
    return sample, expert_id


def _paths(record: Mapping[str, object]) -> tuple[Path, ...]:
    values = record.get("observation_paths")
    if not isinstance(values, (list, tuple)) or not all(isinstance(value, str) for value in values):
        raise TrainingSchemaError("observation_paths must be a list of strings")
    return tuple(Path(value) for value in values)


def _required_string(record: Mapping[str, object], name: str) -> str:
    value = record.get(name)
    if not isinstance(value, str):
        raise TrainingSchemaError(f"{name} must be a string")
    return value


def _required_int(record: Mapping[str, object], name: str) -> int:
    value = record.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TrainingSchemaError(f"{name} must be an integer")
    return value


def _connected_groups(samples: Iterable[TrainingSample]) -> tuple[tuple[TrainingSample, ...], ...]:
    ordered = sorted(
        samples,
        key=lambda sample: (
            sample.sample_id,
            sample.trajectory_id,
            sample.scene_id,
            sample.content_sha256,
        ),
    )
    parents = list(range(len(ordered)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    trajectory_owner: dict[str, int] = {}
    scene_owner: dict[str, int] = {}
    for index, sample in enumerate(ordered):
        for owner, provenance_id in (
            (trajectory_owner, sample.trajectory_id),
            (scene_owner, sample.scene_id),
        ):
            previous = owner.setdefault(provenance_id, index)
            union(index, previous)

    components: dict[int, list[TrainingSample]] = {}
    for index, sample in enumerate(ordered):
        components.setdefault(find(index), []).append(sample)
    return tuple(tuple(component) for component in components.values())


def _group_rank(group: tuple[TrainingSample, ...], seed: int) -> tuple[str, str, str]:
    trajectories = ",".join(sorted({sample.trajectory_id for sample in group}))
    scenes = ",".join(sorted({sample.scene_id for sample in group}))
    digest = sha256(f"{seed}\0{trajectories}\0{scenes}".encode("utf-8")).hexdigest()
    return digest, trajectories, scenes


def _duplicates(samples: tuple[TrainingSample, ...], attribute: str) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for sample in samples:
        value = getattr(sample, attribute)
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _evaluation_group_count(group_count: int, eval_fraction: float) -> int:
    if eval_fraction == 0.0 or group_count == 1:
        return 0
    requested = round(group_count * eval_fraction)
    return min(group_count - 1, max(1, requested))

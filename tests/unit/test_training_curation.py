import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from evoworld.contracts import DiagnosisLabel
from training.common.curate import (
    DataLeakageError,
    curate_records,
    group_split,
    validate_no_leakage,
)
from training.common.schema import TrainingSample, TrainingSchemaError


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "training_records.jsonl"


def _record(
    diagnosis: DiagnosisLabel,
    *,
    expert_id: str | None = None,
    sample_id: str = "sample-001",
    trajectory_id: str = "trajectory-001",
    scene_id: str = "scene-001",
    content_sha256: str = "a" * 64,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "sample_id": sample_id,
        "target": "untrusted-directory-name",
        "expert_id": expert_id,
        "task_id": "task-001",
        "instruction": "place the mug on the tray",
        "skill_family": "pick_place",
        "scene_id": scene_id,
        "trajectory_id": trajectory_id,
        "dataset_root": "dataset",
        "observation_paths": ["frames/000.png"],
        "action_path": "actions/000.npy",
        "action_schema": "joint_velocity",
        "action_dimension": 7,
        "action_horizon": 10,
        "source_type": "real",
        "source_checkpoint": "checkpoint-a",
        "diagnosis": diagnosis.value,
        "split": "pending",
        "content_sha256": content_sha256,
    }


def _sample(split: str, **changes: object) -> TrainingSample:
    record = _record(DiagnosisLabel.PF, expert_id="pi05")
    record["split"] = split
    record.update(changes)
    return curate_records([record])[0]


def test_diagnoses_route_only_to_their_authorized_training_targets() -> None:
    assert curate_records([_record(DiagnosisLabel.TSM)]) == ()
    assert {
        sample.target for sample in curate_records([_record(DiagnosisLabel.PF, expert_id="pi05")])
    } == {"pi05"}
    assert {sample.target for sample in curate_records([_record(DiagnosisLabel.WMF)])} == {
        "ctrl_world"
    }


def test_none_requires_the_explicit_evaluator_flag() -> None:
    record = _record(DiagnosisLabel.NONE)

    assert curate_records([record]) == ()
    assert {
        sample.target for sample in curate_records([record], include_verified_evaluator=True)
    } == {"evaluator"}


def test_curation_never_uses_an_untrusted_target_field() -> None:
    record = _record(DiagnosisLabel.PF, expert_id="pi05")
    record["target"] = "ctrl_world"

    assert curate_records([record])[0].target == "pi05"


def test_curation_rejects_a_non_string_action_path() -> None:
    record = _record(DiagnosisLabel.PF, expert_id="pi05")
    record["action_path"] = 3

    with pytest.raises(TrainingSchemaError, match="action_path"):
        curate_records([record])


def test_training_records_fixture_contains_representative_diagnosis_mappings() -> None:
    records = [json.loads(line) for line in FIXTURE_PATH.read_text().splitlines() if line]

    samples = curate_records(records)

    assert len(samples) >= 2
    assert {(sample.diagnosis, sample.target) for sample in samples} >= {
        (DiagnosisLabel.PF, "pi05"),
        (DiagnosisLabel.WMF, "ctrl_world"),
    }


def test_group_split_is_seeded_and_independent_of_input_order() -> None:
    samples = tuple(
        _sample(
            "pending",
            sample_id=f"sample-{index}",
            trajectory_id=f"trajectory-{index // 2}",
            scene_id=f"scene-{index // 2}",
            content_sha256=f"{index:064x}",
        )
        for index in range(6)
    )

    first = group_split(samples, seed=7, eval_fraction=0.34)
    second = group_split(tuple(reversed(samples)), seed=7, eval_fraction=0.34)

    assert {sample.sample_id: sample.split for sample in first} == {
        sample.sample_id: sample.split for sample in second
    }
    assert all(first[index].split == first[index + 1].split for index in range(0, 6, 2))


def test_group_split_keeps_different_trajectories_from_the_same_scene_together() -> None:
    samples = (
        _sample(
            "pending",
            sample_id="shared-scene-a",
            trajectory_id="trajectory-a",
            scene_id="shared-scene",
            content_sha256="1" * 64,
        ),
        _sample(
            "pending",
            sample_id="shared-scene-b",
            trajectory_id="trajectory-b",
            scene_id="shared-scene",
            content_sha256="2" * 64,
        ),
        _sample(
            "pending",
            sample_id="independent",
            trajectory_id="trajectory-c",
            scene_id="independent-scene",
            content_sha256="3" * 64,
        ),
    )

    for seed in range(12):
        split_by_id = {
            sample.sample_id: sample.split
            for sample in group_split(samples, seed=seed, eval_fraction=0.5)
        }
        assert split_by_id["shared-scene-a"] == split_by_id["shared-scene-b"]


def test_group_split_closes_transitive_trajectory_and_scene_connections() -> None:
    samples = (
        _sample(
            "pending",
            sample_id="trajectory-link-a",
            trajectory_id="trajectory-a",
            scene_id="scene-a",
            content_sha256="6" * 64,
        ),
        _sample(
            "pending",
            sample_id="trajectory-link-b",
            trajectory_id="trajectory-a",
            scene_id="scene-b",
            content_sha256="7" * 64,
        ),
        _sample(
            "pending",
            sample_id="scene-link",
            trajectory_id="trajectory-b",
            scene_id="scene-b",
            content_sha256="8" * 64,
        ),
        _sample(
            "pending",
            sample_id="independent-component",
            trajectory_id="trajectory-c",
            scene_id="scene-c",
            content_sha256="9" * 64,
        ),
    )

    split_by_id = {
        sample.sample_id: sample.split for sample in group_split(samples, seed=3, eval_fraction=0.5)
    }

    assert (
        len(
            {
                split_by_id["trajectory-link-a"],
                split_by_id["trajectory-link-b"],
                split_by_id["scene-link"],
            }
        )
        == 1
    )


@pytest.mark.parametrize(
    ("key_type", "train", "evaluation"),
    [
        ("sample_id", _sample("train"), _sample("eval")),
        (
            "trajectory_id",
            _sample(
                "train", sample_id="train", trajectory_id="shared-traj", content_sha256="b" * 64
            ),
            _sample("eval", sample_id="eval", trajectory_id="shared-traj", content_sha256="c" * 64),
        ),
        (
            "content_sha256",
            _sample(
                "train", sample_id="train", trajectory_id="train-traj", content_sha256="d" * 64
            ),
            _sample("eval", sample_id="eval", trajectory_id="eval-traj", content_sha256="d" * 64),
        ),
    ],
)
def test_split_rejects_sample_trajectory_or_hash_leakage(
    key_type: str, train: TrainingSample, evaluation: TrainingSample
) -> None:
    with pytest.raises(
        DataLeakageError, match=f"{key_type} leakage between train and eval"
    ) as error:
        validate_no_leakage([train], [evaluation])
    assert error.value.key_type == key_type
    assert error.value.scope == "between train and eval"


@pytest.mark.parametrize("scope", ["train", "eval"])
@pytest.mark.parametrize("key_type", ["sample_id", "trajectory_id", "content_sha256"])
def test_split_rejects_duplicate_provenance_within_each_split(key_type: str, scope: str) -> None:
    first_values: dict[str, object] = {
        "sample_id": "first",
        "trajectory_id": "trajectory-first",
        "content_sha256": "4" * 64,
    }
    second_values: dict[str, object] = {
        "sample_id": "second",
        "trajectory_id": "trajectory-second",
        "content_sha256": "5" * 64,
    }
    second_values[key_type] = first_values[key_type]
    samples = (_sample(scope, **first_values), _sample(scope, **second_values))
    train = samples if scope == "train" else ()
    evaluation = samples if scope == "eval" else ()

    with pytest.raises(DataLeakageError, match=f"{key_type} leakage within {scope}") as error:
        validate_no_leakage(train, evaluation)
    assert error.value.key_type == key_type
    assert error.value.scope == f"within {scope}"


def test_split_accepts_distinct_samples_trajectories_and_content() -> None:
    train = _sample("train", sample_id="train", trajectory_id="train-traj", content_sha256="e" * 64)
    evaluation = _sample(
        "eval", sample_id="eval", trajectory_id="eval-traj", content_sha256="f" * 64
    )

    validate_no_leakage([train], [evaluation])

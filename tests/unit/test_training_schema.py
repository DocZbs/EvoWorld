from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from evoworld.contracts import DiagnosisLabel
from training.common.schema import TrainingManifest, TrainingSample, TrainingSchemaError


def _sample(**changes: object) -> TrainingSample:
    values: dict[str, object] = {
        "schema_version": 1,
        "sample_id": "sample-001",
        "target": "pi05",
        "task_id": "task-001",
        "instruction": "place the mug on the tray",
        "skill_family": "pick_place",
        "scene_id": "scene-001",
        "trajectory_id": "trajectory-001",
        "dataset_root": Path("dataset"),
        "observation_paths": (Path("frames/000.png"),),
        "action_path": Path("actions/000.npy"),
        "action_schema": "joint_velocity",
        "action_dimension": 7,
        "action_horizon": 10,
        "source_type": "real",
        "source_checkpoint": "checkpoint-a",
        "diagnosis": DiagnosisLabel.PF,
        "split": "train",
        "content_sha256": "a" * 64,
    }
    values.update(changes)
    return TrainingSample(**values)  # type: ignore[arg-type]


def test_training_sample_rejects_parent_media_path() -> None:
    with pytest.raises(TrainingSchemaError, match="relative"):
        _sample(observation_paths=(Path("../frame.png"),))


def test_training_sample_keeps_media_paths_unresolved() -> None:
    sample = _sample(observation_paths=(Path("frames/000.png"),))

    assert sample.observation_paths == (Path("frames/000.png"),)


def test_manifest_hash_changes_when_sample_provenance_changes() -> None:
    first = TrainingManifest(schema_version=1, manifest_id="manifest-001", samples=(_sample(),))
    second = TrainingManifest(
        schema_version=1,
        manifest_id="manifest-001",
        samples=(_sample(source_checkpoint="checkpoint-b"),),
    )

    assert first.sha256 != second.sha256

"""Tests for durable, resumable EvoWorld run storage."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from evoworld.contracts import (
    ActionChunk,
    CandidateScore,
    CandidateState,
    Diagnosis,
    DiagnosisLabel,
    ExecutionMode,
    ExperienceRecord,
    ExpertSpec,
    Rollout,
    RoundStage,
    RoundState,
    SceneRecord,
    TaskSpec,
    Trajectory,
    UpdateDecision,
    UpdateTarget,
)
from evoworld.storage import JsonRunStore, JsonlExperienceMemory, atomic_write_json


def _round_state(round_id: str = "round-000001") -> RoundState:
    actions = ActionChunk("pi05", ((0.1,),), (0.0,), "joint_velocity")
    return RoundState(
        round_id=round_id,
        stage=RoundStage.COMPLETE,
        task=TaskSpec("task-1", "place cup"),
        scene=SceneRecord("scene-1", (Path("observation.jpg"),), {"split": "test"}),
        experts=(ExpertSpec("pi05", "local", ("manipulation",), None, "joint_velocity"),),
        candidates=(
            CandidateState(
                ExpertSpec("pi05", "local", ("manipulation",), None, "joint_velocity"),
                actions,
                Rollout("rollout-1", "pi05", "scene-1", (Path("imagined.jpg"),), actions),
                CandidateScore("pi05", "rollout-1", 0.8, 0.7, 0.75),
            ),
        ),
        selected_expert_id="pi05",
        trajectory=Trajectory("trajectory-1", "rollout-1", (Path("frame.jpg"),), actions),
        diagnosis=Diagnosis(0.9, DiagnosisLabel.NONE, "verified"),
        update=UpdateDecision(
            "update-1",
            UpdateTarget.NONE,
            "task-1",
            "scene-1",
            "trajectory-1",
            ExecutionMode.PLAN,
        ),
        started_at="2026-07-17T00:00:00+00:00",
        seed=7,
    )


def _experience_record(record_id: str = "exp-1", *, score: float = 0.9) -> ExperienceRecord:
    return ExperienceRecord(
        record_id=record_id,
        task_id="task-1",
        scene_id="scene-1",
        expert_id="pi05",
        score=CandidateScore(
            expert_id="pi05",
            rollout_id="rollout-1",
            router_score=0.8,
            world_model_score=0.7,
            joint_score=0.75,
        ),
        diagnosis=Diagnosis(score=score, label=DiagnosisLabel.NONE, reason="complete"),
        artifact_refs=(Path("relative/video.mp4"),),
    )


def test_atomic_write_replaces_json_without_temporary_files(tmp_path: Path) -> None:
    """State writes replace a complete file and leave no interrupted-write residue."""
    state_path = tmp_path / "state.json"

    atomic_write_json(state_path, {"round": "old"})
    atomic_write_json(state_path, {"round": "new"})

    assert state_path.read_text(encoding="utf-8") == '{\n  "round": "new"\n}\n'
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_cleans_up_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed atomic replace is re-raised without retaining the temporary file."""
    state_path = tmp_path / "state.json"

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        atomic_write_json(state_path, {"round": "new"})

    assert not state_path.exists()
    assert not any(path.suffix == ".tmp" for path in tmp_path.iterdir())


@pytest.mark.parametrize("run_id", ["", " ", ".", "..", "nested/run", r"nested\run", "/absolute"])
def test_run_store_rejects_unsafe_run_identifiers(tmp_path: Path, run_id: str) -> None:
    """A run identifier cannot resolve outside its configured run root."""
    run_root = tmp_path / "runs"

    with pytest.raises(ValueError, match="run_id"):
        JsonRunStore.create(run_root, run_id, {}, "digest")


def test_run_store_rejects_run_directory_symlink_outside_root(tmp_path: Path) -> None:
    """A valid run component cannot follow a pre-existing symlink out of the run root."""
    run_root = tmp_path / "runs"
    outside = tmp_path / "outside"
    run_root.mkdir()
    outside.mkdir()
    try:
        (run_root / "linked").symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")

    with pytest.raises(ValueError, match="run directory"):
        JsonRunStore.create(run_root, "linked", {}, "digest")

    assert not (outside / "manifest.json").exists()


def test_run_store_round_trips_state_and_rejects_changed_configuration(tmp_path: Path) -> None:
    """Resume reads the complete state only when it belongs to the same configuration."""
    store = JsonRunStore.create(tmp_path, "run-1", {"mode": "dry_run"}, "digest-a")
    store.write_state(_round_state())

    reopened = JsonRunStore.open(store.run_dir, "digest-a")

    assert reopened.read_state() == _round_state()
    with pytest.raises(ValueError, match="config digest"):
        JsonRunStore.open(store.run_dir, "digest-b")


def test_run_store_manifest_is_created_exclusively(tmp_path: Path) -> None:
    """A run identifier cannot silently replace another invocation's manifest."""
    JsonRunStore.create(tmp_path, "run-1", {}, "digest")

    with pytest.raises(FileExistsError):
        JsonRunStore.create(tmp_path, "run-1", {}, "digest")


@pytest.mark.parametrize(
    "round_id", ["", " ", ".", "..", "nested/round", r"nested\round", "/absolute"]
)
def test_run_store_rejects_unsafe_round_identifiers(tmp_path: Path, round_id: str) -> None:
    """Round artifacts cannot escape the run's rounds directory."""
    store = JsonRunStore.create(tmp_path, "run-1", {}, "digest")

    with pytest.raises(ValueError, match="round_id"):
        store.write_artifact(round_id, "grounding", {})


@pytest.mark.parametrize(
    "name",
    ["", " ", ".", "..", "./artifact", "nested/artifact", r"nested\artifact", "/absolute"],
)
def test_run_store_rejects_unsafe_artifact_names(tmp_path: Path, name: str) -> None:
    """Artifact names must be safe portable filename components."""
    store = JsonRunStore.create(tmp_path, "run-1", {}, "digest")

    with pytest.raises(ValueError, match="artifact name"):
        store.write_artifact("round-1", name, {})


@pytest.mark.parametrize("symlink_level", ["rounds", "round"])
def test_run_store_rejects_artifact_symlink_outside_run(tmp_path: Path, symlink_level: str) -> None:
    """Round artifact writes cannot follow directory symlinks outside the run."""
    store = JsonRunStore.create(tmp_path, "run-1", {}, "digest")
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        if symlink_level == "rounds":
            (store.run_dir / "rounds").symlink_to(outside, target_is_directory=True)
        else:
            (store.run_dir / "rounds").mkdir()
            (store.run_dir / "rounds" / "round-1").symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")

    with pytest.raises(ValueError, match="artifact path"):
        store.write_artifact("round-1", "grounding", {})

    assert not (outside / "grounding.json").exists()


def test_memory_append_is_idempotent_and_preserves_absolute_provenance(tmp_path: Path) -> None:
    """A retry is ignored while stored artifact references are portable and absolute."""
    memory = JsonlExperienceMemory(tmp_path / "memory.jsonl")
    record = _experience_record()

    memory.append(record)
    memory.append(record)

    records = memory.lookup(TaskSpec("task-1", "place cup"), SceneRecord("scene-1"))
    assert records == (
        ExperienceRecord(
            record_id="exp-1",
            task_id="task-1",
            scene_id="scene-1",
            expert_id="pi05",
            score=record.score,
            diagnosis=record.diagnosis,
            artifact_refs=((Path.cwd() / "relative/video.mp4").resolve(),),
        ),
    )
    assert len(memory.path.read_text(encoding="utf-8").splitlines()) == 1


def test_memory_rejects_different_record_with_same_deterministic_id(tmp_path: Path) -> None:
    """An ID collision cannot corrupt append-only experience history."""
    memory = JsonlExperienceMemory(tmp_path / "memory.jsonl")
    memory.append(_experience_record("exp-1", score=0.9))

    with pytest.raises(ValueError, match="different record"):
        memory.append(_experience_record("exp-1", score=0.8))

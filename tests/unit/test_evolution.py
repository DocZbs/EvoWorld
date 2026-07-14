"""Unit tests for diagnosis-driven evolution decisions."""

from __future__ import annotations

import re

import pytest

from evoworld.contracts import (
    ActionChunk,
    Diagnosis,
    DiagnosisLabel,
    ExecutionMode,
    ExpertSpec,
    SceneRecord,
    TaskSpec,
    Trajectory,
    UpdateTarget,
)
from evoworld.evolution import decide_update


def _task(task_id: str = "task-1") -> TaskSpec:
    return TaskSpec(task_id=task_id, instruction="place the cup")


def _scene(scene_id: str = "scene-1") -> SceneRecord:
    return SceneRecord(scene_id=scene_id)


def _expert(expert_id: str = "pi05") -> ExpertSpec:
    return ExpertSpec(
        expert_id=expert_id,
        backend="openpi",
        task_families=("manipulation",),
        checkpoint=None,
        action_schema="joint_position",
    )


def _trajectory(trajectory_id: str = "trajectory-1") -> Trajectory:
    return Trajectory(
        trajectory_id=trajectory_id,
        rollout_id="rollout-1",
        frame_paths=(),
        actions=ActionChunk(
            expert_id="pi05",
            actions=((0.0,),),
            timestamps=(0.0,),
            schema_name="joint_position",
        ),
    )


def _decision(
    label: DiagnosisLabel,
    score: float = 0.2,
    *,
    task_id: str = "task-1",
    scene_id: str = "scene-1",
    trajectory_id: str = "trajectory-1",
    expert_id: str = "pi05",
    execution_mode: ExecutionMode = ExecutionMode.PLAN,
):
    return decide_update(
        diagnosis=Diagnosis(score=score, label=label, reason="unit test"),
        reliability_threshold=0.7,
        task=_task(task_id),
        scene=_scene(scene_id),
        expert=_expert(expert_id),
        trajectory=_trajectory(trajectory_id),
        execution_mode=execution_mode,
    )


@pytest.mark.parametrize(
    ("label", "target"),
    (
        (DiagnosisLabel.TSM, UpdateTarget.GROUNDING_ROUTER),
        (DiagnosisLabel.PF, UpdateTarget.EXPERT_POLICY),
        (DiagnosisLabel.WMF, UpdateTarget.WORLD_MODEL),
    ),
)
def test_failure_targets_only_the_diagnosed_module(
    label: DiagnosisLabel, target: UpdateTarget
) -> None:
    """Each low-reliability branch maps to exactly one update target."""
    assert _decision(label).target is target


def test_score_at_threshold_is_a_noop() -> None:
    """The reliability threshold is inclusive."""
    decision = _decision(DiagnosisLabel.NONE, score=0.7)

    assert decision.target is UpdateTarget.NONE


@pytest.mark.parametrize("threshold", (-0.1, 1.1, float("nan"), float("inf")))
def test_invalid_reliability_threshold_is_rejected(threshold: float) -> None:
    """Thresholds must be finite normalized reliability scores."""
    with pytest.raises(ValueError, match="reliability_threshold"):
        decide_update(
            diagnosis=Diagnosis(score=0.2, label=DiagnosisLabel.TSM, reason="unit test"),
            reliability_threshold=threshold,
            task=_task(),
            scene=_scene(),
            expert=_expert(),
            trajectory=_trajectory(),
            execution_mode=ExecutionMode.PLAN,
        )


@pytest.mark.parametrize(
    "diagnosis",
    (
        Diagnosis(score=0.8, label=DiagnosisLabel.TSM, reason="inconsistent"),
        Diagnosis(score=0.2, label=DiagnosisLabel.NONE, reason="inconsistent"),
    ),
)
def test_inconsistent_diagnosis_is_rejected(diagnosis: Diagnosis) -> None:
    """Failure labels and no-op labels must agree with score state."""
    with pytest.raises(ValueError, match="diagnosis"):
        decide_update(
            diagnosis=diagnosis,
            reliability_threshold=0.7,
            task=_task(),
            scene=_scene(),
            expert=_expert(),
            trajectory=_trajectory(),
            execution_mode=ExecutionMode.PLAN,
        )


def test_policy_failure_keeps_only_the_selected_expert() -> None:
    """PF updates train the selected policy and no other expert."""
    decision = _decision(DiagnosisLabel.PF)

    assert decision.expert_id == "pi05"


def test_tsm_has_no_training_artifacts() -> None:
    """TSM reroutes grounding without manufacturing a GPU training plan."""
    decision = _decision(DiagnosisLabel.TSM)

    assert decision.training_dataset is None
    assert decision.command == ()


def test_wmf_targets_the_world_model() -> None:
    """WMF maps to the Ctrl-World semantic target, not an expert policy."""
    decision = _decision(DiagnosisLabel.WMF)

    assert decision.target is UpdateTarget.WORLD_MODEL
    assert decision.expert_id is None


def test_decision_id_and_provenance_are_deterministic() -> None:
    """Identical inputs produce the same traceable update request."""
    first = _decision(DiagnosisLabel.PF)
    second = _decision(DiagnosisLabel.PF)

    assert first.update_id == second.update_id
    assert (first.task_id, first.scene_id, first.trajectory_id) == (
        "task-1",
        "scene-1",
        "trajectory-1",
    )
    assert first.execution_mode is ExecutionMode.PLAN


def test_decision_id_retains_the_full_sha256_digest() -> None:
    """Update identifiers keep all 256 bits of the stable content hash."""
    decision = _decision(DiagnosisLabel.PF)

    assert re.fullmatch(r"update-[0-9a-f]{64}", decision.update_id)


def test_decision_id_changes_for_each_identity_dimension() -> None:
    """Every field that defines an update request participates in its identity."""
    decisions = (
        _decision(DiagnosisLabel.PF),
        _decision(DiagnosisLabel.PF, task_id="task-2"),
        _decision(DiagnosisLabel.PF, scene_id="scene-2"),
        _decision(DiagnosisLabel.PF, trajectory_id="trajectory-2"),
        _decision(DiagnosisLabel.TSM),
        _decision(DiagnosisLabel.PF, expert_id="pi0"),
        _decision(DiagnosisLabel.PF, execution_mode=ExecutionMode.EXECUTE),
    )

    assert len({decision.update_id for decision in decisions}) == len(decisions)


def test_decision_id_has_unambiguous_field_boundaries() -> None:
    """Delimiter characters in IDs cannot move content between identity fields."""
    first = _decision(DiagnosisLabel.PF, task_id="a|b", scene_id="c")
    second = _decision(DiagnosisLabel.PF, task_id="a", scene_id="b|c")

    assert first.update_id != second.update_id

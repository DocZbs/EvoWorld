"""Unit tests for backend-independent EvoWorld domain contracts."""

from __future__ import annotations

import pytest

from pathlib import Path

from evoworld.contracts import (
    ActionChunk,
    CandidateState,
    CandidateScore,
    Diagnosis,
    DiagnosisLabel,
    ExecutionMode,
    ExpertSpec,
    OperationIntent,
    OperationKind,
    OperationReceipt,
    Rollout,
    RoundStage,
    RoundState,
    SceneRecord,
    TaskSpec,
    Trajectory,
    UpdateDecision,
    UpdateTarget,
)
from evoworld.codec import (
    StateDecodeError,
    input_digest,
    round_state_from_json,
    round_state_to_json,
)


def test_task_spec_rejects_blank_instruction() -> None:
    """Tasks require a human-readable instruction."""
    with pytest.raises(ValueError, match="instruction"):
        TaskSpec(task_id="task-1", instruction="   ")


def test_candidate_score_combines_router_and_world_model_scores() -> None:
    """The joint score uses the configured convex weight."""
    score = CandidateScore.from_components("pi05", "r1", 0.8, 0.2, 0.25)

    assert score.joint_score == pytest.approx(0.35)


def test_candidate_score_records_an_invalid_rollout_error() -> None:
    """A failed rollout is retained without inventing a rollout identifier."""
    score = CandidateScore(
        expert_id="pi05",
        rollout_id=None,
        router_score=0.0,
        world_model_score=0.0,
        joint_score=0.0,
        valid=False,
        error="world model timed out",
    )

    assert score.rollout_id is None
    assert score.error == "world model timed out"


def test_invalid_candidate_requires_a_nonblank_error() -> None:
    """Failed candidates must preserve the reason that they could not run."""
    with pytest.raises(ValueError, match="error"):
        CandidateScore(
            expert_id="pi05",
            rollout_id=None,
            router_score=0.0,
            world_model_score=0.0,
            joint_score=0.0,
            valid=False,
            error=" ",
        )


def test_valid_candidate_requires_a_rollout_identifier() -> None:
    """Scores for completed candidates must reference their rollout artifact."""
    with pytest.raises(ValueError, match="rollout_id"):
        CandidateScore(
            expert_id="pi05",
            rollout_id=None,
            router_score=0.0,
            world_model_score=0.0,
            joint_score=0.0,
        )


def test_valid_candidate_rejects_an_error() -> None:
    """A successful candidate cannot simultaneously report a failure."""
    with pytest.raises(ValueError, match="error"):
        CandidateScore(
            expert_id="pi05",
            rollout_id="r1",
            router_score=0.8,
            world_model_score=0.2,
            joint_score=0.5,
            error="timed out",
        )


def test_diagnosis_rejects_out_of_range_score() -> None:
    """Diagnosis confidence is normalized to the unit interval."""
    with pytest.raises(ValueError, match="score"):
        Diagnosis(score=1.1, label=DiagnosisLabel.NONE, reason="invalid")


def test_update_decision_preserves_provenance_and_requires_the_selected_expert() -> None:
    """Only an expert-policy repair retains the expert selected for training."""
    decision = UpdateDecision(
        update_id="update-1",
        target=UpdateTarget.EXPERT_POLICY,
        task_id="task-1",
        scene_id="scene-1",
        trajectory_id="trajectory-1",
        expert_id="pi05",
        training_dataset=Path("/tmp/training.jsonl"),
        command=("train", "--steps", "5000"),
        execution_mode=ExecutionMode.PLAN,
    )

    assert decision.target is UpdateTarget.EXPERT_POLICY
    assert decision.expert_id == "pi05"


@pytest.mark.parametrize(
    "target",
    (UpdateTarget.NONE, UpdateTarget.GROUNDING_ROUTER, UpdateTarget.WORLD_MODEL),
)
def test_non_policy_update_rejects_an_expert_identifier(target: UpdateTarget) -> None:
    """Router, world-model, and no-op decisions must not target an expert."""
    with pytest.raises(ValueError, match="expert_id"):
        UpdateDecision(
            update_id="update-1",
            target=target,
            task_id="task-1",
            scene_id="scene-1",
            trajectory_id="trajectory-1",
            expert_id="pi05",
            execution_mode=ExecutionMode.PLAN,
        )


def test_round_state_requires_typed_stage_and_stage_outputs() -> None:
    """Resumable state cannot claim a completed stage without its typed output."""
    with pytest.raises(ValueError, match="stage"):
        RoundState(
            round_id="round-1",
            stage="grounded",  # type: ignore[arg-type]
            task=TaskSpec("task-1", "place cup"),
            scene=SceneRecord("scene-1"),
            started_at="2026-07-17T00:00:00+00:00",
            seed=7,
        )

    with pytest.raises(ValueError, match="candidates"):
        RoundState(
            round_id="round-1",
            stage=RoundStage.CANDIDATES_SCORED,
            task=TaskSpec("task-1", "place cup"),
            scene=SceneRecord("scene-1"),
            started_at="2026-07-17T00:00:00+00:00",
            seed=7,
        )


@pytest.mark.parametrize(
    "expert_id",
    ("CON", "con.txt", "PRN", "AUX.log", "NUL", "COM1", "com9.bin", "LPT1", "lpt9.txt", "safe."),
)
def test_expert_id_rejects_nonportable_windows_names(expert_id: str) -> None:
    with pytest.raises(ValueError, match="portable"):
        ExpertSpec(expert_id, "local", (), None, "joint_velocity")


def test_candidates_state_rejects_trajectory_action_mismatch() -> None:
    expert = ExpertSpec("pi05", "local", (), None, "joint_velocity")
    actions = ActionChunk("pi05", ((0.1,),), (0.0,), "joint_velocity")
    other = ActionChunk("pi05", ((0.2,),), (0.0,), "joint_velocity")
    rollout = Rollout("rollout-1", "pi05", "scene-1", (), actions)
    candidate = CandidateState(
        expert,
        actions,
        rollout,
        CandidateScore("pi05", "rollout-1", 0.8, 0.8, 0.8),
    )
    with pytest.raises(ValueError, match="selected rollout"):
        RoundState(
            round_id="round-1",
            stage=RoundStage.CANDIDATES_SCORED,
            started_at="2026-07-17T00:00:00+00:00",
            seed=7,
            task=TaskSpec("task-1", "place cup"),
            scene=SceneRecord("scene-1"),
            experts=(expert,),
            candidates=(candidate,),
            selected_expert_id="pi05",
            trajectory=Trajectory("trajectory-1", "rollout-1", (), other),
        )


def test_candidate_score_rollout_must_match_candidate_rollout() -> None:
    expert = ExpertSpec("pi05", "local", (), None, "joint_velocity")
    actions = ActionChunk("pi05", ((0.1,),), (0.0,), "joint_velocity")
    rollout = Rollout("rollout-1", "pi05", "scene-1", (), actions)

    with pytest.raises(ValueError, match="score rollout"):
        CandidateState(
            expert,
            actions,
            rollout,
            CandidateScore("pi05", "rollout-forged", 0.8, 0.8, 0.8),
        )


def test_valid_candidate_score_requires_a_durable_rollout() -> None:
    expert = ExpertSpec("pi05", "local", (), None, "joint_velocity")

    with pytest.raises(ValueError, match="valid score requires"):
        CandidateState(
            expert,
            score=CandidateScore("pi05", "rollout-missing", 0.8, 0.8, 0.8),
        )

    failed = CandidateState(
        expert,
        score=CandidateScore("pi05", None, 0.0, 0.0, 0.0, valid=False, error="rollout failed"),
    )
    assert failed.rollout is None


def test_in_doubt_evaluate_without_trajectory_is_typed_decode_error() -> None:
    payload = round_state_to_json(
        RoundState(
            round_id="round-1",
            stage=RoundStage.STARTED,
            started_at="2026-07-17T00:00:00+00:00",
            seed=7,
        )
    )
    payload["stage"] = RoundStage.IN_DOUBT.value
    payload["in_doubt_reason"] = "manual reconciliation required"
    payload["pending_operation"] = {
        "operation_id": "round-1:evaluate",
        "kind": OperationKind.EVALUATE.value,
        "input_digest": input_digest({"task": {}, "scene": {}, "trajectory": {}}),
        "input_payload": {"task": {}, "scene": {}, "trajectory": {}},
        "expert_id": None,
    }
    with pytest.raises(StateDecodeError):
        round_state_from_json(payload)


def test_operation_digest_must_match_its_canonical_input_payload() -> None:
    with pytest.raises(ValueError, match="canonical input_payload"):
        OperationIntent(
            "round-1:task",
            OperationKind.TASK,
            "f" * 64,
            {},
        )


def test_round_state_decode_rejects_mutated_receipt_digest() -> None:
    state = RoundState(
        round_id="round-1",
        stage=RoundStage.STARTED,
        started_at="2026-07-17T00:00:00+00:00",
        seed=7,
        task=TaskSpec("task-1", "place cup"),
        receipts=(
            OperationReceipt(
                "round-1:task",
                OperationKind.TASK,
                input_digest({}),
                {},
            ),
        ),
    )
    payload = round_state_to_json(state)
    receipts = payload["receipts"]
    assert isinstance(receipts, list) and isinstance(receipts[0], dict)
    receipts[0]["input_digest"] = "f" * 64

    with pytest.raises(StateDecodeError, match="domain contracts"):
        round_state_from_json(payload)


def test_round_state_rejects_semantically_mismatched_receipt_payload() -> None:
    wrong_input = {
        "task": {
            "task_id": "other-task",
            "instruction": "place cup",
            "skill_hints": [],
            "metadata": {},
        }
    }
    with pytest.raises(ValueError, match="durable state"):
        RoundState(
            round_id="round-1",
            stage=RoundStage.STARTED,
            started_at="2026-07-17T00:00:00+00:00",
            seed=7,
            task=TaskSpec("task-1", "place cup"),
            scene=SceneRecord("scene-1"),
            receipts=(
                OperationReceipt(
                    "round-1:anchor",
                    OperationKind.ANCHOR,
                    input_digest(wrong_input),
                    wrong_input,
                ),
            ),
        )

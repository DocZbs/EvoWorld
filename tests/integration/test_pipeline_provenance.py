"""Untrusted adapter provenance must fail before physical execution."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from evoworld.adapters.deterministic import (
    DeterministicScenario,
    build_deterministic_adapters,
)
from evoworld.config import EvoWorldConfig
from evoworld.codec import (
    StateDecodeError,
    input_digest,
    round_state_from_json,
    round_state_to_json,
    update_to_json,
)
from evoworld.contracts import (
    ActionChunk,
    CandidateScore,
    DiagnosisLabel,
    ExecutionMode,
    ExpertSpec,
    GroundingMode,
    Rollout,
    OperationKind,
    RoundStage,
    SceneRecord,
    TaskSpec,
    Trajectory,
    UpdateDecision,
)
from evoworld.pipeline import EvoWorldPipeline
from evoworld.evolution import decide_update
from evoworld.routing import NoValidCandidateError


@pytest.mark.parametrize("malicious_stage", ("action", "rollout", "score"))
def test_mismatched_candidate_provenance_never_reaches_executor(
    tmp_path: Path, malicious_stage: str
) -> None:
    config = _config(tmp_path)
    adapters = build_deterministic_adapters(config, tmp_path / malicious_stage)
    executor = Mock(wraps=adapters.executor)
    changes: dict[str, object] = {"executor": executor}
    if malicious_stage == "action":
        changes["policies"] = {expert.expert_id: _WrongActionPolicy() for expert in config.experts}
    elif malicious_stage == "rollout":
        changes["world_model"] = _WrongSceneWorldModel()
    else:
        changes["rollout_scorer"] = _WrongExpertScorer()

    with pytest.raises(NoValidCandidateError):
        EvoWorldPipeline(config, replace(adapters, **changes)).run(1)

    executor.execute.assert_not_called()


def test_executor_must_preserve_the_selected_action_chunk(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / "executor-actions"
    adapters = build_deterministic_adapters(config, run_dir)
    malicious = _WrongActionExecutor(adapters.executor)

    with pytest.raises(RuntimeError, match="ValueError: execute failed"):
        EvoWorldPipeline(config, replace(adapters, executor=malicious)).run_round("round-000001")

    failed = adapters.store.load_state("round-000001")
    assert failed is not None and failed.trajectory is None
    assert all(receipt.kind is not OperationKind.EXECUTE for receipt in failed.receipts)
    resumed = EvoWorldPipeline(config, build_deterministic_adapters(config, run_dir)).resume()
    assert resumed is not None and resumed.trajectory.actions


def test_trainer_may_enrich_artifacts_but_cannot_rewrite_identity(tmp_path: Path) -> None:
    config = _config(tmp_path)
    adapters = build_deterministic_adapters(
        config,
        tmp_path / "trainer-identity",
        DeterministicScenario(
            evaluation_score=0.2,
            evaluation_label=DiagnosisLabel.PF,
        ),
    )

    with pytest.raises(RuntimeError, match="ValueError: train_plan failed"):
        EvoWorldPipeline(config, replace(adapters, trainer=_WrongIdentityTrainer())).run(1)


def test_forged_rollout_score_and_trajectory_chain_is_rejected_on_decode(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    adapters = build_deterministic_adapters(config, tmp_path / "forged-chain")
    EvoWorldPipeline(config, adapters).run(1)
    state = adapters.store.read_state()
    assert state is not None
    payload = round_state_to_json(state)
    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    selected_id = payload["selected_expert_id"]
    selected = next(
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and isinstance(candidate.get("expert"), dict)
        and candidate["expert"].get("expert_id") == selected_id
    )
    assert isinstance(selected["rollout"], dict)
    assert isinstance(selected["score"], dict)
    assert isinstance(payload["trajectory"], dict)
    selected["rollout"]["rollout_id"] = "rollout-forged"
    selected["score"]["rollout_id"] = "rollout-forged"
    payload["trajectory"]["rollout_id"] = "rollout-forged"

    with pytest.raises(StateDecodeError):
        round_state_from_json(payload)


def test_valid_scored_candidate_without_rollout_is_typed_decode_error(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    adapters = build_deterministic_adapters(config, tmp_path / "missing-valid-rollout")
    EvoWorldPipeline(config, adapters).run(1)
    state = adapters.store.read_state()
    assert state is not None
    payload = round_state_to_json(state)
    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    selected = next(
        item
        for item in candidates
        if isinstance(item, dict)
        and isinstance(item.get("expert"), dict)
        and item["expert"].get("expert_id") == payload["selected_expert_id"]
    )
    selected["rollout"] = None

    with pytest.raises(StateDecodeError):
        round_state_from_json(payload)


@pytest.mark.parametrize(
    ("kind", "mode"),
    (
        (OperationKind.TRAIN_PLAN, ExecutionMode.PLAN),
        (OperationKind.TRAIN_EXECUTE, ExecutionMode.EXECUTE),
    ),
)
def test_recomputed_training_receipt_digest_cannot_hide_changed_command(
    tmp_path: Path, kind: OperationKind, mode: ExecutionMode
) -> None:
    config = _config(tmp_path)
    adapters = build_deterministic_adapters(
        config,
        tmp_path / kind.value,
        DeterministicScenario(
            evaluation_score=0.2,
            evaluation_label=DiagnosisLabel.PF,
        ),
    )
    EvoWorldPipeline(config, adapters, execution_mode=mode).run(1)
    state = adapters.store.read_state()
    assert state is not None
    payload = round_state_to_json(state)
    receipts = payload["receipts"]
    assert isinstance(receipts, list)
    receipt = next(
        item for item in receipts if isinstance(item, dict) and item.get("kind") == kind.value
    )
    operation_input = receipt["input_payload"]
    assert isinstance(operation_input, dict) and isinstance(operation_input["update"], dict)
    operation_input["update"]["command"] = ["unexpected-command"]
    receipt["input_digest"] = input_digest(operation_input)

    with pytest.raises(StateDecodeError):
        round_state_from_json(payload)


def test_in_doubt_train_execute_requires_diagnosis_and_consistent_plan(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    adapters = build_deterministic_adapters(
        config,
        tmp_path / "train-execute-in-doubt",
        DeterministicScenario(
            evaluation_score=0.2,
            evaluation_label=DiagnosisLabel.PF,
        ),
    )
    EvoWorldPipeline(config, adapters, execution_mode=ExecutionMode.EXECUTE).run(1)
    state = adapters.store.read_state()
    assert state is not None
    payload = round_state_to_json(state)
    receipts = payload["receipts"]
    assert isinstance(receipts, list)
    execute_receipt = next(
        item
        for item in receipts
        if isinstance(item, dict) and item.get("kind") == OperationKind.TRAIN_EXECUTE.value
    )
    payload["receipts"] = [item for item in receipts if item is not execute_receipt]
    payload["pending_operation"] = {
        **execute_receipt,
        "expert_id": None,
    }
    payload["stage"] = RoundStage.IN_DOUBT.value
    payload["in_doubt_reason"] = "manual reconciliation required"
    payload["diagnosis"] = None

    with pytest.raises(StateDecodeError):
        round_state_from_json(payload)


def test_plan_round_cannot_be_upgraded_by_recomputed_execute_decision(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    adapters = build_deterministic_adapters(
        config,
        tmp_path / "plan-upgrade",
        DeterministicScenario(
            evaluation_score=0.2,
            evaluation_label=DiagnosisLabel.PF,
        ),
    )
    EvoWorldPipeline(config, adapters, execution_mode=ExecutionMode.PLAN).run(1)
    state = adapters.store.read_state()
    assert (
        state is not None
        and state.task is not None
        and state.scene is not None
        and state.trajectory is not None
        and state.diagnosis is not None
    )
    selected = next(
        candidate.expert
        for candidate in state.candidates
        if candidate.expert.expert_id == state.selected_expert_id
    )
    forged = decide_update(
        diagnosis=state.diagnosis,
        reliability_threshold=state.reliability_threshold,
        task=state.task,
        scene=state.scene,
        expert=selected,
        trajectory=state.trajectory,
        execution_mode=ExecutionMode.EXECUTE,
    )
    forged_json = update_to_json(forged)
    payload = round_state_to_json(state)
    payload["update"] = forged_json
    receipts = payload["receipts"]
    assert isinstance(receipts, list)
    plan_receipt = next(
        item
        for item in receipts
        if isinstance(item, dict) and item.get("kind") == OperationKind.TRAIN_PLAN.value
    )
    plan_receipt["input_payload"] = {"update": forged_json}
    plan_receipt["input_digest"] = input_digest(plan_receipt["input_payload"])

    with pytest.raises(StateDecodeError):
        round_state_from_json(payload)


def test_plan_round_rejects_in_doubt_train_execute(tmp_path: Path) -> None:
    config = _config(tmp_path)
    adapters = build_deterministic_adapters(
        config,
        tmp_path / "plan-in-doubt-execute",
        DeterministicScenario(
            evaluation_score=0.2,
            evaluation_label=DiagnosisLabel.PF,
        ),
    )
    EvoWorldPipeline(config, adapters, execution_mode=ExecutionMode.PLAN).run(1)
    state = adapters.store.read_state()
    assert state is not None and state.update is not None
    update_payload = {"update": update_to_json(state.update)}
    payload = round_state_to_json(state)
    payload["stage"] = RoundStage.IN_DOUBT.value
    payload["in_doubt_reason"] = "manual reconciliation required"
    payload["pending_operation"] = {
        "operation_id": f"{state.round_id}:train_execute",
        "kind": OperationKind.TRAIN_EXECUTE.value,
        "input_digest": input_digest(update_payload),
        "input_payload": update_payload,
        "expert_id": None,
    }

    with pytest.raises(StateDecodeError):
        round_state_from_json(payload)


class _WrongActionPolicy:
    def act(self, task: TaskSpec, observation: SceneRecord, *, operation_id: str) -> ActionChunk:
        del task, observation, operation_id
        return ActionChunk("intruder", ((0.1,),), (0.0,), "joint_velocity")


class _WrongSceneWorldModel:
    def imagine(
        self,
        task: TaskSpec,
        scene: SceneRecord,
        expert: ExpertSpec,
        actions: ActionChunk,
        *,
        operation_id: str,
    ) -> Rollout:
        del task, scene, operation_id
        return Rollout(
            f"rollout-{expert.expert_id}",
            expert.expert_id,
            "wrong-scene",
            (),
            actions,
        )


class _WrongExpertScorer:
    def score(self, task: TaskSpec, rollout: Rollout, *, operation_id: str) -> CandidateScore:
        del task, operation_id
        return CandidateScore("intruder", rollout.rollout_id, 0.5, 0.5, 0.5)


class _WrongActionExecutor:
    def __init__(self, delegate) -> None:
        self._delegate = delegate

    def execute(
        self,
        task: TaskSpec,
        scene: SceneRecord,
        expert: ExpertSpec,
        actions: ActionChunk,
        *,
        operation_id: str,
    ) -> Trajectory:
        trajectory = self._delegate.execute(task, scene, expert, actions, operation_id=operation_id)
        wrong = ActionChunk(
            actions.expert_id,
            tuple((value + 0.01,) for (value,) in actions.actions),
            actions.timestamps,
            actions.schema_name,
        )
        return replace(trajectory, actions=wrong)


class _WrongIdentityTrainer:
    def plan(self, update: UpdateDecision, *, operation_id: str) -> UpdateDecision:
        del operation_id
        return replace(update, task_id="wrong-task")

    def execute(self, plan: UpdateDecision, *, operation_id: str) -> UpdateDecision:
        del operation_id
        return plan


def _config(tmp_path: Path) -> EvoWorldConfig:
    return EvoWorldConfig(
        schema_version=1,
        seed=7,
        run_root=(tmp_path / "runs").resolve(),
        mode="dry_run",
        grounding_mode=GroundingMode.NEAREST,
        grounding_temperature=0.5,
        router_weight=0.25,
        reliability_threshold=0.7,
        rollout_horizon=3,
        executor=None,
        experts=tuple(
            ExpertSpec(
                expert_id,
                "local",
                ("manipulation",),
                None,
                "joint_velocity",
            )
            for expert_id in ("groot", "pi05")
        ),
    )

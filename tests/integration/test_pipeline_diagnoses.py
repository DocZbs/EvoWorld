"""Integration coverage for diagnosis routing and candidate failure isolation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest

from evoworld.adapters.deterministic import (
    AdapterSet,
    DeterministicScenario,
    build_deterministic_adapters,
)
from evoworld.codec import (
    StateDecodeError,
    action_to_json,
    expert_to_json,
    input_digest,
    round_state_from_json,
    round_state_to_json,
    scene_to_json,
    task_to_json,
    update_to_json,
)
from evoworld.config import EvoWorldConfig
from evoworld.contracts import (
    ActionChunk,
    DiagnosisLabel,
    ExecutionMode,
    ExpertSpec,
    GroundingMode,
    JsonObject,
    OperationKind,
    Rollout,
    RoundStage,
    RoundState,
    SceneRecord,
    TaskSpec,
    Trajectory,
    UpdateTarget,
)
from evoworld.pipeline import EvoWorldPipeline
from evoworld.ports import OperationInDoubtError, WorldModel
from evoworld.routing import NoValidCandidateError
from evoworld.storage import JsonRunStore

_SENSITIVE_FRAGMENTS = (
    "space separated secret value",
    "file:///Users/private/a folder/model.bin",
    "/Users/private/a folder/model.bin",
    "AKIAIOSFODNN7EXAMPLE",
    "hf_super_secret_model_token",
    "xoxb-123456789-secret",
    "arbitrary-unstructured-secret",
)
_DYNAMIC_SECRET_CLASS_NAMES = (
    "hf_super_secret_model_token",
    "AKIAIOSFODNN7EXAMPLE",
    "xoxb_123456789_secret",
)


@pytest.mark.parametrize(
    ("label", "target", "expected_expert"),
    (
        (DiagnosisLabel.TSM, UpdateTarget.GROUNDING_ROUTER, None),
        (DiagnosisLabel.PF, UpdateTarget.EXPERT_POLICY, "pi05"),
        (DiagnosisLabel.WMF, UpdateTarget.WORLD_MODEL, None),
    ),
)
def test_plan_mode_persists_only_the_targeted_diagnosis_update(
    tmp_path: Path,
    label: DiagnosisLabel,
    target: UpdateTarget,
    expected_expert: str | None,
) -> None:
    pipeline, adapters = _pipeline(
        tmp_path,
        label=label,
        score=0.2,
        execution_mode=ExecutionMode.PLAN,
    )

    result = pipeline.run(1)[0]

    assert not result.verified
    assert result.update.target is target
    assert result.update.expert_id == expected_expert
    assert result.update.execution_mode is ExecutionMode.PLAN
    assert result.update.command == ()
    assert result.update.training_dataset is None
    state = adapters.store.read_state()  # type: ignore[attr-defined]
    assert state is not None
    assert all(receipt.kind is not OperationKind.TRAIN_EXECUTE for receipt in state.receipts)
    assert not (adapters.store.run_dir / "memory.jsonl").exists()

    round_dir = adapters.store.run_dir / "rounds" / result.round_id
    assert (round_dir / "grounding.json").is_file()
    assert (round_dir / "candidates.json").is_file()
    diagnosis = json.loads((round_dir / "diagnosis.json").read_text(encoding="utf-8"))
    update = json.loads((round_dir / "update.json").read_text(encoding="utf-8"))
    assert diagnosis == {
        "diagnosis": {
            "label": label.value,
            "metadata": {"trajectory_id": result.trajectory.trajectory_id},
            "reason": "deterministic local evaluation",
            "score": 0.2,
        },
        "verified": False,
    }
    assert update["target"] == target.value
    assert update["expert_id"] == expected_expert


@pytest.mark.parametrize(
    ("label", "score", "target", "executes_training", "verified"),
    (
        (DiagnosisLabel.TSM, 0.2, UpdateTarget.GROUNDING_ROUTER, False, False),
        (DiagnosisLabel.PF, 0.2, UpdateTarget.EXPERT_POLICY, True, False),
        (DiagnosisLabel.WMF, 0.2, UpdateTarget.WORLD_MODEL, True, False),
        (DiagnosisLabel.NONE, 0.9, UpdateTarget.NONE, False, True),
    ),
)
def test_execute_mode_runs_only_gpu_relevant_failure_updates(
    tmp_path: Path,
    label: DiagnosisLabel,
    score: float,
    target: UpdateTarget,
    executes_training: bool,
    verified: bool,
) -> None:
    pipeline, adapters = _pipeline(
        tmp_path,
        label=label,
        score=score,
        execution_mode=ExecutionMode.EXECUTE,
    )

    result = pipeline.run(1)[0]

    assert result.verified is verified
    assert result.update.target is target
    operation_id = f"{result.round_id}:train_execute"
    assert adapters.receipts.effect_count("train_execute", operation_id) == int(executes_training)
    state = adapters.store.read_state()  # type: ignore[attr-defined]
    assert state is not None
    assert (
        any(receipt.kind is OperationKind.TRAIN_EXECUTE for receipt in state.receipts)
        is executes_training
    )
    if label is DiagnosisLabel.PF:
        assert result.update.expert_id == result.trajectory.actions.expert_id == "pi05"
        assert result.update.expert_id != "groot"
    else:
        assert result.update.expert_id is None
    if label is DiagnosisLabel.WMF:
        assert result.update.target.value == "world_model"
    if label is DiagnosisLabel.TSM:
        assert (adapters.store.run_dir / "rounds" / result.round_id / "grounding.json").is_file()
    if verified:
        assert (adapters.store.run_dir / "verified" / result.trajectory.trajectory_id).is_dir()


def test_partial_candidate_failure_is_sanitized_and_valid_candidate_continues(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    adapters = build_deterministic_adapters(config, tmp_path / "partial")
    world_model = _SelectiveFailureWorldModel(
        adapters.world_model,
        failing_experts=frozenset({"groot"}),
    )

    result = EvoWorldPipeline(config, replace(adapters, world_model=world_model)).run(1)[0]

    assert tuple(score.expert_id for score in result.scores) == ("pi05", "groot")
    failed = result.scores[1]
    assert not failed.valid
    assert failed.rollout_id is None
    error = cast(str, failed.error)
    assert error == "RuntimeError: imagine failed"
    assert len(error) <= 256
    assert all(fragment not in error for fragment in _SENSITIVE_FRAGMENTS)
    assert result.trajectory.actions.expert_id == "pi05"
    assert world_model.calls == ["pi05", "groot"]

    candidates_path = adapters.store.run_dir / "rounds" / result.round_id / "candidates.json"
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    assert [item["expert"]["expert_id"] for item in candidates["candidates"]] == [
        "pi05",
        "groot",
    ]
    failed_artifact = candidates["candidates"][1]
    assert failed_artifact["rollout"] is None
    assert failed_artifact["score"]["error"] == error
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            adapters.store.run_dir / "state.json",
            adapters.store.run_dir / "rounds" / result.round_id / "state.json",
            candidates_path,
        )
    )
    assert all(fragment not in persisted for fragment in _SENSITIVE_FRAGMENTS)


def test_total_candidate_failure_persists_all_errors_without_downstream_effects(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    adapters = build_deterministic_adapters(config, tmp_path / "total")
    world_model = _SelectiveFailureWorldModel(
        adapters.world_model,
        failing_experts=frozenset({"groot", "pi05"}),
    )
    executor = Mock(wraps=adapters.executor)
    evaluator = Mock(wraps=adapters.evaluator)
    trainer = Mock(wraps=adapters.trainer)
    memory = Mock(wraps=adapters.memory)
    pipeline = EvoWorldPipeline(
        config,
        replace(
            adapters,
            world_model=world_model,
            executor=executor,
            evaluator=evaluator,
            trainer=trainer,
            memory=memory,
        ),
    )

    with pytest.raises(NoValidCandidateError) as raised:
        pipeline.run(1)

    assert all(fragment not in str(raised.value) for fragment in _SENSITIVE_FRAGMENTS)

    state = adapters.store.read_state()  # type: ignore[attr-defined]
    assert state is not None
    assert tuple(candidate.expert.expert_id for candidate in state.candidates) == (
        "pi05",
        "groot",
    )
    assert tuple(score.expert_id for score in raised.value.candidates) == (
        "pi05",
        "groot",
    )
    assert all(
        candidate.score is not None and not candidate.score.valid for candidate in state.candidates
    )
    assert all(candidate.rollout is None for candidate in state.candidates)
    assert state.pending_operation is None
    assert len({receipt.operation_id for receipt in state.receipts}) == len(state.receipts)
    assert {receipt.kind for receipt in state.receipts} == {
        OperationKind.TASK,
        OperationKind.ANCHOR,
        OperationKind.ROUTE,
        OperationKind.ACT,
        OperationKind.IMAGINE,
    }
    executor.execute.assert_not_called()
    evaluator.evaluate.assert_not_called()
    trainer.plan.assert_not_called()
    trainer.execute.assert_not_called()
    memory.append.assert_not_called()

    round_dir = adapters.store.run_dir / "rounds" / state.round_id
    assert {path.name for path in round_dir.iterdir()} == {
        ".claim",
        "candidates.json",
        "grounding.json",
        "rollouts",
        "state.json",
    }
    assert all(
        {path.name for path in expert_dir.iterdir()}
        == {".act.receipt.json.lock", "act.receipt.json"}
        for expert_dir in (round_dir / "rollouts").iterdir()
    )
    artifact_before_resume = (round_dir / "candidates.json").read_bytes()
    calls_before_resume = tuple(world_model.calls)

    with pytest.raises(NoValidCandidateError):
        pipeline.resume()

    assert tuple(world_model.calls) == calls_before_resume
    assert (round_dir / "candidates.json").read_bytes() == artifact_before_resume
    executor.execute.assert_not_called()
    evaluator.evaluate.assert_not_called()
    trainer.plan.assert_not_called()
    memory.append.assert_not_called()


def test_candidate_failure_receipt_survives_fresh_process_restart(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / "fresh-restart"
    first = build_deterministic_adapters(config, run_dir)
    world_model = _SelectiveFailureWorldModel(
        first.world_model,
        failing_experts=frozenset({"groot"}),
    )
    interrupted_store = _InterruptAfterCandidatesStore(cast(JsonRunStore, first.store))

    with pytest.raises(_RestartInterruption):
        EvoWorldPipeline(
            config,
            replace(first, world_model=world_model, store=interrupted_store),
        ).run(1)

    persisted = interrupted_store.read_state()
    assert persisted is not None and persisted.stage is RoundStage.CANDIDATES_SCORED
    failed_candidate = next(
        candidate for candidate in persisted.candidates if candidate.expert.expert_id == "groot"
    )
    assert failed_candidate.score is not None and not failed_candidate.score.valid
    failure_receipt = next(
        receipt
        for receipt in persisted.receipts
        if receipt.operation_id == "round-000000:imagine:groot"
    )
    assert failure_receipt.kind is OperationKind.IMAGINE
    assert persisted.task is not None and persisted.scene is not None
    assert failed_candidate.actions is not None
    assert failure_receipt.input_payload == {
        "task": task_to_json(persisted.task),
        "scene": scene_to_json(persisted.scene),
        "expert": expert_to_json(failed_candidate.expert),
        "actions": action_to_json(failed_candidate.actions),
    }
    assert failure_receipt.input_digest == input_digest(failure_receipt.input_payload)
    assert (
        sum(receipt.operation_id == failure_receipt.operation_id for receipt in persisted.receipts)
        == 1
    )
    assert world_model.calls.count("groot") == 1
    for kind in ("act", "imagine", "score"):
        assert first.receipts.effect_count(kind, f"round-000000:{kind}:pi05") == 1

    restarted = build_deterministic_adapters(config, run_dir)
    result = EvoWorldPipeline(config, restarted).resume()

    assert result is not None and result.verified
    assert result.trajectory.actions.expert_id == "pi05"
    assert world_model.calls.count("groot") == 1
    completed = restarted.store.read_state()  # type: ignore[attr-defined]
    assert completed is not None
    assert (
        next(
            receipt
            for receipt in completed.receipts
            if receipt.operation_id == failure_receipt.operation_id
        )
        == failure_receipt
    )
    for expert_id in ("pi05", "groot"):
        for kind in ("act", "imagine", "score"):
            assert restarted.receipts.effect_count(kind, f"round-000000:{kind}:{expert_id}") == 0


def test_in_doubt_backend_message_is_neither_persisted_nor_exposed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    adapters = build_deterministic_adapters(config, tmp_path / "generic-in-doubt")
    pipeline = EvoWorldPipeline(config, replace(adapters, executor=_InDoubtExecutor()))

    with pytest.raises(OperationInDoubtError) as raised:
        pipeline.run(1)

    expected = "operation round-000000:execute:pi05 (execute) is in doubt"
    assert str(raised.value) == expected
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    state = adapters.store.read_state()  # type: ignore[attr-defined]
    assert state is not None and state.in_doubt_reason == expected
    persisted = (adapters.store.run_dir / "state.json").read_text(encoding="utf-8")
    assert all(fragment not in persisted for fragment in _SENSITIVE_FRAGMENTS)
    assert all(fragment not in str(raised.value) for fragment in _SENSITIVE_FRAGMENTS)


def test_non_candidate_backend_failure_exposes_only_generic_context(tmp_path: Path) -> None:
    config = _config(tmp_path)
    adapters = build_deterministic_adapters(config, tmp_path / "generic-failure")
    pipeline = EvoWorldPipeline(config, replace(adapters, executor=_FailingExecutor()))

    with pytest.raises(RuntimeError) as raised:
        pipeline.run(1)

    assert str(raised.value) == "RuntimeError: execute failed"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert all(fragment not in str(raised.value) for fragment in _SENSITIVE_FRAGMENTS)


@pytest.mark.parametrize(
    "error_name",
    ("space separated secret value", *_DYNAMIC_SECRET_CLASS_NAMES),
)
def test_unknown_exception_class_name_falls_back_to_backend_error(
    tmp_path: Path, error_name: str
) -> None:
    config = _config(tmp_path)
    adapters = build_deterministic_adapters(config, tmp_path / f"class-name-{len(error_name)}")
    unsafe_error = type(error_name, (RuntimeError,), {})
    world_model = _SelectiveFailureWorldModel(
        adapters.world_model,
        failing_experts=frozenset({"groot"}),
        error_type=unsafe_error,
    )

    result = EvoWorldPipeline(config, replace(adapters, world_model=world_model)).run(1)[0]

    failed = next(score for score in result.scores if not score.valid)
    assert failed.error == "BackendError: imagine failed"
    assert error_name not in cast(str, failed.error)
    round_dir = adapters.store.run_dir / "rounds" / result.round_id
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            adapters.store.run_dir / "state.json",
            round_dir / "state.json",
            round_dir / "candidates.json",
        )
    )
    assert error_name not in persisted


@pytest.mark.parametrize("error_name", _DYNAMIC_SECRET_CLASS_NAMES)
def test_unknown_exception_class_name_is_not_exposed_by_non_candidate_failure(
    tmp_path: Path, error_name: str
) -> None:
    config = _config(tmp_path)
    adapters = build_deterministic_adapters(config, tmp_path / f"external-{len(error_name)}")
    unsafe_error = type(error_name, (RuntimeError,), {})
    pipeline = EvoWorldPipeline(
        config,
        replace(adapters, executor=_FailingExecutor(error_type=unsafe_error)),
    )

    with pytest.raises(RuntimeError) as raised:
        pipeline.run(1)

    assert str(raised.value) == "BackendError: execute failed"
    persisted = (adapters.store.run_dir / "state.json").read_text(encoding="utf-8")
    forbidden = (error_name, *_SENSITIVE_FRAGMENTS)
    assert all(fragment not in str(raised.value) for fragment in forbidden)
    assert all(fragment not in persisted for fragment in forbidden)


def test_spoofed_repository_exception_metadata_does_not_bypass_allowlist(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    adapters = build_deterministic_adapters(config, tmp_path / "spoofed-class")
    spoofed_error = type(
        "DeterministicRolloutError",
        (RuntimeError,),
        {"__module__": "evoworld.adapters.deterministic"},
    )
    world_model = _SelectiveFailureWorldModel(
        adapters.world_model,
        failing_experts=frozenset({"groot"}),
        error_type=spoofed_error,
    )

    result = EvoWorldPipeline(config, replace(adapters, world_model=world_model)).run(1)[0]

    failed = next(score for score in result.scores if not score.valid)
    assert failed.error == "BackendError: imagine failed"


@pytest.mark.parametrize("label", (DiagnosisLabel.TSM, DiagnosisLabel.NONE))
@pytest.mark.parametrize("representation", ("receipt", "in_doubt"))
def test_non_gpu_update_rejects_forged_train_execute_state(
    tmp_path: Path,
    label: DiagnosisLabel,
    representation: str,
) -> None:
    score = 0.9 if label is DiagnosisLabel.NONE else 0.2
    pipeline, adapters = _pipeline(
        tmp_path,
        label=label,
        score=score,
        execution_mode=ExecutionMode.EXECUTE,
    )
    pipeline.run(1)
    state = adapters.store.read_state()  # type: ignore[attr-defined]
    assert state is not None and state.update is not None
    update_payload: JsonObject = {"update": update_to_json(state.update)}
    operation = {
        "operation_id": f"{state.round_id}:train_execute",
        "kind": OperationKind.TRAIN_EXECUTE.value,
        "input_digest": input_digest(update_payload),
        "input_payload": update_payload,
    }
    payload = round_state_to_json(state)
    if representation == "receipt":
        receipts = payload["receipts"]
        assert isinstance(receipts, list)
        receipts.append(operation)
    else:
        payload["stage"] = RoundStage.IN_DOUBT.value
        payload["pending_operation"] = {**operation, "expert_id": None}
        payload["in_doubt_reason"] = "generic reconciliation required"

    with pytest.raises(StateDecodeError):
        round_state_from_json(payload)


class _SelectiveFailureWorldModel:
    def __init__(
        self,
        delegate: WorldModel,
        *,
        failing_experts: frozenset[str],
        error_type: type[Exception] = RuntimeError,
    ) -> None:
        self._delegate = delegate
        self._failing_experts = failing_experts
        self._error_type = error_type
        self.calls: list[str] = []

    def imagine(
        self,
        task: TaskSpec,
        scene: SceneRecord,
        expert: ExpertSpec,
        actions: ActionChunk,
        *,
        operation_id: str = "",
    ) -> Rollout:
        self.calls.append(expert.expert_id)
        if expert.expert_id in self._failing_experts:
            raise self._error_type(" | ".join(_SENSITIVE_FRAGMENTS))
        return self._delegate.imagine(
            task,
            scene,
            expert,
            actions,
            operation_id=operation_id,
        )


class _InDoubtExecutor:
    def execute(
        self,
        task: TaskSpec,
        scene: SceneRecord,
        expert: ExpertSpec,
        actions: ActionChunk,
        *,
        operation_id: str,
    ) -> Trajectory:
        del task, scene, expert, actions, operation_id
        raise OperationInDoubtError(" | ".join(_SENSITIVE_FRAGMENTS))


class _FailingExecutor:
    def __init__(self, *, error_type: type[Exception] = RuntimeError) -> None:
        self._error_type = error_type

    def execute(
        self,
        task: TaskSpec,
        scene: SceneRecord,
        expert: ExpertSpec,
        actions: ActionChunk,
        *,
        operation_id: str,
    ) -> Trajectory:
        del task, scene, expert, actions, operation_id
        raise self._error_type(" | ".join(_SENSITIVE_FRAGMENTS))


class _RestartInterruption(RuntimeError):
    pass


class _InterruptAfterCandidatesStore:
    def __init__(self, delegate: JsonRunStore) -> None:
        self._delegate = delegate
        self._interrupted = False

    @property
    def run_dir(self) -> Path:
        return self._delegate.run_dir

    def save_state(self, state: RoundState) -> None:
        self._delegate.save_state(state)
        if state.stage is RoundStage.CANDIDATES_SCORED and not self._interrupted:
            self._interrupted = True
            raise _RestartInterruption(state.round_id)

    def read_state(self) -> RoundState | None:
        return self._delegate.read_state()

    def load_state(self, round_id: str) -> RoundState | None:
        return self._delegate.load_state(round_id)

    def claim_round(self, state: RoundState) -> None:
        self._delegate.claim_round(state)

    def write_artifact(self, round_id: str, name: str, metadata: JsonObject) -> Path:
        return self._delegate.write_artifact(round_id, name, metadata)

    def append_task(self, round_id: str, operation_id: str, task: TaskSpec) -> None:
        self._delegate.append_task(round_id, operation_id, task)

    def write_verified(self, trajectory: Trajectory, provenance: JsonObject) -> Path:
        return self._delegate.write_verified(trajectory, provenance)


def _pipeline(
    tmp_path: Path,
    *,
    label: DiagnosisLabel,
    score: float,
    execution_mode: ExecutionMode,
) -> tuple[EvoWorldPipeline, AdapterSet]:
    config = _config(tmp_path)
    scenario = DeterministicScenario(
        router_scores={"groot": 0.1, "pi05": 0.9},
        rollout_scores={"groot": 0.1, "pi05": 0.9},
        evaluation_score=score,
        evaluation_label=label,
    )
    run_dir = tmp_path / f"{execution_mode.value}-{label.value.lower()}"
    adapters = build_deterministic_adapters(config, run_dir, scenario)
    return EvoWorldPipeline(config, adapters, execution_mode=execution_mode), adapters


def _config(tmp_path: Path) -> EvoWorldConfig:
    experts = tuple(
        ExpertSpec(
            expert_id=expert_id,
            backend="local",
            task_families=("manipulation",),
            checkpoint=None,
            action_schema="joint_velocity",
        )
        for expert_id in ("groot", "pi05")
    )
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
        experts=experts,
    )

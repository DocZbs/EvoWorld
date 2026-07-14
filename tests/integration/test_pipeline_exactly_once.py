"""Crash-after-return tests for durable at-least-once operation recovery."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from evoworld.adapters.deterministic import (
    DeterministicScenario,
    build_deterministic_adapters,
)
from evoworld.config import EvoWorldConfig
from evoworld.contracts import (
    DiagnosisLabel,
    ExecutionMode,
    ExpertSpec,
    GroundingMode,
    JsonObject,
    OperationKind,
    RoundState,
    TaskSpec,
    Trajectory,
)
from evoworld.pipeline import EvoWorldPipeline
from evoworld.ports import OperationInDoubtError
from evoworld.storage import JsonRunStore


class ReturnedBeforeReceipt(RuntimeError):
    pass


class CrashBeforeReceiptStore:
    """Drop the first state containing the selected returned-operation receipt."""

    def __init__(self, delegate: JsonRunStore, target: OperationKind) -> None:
        self._delegate = delegate
        self._target = target
        self.operation_id: str | None = None

    @property
    def run_dir(self) -> Path:
        return self._delegate.run_dir

    def save_state(self, state: RoundState) -> None:
        receipt = next((item for item in state.receipts if item.kind is self._target), None)
        if receipt is not None and self.operation_id is None:
            self.operation_id = receipt.operation_id
            raise ReturnedBeforeReceipt(receipt.operation_id)
        self._delegate.save_state(state)

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


@pytest.mark.parametrize(
    "target",
    (
        OperationKind.TASK,
        OperationKind.ANCHOR,
        OperationKind.ROUTE,
        OperationKind.ACT,
        OperationKind.IMAGINE,
        OperationKind.SCORE,
        OperationKind.EXECUTE,
        OperationKind.EVALUATE,
        OperationKind.TRAIN_PLAN,
        OperationKind.MEMORY_APPEND,
    ),
)
def test_resume_reuses_operation_id_without_repeating_effect(
    tmp_path: Path, target: OperationKind
) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / target.value
    adapters = build_deterministic_adapters(config, run_dir)
    store = CrashBeforeReceiptStore(adapters.store, target)  # type: ignore[arg-type]
    pipeline = EvoWorldPipeline(config, replace(adapters, store=store))

    with pytest.raises(ReturnedBeforeReceipt):
        pipeline.run(1)
    restarted = build_deterministic_adapters(config, run_dir)
    result = EvoWorldPipeline(config, restarted).resume()

    assert result is not None and result.verified
    assert store.operation_id is not None
    if target is not OperationKind.MEMORY_APPEND:
        assert adapters.receipts.effect_count(target.value, store.operation_id) == 1
        assert restarted.receipts.effect_count(target.value, store.operation_id) == 0
    memory = store.run_dir / "memory.jsonl"
    assert len(memory.read_text(encoding="utf-8").splitlines()) == 1


def test_execute_mode_runs_targeted_training_once_after_return_crash(tmp_path: Path) -> None:
    config = _config(tmp_path)
    adapters = build_deterministic_adapters(
        config,
        tmp_path / "execute",
        DeterministicScenario(
            evaluation_score=0.2,
            evaluation_label=DiagnosisLabel.PF,
        ),
    )
    store = CrashBeforeReceiptStore(
        adapters.store,
        OperationKind.TRAIN_EXECUTE,  # type: ignore[arg-type]
    )
    pipeline = EvoWorldPipeline(
        config, replace(adapters, store=store), execution_mode=ExecutionMode.EXECUTE
    )

    with pytest.raises(ReturnedBeforeReceipt):
        pipeline.run(1)
    result = pipeline.resume()

    assert result is not None and not result.verified
    assert store.operation_id is not None
    assert adapters.receipts.effect_count("train_execute", store.operation_id) == 1
    assert not (store.run_dir / "verified").exists()


def test_verified_none_target_never_executes_trainer_even_in_execute_mode(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    adapters = build_deterministic_adapters(config, tmp_path / "none")

    result = EvoWorldPipeline(config, adapters, execution_mode=ExecutionMode.EXECUTE).run(1)[0]

    assert result.verified
    assert all(
        receipt.kind is not OperationKind.TRAIN_EXECUTE
        for receipt in adapters.store.read_state().receipts  # type: ignore[union-attr]
    )


def test_fresh_process_adapter_reuses_persisted_physical_execution_receipt(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / "restart"
    first = build_deterministic_adapters(config, run_dir)
    crashing_store = CrashBeforeReceiptStore(
        first.store,
        OperationKind.EXECUTE,  # type: ignore[arg-type]
    )

    with pytest.raises(ReturnedBeforeReceipt):
        EvoWorldPipeline(config, replace(first, store=crashing_store)).run(1)
    assert crashing_store.operation_id is not None

    restarted = build_deterministic_adapters(config, run_dir)
    result = EvoWorldPipeline(config, restarted).resume()

    assert result is not None and result.verified
    assert restarted.receipts.effect_count("execute", crashing_store.operation_id) == 0


def test_resume_persists_in_doubt_and_blocks_physical_replay(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / "in-doubt"
    interrupted = build_deterministic_adapters(
        config,
        run_dir,
        DeterministicScenario(execution_interrupt="after_effect"),
    )
    with pytest.raises(RuntimeError, match="BackendError: execute failed"):
        EvoWorldPipeline(config, interrupted).run_round("round-000001")

    fresh = build_deterministic_adapters(config, run_dir)
    with pytest.raises(OperationInDoubtError):
        EvoWorldPipeline(config, fresh).resume()

    state = fresh.store.load_state("round-000001")
    assert state is not None
    assert state.stage.value == "in_doubt"
    assert state.pending_operation is not None
    assert state.in_doubt_reason


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

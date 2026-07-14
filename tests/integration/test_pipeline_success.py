"""Integration coverage for the durable four-stage EvoWorld pipeline."""

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
from evoworld.config import EvoWorldConfig
from evoworld.contracts import (
    ExpertSpec,
    GroundingMode,
    JsonObject,
    RoundStage,
    RoundState,
    TaskSpec,
    Trajectory,
)
from evoworld.pipeline import EvoWorldPipeline
from evoworld.storage import JsonRunStore


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
        experts=(
            ExpertSpec(
                expert_id="groot",
                backend="local",
                task_families=("manipulation",),
                checkpoint=None,
                action_schema="joint_velocity",
            ),
            ExpertSpec(
                expert_id="pi05",
                backend="local",
                task_families=("manipulation",),
                checkpoint=None,
                action_schema="joint_velocity",
            ),
        ),
    )


def test_successful_round_persists_every_method_stage(tmp_path: Path) -> None:
    config = _config(tmp_path)
    adapters = build_deterministic_adapters(config, tmp_path / "run")
    pipeline = EvoWorldPipeline(config, adapters)

    result = pipeline.run(rounds=1)[0]

    assert result.verified is True
    round_dir = pipeline.store.run_dir / "rounds" / result.round_id
    assert {path.name for path in round_dir.iterdir()} >= {
        "grounding.json",
        "candidates.json",
        "trajectory.json",
        "diagnosis.json",
        "update.json",
    }
    assert tuple(score.expert_id for score in result.scores) == tuple(
        candidate["expert"]["expert_id"]
        for candidate in json.loads((round_dir / "candidates.json").read_text(encoding="utf-8"))[
            "candidates"
        ]
    )
    assert adapters.store.load_state(result.round_id).stage is RoundStage.COMPLETE
    assert len((pipeline.store.run_dir / "memory.jsonl").read_text().splitlines()) == 1
    tasks = (pipeline.store.run_dir / "tasks.jsonl").read_text().splitlines()
    assert len(tasks) == 1
    assert json.loads(tasks[0])["round_id"] == result.round_id
    rollout_root = round_dir / "rollouts"
    assert {path.name for path in rollout_root.iterdir()} == {"groot", "pi05"}
    assert all(
        {path.name for path in expert_dir.iterdir()}
        >= {"act.receipt.json", "imagine.receipt.json", "score.receipt.json"}
        for expert_dir in rollout_root.iterdir()
    )
    verified = pipeline.store.run_dir / "verified" / result.trajectory.trajectory_id
    assert {path.name for path in verified.iterdir()} == {
        "trajectory.json",
        "provenance.json",
    }


def test_pipeline_calls_every_adapter_in_method_order_and_never_executes_training(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    adapters = build_deterministic_adapters(config, tmp_path / "run")
    traced, calls = _trace(adapters)

    result = EvoWorldPipeline(config, traced).run(1)[0]

    candidate_calls = [
        call
        for expert_id in (score.expert_id for score in result.scores)
        for call in (
            f"policies_{expert_id}.act",
            "world_model.imagine",
            "rollout_scorer.score",
        )
    ]
    assert [entry[0] for entry in calls.mock_calls] == [
        "task_source.next_task",
        "scene_anchor.anchor",
        "router.shortlist",
        "memory.lookup",
        *candidate_calls,
        "executor.execute",
        "evaluator.evaluate",
        "trainer.plan",
        "memory.append",
    ]
    assert not traced.trainer.execute.called


def test_failed_candidate_is_persisted_without_aborting_valid_candidate(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    adapters = build_deterministic_adapters(
        config,
        tmp_path / "run",
        DeterministicScenario(failing_rollout_experts=frozenset({"groot"})),
    )

    result = EvoWorldPipeline(config, adapters).run(1)[0]

    failed = next(score for score in result.scores if score.expert_id == "groot")
    assert failed.valid is False
    assert failed.rollout_id is None
    assert failed.error == "BackendError: imagine failed"
    assert result.trajectory.actions.expert_id == "pi05"


@pytest.mark.parametrize(
    "interrupted_stage",
    (
        RoundStage.GROUNDED,
        RoundStage.CANDIDATES_SCORED,
        RoundStage.EXECUTED,
        RoundStage.COMPLETE,
    ),
)
def test_resume_skips_every_completed_side_effect(
    tmp_path: Path, interrupted_stage: RoundStage
) -> None:
    config = _config(tmp_path)
    adapters = build_deterministic_adapters(config, tmp_path / interrupted_stage.value)
    store = InterruptAfterSaveStore(cast(JsonRunStore, adapters.store), interrupted_stage)
    traced, calls = _trace(replace(adapters, store=store))
    pipeline = EvoWorldPipeline(config, traced)

    with pytest.raises(ScriptedInterruption, match=interrupted_stage.value):
        pipeline.run(1)
    persisted = store.read_state()
    assert persisted is not None
    assert persisted.stage is interrupted_stage
    calls_before_resume = tuple(calls.mock_calls)
    writes_before_resume = tuple(store.artifact_writes)

    resumed = pipeline.resume()

    if interrupted_stage is RoundStage.COMPLETE:
        assert resumed is None
    else:
        assert resumed is not None
        assert resumed.round_id == persisted.round_id
        assert store.read_state().stage is RoundStage.COMPLETE
    completed_prefix = _completed_adapter_prefix(interrupted_stage)
    assert (
        tuple(calls.mock_calls[: len(completed_prefix)])
        == calls_before_resume[: len(completed_prefix)]
    )
    before_names = [entry[0] for entry in calls_before_resume]
    after_names = [entry[0] for entry in calls.mock_calls]
    for method_name in set(completed_prefix):
        assert after_names.count(method_name) == before_names.count(method_name)
    assert len(store.artifact_writes) == len(set(store.artifact_writes))
    assert tuple(store.artifact_writes[: len(writes_before_resume)]) == writes_before_resume
    memory_path = store.run_dir / "memory.jsonl"
    if memory_path.exists():
        record_ids = [
            json.loads(line)["record_id"] for line in memory_path.read_text().splitlines()
        ]
        assert len(record_ids) == len(set(record_ids))


def test_resume_returns_none_without_state_and_run_rejects_overwrite(tmp_path: Path) -> None:
    config = _config(tmp_path)
    pipeline = EvoWorldPipeline(config, build_deterministic_adapters(config, tmp_path / "run"))

    assert pipeline.resume() is None
    assert pipeline.run(0) == ()
    pipeline.run_round("round-custom")

    with pytest.raises(FileExistsError, match="round-custom"):
        pipeline.run_round("round-custom")
    with pytest.raises(ValueError, match="nonnegative"):
        pipeline.run(-1)
    with pytest.raises(TypeError, match="integer"):
        pipeline.run(cast(int, True))


def test_multiple_rounds_keep_distinct_verified_trajectory_provenance(tmp_path: Path) -> None:
    config = _config(tmp_path)
    pipeline = EvoWorldPipeline(config, build_deterministic_adapters(config, tmp_path / "multi"))

    results = pipeline.run(2)

    assert tuple(result.round_id for result in results) == (
        "round-000000",
        "round-000001",
    )
    assert len({result.trajectory.trajectory_id for result in results}) == 2


def test_resume_retries_incomplete_final_stage_idempotently(tmp_path: Path) -> None:
    config = _config(tmp_path)
    adapters = build_deterministic_adapters(config, tmp_path / "run")
    store = InterruptAfterSaveStore(
        cast(JsonRunStore, adapters.store), RoundStage.COMPLETE, before_save=True
    )
    traced, calls = _trace(replace(adapters, store=store))
    pipeline = EvoWorldPipeline(config, traced)

    with pytest.raises(ScriptedInterruption, match=RoundStage.COMPLETE.value):
        pipeline.run(1)
    persisted = store.read_state()
    assert persisted is not None and persisted.stage is RoundStage.EXECUTED
    round_dir = store.run_dir / "rounds" / persisted.round_id
    first_diagnosis = (round_dir / "diagnosis.json").read_bytes()
    first_update = (round_dir / "update.json").read_bytes()

    resumed = pipeline.resume()

    assert resumed is not None and resumed.verified is True
    assert (round_dir / "diagnosis.json").read_bytes() == first_diagnosis
    assert (round_dir / "update.json").read_bytes() == first_update
    records = (store.run_dir / "memory.jsonl").read_text().splitlines()
    assert len(records) == 1
    assert [entry[0] for entry in calls.mock_calls].count("executor.execute") == 1


class ScriptedInterruption(RuntimeError):
    """Test-only crash injected after one durable state transition."""


class InterruptAfterSaveStore:
    """Run-store wrapper that interrupts exactly once after a selected save."""

    def __init__(
        self,
        delegate: JsonRunStore,
        stage: RoundStage,
        *,
        before_save: bool = False,
    ) -> None:
        self._delegate = delegate
        self._stage = stage
        self._before_save = before_save
        self._interrupted = False
        self.artifact_writes: list[str] = []

    @property
    def run_dir(self) -> Path:
        return self._delegate.run_dir

    def write_artifact(self, round_id: str, name: str, metadata: JsonObject) -> Path:
        self.artifact_writes.append(name)
        return self._delegate.write_artifact(round_id, name, metadata)

    def save_state(self, state: RoundState) -> None:
        if self._before_save and state.stage is self._stage and not self._interrupted:
            self._interrupted = True
            raise ScriptedInterruption(state.stage.value)
        self._delegate.save_state(state)
        if state.stage is self._stage and not self._interrupted:
            self._interrupted = True
            raise ScriptedInterruption(state.stage.value)

    def load_state(self, round_id: str) -> RoundState | None:
        return self._delegate.load_state(round_id)

    def read_state(self) -> RoundState | None:
        return self._delegate.read_state()

    def claim_round(self, state: RoundState) -> None:
        self._delegate.claim_round(state)

    def append_task(self, round_id: str, operation_id: str, task: object) -> None:
        self._delegate.append_task(round_id, operation_id, cast(TaskSpec, task))

    def write_verified(self, trajectory: object, provenance: JsonObject) -> Path:
        return self._delegate.write_verified(cast(Trajectory, trajectory), provenance)


def _trace(adapters: AdapterSet) -> tuple[AdapterSet, Mock]:
    calls = Mock()

    def attached(name: str, value: object) -> Mock:
        wrapped = Mock(wraps=value)
        calls.attach_mock(wrapped, name)
        return wrapped

    policies = {
        expert_id: attached(f"policies_{expert_id}", policy)
        for expert_id, policy in adapters.policies.items()
    }
    traced = replace(
        adapters,
        task_source=attached("task_source", adapters.task_source),
        scene_anchor=attached("scene_anchor", adapters.scene_anchor),
        router=attached("router", adapters.router),
        policies=policies,
        world_model=attached("world_model", adapters.world_model),
        rollout_scorer=attached("rollout_scorer", adapters.rollout_scorer),
        executor=attached("executor", adapters.executor),
        evaluator=attached("evaluator", adapters.evaluator),
        trainer=attached("trainer", adapters.trainer),
        memory=attached("memory", adapters.memory),
    )
    return traced, calls


def _completed_adapter_prefix(stage: RoundStage) -> tuple[str, ...]:
    prefix = ["task_source.next_task", "scene_anchor.anchor"]
    if stage in {
        RoundStage.CANDIDATES_SCORED,
        RoundStage.EXECUTED,
        RoundStage.COMPLETE,
    }:
        prefix.extend(
            (
                "router.shortlist",
                "memory.lookup",
                "policies_groot.act",
                "world_model.imagine",
                "rollout_scorer.score",
                "policies_pi05.act",
                "world_model.imagine",
                "rollout_scorer.score",
            )
        )
    if stage in {RoundStage.EXECUTED, RoundStage.COMPLETE}:
        prefix.append("executor.execute")
    if stage is RoundStage.COMPLETE:
        prefix.extend(("evaluator.evaluate", "trainer.plan", "memory.append"))
    return tuple(prefix)

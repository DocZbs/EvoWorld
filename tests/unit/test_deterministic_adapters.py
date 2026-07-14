"""Tests for the local no-backend EvoWorld adapter bundle."""

from __future__ import annotations

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from math import isfinite
from pathlib import Path

import pytest

from evoworld.adapters.deterministic import (
    AdapterSet,
    DeterministicExecutionError,
    DeterministicInterruption,
    DeterministicRolloutError,
    DeterministicScenario,
    DeterministicTrainingError,
    build_deterministic_adapters,
)
from evoworld.config import EvoWorldConfig
from evoworld.contracts import (
    DiagnosisLabel,
    Diagnosis,
    ExecutionMode,
    ExperienceRecord,
    ExpertSpec,
    GroundingMode,
    TaskSpec,
    UpdateDecision,
    UpdateTarget,
)
from evoworld.ports import (
    Executor,
    ExperienceMemory,
    ExpertPolicy,
    RolloutScorer,
    OperationInDoubtError,
    OperationConflictError,
    Router,
    RunStore,
    SceneAnchor,
    TaskSource,
    Trainer,
    TrajectoryEvaluator,
    WorldModel,
)


def _config(seed: int = 7) -> EvoWorldConfig:
    return EvoWorldConfig(
        schema_version=1,
        seed=seed,
        run_root=Path("runs").resolve(),
        mode="dry_run",
        grounding_mode=GroundingMode.NEAREST,
        grounding_temperature=0.5,
        router_weight=0.25,
        reliability_threshold=0.7,
        rollout_horizon=3,
        executor=None,
        experts=(_expert("pi05"), _expert("groot")),
    )


def _expert(expert_id: str) -> ExpertSpec:
    return ExpertSpec(
        expert_id=expert_id,
        backend="local",
        task_families=("manipulation",),
        checkpoint=None,
        action_schema="joint_velocity",
    )


def _update(mode: ExecutionMode = ExecutionMode.PLAN) -> UpdateDecision:
    return UpdateDecision(
        update_id="update-1",
        target=UpdateTarget.EXPERT_POLICY,
        task_id="task-1",
        scene_id="scene-1",
        trajectory_id="trajectory-1",
        expert_id="pi05",
        execution_mode=mode,
    )


def test_same_seed_produces_byte_identical_values_independent_of_run_root(tmp_path: Path) -> None:
    first = build_deterministic_adapters(_config(9), tmp_path / "a")
    second = build_deterministic_adapters(_config(9), tmp_path / "b")
    task = first.task_source.next_task()
    scene = first.scene_anchor.anchor(task)

    first_actions = first.policies["pi05"].act(task, scene)
    second_actions = second.policies["pi05"].act(task, scene)
    first_rollout = first.world_model.imagine(task, scene, _expert("pi05"), first_actions)
    second_rollout = second.world_model.imagine(task, scene, _expert("pi05"), second_actions)

    assert first_actions == second_actions
    assert first_rollout.rollout_id == second_rollout.rollout_id
    assert first_rollout.metrics == second_rollout.metrics
    assert first_rollout.frame_paths[0].read_bytes() == second_rollout.frame_paths[0].read_bytes()


def test_different_seed_changes_stochastic_action_values(tmp_path: Path) -> None:
    first = build_deterministic_adapters(_config(1), tmp_path / "one")
    second = build_deterministic_adapters(_config(2), tmp_path / "two")
    task = TaskSpec("task-1", "place the object")
    scene = first.scene_anchor.anchor(task)

    assert first.policies["pi05"].act(task, scene) != second.policies["pi05"].act(task, scene)


def test_adapter_set_implements_every_core_port(tmp_path: Path) -> None:
    adapters = build_deterministic_adapters(_config(), tmp_path / "run")

    assert isinstance(adapters, AdapterSet)
    assert isinstance(adapters.task_source, TaskSource)
    assert isinstance(adapters.scene_anchor, SceneAnchor)
    assert isinstance(adapters.router, Router)
    assert isinstance(adapters.policies["pi05"], ExpertPolicy)
    assert isinstance(adapters.world_model, WorldModel)
    assert isinstance(adapters.rollout_scorer, RolloutScorer)
    assert isinstance(adapters.executor, Executor)
    assert isinstance(adapters.evaluator, TrajectoryEvaluator)
    assert isinstance(adapters.trainer, Trainer)
    assert isinstance(adapters.memory, ExperienceMemory)
    assert isinstance(adapters.store, RunStore)
    assert adapters.store.run_dir.is_absolute()
    assert adapters.store.write_artifact("round-1", "adapters", {"mode": "dry_run"}).is_file()


def test_input_expert_order_does_not_change_routing_or_scores(tmp_path: Path) -> None:
    config = _config()
    reverse = EvoWorldConfig(
        **{field: getattr(config, field) for field in config.__dataclass_fields__}
        | {"experts": tuple(reversed(config.experts))}
    )
    first = build_deterministic_adapters(config, tmp_path / "first")
    second = build_deterministic_adapters(reverse, tmp_path / "second")
    task = first.task_source.next_task()
    scene = first.scene_anchor.anchor(task)

    assert [expert.expert_id for expert in first.router.shortlist(task, scene, first.memory)] == [
        expert.expert_id for expert in second.router.shortlist(task, scene, second.memory)
    ]


def test_successful_memory_changes_the_deterministic_shortlist(tmp_path: Path) -> None:
    adapters = build_deterministic_adapters(
        _config(), tmp_path / "run", DeterministicScenario(memory_weight=1.0)
    )
    task = adapters.task_source.next_task()
    scene = adapters.scene_anchor.anchor(task)
    rollout = adapters.world_model.imagine(
        task, scene, _expert("groot"), adapters.policies["groot"].act(task, scene)
    )
    score = adapters.rollout_scorer.score(task, rollout)
    adapters.memory.append(
        ExperienceRecord(
            record_id="experience-groot",
            task_id=task.task_id,
            scene_id=scene.scene_id,
            expert_id="groot",
            score=replace(score, joint_score=1.0),
            diagnosis=Diagnosis(1.0, DiagnosisLabel.NONE, "verified"),
        )
    )

    assert adapters.router.shortlist(task, scene, adapters.memory)[0].expert_id == "groot"


def test_actions_change_with_stable_scene_observation_content(tmp_path: Path) -> None:
    adapters = build_deterministic_adapters(_config(), tmp_path / "run")
    task = adapters.task_source.next_task()
    first_scene = adapters.scene_anchor.anchor(task)
    alternate_observation = tmp_path / "alternate.txt"
    alternate_observation.write_text("different scene", encoding="utf-8")
    second_scene = replace(
        first_scene, scene_id="scene-2", observation_paths=(alternate_observation,)
    )

    assert adapters.policies["pi05"].act(task, first_scene) != adapters.policies["pi05"].act(
        task, second_scene
    )


def test_trajectory_identity_and_frame_change_with_observation_content(tmp_path: Path) -> None:
    adapters = build_deterministic_adapters(_config(), tmp_path / "run")
    task = adapters.task_source.next_task()
    first_scene = adapters.scene_anchor.anchor(task)
    alternate_observation = tmp_path / "alternate.txt"
    alternate_observation.write_text("different scene", encoding="utf-8")
    second_scene = replace(first_scene, observation_paths=(alternate_observation,))

    first = adapters.executor.execute(task, first_scene, _expert("pi05"))
    first_bytes = first.frame_paths[0].read_bytes()
    second = adapters.executor.execute(task, second_scene, _expert("pi05"))

    assert first.trajectory_id != second.trajectory_id
    assert first.frame_paths[0] != second.frame_paths[0]
    assert first.frame_paths[0].read_bytes() == first_bytes
    assert first.frame_paths[0].is_file() and second.frame_paths[0].is_file()


def test_trajectory_identity_is_independent_of_output_root(tmp_path: Path) -> None:
    first = build_deterministic_adapters(_config(9), tmp_path / "a")
    second = build_deterministic_adapters(_config(9), tmp_path / "b")
    first_task = first.task_source.next_task()
    second_task = second.task_source.next_task()

    first_trajectory = first.executor.execute(
        first_task, first.scene_anchor.anchor(first_task), _expert("pi05")
    )
    second_trajectory = second.executor.execute(
        second_task, second.scene_anchor.anchor(second_task), _expert("pi05")
    )

    assert first_trajectory.trajectory_id == second_trajectory.trajectory_id
    assert (
        first_trajectory.frame_paths[0].read_bytes()
        == second_trajectory.frame_paths[0].read_bytes()
    )


def test_rollout_identity_includes_the_full_action_payload(tmp_path: Path) -> None:
    adapters = build_deterministic_adapters(_config(), tmp_path / "run")
    task = adapters.task_source.next_task()
    scene = adapters.scene_anchor.anchor(task)
    actions = adapters.policies["pi05"].act(task, scene)
    changed_actions = replace(actions, actions=((0.0,), *actions.actions[1:]))

    first = adapters.world_model.imagine(task, scene, _expert("pi05"), actions)
    second = adapters.world_model.imagine(task, scene, _expert("pi05"), changed_actions)

    assert first.rollout_id != second.rollout_id


def test_world_model_rejects_actions_for_another_expert_or_schema(tmp_path: Path) -> None:
    adapters = build_deterministic_adapters(_config(), tmp_path / "run")
    task = adapters.task_source.next_task()
    scene = adapters.scene_anchor.anchor(task)
    actions = adapters.policies["pi05"].act(task, scene)

    with pytest.raises(ValueError, match="expert_id"):
        adapters.world_model.imagine(task, scene, _expert("groot"), actions)
    with pytest.raises(ValueError, match="schema"):
        adapters.world_model.imagine(
            task, scene, _expert("pi05"), replace(actions, schema_name="cartesian_pose")
        )


def test_scenario_can_fail_one_rollout(tmp_path: Path) -> None:
    adapters = build_deterministic_adapters(
        _config(),
        tmp_path / "run",
        DeterministicScenario(failing_rollout_experts=frozenset({"pi05"})),
    )
    task = adapters.task_source.next_task()
    scene = adapters.scene_anchor.anchor(task)

    with pytest.raises(DeterministicRolloutError, match="pi05"):
        adapters.world_model.imagine(
            task, scene, _expert("pi05"), adapters.policies["pi05"].act(task, scene)
        )


def test_scripted_diagnosis_and_execution_failure_are_reproducible(tmp_path: Path) -> None:
    scenario = DeterministicScenario(
        evaluation_score=0.2,
        evaluation_label=DiagnosisLabel.PF,
        execution_failure_experts=frozenset({"pi05"}),
    )
    adapters = build_deterministic_adapters(_config(), tmp_path / "run", scenario)
    task = adapters.task_source.next_task()
    scene = adapters.scene_anchor.anchor(task)
    actions = adapters.policies["groot"].act(task, scene)
    rollout = adapters.world_model.imagine(task, scene, _expert("groot"), actions)
    trajectory = adapters.executor.execute(task, scene, _expert("groot"))

    assert (
        adapters.evaluator.evaluate(task, scene, trajectory, adapters.memory).label
        is DiagnosisLabel.PF
    )
    assert adapters.rollout_scorer.score(task, rollout).valid
    successful_frames = set((adapters.store.run_dir / "frames").glob("trajectory-*.txt"))
    with pytest.raises(DeterministicExecutionError, match="pi05") as first_failure:
        adapters.executor.execute(task, scene, _expert("pi05"))
    with pytest.raises(OperationInDoubtError, match="claimed"):
        adapters.executor.execute(task, scene, _expert("pi05"))

    assert first_failure.value.identity
    assert set((adapters.store.run_dir / "frames").glob("trajectory-*.txt")) == successful_frames


def test_trainer_is_plan_only_unless_the_update_explicitly_allows_execution(tmp_path: Path) -> None:
    trainer = build_deterministic_adapters(_config(), tmp_path / "run").trainer

    assert trainer.plan(_update()) == _update()
    with pytest.raises(DeterministicTrainingError, match="execute"):
        trainer.execute(_update())
    assert trainer.execute(_update(ExecutionMode.EXECUTE)) == _update(ExecutionMode.EXECUTE)


def test_physical_operation_id_returns_one_receipted_effect(tmp_path: Path) -> None:
    adapters = build_deterministic_adapters(_config(), tmp_path / "run")
    task = adapters.task_source.next_task(operation_id="round-1:task")
    scene = adapters.scene_anchor.anchor(task, operation_id="round-1:anchor")
    operation_id = "round-1:execute:pi05"

    first = adapters.executor.execute(task, scene, _expert("pi05"), operation_id=operation_id)
    second = adapters.executor.execute(task, scene, _expert("pi05"), operation_id=operation_id)

    assert second == first
    assert adapters.receipts.effect_count("execute", operation_id) == 1


def test_fresh_adapter_reads_disk_receipt_and_rejects_changed_input(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    first = build_deterministic_adapters(
        _config(), run_dir, DeterministicScenario(task_id="task-original")
    )
    task = first.task_source.next_task(operation_id="round-1:task")

    fresh = build_deterministic_adapters(
        _config(), run_dir, DeterministicScenario(task_id="task-replacement")
    )
    assert fresh.task_source.next_task(operation_id="round-1:task") == task
    scene = fresh.scene_anchor.anchor(task, operation_id="round-1:anchor")
    with pytest.raises(OperationConflictError, match="different input"):
        fresh.scene_anchor.anchor(
            TaskSpec("different-task", "different"), operation_id="round-1:anchor"
        )
    assert scene.scene_id == "scene-000001"


@pytest.mark.parametrize("window", ("after_claim", "after_effect"))
def test_fresh_executor_never_replays_an_in_doubt_physical_claim(
    tmp_path: Path, window: str
) -> None:
    run_dir = tmp_path / window
    interrupted = build_deterministic_adapters(
        _config(), run_dir, DeterministicScenario(execution_interrupt=window)
    )
    task = interrupted.task_source.next_task()
    scene = interrupted.scene_anchor.anchor(task)
    actions = interrupted.policies["pi05"].act(task, scene)
    operation_id = "round-1:execute:pi05"
    with pytest.raises(DeterministicInterruption):
        interrupted.executor.execute(
            task, scene, _expert("pi05"), actions, operation_id=operation_id
        )

    fresh = build_deterministic_adapters(_config(), run_dir)
    with pytest.raises(OperationInDoubtError, match="claimed"):
        fresh.executor.execute(task, scene, _expert("pi05"), actions, operation_id=operation_id)
    assert fresh.receipts.effect_count("execute", operation_id) == 0


def test_concurrent_duplicate_execution_has_one_physical_effect(tmp_path: Path) -> None:
    adapters = build_deterministic_adapters(_config(), tmp_path / "run")
    task = adapters.task_source.next_task()
    scene = adapters.scene_anchor.anchor(task)
    actions = adapters.policies["pi05"].act(task, scene)
    operation_id = "round-1:execute:pi05"

    def execute():
        return adapters.executor.execute(
            task, scene, _expert("pi05"), actions, operation_id=operation_id
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(lambda _index: execute(), range(2)))

    assert results[0] == results[1]
    assert adapters.receipts.effect_count("execute", operation_id) == 1


def test_frame_artifacts_remain_contained_when_a_child_is_a_symlink(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    outside = tmp_path / "outside"
    run_dir.mkdir()
    outside.mkdir()
    try:
        (run_dir / "frames").symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")

    with pytest.raises(ValueError, match="artifact path"):
        build_deterministic_adapters(_config(), run_dir)

    assert not tuple(outside.iterdir())


@pytest.mark.parametrize("directory", ("adapter_receipts", "adapter_claims"))
@pytest.mark.parametrize("target_kind", ("outside", "inside"))
def test_operation_receipt_and_claim_directories_reject_all_symlinks(
    tmp_path: Path, directory: str, target_kind: str
) -> None:
    adapters = build_deterministic_adapters(_config(), tmp_path / f"{directory}-{target_kind}")
    task = scene = actions = None
    if directory == "adapter_claims":
        task = adapters.task_source.next_task(operation_id="setup:task")
        scene = adapters.scene_anchor.anchor(task, operation_id="setup:anchor")
        actions = adapters.policies["pi05"].act(task, scene, operation_id="setup:act:pi05")
    target = tmp_path / "outside" if target_kind == "outside" else adapters.store.run_dir / "inside"
    target.mkdir(exist_ok=True)
    link = adapters.store.run_dir / directory
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")

    before = tuple(target.iterdir())
    with pytest.raises(ValueError, match="symlink"):
        if directory == "adapter_receipts":
            adapters.task_source.next_task(operation_id="round-1:task")
        else:
            assert task is not None and scene is not None and actions is not None
            adapters.executor.execute(
                task,
                scene,
                _expert("pi05"),
                actions,
                operation_id="round-1:execute:pi05",
            )
    assert tuple(target.iterdir()) == before


def test_adapters_do_not_require_global_rng_environment_or_network(tmp_path: Path) -> None:
    adapters = build_deterministic_adapters(_config(), tmp_path / "run")
    task = adapters.task_source.next_task()
    scene = adapters.scene_anchor.anchor(task)

    assert adapters.policies["pi05"].act(task, scene).actions


def test_scenario_copies_mutable_inputs_and_normalizes_finite_real_scores() -> None:
    failures = {"pi05"}
    scores = {"pi05": 1}
    scenario = DeterministicScenario(
        failing_rollout_experts=failures,
        router_scores=scores,
    )
    failures.add("groot")
    scores["groot"] = 0.1

    assert scenario.failing_rollout_experts == frozenset({"pi05"})
    assert dict(scenario.router_scores) == {"pi05": 1.0}
    assert all(
        isinstance(score, float) and isfinite(score) for score in scenario.router_scores.values()
    )
    with pytest.raises(ValueError, match="finite"):
        DeterministicScenario(router_scores={"pi05": float("inf")})
    with pytest.raises(ValueError, match="number"):
        DeterministicScenario(rollout_scores={"pi05": True})
    with pytest.raises(ValueError, match="blank"):
        DeterministicScenario(execution_failure_experts=frozenset({" "}))

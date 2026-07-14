"""Deterministic, local implementations of the EvoWorld adapter ports."""

from __future__ import annotations

import json
import os
import random
import tempfile
import threading
import fcntl
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterator, Mapping, TypeVar, cast

from evoworld.codec import (
    action_to_json,
    action_from_json,
    canonical_json,
    decode_json_object,
    diagnosis_from_json,
    diagnosis_to_json,
    expert_from_json,
    expert_to_json,
    input_digest,
    rollout_from_json,
    rollout_to_json,
    scene_from_json,
    scene_to_json,
    score_from_json,
    score_to_json,
    task_to_json,
    task_from_json,
    trajectory_from_json,
    trajectory_to_json,
    update_from_json,
    update_to_json,
)
from evoworld.config import EvoWorldConfig, config_digest, resolved_manifest, validate_config
from evoworld.contracts import (
    ActionChunk,
    CandidateScore,
    Diagnosis,
    DiagnosisLabel,
    ExecutionMode,
    ExpertSpec,
    JsonObject,
    JsonValue,
    Rollout,
    SceneRecord,
    TaskSpec,
    Trajectory,
    UpdateDecision,
)
from evoworld.ports import (
    Executor,
    ExperienceMemory,
    ExpertPolicy,
    RolloutScorer,
    Router,
    RunStore,
    SceneAnchor,
    TaskSource,
    Trainer,
    TrajectoryEvaluator,
    WorldModel,
    OperationConflictError,
    OperationInDoubtError,
)
from evoworld.routing import MemoryAwareRouter
from evoworld.storage import JsonRunStore, JsonlExperienceMemory, atomic_write_json

T = TypeVar("T")


class DeterministicAdapterError(RuntimeError):
    """Base exception carrying the deterministic identity that caused an error."""

    def __init__(self, message: str, *, expert_id: str | None = None, identity: str) -> None:
        self.expert_id = expert_id
        self.identity = identity
        super().__init__(message)


class DeterministicRolloutError(DeterministicAdapterError):
    """Raised when the scenario scripts an imagined rollout failure."""


class DeterministicExecutionError(DeterministicAdapterError):
    """Raised when the scenario scripts an execution failure."""


class DeterministicTrainingError(DeterministicAdapterError):
    """Raised when a plan-only update is asked to execute."""


class DeterministicInterruption(RuntimeError):
    """Test-only interruption at a physical execution durability boundary."""


@dataclass(frozen=True, slots=True)
class DeterministicScenario:
    """Explicit local inputs used to exercise normal and failure pipeline paths."""

    task_id: str = "task-000001"
    instruction: str = "place the object in the target area"
    scene_id: str = "scene-000001"
    router_scores: Mapping[str, float] = field(default_factory=dict)
    rollout_scores: Mapping[str, float] = field(default_factory=dict)
    memory_weight: float = 0.5
    evaluation_score: float = 0.9
    evaluation_label: DiagnosisLabel = DiagnosisLabel.NONE
    evaluation_reason: str = "deterministic local evaluation"
    failing_rollout_experts: frozenset[str] = frozenset()
    execution_failure_experts: frozenset[str] = frozenset()
    execution_interrupt: str | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.task_id, "task_id"),
            (self.instruction, "instruction"),
            (self.scene_id, "scene_id"),
            (self.evaluation_reason, "evaluation_reason"),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be blank")
        object.__setattr__(
            self, "router_scores", _normalized_scores(self.router_scores, "router_scores")
        )
        object.__setattr__(
            self, "rollout_scores", _normalized_scores(self.rollout_scores, "rollout_scores")
        )
        for name in ("failing_rollout_experts", "execution_failure_experts"):
            values = frozenset(getattr(self, name))
            if any(not isinstance(expert_id, str) or not expert_id.strip() for expert_id in values):
                raise ValueError(f"{name} values must be nonblank strings")
            object.__setattr__(self, name, values)
        if not _unit_real(self.evaluation_score):
            raise ValueError("evaluation_score must be between 0 and 1")
        object.__setattr__(self, "evaluation_score", float(self.evaluation_score))
        if not _unit_real(self.memory_weight):
            raise ValueError("memory_weight must be a finite value between 0 and 1")
        object.__setattr__(self, "memory_weight", float(self.memory_weight))
        if not isinstance(self.evaluation_label, DiagnosisLabel):
            raise ValueError("evaluation_label must be a DiagnosisLabel")
        if self.execution_interrupt not in {None, "after_claim", "after_effect"}:
            raise ValueError("execution_interrupt must be after_claim, after_effect, or None")


@dataclass(frozen=True, slots=True)
class AdapterSet:
    """One complete, backend-free dependency bundle for an EvoWorld pipeline."""

    task_source: TaskSource
    scene_anchor: SceneAnchor
    router: Router
    policies: Mapping[str, ExpertPolicy]
    world_model: WorldModel
    rollout_scorer: RolloutScorer
    executor: Executor
    evaluator: TrajectoryEvaluator
    trainer: Trainer
    memory: ExperienceMemory
    store: RunStore
    receipts: DeterministicReceipts


class DeterministicReceipts:
    """Disk-backed receipts binding a stable operation ID to canonical input."""

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir.resolve(strict=True)
        self._effect_counts: dict[tuple[str, str], int] = {}
        self._lock = threading.RLock()

    def perform(
        self,
        kind: str,
        operation_id: str,
        inputs: JsonObject,
        effect: Callable[[], T],
        encode: Callable[[T], JsonValue],
        decode: Callable[[JsonValue], T],
    ) -> T:
        if not operation_id.strip():
            raise ValueError("operation_id must not be blank")
        key = (kind, operation_id)
        digest = input_digest(inputs)
        with self._lock, self._operation_lock(operation_id):
            persisted = self.load(kind, operation_id, digest, decode)
            if persisted is not None:
                return persisted
            value = effect()
            self._effect_counts[key] = self._effect_counts.get(key, 0) + 1
            self.complete(kind, operation_id, digest, encode(value))
            return value

    def load(
        self,
        kind: str,
        operation_id: str,
        digest: str,
        decode: Callable[[JsonValue], T],
    ) -> T | None:
        path = self._receipt_path(operation_id)
        if not path.exists():
            return None
        receipt = decode_json_object(path.read_text(encoding="utf-8"), source=str(path))
        expected = {"schema_version", "operation_id", "kind", "input_digest", "output"}
        if set(receipt) != expected:
            raise ValueError(f"invalid deterministic receipt at {path}")
        if receipt.get("operation_id") != operation_id or receipt.get("kind") != kind:
            raise OperationConflictError("operation receipt identity mismatch")
        if receipt.get("input_digest") != digest:
            raise OperationConflictError("operation ID was reused with different input")
        return decode(receipt["output"])

    def complete(self, kind: str, operation_id: str, digest: str, output: JsonValue) -> None:
        path = self._receipt_path(operation_id)
        atomic_write_json(
            path,
            {
                "schema_version": 1,
                "operation_id": operation_id,
                "kind": kind,
                "input_digest": digest,
                "output": output,
            },
        )

    def claim_path(self, operation_id: str) -> Path:
        return _safe_operation_path(
            self._run_dir,
            "adapter_claims",
            f"{sha256(operation_id.encode()).hexdigest()}.json",
        )

    def _receipt_path(self, operation_id: str) -> Path:
        return _safe_operation_path(
            self._run_dir,
            "adapter_receipts",
            f"{sha256(operation_id.encode()).hexdigest()}.json",
        )

    @contextmanager
    def _operation_lock(self, operation_id: str) -> Iterator[None]:
        path = _safe_operation_path(
            self._run_dir,
            ".locks",
            f"operation-{sha256(operation_id.encode()).hexdigest()}.lock",
        )
        with path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def effect_count(self, kind: str, operation_id: str) -> int:
        return self._effect_counts.get((kind, operation_id), 0)

    def record_effect(self, kind: str, operation_id: str) -> None:
        key = (kind, operation_id)
        self._effect_counts[key] = self._effect_counts.get(key, 0) + 1


class _TaskSource:
    def __init__(self, task: TaskSpec, receipts: DeterministicReceipts) -> None:
        self._task = task
        self._receipts = receipts

    def next_task(self, *, operation_id: str = "direct:task") -> TaskSpec:
        return self._receipts.perform(
            "task", operation_id, {}, lambda: self._task, task_to_json, _decode_task
        )


class _SceneAnchor:
    def __init__(self, scene: SceneRecord, receipts: DeterministicReceipts) -> None:
        self._scene = scene
        self._receipts = receipts

    def anchor(self, task: TaskSpec, *, operation_id: str = "") -> SceneRecord:
        operation_id = operation_id or f"direct:anchor:{task.task_id}"
        return self._receipts.perform(
            "anchor",
            operation_id,
            {"task": task_to_json(task)},
            lambda: self._scene,
            scene_to_json,
            _decode_scene,
        )


class _Router:
    def __init__(self, router: Router, receipts: DeterministicReceipts) -> None:
        self._router = router
        self._receipts = receipts

    def shortlist(
        self,
        task: TaskSpec,
        scene: SceneRecord,
        memory: ExperienceMemory,
        *,
        operation_id: str = "",
    ) -> tuple[ExpertSpec, ...]:
        operation_id = operation_id or f"direct:route:{task.task_id}:{scene.scene_id}"
        return self._receipts.perform(
            "route",
            operation_id,
            {"task": task_to_json(task), "scene": scene_to_json(scene)},
            lambda: tuple(
                self._router.shortlist(task, scene, memory, operation_id=f"{operation_id}:memory")
            ),
            lambda experts: [expert_to_json(expert) for expert in experts],
            _decode_experts,
        )


class _Policy:
    def __init__(
        self,
        seed: int,
        expert: ExpertSpec,
        horizon: int,
        receipts: DeterministicReceipts,
        run_dir: Path,
    ) -> None:
        self._seed = seed
        self._expert = expert
        self._horizon = horizon
        self._receipts = receipts
        self._run_dir = run_dir

    def act(
        self, task: TaskSpec, observation: SceneRecord, *, operation_id: str = ""
    ) -> ActionChunk:
        operation_id = operation_id or (
            f"direct:act:{self._expert.expert_id}:"
            f"{_digest(self._seed, task.task_id, observation.scene_id, _observation_identity(observation))}"
        )
        return self._receipts.perform(
            "act",
            operation_id,
            {
                "task": task_to_json(task),
                "scene": scene_to_json(observation),
                "expert": expert_to_json(self._expert),
            },
            lambda: self._act(task, observation, operation_id),
            action_to_json,
            _decode_action,
        )

    def _act(self, task: TaskSpec, observation: SceneRecord, operation_id: str) -> ActionChunk:
        observation_identity = _observation_identity(observation)
        actions = tuple(
            (
                _unit_value(
                    self._seed,
                    "action",
                    self._expert.expert_id,
                    task.task_id,
                    observation.scene_id,
                    observation_identity,
                    step,
                ),
            )
            for step in range(self._horizon)
        )
        timestamps = tuple(float(step) for step in range(self._horizon))
        result = ActionChunk(
            expert_id=self._expert.expert_id,
            actions=actions,
            timestamps=timestamps,
            schema_name=self._expert.action_schema,
        )
        _write_receipt(
            self._run_dir,
            operation_id,
            self._expert.expert_id,
            "act",
            {"operation_id": operation_id, "actions": action_to_json(result)},
        )
        return result


class _WorldModel:
    def __init__(
        self,
        seed: int,
        scenario: DeterministicScenario,
        run_dir: Path,
        receipts: DeterministicReceipts,
    ) -> None:
        self._seed = seed
        self._scenario = scenario
        self._run_dir = run_dir
        self._receipts = receipts

    def imagine(
        self,
        task: TaskSpec,
        scene: SceneRecord,
        expert: ExpertSpec,
        actions: ActionChunk,
        *,
        operation_id: str = "",
    ) -> Rollout:
        operation_id = operation_id or (
            f"direct:imagine:{expert.expert_id}:"
            f"{_digest(self._seed, task.task_id, scene.scene_id, _action_payload(expert, actions))}"
        )
        return self._receipts.perform(
            "imagine",
            operation_id,
            {
                "task": task_to_json(task),
                "scene": scene_to_json(scene),
                "expert": expert_to_json(expert),
                "actions": action_to_json(actions),
            },
            lambda: self._imagine(task, scene, expert, actions, operation_id),
            rollout_to_json,
            _decode_rollout,
        )

    def _imagine(
        self,
        task: TaskSpec,
        scene: SceneRecord,
        expert: ExpertSpec,
        actions: ActionChunk,
        operation_id: str,
    ) -> Rollout:
        action_payload = _action_payload(expert, actions)
        identity = _digest(
            self._seed,
            "rollout",
            task.task_id,
            scene.scene_id,
            expert.expert_id,
            action_payload,
        )
        if expert.expert_id in self._scenario.failing_rollout_experts:
            raise DeterministicRolloutError(
                f"scripted rollout failure for {expert.expert_id}",
                expert_id=expert.expert_id,
                identity=identity,
            )
        rollout_id = f"rollout-{identity}"
        frame = _write_text_artifact(
            self._run_dir,
            f"rollout-{identity}.txt",
            f"rollout_id={rollout_id}\nexpert_id={expert.expert_id}\nscene_id={scene.scene_id}\n",
            relative_dir=Path("rounds") / _round_id(operation_id) / "rollouts" / expert.expert_id,
        )
        result = Rollout(
            rollout_id=rollout_id,
            expert_id=expert.expert_id,
            scene_id=scene.scene_id,
            frame_paths=(frame,),
            actions=actions,
            metrics={
                "router_score": _score(
                    self._scenario.router_scores, self._seed, "router", expert.expert_id
                ),
                "world_model_score": _score(
                    self._scenario.rollout_scores, self._seed, "rollout_score", expert.expert_id
                ),
            },
        )
        _write_receipt(
            self._run_dir,
            operation_id,
            expert.expert_id,
            "imagine",
            {"operation_id": operation_id, "rollout": rollout_to_json(result)},
        )
        return result


class _RolloutScorer:
    def __init__(
        self, router_weight: float, receipts: DeterministicReceipts, run_dir: Path
    ) -> None:
        self._router_weight = router_weight
        self._receipts = receipts
        self._run_dir = run_dir

    def score(self, task: TaskSpec, rollout: Rollout, *, operation_id: str = "") -> CandidateScore:
        operation_id = operation_id or f"direct:score:{rollout.expert_id}:{rollout.rollout_id}"
        return self._receipts.perform(
            "score",
            operation_id,
            {"task": task_to_json(task), "rollout": rollout_to_json(rollout)},
            lambda: self._score(rollout, operation_id),
            score_to_json,
            _decode_score,
        )

    def _score(self, rollout: Rollout, operation_id: str) -> CandidateScore:
        router_score = _metric(rollout, "router_score")
        rollout_score = _metric(rollout, "world_model_score")
        result = CandidateScore.from_components(
            rollout.expert_id,
            rollout.rollout_id,
            router_score,
            rollout_score,
            self._router_weight,
        )
        _write_receipt(
            self._run_dir,
            operation_id,
            rollout.expert_id,
            "score",
            {"operation_id": operation_id, "score": score_to_json(result)},
        )
        return result


class _Executor:
    def __init__(
        self,
        seed: int,
        scenario: DeterministicScenario,
        policies: Mapping[str, ExpertPolicy],
        run_dir: Path,
        receipts: DeterministicReceipts,
    ) -> None:
        self._seed = seed
        self._scenario = scenario
        self._policies = policies
        self._run_dir = run_dir
        self._receipts = receipts

    def execute(
        self,
        task: TaskSpec,
        scene: SceneRecord,
        expert: ExpertSpec,
        actions: ActionChunk | None = None,
        *,
        operation_id: str = "",
    ) -> Trajectory:
        operation_id = operation_id or (
            f"direct:execute:{expert.expert_id}:"
            f"{_digest(self._seed, task.task_id, scene.scene_id, _observation_identity(scene))}"
        )
        if actions is None:
            policy = self._policies.get(expert.expert_id)
            if policy is None:
                raise ValueError(f"no deterministic policy configured for {expert.expert_id}")
            actions = policy.act(task, scene, operation_id=f"{operation_id}:actions")
        inputs: JsonObject = {
            "task": task_to_json(task),
            "scene": scene_to_json(scene),
            "expert": expert_to_json(expert),
            "actions": action_to_json(actions),
        }
        digest = input_digest(inputs)
        with self._receipts._operation_lock(operation_id):
            completed = self._receipts.load("execute", operation_id, digest, _decode_trajectory)
            if completed is not None:
                return completed
            claim_path = self._receipts.claim_path(operation_id)
            if claim_path.exists():
                claim = decode_json_object(
                    claim_path.read_text(encoding="utf-8"), source=str(claim_path)
                )
                expected = {"schema_version", "operation_id", "kind", "input_digest", "state"}
                if set(claim) != expected:
                    raise ValueError("invalid physical operation claim")
                if claim.get("operation_id") != operation_id or claim.get("kind") != "execute":
                    raise OperationConflictError("physical operation claim identity mismatch")
                if claim.get("input_digest") != digest:
                    raise OperationConflictError("operation ID was reused with different input")
                raise OperationInDoubtError(
                    f"physical execution {operation_id} was claimed without a completion receipt"
                )
            _exclusive_write_json(
                claim_path,
                {
                    "schema_version": 1,
                    "operation_id": operation_id,
                    "kind": "execute",
                    "input_digest": digest,
                    "state": "claimed",
                },
            )
            if self._scenario.execution_interrupt == "after_claim":
                raise DeterministicInterruption("interrupted after physical execution claim")
            result = self._execute(task, scene, expert, actions, operation_id)
            self._receipts.record_effect("execute", operation_id)
            if self._scenario.execution_interrupt == "after_effect":
                raise DeterministicInterruption("interrupted after physical execution effect")
            self._receipts.complete("execute", operation_id, digest, trajectory_to_json(result))
            return result

    def _execute(
        self,
        task: TaskSpec,
        scene: SceneRecord,
        expert: ExpertSpec,
        actions: ActionChunk,
        operation_id: str,
    ) -> Trajectory:
        action_payload = _action_payload(expert, actions)
        request_identity = _digest(
            self._seed,
            "execution_request",
            task.task_id,
            scene.scene_id,
            expert.expert_id,
            _observation_identity(scene),
        )
        if expert.expert_id in self._scenario.execution_failure_experts:
            raise DeterministicExecutionError(
                f"scripted execution failure for {expert.expert_id}",
                expert_id=expert.expert_id,
                identity=request_identity,
            )
        rollout_identity = _digest(
            self._seed,
            "rollout",
            task.task_id,
            scene.scene_id,
            expert.expert_id,
            action_payload,
        )
        identity = _digest(
            self._seed,
            "trajectory",
            task.task_id,
            scene.scene_id,
            expert.expert_id,
            action_payload,
            operation_id,
        )
        rollout_id = f"rollout-{rollout_identity}"
        trajectory_id = f"trajectory-{identity}"
        frame = _write_text_artifact(
            self._run_dir,
            f"trajectory-{identity}.txt",
            f"trajectory_id={trajectory_id}\nexpert_id={expert.expert_id}\nscene_id={scene.scene_id}\n",
        )
        result = Trajectory(
            trajectory_id=trajectory_id,
            rollout_id=rollout_id,
            frame_paths=(frame,),
            actions=actions,
            metadata={"scenario": "deterministic"},
        )
        return result


class _Evaluator:
    def __init__(self, scenario: DeterministicScenario, receipts: DeterministicReceipts) -> None:
        self._scenario = scenario
        self._receipts = receipts

    def evaluate(
        self,
        task: TaskSpec,
        scene: SceneRecord,
        trajectory: Trajectory,
        memory: ExperienceMemory,
        *,
        operation_id: str = "",
    ) -> Diagnosis:
        operation_id = operation_id or f"direct:evaluate:{trajectory.trajectory_id}"
        return self._receipts.perform(
            "evaluate",
            operation_id,
            {
                "task": task_to_json(task),
                "scene": scene_to_json(scene),
                "trajectory": trajectory_to_json(trajectory),
            },
            lambda: Diagnosis(
                score=self._scenario.evaluation_score,
                label=self._scenario.evaluation_label,
                reason=self._scenario.evaluation_reason,
                metadata={"trajectory_id": trajectory.trajectory_id},
            ),
            diagnosis_to_json,
            _decode_diagnosis,
        )


class _Trainer:
    def __init__(self, receipts: DeterministicReceipts) -> None:
        self._receipts = receipts

    def plan(self, update: UpdateDecision, *, operation_id: str = "") -> UpdateDecision:
        operation_id = operation_id or f"direct:train_plan:{update.update_id}"
        return self._receipts.perform(
            "train_plan",
            operation_id,
            {"update": update_to_json(update)},
            lambda: update,
            update_to_json,
            _decode_update,
        )

    def execute(self, plan: UpdateDecision, *, operation_id: str = "") -> UpdateDecision:
        operation_id = operation_id or f"direct:train_execute:{plan.update_id}"
        return self._receipts.perform(
            "train_execute",
            operation_id,
            {"update": update_to_json(plan)},
            lambda: self._execute(plan),
            update_to_json,
            _decode_update,
        )

    def _execute(self, plan: UpdateDecision) -> UpdateDecision:
        if plan.execution_mode is not ExecutionMode.EXECUTE:
            raise DeterministicTrainingError(
                "deterministic trainer is plan-only; set execution_mode to execute explicitly",
                identity=plan.update_id,
            )
        return plan


def build_deterministic_adapters(
    config: EvoWorldConfig,
    run_dir: Path,
    scenario: DeterministicScenario | None = None,
) -> AdapterSet:
    """Build deterministic ports without inspecting hardware, environment, or network state."""
    validate_config(config)
    if config.mode != "dry_run":
        raise ValueError("deterministic adapters are only available for dry_run mode")
    scenario = scenario or DeterministicScenario()
    experts = tuple(sorted(config.experts, key=lambda expert: expert.expert_id))
    _validate_scenario_experts(scenario, experts)
    store = _open_store(config, run_dir)
    receipts = DeterministicReceipts(store.run_dir)
    scene_identity = _digest(config.seed, "scene", scenario.scene_id)
    observation = _write_text_artifact(
        store.run_dir,
        f"scene-{scene_identity}.txt",
        f"scene_id={scenario.scene_id}\nseed={config.seed}\n",
    )
    task = TaskSpec(task_id=scenario.task_id, instruction=scenario.instruction)
    scene = SceneRecord(scene_id=scenario.scene_id, observation_paths=(observation,))
    policies: Mapping[str, ExpertPolicy] = MappingProxyType(
        {
            expert.expert_id: _Policy(
                config.seed,
                expert,
                config.rollout_horizon,
                receipts,
                store.run_dir,
            )
            for expert in experts
        }
    )
    router_scores = {
        expert.expert_id: _score(scenario.router_scores, config.seed, "router", expert.expert_id)
        for expert in experts
    }
    memory = JsonlExperienceMemory(store.run_dir / "memory.jsonl", containment_root=store.run_dir)
    return AdapterSet(
        task_source=_TaskSource(task, receipts),
        scene_anchor=_SceneAnchor(scene, receipts),
        router=_Router(
            MemoryAwareRouter(
                experts=experts,
                semantic_logits=router_scores,
                memory_weight=scenario.memory_weight,
                rng=random.Random(_router_seed(config.seed, scenario)),
            ),
            receipts,
        ),
        policies=policies,
        world_model=_WorldModel(config.seed, scenario, store.run_dir, receipts),
        rollout_scorer=_RolloutScorer(config.router_weight, receipts, store.run_dir),
        executor=_Executor(config.seed, scenario, policies, store.run_dir, receipts),
        evaluator=_Evaluator(scenario, receipts),
        trainer=_Trainer(receipts),
        memory=memory,
        store=store,
        receipts=receipts,
    )


def _open_store(config: EvoWorldConfig, run_dir: Path) -> JsonRunStore:
    resolved_run_dir = run_dir.resolve()
    resolved_run_dir.parent.mkdir(parents=True, exist_ok=True)
    digest = config_digest(config)
    manifest = resolved_run_dir / "manifest.json"
    if manifest.exists():
        return JsonRunStore.open(resolved_run_dir, digest)
    return JsonRunStore.create(
        resolved_run_dir.parent,
        resolved_run_dir.name,
        resolved_manifest(config),
        digest,
    )


def _validate_scenario_experts(
    scenario: DeterministicScenario, experts: tuple[ExpertSpec, ...]
) -> None:
    expert_ids = {expert.expert_id for expert in experts}
    scripted_ids = (
        set(scenario.router_scores)
        | set(scenario.rollout_scores)
        | set(scenario.failing_rollout_experts)
        | set(scenario.execution_failure_experts)
    )
    unknown = sorted(scripted_ids - expert_ids)
    if unknown:
        raise ValueError(f"scenario references unknown expert(s): {', '.join(unknown)}")


def _score(scores: Mapping[str, float], seed: int, stage: str, expert_id: str) -> float:
    return scores.get(expert_id, _unit_value(seed, stage, expert_id))


def _metric(rollout: Rollout, key: str) -> float:
    value = rollout.metrics.get(key)
    if not isinstance(value, float):
        raise ValueError(f"deterministic rollout is missing numeric {key}")
    return value


def _digest(seed: int, *identity: object) -> str:
    payload = json.dumps(
        [seed, *identity], ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _unit_value(seed: int, *identity: object) -> float:
    return int(_digest(seed, *identity)[:16], 16) / ((1 << 64) - 1)


def _normalized_scores(scores: Mapping[str, float], name: str) -> Mapping[str, float]:
    normalized: dict[str, float] = {}
    for expert_id, score in scores.items():
        if not isinstance(expert_id, str) or not expert_id.strip():
            raise ValueError(f"{name} keys must be nonblank strings")
        if not _unit_real(score):
            raise ValueError(f"{name} values must be finite numbers between 0 and 1")
        normalized[expert_id] = float(score)
    return MappingProxyType(normalized)


def _unit_real(value: object) -> bool:
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        numeric = float(value)
    except OverflowError:
        return False
    return isfinite(numeric) and 0.0 <= numeric <= 1.0


def _router_seed(seed: int, scenario: DeterministicScenario) -> int:
    return int(_digest(seed, "router", scenario.task_id, scenario.scene_id)[:16], 16)


def _observation_identity(observation: SceneRecord) -> tuple[tuple[str, str], ...]:
    identities: list[tuple[str, str]] = []
    for path in observation.observation_paths:
        try:
            content = path.read_bytes()
        except OSError as error:
            raise ValueError(f"deterministic observation is unreadable: {path}") from error
        identities.append((path.name, sha256(content).hexdigest()))
    return tuple(identities)


def _action_payload(expert: ExpertSpec, actions: ActionChunk) -> dict[str, object]:
    if actions.expert_id != expert.expert_id:
        raise ValueError("actions expert_id must match the imagined expert")
    if actions.schema_name != expert.action_schema:
        raise ValueError("actions schema must match the imagined expert")
    if not actions.actions:
        raise ValueError("actions must not be empty")
    dimensions = {len(action) for action in actions.actions}
    if len(dimensions) != 1 or 0 in dimensions:
        raise ValueError("actions must have one nonempty, consistent dimension")
    if any(not _finite_number(value) for action in actions.actions for value in action):
        raise ValueError("actions must contain only finite numbers")
    if any(not _finite_number(timestamp) for timestamp in actions.timestamps):
        raise ValueError("action timestamps must contain only finite numbers")
    return {
        "actions": [[float(value) for value in action] for action in actions.actions],
        "expert_id": actions.expert_id,
        "schema_name": actions.schema_name,
        "timestamps": [float(timestamp) for timestamp in actions.timestamps],
    }


def _finite_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)


def _as_object(value: JsonValue, name: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"persisted {name} output must be a JSON object")
    return value


def _decode_task(value: JsonValue) -> TaskSpec:
    return task_from_json(_as_object(value, "task"))


def _decode_scene(value: JsonValue) -> SceneRecord:
    return scene_from_json(_as_object(value, "scene"))


def _decode_experts(value: JsonValue) -> tuple[ExpertSpec, ...]:
    if not isinstance(value, list):
        raise ValueError("persisted route output must be a JSON array")
    return tuple(expert_from_json(_as_object(item, "expert")) for item in value)


def _decode_action(value: JsonValue) -> ActionChunk:
    return action_from_json(_as_object(value, "action"))


def _decode_rollout(value: JsonValue) -> Rollout:
    return rollout_from_json(_as_object(value, "rollout"))


def _decode_score(value: JsonValue) -> CandidateScore:
    return score_from_json(_as_object(value, "score"))


def _decode_trajectory(value: JsonValue) -> Trajectory:
    return trajectory_from_json(_as_object(value, "trajectory"))


def _decode_diagnosis(value: JsonValue) -> Diagnosis:
    return diagnosis_from_json(_as_object(value, "diagnosis"))


def _decode_update(value: JsonValue) -> UpdateDecision:
    return update_from_json(_as_object(value, "update"))


def _write_text_artifact(
    run_dir: Path,
    name: str,
    content: str,
    *,
    relative_dir: Path = Path("frames"),
) -> Path:
    root = run_dir.resolve(strict=True)
    frame_dir = root / relative_dir
    frame_dir.mkdir(parents=True, exist_ok=True)
    path = (frame_dir / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("artifact path must remain within the resolved run directory")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _exclusive_write_json(path: Path, value: JsonObject) -> None:
    payload = (canonical_json(value, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _safe_operation_path(run_dir: Path, directory: str, filename: str) -> Path:
    root = run_dir.resolve(strict=True)
    parent = root / directory
    if parent.is_symlink():
        raise ValueError(f"operation storage directory must not be a symlink: {parent}")
    try:
        parent.mkdir(mode=0o755)
    except FileExistsError:
        pass
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"operation storage directory must not be a symlink: {parent}")
    if not parent.resolve(strict=True).is_relative_to(root):
        raise ValueError("operation storage directory must remain within the run directory")
    path = parent / filename
    if path.is_symlink():
        raise ValueError(f"operation storage file must not be a symlink: {path}")
    if not path.resolve().is_relative_to(root):
        raise ValueError("operation storage path must remain within the run directory")
    return path


def _round_id(operation_id: str) -> str:
    round_id = operation_id.split(":", maxsplit=1)[0]
    if not round_id or "/" in round_id or "\\" in round_id or round_id in {".", ".."}:
        raise ValueError("operation_id must start with a safe round ID")
    return round_id


def _write_receipt(
    run_dir: Path,
    operation_id: str,
    expert_id: str,
    name: str,
    value: dict[str, object],
) -> None:
    path = _safe_nested_file(
        run_dir,
        Path("rounds") / _round_id(operation_id) / "rollouts" / expert_id / f"{name}.receipt.json",
    )
    atomic_write_json(path, cast(JsonValue, value))


def _safe_nested_file(run_dir: Path, relative: Path) -> Path:
    root = run_dir.resolve(strict=True)
    current = root
    for part in relative.parent.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"receipt directory must not be a symlink: {current}")
        try:
            current.mkdir()
        except FileExistsError:
            pass
        if current.is_symlink() or not current.is_dir():
            raise ValueError(f"receipt directory must not be a symlink: {current}")
        if not current.resolve(strict=True).is_relative_to(root):
            raise ValueError("receipt directory must remain within the run directory")
    path = current / relative.name
    if path.is_symlink():
        raise ValueError(f"receipt file must not be a symlink: {path}")
    if not path.resolve().is_relative_to(root):
        raise ValueError("receipt path must remain within the run directory")
    return path

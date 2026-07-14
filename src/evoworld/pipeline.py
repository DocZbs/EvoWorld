"""Durable, intent-driven four-stage orchestration for EvoWorld."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Protocol, TypeVar, runtime_checkable

from evoworld.codec import (
    action_to_json,
    candidate_to_json,
    diagnosis_to_json,
    expert_to_json,
    input_digest,
    rollout_to_json,
    scene_to_json,
    score_to_json,
    task_to_json,
    trajectory_to_json,
    update_to_json,
)
from evoworld.config import EvoWorldConfig
from evoworld.contracts import (
    ActionChunk,
    CandidateScore,
    CandidateState,
    Diagnosis,
    ExecutionMode,
    ExperienceRecord,
    ExpertSpec,
    JsonObject,
    OperationIntent,
    OperationKind,
    OperationReceipt,
    Rollout,
    RoundResult,
    RoundStage,
    RoundState,
    SceneRecord,
    TaskSpec,
    Trajectory,
    UpdateDecision,
    UpdateTarget,
)
from evoworld.evolution import decide_update
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
from evoworld.routing import select_candidate

T = TypeVar("T")

_MAX_PERSISTED_ERROR_LENGTH = 256
_PUBLIC_EXCEPTION_NAMES: Mapping[type[Exception], str] = {
    FileNotFoundError: "FileNotFoundError",
    KeyError: "KeyError",
    OperationConflictError: "OperationConflictError",
    RuntimeError: "RuntimeError",
    TimeoutError: "TimeoutError",
    TypeError: "TypeError",
    ValueError: "ValueError",
}


class _OperationFailure(RuntimeError):
    def __init__(self, error: Exception, kind: OperationKind) -> None:
        self.public_message = _public_operation_error(error, kind)
        super().__init__(self.public_message)


@runtime_checkable
class ResumableRunStore(RunStore, Protocol):
    def read_state(self) -> RoundState | None: ...


class PipelineAdapters(Protocol):
    @property
    def task_source(self) -> TaskSource: ...

    @property
    def scene_anchor(self) -> SceneAnchor: ...

    @property
    def router(self) -> Router: ...

    @property
    def policies(self) -> Mapping[str, ExpertPolicy]: ...

    @property
    def world_model(self) -> WorldModel: ...

    @property
    def rollout_scorer(self) -> RolloutScorer: ...

    @property
    def executor(self) -> Executor: ...

    @property
    def evaluator(self) -> TrajectoryEvaluator: ...

    @property
    def trainer(self) -> Trainer: ...

    @property
    def memory(self) -> ExperienceMemory: ...

    @property
    def store(self) -> RunStore: ...


class EvoWorldPipeline:
    """Run every external operation behind a durable idempotency intent."""

    def __init__(
        self,
        config: EvoWorldConfig,
        adapters: PipelineAdapters,
        *,
        execution_mode: ExecutionMode = ExecutionMode.PLAN,
    ) -> None:
        if not isinstance(adapters.store, ResumableRunStore):
            raise TypeError("pipeline store must support reading the latest state")
        if not isinstance(execution_mode, ExecutionMode):
            raise TypeError("execution_mode must be an ExecutionMode")
        self._config = config
        self._task_source = adapters.task_source
        self._scene_anchor = adapters.scene_anchor
        self._router = adapters.router
        self._policies = adapters.policies
        self._world_model = adapters.world_model
        self._rollout_scorer = adapters.rollout_scorer
        self._executor = adapters.executor
        self._evaluator = adapters.evaluator
        self._trainer = adapters.trainer
        self._memory = adapters.memory
        self.store = adapters.store
        self._execution_mode = execution_mode

    def run(self, rounds: int) -> tuple[RoundResult, ...]:
        if not isinstance(rounds, int) or isinstance(rounds, bool):
            raise TypeError("rounds must be an integer")
        if rounds < 0:
            raise ValueError("rounds must be nonnegative")
        return tuple(self.run_round(f"round-{index:06d}") for index in range(rounds))

    def run_round(self, round_id: str, state: RoundState | None = None) -> RoundResult:
        _validate_round_id(round_id)
        if state is None:
            current = RoundState(
                round_id=round_id,
                stage=RoundStage.STARTED,
                started_at=datetime.now(timezone.utc).isoformat(),
                seed=self._config.seed,
                reliability_threshold=self._config.reliability_threshold,
                execution_mode=self._execution_mode,
            )
            self.store.claim_round(current)
            self.store.save_state(current)
        else:
            if (
                state.round_id != round_id
                or state.seed != self._config.seed
                or state.reliability_threshold != self._config.reliability_threshold
                or state.execution_mode is not self._execution_mode
            ):
                raise ValueError("round state does not match the active run")
            current = state

        while current.stage is not RoundStage.COMPLETE:
            if current.stage is RoundStage.IN_DOUBT:
                raise OperationInDoubtError(
                    current.in_doubt_reason or "round contains an in-doubt operation"
                )
            if current.stage is RoundStage.STARTED:
                current = self._ground(current)
            elif current.stage is RoundStage.GROUNDED:
                current = self._score_candidates(current)
            elif current.stage is RoundStage.CANDIDATES_SCORED:
                current = self._execute(current)
            elif current.stage is RoundStage.EXECUTED:
                current = self._evaluate_and_evolve(current)
        return self._result(current)

    def resume(self) -> RoundResult | None:
        state = self.store.read_state()
        if state is None or state.stage is RoundStage.COMPLETE:
            return None
        return self.run_round(state.round_id, state)

    def _ground(self, state: RoundState) -> RoundState:
        current = state
        if current.task is None:
            current = self._invoke(
                current,
                OperationKind.TASK,
                {},
                lambda operation_id: self._task_source.next_task(operation_id=operation_id),
                lambda snapshot, task, operation_id: self._record_task(
                    snapshot, task, operation_id
                ),
            )
        if current.scene is None:
            task = _required_task(current)
            current = self._invoke(
                current,
                OperationKind.ANCHOR,
                {"task": task_to_json(task)},
                lambda operation_id: self._scene_anchor.anchor(task, operation_id=operation_id),
                lambda snapshot, scene, _operation_id: replace(snapshot, scene=scene),
            )
        task = _required_task(current)
        scene = _required_scene(current)
        self.store.write_artifact(
            current.round_id,
            "grounding",
            {"task": task_to_json(task), "scene": scene_to_json(scene)},
        )
        return self._persist(replace(current, stage=RoundStage.GROUNDED))

    def _record_task(self, state: RoundState, task: TaskSpec, operation_id: str) -> RoundState:
        self.store.append_task(state.round_id, operation_id, task)
        return replace(state, task=task)

    def _score_candidates(self, state: RoundState) -> RoundState:
        current = state
        if not current.experts:
            task, scene = _task_scene(current)
            current = self._invoke(
                current,
                OperationKind.ROUTE,
                {"task": task_to_json(task), "scene": scene_to_json(scene)},
                lambda operation_id: tuple(
                    self._router.shortlist(task, scene, self._memory, operation_id=operation_id)
                ),
                self._record_route,
            )
        for index, candidate in enumerate(current.candidates):
            if candidate.score is not None:
                continue
            current = self._complete_candidate(current, index)

        evidence: JsonObject = {
            "candidates": [candidate_to_json(item) for item in current.candidates],
            "selected_expert_id": None,
        }
        try:
            selected = select_candidate(current.scores)
        except Exception:
            self.store.write_artifact(current.round_id, "candidates", evidence)
            raise
        evidence["selected_expert_id"] = selected.expert_id
        self.store.write_artifact(current.round_id, "candidates", evidence)
        return self._persist(
            replace(
                current,
                stage=RoundStage.CANDIDATES_SCORED,
                selected_expert_id=selected.expert_id,
            )
        )

    def _record_route(
        self,
        state: RoundState,
        experts: tuple[ExpertSpec, ...],
        _operation_id: str,
    ) -> RoundState:
        if not experts or len({expert.expert_id for expert in experts}) != len(experts):
            raise ValueError("router must return nonempty unique expert IDs")
        return replace(
            state,
            experts=experts,
            candidates=tuple(CandidateState(expert) for expert in experts),
        )

    def _complete_candidate(self, state: RoundState, index: int) -> RoundState:
        current = state
        candidate = current.candidates[index]
        expert = candidate.expert
        try:
            if candidate.actions is None:
                task, scene = _task_scene(current)
                policy = self._policies[expert.expert_id]
                current = self._invoke(
                    current,
                    OperationKind.ACT,
                    {
                        "task": task_to_json(task),
                        "scene": scene_to_json(scene),
                        "expert": expert_to_json(expert),
                    },
                    lambda operation_id: policy.act(task, scene, operation_id=operation_id),
                    lambda snapshot, actions, _operation_id: self._set_actions(
                        snapshot, index, actions
                    ),
                    expert_id=expert.expert_id,
                )
                candidate = current.candidates[index]
            if candidate.rollout is None:
                task, scene = _task_scene(current)
                actions = _required_actions(candidate)
                current = self._invoke(
                    current,
                    OperationKind.IMAGINE,
                    {
                        "task": task_to_json(task),
                        "scene": scene_to_json(scene),
                        "expert": expert_to_json(expert),
                        "actions": action_to_json(actions),
                    },
                    lambda operation_id: self._world_model.imagine(
                        task,
                        scene,
                        expert,
                        actions,
                        operation_id=operation_id,
                    ),
                    lambda snapshot, rollout, _operation_id: self._set_rollout(
                        snapshot, index, rollout
                    ),
                    expert_id=expert.expert_id,
                )
                candidate = current.candidates[index]
            rollout = _required_rollout(candidate)
            task = _required_task(current)
            current = self._invoke(
                current,
                OperationKind.SCORE,
                {"task": task_to_json(task), "rollout": rollout_to_json(rollout)},
                lambda operation_id: self._rollout_scorer.score(
                    task, rollout, operation_id=operation_id
                ),
                lambda snapshot, score, _operation_id: self._set_score(snapshot, index, score),
                expert_id=expert.expert_id,
            )
        except _OperationFailure as failure:
            persisted = self.store.read_state()
            if persisted is not None and persisted.round_id == current.round_id:
                current = persisted
            current = self._record_candidate_failure(current, index, failure)
        return current

    def _set_actions(self, state: RoundState, index: int, actions: ActionChunk) -> RoundState:
        candidate = state.candidates[index]
        if (
            actions.expert_id != candidate.expert.expert_id
            or actions.schema_name != candidate.expert.action_schema
        ):
            raise ValueError("candidate action provenance does not match the expert")
        return _replace_candidate(state, index, replace(candidate, actions=actions))

    def _set_rollout(self, state: RoundState, index: int, rollout: Rollout) -> RoundState:
        candidate = state.candidates[index]
        scene = _required_scene(state)
        if (
            rollout.expert_id != candidate.expert.expert_id
            or rollout.scene_id != scene.scene_id
            or rollout.actions != candidate.actions
            or not rollout.valid
        ):
            raise ValueError("candidate rollout provenance is invalid")
        return _replace_candidate(state, index, replace(candidate, rollout=rollout))

    def _set_score(self, state: RoundState, index: int, score: CandidateScore) -> RoundState:
        candidate = state.candidates[index]
        rollout = _required_rollout(candidate)
        if score.expert_id != candidate.expert.expert_id or score.rollout_id != rollout.rollout_id:
            raise ValueError("candidate score provenance is invalid")
        return _replace_candidate(state, index, replace(candidate, score=score))

    def _record_candidate_failure(
        self, state: RoundState, index: int, error: _OperationFailure
    ) -> RoundState:
        candidate = state.candidates[index]
        failed = CandidateScore(
            expert_id=candidate.expert.expert_id,
            rollout_id=None,
            router_score=0.0,
            world_model_score=0.0,
            joint_score=0.0,
            valid=False,
            error=_candidate_error(error),
        )
        cleared = replace(candidate, rollout=None, score=failed)
        pending = state.pending_operation
        if pending is None:
            return self._persist(_replace_candidate(state, index, cleared))
        receipt = OperationReceipt(
            pending.operation_id,
            pending.kind,
            pending.input_digest,
            pending.input_payload,
        )
        updated = _replace_candidate(state, index, cleared)
        return self._persist(
            replace(
                updated,
                pending_operation=None,
                receipts=(*updated.receipts, receipt),
            )
        )

    def _execute(self, state: RoundState) -> RoundState:
        task, scene = _task_scene(state)
        expert = _selected_expert(state)
        selected = _selected_candidate(state)
        if selected.score is None or selected.rollout is None or not selected.score.valid:
            raise ValueError("selected candidate provenance is incomplete")
        selected_rollout = selected.rollout
        current = state
        if current.trajectory is None:
            current = self._invoke(
                current,
                OperationKind.EXECUTE,
                {
                    "task": task_to_json(task),
                    "scene": scene_to_json(scene),
                    "expert": expert_to_json(expert),
                    "actions": action_to_json(selected_rollout.actions),
                },
                lambda operation_id: self._executor.execute(
                    task,
                    scene,
                    expert,
                    selected_rollout.actions,
                    operation_id=operation_id,
                ),
                lambda snapshot, trajectory, _operation_id: replace(
                    snapshot, trajectory=trajectory
                ),
                validate_result=lambda trajectory: _validate_execution_output(
                    selected_rollout, expert, trajectory
                ),
                expert_id=expert.expert_id,
            )
        trajectory = _required_trajectory(current)
        _validate_execution_output(selected_rollout, expert, trajectory)
        self.store.write_artifact(current.round_id, "trajectory", trajectory_to_json(trajectory))
        return self._persist(replace(current, stage=RoundStage.EXECUTED))

    def _evaluate_and_evolve(self, state: RoundState) -> RoundState:
        current = state
        task, scene = _task_scene(current)
        trajectory = _required_trajectory(current)
        expert = _selected_expert(current)
        if current.diagnosis is None:
            current = self._invoke(
                current,
                OperationKind.EVALUATE,
                {
                    "task": task_to_json(task),
                    "scene": scene_to_json(scene),
                    "trajectory": trajectory_to_json(trajectory),
                },
                lambda operation_id: self._evaluator.evaluate(
                    task,
                    scene,
                    trajectory,
                    self._memory,
                    operation_id=operation_id,
                ),
                lambda snapshot, diagnosis, _operation_id: replace(snapshot, diagnosis=diagnosis),
            )
        diagnosis = _required_diagnosis(current)
        if current.update is None:
            decision = decide_update(
                diagnosis=diagnosis,
                reliability_threshold=self._config.reliability_threshold,
                task=task,
                scene=scene,
                expert=expert,
                trajectory=trajectory,
                execution_mode=self._execution_mode,
            )
            current = self._invoke(
                current,
                OperationKind.TRAIN_PLAN,
                {"update": update_to_json(decision)},
                lambda operation_id: self._trainer.plan(decision, operation_id=operation_id),
                lambda snapshot, update, _operation_id: replace(
                    snapshot, update=_validate_update_enrichment(decision, update)
                ),
            )
        update = _required_update(current)
        if (
            self._execution_mode is ExecutionMode.EXECUTE
            and update.target in {UpdateTarget.EXPERT_POLICY, UpdateTarget.WORLD_MODEL}
            and not _has_receipt(
                current, OperationKind.TRAIN_EXECUTE, {"update": update_to_json(update)}
            )
        ):
            current = self._invoke(
                current,
                OperationKind.TRAIN_EXECUTE,
                {"update": update_to_json(update)},
                lambda operation_id: self._trainer.execute(update, operation_id=operation_id),
                lambda snapshot, executed, _operation_id: replace(
                    snapshot, update=_validate_executed_update(update, executed)
                ),
            )
            update = _required_update(current)

        verified = diagnosis.score >= self._config.reliability_threshold
        diagnosis_path = self.store.write_artifact(
            current.round_id,
            "diagnosis",
            {"diagnosis": diagnosis_to_json(diagnosis), "verified": verified},
        )
        update_path = self.store.write_artifact(current.round_id, "update", update_to_json(update))
        if verified:
            provenance: JsonObject = {
                "round_id": current.round_id,
                "task": task_to_json(task),
                "scene": scene_to_json(scene),
                "expert_id": expert.expert_id,
                "score": candidate_to_json(_selected_candidate(current))["score"],
                "diagnosis": diagnosis_to_json(diagnosis),
            }
            verified_dir = self.store.write_verified(trajectory, provenance)
            record = ExperienceRecord(
                record_id=_experience_id(current, trajectory, diagnosis),
                task_id=task.task_id,
                scene_id=scene.scene_id,
                expert_id=expert.expert_id,
                score=_selected_score(current),
                diagnosis=diagnosis,
                artifact_refs=(verified_dir, diagnosis_path, update_path),
            )
            memory_input = _experience_payload(record)
            if not _has_receipt(current, OperationKind.MEMORY_APPEND, memory_input):
                current = self._invoke(
                    current,
                    OperationKind.MEMORY_APPEND,
                    memory_input,
                    lambda operation_id: self._append_memory(record, operation_id),
                    lambda snapshot, _result, _operation_id: snapshot,
                )
        return self._persist(replace(current, stage=RoundStage.COMPLETE))

    def _append_memory(self, record: ExperienceRecord, operation_id: str) -> None:
        self._memory.append(record, operation_id=operation_id)

    def _invoke(
        self,
        state: RoundState,
        kind: OperationKind,
        inputs: JsonObject,
        call: Callable[[str], T],
        apply_result: Callable[[RoundState, T, str], RoundState],
        *,
        validate_result: Callable[[T], None] | None = None,
        expert_id: str | None = None,
    ) -> RoundState:
        current, intent = self._ensure_intent(state, kind, inputs, expert_id)
        failure: _OperationFailure | None = None
        in_doubt_reason: str | None = None
        try:
            result = call(intent.operation_id)
            if validate_result is not None:
                validate_result(result)
            updated = apply_result(current, result, intent.operation_id)
        except OperationInDoubtError:
            in_doubt_reason = _in_doubt_reason(intent)
            blocked = self._persist(
                replace(
                    current,
                    stage=RoundStage.IN_DOUBT,
                    in_doubt_reason=in_doubt_reason,
                )
            )
            in_doubt_reason = blocked.in_doubt_reason
        except Exception as error:
            failure = _OperationFailure(error, kind)
        else:
            return self._persist(
                replace(
                    updated,
                    pending_operation=None,
                    receipts=(
                        *updated.receipts,
                        OperationReceipt(
                            intent.operation_id,
                            intent.kind,
                            intent.input_digest,
                            intent.input_payload,
                        ),
                    ),
                )
            )
        if in_doubt_reason is not None:
            raise OperationInDoubtError(in_doubt_reason)
        assert failure is not None
        raise failure

    def _ensure_intent(
        self,
        state: RoundState,
        kind: OperationKind,
        inputs: JsonObject,
        expert_id: str | None,
    ) -> tuple[RoundState, OperationIntent]:
        operation_id = _operation_id(state.round_id, kind, expert_id)
        expected = OperationIntent(
            operation_id,
            kind,
            input_digest(inputs),
            inputs,
            expert_id,
        )
        if state.pending_operation is not None:
            if state.pending_operation != expected:
                raise RuntimeError("persisted operation intent does not match pipeline progress")
            return state, state.pending_operation
        persisted = self._persist(replace(state, pending_operation=expected))
        return persisted, expected

    def _persist(self, state: RoundState) -> RoundState:
        updated = replace(state, revision=state.revision + 1)
        self.store.save_state(updated)
        return updated

    def _result(self, state: RoundState) -> RoundResult:
        return RoundResult(
            round_id=state.round_id,
            trajectory=_required_trajectory(state),
            scores=state.scores,
            diagnosis=_required_diagnosis(state),
            verified=_required_diagnosis(state).score >= self._config.reliability_threshold,
            update=_required_update(state),
        )


def _operation_id(round_id: str, kind: OperationKind, expert_id: str | None = None) -> str:
    return f"{round_id}:{kind.value}" + (f":{expert_id}" if expert_id is not None else "")


def _replace_candidate(state: RoundState, index: int, candidate: CandidateState) -> RoundState:
    candidates = list(state.candidates)
    candidates[index] = candidate
    return replace(state, candidates=tuple(candidates))


def _validate_execution_output(
    selected_rollout: Rollout, expert: ExpertSpec, trajectory: Trajectory
) -> None:
    if (
        trajectory.rollout_id != selected_rollout.rollout_id
        or trajectory.actions != selected_rollout.actions
        or trajectory.actions.schema_name != expert.action_schema
    ):
        raise ValueError("executor returned a trajectory with invalid provenance")


def _has_receipt(state: RoundState, kind: OperationKind, inputs: JsonObject) -> bool:
    operation_id = _operation_id(state.round_id, kind)
    digest = input_digest(inputs)
    return any(
        receipt.kind is kind
        and receipt.operation_id == operation_id
        and receipt.input_digest == digest
        for receipt in state.receipts
    )


def _required_task(state: RoundState) -> TaskSpec:
    if state.task is None:
        raise RuntimeError("round is missing its task")
    return state.task


def _required_scene(state: RoundState) -> SceneRecord:
    if state.scene is None:
        raise RuntimeError("round is missing its scene")
    return state.scene


def _task_scene(state: RoundState) -> tuple[TaskSpec, SceneRecord]:
    return _required_task(state), _required_scene(state)


def _selected_candidate(state: RoundState) -> CandidateState:
    return next(
        candidate
        for candidate in state.candidates
        if candidate.expert.expert_id == state.selected_expert_id
    )


def _selected_expert(state: RoundState) -> ExpertSpec:
    return _selected_candidate(state).expert


def _selected_score(state: RoundState) -> CandidateScore:
    score = _selected_candidate(state).score
    if score is None:
        raise RuntimeError("selected candidate has no score")
    return score


def _required_actions(candidate: CandidateState) -> ActionChunk:
    if candidate.actions is None:
        raise RuntimeError("candidate has no actions")
    return candidate.actions


def _required_rollout(candidate: CandidateState) -> Rollout:
    if candidate.rollout is None:
        raise RuntimeError("candidate has no rollout")
    return candidate.rollout


def _required_trajectory(state: RoundState) -> Trajectory:
    if state.trajectory is None:
        raise RuntimeError("round has no trajectory")
    return state.trajectory


def _required_diagnosis(state: RoundState) -> Diagnosis:
    if state.diagnosis is None:
        raise RuntimeError("round has no diagnosis")
    return state.diagnosis


def _required_update(state: RoundState) -> UpdateDecision:
    if state.update is None:
        raise RuntimeError("round has no update")
    return state.update


def _candidate_error(error: _OperationFailure) -> str:
    """Return the already generic operation error; backend messages are never persisted."""
    return error.public_message


def _public_operation_error(error: Exception, kind: OperationKind) -> str:
    error_name = _PUBLIC_EXCEPTION_NAMES.get(type(error), "BackendError")
    rendered = f"{error_name}: {kind.value} failed"
    if len(rendered) > _MAX_PERSISTED_ERROR_LENGTH:
        raise AssertionError("generic operation error exceeded its fixed bound")
    return rendered


def _in_doubt_reason(intent: OperationIntent) -> str:
    prefix = "operation "
    suffix = f" ({intent.kind.value}) is in doubt"
    available = _MAX_PERSISTED_ERROR_LENGTH - len(prefix) - len(suffix)
    operation_id = intent.operation_id
    if len(operation_id) > available:
        digest = sha256(operation_id.encode("utf-8")).hexdigest()[:12]
        operation_id = f"{operation_id[: available - len(digest) - 1]}#{digest}"
    return f"{prefix}{operation_id}{suffix}"


def _experience_id(state: RoundState, trajectory: Trajectory, diagnosis: Diagnosis) -> str:
    task, scene = _task_scene(state)
    payload = json.dumps(
        {
            "diagnosis_label": diagnosis.label.value,
            "diagnosis_score": diagnosis.score,
            "round_id": state.round_id,
            "scene_id": scene.scene_id,
            "task_id": task.task_id,
            "trajectory_id": trajectory.trajectory_id,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"experience-{sha256(payload.encode('utf-8')).hexdigest()}"


def _experience_payload(record: ExperienceRecord) -> JsonObject:
    return {
        "record_id": record.record_id,
        "task_id": record.task_id,
        "scene_id": record.scene_id,
        "expert_id": record.expert_id,
        "score": score_to_json(record.score),
        "diagnosis": diagnosis_to_json(record.diagnosis),
        "artifact_refs": [str(path.resolve()) for path in record.artifact_refs],
    }


def _validate_update_enrichment(
    requested: UpdateDecision, returned: UpdateDecision
) -> UpdateDecision:
    identity_fields = (
        "update_id",
        "target",
        "task_id",
        "scene_id",
        "trajectory_id",
        "execution_mode",
        "expert_id",
    )
    if any(getattr(requested, field) != getattr(returned, field) for field in identity_fields):
        raise ValueError("trainer changed immutable update identity fields")
    if requested.training_dataset is not None and (
        returned.training_dataset != requested.training_dataset
    ):
        raise ValueError("trainer replaced an existing training dataset")
    if requested.command and returned.command != requested.command:
        raise ValueError("trainer replaced an existing training command")
    return returned


def _validate_executed_update(planned: UpdateDecision, executed: UpdateDecision) -> UpdateDecision:
    if executed != planned:
        raise ValueError("trainer execute must preserve the exact planned update")
    return executed


def _validate_round_id(round_id: str) -> None:
    if not isinstance(round_id, str):
        raise TypeError("round_id must be a string")
    candidate = Path(round_id)
    if (
        not round_id.strip()
        or round_id in {".", ".."}
        or candidate.is_absolute()
        or "/" in round_id
        or "\\" in round_id
        or ":" in round_id
        or candidate.name != round_id
    ):
        raise ValueError("round_id must be a safe single path component")

"""Immutable, backend-independent values used by the EvoWorld pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from typing import TypeAlias

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


def _required(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _unit_interval(value: float, field_name: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool) or not isfinite(value):
        raise ValueError(f"{field_name} must be a finite number between 0 and 1")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")


def _finite_json(value: JsonValue, field_name: str) -> None:
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{field_name} must contain only finite numbers")
    if isinstance(value, list):
        for item in value:
            _finite_json(item, field_name)
    elif isinstance(value, dict):
        for item in value.values():
            _finite_json(item, field_name)


def _input_digest(value: JsonObject) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("operation input_payload must be canonical finite JSON") from error
    return sha256(payload.encode("utf-8")).hexdigest()


class DiagnosisLabel(str, Enum):
    """The three actionable diagnosis branches and a no-action result."""

    TSM = "TSM"
    PF = "PF"
    WMF = "WMF"
    NONE = "NONE"


class GroundingMode(str, Enum):
    """The supported scene-selection rules."""

    NEAREST = "nearest"
    SOFTMAX = "softmax"


class UpdateTarget(str, Enum):
    """The single EvoWorld component selected for a diagnosis response."""

    NONE = "none"
    GROUNDING_ROUTER = "grounding_router"
    EXPERT_POLICY = "expert_policy"
    WORLD_MODEL = "world_model"


class ExecutionMode(str, Enum):
    """Whether an update decision is only planned or may be executed."""

    PLAN = "plan"
    EXECUTE = "execute"


class RoundStage(str, Enum):
    """Last durably completed method stage for a resumable round."""

    STARTED = "started"
    GROUNDED = "grounded"
    CANDIDATES_SCORED = "candidates_scored"
    EXECUTED = "executed"
    COMPLETE = "complete"
    IN_DOUBT = "in_doubt"


class OperationKind(str, Enum):
    """Idempotent external operation represented by a durable intent."""

    TASK = "task"
    ANCHOR = "anchor"
    ROUTE = "route"
    ACT = "act"
    IMAGINE = "imagine"
    SCORE = "score"
    EXECUTE = "execute"
    EVALUATE = "evaluate"
    TRAIN_PLAN = "train_plan"
    TRAIN_EXECUTE = "train_execute"
    MEMORY_APPEND = "memory_append"


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """A task selected for one EvoWorld round."""

    task_id: str
    instruction: str
    skill_hints: tuple[str, ...] = ()
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required(self.task_id, "task_id")
        _required(self.instruction, "instruction")
        _finite_json(self.metadata, "task metadata")


@dataclass(frozen=True, slots=True)
class SceneRecord:
    """Grounded observations and context for a task instance."""

    scene_id: str
    observation_paths: tuple[Path, ...] = ()
    context: JsonObject = field(default_factory=dict)
    embedding_ref: str | None = None

    def __post_init__(self) -> None:
        _required(self.scene_id, "scene_id")
        if self.embedding_ref is not None:
            _required(self.embedding_ref, "embedding_ref")
        _finite_json(self.context, "scene context")


@dataclass(frozen=True, slots=True)
class ExpertSpec:
    """Static description of one available VLA expert."""

    expert_id: str
    backend: str
    task_families: tuple[str, ...]
    checkpoint: Path | None
    action_schema: str

    def __post_init__(self) -> None:
        _required(self.expert_id, "expert_id")
        candidate = Path(self.expert_id)
        windows_base = self.expert_id.split(".", maxsplit=1)[0].upper()
        reserved = {"CON", "PRN", "AUX", "NUL"} | {
            f"{prefix}{index}" for prefix in ("COM", "LPT") for index in range(1, 10)
        }
        if (
            self.expert_id in {".", ".."}
            or candidate.is_absolute()
            or "/" in self.expert_id
            or "\\" in self.expert_id
            or candidate.name != self.expert_id
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.expert_id) is None
            or self.expert_id.endswith((".", " "))
            or windows_base in reserved
        ):
            raise ValueError("expert_id must be a safe portable path component")
        _required(self.backend, "backend")
        _required(self.action_schema, "action_schema")


@dataclass(frozen=True, slots=True)
class ActionChunk:
    """An ordered action proposal expressed in a named action schema."""

    expert_id: str
    actions: tuple[tuple[float, ...], ...]
    timestamps: tuple[float, ...]
    schema_name: str

    def __post_init__(self) -> None:
        _required(self.expert_id, "expert_id")
        _required(self.schema_name, "schema_name")
        if len(self.actions) != len(self.timestamps):
            raise ValueError("actions and timestamps must have the same length")
        if not self.actions or not self.actions[0]:
            raise ValueError("actions must be nonempty")
        action_dimension = len(self.actions[0])
        if any(len(action) != action_dimension for action in self.actions):
            raise ValueError("actions must use one consistent dimension")
        if any(
            not isinstance(value, int | float) or isinstance(value, bool) or not isfinite(value)
            for action in self.actions
            for value in action
        ):
            raise ValueError("actions must contain only finite numbers")
        if any(
            not isinstance(timestamp, int | float)
            or isinstance(timestamp, bool)
            or not isfinite(timestamp)
            for timestamp in self.timestamps
        ):
            raise ValueError("timestamps must contain only finite numbers")


@dataclass(frozen=True, slots=True)
class Rollout:
    """An imagined execution associated with one expert and scene."""

    rollout_id: str
    expert_id: str
    scene_id: str
    frame_paths: tuple[Path, ...]
    actions: ActionChunk
    metrics: JsonObject = field(default_factory=dict)
    valid: bool = True

    def __post_init__(self) -> None:
        _required(self.rollout_id, "rollout_id")
        _required(self.expert_id, "expert_id")
        _required(self.scene_id, "scene_id")
        if self.actions.expert_id != self.expert_id:
            raise ValueError("rollout actions must belong to the rollout expert")
        _finite_json(self.metrics, "rollout metrics")


@dataclass(frozen=True, slots=True)
class Trajectory:
    """An executed trajectory retained as artifacts rather than media bytes."""

    trajectory_id: str
    rollout_id: str
    frame_paths: tuple[Path, ...]
    actions: ActionChunk
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required(self.trajectory_id, "trajectory_id")
        _required(self.rollout_id, "rollout_id")
        _finite_json(self.metadata, "trajectory metadata")


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """Normalized router and world-model scores for a candidate rollout."""

    expert_id: str
    rollout_id: str | None
    router_score: float
    world_model_score: float
    joint_score: float
    valid: bool = True
    error: str | None = None

    def __post_init__(self) -> None:
        _required(self.expert_id, "expert_id")
        if self.valid and self.rollout_id is None:
            raise ValueError("rollout_id is required for a valid candidate")
        if self.valid and self.error is not None:
            raise ValueError("error must be None for a valid candidate")
        if self.rollout_id is not None:
            _required(self.rollout_id, "rollout_id")
        if not self.valid:
            if self.error is None:
                raise ValueError("error is required for an invalid candidate")
            _required(self.error, "error")
        _unit_interval(self.router_score, "router_score")
        _unit_interval(self.world_model_score, "world_model_score")
        _unit_interval(self.joint_score, "joint_score")

    @classmethod
    def from_components(
        cls,
        expert_id: str,
        rollout_id: str,
        router_score: float,
        world_model_score: float,
        weight: float,
    ) -> CandidateScore:
        """Build a score using the configured router/world-model convex weight."""
        _unit_interval(weight, "weight")
        return cls(
            expert_id=expert_id,
            rollout_id=rollout_id,
            router_score=router_score,
            world_model_score=world_model_score,
            joint_score=weight * router_score + (1.0 - weight) * world_model_score,
        )


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """Reliability assessment that selects a recovery branch when needed."""

    score: float
    label: DiagnosisLabel
    reason: str
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        _unit_interval(self.score, "score")
        _required(self.reason, "reason")
        _finite_json(self.metadata, "diagnosis metadata")


@dataclass(frozen=True, slots=True)
class ExperienceRecord:
    """A persisted task outcome available to future routing decisions."""

    record_id: str
    task_id: str
    scene_id: str
    expert_id: str
    score: CandidateScore
    diagnosis: Diagnosis
    artifact_refs: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        _required(self.record_id, "record_id")
        _required(self.task_id, "task_id")
        _required(self.scene_id, "scene_id")
        _required(self.expert_id, "expert_id")


@dataclass(frozen=True, slots=True)
class UpdateDecision:
    """A bounded fine-tuning request produced by a diagnosis branch."""

    update_id: str
    target: UpdateTarget
    task_id: str
    scene_id: str
    trajectory_id: str
    execution_mode: ExecutionMode
    expert_id: str | None = None
    training_dataset: Path | None = None
    command: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _required(self.update_id, "update_id")
        _required(self.task_id, "task_id")
        _required(self.scene_id, "scene_id")
        _required(self.trajectory_id, "trajectory_id")
        if not isinstance(self.target, UpdateTarget):
            raise ValueError("target must be an UpdateTarget")
        if not isinstance(self.execution_mode, ExecutionMode):
            raise ValueError("execution_mode must be an ExecutionMode")
        if self.target is UpdateTarget.EXPERT_POLICY:
            if self.expert_id is None:
                raise ValueError("expert_id is required for EXPERT_POLICY")
            _required(self.expert_id, "expert_id")
        elif self.expert_id is not None:
            raise ValueError("expert_id is only valid for EXPERT_POLICY")
        if self.training_dataset is not None and not isinstance(self.training_dataset, Path):
            raise ValueError("training_dataset must be a Path or None")
        if not isinstance(self.command, tuple):
            raise ValueError("command must be a tuple")
        for entry in self.command:
            _required(entry, "command entry")
        if self.target in {UpdateTarget.NONE, UpdateTarget.GROUNDING_ROUTER}:
            if self.training_dataset is not None or self.command:
                raise ValueError(f"{self.target.value} must not include training artifacts")


@dataclass(frozen=True, slots=True)
class OperationIntent:
    """Durable idempotency key written before invoking an external port."""

    operation_id: str
    kind: OperationKind
    input_digest: str
    input_payload: JsonObject
    expert_id: str | None = None

    def __post_init__(self) -> None:
        _required(self.operation_id, "operation_id")
        if not isinstance(self.kind, OperationKind):
            raise ValueError("kind must be an OperationKind")
        if re.fullmatch(r"[0-9a-f]{64}", self.input_digest) is None:
            raise ValueError("input_digest must be a lowercase SHA-256 digest")
        if not isinstance(self.input_payload, dict):
            raise ValueError("input_payload must be a JSON object")
        _finite_json(self.input_payload, "operation input_payload")
        if self.input_digest != _input_digest(self.input_payload):
            raise ValueError("input_digest does not match canonical input_payload")
        if self.expert_id is not None:
            _required(self.expert_id, "expert_id")


@dataclass(frozen=True, slots=True)
class OperationReceipt:
    """Evidence that one intent returned and its output entered round state."""

    operation_id: str
    kind: OperationKind
    input_digest: str
    input_payload: JsonObject

    def __post_init__(self) -> None:
        _required(self.operation_id, "operation_id")
        if not isinstance(self.kind, OperationKind):
            raise ValueError("kind must be an OperationKind")
        if re.fullmatch(r"[0-9a-f]{64}", self.input_digest) is None:
            raise ValueError("input_digest must be a lowercase SHA-256 digest")
        if not isinstance(self.input_payload, dict):
            raise ValueError("input_payload must be a JSON object")
        _finite_json(self.input_payload, "receipt input_payload")
        if self.input_digest != _input_digest(self.input_payload):
            raise ValueError("input_digest does not match canonical input_payload")


@dataclass(frozen=True, slots=True)
class CandidateState:
    """Typed, resumable progress for one routed expert candidate."""

    expert: ExpertSpec
    actions: ActionChunk | None = None
    rollout: Rollout | None = None
    score: CandidateScore | None = None

    def __post_init__(self) -> None:
        if self.actions is not None and self.actions.expert_id != self.expert.expert_id:
            raise ValueError("candidate actions must belong to the candidate expert")
        if self.rollout is not None:
            if self.actions is None or self.rollout.actions != self.actions:
                raise ValueError("candidate rollout must preserve candidate actions")
            if self.rollout.expert_id != self.expert.expert_id:
                raise ValueError("candidate rollout expert must match")
        if self.score is not None and self.score.expert_id != self.expert.expert_id:
            raise ValueError("candidate score expert must match")
        if self.score is not None and self.score.valid and self.rollout is None:
            raise ValueError("candidate valid score requires a durable rollout")
        if (
            self.score is not None
            and self.score.valid
            and self.rollout is not None
            and self.score.rollout_id != self.rollout.rollout_id
        ):
            raise ValueError("candidate score rollout must match candidate rollout")


@dataclass(frozen=True, slots=True)
class RoundState:
    """Typed domain snapshot at the last durably completed method stage."""

    round_id: str
    stage: RoundStage
    started_at: str
    seed: int
    reliability_threshold: float = 0.7
    execution_mode: ExecutionMode = ExecutionMode.PLAN
    revision: int = 0
    task: TaskSpec | None = None
    scene: SceneRecord | None = None
    experts: tuple[ExpertSpec, ...] = ()
    candidates: tuple[CandidateState, ...] = ()
    selected_expert_id: str | None = None
    trajectory: Trajectory | None = None
    diagnosis: Diagnosis | None = None
    update: UpdateDecision | None = None
    pending_operation: OperationIntent | None = None
    receipts: tuple[OperationReceipt, ...] = ()
    in_doubt_reason: str | None = None

    def __post_init__(self) -> None:
        _required(self.round_id, "round_id")
        if not isinstance(self.stage, RoundStage):
            raise ValueError("stage must be a RoundStage")
        _required(self.started_at, "started_at")
        try:
            started_at = datetime.fromisoformat(self.started_at)
        except ValueError as error:
            raise ValueError("started_at must be a timezone-aware ISO timestamp") from error
        if started_at.tzinfo is None or started_at.utcoffset() is None:
            raise ValueError("started_at must be a timezone-aware ISO timestamp")
        _unit_interval(self.reliability_threshold, "reliability_threshold")
        if not isinstance(self.execution_mode, ExecutionMode):
            raise ValueError("execution_mode must be an ExecutionMode")
        if (
            not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or self.revision < 0
        ):
            raise ValueError("revision must be a nonnegative integer")
        receipt_ids = [receipt.operation_id for receipt in self.receipts]
        if len(receipt_ids) != len(set(receipt_ids)):
            raise ValueError("operation receipt IDs must be unique")
        for receipt in self.receipts:
            _validate_operation_prerequisites(self, receipt)
            expected = f"{self.round_id}:{receipt.kind.value}"
            if receipt.operation_id != expected and not receipt.operation_id.startswith(
                f"{expected}:"
            ):
                raise ValueError("operation receipt does not belong to this round or kind")
            suffix = receipt.operation_id.removeprefix(expected)
            expert_id = suffix.removeprefix(":") if suffix else None
            expert_kinds = {
                OperationKind.ACT,
                OperationKind.IMAGINE,
                OperationKind.SCORE,
                OperationKind.EXECUTE,
            }
            if receipt.kind not in expert_kinds and expert_id is not None:
                raise ValueError("non-expert operation receipt has an unexpected suffix")
            candidate = next(
                (item for item in self.candidates if item.expert.expert_id == expert_id),
                None,
            )
            if receipt.kind in expert_kinds and candidate is None:
                raise ValueError("expert operation receipt has no matching candidate")
            if receipt.kind is OperationKind.EXECUTE and expert_id != self.selected_expert_id:
                raise ValueError("execute receipt must belong to the selected expert")
            failed = (
                candidate is not None and candidate.score is not None and not candidate.score.valid
            )
            output_present = {
                OperationKind.TASK: self.task is not None,
                OperationKind.ANCHOR: self.scene is not None,
                OperationKind.ROUTE: bool(self.experts),
                OperationKind.ACT: candidate is not None
                and (candidate.actions is not None or failed),
                OperationKind.IMAGINE: candidate is not None
                and (candidate.rollout is not None or failed),
                OperationKind.SCORE: candidate is not None and candidate.score is not None,
                OperationKind.EXECUTE: self.trajectory is not None,
                OperationKind.EVALUATE: self.diagnosis is not None,
                OperationKind.TRAIN_PLAN: self.update is not None,
                OperationKind.TRAIN_EXECUTE: self.update is not None,
                OperationKind.MEMORY_APPEND: (
                    self.trajectory is not None
                    and self.diagnosis is not None
                    and self.update is not None
                ),
            }[receipt.kind]
            if not output_present:
                raise ValueError("operation receipt has no associated durable output")
        if self.pending_operation is not None:
            _validate_operation_prerequisites(self, self.pending_operation)
            expected = f"{self.round_id}:{self.pending_operation.kind.value}"
            if (
                self.pending_operation.operation_id != expected
                and not self.pending_operation.operation_id.startswith(f"{expected}:")
            ):
                raise ValueError("pending operation does not belong to this round or kind")
            suffix = self.pending_operation.operation_id.removeprefix(expected)
            expert_kinds = {
                OperationKind.ACT,
                OperationKind.IMAGINE,
                OperationKind.SCORE,
                OperationKind.EXECUTE,
            }
            if self.pending_operation.kind in expert_kinds:
                expert_id = suffix.removeprefix(":") if suffix else None
                if self.pending_operation.expert_id != expert_id:
                    raise ValueError("pending operation expert_id does not match operation_id")
                if expert_id not in {candidate.expert.expert_id for candidate in self.candidates}:
                    raise ValueError("pending expert operation has no matching candidate")
            elif suffix or self.pending_operation.expert_id is not None:
                raise ValueError("pending non-expert operation has unexpected expert identity")
            if any(
                receipt.operation_id == self.pending_operation.operation_id
                for receipt in self.receipts
            ):
                raise ValueError("pending operation must not already have a receipt")
        if self.stage is RoundStage.IN_DOUBT:
            if self.pending_operation is None or self.in_doubt_reason is None:
                raise ValueError("in_doubt stage requires a pending operation and reason")
            _required(self.in_doubt_reason, "in_doubt_reason")
        elif self.in_doubt_reason is not None:
            raise ValueError("in_doubt_reason is only valid for the in_doubt stage")
        ordered_stages = (
            RoundStage.STARTED,
            RoundStage.GROUNDED,
            RoundStage.CANDIDATES_SCORED,
            RoundStage.EXECUTED,
            RoundStage.COMPLETE,
        )
        effective_stage = self.stage
        if self.stage is RoundStage.IN_DOUBT:
            assert self.pending_operation is not None
            effective_stage = {
                OperationKind.TASK: RoundStage.STARTED,
                OperationKind.ANCHOR: RoundStage.STARTED,
                OperationKind.ROUTE: RoundStage.GROUNDED,
                OperationKind.ACT: RoundStage.GROUNDED,
                OperationKind.IMAGINE: RoundStage.GROUNDED,
                OperationKind.SCORE: RoundStage.GROUNDED,
                OperationKind.EXECUTE: RoundStage.CANDIDATES_SCORED,
                OperationKind.EVALUATE: RoundStage.EXECUTED,
                OperationKind.TRAIN_PLAN: RoundStage.EXECUTED,
                OperationKind.TRAIN_EXECUTE: RoundStage.EXECUTED,
                OperationKind.MEMORY_APPEND: RoundStage.EXECUTED,
            }[self.pending_operation.kind]
        stage_index = ordered_stages.index(effective_stage)
        if stage_index >= ordered_stages.index(RoundStage.GROUNDED):
            if self.task is None or self.scene is None:
                raise ValueError("grounded stage requires task and scene")
        if stage_index >= ordered_stages.index(RoundStage.CANDIDATES_SCORED):
            if not self.experts or len(self.experts) != len(self.candidates):
                raise ValueError("candidates stage requires one score per expert")
            expert_ids = {expert.expert_id for expert in self.experts}
            if len(expert_ids) != len(self.experts):
                raise ValueError("candidate experts must be unique")
            if {candidate.expert.expert_id for candidate in self.candidates} != expert_ids:
                raise ValueError("candidate scores must match candidate experts")
            if any(candidate.score is None for candidate in self.candidates):
                raise ValueError("candidates stage requires completed scores")
            if self.selected_expert_id not in expert_ids:
                raise ValueError("selected_expert_id must identify a candidate expert")
            selected = next(
                candidate
                for candidate in self.candidates
                if candidate.expert.expert_id == self.selected_expert_id
            )
            if selected.score is None or not selected.score.valid:
                raise ValueError("selected candidate must be valid")
        elif self.selected_expert_id is not None:
            raise ValueError("selection is only valid after candidates are scored")

        if self.trajectory is not None:
            selected = next(
                candidate
                for candidate in self.candidates
                if candidate.expert.expert_id == self.selected_expert_id
            )
            if selected.score is None or self.trajectory.rollout_id != selected.score.rollout_id:
                raise ValueError("trajectory must execute the selected rollout")
            if (
                selected.rollout is None
                or self.trajectory.rollout_id != selected.rollout.rollout_id
            ):
                raise ValueError("trajectory rollout must directly match selected rollout")
            if selected.rollout is None or self.trajectory.actions != selected.rollout.actions:
                raise ValueError("trajectory actions must equal the selected rollout actions")
            if self.trajectory.actions.expert_id != self.selected_expert_id:
                raise ValueError("trajectory actions must belong to the selected expert")
        elif stage_index >= ordered_stages.index(RoundStage.EXECUTED):
            raise ValueError("executed stage requires a trajectory")

        if self.stage is RoundStage.COMPLETE:
            if self.diagnosis is None or self.update is None:
                raise ValueError("complete stage requires diagnosis and update outputs")
        if self.update is not None:
            if (
                self.task is None
                or self.scene is None
                or self.update.task_id != self.task.task_id
                or self.update.scene_id != self.scene.scene_id
                or self.trajectory is None
                or self.update.trajectory_id != self.trajectory.trajectory_id
            ):
                raise ValueError("update provenance must match the completed round")

    @property
    def scores(self) -> tuple[CandidateScore, ...]:
        """Return completed scores in routed candidate order."""
        return tuple(
            candidate.score for candidate in self.candidates if candidate.score is not None
        )


def _validate_operation_prerequisites(
    state: RoundState, operation: OperationIntent | OperationReceipt
) -> None:
    payload = operation.input_payload
    pending = isinstance(operation, OperationIntent)
    pending_in_doubt = pending and state.stage is RoundStage.IN_DOUBT
    expected_fields = {
        OperationKind.TASK: set(),
        OperationKind.ANCHOR: {"task"},
        OperationKind.ROUTE: {"task", "scene"},
        OperationKind.ACT: {"task", "scene", "expert"},
        OperationKind.IMAGINE: {"task", "scene", "expert", "actions"},
        OperationKind.SCORE: {"task", "rollout"},
        OperationKind.EXECUTE: {"task", "scene", "expert", "actions"},
        OperationKind.EVALUATE: {"task", "scene", "trajectory"},
        OperationKind.TRAIN_PLAN: {"update"},
        OperationKind.TRAIN_EXECUTE: {"update"},
        OperationKind.MEMORY_APPEND: {
            "record_id",
            "task_id",
            "scene_id",
            "expert_id",
            "score",
            "diagnosis",
            "artifact_refs",
        },
    }[operation.kind]
    if set(payload) != expected_fields:
        raise ValueError("operation input_payload fields do not match its kind")
    if operation.kind is OperationKind.TASK:
        if pending_in_doubt and state.task is not None:
            raise ValueError("pending task operation must not already have task output")
        return
    if operation.kind in {
        OperationKind.ANCHOR,
        OperationKind.ROUTE,
        OperationKind.ACT,
        OperationKind.IMAGINE,
        OperationKind.SCORE,
        OperationKind.EXECUTE,
        OperationKind.EVALUATE,
    }:
        if state.task is None:
            raise ValueError("operation requires a durable task")
        _require_nested_value(payload, "task", _task_json(state.task))
    if operation.kind in {
        OperationKind.ROUTE,
        OperationKind.ACT,
        OperationKind.IMAGINE,
        OperationKind.EXECUTE,
        OperationKind.EVALUATE,
    }:
        if state.scene is None:
            raise ValueError("operation requires a durable scene")
        _require_nested_value(payload, "scene", _scene_json(state.scene))
    if operation.kind in {
        OperationKind.ACT,
        OperationKind.IMAGINE,
        OperationKind.SCORE,
        OperationKind.EXECUTE,
    }:
        expert_id = operation.operation_id.rsplit(":", maxsplit=1)[-1]
        candidate = next(
            (item for item in state.candidates if item.expert.expert_id == expert_id),
            None,
        )
        if candidate is None:
            raise ValueError("operation requires a matching durable candidate")
        if operation.kind is not OperationKind.SCORE:
            _require_nested_value(payload, "expert", _expert_json(candidate.expert))
        if operation.kind in {OperationKind.IMAGINE, OperationKind.EXECUTE}:
            if candidate.actions is None or payload.get("actions") != _action_json(
                candidate.actions
            ):
                raise ValueError("operation actions do not match durable candidate actions")
        if operation.kind is OperationKind.SCORE:
            failed = candidate.score is not None and not candidate.score.valid
            if candidate.rollout is None and not failed:
                raise ValueError("score operation requires a durable rollout")
            if candidate.rollout is not None:
                _require_nested_value(payload, "rollout", _rollout_json(candidate.rollout))
    if operation.kind is OperationKind.EXECUTE:
        if state.selected_expert_id != expert_id:
            raise ValueError("execute operation requires the selected expert")
        if pending_in_doubt and state.trajectory is not None:
            raise ValueError("pending execute must not already have trajectory output")
    if operation.kind is OperationKind.EVALUATE:
        if state.trajectory is None:
            raise ValueError("evaluate operation requires a durable trajectory")
        _require_nested_value(payload, "trajectory", _trajectory_json(state.trajectory))
        if pending_in_doubt and state.diagnosis is not None:
            raise ValueError("pending evaluate must not already have diagnosis output")
    if operation.kind in {OperationKind.TRAIN_PLAN, OperationKind.TRAIN_EXECUTE}:
        update = payload.get("update")
        if not isinstance(update, dict):
            raise ValueError("training operation requires an update object")
        if state.diagnosis is None:
            raise ValueError("training operation requires a durable diagnosis")
        expected = _expected_update_json(state)
        if operation.kind is OperationKind.TRAIN_PLAN and update != expected:
            raise ValueError("train plan input must equal the diagnosis-derived decision")
        if (
            operation.kind is OperationKind.TRAIN_PLAN
            and pending_in_doubt
            and state.update is not None
        ):
            raise ValueError("pending train plan must not already have update output")
        if operation.kind is OperationKind.TRAIN_PLAN and state.update is not None:
            _validate_update_enrichment_payload(expected, _update_json(state.update))
        if operation.kind is OperationKind.TRAIN_EXECUTE:
            if state.execution_mode is not ExecutionMode.EXECUTE:
                raise ValueError("train execute is forbidden for a PLAN round")
            if state.update is None:
                raise ValueError("train execute requires the durable update plan")
            if state.update.target not in {
                UpdateTarget.EXPERT_POLICY,
                UpdateTarget.WORLD_MODEL,
            }:
                raise ValueError("train execute requires a GPU-relevant update target")
            planned = _update_json(state.update)
            _validate_update_enrichment_payload(expected, planned)
            if update != planned:
                raise ValueError("train execute input must exactly match the durable update plan")
    if operation.kind is OperationKind.MEMORY_APPEND:
        if (
            state.task is None
            or payload.get("task_id") != state.task.task_id
            or state.scene is None
            or payload.get("scene_id") != state.scene.scene_id
            or state.trajectory is None
            or state.diagnosis is None
            or state.update is None
            or payload.get("expert_id") != state.selected_expert_id
            or payload.get("score") != _score_json(_selected_state_score(state))
            or payload.get("diagnosis") != _diagnosis_json(state.diagnosis)
        ):
            raise ValueError("memory append input does not match durable round outputs")

    if not pending_in_doubt:
        return
    if operation.kind is OperationKind.ANCHOR and state.scene is not None:
        raise ValueError("pending anchor must not already have scene output")
    if operation.kind is OperationKind.ROUTE and state.experts:
        raise ValueError("pending route must not already have expert output")
    if operation.kind in {
        OperationKind.ACT,
        OperationKind.IMAGINE,
        OperationKind.SCORE,
    }:
        expert_id = operation.operation_id.rsplit(":", maxsplit=1)[-1]
        candidate = next(item for item in state.candidates if item.expert.expert_id == expert_id)
        output = {
            OperationKind.ACT: candidate.actions,
            OperationKind.IMAGINE: candidate.rollout,
            OperationKind.SCORE: candidate.score,
        }[operation.kind]
        if output is not None:
            raise ValueError("pending candidate operation already has durable output")


def _require_nested_value(payload: JsonObject, object_name: str, expected: JsonObject) -> None:
    if payload.get(object_name) != expected:
        raise ValueError(f"operation {object_name} does not match durable state")


def _action_json(action: ActionChunk) -> JsonObject:
    return {
        "expert_id": action.expert_id,
        "actions": [list(row) for row in action.actions],
        "timestamps": list(action.timestamps),
        "schema_name": action.schema_name,
    }


def _task_json(task: TaskSpec) -> JsonObject:
    return {
        "task_id": task.task_id,
        "instruction": task.instruction,
        "skill_hints": list(task.skill_hints),
        "metadata": task.metadata,
    }


def _scene_json(scene: SceneRecord) -> JsonObject:
    return {
        "scene_id": scene.scene_id,
        "observation_paths": [str(path) for path in scene.observation_paths],
        "context": scene.context,
        "embedding_ref": scene.embedding_ref,
    }


def _expert_json(expert: ExpertSpec) -> JsonObject:
    return {
        "expert_id": expert.expert_id,
        "backend": expert.backend,
        "task_families": list(expert.task_families),
        "checkpoint": str(expert.checkpoint) if expert.checkpoint is not None else None,
        "action_schema": expert.action_schema,
    }


def _rollout_json(rollout: Rollout) -> JsonObject:
    return {
        "rollout_id": rollout.rollout_id,
        "expert_id": rollout.expert_id,
        "scene_id": rollout.scene_id,
        "frame_paths": [str(path) for path in rollout.frame_paths],
        "actions": _action_json(rollout.actions),
        "metrics": rollout.metrics,
        "valid": rollout.valid,
    }


def _trajectory_json(trajectory: Trajectory) -> JsonObject:
    return {
        "trajectory_id": trajectory.trajectory_id,
        "rollout_id": trajectory.rollout_id,
        "frame_paths": [str(path) for path in trajectory.frame_paths],
        "actions": _action_json(trajectory.actions),
        "metadata": trajectory.metadata,
    }


def _score_json(score: CandidateScore) -> JsonObject:
    return {
        "expert_id": score.expert_id,
        "rollout_id": score.rollout_id,
        "router_score": score.router_score,
        "world_model_score": score.world_model_score,
        "joint_score": score.joint_score,
        "valid": score.valid,
        "error": score.error,
    }


def _diagnosis_json(diagnosis: Diagnosis) -> JsonObject:
    return {
        "score": diagnosis.score,
        "label": diagnosis.label.value,
        "reason": diagnosis.reason,
        "metadata": diagnosis.metadata,
    }


def _update_json(update: UpdateDecision) -> JsonObject:
    return {
        "update_id": update.update_id,
        "target": update.target.value,
        "task_id": update.task_id,
        "scene_id": update.scene_id,
        "trajectory_id": update.trajectory_id,
        "execution_mode": update.execution_mode.value,
        "expert_id": update.expert_id,
        "training_dataset": (
            str(update.training_dataset) if update.training_dataset is not None else None
        ),
        "command": list(update.command),
    }


def _expected_update_json(state: RoundState) -> JsonObject:
    execution_mode = state.execution_mode
    if (
        state.task is None
        or state.scene is None
        or state.trajectory is None
        or state.diagnosis is None
    ):
        raise ValueError("training operation is missing durable round prerequisites")
    if state.diagnosis.score >= state.reliability_threshold:
        if state.diagnosis.label is not DiagnosisLabel.NONE:
            raise ValueError("reliable diagnosis must use the NONE label")
        target = UpdateTarget.NONE
    else:
        targets = {
            DiagnosisLabel.TSM: UpdateTarget.GROUNDING_ROUTER,
            DiagnosisLabel.PF: UpdateTarget.EXPERT_POLICY,
            DiagnosisLabel.WMF: UpdateTarget.WORLD_MODEL,
        }
        if state.diagnosis.label not in targets:
            raise ValueError("unreliable diagnosis must identify a failure target")
        target = targets[state.diagnosis.label]
    expert_id = state.selected_expert_id if target is UpdateTarget.EXPERT_POLICY else None
    identity: JsonObject = {
        "execution_mode": execution_mode.value,
        "expert_id": expert_id,
        "scene_id": state.scene.scene_id,
        "target": target.value,
        "task_id": state.task.task_id,
        "trajectory_id": state.trajectory.trajectory_id,
    }
    update_id = f"update-{sha256(json.dumps(identity, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode('utf-8')).hexdigest()}"
    return {
        "update_id": update_id,
        "target": target.value,
        "task_id": state.task.task_id,
        "scene_id": state.scene.scene_id,
        "trajectory_id": state.trajectory.trajectory_id,
        "execution_mode": execution_mode.value,
        "expert_id": expert_id,
        "training_dataset": None,
        "command": [],
    }


def _validate_update_enrichment_payload(requested: JsonObject, returned: JsonObject) -> None:
    identity_fields = {
        "update_id",
        "target",
        "task_id",
        "scene_id",
        "trajectory_id",
        "execution_mode",
        "expert_id",
    }
    if any(requested.get(field) != returned.get(field) for field in identity_fields):
        raise ValueError("training output changed immutable decision identity")
    if requested.get("training_dataset") is not None and (
        returned.get("training_dataset") != requested.get("training_dataset")
    ):
        raise ValueError("training output replaced the requested dataset")
    requested_command = requested.get("command")
    if requested_command and returned.get("command") != requested_command:
        raise ValueError("training output replaced the requested command")


def _selected_state_score(state: RoundState) -> CandidateScore:
    selected = next(
        candidate
        for candidate in state.candidates
        if candidate.expert.expert_id == state.selected_expert_id
    )
    if selected.score is None:
        raise ValueError("selected candidate has no durable score")
    return selected.score


@dataclass(frozen=True, slots=True)
class RoundResult:
    """The terminal result and optional update request for one round."""

    round_id: str
    trajectory: Trajectory
    scores: tuple[CandidateScore, ...]
    diagnosis: Diagnosis
    verified: bool
    update: UpdateDecision | None = None

    def __post_init__(self) -> None:
        _required(self.round_id, "round_id")
        if not isinstance(self.verified, bool):
            raise ValueError("verified must be a boolean")

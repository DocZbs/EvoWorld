"""Canonical, schema-versioned JSON codecs for EvoWorld domain values."""

from __future__ import annotations

import json
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Mapping, cast

from evoworld.contracts import (
    ActionChunk,
    CandidateScore,
    CandidateState,
    Diagnosis,
    DiagnosisLabel,
    ExecutionMode,
    ExpertSpec,
    JsonObject,
    JsonValue,
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

STATE_SCHEMA_VERSION = 6


class CodecError(ValueError):
    """Base error for invalid persisted domain data."""


class StateDecodeError(CodecError):
    """State JSON is truncated, corrupt, or violates domain contracts."""


class UnsupportedStateSchemaError(StateDecodeError):
    """State cannot be resumed because its schema is not supported."""


def canonical_json(value: JsonValue, *, indent: int | None = None) -> str:
    """Encode finite JSON with deterministic key order and no NaN extensions."""
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            separators=None if indent else (",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise CodecError("value is not canonical finite JSON") from error


def decode_json_object(text: str, *, source: str) -> JsonObject:
    """Decode one JSON object with a source-aware error."""
    try:
        return _json_object(cast(object, json.loads(text, parse_constant=_reject_constant)), source)
    except (json.JSONDecodeError, ValueError, TypeError) as error:
        raise StateDecodeError(f"invalid JSON object in {source}") from error


def input_digest(value: JsonObject) -> str:
    """Return the stable SHA-256 digest used to bind operation IDs to input."""
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def task_to_json(task: TaskSpec) -> JsonObject:
    return {
        "task_id": task.task_id,
        "instruction": task.instruction,
        "skill_hints": list(task.skill_hints),
        "metadata": task.metadata,
    }


def task_from_json(value: JsonObject) -> TaskSpec:
    _expect_fields(value, {"task_id", "instruction", "skill_hints", "metadata"}, "task")
    return TaskSpec(
        task_id=_text(value.get("task_id"), "task_id"),
        instruction=_text(value.get("instruction"), "instruction"),
        skill_hints=tuple(
            _text(item, "skill hint")
            for item in _json_list(value.get("skill_hints"), "skill_hints")
        ),
        metadata=_json_object(value.get("metadata"), "task metadata"),
    )


def scene_to_json(scene: SceneRecord) -> JsonObject:
    return {
        "scene_id": scene.scene_id,
        "observation_paths": [str(path) for path in scene.observation_paths],
        "context": scene.context,
        "embedding_ref": scene.embedding_ref,
    }


def scene_from_json(value: JsonObject) -> SceneRecord:
    _expect_fields(
        value,
        {"scene_id", "observation_paths", "context", "embedding_ref"},
        "scene",
    )
    return SceneRecord(
        scene_id=_text(value.get("scene_id"), "scene_id"),
        observation_paths=tuple(
            Path(_text(item, "observation path"))
            for item in _json_list(value.get("observation_paths"), "observation_paths")
        ),
        context=_json_object(value.get("context"), "scene context"),
        embedding_ref=_optional_text(value.get("embedding_ref"), "embedding_ref"),
    )


def expert_to_json(expert: ExpertSpec) -> JsonObject:
    return {
        "expert_id": expert.expert_id,
        "backend": expert.backend,
        "task_families": list(expert.task_families),
        "checkpoint": str(expert.checkpoint) if expert.checkpoint is not None else None,
        "action_schema": expert.action_schema,
    }


def expert_from_json(value: JsonObject) -> ExpertSpec:
    _expect_fields(
        value,
        {"expert_id", "backend", "task_families", "checkpoint", "action_schema"},
        "expert",
    )
    checkpoint = _optional_text(value.get("checkpoint"), "checkpoint")
    return ExpertSpec(
        expert_id=_text(value.get("expert_id"), "expert_id"),
        backend=_text(value.get("backend"), "backend"),
        task_families=tuple(
            _text(item, "task family")
            for item in _json_list(value.get("task_families"), "task_families")
        ),
        checkpoint=Path(checkpoint) if checkpoint is not None else None,
        action_schema=_text(value.get("action_schema"), "action_schema"),
    )


def action_to_json(action: ActionChunk) -> JsonObject:
    return {
        "expert_id": action.expert_id,
        "actions": [list(row) for row in action.actions],
        "timestamps": list(action.timestamps),
        "schema_name": action.schema_name,
    }


def action_from_json(value: JsonObject) -> ActionChunk:
    _expect_fields(value, {"expert_id", "actions", "timestamps", "schema_name"}, "action")
    return ActionChunk(
        expert_id=_text(value.get("expert_id"), "action expert_id"),
        actions=tuple(
            tuple(_number(item, "action value") for item in _json_list(row, "action row"))
            for row in _json_list(value.get("actions"), "actions")
        ),
        timestamps=tuple(
            _number(item, "timestamp") for item in _json_list(value.get("timestamps"), "timestamps")
        ),
        schema_name=_text(value.get("schema_name"), "schema_name"),
    )


def rollout_to_json(rollout: Rollout) -> JsonObject:
    return {
        "rollout_id": rollout.rollout_id,
        "expert_id": rollout.expert_id,
        "scene_id": rollout.scene_id,
        "frame_paths": [str(path) for path in rollout.frame_paths],
        "actions": action_to_json(rollout.actions),
        "metrics": rollout.metrics,
        "valid": rollout.valid,
    }


def rollout_from_json(value: JsonObject) -> Rollout:
    _expect_fields(
        value,
        {"rollout_id", "expert_id", "scene_id", "frame_paths", "actions", "metrics", "valid"},
        "rollout",
    )
    return Rollout(
        rollout_id=_text(value.get("rollout_id"), "rollout_id"),
        expert_id=_text(value.get("expert_id"), "rollout expert_id"),
        scene_id=_text(value.get("scene_id"), "rollout scene_id"),
        frame_paths=tuple(
            Path(_text(item, "rollout frame path"))
            for item in _json_list(value.get("frame_paths"), "rollout frame paths")
        ),
        actions=action_from_json(_json_object(value.get("actions"), "rollout actions")),
        metrics=_json_object(value.get("metrics"), "rollout metrics"),
        valid=_boolean(value.get("valid"), "rollout valid"),
    )


def score_to_json(score: CandidateScore) -> JsonObject:
    return {
        "expert_id": score.expert_id,
        "rollout_id": score.rollout_id,
        "router_score": score.router_score,
        "world_model_score": score.world_model_score,
        "joint_score": score.joint_score,
        "valid": score.valid,
        "error": score.error,
    }


def score_from_json(value: JsonObject) -> CandidateScore:
    _expect_fields(
        value,
        {
            "expert_id",
            "rollout_id",
            "router_score",
            "world_model_score",
            "joint_score",
            "valid",
            "error",
        },
        "score",
    )
    return CandidateScore(
        expert_id=_text(value.get("expert_id"), "score expert_id"),
        rollout_id=_optional_text(value.get("rollout_id"), "score rollout_id"),
        router_score=_number(value.get("router_score"), "router_score"),
        world_model_score=_number(value.get("world_model_score"), "world_model_score"),
        joint_score=_number(value.get("joint_score"), "joint_score"),
        valid=_boolean(value.get("valid"), "valid"),
        error=_optional_text(value.get("error"), "score error"),
    )


def candidate_to_json(candidate: CandidateState) -> JsonObject:
    return {
        "expert": expert_to_json(candidate.expert),
        "actions": action_to_json(candidate.actions) if candidate.actions is not None else None,
        "rollout": (rollout_to_json(candidate.rollout) if candidate.rollout is not None else None),
        "score": score_to_json(candidate.score) if candidate.score is not None else None,
    }


def candidate_from_json(value: JsonObject) -> CandidateState:
    _expect_fields(value, {"expert", "actions", "rollout", "score"}, "candidate")
    actions = value.get("actions")
    rollout = value.get("rollout")
    score = value.get("score")
    return CandidateState(
        expert=expert_from_json(_json_object(value.get("expert"), "candidate expert")),
        actions=(
            action_from_json(_json_object(actions, "candidate actions"))
            if actions is not None
            else None
        ),
        rollout=(
            rollout_from_json(_json_object(rollout, "candidate rollout"))
            if rollout is not None
            else None
        ),
        score=(
            score_from_json(_json_object(score, "candidate score")) if score is not None else None
        ),
    )


def trajectory_to_json(trajectory: Trajectory) -> JsonObject:
    return {
        "trajectory_id": trajectory.trajectory_id,
        "rollout_id": trajectory.rollout_id,
        "frame_paths": [str(path) for path in trajectory.frame_paths],
        "actions": action_to_json(trajectory.actions),
        "metadata": trajectory.metadata,
    }


def trajectory_from_json(value: JsonObject) -> Trajectory:
    _expect_fields(
        value,
        {"trajectory_id", "rollout_id", "frame_paths", "actions", "metadata"},
        "trajectory",
    )
    return Trajectory(
        trajectory_id=_text(value.get("trajectory_id"), "trajectory_id"),
        rollout_id=_text(value.get("rollout_id"), "trajectory rollout_id"),
        frame_paths=tuple(
            Path(_text(item, "trajectory frame path"))
            for item in _json_list(value.get("frame_paths"), "trajectory frame paths")
        ),
        actions=action_from_json(_json_object(value.get("actions"), "trajectory actions")),
        metadata=_json_object(value.get("metadata"), "trajectory metadata"),
    )


def diagnosis_to_json(diagnosis: Diagnosis) -> JsonObject:
    return {
        "score": diagnosis.score,
        "label": diagnosis.label.value,
        "reason": diagnosis.reason,
        "metadata": diagnosis.metadata,
    }


def diagnosis_from_json(value: JsonObject) -> Diagnosis:
    _expect_fields(value, {"score", "label", "reason", "metadata"}, "diagnosis")
    return Diagnosis(
        score=_number(value.get("score"), "diagnosis score"),
        label=DiagnosisLabel(_text(value.get("label"), "diagnosis label")),
        reason=_text(value.get("reason"), "diagnosis reason"),
        metadata=_json_object(value.get("metadata"), "diagnosis metadata"),
    )


def update_to_json(update: UpdateDecision) -> JsonObject:
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


def update_from_json(value: JsonObject) -> UpdateDecision:
    _expect_fields(
        value,
        {
            "update_id",
            "target",
            "task_id",
            "scene_id",
            "trajectory_id",
            "execution_mode",
            "expert_id",
            "training_dataset",
            "command",
        },
        "update",
    )
    training_dataset = _optional_text(value.get("training_dataset"), "training_dataset")
    return UpdateDecision(
        update_id=_text(value.get("update_id"), "update_id"),
        target=UpdateTarget(_text(value.get("target"), "update target")),
        task_id=_text(value.get("task_id"), "update task_id"),
        scene_id=_text(value.get("scene_id"), "update scene_id"),
        trajectory_id=_text(value.get("trajectory_id"), "update trajectory_id"),
        execution_mode=ExecutionMode(_text(value.get("execution_mode"), "execution_mode")),
        expert_id=_optional_text(value.get("expert_id"), "update expert_id"),
        training_dataset=Path(training_dataset) if training_dataset is not None else None,
        command=tuple(
            _text(item, "command entry") for item in _json_list(value.get("command"), "command")
        ),
    )


def round_state_to_json(state: RoundState) -> JsonObject:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "round_id": state.round_id,
        "stage": state.stage.value,
        "started_at": state.started_at,
        "seed": state.seed,
        "reliability_threshold": state.reliability_threshold,
        "execution_mode": state.execution_mode.value,
        "revision": state.revision,
        "task": task_to_json(state.task) if state.task is not None else None,
        "scene": scene_to_json(state.scene) if state.scene is not None else None,
        "experts": [expert_to_json(expert) for expert in state.experts],
        "candidates": [candidate_to_json(candidate) for candidate in state.candidates],
        "selected_expert_id": state.selected_expert_id,
        "trajectory": (
            trajectory_to_json(state.trajectory) if state.trajectory is not None else None
        ),
        "diagnosis": (diagnosis_to_json(state.diagnosis) if state.diagnosis is not None else None),
        "update": update_to_json(state.update) if state.update is not None else None,
        "pending_operation": (
            _intent_to_json(state.pending_operation)
            if state.pending_operation is not None
            else None
        ),
        "receipts": [_receipt_to_json(receipt) for receipt in state.receipts],
        "in_doubt_reason": state.in_doubt_reason,
    }


def round_state_from_json(value: JsonObject) -> RoundState:
    schema = value.get("schema_version")
    if schema is None:
        raise UnsupportedStateSchemaError(
            "legacy state has no schema_version and is not resumable; start a new run"
        )
    if not isinstance(schema, int) or isinstance(schema, bool) or schema != STATE_SCHEMA_VERSION:
        raise UnsupportedStateSchemaError(
            f"state schema {schema!r} is unsupported; start a new run"
        )
    try:
        _expect_fields(
            value,
            {
                "schema_version",
                "round_id",
                "stage",
                "started_at",
                "seed",
                "revision",
                "reliability_threshold",
                "execution_mode",
                "task",
                "scene",
                "experts",
                "candidates",
                "selected_expert_id",
                "trajectory",
                "diagnosis",
                "update",
                "pending_operation",
                "receipts",
                "in_doubt_reason",
            },
            "round state",
        )
        task = value.get("task")
        scene = value.get("scene")
        trajectory = value.get("trajectory")
        diagnosis = value.get("diagnosis")
        update = value.get("update")
        pending = value.get("pending_operation")
        return RoundState(
            round_id=_text(value.get("round_id"), "round_id"),
            stage=RoundStage(_text(value.get("stage"), "stage")),
            started_at=_text(value.get("started_at"), "started_at"),
            seed=_integer(value.get("seed"), "seed"),
            reliability_threshold=_number(
                value.get("reliability_threshold"), "reliability_threshold"
            ),
            execution_mode=ExecutionMode(
                _text(value.get("execution_mode"), "round execution_mode")
            ),
            revision=_integer(value.get("revision"), "revision"),
            task=(task_from_json(_json_object(task, "task")) if task is not None else None),
            scene=(scene_from_json(_json_object(scene, "scene")) if scene is not None else None),
            experts=tuple(
                expert_from_json(_json_object(item, "expert"))
                for item in _json_list(value.get("experts"), "experts")
            ),
            candidates=tuple(
                candidate_from_json(_json_object(item, "candidate"))
                for item in _json_list(value.get("candidates"), "candidates")
            ),
            selected_expert_id=_optional_text(
                value.get("selected_expert_id"), "selected_expert_id"
            ),
            trajectory=(
                trajectory_from_json(_json_object(trajectory, "trajectory"))
                if trajectory is not None
                else None
            ),
            diagnosis=(
                diagnosis_from_json(_json_object(diagnosis, "diagnosis"))
                if diagnosis is not None
                else None
            ),
            update=(
                update_from_json(_json_object(update, "update")) if update is not None else None
            ),
            pending_operation=(
                _intent_from_json(_json_object(pending, "pending operation"))
                if pending is not None
                else None
            ),
            receipts=tuple(
                _receipt_from_json(_json_object(item, "operation receipt"))
                for item in _json_list(value.get("receipts"), "receipts")
            ),
            in_doubt_reason=_optional_text(value.get("in_doubt_reason"), "in_doubt_reason"),
        )
    except (StopIteration, TypeError, ValueError) as error:
        if isinstance(error, UnsupportedStateSchemaError):
            raise
        raise StateDecodeError("state violates schema or domain contracts") from error


def _intent_to_json(intent: OperationIntent) -> JsonObject:
    return {
        "operation_id": intent.operation_id,
        "kind": intent.kind.value,
        "input_digest": intent.input_digest,
        "input_payload": intent.input_payload,
        "expert_id": intent.expert_id,
    }


def _intent_from_json(value: JsonObject) -> OperationIntent:
    _expect_fields(
        value,
        {"operation_id", "kind", "input_digest", "input_payload", "expert_id"},
        "intent",
    )
    return OperationIntent(
        operation_id=_text(value.get("operation_id"), "operation_id"),
        kind=OperationKind(_text(value.get("kind"), "operation kind")),
        input_digest=_text(value.get("input_digest"), "operation input_digest"),
        input_payload=_json_object(value.get("input_payload"), "operation input_payload"),
        expert_id=_optional_text(value.get("expert_id"), "operation expert_id"),
    )


def _receipt_to_json(receipt: OperationReceipt) -> JsonObject:
    return {
        "operation_id": receipt.operation_id,
        "kind": receipt.kind.value,
        "input_digest": receipt.input_digest,
        "input_payload": receipt.input_payload,
    }


def _receipt_from_json(value: JsonObject) -> OperationReceipt:
    _expect_fields(
        value,
        {"operation_id", "kind", "input_digest", "input_payload"},
        "receipt",
    )
    return OperationReceipt(
        operation_id=_text(value.get("operation_id"), "receipt operation_id"),
        kind=OperationKind(_text(value.get("kind"), "receipt operation kind")),
        input_digest=_text(value.get("input_digest"), "receipt input_digest"),
        input_payload=_json_object(value.get("input_payload"), "receipt input_payload"),
    )


def _json_object(value: object, field_name: str) -> JsonObject:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be a JSON object")
    return {key: _json_value(item, field_name) for key, item in value.items()}


def _json_list(value: object, field_name: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON array")
    return [_json_value(item, field_name) for item in value]


def _json_value(value: object, field_name: str) -> JsonValue:
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{field_name} contains a non-finite number")
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_json_value(item, field_name) for item in value]
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return {key: _json_value(item, field_name) for key, item in value.items()}
    raise ValueError(f"{field_name} contains a non-JSON value")


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _optional_text(value: object, field_name: str) -> str | None:
    return None if value is None else _text(value, field_name)


def _number(value: object, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not isfinite(value):
        raise ValueError(f"{field_name} must be a finite number")
    return float(value)


def _integer(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _boolean(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")


def _expect_fields(value: JsonObject, expected: set[str], field_name: str) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown or missing:
        details: list[str] = []
        if unknown:
            details.append(f"unknown fields: {', '.join(sorted(unknown))}")
        if missing:
            details.append(f"missing fields: {', '.join(sorted(missing))}")
        raise ValueError(f"{field_name} has invalid fields ({'; '.join(details)})")

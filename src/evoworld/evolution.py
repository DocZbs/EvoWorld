"""Pure diagnosis-to-update decisions for the EvoWorld evolution stage."""

from __future__ import annotations

from hashlib import sha256
from json import dumps
from math import isfinite

from evoworld.contracts import (
    Diagnosis,
    DiagnosisLabel,
    ExecutionMode,
    ExpertSpec,
    SceneRecord,
    TaskSpec,
    Trajectory,
    UpdateDecision,
    UpdateTarget,
)

_FAILURE_TARGETS = {
    DiagnosisLabel.TSM: UpdateTarget.GROUNDING_ROUTER,
    DiagnosisLabel.PF: UpdateTarget.EXPERT_POLICY,
    DiagnosisLabel.WMF: UpdateTarget.WORLD_MODEL,
}


def decide_update(
    *,
    diagnosis: Diagnosis,
    reliability_threshold: float,
    task: TaskSpec,
    scene: SceneRecord,
    expert: ExpertSpec,
    trajectory: Trajectory,
    execution_mode: ExecutionMode,
) -> UpdateDecision:
    """Return the one component that should respond to a trajectory diagnosis.

    This function only produces a traceable plan. It never imports or invokes a
    training backend, so grounding repairs cannot accidentally allocate GPU work.
    """
    _validate_threshold(reliability_threshold)
    target = _select_target(diagnosis, reliability_threshold)
    expert_id = expert.expert_id if target is UpdateTarget.EXPERT_POLICY else None
    return UpdateDecision(
        update_id=_update_id(task, scene, trajectory, target, expert_id, execution_mode),
        target=target,
        task_id=task.task_id,
        scene_id=scene.scene_id,
        trajectory_id=trajectory.trajectory_id,
        expert_id=expert_id,
        execution_mode=execution_mode,
    )


def _validate_threshold(reliability_threshold: float) -> None:
    if (
        not isinstance(reliability_threshold, float | int)
        or isinstance(reliability_threshold, bool)
        or not isfinite(reliability_threshold)
        or not 0.0 <= reliability_threshold <= 1.0
    ):
        raise ValueError("reliability_threshold must be a finite value between 0 and 1")


def _select_target(diagnosis: Diagnosis, reliability_threshold: float) -> UpdateTarget:
    if diagnosis.score >= reliability_threshold:
        if diagnosis.label is not DiagnosisLabel.NONE:
            raise ValueError("diagnosis label must be NONE at or above the reliability threshold")
        return UpdateTarget.NONE
    if diagnosis.label is DiagnosisLabel.NONE:
        raise ValueError("diagnosis label must identify a failure below the reliability threshold")
    return _FAILURE_TARGETS[diagnosis.label]


def _update_id(
    task: TaskSpec,
    scene: SceneRecord,
    trajectory: Trajectory,
    target: UpdateTarget,
    expert_id: str | None,
    execution_mode: ExecutionMode,
) -> str:
    identity_payload = dumps(
        {
            "execution_mode": execution_mode.value,
            "expert_id": expert_id,
            "scene_id": scene.scene_id,
            "target": target.value,
            "task_id": task.task_id,
            "trajectory_id": trajectory.trajectory_id,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"update-{sha256(identity_payload.encode('utf-8')).hexdigest()}"

"""Backend-agnostic protocols that connect EvoWorld domain services.

Every external operation receives a stable ``operation_id``. Implementations bind
that key to canonical input and provide at-least-once delivery. Physical adapters
must surface an explicit in-doubt result when replay safety cannot be proven.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from evoworld.contracts import (
    ActionChunk,
    CandidateScore,
    Diagnosis,
    ExperienceRecord,
    ExpertSpec,
    JsonObject,
    Rollout,
    RoundState,
    SceneRecord,
    TaskSpec,
    Trajectory,
    UpdateDecision,
)


class OperationConflictError(RuntimeError):
    """An operation ID was reused with different canonical input."""


class OperationInDoubtError(RuntimeError):
    """A physical operation was claimed but completion cannot be proven."""


@runtime_checkable
class TaskSource(Protocol):
    """Supplies the next task to run."""

    def next_task(self, *, operation_id: str) -> TaskSpec:
        """Return the task idempotently for the stable operation ID."""
        ...


@runtime_checkable
class SceneAnchor(Protocol):
    """Grounds a task in a scene."""

    def anchor(self, task: TaskSpec, *, operation_id: str) -> SceneRecord: ...


@runtime_checkable
class Router(Protocol):
    """Selects an ordered, small set of experts for a scene."""

    def shortlist(
        self,
        task: TaskSpec,
        scene: SceneRecord,
        memory: ExperienceMemory,
        *,
        operation_id: str,
    ) -> Sequence[ExpertSpec]: ...


@runtime_checkable
class ExpertPolicy(Protocol):
    """Produces an action chunk from a task observation."""

    def act(
        self, task: TaskSpec, observation: SceneRecord, *, operation_id: str
    ) -> ActionChunk: ...


@runtime_checkable
class ActionProjector(Protocol):
    """Projects one immutable raw expert chunk into the world-model action space."""

    @property
    def identity(self) -> str: ...

    @property
    def content_digest(self) -> str: ...

    @property
    def source_schema(self) -> str: ...

    @property
    def target_schema(self) -> str: ...

    def project(
        self,
        task: TaskSpec,
        observation: SceneRecord,
        actions: ActionChunk,
        *,
        operation_id: str,
    ) -> ActionChunk: ...


@runtime_checkable
class WorldModel(Protocol):
    """Imagines a candidate rollout before physical execution."""

    def imagine(
        self,
        task: TaskSpec,
        scene: SceneRecord,
        expert: ExpertSpec,
        actions: ActionChunk,
        *,
        operation_id: str,
    ) -> Rollout: ...


@runtime_checkable
class RolloutScorer(Protocol):
    """Scores an imagined rollout."""

    def score(self, task: TaskSpec, rollout: Rollout, *, operation_id: str) -> CandidateScore: ...


@runtime_checkable
class Executor(Protocol):
    """Executes a chosen expert in the anchored scene."""

    def execute(
        self,
        task: TaskSpec,
        scene: SceneRecord,
        expert: ExpertSpec,
        actions: ActionChunk,
        *,
        operation_id: str,
    ) -> Trajectory: ...


@runtime_checkable
class TrajectoryEvaluator(Protocol):
    """Assesses a completed trajectory using prior experience."""

    def evaluate(
        self,
        task: TaskSpec,
        scene: SceneRecord,
        trajectory: Trajectory,
        memory: ExperienceMemory,
        *,
        operation_id: str,
    ) -> Diagnosis: ...


@runtime_checkable
class Trainer(Protocol):
    """Plans and explicitly executes a fine-tuning update."""

    def plan(self, update: UpdateDecision, *, operation_id: str) -> UpdateDecision: ...

    def execute(self, plan: UpdateDecision, *, operation_id: str) -> UpdateDecision: ...


@runtime_checkable
class ExperienceMemory(Protocol):
    """Stores and retrieves prior task outcomes."""

    def lookup(
        self, task: TaskSpec, scene: SceneRecord, *, operation_id: str
    ) -> Sequence[ExperienceRecord]: ...

    def append(self, record: ExperienceRecord, *, operation_id: str) -> None: ...


@runtime_checkable
class RunStore(Protocol):
    """Persists artifacts and resumable round state."""

    @property
    def run_dir(self) -> Path:
        """Absolute directory containing this run's durable artifacts."""

    def write_artifact(self, round_id: str, name: str, metadata: JsonObject) -> Path: ...

    def save_state(self, state: RoundState) -> None: ...

    def load_state(self, round_id: str) -> RoundState | None: ...

    def claim_round(self, state: RoundState) -> None: ...

    def append_task(self, round_id: str, operation_id: str, task: TaskSpec) -> None: ...

    def write_verified(self, trajectory: Trajectory, provenance: JsonObject) -> Path: ...

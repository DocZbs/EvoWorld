"""Tests for memory-aware expert ordering and rollout selection."""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

from evoworld.contracts import (
    CandidateScore,
    Diagnosis,
    DiagnosisLabel,
    ExperienceRecord,
    ExpertSpec,
    SceneRecord,
    TaskSpec,
)
from evoworld.routing import (
    MemoryAwareRouter,
    NoValidCandidateError,
    experience_score,
    select_candidate,
)


def _expert(expert_id: str, *, schema: str = "joint_velocity") -> ExpertSpec:
    return ExpertSpec(
        expert_id=expert_id,
        backend="deterministic",
        task_families=("pick",),
        checkpoint=None,
        action_schema=schema,
    )


def _task() -> TaskSpec:
    return TaskSpec(task_id="task-1", instruction="pick the block")


def _scene() -> SceneRecord:
    return SceneRecord(scene_id="scene-1")


def _candidate(expert_id: str, joint: float, *, valid: bool = True) -> CandidateScore:
    return CandidateScore(
        expert_id=expert_id,
        rollout_id=f"rollout-{expert_id}" if valid else None,
        router_score=joint if valid else 0.0,
        world_model_score=joint if valid else 0.0,
        joint_score=joint if valid else 0.0,
        valid=valid,
        error=None if valid else "rollout failed",
    )


def _experience(
    expert_id: str,
    joint: float,
    *,
    task_id: str = "task-1",
    scene_id: str = "scene-1",
    label: DiagnosisLabel = DiagnosisLabel.NONE,
    valid: bool = True,
) -> ExperienceRecord:
    return ExperienceRecord(
        record_id=f"experience-{expert_id}-{task_id}-{scene_id}-{label.value}-{valid}",
        task_id=task_id,
        scene_id=scene_id,
        expert_id=expert_id,
        score=_candidate(expert_id, joint, valid=valid),
        diagnosis=Diagnosis(score=joint, label=label, reason="completed"),
    )


@dataclass(frozen=True)
class _Memory:
    records: tuple[ExperienceRecord, ...]

    def lookup(
        self, task: TaskSpec, scene: SceneRecord, *, operation_id: str
    ) -> tuple[ExperienceRecord, ...]:
        del task, scene, operation_id
        return self.records

    def append(self, record: ExperienceRecord, *, operation_id: str) -> None:
        raise AssertionError(f"not used: {record} ({operation_id})")


def test_candidate_selection_ignores_invalid_rollouts() -> None:
    valid = _candidate("pi05", 0.4)
    invalid = _candidate("groot", 0.9, valid=False)

    assert select_candidate((invalid, valid)) == valid


def test_candidate_tie_breaks_by_expert_id() -> None:
    selected = select_candidate((_candidate("pi05", 0.5), _candidate("groot", 0.5)))

    assert selected.expert_id == "groot"


def test_all_invalid_candidates_preserve_diagnostics() -> None:
    candidates = (_candidate("pi05", 0.1, valid=False), _candidate("groot", 0.2, valid=False))

    with pytest.raises(NoValidCandidateError) as error:
        select_candidate(candidates)

    assert error.value.candidates == candidates


def test_unfamiliar_context_uses_reproducible_seeded_exploration() -> None:
    experts = (_expert("groot"), _expert("pi05"))
    first = MemoryAwareRouter(
        experts=experts,
        semantic_logits={"groot": 0.0, "pi05": 1.0},
        memory_weight=0.5,
        rng=random.Random(7),
    )
    second = MemoryAwareRouter(
        experts=tuple(reversed(experts)),
        semantic_logits={"pi05": 1.0, "groot": 0.0},
        memory_weight=0.5,
        rng=random.Random(7),
    )
    memory = _Memory(())

    assert first.shortlist(_task(), _scene(), memory) == second.shortlist(_task(), _scene(), memory)


def test_recurrent_context_ranks_successful_history_with_memory_weight() -> None:
    router = MemoryAwareRouter(
        experts=(_expert("groot"), _expert("pi05")),
        semantic_logits={"groot": 2.0, "pi05": 0.0},
        memory_weight=1.0,
        rng=random.Random(1),
    )
    memory = _Memory((_experience("pi05", 0.9), _experience("groot", 0.2)))

    assert [expert.expert_id for expert in router.shortlist(_task(), _scene(), memory)] == [
        "pi05",
        "groot",
    ]


def test_history_filters_failures_and_irrelevant_contexts() -> None:
    records = (
        _experience("pi05", 0.8),
        _experience("pi05", 1.0, label=DiagnosisLabel.PF),
        _experience("pi05", 1.0, task_id="other-task"),
        _experience("pi05", 1.0, scene_id="other-scene"),
        _experience("pi05", 0.9, valid=False),
    )

    assert experience_score("pi05", records, task=_task(), scene=_scene()) == pytest.approx(0.8)


def test_disabled_and_incompatible_experts_are_excluded() -> None:
    router = MemoryAwareRouter(
        experts=(_expert("groot"), _expert("pi05"), _expert("wrong", schema="cartesian")),
        semantic_logits={"groot": 3.0, "pi05": 2.0, "wrong": 1.0},
        memory_weight=0.0,
        action_schema="joint_velocity",
        disabled_expert_ids=frozenset({"groot"}),
        rng=random.Random(3),
    )

    assert [expert.expert_id for expert in router.shortlist(_task(), _scene(), _Memory(()))] == [
        "pi05"
    ]


def test_router_rejects_duplicate_ids_and_nonfinite_logits() -> None:
    with pytest.raises(ValueError, match="unique"):
        MemoryAwareRouter(
            experts=(_expert("pi05"), _expert("pi05")),
            semantic_logits={"pi05": 0.0},
            memory_weight=0.5,
            rng=random.Random(1),
        )
    with pytest.raises(ValueError, match="finite"):
        MemoryAwareRouter(
            experts=(_expert("pi05"),),
            semantic_logits={"pi05": float("nan")},
            memory_weight=0.5,
            rng=random.Random(1),
        )


def test_candidate_selection_rejects_empty_and_duplicate_experts() -> None:
    with pytest.raises(ValueError, match="empty"):
        select_candidate(())
    with pytest.raises(ValueError, match="unique"):
        select_candidate((_candidate("pi05", 0.4), _candidate("pi05", 0.5)))

"""Memory-aware expert routing and deterministic rollout selection."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence

from evoworld.contracts import (
    CandidateScore,
    DiagnosisLabel,
    ExperienceRecord,
    ExpertSpec,
    SceneRecord,
    TaskSpec,
)
from evoworld.grounding import softmax_probabilities
from evoworld.ports import ExperienceMemory


class NoValidCandidateError(RuntimeError):
    """Raised when a round has no rollout that is safe to execute."""

    def __init__(self, candidates: Sequence[CandidateScore]) -> None:
        self.candidates = tuple(candidates)
        details = ", ".join(
            f"{candidate.expert_id}: {candidate.error or 'invalid rollout'}"
            for candidate in self.candidates
        )
        super().__init__(f"no valid rollout candidates: {details}")


def experience_score(
    expert_id: str,
    records: Sequence[ExperienceRecord],
    *,
    task: TaskSpec,
    scene: SceneRecord,
) -> float:
    """Average successful historical joint score for one exact task and scene."""
    if not expert_id.strip():
        raise ValueError("expert_id must not be blank")

    scores = [
        record.score.joint_score
        for record in records
        if _is_successful(record, expert_id=expert_id, task=task, scene=scene)
    ]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def select_candidate(candidates: Sequence[CandidateScore]) -> CandidateScore:
    """Return the highest scoring valid rollout with a stable expert-ID tie break."""
    scores = tuple(candidates)
    if not scores:
        raise ValueError("candidates must not be empty")
    _require_unique_expert_ids(score.expert_id for score in scores)

    valid = tuple(score for score in scores if score.valid)
    if not valid:
        raise NoValidCandidateError(scores)
    return min(valid, key=lambda score: (-score.joint_score, score.expert_id))


class MemoryAwareRouter:
    """Rank experts from semantic logits and successful, relevant experience."""

    def __init__(
        self,
        *,
        experts: Sequence[ExpertSpec],
        semantic_logits: Mapping[str, float],
        memory_weight: float,
        rng: random.Random,
        action_schema: str | None = None,
        disabled_expert_ids: frozenset[str] = frozenset(),
    ) -> None:
        configured_experts = tuple(experts)
        if not configured_experts:
            raise ValueError("experts must not be empty")
        _require_unique_expert_ids(expert.expert_id for expert in configured_experts)
        if not 0.0 <= memory_weight <= 1.0 or not math.isfinite(memory_weight):
            raise ValueError("memory_weight must be finite and between 0 and 1")
        if not isinstance(rng, random.Random):
            raise TypeError("rng must be an injected random.Random instance")
        if action_schema is not None and not action_schema.strip():
            raise ValueError("action_schema must not be blank")
        if any(not expert_id.strip() for expert_id in disabled_expert_ids):
            raise ValueError("disabled expert IDs must not be blank")

        expert_ids = {expert.expert_id for expert in configured_experts}
        if set(semantic_logits) != expert_ids:
            raise ValueError("semantic logits must match configured expert IDs exactly")
        if not all(math.isfinite(logit) for logit in semantic_logits.values()):
            raise ValueError("semantic logits must contain only finite values")
        unknown_disabled = disabled_expert_ids - expert_ids
        if unknown_disabled:
            raise ValueError("disabled expert IDs must be configured experts")

        self._experts = configured_experts
        self._semantic_logits = dict(semantic_logits)
        self._memory_weight = memory_weight
        self._rng = rng
        self._action_schema = action_schema
        self._disabled_expert_ids = disabled_expert_ids

    def shortlist(
        self,
        task: TaskSpec,
        scene: SceneRecord,
        memory: ExperienceMemory,
        *,
        operation_id: str = "direct:route",
    ) -> tuple[ExpertSpec, ...]:
        """Explore semantic preferences or exploit successful local experience."""
        eligible = tuple(
            expert
            for expert in self._experts
            if expert.expert_id not in self._disabled_expert_ids
            and (self._action_schema is None or expert.action_schema == self._action_schema)
        )
        if not eligible:
            raise ValueError("no enabled experts match the required action schema")

        history = tuple(memory.lookup(task, scene, operation_id=f"{operation_id}:lookup"))
        semantic = softmax_probabilities(
            {expert.expert_id: self._semantic_logits[expert.expert_id] for expert in eligible},
            temperature=1.0,
        )
        historical = {
            expert.expert_id: experience_score(expert.expert_id, history, task=task, scene=scene)
            for expert in eligible
        }
        eligible_ids = {expert.expert_id for expert in eligible}
        has_relevant_history = any(
            _is_successful(record, task=task, scene=scene) and record.expert_id in eligible_ids
            for record in history
        )
        if not has_relevant_history:
            return _sample_then_rank(eligible, semantic, self._rng)

        combined = {
            expert.expert_id: (1.0 - self._memory_weight) * semantic[expert.expert_id]
            + self._memory_weight * historical[expert.expert_id]
            for expert in eligible
        }
        return tuple(
            sorted(eligible, key=lambda expert: (-combined[expert.expert_id], expert.expert_id))
        )


def _is_successful(
    record: ExperienceRecord,
    *,
    task: TaskSpec,
    scene: SceneRecord,
    expert_id: str | None = None,
) -> bool:
    return (
        record.task_id == task.task_id
        and record.scene_id == scene.scene_id
        and (expert_id is None or record.expert_id == expert_id)
        and record.score.expert_id == record.expert_id
        and record.score.valid
        and record.diagnosis.label is DiagnosisLabel.NONE
    )


def _sample_then_rank(
    experts: Sequence[ExpertSpec], probabilities: Mapping[str, float], rng: random.Random
) -> tuple[ExpertSpec, ...]:
    ordered = tuple(sorted(experts, key=lambda expert: expert.expert_id))
    threshold = rng.random()
    cumulative_probability = 0.0
    selected = ordered[-1]
    for expert in ordered:
        cumulative_probability += probabilities[expert.expert_id]
        if threshold < cumulative_probability:
            selected = expert
            break
    remaining = tuple(expert for expert in ordered if expert.expert_id != selected.expert_id)
    return (selected, *remaining)


def _require_unique_expert_ids(expert_ids: Iterable[str]) -> None:
    identifiers = tuple(expert_ids)
    if any(not expert_id.strip() for expert_id in identifiers):
        raise ValueError("expert IDs must not be blank")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("expert IDs must be unique")

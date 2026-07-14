"""Deterministic retrieval and seeded sampling for contextual scene anchoring."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from evoworld.contracts import GroundingMode, SceneRecord, TaskSpec


Embedding = Sequence[float]
TextEncoder = Callable[[str], Embedding]
FloatVector: TypeAlias = NDArray[np.float64]


def _validated_temperature(temperature: float) -> float:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and greater than zero")
    return temperature


def _vector(values: Embedding, name: str) -> FloatVector:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    if float(np.max(np.abs(vector))) == 0.0:
        raise ValueError(f"{name} must not be a zero vector")
    return vector


def cosine_similarity(left: Embedding, right: Embedding) -> float:
    """Return the cosine similarity of two finite, nonzero equal-length vectors."""
    left_vector = _vector(left, "left vector")
    right_vector = _vector(right, "right vector")
    if left_vector.size != right_vector.size:
        raise ValueError("vector dimensions must match")
    scaled_left = left_vector / np.max(np.abs(left_vector))
    scaled_right = right_vector / np.max(np.abs(right_vector))
    unit_left = scaled_left / np.linalg.vector_norm(scaled_left)
    unit_right = scaled_right / np.linalg.vector_norm(scaled_right)
    return float(np.clip(np.dot(unit_left, unit_right), -1.0, 1.0))


def softmax_probabilities(
    similarities: Mapping[str, float], *, temperature: float
) -> dict[str, float]:
    """Convert scene similarities into stable, temperature-scaled probabilities."""
    validated_temperature = _validated_temperature(temperature)
    if not similarities:
        raise ValueError("similarities must not be empty")

    scene_ids = tuple(sorted(similarities))
    if any(not scene_id.strip() for scene_id in scene_ids):
        raise ValueError("scene IDs must not be blank")
    logits = np.asarray([similarities[scene_id] for scene_id in scene_ids], dtype=np.float64)
    if not np.all(np.isfinite(logits)):
        raise ValueError("similarities must contain only finite values")
    centered_scores = logits - np.max(logits)
    with np.errstate(over="ignore"):
        scaled_logits = centered_scores / validated_temperature
    unnormalized = np.exp(scaled_logits)
    normalizer = float(np.sum(unnormalized))
    if not math.isfinite(normalizer) or normalizer == 0.0:
        raise ValueError("similarities cannot be normalized")
    return {
        scene_id: float(probability / normalizer)
        for scene_id, probability in zip(scene_ids, unnormalized, strict=True)
    }


def select_scene_id(
    task_embedding: Embedding,
    scene_embeddings: Mapping[str, Embedding],
    *,
    mode: GroundingMode,
    temperature: float,
    rng: random.Random,
) -> str:
    """Select a scene by nearest cosine similarity or seeded softmax sampling."""
    _validated_temperature(temperature)
    if not scene_embeddings:
        raise ValueError("scene embeddings must not be empty")
    if not isinstance(rng, random.Random):
        raise TypeError("rng must be an injected random.Random instance")

    similarities = {
        scene_id: cosine_similarity(task_embedding, embedding)
        for scene_id, embedding in scene_embeddings.items()
    }
    if mode is GroundingMode.NEAREST:
        best_similarity = max(similarities.values())
        return min(
            scene_id
            for scene_id, similarity in similarities.items()
            if similarity == best_similarity
        )
    if mode is not GroundingMode.SOFTMAX:
        raise ValueError(f"unsupported grounding mode: {mode}")

    probabilities = softmax_probabilities(similarities, temperature=temperature)
    threshold = rng.random()
    cumulative_probability = 0.0
    for scene_id, probability in probabilities.items():
        cumulative_probability += probability
        if threshold < cumulative_probability:
            return scene_id
    return next(reversed(probabilities))


class IndexedSceneAnchor:
    """Anchor task instructions using injected embeddings and immutable scene records."""

    def __init__(
        self,
        *,
        text_encoder: TextEncoder,
        scenes: Sequence[SceneRecord],
        embeddings: Mapping[str, Embedding],
        mode: GroundingMode,
        temperature: float,
        rng: random.Random,
    ) -> None:
        _validated_temperature(temperature)
        if not isinstance(rng, random.Random):
            raise TypeError("rng must be an injected random.Random instance")
        indexed_scenes = tuple(scenes)
        if not indexed_scenes:
            raise ValueError("scenes must not be empty")
        scene_ids = tuple(scene.scene_id for scene in indexed_scenes)
        if len(set(scene_ids)) != len(scene_ids):
            raise ValueError("scene IDs must be unique")
        if set(scene_ids) != set(embeddings):
            raise ValueError("scene records and embeddings must match exactly")

        frozen_embeddings = {
            scene_id: tuple(_vector(embedding, f"embedding for {scene_id}").tolist())
            for scene_id, embedding in embeddings.items()
        }
        embedding_dimensions = {len(embedding) for embedding in frozen_embeddings.values()}
        if len(embedding_dimensions) != 1:
            raise ValueError("scene embedding dimensions must match")

        self._text_encoder = text_encoder
        self._scenes = MappingProxyType({scene.scene_id: scene for scene in indexed_scenes})
        self._embeddings = MappingProxyType(frozen_embeddings)
        self._mode = mode
        self._temperature = temperature
        self._rng = rng

    def anchor(self, task: TaskSpec, *, operation_id: str = "direct:anchor") -> SceneRecord:
        """Encode a task instruction and return the selected indexed scene."""
        if not operation_id.strip():
            raise ValueError("operation_id must not be blank")
        scene_id = select_scene_id(
            self._text_encoder(task.instruction),
            self._embeddings,
            mode=self._mode,
            temperature=self._temperature,
            rng=self._rng,
        )
        return self._scenes[scene_id]

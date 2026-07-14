"""Tests for deterministic contextual scene anchoring."""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from evoworld.contracts import GroundingMode, SceneRecord, TaskSpec
from evoworld.grounding import (
    IndexedSceneAnchor,
    cosine_similarity,
    select_scene_id,
    softmax_probabilities,
)


def test_cosine_similarity_and_nearest_scene() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    selected = select_scene_id(
        [1.0, 0.0],
        {"b": [0.0, 1.0], "a": [1.0, 0.0]},
        mode=GroundingMode.NEAREST,
        temperature=0.5,
        rng=random.Random(7),
    )
    assert selected == "a"


def test_softmax_sampling_is_seeded() -> None:
    scene_embeddings = {"a": [1.0], "b": [0.1]}
    first = select_scene_id(
        [1.0],
        scene_embeddings,
        mode=GroundingMode.SOFTMAX,
        temperature=0.8,
        rng=random.Random(11),
    )
    second = select_scene_id(
        [1.0],
        scene_embeddings,
        mode=GroundingMode.SOFTMAX,
        temperature=0.8,
        rng=random.Random(11),
    )
    assert first == second


def test_softmax_probabilities_are_normalized_and_stable() -> None:
    probabilities = softmax_probabilities({"a": 1_000.0, "b": 999.0}, temperature=0.5)

    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities["a"] > probabilities["b"] > 0.0


def test_softmax_handles_finite_subnormal_temperature() -> None:
    probabilities = softmax_probabilities({"a": 1.0, "b": 0.0}, temperature=1e-320)

    assert all(math.isfinite(probability) for probability in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities == {"a": pytest.approx(1.0), "b": pytest.approx(0.0)}


def test_nearest_scene_breaks_ties_by_scene_id() -> None:
    selected = select_scene_id(
        [1.0, 0.0],
        {"z": [1.0, 0.0], "a": [1.0, 0.0]},
        mode=GroundingMode.NEAREST,
        temperature=1.0,
        rng=random.Random(1),
    )

    assert selected == "a"


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        ([], [], "empty"),
        ([0.0], [1.0], "zero"),
        ([float("nan")], [1.0], "finite"),
        ([1.0], [1.0, 0.0], "dimensions"),
    ],
)
def test_cosine_similarity_rejects_invalid_vectors(
    left: list[float], right: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        cosine_similarity(left, right)


def test_cosine_similarity_scales_finite_large_vectors() -> None:
    assert cosine_similarity([1e308, 1e308], [1e308, 1e308]) == pytest.approx(1.0)


def test_cosine_similarity_scales_finite_subnormal_vectors() -> None:
    assert cosine_similarity([1e-320], [1e-320]) == pytest.approx(1.0)


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_softmax_rejects_invalid_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        softmax_probabilities({"a": 1.0}, temperature=temperature)


def test_indexed_scene_anchor_returns_injected_scene() -> None:
    scene = SceneRecord(scene_id="scene-a", observation_paths=(Path("frame.jpg"),))
    anchor = IndexedSceneAnchor(
        text_encoder=lambda instruction: [1.0, 0.0] if instruction else [0.0, 1.0],
        scenes=(scene,),
        embeddings={"scene-a": [1.0, 0.0]},
        mode=GroundingMode.NEAREST,
        temperature=1.0,
        rng=random.Random(0),
    )

    assert anchor.anchor(TaskSpec(task_id="task-1", instruction="pick up")) is scene


def test_indexed_scene_anchor_rejects_inconsistent_index() -> None:
    with pytest.raises(ValueError, match="match"):
        IndexedSceneAnchor(
            text_encoder=lambda instruction: [1.0],
            scenes=(SceneRecord(scene_id="scene-a"),),
            embeddings={"scene-b": [1.0]},
            mode=GroundingMode.NEAREST,
            temperature=1.0,
            rng=random.Random(0),
        )


def test_indexed_scene_anchor_rejects_duplicate_scene_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        IndexedSceneAnchor(
            text_encoder=lambda instruction: [1.0],
            scenes=(SceneRecord(scene_id="scene-a"), SceneRecord(scene_id="scene-a")),
            embeddings={"scene-a": [1.0]},
            mode=GroundingMode.NEAREST,
            temperature=1.0,
            rng=random.Random(0),
        )

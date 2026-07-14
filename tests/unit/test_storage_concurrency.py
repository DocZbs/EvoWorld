"""Concurrency, conflict, and schema tests for durable run storage."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from evoworld.codec import (
    CodecError,
    StateDecodeError,
    UnsupportedStateSchemaError,
    input_digest,
)
from evoworld.contracts import (
    ActionChunk,
    CandidateScore,
    Diagnosis,
    DiagnosisLabel,
    ExperienceRecord,
    OperationKind,
    OperationReceipt,
    RoundStage,
    RoundState,
    SceneRecord,
    TaskSpec,
    Trajectory,
)
from evoworld.storage import (
    ArtifactConflictError,
    JsonRunStore,
    JsonlExperienceMemory,
    StateConflictError,
    atomic_write_json,
)


def test_artifact_write_is_idempotent_but_rejects_different_bytes(tmp_path: Path) -> None:
    store = JsonRunStore.create(tmp_path, "run", {}, "digest")
    store.claim_round(_claim_state())

    first = store.write_artifact("round-1", "grounding", {"value": 1})
    second = store.write_artifact("round-1", "grounding", {"value": 1})

    assert first == second
    with pytest.raises(ArtifactConflictError):
        store.write_artifact("round-1", "grounding", {"value": 2})


def test_round_claim_is_exclusive_across_processes(tmp_path: Path) -> None:
    store = JsonRunStore.create(tmp_path, "run", {}, "digest")
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    processes = [
        context.Process(target=_claim_round, args=(store.run_dir, queue)) for _ in range(2)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=5)

    assert sorted(queue.get(timeout=1) for _ in processes) == ["claimed", "conflict"]
    assert all(process.exitcode == 0 for process in processes)


def test_orphan_round_claim_recovers_complete_initial_state(tmp_path: Path) -> None:
    store = JsonRunStore.create(tmp_path, "run", {}, "digest")
    claimed = _claim_state()
    store.claim_round(claimed)

    assert not (store.run_dir / "state.json").exists()
    assert store.read_state() == claimed
    assert store.load_state(claimed.round_id) == claimed


def test_round_state_rejects_receipt_without_associated_output() -> None:
    with pytest.raises(ValueError, match="associated durable output"):
        RoundState(
            round_id="round-1",
            stage=RoundStage.STARTED,
            started_at="2026-07-17T00:00:00+00:00",
            seed=7,
            receipts=(OperationReceipt("round-1:task", OperationKind.TASK, input_digest({}), {}),),
        )


def test_state_cas_allows_only_one_concurrent_revision(tmp_path: Path) -> None:
    store = JsonRunStore.create(tmp_path, "run", {}, "digest")
    initial = _state()
    store.save_state(initial)
    proposals = (
        replace(initial, revision=1, started_at="2026-07-17T00:00:01+00:00"),
        replace(initial, revision=1, started_at="2026-07-17T00:00:02+00:00"),
    )

    def save(proposal: RoundState) -> str:
        try:
            store.save_state(proposal)
        except StateConflictError:
            return "conflict"
        return "saved"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(save, proposals))

    assert sorted(outcomes) == ["conflict", "saved"]
    assert store.read_state().revision == 1  # type: ignore[union-attr]


def test_duplicate_experience_append_is_one_line_across_processes(tmp_path: Path) -> None:
    path = tmp_path / "memory.jsonl"
    context = multiprocessing.get_context("fork")
    processes = [
        context.Process(target=_append_experience, args=(path, f"operation-{index}"))
        for index in range(4)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


@pytest.mark.parametrize(
    "payload",
    (
        '{"round_id":"legacy"}',
        '{"schema_version":999}',
    ),
)
def test_unsupported_state_schema_has_explicit_nonresumable_error(
    tmp_path: Path, payload: str
) -> None:
    store = JsonRunStore.create(tmp_path, "run", {}, "digest")
    (store.run_dir / "state.json").write_text(payload, encoding="utf-8")

    with pytest.raises(UnsupportedStateSchemaError, match="start a new run"):
        store.read_state()


@pytest.mark.parametrize("payload", ("{", "[]", '{"schema_version":2,"stage":4}'))
def test_corrupt_or_wrongly_typed_state_is_never_silently_resumed(
    tmp_path: Path, payload: str
) -> None:
    store = JsonRunStore.create(tmp_path, "run", {}, "digest")
    (store.run_dir / "state.json").write_text(payload, encoding="utf-8")

    with pytest.raises(StateDecodeError):
        store.read_state()


def test_nonfinite_domain_and_json_values_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite"):
        ActionChunk("pi05", ((float("nan"),),), (0.0,), "joint_velocity")
    with pytest.raises(CodecError, match="finite"):
        atomic_write_json(tmp_path / "bad.json", {"score": float("inf")})


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: payload.replace('"seed": 7', '"seed": NaN'),
        lambda payload: payload.replace('"round_id":', '"unknown":true,"round_id":'),
    ),
)
def test_state_decoder_rejects_nonstandard_numbers_and_unknown_fields(
    tmp_path: Path, mutation
) -> None:
    store = JsonRunStore.create(tmp_path, "run", {}, "digest")
    store.claim_round(_claim_state())
    claim = store.run_dir / "rounds" / "round-1" / ".claim"
    payload = claim.read_text(encoding="utf-8").strip()
    claim.write_text(mutation(payload), encoding="utf-8")

    with pytest.raises(StateDecodeError):
        store.read_state()


def test_verified_write_rejects_symlink_escape_before_creating_outside_data(
    tmp_path: Path,
) -> None:
    store = JsonRunStore.create(tmp_path, "run", {}, "digest")
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (store.run_dir / "verified").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")
    actions = ActionChunk("pi05", ((0.1,),), (0.0,), "joint_velocity")
    trajectory = Trajectory("trajectory-1", "rollout-1", (), actions)

    with pytest.raises(ValueError, match="verified"):
        store.write_verified(trajectory, {"round_id": "round-1"})

    assert list(outside.iterdir()) == []


def test_task_and_memory_journals_reject_symlink_escape(tmp_path: Path) -> None:
    store = JsonRunStore.create(tmp_path, "run", {}, "digest")
    outside_task = tmp_path / "outside-tasks.jsonl"
    outside_memory = tmp_path / "outside-memory.jsonl"
    outside_task.write_text("", encoding="utf-8")
    outside_memory.write_text("", encoding="utf-8")
    try:
        (store.run_dir / "tasks.jsonl").symlink_to(outside_task)
        (store.run_dir / "memory.jsonl").symlink_to(outside_memory)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")

    with pytest.raises(ValueError, match="task journal"):
        store.append_task("round-1", "round-1:task", TaskSpec("task-1", "place cup"))
    memory = JsonlExperienceMemory(store.run_dir / "memory.jsonl", containment_root=store.run_dir)
    with pytest.raises(ValueError, match="experience memory"):
        memory.append(_experience(), operation_id="round-1:memory_append")

    assert outside_task.read_bytes() == b""
    assert outside_memory.read_bytes() == b""


def _claim_round(run_dir: Path, queue: object) -> None:
    store = JsonRunStore.open(run_dir, "digest")
    try:
        store.claim_round(_claim_state())
    except FileExistsError:
        queue.put("conflict")  # type: ignore[attr-defined]
    else:
        queue.put("claimed")  # type: ignore[attr-defined]


def _append_experience(path: Path, operation_id: str) -> None:
    JsonlExperienceMemory(path).append(_experience(), operation_id=operation_id)


def _experience() -> ExperienceRecord:
    return ExperienceRecord(
        record_id="experience-1",
        task_id="task-1",
        scene_id="scene-1",
        expert_id="pi05",
        score=CandidateScore("pi05", "rollout-1", 0.8, 0.8, 0.8),
        diagnosis=Diagnosis(0.9, DiagnosisLabel.NONE, "verified"),
    )


def _state() -> RoundState:
    return RoundState(
        round_id="round-1",
        stage=RoundStage.GROUNDED,
        started_at="2026-07-17T00:00:00+00:00",
        seed=7,
        task=TaskSpec("task-1", "place cup"),
        scene=SceneRecord("scene-1"),
    )


def _claim_state() -> RoundState:
    return RoundState(
        round_id="round-1",
        stage=RoundStage.STARTED,
        started_at="2026-07-17T00:00:00+00:00",
        seed=7,
    )

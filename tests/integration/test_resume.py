"""Fresh-process recovery coverage for interrupted EvoWorld rounds."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest

import evoworld.storage as storage_module
from evoworld.adapters.deterministic import (
    AdapterSet,
    DeterministicScenario,
    build_deterministic_adapters,
)
from evoworld.codec import (
    StateDecodeError,
    UnsupportedStateSchemaError,
    round_state_to_json,
)
from evoworld.config import EvoWorldConfig
from evoworld.contracts import (
    DiagnosisLabel,
    ExecutionMode,
    ExpertSpec,
    GroundingMode,
    JsonObject,
    OperationKind,
    RoundStage,
    RoundState,
    TaskSpec,
    Trajectory,
)
from evoworld.pipeline import EvoWorldPipeline
from evoworld.ports import OperationInDoubtError
from evoworld.storage import JsonRunStore, StateConflictError, atomic_write_json


class ScriptedInterruption(RuntimeError):
    """Test-only process loss at a selected durability boundary."""


class _DelegatingStore:
    def __init__(self, delegate: JsonRunStore) -> None:
        self._delegate = delegate

    @property
    def run_dir(self) -> Path:
        return self._delegate.run_dir

    def save_state(self, state: RoundState) -> None:
        self._delegate.save_state(state)

    def read_state(self) -> RoundState | None:
        return self._delegate.read_state()

    def load_state(self, round_id: str) -> RoundState | None:
        return self._delegate.load_state(round_id)

    def claim_round(self, state: RoundState) -> None:
        self._delegate.claim_round(state)

    def write_artifact(self, round_id: str, name: str, metadata: JsonObject) -> Path:
        return self._delegate.write_artifact(round_id, name, metadata)

    def append_task(self, round_id: str, operation_id: str, task: TaskSpec) -> None:
        self._delegate.append_task(round_id, operation_id, task)

    def write_verified(self, trajectory: Trajectory, provenance: JsonObject) -> Path:
        return self._delegate.write_verified(trajectory, provenance)


class _InterruptAfterStageStore(_DelegatingStore):
    def __init__(self, delegate: JsonRunStore, stage: RoundStage) -> None:
        super().__init__(delegate)
        self._stage = stage
        self._interrupted = False

    def save_state(self, state: RoundState) -> None:
        super().save_state(state)
        if state.stage is self._stage and not self._interrupted:
            self._interrupted = True
            raise ScriptedInterruption(state.stage.value)


class _InterruptBeforeCompleteStore(_DelegatingStore):
    def save_state(self, state: RoundState) -> None:
        if state.stage is RoundStage.COMPLETE:
            raise ScriptedInterruption("before complete")
        super().save_state(state)


class _InterruptAfterIntentStore(_DelegatingStore):
    def __init__(self, delegate: JsonRunStore, kind: OperationKind) -> None:
        super().__init__(delegate)
        self._kind = kind
        self._interrupted = False

    def save_state(self, state: RoundState) -> None:
        super().save_state(state)
        pending = state.pending_operation
        if pending is not None and pending.kind is self._kind and not self._interrupted:
            self._interrupted = True
            raise ScriptedInterruption("after intent")


class _KeyboardInterruptBeforeReceiptStore(_DelegatingStore):
    def __init__(self, delegate: JsonRunStore, kind: OperationKind) -> None:
        super().__init__(delegate)
        self._kind = kind
        self.operation_id: str | None = None

    def save_state(self, state: RoundState) -> None:
        receipt = next(
            (receipt for receipt in state.receipts if receipt.kind is self._kind),
            None,
        )
        if receipt is not None and self.operation_id is None:
            self.operation_id = receipt.operation_id
            raise KeyboardInterrupt
        super().save_state(state)


class _InterruptAfterReceiptStore(_DelegatingStore):
    def __init__(self, delegate: JsonRunStore, kind: OperationKind) -> None:
        super().__init__(delegate)
        self._kind = kind
        self._interrupted = False

    def save_state(self, state: RoundState) -> None:
        super().save_state(state)
        if not self._interrupted and any(receipt.kind is self._kind for receipt in state.receipts):
            self._interrupted = True
            raise ScriptedInterruption("after receipt")


class _RecordingStore(_DelegatingStore):
    def __init__(self, delegate: JsonRunStore) -> None:
        super().__init__(delegate)
        self.revisions: list[int] = []

    def save_state(self, state: RoundState) -> None:
        super().save_state(state)
        self.revisions.append(state.revision)


@pytest.mark.parametrize(
    "stage",
    (
        RoundStage.GROUNDED,
        RoundStage.CANDIDATES_SCORED,
        RoundStage.EXECUTED,
        RoundStage.COMPLETE,
    ),
)
def test_fresh_reconstruction_restores_each_completed_stage(
    tmp_path: Path, stage: RoundStage
) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / stage.value
    first = build_deterministic_adapters(config, run_dir)
    interrupted_store = _InterruptAfterStageStore(cast(JsonRunStore, first.store), stage)

    with pytest.raises(ScriptedInterruption, match=stage.value):
        EvoWorldPipeline(config, replace(first, store=interrupted_store)).run(1)
    persisted = interrupted_store.read_state()
    assert persisted is not None and persisted.stage is stage
    durable_before = _immutable_round_bytes(run_dir, persisted.round_id)
    verified_before = _tree_bytes(run_dir / "verified")

    restarted = build_deterministic_adapters(config, run_dir)
    spied, spies = _spied_adapters(restarted)
    recording = _RecordingStore(cast(JsonRunStore, spied.store))
    result = EvoWorldPipeline(config, replace(spied, store=recording)).resume()

    completed = recording.read_state()
    assert completed is not None and completed.stage is RoundStage.COMPLETE
    if stage is RoundStage.COMPLETE:
        assert result is None
    else:
        assert result is not None and result.round_id == persisted.round_id
    assert recording.revisions == list(range(persisted.revision + 1, completed.revision + 1))
    for relative, payload in durable_before.items():
        assert (run_dir / "rounds" / persisted.round_id / relative).read_bytes() == payload
    if verified_before:
        assert _tree_bytes(run_dir / "verified") == verified_before
    _assert_completed_adapters_were_not_called(stage, spies)
    _assert_unique_durable_ids(run_dir)


@pytest.mark.parametrize("target", tuple(OperationKind))
def test_each_operation_intent_survives_before_adapter_call(
    tmp_path: Path, target: OperationKind
) -> None:
    config, run_dir, first, execution_mode = _operation_case(tmp_path, target, "intent")
    store = _InterruptAfterIntentStore(cast(JsonRunStore, first.store), target)

    with pytest.raises(ScriptedInterruption, match="after intent"):
        EvoWorldPipeline(
            config,
            replace(first, store=store),
            execution_mode=execution_mode,
        ).run(1)
    interrupted = store.read_state()
    assert interrupted is not None and interrupted.pending_operation is not None
    assert interrupted.pending_operation.kind is target
    operation_id = interrupted.pending_operation.operation_id

    restarted = build_deterministic_adapters(config, run_dir, _scenario(target))
    resumed = EvoWorldPipeline(config, restarted, execution_mode=execution_mode).resume()

    assert resumed is not None
    if target is not OperationKind.MEMORY_APPEND:
        assert restarted.receipts.effect_count(target.value, operation_id) == 1
    completed = cast(JsonRunStore, restarted.store).read_state_required()
    assert sum(receipt.operation_id == operation_id for receipt in completed.receipts) == 1
    _assert_journal_content_unique(run_dir / "tasks.jsonl")
    _assert_journal_content_unique(run_dir / "memory.jsonl")


@pytest.mark.parametrize("target", tuple(OperationKind))
def test_keyboard_interrupt_after_each_adapter_return_reuses_durable_effect(
    tmp_path: Path, target: OperationKind
) -> None:
    config, run_dir, first, execution_mode = _operation_case(tmp_path, target, "keyboard-interrupt")
    store = _KeyboardInterruptBeforeReceiptStore(cast(JsonRunStore, first.store), target)

    with pytest.raises(KeyboardInterrupt):
        EvoWorldPipeline(
            config,
            replace(first, store=store),
            execution_mode=execution_mode,
        ).run(1)
    pending = store.read_state()
    assert pending is not None and pending.pending_operation is not None
    assert pending.pending_operation.kind is target
    operation_id = pending.pending_operation.operation_id
    assert store.operation_id == operation_id

    restarted = build_deterministic_adapters(config, run_dir, _scenario(target))
    resumed = EvoWorldPipeline(config, restarted, execution_mode=execution_mode).resume()

    assert resumed is not None
    if target is not OperationKind.MEMORY_APPEND:
        assert restarted.receipts.effect_count(target.value, operation_id) == 0
    completed = cast(JsonRunStore, restarted.store).read_state_required()
    assert sum(receipt.operation_id == operation_id for receipt in completed.receipts) == 1
    _assert_journal_content_unique(run_dir / "tasks.jsonl")
    _assert_journal_content_unique(run_dir / "memory.jsonl")


@pytest.mark.parametrize("target", tuple(OperationKind))
def test_each_receipt_before_next_cas_restores_output_without_reinvocation(
    tmp_path: Path, target: OperationKind
) -> None:
    config, run_dir, first, execution_mode = _operation_case(tmp_path, target, "after-receipt")
    store = _InterruptAfterReceiptStore(cast(JsonRunStore, first.store), target)

    with pytest.raises(ScriptedInterruption, match="after receipt"):
        EvoWorldPipeline(
            config,
            replace(first, store=store),
            execution_mode=execution_mode,
        ).run(1)
    persisted = store.read_state()
    assert persisted is not None
    receipt = next(receipt for receipt in persisted.receipts if receipt.kind is target)

    restarted, spies = _spied_adapters(
        build_deterministic_adapters(config, run_dir, _scenario(target))
    )
    resumed = EvoWorldPipeline(config, restarted, execution_mode=execution_mode).resume()

    assert resumed is not None
    _assert_operation_id_not_called(target, receipt.operation_id, spies)
    _assert_journal_content_unique(run_dir / "tasks.jsonl")
    _assert_journal_content_unique(run_dir / "memory.jsonl")


def test_final_artifacts_survive_crash_before_complete_cas(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / "final-artifacts"
    first = build_deterministic_adapters(config, run_dir)
    store = _InterruptBeforeCompleteStore(cast(JsonRunStore, first.store))

    with pytest.raises(ScriptedInterruption, match="before complete"):
        EvoWorldPipeline(config, replace(first, store=store)).run(1)
    persisted = store.read_state()
    assert persisted is not None and persisted.stage is RoundStage.EXECUTED
    durable_before = _immutable_round_bytes(run_dir, persisted.round_id)
    verified_before = _tree_bytes(run_dir / "verified")

    restarted, spies = _spied_adapters(build_deterministic_adapters(config, run_dir))
    resumed = EvoWorldPipeline(config, restarted).resume()

    assert resumed is not None and resumed.verified
    for relative, payload in durable_before.items():
        assert (run_dir / "rounds" / persisted.round_id / relative).read_bytes() == payload
    assert _tree_bytes(run_dir / "verified") == verified_before
    spies["evaluator"].evaluate.assert_not_called()
    spies["trainer"].plan.assert_not_called()
    spies["memory"].append.assert_not_called()
    _assert_unique_durable_ids(run_dir)


def test_complete_and_absent_runs_resume_without_invoking_adapters(tmp_path: Path) -> None:
    config = _config(tmp_path)
    absent, absent_spies = _spied_adapters(
        build_deterministic_adapters(config, tmp_path / "absent")
    )
    assert EvoWorldPipeline(config, absent).resume() is None
    assert all(not spy.mock_calls for spy in absent_spies.values())

    run_dir = tmp_path / "complete"
    EvoWorldPipeline(config, build_deterministic_adapters(config, run_dir)).run(1)
    complete, complete_spies = _spied_adapters(build_deterministic_adapters(config, run_dir))
    assert EvoWorldPipeline(config, complete).resume() is None
    assert all(not spy.mock_calls for spy in complete_spies.values())


def test_newest_incomplete_claim_wins_over_completed_older_round(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / "multiple"
    first = build_deterministic_adapters(config, run_dir)
    EvoWorldPipeline(config, first).run_round("round-000001")
    newer = RoundState(
        round_id="round-000002",
        stage=RoundStage.STARTED,
        started_at="9999-12-31T23:59:59+00:00",
        seed=config.seed,
        reliability_threshold=config.reliability_threshold,
    )
    first.store.claim_round(newer)

    restarted = build_deterministic_adapters(config, run_dir)
    resumed = EvoWorldPipeline(config, restarted).resume()

    assert resumed is not None and resumed.round_id == newer.round_id
    _assert_unique_durable_ids(run_dir)


def test_latest_completed_claim_does_not_resurrect_older_orphan(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / "completed-after-orphan"
    adapters = build_deterministic_adapters(config, run_dir)
    store = cast(JsonRunStore, adapters.store)
    store.claim_round(
        RoundState(
            round_id="round-old-orphan",
            stage=RoundStage.STARTED,
            started_at="2026-07-17T00:00:00+00:00",
            seed=config.seed,
            reliability_threshold=config.reliability_threshold,
        )
    )

    EvoWorldPipeline(config, adapters).run_round("round-new-complete")
    completed = store.load_state("round-new-complete")

    assert completed is not None and completed.stage is RoundStage.COMPLETE
    assert store.read_state() == completed
    assert EvoWorldPipeline(config, adapters).resume() is None


def test_changed_config_digest_is_rejected_before_resume(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / "config-mismatch"
    EvoWorldPipeline(config, build_deterministic_adapters(config, run_dir)).run(1)

    with pytest.raises(ValueError, match="config digest"):
        build_deterministic_adapters(replace(config, rollout_horizon=4), run_dir)


@pytest.mark.parametrize(
    ("payload", "error_type"),
    (
        ("{", StateDecodeError),
        ('{"schema_version":999}', UnsupportedStateSchemaError),
    ),
)
def test_corrupt_and_unknown_state_schemas_raise_typed_errors(
    tmp_path: Path, payload: str, error_type: type[Exception]
) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / error_type.__name__
    EvoWorldPipeline(config, build_deterministic_adapters(config, run_dir)).run(1)
    (run_dir / "state.json").write_text(payload, encoding="utf-8")

    restarted = build_deterministic_adapters(config, run_dir)
    with pytest.raises(error_type):
        EvoWorldPipeline(config, restarted).resume()


def test_physical_in_doubt_round_never_auto_replays(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / "in-doubt"
    interrupted = build_deterministic_adapters(
        config,
        run_dir,
        DeterministicScenario(execution_interrupt="after_effect"),
    )
    with pytest.raises(RuntimeError, match="execute failed"):
        EvoWorldPipeline(config, interrupted).run(1)

    restarted = build_deterministic_adapters(config, run_dir)
    with pytest.raises(OperationInDoubtError):
        EvoWorldPipeline(config, restarted).resume()
    state = restarted.store.read_state()  # type: ignore[attr-defined]
    assert state is not None and state.stage is RoundStage.IN_DOUBT

    blocked, spies = _spied_adapters(build_deterministic_adapters(config, run_dir))
    with pytest.raises(OperationInDoubtError):
        EvoWorldPipeline(config, blocked).resume()
    spies["executor"].execute.assert_not_called()


def test_resume_store_selects_newest_incomplete_orphan_claim(tmp_path: Path) -> None:
    store = JsonRunStore.create(tmp_path, "run", {}, "digest")
    older = RoundState(
        round_id="round-000001",
        stage=RoundStage.STARTED,
        started_at="2026-07-17T00:00:00+00:00",
        seed=7,
    )
    newer = RoundState(
        round_id="round-000002",
        stage=RoundStage.STARTED,
        started_at="2026-07-17T00:00:01+00:00",
        seed=7,
    )
    store.claim_round(older)
    store.save_state(older)
    store.claim_round(newer)

    recovered = JsonRunStore.open(store.run_dir, "digest").read_state()

    assert recovered == newer


def test_same_round_higher_revision_wins_even_when_both_states_are_complete(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / "complete-revision"
    adapters = build_deterministic_adapters(config, run_dir)
    EvoWorldPipeline(config, adapters).run(1)
    latest = cast(JsonRunStore, adapters.store).read_state_required()
    assert latest.stage is RoundStage.COMPLETE and latest.revision > 0
    atomic_write_json(run_dir / "state.json", round_state_to_json(replace(latest, revision=0)))

    assert cast(JsonRunStore, adapters.store).read_state() == latest


def test_same_round_equal_revision_with_different_bytes_is_a_conflict(
    tmp_path: Path,
) -> None:
    store = JsonRunStore.create(tmp_path, "run", {}, "digest")
    initial = RoundState(
        round_id="round-000001",
        stage=RoundStage.STARTED,
        started_at="2026-07-17T00:00:00+00:00",
        seed=7,
    )
    store.claim_round(initial)
    store.save_state(initial)
    divergent = replace(initial, started_at="2026-07-17T00:00:01+00:00")
    atomic_write_json(store.run_dir / "state.json", round_state_to_json(divergent))

    with pytest.raises(StateConflictError, match="equal revision"):
        store.read_state()


def test_claim_and_round_snapshot_equal_revision_divergence_is_a_conflict(
    tmp_path: Path,
) -> None:
    store = JsonRunStore.create(tmp_path, "run", {}, "digest")
    initial = RoundState("round-1", RoundStage.STARTED, "2026-07-17T00:00:00+00:00", 7)
    store.claim_round(initial)
    divergent = replace(initial, started_at="2026-07-17T00:00:01+00:00")
    atomic_write_json(
        store.run_dir / "rounds" / initial.round_id / "state.json",
        round_state_to_json(divergent),
    )

    with pytest.raises(StateConflictError, match="equal revision"):
        store.read_state()


@pytest.mark.parametrize(
    ("first_time", "second_time"),
    (
        ("2026-07-17T00:00:00+14:00", "2025-01-01T00:00:00+00:00"),
        ("2026-07-17T00:00:00+00:00", "2026-07-17T00:00:00+00:00"),
    ),
)
def test_later_claim_sequence_wins_regardless_of_time_or_round_id(
    tmp_path: Path, first_time: str, second_time: str
) -> None:
    store = JsonRunStore.create(tmp_path, "run", {}, "digest")
    first = RoundState("round-z", RoundStage.STARTED, first_time, 7)
    second = RoundState("round-a", RoundStage.STARTED, second_time, 7)
    store.claim_round(first)
    store.claim_round(second)

    assert store.read_state() == second


@pytest.mark.parametrize("started_at", ("2026-07-17T00:00:00", "not-a-timestamp"))
def test_round_started_at_requires_timezone_aware_iso8601(started_at: str) -> None:
    with pytest.raises(ValueError, match="timezone-aware ISO"):
        RoundState("round-1", RoundStage.STARTED, started_at, 7)


@pytest.mark.parametrize(
    ("payload", "error_type"),
    (
        ("{", StateDecodeError),
        (
            '{"claim_schema_version":2,"claim_sequence":1,"state":{},"extra":true}',
            StateDecodeError,
        ),
        ('{"claim_schema_version":2,"state":{}}', StateDecodeError),
        (
            '{"claim_schema_version":2,"claim_sequence":"1","state":{}}',
            StateDecodeError,
        ),
        ('{"claim_schema_version":999,"claim_sequence":1,"state":{}}', UnsupportedStateSchemaError),
        ('{"claim_sequence":1,"state":{}}', UnsupportedStateSchemaError),
    ),
)
def test_claim_wrapper_failures_are_typed(
    tmp_path: Path, payload: str, error_type: type[Exception]
) -> None:
    store = JsonRunStore.create(tmp_path, "run", {}, "digest")
    claim = store.run_dir / "rounds" / "round-1" / ".claim"
    claim.parent.mkdir(parents=True)
    claim.write_text(payload, encoding="utf-8")

    with pytest.raises(error_type):
        store.read_state()


@pytest.mark.parametrize("target", ("claim", "state"))
def test_round_directory_rejects_embedded_round_id_mismatch(tmp_path: Path, target: str) -> None:
    store = JsonRunStore.create(tmp_path, "run", {}, "digest")
    claimed = RoundState("round-1", RoundStage.STARTED, "2026-07-17T00:00:00+00:00", 7)
    store.claim_round(claimed)
    if target == "state":
        store.save_state(claimed)
        path = store.run_dir / "rounds" / "round-1" / "state.json"
        payload = round_state_to_json(claimed)
    else:
        path = store.run_dir / "rounds" / "round-1" / ".claim"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict) and isinstance(payload["state"], dict)
        payload = cast(dict[str, object], payload)
        state_payload = cast(dict[str, object], payload["state"])
        state_payload["round_id"] = "round-elsewhere"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(StateDecodeError, match="directory"):
            store.read_state()
        return
    payload["round_id"] = "round-elsewhere"
    atomic_write_json(path, payload)

    with pytest.raises(StateDecodeError, match="directory"):
        store.read_state()


def test_duplicate_claim_sequences_raise_typed_conflict(tmp_path: Path) -> None:
    store = JsonRunStore.create(tmp_path, "run", {}, "digest")
    for round_id in ("round-1", "round-2"):
        store.claim_round(RoundState(round_id, RoundStage.STARTED, "2026-07-17T00:00:00+00:00", 7))
    first_path = store.run_dir / "rounds" / "round-1" / ".claim"
    second_path = store.run_dir / "rounds" / "round-2" / ".claim"
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    second["claim_sequence"] = first["claim_sequence"]
    second_path.write_text(json.dumps(second), encoding="utf-8")

    with pytest.raises(StateConflictError, match="duplicate claim sequence"):
        store.read_state()


def test_claim_crash_after_counter_allocation_leaves_safe_sequence_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonRunStore.create(tmp_path, "run", {}, "digest")
    atomic_create = storage_module._atomic_create_exclusive

    def interrupt_claim(_path: Path, _payload: bytes) -> None:
        raise OSError("crash after sequence allocation")

    monkeypatch.setattr(storage_module, "_atomic_create_exclusive", interrupt_claim)
    with pytest.raises(OSError, match="sequence allocation"):
        store.claim_round(RoundState("round-1", RoundStage.STARTED, "2026-07-17T00:00:00+00:00", 7))
    monkeypatch.setattr(storage_module, "_atomic_create_exclusive", atomic_create)
    store.claim_round(RoundState("round-2", RoundStage.STARTED, "2026-07-17T00:00:00+00:00", 7))

    assert not (store.run_dir / "rounds" / "round-1" / ".claim").exists()
    claim = json.loads(
        (store.run_dir / "rounds" / "round-2" / ".claim").read_text(encoding="utf-8")
    )
    assert claim["claim_sequence"] == 2
    assert store.read_state().round_id == "round-2"  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "failure",
    (KeyboardInterrupt("mid-write"), OSError("mid-write")),
    ids=("keyboard-interrupt", "write-error"),
)
def test_claim_mid_write_failure_never_publishes_truncated_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    store = JsonRunStore.create(tmp_path, "run", {}, "digest")
    claimed = RoundState("round-1", RoundStage.STARTED, "2026-07-17T00:00:00+00:00", 7)

    def interrupt_write(descriptor: int, payload: bytes) -> None:
        os.write(descriptor, payload[:8])
        raise failure

    monkeypatch.setattr(
        storage_module,
        "_write_descriptor_and_sync",
        interrupt_write,
        raising=False,
    )
    with pytest.raises(type(failure), match="mid-write"):
        store.claim_round(claimed)

    claim_path = store.run_dir / "rounds" / claimed.round_id / ".claim"
    assert not claim_path.exists()
    assert list(claim_path.parent.glob(".claim.*.tmp")) == []

    monkeypatch.undo()
    store.claim_round(claimed)
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    assert claim["claim_sequence"] == 2
    assert store.read_state() == claimed


def test_claim_publish_never_closes_a_reused_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonRunStore.create(tmp_path, "run", {}, "digest")
    claimed = RoundState("round-1", RoundStage.STARTED, "2026-07-17T00:00:00+00:00", 7)
    sentinel_path = tmp_path / "sentinel"
    claim_descriptor: int | None = None
    sentinel_descriptor: int | None = None
    write_and_sync = storage_module._write_descriptor_and_sync
    link = os.link

    def capture_descriptor(descriptor: int, payload: bytes) -> None:
        nonlocal claim_descriptor
        claim_descriptor = descriptor
        write_and_sync(descriptor, payload)

    def reuse_descriptor(source: Path, target: Path) -> None:
        nonlocal sentinel_descriptor
        sentinel_descriptor = os.open(sentinel_path, os.O_RDWR | os.O_CREAT, 0o600)
        link(source, target)

    monkeypatch.setattr(storage_module, "_write_descriptor_and_sync", capture_descriptor)
    monkeypatch.setattr(storage_module.os, "link", reuse_descriptor)
    try:
        store.claim_round(claimed)

        assert sentinel_descriptor is not None
        assert sentinel_descriptor != claim_descriptor
        os.fstat(sentinel_descriptor)
        assert os.write(sentinel_descriptor, b"sentinel") == len(b"sentinel")
    finally:
        if sentinel_descriptor is not None:
            try:
                os.close(sentinel_descriptor)
            except OSError:
                pass


def test_claim_writer_retries_eintr_and_partial_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonRunStore.create(tmp_path, "run", {}, "digest")
    claimed = RoundState("round-1", RoundStage.STARTED, "2026-07-17T00:00:00+00:00", 7)
    write = os.write
    calls = 0

    def interrupted_partial_write(descriptor: int, payload: bytes) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError
        return write(descriptor, payload[:3])

    monkeypatch.setattr(storage_module.os, "write", interrupted_partial_write)

    store.claim_round(claimed)

    assert calls > 2
    assert store.read_state() == claimed


def _spied_adapters(adapters: AdapterSet) -> tuple[AdapterSet, dict[str, Mock]]:
    spies = {
        "task_source": Mock(wraps=adapters.task_source),
        "scene_anchor": Mock(wraps=adapters.scene_anchor),
        "router": Mock(wraps=adapters.router),
        "world_model": Mock(wraps=adapters.world_model),
        "rollout_scorer": Mock(wraps=adapters.rollout_scorer),
        "executor": Mock(wraps=adapters.executor),
        "evaluator": Mock(wraps=adapters.evaluator),
        "trainer": Mock(wraps=adapters.trainer),
        "memory": Mock(wraps=adapters.memory),
    }
    policies: dict[str, Mock] = {}
    for expert_id, policy in adapters.policies.items():
        spy = Mock(wraps=policy)
        policies[expert_id] = spy
        spies[f"policy:{expert_id}"] = spy
    return (
        replace(
            adapters,
            task_source=spies["task_source"],
            scene_anchor=spies["scene_anchor"],
            router=spies["router"],
            policies=policies,
            world_model=spies["world_model"],
            rollout_scorer=spies["rollout_scorer"],
            executor=spies["executor"],
            evaluator=spies["evaluator"],
            trainer=spies["trainer"],
            memory=spies["memory"],
        ),
        spies,
    )


def _assert_completed_adapters_were_not_called(stage: RoundStage, spies: dict[str, Mock]) -> None:
    spies["task_source"].next_task.assert_not_called()
    spies["scene_anchor"].anchor.assert_not_called()
    if stage in {
        RoundStage.CANDIDATES_SCORED,
        RoundStage.EXECUTED,
        RoundStage.COMPLETE,
    }:
        spies["router"].shortlist.assert_not_called()
        spies["memory"].lookup.assert_not_called()
        spies["world_model"].imagine.assert_not_called()
        spies["rollout_scorer"].score.assert_not_called()
        for name, spy in spies.items():
            if name.startswith("policy:"):
                spy.act.assert_not_called()
    if stage in {RoundStage.EXECUTED, RoundStage.COMPLETE}:
        spies["executor"].execute.assert_not_called()
    if stage is RoundStage.COMPLETE:
        spies["evaluator"].evaluate.assert_not_called()
        spies["trainer"].plan.assert_not_called()
        spies["memory"].append.assert_not_called()


def _assert_operation_id_not_called(
    kind: OperationKind, operation_id: str, spies: dict[str, Mock]
) -> None:
    methods: list[Mock]
    if kind is OperationKind.TASK:
        methods = [spies["task_source"].next_task]
    elif kind is OperationKind.ANCHOR:
        methods = [spies["scene_anchor"].anchor]
    elif kind is OperationKind.ROUTE:
        methods = [spies["router"].shortlist]
    elif kind is OperationKind.ACT:
        methods = [spy.act for name, spy in spies.items() if name.startswith("policy:")]
    elif kind is OperationKind.IMAGINE:
        methods = [spies["world_model"].imagine]
    elif kind is OperationKind.SCORE:
        methods = [spies["rollout_scorer"].score]
    elif kind is OperationKind.EXECUTE:
        methods = [spies["executor"].execute]
    elif kind is OperationKind.EVALUATE:
        methods = [spies["evaluator"].evaluate]
    elif kind is OperationKind.TRAIN_PLAN:
        methods = [spies["trainer"].plan]
    elif kind is OperationKind.TRAIN_EXECUTE:
        methods = [spies["trainer"].execute]
    else:
        methods = [spies["memory"].append]
    assert all(
        call.kwargs.get("operation_id") != operation_id
        for method in methods
        for call in method.call_args_list
    )


def _immutable_round_bytes(run_dir: Path, round_id: str) -> dict[Path, bytes]:
    round_dir = run_dir / "rounds" / round_id
    return {
        path.relative_to(round_dir): path.read_bytes()
        for path in round_dir.rglob("*")
        if path.is_file() and path.name != "state.json"
    }


def _tree_bytes(root: Path) -> dict[Path, bytes]:
    if not root.exists():
        return {}
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _assert_journal_content_unique(path: Path) -> None:
    if not path.exists():
        return
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    canonical = [json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records]
    assert len(canonical) == len(set(canonical))


def _assert_unique_durable_ids(run_dir: Path) -> None:
    task_path = run_dir / "tasks.jsonl"
    memory_path = run_dir / "memory.jsonl"
    _assert_journal_content_unique(task_path)
    _assert_journal_content_unique(memory_path)
    tasks = [json.loads(line) for line in task_path.read_text(encoding="utf-8").splitlines()]
    memory = [json.loads(line) for line in memory_path.read_text(encoding="utf-8").splitlines()]
    assert len({record["operation_id"] for record in tasks}) == len(tasks)
    assert len({record["record_id"] for record in memory}) == len(memory)
    assert len({record["operation_id"] for record in memory}) == len(memory)

    verified_records: list[tuple[str, str]] = []
    provenance_rounds: list[str] = []
    for directory in sorted(path for path in (run_dir / "verified").iterdir() if path.is_dir()):
        trajectory_text = (directory / "trajectory.json").read_text(encoding="utf-8")
        provenance_text = (directory / "provenance.json").read_text(encoding="utf-8")
        trajectory = json.loads(trajectory_text)
        provenance = json.loads(provenance_text)
        assert trajectory["trajectory_id"] == directory.name
        verified_records.append((trajectory_text, provenance_text))
        provenance_rounds.append(provenance["round_id"])
    assert len(verified_records) == len(set(verified_records))
    assert len(provenance_rounds) == len(set(provenance_rounds))


def _scenario(target: OperationKind) -> DeterministicScenario:
    if target is OperationKind.TRAIN_EXECUTE:
        return DeterministicScenario(
            evaluation_score=0.2,
            evaluation_label=DiagnosisLabel.PF,
        )
    return DeterministicScenario()


def _operation_case(
    tmp_path: Path, target: OperationKind, prefix: str
) -> tuple[EvoWorldConfig, Path, AdapterSet, ExecutionMode]:
    config = _config(tmp_path)
    run_dir = tmp_path / f"{prefix}-{target.value}"
    adapters = build_deterministic_adapters(config, run_dir, _scenario(target))
    execution_mode = (
        ExecutionMode.EXECUTE if target is OperationKind.TRAIN_EXECUTE else ExecutionMode.PLAN
    )
    return config, run_dir, adapters, execution_mode


def _config(tmp_path: Path) -> EvoWorldConfig:
    return EvoWorldConfig(
        schema_version=1,
        seed=7,
        run_root=(tmp_path / "runs").resolve(),
        mode="dry_run",
        grounding_mode=GroundingMode.NEAREST,
        grounding_temperature=0.5,
        router_weight=0.25,
        reliability_threshold=0.7,
        rollout_horizon=3,
        executor=None,
        experts=tuple(
            ExpertSpec(
                expert_id=expert_id,
                backend="local",
                task_families=("manipulation",),
                checkpoint=None,
                action_schema="joint_velocity",
            )
            for expert_id in ("groot", "pi05")
        ),
    )

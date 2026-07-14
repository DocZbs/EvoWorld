"""Durability contracts shared by every process-backed adapter."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
from pathlib import Path
import threading
import time

import pytest

from evoworld.adapters.operations import OperationLedger, OperationReceiptError
from evoworld.ports import OperationConflictError, OperationInDoubtError


def _encode(value: dict[str, object]) -> dict[str, object]:
    return value


def _decode(value: dict[str, object]) -> dict[str, object]:
    return value


def _process_claim(
    root: str,
    counter: str,
    barrier: object,
    results: object,
) -> None:
    ledger = OperationLedger(Path(root))
    getattr(barrier, "wait")()

    def execute() -> dict[str, object]:
        with Path(counter).open("a", encoding="utf-8") as handle:
            handle.write("effect\n")
            handle.flush()
        time.sleep(0.1)
        return {"ok": True}

    value = ledger.perform(
        "qwen.evaluate",
        "cross-process-operation",
        {"checkpoint": "qwen"},
        execute,
        _encode,
        _decode,
    )
    getattr(results, "put")(value)


def test_completed_retry_returns_receipt_without_repeating_effect(tmp_path: Path) -> None:
    ledger = OperationLedger(tmp_path)
    calls = 0

    def execute() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"value": 7}

    first = ledger.perform(
        "openpi.act", "round-1:act", {"model": "pi05"}, execute, _encode, _decode
    )
    second = ledger.perform(
        "openpi.act", "round-1:act", {"model": "pi05"}, execute, _encode, _decode
    )

    assert first == second == {"value": 7}
    assert calls == 1
    receipt = json.loads(
        (ledger.entry_dir("openpi.act", "round-1:act") / "receipt.json").read_text()
    )
    assert receipt["operation_id"] == "round-1:act"


def test_same_id_with_different_canonical_input_conflicts(tmp_path: Path) -> None:
    ledger = OperationLedger(tmp_path)
    ledger.perform(
        "groot.act", "operation-1", {"model": "n1.6"}, lambda: {"ok": True}, _encode, _decode
    )

    with pytest.raises(OperationConflictError):
        ledger.perform(
            "groot.act", "operation-1", {"model": "other"}, lambda: {"ok": True}, _encode, _decode
        )


def test_claim_without_receipt_is_honestly_in_doubt(tmp_path: Path) -> None:
    ledger = OperationLedger(tmp_path)

    with pytest.raises(KeyboardInterrupt):
        ledger.perform(
            "ctrl_world.imagine",
            "operation-1",
            {"actions": [[0.0] * 7]},
            lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
            _encode,
            _decode,
        )

    with pytest.raises(OperationInDoubtError):
        ledger.perform(
            "ctrl_world.imagine",
            "operation-1",
            {"actions": [[0.0] * 7]},
            lambda: {"must": "not run"},
            _encode,
            _decode,
        )


def test_concurrent_callers_share_one_completed_effect(tmp_path: Path) -> None:
    ledger = OperationLedger(tmp_path)
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def execute() -> dict[str, object]:
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(2)
        return {"ok": True}

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            ledger.perform,
            "qwen.evaluate",
            "operation-1",
            {"checkpoint": "qwen"},
            execute,
            _encode,
            _decode,
        )
        assert started.wait(2)
        second = pool.submit(
            ledger.perform,
            "qwen.evaluate",
            "operation-1",
            {"checkpoint": "qwen"},
            execute,
            _encode,
            _decode,
        )
        release.set()
        assert first.result() == second.result() == {"ok": True}
    assert calls == 1


def test_cross_process_callers_are_serialized_by_the_operation_lock(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    counter = tmp_path / "effects.log"
    workers = [
        context.Process(
            target=_process_claim,
            args=(str(tmp_path), str(counter), barrier, results),
        )
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(5)
        assert worker.exitcode == 0

    assert [results.get(timeout=1), results.get(timeout=1)] == [
        {"ok": True},
        {"ok": True},
    ]
    assert counter.read_text(encoding="utf-8") == "effect\n"


def test_entries_use_safe_hashes_and_distinct_ids_are_isolated(tmp_path: Path) -> None:
    ledger = OperationLedger(tmp_path)
    first = ledger.entry_dir("openpi.act", "../../escape")
    second = ledger.entry_dir("openpi.act", "other")

    assert first.parent == second.parent == tmp_path.resolve() / ".operations" / "entries"
    assert first != second
    assert len(first.name) == 64
    assert "/" not in first.name


def test_corrupt_or_symlinked_receipts_fail_closed(tmp_path: Path) -> None:
    ledger = OperationLedger(tmp_path)
    inputs = {"model": "pi05"}
    ledger.perform("openpi.act", "operation-1", inputs, lambda: {"ok": True}, _encode, _decode)
    entry = ledger.entry_dir("openpi.act", "operation-1")
    (entry / "receipt.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(OperationReceiptError):
        ledger.perform("openpi.act", "operation-1", inputs, lambda: {"ok": False}, _encode, _decode)

    other_root = tmp_path / "other"
    other_root.mkdir()
    other = OperationLedger(other_root)
    other.perform("openpi.act", "operation-2", inputs, lambda: {"ok": True}, _encode, _decode)
    receipt = other.entry_dir("openpi.act", "operation-2") / "receipt.json"
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    receipt.unlink()
    receipt.symlink_to(outside)
    with pytest.raises(OperationReceiptError, match="symlink"):
        other.perform("openpi.act", "operation-2", inputs, lambda: {"ok": False}, _encode, _decode)


def test_nonfinite_input_is_rejected_before_claim(tmp_path: Path) -> None:
    ledger = OperationLedger(tmp_path)
    with pytest.raises(OperationReceiptError, match="finite JSON"):
        ledger.perform(
            "openpi.act", "operation-1", {"value": float("nan")}, lambda: {}, _encode, _decode
        )
    assert not ledger.entry_dir("openpi.act", "operation-1").exists()

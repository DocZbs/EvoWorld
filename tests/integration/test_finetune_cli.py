from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from evoworld.cli import main
from training.evaluator.qwen_vl import EvaluatorConfigError
from training.experts.groot import GrootConfigError
from training.experts.openpi import OpenPiConfigError
from training.common.schema import TrainingPlan
from training.world_model.ctrl_world import CtrlWorldConfigError


TARGETS = ("pi05", "pi0", "pi0_fast", "groot_n16", "ctrl_world", "qwen_evaluator")
TARGET_ERRORS = {
    "pi05": OpenPiConfigError,
    "pi0": OpenPiConfigError,
    "pi0_fast": OpenPiConfigError,
    "groot_n16": GrootConfigError,
    "ctrl_world": CtrlWorldConfigError,
    "qwen_evaluator": EvaluatorConfigError,
}


def _resign_job(job: Path, mutate: object) -> None:
    payload = json.loads(job.read_text(encoding="utf-8"))
    mutate(payload)  # type: ignore[operator]
    payload["plan_sha256"] = sha256(
        json.dumps(payload["plan"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload.pop("job_sha256", None)
    payload["job_sha256"] = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    job.write_text(json.dumps(payload), encoding="utf-8")


def _config(tmp_path: Path, target: str = "pi05") -> Path:
    config = tmp_path / "finetune.toml"
    text = (
        "[finetune]\n"
        f'job_root = "{(tmp_path / "jobs").as_posix()}"\n'
        'recipe_config = "recipe.toml"\n'
        'manifest = "manifest.json"\n'
    )
    if target != "qwen_evaluator":
        text += 'dataset_root = "dataset"\noutput_root = "outputs"\nbackend_root = "backend"\n'
    if target == "ctrl_world":
        text += (
            'repository_root = "repo"\n'
            'dataset_meta_info_root = "metadata"\n'
            'svd_model = "models/svd"\n'
            'clip_model = "models/clip"\n'
            'policy_checkpoint = "models/pi.pt"\n'
            'action_adapter = "models/adapter.pt"\n'
            'synthetic_output_root = "synthetic"\n'
            'starting_checkpoint = "models/ctrl.pt"\n'
            'ctrl_world_python = "backend-python"\n'
            'dataset_name = "evoworld"\n'
            'tag = "evoworld"\n'
            'wandb_run_name = "evoworld"\n'
            'wandb_project_name = "evoworld"\n'
        )
    if target == "qwen_evaluator":
        text += (
            'workspace_root = "."\n'
            'validation_export = "validation.jsonl"\n'
            'export_dataset_root = "dataset"\n'
            f'export_sha256 = "{"a" * 64}"\n'
            f'train_sha256 = "{"b" * 64}"\n'
            f'validation_sha256 = "{"c" * 64}"\n'
            f'dataset_revision = "{"d" * 64}"\n'
            "example_count = 2\n"
        )
    config.write_text(text, encoding="utf-8")
    return config


def _fake_plan(target: str, tmp_path: Path) -> TrainingPlan:
    return TrainingPlan(
        schema_version=1,
        job_id=f"{target}-job",
        target=target,
        manifest_sha256="a" * 64,
        upstream_revision="b" * 40,
        cwd=tmp_path.resolve(),
        argv=("fake-train", target),
        environment={},
        expected_outputs=(Path("checkpoint"),),
    )


@pytest.mark.parametrize("target", TARGETS)
def test_finetune_plan_writes_job_without_execution(
    tmp_path: Path, target: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []

    def compose(settings: object, selected: str) -> object:
        del settings
        calls.append(selected)
        from evoworld.training import PlannedTarget

        return PlannedTarget(_fake_plan(selected, tmp_path), None)

    monkeypatch.setattr("evoworld.training._compose_target", compose)
    assert (
        main(["finetune", "plan", "--config", str(_config(tmp_path, target)), "--target", target])
        == 0
    )
    assert calls == [target]
    job = tmp_path / "jobs" / f"{target}-job" / "job.json"
    assert job.is_file()
    payload = json.loads(job.read_text(encoding="utf-8"))
    assert payload["target"] == target
    assert payload["status"] == "planned"
    assert json.loads(capsys.readouterr().out)["job"] == str(job.resolve())


def test_execute_success_records_logs_receipt_and_rejects_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    plan = _fake_plan("pi05", tmp_path)
    from evoworld.training import PlannedTarget

    monkeypatch.setattr(
        "evoworld.training._compose_target", lambda settings, target: PlannedTarget(plan, None)
    )
    assert main(["finetune", "plan", "--config", str(config), "--target", "pi05"]) == 0
    job = tmp_path / "jobs/pi05-job/job.json"

    def run(actual: TrainingPlan, stdout: object, stderr: object) -> int:
        assert actual == plan
        Path(tmp_path / "checkpoint").write_text("weights", encoding="utf-8")
        stdout.write("trained\n")  # type: ignore[attr-defined]
        stderr.write("warning\n")  # type: ignore[attr-defined]
        return 0

    monkeypatch.setattr("evoworld.training._run_generic", run)
    assert main(["finetune", "execute", "--job", str(job)]) == 0
    receipt = json.loads((job.parent / "execution.json").read_text(encoding="utf-8"))
    assert receipt["exit_code"] == 0
    assert receipt["status"] == "completed"
    assert set(receipt) == {
        "attempt_index",
        "exit_code",
        "job_id",
        "outputs",
        "plan_sha256",
        "receipt_sha256",
        "receipt_version",
        "recorded_at",
        "schema_version",
        "status",
        "stderr",
        "stdout",
        "target",
    }
    assert (job.parent / "attempts/0001/stdout.log").read_text() == "trained\n"
    assert (job.parent / "attempts/0001/receipt.json").read_bytes() == (
        job.parent / "execution.json"
    ).read_bytes()
    assert main(["finetune", "execute", "--job", str(job)]) == 2


def test_execute_nonzero_can_retry_only_when_outputs_are_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    plan = _fake_plan("pi05", tmp_path)
    from evoworld.training import PlannedTarget

    monkeypatch.setattr(
        "evoworld.training._compose_target", lambda settings, target: PlannedTarget(plan, None)
    )
    assert main(["finetune", "plan", "--config", str(config), "--target", "pi05"]) == 0
    job = tmp_path / "jobs/pi05-job/job.json"
    exits = iter((9, 0))

    def run(actual: TrainingPlan, stdout: object, stderr: object) -> int:
        del actual, stdout, stderr
        code = next(exits)
        if code == 0:
            (tmp_path / "checkpoint").write_text("ok", encoding="utf-8")
        return code

    monkeypatch.setattr("evoworld.training._run_generic", run)
    assert main(["finetune", "execute", "--job", str(job)]) == 1
    assert main(["finetune", "execute", "--job", str(job)]) == 0
    assert (job.parent / "attempts/0002/receipt.json").is_file()


def test_execute_rejects_tamper_symlink_claim_and_output_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    plan = _fake_plan("pi05", tmp_path)
    from evoworld.training import PlannedTarget

    monkeypatch.setattr(
        "evoworld.training._compose_target", lambda settings, target: PlannedTarget(plan, None)
    )
    assert main(["finetune", "plan", "--config", str(config), "--target", "pi05"]) == 0
    job = tmp_path / "jobs/pi05-job/job.json"
    original = job.read_text()
    payload = json.loads(original)
    payload["plan"]["argv"].append("--tampered")
    job.write_text(json.dumps(payload), encoding="utf-8")
    assert main(["finetune", "execute", "--job", str(job)]) == 2
    job.write_text(original, encoding="utf-8")
    (job.parent / ".execute.claim").write_text("claimed", encoding="utf-8")
    assert main(["finetune", "execute", "--job", str(job)]) == 2
    (job.parent / ".execute.claim").unlink()
    (tmp_path / "checkpoint").write_text("partial", encoding="utf-8")
    assert main(["finetune", "execute", "--job", str(job)]) == 2

    symlink_job = tmp_path / "job-link.json"
    symlink_job.symlink_to(job)
    assert main(["finetune", "execute", "--job", str(symlink_job)]) == 2


def test_execute_claim_allows_only_one_concurrent_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    plan = _fake_plan("pi05", tmp_path)
    from evoworld.training import PlannedTarget

    monkeypatch.setattr(
        "evoworld.training._compose_target", lambda settings, target: PlannedTarget(plan, None)
    )
    assert main(["finetune", "plan", "--config", str(config), "--target", "pi05"]) == 0
    job = tmp_path / "jobs/pi05-job/job.json"
    entered = threading.Event()
    release = threading.Event()
    results: list[int] = []

    def run(actual: TrainingPlan, stdout: object, stderr: object) -> int:
        del actual, stdout, stderr
        entered.set()
        assert release.wait(timeout=5)
        (tmp_path / "checkpoint").write_text("ok", encoding="utf-8")
        return 0

    monkeypatch.setattr("evoworld.training._run_generic", run)
    worker = threading.Thread(
        target=lambda: results.append(main(["finetune", "execute", "--job", str(job)]))
    )
    worker.start()
    assert entered.wait(timeout=5)
    assert main(["finetune", "execute", "--job", str(job)]) == 2
    release.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert results == [0]


def test_delayed_preclaim_caller_rechecks_completion_before_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    plan = _fake_plan("pi05", tmp_path)
    from evoworld import training as orchestration
    from evoworld.training import PlannedTarget

    monkeypatch.setattr(
        "evoworld.training._compose_target", lambda settings, target: PlannedTarget(plan, None)
    )
    assert main(["finetune", "plan", "--config", str(config), "--target", "pi05"]) == 0
    job = tmp_path / "jobs/pi05-job/job.json"
    delayed = threading.Event()
    release = threading.Event()
    runner_calls: list[str] = []
    first_ident: list[int] = []
    original_claim = orchestration._claim

    def claim(path: Path) -> None:
        if threading.get_ident() == first_ident[0]:
            delayed.set()
            assert release.wait(timeout=5)
        original_claim(path)

    def run(actual: TrainingPlan, stdout: object, stderr: object) -> int:
        del actual, stdout, stderr
        runner_calls.append("run")
        (tmp_path / "checkpoint").write_text("ok", encoding="utf-8")
        return 0

    monkeypatch.setattr("evoworld.training._claim", claim)
    monkeypatch.setattr("evoworld.training._run_generic", run)

    first_results: list[int] = []

    def first() -> None:
        first_ident.append(threading.get_ident())
        first_results.append(main(["finetune", "execute", "--job", str(job)]))

    worker = threading.Thread(target=first)
    worker.start()
    assert delayed.wait(timeout=5)
    assert main(["finetune", "execute", "--job", str(job)]) == 0
    release.set()
    worker.join(timeout=5)
    assert first_results == [2]
    assert runner_calls == ["run"]


def test_retry_rejects_tampered_failed_receipt_and_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    plan = _fake_plan("pi05", tmp_path)
    from evoworld.training import PlannedTarget

    monkeypatch.setattr(
        "evoworld.training._compose_target", lambda settings, target: PlannedTarget(plan, None)
    )
    monkeypatch.setattr("evoworld.training._run_generic", lambda plan, stdout, stderr: 7)
    assert main(["finetune", "plan", "--config", str(config), "--target", "pi05"]) == 0
    job = tmp_path / "jobs/pi05-job/job.json"
    assert main(["finetune", "execute", "--job", str(job)]) == 1
    receipt_path = job.parent / "attempts/0001/receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["job_id"] == plan.job_id
    assert receipt["target"] == plan.target
    assert receipt["attempt_index"] == 1
    assert receipt["exit_code"] == 7
    assert receipt["stdout"]["path"] == "attempts/0001/stdout.log"
    (job.parent / "attempts/0001/stdout.log").write_text("tampered", encoding="utf-8")
    assert main(["finetune", "execute", "--job", str(job)]) == 2


def test_retry_rejects_foreign_log_path_even_with_recomputed_receipt_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    plan = _fake_plan("pi05", tmp_path)
    from evoworld.training import PlannedTarget

    monkeypatch.setattr(
        "evoworld.training._compose_target", lambda settings, target: PlannedTarget(plan, None)
    )
    monkeypatch.setattr("evoworld.training._run_generic", lambda plan, stdout, stderr: 3)
    assert main(["finetune", "plan", "--config", str(config), "--target", "pi05"]) == 0
    job = tmp_path / "jobs/pi05-job/job.json"
    assert main(["finetune", "execute", "--job", str(job)]) == 1
    receipt_path = job.parent / "attempts/0001/receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["stdout"]["path"] = "../foreign.log"
    receipt.pop("receipt_sha256")
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)
    receipt["receipt_sha256"] = sha256(canonical.encode()).hexdigest()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert main(["finetune", "execute", "--job", str(job)]) == 2


@pytest.mark.parametrize("mutation", ["minimal", "symlink"])
def test_retry_rejects_minimal_receipt_or_symlinked_log(
    tmp_path: Path, mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    plan = _fake_plan("pi05", tmp_path)
    from evoworld.training import PlannedTarget

    monkeypatch.setattr(
        "evoworld.training._compose_target", lambda settings, target: PlannedTarget(plan, None)
    )
    monkeypatch.setattr("evoworld.training._run_generic", lambda plan, stdout, stderr: 4)
    assert main(["finetune", "plan", "--config", str(config), "--target", "pi05"]) == 0
    job = tmp_path / "jobs/pi05-job/job.json"
    assert main(["finetune", "execute", "--job", str(job)]) == 1
    attempt = job.parent / "attempts/0001"
    if mutation == "minimal":
        (attempt / "receipt.json").write_text('{"status":"failed"}', encoding="utf-8")
    else:
        outside = tmp_path / "outside.log"
        outside.write_text("foreign", encoding="utf-8")
        (attempt / "stderr.log").unlink()
        (attempt / "stderr.log").symlink_to(outside)
    assert main(["finetune", "execute", "--job", str(job)]) == 2


@pytest.mark.parametrize("target", TARGETS)
def test_target_validation_errors_are_config_errors(
    tmp_path: Path, target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = TARGET_ERRORS[target]

    def reject(settings: object, selected: str) -> object:
        del settings, selected
        raise error("invalid target fixture")

    monkeypatch.setattr("evoworld.training._compose_target", reject)
    assert (
        main(["finetune", "plan", "--config", str(_config(tmp_path, target)), "--target", target])
        == 2
    )


@pytest.mark.parametrize("target", TARGETS)
def test_execute_target_drift_is_a_config_error(
    tmp_path: Path, target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evoworld.training import PlannedTarget

    calls = 0

    def compose(settings: object, selected: str) -> PlannedTarget:
        nonlocal calls
        del settings
        calls += 1
        if calls > 1:
            raise TARGET_ERRORS[target]("stale target fixture")
        return PlannedTarget(_fake_plan(selected, tmp_path), None)

    config = _config(tmp_path, target)
    monkeypatch.setattr("evoworld.training._compose_target", compose)
    assert main(["finetune", "plan", "--config", str(config), "--target", target]) == 0
    job = tmp_path / "jobs" / f"{target}-job" / "job.json"
    assert main(["finetune", "execute", "--job", str(job)]) == 2
    assert not (job.parent / ".execute.claim").exists()


def test_runner_os_error_is_runtime_failure_with_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    plan = _fake_plan("pi05", tmp_path)
    from evoworld.training import PlannedTarget

    monkeypatch.setattr(
        "evoworld.training._compose_target", lambda settings, target: PlannedTarget(plan, None)
    )
    monkeypatch.setattr(
        "evoworld.training._run_generic",
        lambda plan, stdout, stderr: (_ for _ in ()).throw(OSError("cannot spawn")),
    )
    assert main(["finetune", "plan", "--config", str(config), "--target", "pi05"]) == 0
    job = tmp_path / "jobs/pi05-job/job.json"
    assert main(["finetune", "execute", "--job", str(job)]) == 1
    receipt = json.loads((job.parent / "attempts/0001/receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["exit_code"] == 1


def test_runner_interruption_retains_in_doubt_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    plan = _fake_plan("pi05", tmp_path)
    from evoworld.training import PlannedTarget, execute_finetune

    monkeypatch.setattr(
        "evoworld.training._compose_target", lambda settings, target: PlannedTarget(plan, None)
    )

    def interrupt(plan: object, stdout: object, stderr: object) -> int:
        del plan, stdout, stderr
        raise KeyboardInterrupt

    monkeypatch.setattr("evoworld.training._run_generic", interrupt)
    assert main(["finetune", "plan", "--config", str(config), "--target", "pi05"]) == 0
    job = tmp_path / "jobs/pi05-job/job.json"
    with pytest.raises(KeyboardInterrupt):
        execute_finetune(job)
    assert (job.parent / ".execute.claim").is_file()


def test_prelaunch_target_validation_is_cleanly_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    plan = _fake_plan("pi05", tmp_path)
    from evoworld.training import PlannedTarget

    calls = 0

    def executor(plan: object, stdout: object, stderr: object, mark_launched: object) -> int:
        nonlocal calls
        del plan, stdout, stderr
        calls += 1
        if calls == 1:
            raise EvaluatorConfigError("preflight rejected")
        mark_launched()  # type: ignore[operator]
        (tmp_path / "checkpoint").write_text("ok", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        "evoworld.training._compose_target",
        lambda settings, target: PlannedTarget(plan, executor),
    )
    assert main(["finetune", "plan", "--config", str(config), "--target", "pi05"]) == 0
    job = tmp_path / "jobs/pi05-job/job.json"
    assert main(["finetune", "execute", "--job", str(job)]) == 2
    assert not (job.parent / ".execute.claim").exists()
    assert list((job.parent / "attempts").iterdir()) == []
    assert main(["finetune", "execute", "--job", str(job)]) == 0
    assert (job.parent / "attempts/0001/receipt.json").is_file()


def test_postlaunch_target_validation_retains_in_doubt_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    plan = _fake_plan("pi05", tmp_path)
    from evoworld.training import PlannedTarget

    def executor(plan: object, stdout: object, stderr: object, mark_launched: object) -> int:
        del plan, stdout, stderr
        mark_launched()  # type: ignore[operator]
        raise EvaluatorConfigError("post-launch validation rejected")

    monkeypatch.setattr(
        "evoworld.training._compose_target",
        lambda settings, target: PlannedTarget(plan, executor),
    )
    assert main(["finetune", "plan", "--config", str(config), "--target", "pi05"]) == 0
    job = tmp_path / "jobs/pi05-job/job.json"
    assert main(["finetune", "execute", "--job", str(job)]) == 2
    assert (job.parent / ".execute.claim").is_file()
    assert not (job.parent / "attempts/0001/receipt.json").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["config"].__setitem__("path", 7),
        lambda payload: payload.__setitem__("target", []),
        lambda payload: payload["plan"].__setitem__("cwd", 7),
        lambda payload: payload["plan"].__setitem__("unknown", "field"),
        lambda payload: payload.__setitem__("output_snapshot", [{"path": 7}]),
        lambda payload: payload["output_snapshot"][0].__setitem__("fingerprint", {"kind": []}),
        lambda payload: payload["output_snapshot"][0].__setitem__(
            "fingerprint", {"kind": "file", "sha256": float("nan")}
        ),
    ],
    ids=(
        "config-path-7",
        "unhashable-target",
        "plan-path-7",
        "unknown-plan-field",
        "malformed-snapshot",
        "unhashable-fingerprint-kind",
        "nonfinite-job-json",
    ),
)
def test_malicious_resigned_job_is_rejected_before_claim(
    tmp_path: Path, mutation: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    plan = _fake_plan("pi05", tmp_path)
    from evoworld.training import PlannedTarget

    monkeypatch.setattr(
        "evoworld.training._compose_target", lambda settings, target: PlannedTarget(plan, None)
    )
    assert main(["finetune", "plan", "--config", str(config), "--target", "pi05"]) == 0
    job = tmp_path / "jobs/pi05-job/job.json"
    _resign_job(job, mutation)
    assert main(["finetune", "execute", "--job", str(job)]) == 2
    assert not (job.parent / ".execute.claim").exists()


def test_relative_parent_job_root_is_canonical_across_plan_and_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = _config(config_dir)
    config.write_text(
        config.read_text().replace(str(config_dir / "jobs"), "../jobs"), encoding="utf-8"
    )
    plan = _fake_plan("pi05", tmp_path)
    from evoworld.training import PlannedTarget

    monkeypatch.setattr(
        "evoworld.training._compose_target", lambda settings, target: PlannedTarget(plan, None)
    )

    def run(plan: object, stdout: object, stderr: object) -> int:
        del plan, stdout, stderr
        (tmp_path / "checkpoint").write_text("ok", encoding="utf-8")
        return 0

    monkeypatch.setattr("evoworld.training._run_generic", run)
    assert main(["finetune", "plan", "--config", str(config), "--target", "pi05"]) == 0
    job = tmp_path / "jobs/pi05-job/job.json"
    assert job.is_file()
    assert main(["finetune", "execute", "--job", str(job)]) == 0


def test_completed_receipt_cannot_be_rebound_to_unrelated_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    plan = _fake_plan("pi05", tmp_path)
    from evoworld.training import PlannedTarget

    runner_calls = 0

    def run(plan: object, stdout: object, stderr: object) -> int:
        nonlocal runner_calls
        del plan, stdout, stderr
        runner_calls += 1
        (tmp_path / "checkpoint").write_text("ok", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        "evoworld.training._compose_target", lambda settings, target: PlannedTarget(plan, None)
    )
    monkeypatch.setattr("evoworld.training._run_generic", run)
    assert main(["finetune", "plan", "--config", str(config), "--target", "pi05"]) == 0
    job = tmp_path / "jobs/pi05-job/job.json"
    assert main(["finetune", "execute", "--job", str(job)]) == 0
    unrelated = tmp_path / "unrelated.bin"
    unrelated.write_bytes(b"unrelated")
    for receipt_path in (
        job.parent / "attempts/0001/receipt.json",
        job.parent / "execution.json",
    ):
        receipt = json.loads(receipt_path.read_text())
        receipt["outputs"] = [
            {
                "path": str(unrelated.resolve()),
                "fingerprint": {"kind": "file", "sha256": sha256(b"unrelated").hexdigest()},
            }
        ]
        receipt.pop("receipt_sha256")
        receipt["receipt_sha256"] = sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert main(["finetune", "execute", "--job", str(job)]) == 2
    assert runner_calls == 1


@pytest.mark.parametrize("exception", [RuntimeError("prelaunch"), KeyboardInterrupt()])
def test_premarker_base_exception_cleans_attempt_and_claim(
    tmp_path: Path, exception: BaseException, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    plan = _fake_plan("pi05", tmp_path)
    from evoworld.training import PlannedTarget, execute_finetune

    def executor(plan: object, stdout: object, stderr: object, mark_launched: object) -> int:
        del plan, stdout, stderr, mark_launched
        raise exception

    monkeypatch.setattr(
        "evoworld.training._compose_target",
        lambda settings, target: PlannedTarget(plan, executor),
    )
    assert main(["finetune", "plan", "--config", str(config), "--target", "pi05"]) == 0
    job = tmp_path / "jobs/pi05-job/job.json"
    if isinstance(exception, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            execute_finetune(job)
    else:
        assert main(["finetune", "execute", "--job", str(job)]) == 1
    assert not (job.parent / ".execute.claim").exists()
    assert list((job.parent / "attempts").iterdir()) == []


def test_premarker_foreign_attempt_artifact_retains_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    plan = _fake_plan("pi05", tmp_path)
    from evoworld.training import PlannedTarget

    job = tmp_path / "jobs/pi05-job/job.json"

    def executor(plan: object, stdout: object, stderr: object, mark_launched: object) -> int:
        del plan, stdout, stderr, mark_launched
        (job.parent / "attempts/0001/foreign.bin").write_bytes(b"foreign")
        raise RuntimeError("prelaunch failure with foreign artifact")

    monkeypatch.setattr(
        "evoworld.training._compose_target",
        lambda settings, target: PlannedTarget(plan, executor),
    )
    assert main(["finetune", "plan", "--config", str(config), "--target", "pi05"]) == 0
    assert main(["finetune", "execute", "--job", str(job)]) == 2
    assert (job.parent / ".execute.claim").is_file()
    assert (job.parent / "attempts/0001/foreign.bin").is_file()


@pytest.mark.parametrize("mutation", ["old-log", "extra-attempt"])
def test_completed_job_validates_entire_attempt_history(
    tmp_path: Path, mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    plan = _fake_plan("pi05", tmp_path)
    from evoworld.training import PlannedTarget

    exits = iter((6, 0))
    runner_calls = 0

    def run(plan: object, stdout: object, stderr: object) -> int:
        nonlocal runner_calls
        del plan, stdout, stderr
        runner_calls += 1
        code = next(exits)
        if code == 0:
            (tmp_path / "checkpoint").write_text("ok", encoding="utf-8")
        return code

    monkeypatch.setattr(
        "evoworld.training._compose_target", lambda settings, target: PlannedTarget(plan, None)
    )
    monkeypatch.setattr("evoworld.training._run_generic", run)
    assert main(["finetune", "plan", "--config", str(config), "--target", "pi05"]) == 0
    job = tmp_path / "jobs/pi05-job/job.json"
    assert main(["finetune", "execute", "--job", str(job)]) == 1
    assert main(["finetune", "execute", "--job", str(job)]) == 0
    if mutation == "old-log":
        (job.parent / "attempts/0001/stdout.log").write_text("tampered", encoding="utf-8")
    else:
        (job.parent / "attempts/0003").mkdir()
    assert main(["finetune", "execute", "--job", str(job)]) == 2
    assert runner_calls == 2


def test_nonfinite_failed_receipt_json_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    plan = _fake_plan("pi05", tmp_path)
    from evoworld.training import PlannedTarget

    monkeypatch.setattr(
        "evoworld.training._compose_target", lambda settings, target: PlannedTarget(plan, None)
    )
    monkeypatch.setattr("evoworld.training._run_generic", lambda plan, stdout, stderr: 5)
    assert main(["finetune", "plan", "--config", str(config), "--target", "pi05"]) == 0
    job = tmp_path / "jobs/pi05-job/job.json"
    assert main(["finetune", "execute", "--job", str(job)]) == 1
    receipt = job.parent / "attempts/0001/receipt.json"
    receipt.write_text('{"exit_code":NaN}', encoding="utf-8")
    assert main(["finetune", "execute", "--job", str(job)]) == 2


@pytest.mark.parametrize(
    ("target", "failure"),
    [("pi05", "return"), ("qwen_evaluator", "called-process")],
)
def test_premarker_nonzero_is_receipted_runtime_failure_and_retryable(
    tmp_path: Path, target: str, failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, target)
    plan = _fake_plan(target, tmp_path)
    from evoworld.training import PlannedTarget

    calls = 0

    def executor(plan: object, stdout: object, stderr: object, mark_launched: object) -> int:
        nonlocal calls
        del plan, stdout, stderr
        calls += 1
        if calls == 1:
            if failure == "called-process":
                raise subprocess.CalledProcessError(9, ("uvx", "runtime-probe"))
            return 9
        mark_launched()  # type: ignore[operator]
        (tmp_path / "checkpoint").write_text("ok", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        "evoworld.training._compose_target",
        lambda settings, selected: PlannedTarget(plan, executor),
    )
    assert main(["finetune", "plan", "--config", str(config), "--target", target]) == 0
    job = tmp_path / "jobs" / f"{target}-job" / "job.json"
    assert main(["finetune", "execute", "--job", str(job)]) == 1
    receipt = json.loads((job.parent / "attempts/0001/receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["exit_code"] == 9
    assert receipt["attempt_index"] == 1
    assert not (job.parent / ".execute.claim").exists()
    assert main(["finetune", "execute", "--job", str(job)]) == 0
    assert (job.parent / "attempts/0002/receipt.json").is_file()


def test_execute_rejects_changed_config_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    plan = _fake_plan("pi05", tmp_path)
    from evoworld.training import PlannedTarget

    monkeypatch.setattr(
        "evoworld.training._compose_target", lambda settings, target: PlannedTarget(plan, None)
    )
    assert main(["finetune", "plan", "--config", str(config), "--target", "pi05"]) == 0
    config.write_text(config.read_text() + "# drift\n", encoding="utf-8")
    assert main(["finetune", "execute", "--job", str(tmp_path / "jobs/pi05-job/job.json")]) == 2


def test_console_and_python_module_parser_parity() -> None:
    console = str(Path(sys.executable).parent / "evoworld")
    direct = subprocess.run(
        [console, "finetune", "plan"], capture_output=True, text=True, check=False
    )
    module = subprocess.run(
        [sys.executable, "-m", "evoworld", "finetune", "plan"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (direct.returncode, direct.stdout, direct.stderr) == (
        module.returncode,
        module.stdout,
        module.stderr,
    )


def test_job_digest_is_canonical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    plan = _fake_plan("pi05", tmp_path)
    from evoworld.training import PlannedTarget

    monkeypatch.setattr(
        "evoworld.training._compose_target", lambda settings, target: PlannedTarget(plan, None)
    )
    assert main(["finetune", "plan", "--config", str(config), "--target", "pi05"]) == 0
    job = tmp_path / "jobs/pi05-job/job.json"
    payload = json.loads(job.read_text())
    digest = payload.pop("job_sha256")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert digest == sha256(canonical.encode()).hexdigest()

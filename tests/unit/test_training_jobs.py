import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from training.common.jobs import JobExecutionError, StaleJobError, execute_job, write_job
from training.common.schema import TrainingPlan, TrainingSchemaError


def _plan(**changes: object) -> TrainingPlan:
    values: dict[str, object] = {
        "schema_version": 1,
        "job_id": "job-001",
        "target": "pi05",
        "manifest_sha256": "a" * 64,
        "upstream_revision": "b" * 40,
        "cwd": Path("/tmp/evoworld-training"),
        "argv": ("python", "train.py", "--steps", "5000"),
        "environment": {"WANDB_MODE": "offline"},
        "expected_outputs": (Path("checkpoint"),),
    }
    values.update(changes)
    return TrainingPlan(**values)  # type: ignore[arg-type]


def test_execute_rejects_changed_manifest(tmp_path: Path) -> None:
    job = write_job(tmp_path, _plan(manifest_sha256="a" * 64))

    with pytest.raises(StaleJobError, match="manifest"):
        execute_job(
            job,
            current_manifest_sha256="c" * 64,
            current_upstream_revision="b" * 40,
            runner=lambda _: (),
        )


def test_execute_rejects_changed_upstream_revision(tmp_path: Path) -> None:
    job = write_job(tmp_path, _plan())

    with pytest.raises(StaleJobError, match="upstream"):
        execute_job(
            job,
            current_manifest_sha256="a" * 64,
            current_upstream_revision="c" * 40,
            runner=lambda _: (),
        )


def test_completed_job_is_idempotent(tmp_path: Path) -> None:
    job = write_job(tmp_path, _plan())
    calls: list[TrainingPlan] = []

    def runner(plan: TrainingPlan) -> tuple[Path, ...]:
        calls.append(plan)
        return (Path("checkpoint"),)

    first = execute_job(
        job,
        current_manifest_sha256="a" * 64,
        current_upstream_revision="b" * 40,
        runner=runner,
    )
    second = execute_job(
        job,
        current_manifest_sha256="a" * 64,
        current_upstream_revision="b" * 40,
        runner=runner,
    )

    assert first.status == second.status == "completed"
    assert calls == [_plan()]


def test_plan_rejects_secret_environment_key() -> None:
    with pytest.raises(TrainingSchemaError, match="allowlisted"):
        _plan(environment={"HF_TOKEN": "secret"})


@pytest.mark.parametrize(
    "job_id", ("/tmp/escape", "../escape", "nested/job", "nested\\job", ".", "..", " ")
)
def test_plan_rejects_job_id_that_is_not_a_safe_single_segment(job_id: str) -> None:
    with pytest.raises(TrainingSchemaError, match="job_id"):
        _plan(job_id=job_id)


def test_upstream_revision_is_required_at_execution(tmp_path: Path) -> None:
    job = write_job(tmp_path, _plan())

    with pytest.raises(TypeError, match="current_upstream_revision"):
        execute_job(job, current_manifest_sha256="a" * 64, runner=lambda _: ())  # type: ignore[call-arg]


def test_failed_runner_is_recorded_in_job_log(tmp_path: Path) -> None:
    job = write_job(tmp_path, _plan())

    def runner(_: TrainingPlan) -> None:
        raise RuntimeError("backend unavailable")

    with pytest.raises(JobExecutionError, match="failed"):
        execute_job(
            job,
            current_manifest_sha256="a" * 64,
            current_upstream_revision="b" * 40,
            runner=runner,
        )

    assert "backend unavailable" in job.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "output",
    (Path("."), Path("../checkpoint"), Path("nested\\checkpoint"), Path("/tmp/checkpoint")),
)
def test_execute_rejects_unsafe_output_reference(tmp_path: Path, output: Path) -> None:
    job = write_job(tmp_path, _plan())

    with pytest.raises(TrainingSchemaError, match="output"):
        execute_job(
            job,
            current_manifest_sha256="a" * 64,
            current_upstream_revision="b" * 40,
            runner=lambda _: (output,),
        )


def test_existing_artifact_revalidates_output_references(tmp_path: Path) -> None:
    job = write_job(tmp_path, _plan())
    payload = json.loads(job.read_text(encoding="utf-8"))
    payload["status"] = "completed"
    payload["output_references"] = ["."]
    job.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TrainingSchemaError, match="output"):
        execute_job(
            job,
            current_manifest_sha256="a" * 64,
            current_upstream_revision="b" * 40,
            runner=lambda _: (),
        )

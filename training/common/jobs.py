"""Persistent, idempotent job artifacts for fine-tuning launchers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

from training.common.schema import (
    TrainingPlan,
    TrainingSchemaError,
    validate_relative_posix_path,
)


class StaleJobError(RuntimeError):
    """Raised when a job no longer matches its manifest or upstream revision."""


class JobExecutionError(RuntimeError):
    """Raised after persisting a failed runner invocation in the job artifact."""


Runner = Callable[[TrainingPlan], tuple[Path, ...] | None]


@dataclass(frozen=True, slots=True)
class TrainingJob:
    """A persisted plan and its append-only execution state."""

    path: Path
    plan: TrainingPlan
    status: str
    logs: tuple[str, ...]
    output_references: tuple[Path, ...]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _payload(job: TrainingJob) -> dict[str, object]:
    return {
        "plan": job.plan.to_dict(),
        "status": job.status,
        "logs": list(job.logs),
        "output_references": [path.as_posix() for path in job.output_references],
    }


def _read_job(path: Path) -> TrainingJob:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        plan_values = raw["plan"]
        if not isinstance(plan_values, dict):
            raise TypeError("plan must be an object")
        plan = TrainingPlan(
            schema_version=plan_values["schema_version"],
            job_id=plan_values["job_id"],
            target=plan_values["target"],
            manifest_sha256=plan_values["manifest_sha256"],
            upstream_revision=plan_values["upstream_revision"],
            cwd=Path(plan_values["cwd"]),
            argv=tuple(plan_values["argv"]),
            environment=plan_values["environment"],
            expected_outputs=tuple(Path(value) for value in plan_values["expected_outputs"]),
        )
        status = raw["status"]
        logs = tuple(raw["logs"])
        outputs = tuple(Path(value) for value in raw["output_references"])
    except (KeyError, TypeError, json.JSONDecodeError, TrainingSchemaError) as error:
        raise TrainingSchemaError(f"invalid job artifact {path}") from error
    if status not in {"planned", "completed", "failed"}:
        raise TrainingSchemaError(f"invalid job status {status!r}")
    if not all(isinstance(log, str) for log in logs):
        raise TrainingSchemaError("job logs must contain strings")
    for output in outputs:
        validate_relative_posix_path(output, "output_references")
    return TrainingJob(path=path, plan=plan, status=status, logs=logs, output_references=outputs)


def write_job(job_root: Path, plan: TrainingPlan) -> Path:
    """Persist a planned job at ``<job_root>/<job_id>/job.json`` without execution."""
    path = job_root / plan.job_id / "job.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = _read_job(path)
        if existing.plan == plan:
            return path
        raise TrainingSchemaError(f"job_id {plan.job_id!r} already has a different immutable plan")
    job = TrainingJob(path=path, plan=plan, status="planned", logs=(), output_references=())
    _write_json(path, _payload(job))
    return path


def execute_job(
    job_path: Path,
    *,
    current_manifest_sha256: str,
    current_upstream_revision: str,
    runner: Runner,
) -> TrainingJob:
    """Run a current job once, recording output references or a failure log."""
    job = _read_job(job_path)
    if job.plan.manifest_sha256 != current_manifest_sha256:
        raise StaleJobError("job manifest revision is stale")
    if job.plan.upstream_revision != current_upstream_revision:
        raise StaleJobError("job upstream revision is stale")
    if job.status == "completed":
        return job
    try:
        outputs = tuple(runner(job.plan) or ())
    except Exception as error:
        failed = TrainingJob(
            path=job.path,
            plan=job.plan,
            status="failed",
            logs=(*job.logs, f"runner failed: {error}"),
            output_references=job.output_references,
        )
        _write_json(job.path, _payload(failed))
        raise JobExecutionError(f"training job {job.plan.job_id} failed") from error
    for path in outputs:
        validate_relative_posix_path(path, "runner output references")
    completed = TrainingJob(
        path=job.path,
        plan=job.plan,
        status="completed",
        logs=(*job.logs, "runner completed"),
        output_references=outputs,
    )
    _write_json(job.path, _payload(completed))
    return completed

"""Plan-first, fail-closed orchestration for EvoWorld fine-tuning targets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import tomllib
from typing import IO, cast

from evoworld.contracts import JsonObject
from training.common.schema import TrainingPlan, TrainingSchemaError
from training.evaluator.qwen_vl import (
    EvaluatorConfigError,
    load_qwen_config,
    plan_qwen_training,
    run_qwen_training,
)
from training.evaluator.schema import EvaluatorExport, EvaluatorSchemaError
from training.experts.groot import GrootConfigError, load_groot_config, plan_groot_training
from training.experts.openpi import (
    OpenPiConfigError,
    load_openpi_config,
    load_openpi_manifest,
    plan_openpi_training,
)
from training.world_model.ctrl_world import (
    CtrlWorldConfigError,
    CtrlWorldTrainConfig,
    load_ctrl_world_config,
    plan_ctrl_world_training,
)


TARGETS = ("pi05", "pi0", "pi0_fast", "groot_n16", "ctrl_world", "qwen_evaluator")
_TARGET_SET = frozenset(TARGETS)
_JOB_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "target",
        "config",
        "plan",
        "plan_sha256",
        "output_snapshot",
        "job_sha256",
    }
)
_COMMON_FIELDS = frozenset(
    {"job_root", "recipe_config", "manifest", "dataset_root", "output_root", "backend_root"}
)
_BASE_FIELDS = frozenset({"job_root", "recipe_config", "manifest"})
_CTRL_FIELDS = frozenset(
    {
        "repository_root",
        "dataset_meta_info_root",
        "svd_model",
        "clip_model",
        "policy_checkpoint",
        "action_adapter",
        "synthetic_output_root",
        "starting_checkpoint",
        "ctrl_world_python",
        "dataset_name",
        "tag",
        "wandb_run_name",
        "wandb_project_name",
    }
)
_QWEN_FIELDS = frozenset(
    {
        "workspace_root",
        "validation_export",
        "export_dataset_root",
        "export_sha256",
        "train_sha256",
        "validation_sha256",
        "dataset_revision",
        "example_count",
    }
)
_RECEIPT_VERSION = 1
_FAILED_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "receipt_version",
        "recorded_at",
        "job_id",
        "target",
        "plan_sha256",
        "attempt_index",
        "exit_code",
        "status",
        "stdout",
        "stderr",
        "receipt_sha256",
    }
)
_COMPLETED_RECEIPT_FIELDS = _FAILED_RECEIPT_FIELDS | {"outputs"}
_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "job_id",
        "target",
        "manifest_sha256",
        "upstream_revision",
        "cwd",
        "argv",
        "environment",
        "expected_outputs",
    }
)
_TARGET_VALIDATION_ERRORS = (
    OpenPiConfigError,
    GrootConfigError,
    CtrlWorldConfigError,
    EvaluatorConfigError,
    EvaluatorSchemaError,
    TrainingSchemaError,
)


class FineTuneConfigError(ValueError):
    """Raised for invalid, stale, claimed, or tampered fine-tuning artifacts."""


class FineTuneRuntimeError(RuntimeError):
    """Raised when an authorized training process exits unsuccessfully."""


LaunchMarker = Callable[[], None]
Executor = Callable[[TrainingPlan, IO[str], IO[str], LaunchMarker], int]


@dataclass(frozen=True, slots=True)
class FineTuneSettings:
    """A strict target-specific TOML table with paths resolved at its source."""

    path: Path
    sha256: str
    values: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PlannedTarget:
    """A real target plan plus its target-owned execution adapter, when needed."""

    plan: TrainingPlan
    executor: Executor | None


def plan_finetune(config_path: Path, target: str) -> JsonObject:
    """Validate and persist one immutable offline plan without running a backend."""
    if target not in _TARGET_SET:
        raise FineTuneConfigError(f"unsupported fine-tuning target {target!r}")
    settings = _load_settings(config_path, target)
    planned = _compose_checked(settings, target)
    plan = planned.plan
    if plan.target != target:
        raise FineTuneConfigError("target planner returned a mismatched target")
    job_root = _path(settings, "job_root")
    job_path = job_root / plan.job_id / "job.json"
    _reject_symlink_chain(job_path)
    _validate_initial_outputs(plan, settings)
    job_dir = job_path.parent.absolute()
    if any(_paths_overlap(job_dir, output) for output in _output_roots(plan, settings)):
        raise FineTuneConfigError("job_root and target output paths must not overlap")
    plan_payload = plan.to_dict()
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "planned",
        "target": target,
        "config": {"path": str(settings.path), "sha256": settings.sha256},
        "plan": plan_payload,
        "plan_sha256": _digest(plan_payload),
        "output_snapshot": _output_snapshot(plan, settings),
    }
    payload["job_sha256"] = _digest(payload)
    _write_exclusive_json(job_path, payload)
    return {
        "job": str(job_path.resolve()),
        "job_id": plan.job_id,
        "plan_sha256": cast(str, payload["plan_sha256"]),
        "status": "planned",
        "target": target,
    }


def execute_finetune(job_path: Path) -> JsonObject:
    """Claim, revalidate, and execute one immutable plan with durable receipts."""
    # This first read is only a cheap location/schema preflight. Every authoritative
    # decision is repeated below while holding the exclusive claim.
    preflight_path, _, _ = _load_job(job_path)
    claim = preflight_path.parent / ".execute.claim"
    _claim(claim)
    backend_launched = False
    attempt: Path | None = None
    attempt_created = False
    owned_logs: set[Path] = set()
    failure_receipted = False
    claim_released = False

    def mark_backend_launched() -> None:
        nonlocal backend_launched
        backend_launched = True

    try:
        path, payload, recorded_plan = _load_job(job_path)
        if path != preflight_path:
            raise FineTuneConfigError("job path changed while acquiring its execution claim")
        config_binding = cast(dict[str, object], payload["config"])
        config_path = Path(cast(str, config_binding["path"]))
        target = cast(str, payload["target"])
        settings = _load_settings(config_path, target)
        if settings.sha256 != config_binding["sha256"]:
            raise FineTuneConfigError("fine-tuning configuration bytes changed after planning")
        expected_job = (_path(settings, "job_root") / recorded_plan.job_id / "job.json").absolute()
        if path != expected_job:
            raise FineTuneConfigError("job path does not match its configured job root")
        if _completion_present(path.parent):
            _reject_completed(path.parent, recorded_plan, payload, settings)
        _validate_attempt_history(path.parent, recorded_plan, cast(str, payload["plan_sha256"]))
        fresh = _compose_checked(settings, target)
        if fresh.plan != recorded_plan:
            raise FineTuneConfigError(
                "fine-tuning source, inputs, revision, or environment drifted"
            )
        if _output_snapshot(recorded_plan, settings) != payload["output_snapshot"]:
            raise FineTuneConfigError("fine-tuning output is partial or changed after planning")

        attempt = _next_attempt(path.parent)
        attempt.mkdir(parents=False, exist_ok=False)
        attempt_created = True
        stdout_path = attempt / "stdout.log"
        stderr_path = attempt / "stderr.log"
        stdout = stdout_path.open("x", encoding="utf-8")
        owned_logs.add(stdout_path)
        try:
            stderr = stderr_path.open("x", encoding="utf-8")
            owned_logs.add(stderr_path)
        except BaseException:
            stdout.close()
            raise
        with stdout, stderr:
            try:
                if fresh.executor is not None:
                    exit_code = fresh.executor(recorded_plan, stdout, stderr, mark_backend_launched)
                else:
                    mark_backend_launched()
                    exit_code = _run_generic(recorded_plan, stdout, stderr)
            except subprocess.CalledProcessError as error:
                exit_code = int(error.returncode)
            except OSError as error:
                stderr.write(f"{type(error).__name__}: {error}\n")
                exit_code = 1
            except _TARGET_VALIDATION_ERRORS as error:
                raise FineTuneConfigError("target runtime validation failed") from error
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise FineTuneRuntimeError("fine-tuning runner returned an invalid exit status")
        receipt = _receipt(
            path.parent,
            recorded_plan,
            cast(str, payload["plan_sha256"]),
            int(attempt.name),
            exit_code,
            stdout_path,
            stderr_path,
        )
        if exit_code:
            _write_exclusive_json(attempt / "receipt.json", receipt)
            failure_receipted = True
            _release_claim(claim)
            claim_released = True
            raise FineTuneRuntimeError(f"fine-tuning process exited with status {exit_code}")
        outputs = _verified_outputs(recorded_plan, settings)
        receipt.pop("receipt_sha256")
        receipt["outputs"] = outputs
        receipt["receipt_sha256"] = _digest(receipt)
        _write_exclusive_json(attempt / "receipt.json", receipt)
        _write_exclusive_json(path.parent / "execution.json", receipt)
        _release_claim(claim)
        claim_released = True
        return {
            "exit_code": 0,
            "job": str(path),
            "receipt": str((path.parent / "execution.json").resolve()),
            "status": "completed",
            "target": target,
        }
    except BaseException:
        if failure_receipted and claim_released:
            raise
        # Before the explicit launch marker, no backend can have observed this
        # attempt, so expected local scaffolding is safe to remove. After the
        # marker, every unknown outcome remains durably in doubt.
        if not backend_launched:
            if attempt is not None:
                _discard_unlaunched_attempt(
                    attempt,
                    attempt_created=attempt_created,
                    owned_logs=owned_logs,
                )
            if not claim_released:
                _release_claim(claim)
        raise


def _load_settings(path: Path, target: str) -> FineTuneSettings:
    try:
        if path.is_symlink():
            raise FineTuneConfigError("fine-tuning config must not be a symlink")
        _reject_symlink_chain(path)
        resolved = path.resolve(strict=True)
        raw_bytes = resolved.read_bytes()
        document = tomllib.loads(raw_bytes.decode("utf-8"))
        values = document["finetune"]
    except FineTuneConfigError:
        raise
    except (OSError, UnicodeError, KeyError, tomllib.TOMLDecodeError) as error:
        raise FineTuneConfigError("fine-tuning config could not be loaded") from error
    if set(document) != {"finetune"} or not isinstance(values, dict):
        raise FineTuneConfigError("config must contain only one [finetune] table")
    allowed = set(_COMMON_FIELDS)
    required = set(_COMMON_FIELDS)
    if target == "ctrl_world":
        allowed.update(_CTRL_FIELDS)
        required.update(_CTRL_FIELDS)
    elif target == "qwen_evaluator":
        allowed = set(_BASE_FIELDS | _QWEN_FIELDS)
        required = set(_BASE_FIELDS | _QWEN_FIELDS)
    missing = required - set(values)
    unknown = set(values) - allowed
    if missing or unknown:
        raise FineTuneConfigError(
            f"invalid [finetune] fields; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    normalized: dict[str, object] = {}
    path_fields = (_COMMON_FIELDS | _CTRL_FIELDS | _QWEN_FIELDS) - {
        "dataset_name",
        "tag",
        "wandb_run_name",
        "wandb_project_name",
        "export_sha256",
        "train_sha256",
        "validation_sha256",
        "dataset_revision",
        "example_count",
    }
    for key, value in values.items():
        if key in path_fields:
            if not isinstance(value, str) or not value.strip():
                raise FineTuneConfigError(f"{key} must be a nonblank path string")
            candidate = Path(value)
            absolute = candidate if candidate.is_absolute() else resolved.parent / candidate
            _reject_symlink_chain(absolute)
            normalized[key] = absolute.resolve(strict=False)
        else:
            normalized[key] = value
    return FineTuneSettings(resolved, sha256(raw_bytes).hexdigest(), normalized)


def _compose_target(settings: FineTuneSettings, target: str) -> PlannedTarget:
    if target in {"pi05", "pi0", "pi0_fast"}:
        manifest = load_openpi_manifest(_path(settings, "manifest"))
        plan = plan_openpi_training(
            manifest,
            load_openpi_config(_path(settings, "recipe_config"), target),
            openpi_root=_path(settings, "backend_root"),
            output_root=_path(settings, "output_root"),
            dataset_root=_path(settings, "dataset_root"),
            manifest_path=_path(settings, "manifest"),
            config_path=_path(settings, "recipe_config"),
        )
        return PlannedTarget(plan, None)
    if target == "groot_n16":
        manifest = load_openpi_manifest(_path(settings, "manifest"))
        plan = plan_groot_training(
            manifest,
            load_groot_config(_path(settings, "recipe_config")),
            groot_root=_path(settings, "backend_root"),
            dataset_root=_path(settings, "dataset_root"),
            output_root=_path(settings, "output_root"),
            manifest_path=_path(settings, "manifest"),
        )
        return PlannedTarget(plan, None)
    if target == "ctrl_world":
        manifest = load_openpi_manifest(_path(settings, "manifest"))
        baseline = load_ctrl_world_config(_path(settings, "recipe_config"))
        config = CtrlWorldTrainConfig(
            dataset_root_path=_path(settings, "dataset_root"),
            dataset_meta_info_path=_path(settings, "dataset_meta_info_root"),
            manifest_path=_path(settings, "manifest"),
            output_root=_path(settings, "output_root"),
            svd_model_path=_path(settings, "svd_model"),
            clip_model_path=_path(settings, "clip_model"),
            policy_checkpoint_path=_path(settings, "policy_checkpoint"),
            action_adapter_path=_path(settings, "action_adapter"),
            synthetic_output_root=_path(settings, "synthetic_output_root"),
            starting_checkpoint=_path(settings, "starting_checkpoint"),
            ctrl_world_python=_path(settings, "ctrl_world_python"),
            dataset_name=_string(settings, "dataset_name"),
            annotation_name=cast(str, baseline["annotation_name"]),
            tag=_string(settings, "tag"),
            wandb_run_name=_string(settings, "wandb_run_name"),
            wandb_project_name=_string(settings, "wandb_project_name"),
            max_train_steps=cast(int, baseline["max_train_steps"]),
            world_size=cast(int, baseline["world_size"]),
        )
        plan = plan_ctrl_world_training(
            manifest,
            config,
            ctrl_world_root=_path(settings, "backend_root"),
            repository_root=_path(settings, "repository_root"),
        )
        return PlannedTarget(plan, None)
    if target == "qwen_evaluator":
        export = EvaluatorExport(
            path=_path(settings, "manifest"),
            validation_path=_path(settings, "validation_export"),
            dataset_root=_path(settings, "export_dataset_root"),
            sha256=_string(settings, "export_sha256"),
            train_sha256=_string(settings, "train_sha256"),
            validation_sha256=_string(settings, "validation_sha256"),
            dataset_revision=_string(settings, "dataset_revision"),
            example_count=_integer(settings, "example_count"),
        )
        qwen_config = load_qwen_config(_path(settings, "recipe_config"))
        workspace = _path(settings, "workspace_root")
        plan = plan_qwen_training(export, qwen_config, working_directory=workspace)

        def execute_qwen(
            actual: TrainingPlan,
            stdout: IO[str],
            stderr: IO[str],
            mark_launched: LaunchMarker,
        ) -> int:
            def command_runner(command_plan: TrainingPlan) -> None:
                mark_launched()
                result = subprocess.run(
                    command_plan.argv,
                    cwd=command_plan.cwd,
                    env={**os.environ, **command_plan.environment},
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                    text=True,
                )
                if result.returncode:
                    raise subprocess.CalledProcessError(result.returncode, command_plan.argv)

            run_qwen_training(
                actual,
                export,
                qwen_config,
                workspace_root=workspace,
                command_runner=command_runner,
            )
            return 0

        return PlannedTarget(plan, execute_qwen)
    raise FineTuneConfigError(f"unsupported fine-tuning target {target!r}")


def _compose_checked(settings: FineTuneSettings, target: str) -> PlannedTarget:
    try:
        return _compose_target(settings, target)
    except FineTuneConfigError:
        raise
    except _TARGET_VALIDATION_ERRORS as error:
        raise FineTuneConfigError(f"{target} fine-tuning validation failed") from error


def _run_generic(plan: TrainingPlan, stdout: IO[str], stderr: IO[str]) -> int:
    result = subprocess.run(
        plan.argv,
        cwd=plan.cwd,
        env={**os.environ, **plan.environment},
        stdout=stdout,
        stderr=stderr,
        check=False,
        text=True,
    )
    return result.returncode


def _load_job(path: Path) -> tuple[Path, dict[str, object], TrainingPlan]:
    try:
        if path.is_symlink():
            raise FineTuneConfigError("job path must not be a symlink")
        resolved = path.resolve(strict=True)
        _reject_symlink_chain(resolved)
        raw = json.loads(resolved.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except FineTuneConfigError:
        raise
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise FineTuneConfigError("job artifact could not be loaded") from error
    if not isinstance(raw, dict) or set(raw) != _JOB_FIELDS:
        raise FineTuneConfigError("job artifact has an invalid schema")
    payload = cast(dict[str, object], raw)
    digest = payload["job_sha256"]
    unsigned = {key: value for key, value in payload.items() if key != "job_sha256"}
    if not _valid_hash(digest, 64) or digest != _digest(unsigned):
        raise FineTuneConfigError("job artifact digest does not match its content")
    schema_version = payload["schema_version"]
    status = payload["status"]
    target = payload["target"]
    plan_sha256 = payload["plan_sha256"]
    if (
        isinstance(schema_version, bool)
        or schema_version != 1
        or not isinstance(status, str)
        or status != "planned"
        or not isinstance(target, str)
        or target not in _TARGET_SET
        or not _valid_hash(plan_sha256, 64)
    ):
        raise FineTuneConfigError("job artifact is not a supported planned job")
    config = payload["config"]
    if not isinstance(config, dict) or set(config) != {"path", "sha256"}:
        raise FineTuneConfigError("job config binding is invalid")
    config_path = config.get("path")
    config_sha256 = config.get("sha256")
    if (
        not isinstance(config_path, str)
        or not config_path.strip()
        or not Path(config_path).is_absolute()
        or not _valid_hash(config_sha256, 64)
    ):
        raise FineTuneConfigError("job config binding has invalid path or digest values")
    plan_values = payload["plan"]
    if (
        not isinstance(plan_values, dict)
        or set(plan_values) != _PLAN_FIELDS
        or plan_sha256 != _digest(plan_values)
    ):
        raise FineTuneConfigError("job plan digest is invalid")
    schema_version = plan_values.get("schema_version")
    cwd = plan_values.get("cwd")
    argv = plan_values.get("argv")
    environment = plan_values.get("environment")
    expected_outputs = plan_values.get("expected_outputs")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or not isinstance(cwd, str)
        or not cwd.strip()
        or not Path(cwd).is_absolute()
        or not isinstance(argv, list)
        or not all(isinstance(item, str) for item in argv)
        or not isinstance(environment, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in environment.items()
        )
        or not isinstance(expected_outputs, list)
        or not all(isinstance(item, str) for item in expected_outputs)
    ):
        raise FineTuneConfigError("job plan contains invalid raw field types")
    for name in ("job_id", "target", "manifest_sha256", "upstream_revision"):
        if not isinstance(plan_values.get(name), str):
            raise FineTuneConfigError(f"job plan {name} must be a string")
    try:
        plan = TrainingPlan(
            schema_version=schema_version,
            job_id=cast(str, plan_values["job_id"]),
            target=cast(str, plan_values["target"]),
            manifest_sha256=cast(str, plan_values["manifest_sha256"]),
            upstream_revision=cast(str, plan_values["upstream_revision"]),
            cwd=Path(cwd),
            argv=tuple(cast(list[str], argv)),
            environment=cast(dict[str, str], environment),
            expected_outputs=tuple(Path(item) for item in cast(list[str], expected_outputs)),
        )
    except (KeyError, TypeError, ValueError, TrainingSchemaError) as error:
        raise FineTuneConfigError("job plan is invalid") from error
    _validate_output_snapshot(payload["output_snapshot"])
    if plan.target != target or resolved.name != "job.json" or resolved.parent.name != plan.job_id:
        raise FineTuneConfigError("job path or target does not match its immutable plan")
    return resolved, payload, plan


def _output_snapshot(plan: TrainingPlan, settings: FineTuneSettings) -> object:
    return [
        {"path": str(path), "fingerprint": _fingerprint(path)}
        for path in _output_roots(plan, settings)
    ]


def _validate_output_snapshot(value: object) -> None:
    if not isinstance(value, list) or not value:
        raise FineTuneConfigError("job output snapshot must be a nonempty list")
    seen_paths: set[str] = set()
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"path", "fingerprint"}:
            raise FineTuneConfigError("job output snapshot entry has an invalid schema")
        path = entry.get("path")
        if (
            not isinstance(path, str)
            or not path.strip()
            or not Path(path).is_absolute()
            or path in seen_paths
        ):
            raise FineTuneConfigError("job output snapshot path is invalid or duplicated")
        seen_paths.add(path)
        _validate_fingerprint(entry.get("fingerprint"), "job output snapshot")


def _validate_fingerprint(value: object, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise FineTuneConfigError(f"{name} fingerprint is invalid")
    kind = value.get("kind")
    if not isinstance(kind, str) or kind not in {"file", "directory"}:
        raise FineTuneConfigError(f"{name} fingerprint is invalid")
    if kind == "file":
        if set(value) != {"kind", "sha256"} or not _valid_hash(value.get("sha256"), 64):
            raise FineTuneConfigError(f"{name} file fingerprint is invalid")
        return
    if set(value) != {"kind", "entries"} or not isinstance(value.get("entries"), list):
        raise FineTuneConfigError(f"{name} directory fingerprint is invalid")
    paths: list[str] = []
    for item in cast(list[object], value["entries"]):
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise FineTuneConfigError(f"{name} directory entry is invalid")
        relative = item.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or "\\" in relative
            or ".." in Path(relative).parts
            or not _valid_hash(item.get("sha256"), 64)
        ):
            raise FineTuneConfigError(f"{name} directory entry path or digest is invalid")
        paths.append(relative)
    if paths != sorted(set(paths)):
        raise FineTuneConfigError(f"{name} directory entries must be sorted and unique")


def _validate_initial_outputs(plan: TrainingPlan, settings: FineTuneSettings) -> None:
    roots = _output_roots(plan, settings)
    if plan.target == "ctrl_world":
        for root in roots:
            for relative in plan.expected_outputs:
                if (root / relative).exists():
                    raise FineTuneConfigError("Ctrl-World final output already exists")
        return
    if any(_fingerprint(root) is not None for root in roots):
        raise FineTuneConfigError("target output already exists; refusing to plan an overwrite")


def _output_roots(plan: TrainingPlan, settings: FineTuneSettings) -> tuple[Path, ...]:
    if plan.target == "qwen_evaluator":
        return tuple(
            (_path(settings, "workspace_root") / relative).resolve()
            for relative in plan.expected_outputs
        )
    for flag in ("--output-dir", "--output_dir"):
        if flag in plan.argv:
            index = plan.argv.index(flag)
            try:
                return (Path(plan.argv[index + 1]).resolve(),)
            except IndexError as error:
                raise FineTuneConfigError("training plan has a malformed output flag") from error
    if "--request-path" in plan.argv:
        index = plan.argv.index("--request-path")
        try:
            request = json.loads(
                Path(plan.argv[index + 1]).read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
            )
            output = request["output_dir"]
        except (
            IndexError,
            OSError,
            ValueError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
        ) as error:
            raise FineTuneConfigError("Ctrl-World request has an invalid output binding") from error
        if not isinstance(output, str):
            raise FineTuneConfigError("Ctrl-World output binding must be a string")
        return (Path(output).resolve(),)
    return tuple((plan.cwd / relative).resolve() for relative in plan.expected_outputs)


def _verified_outputs(plan: TrainingPlan, settings: FineTuneSettings) -> list[dict[str, object]]:
    roots = _output_roots(plan, settings)
    paths: tuple[Path, ...]
    if plan.target == "qwen_evaluator" or not any(
        flag in plan.argv for flag in ("--output-dir", "--output_dir", "--request-path")
    ):
        paths = roots
    else:
        paths = tuple(root / relative for root in roots for relative in plan.expected_outputs)
    result: list[dict[str, object]] = []
    for path in paths:
        fingerprint = _fingerprint(path)
        if fingerprint is None:
            raise FineTuneRuntimeError(f"training process did not produce {path}")
        result.append({"path": str(path), "fingerprint": fingerprint})
    return result


def _fingerprint(path: Path) -> object:
    if path.is_symlink():
        raise FineTuneConfigError(f"output path must not be a symlink: {path}")
    if not path.exists():
        return None
    if path.is_file():
        return {"kind": "file", "sha256": _file_hash(path)}
    if not path.is_dir():
        raise FineTuneConfigError(f"output path has an unsupported type: {path}")
    entries: list[dict[str, str]] = []
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise FineTuneConfigError(f"output tree contains a symlink: {item}")
        if item.is_file():
            entries.append({"path": item.relative_to(path).as_posix(), "sha256": _file_hash(item)})
    return {"kind": "directory", "entries": entries}


def _claim(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FineTuneConfigError("fine-tuning job is already claimed or in doubt") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps({"pid": os.getpid()}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _release_claim(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError as error:
        raise FineTuneConfigError("execution claim disappeared while it was owned") from error


def _discard_unlaunched_attempt(
    attempt: Path, *, attempt_created: bool, owned_logs: set[Path]
) -> None:
    if not attempt_created:
        if attempt.exists() or attempt.is_symlink():
            raise FineTuneConfigError("unlaunched attempt path was created by another actor")
        return
    if attempt.is_symlink() or not attempt.is_dir():
        raise FineTuneConfigError("unlaunched attempt directory is unsafe")
    children = {child.name: child for child in attempt.iterdir()}
    if not set(children) <= {"stdout.log", "stderr.log"}:
        raise FineTuneConfigError("unlaunched attempt contains unexpected artifacts")
    for child in children.values():
        if child not in owned_logs or child.is_symlink() or not child.is_file():
            raise FineTuneConfigError("unlaunched attempt log is unsafe")
    for child in children.values():
        child.unlink()
    attempt.rmdir()


def _next_attempt(job_dir: Path) -> Path:
    attempts = job_dir / "attempts"
    attempts.mkdir(exist_ok=True)
    indices = []
    for path in attempts.iterdir():
        if not path.is_dir() or not path.name.isdigit():
            raise FineTuneConfigError("attempt history contains an invalid entry")
        indices.append(int(path.name))
    return attempts / f"{max(indices, default=0) + 1:04d}"


def _validate_attempt_history(job_dir: Path, plan: TrainingPlan, plan_sha256: str) -> None:
    attempts = job_dir / "attempts"
    if attempts.is_symlink():
        raise FineTuneConfigError("attempt history must not be a symlink")
    if not attempts.exists():
        return
    if not attempts.is_dir():
        raise FineTuneConfigError("attempt history is not a directory")
    expected_index = 1
    for attempt in sorted(attempts.iterdir()):
        if attempt.is_symlink() or not attempt.is_dir() or attempt.name != f"{expected_index:04d}":
            raise FineTuneConfigError("attempt history is non-contiguous or unsafe")
        expected_index += 1
        children = {child.name for child in attempt.iterdir()}
        if children != {"receipt.json", "stderr.log", "stdout.log"}:
            raise FineTuneConfigError("attempt history contains a partial attempt")
        for name in children:
            child = attempt / name
            if child.is_symlink() or not child.is_file():
                raise FineTuneConfigError("attempt history contains an unsafe artifact")
        try:
            receipt = json.loads(
                (attempt / "receipt.json").read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise FineTuneConfigError("attempt receipt is corrupt") from error
        _validate_receipt(
            cast(object, receipt),
            job_dir=job_dir,
            plan=plan,
            plan_sha256=plan_sha256,
            attempt_index=expected_index - 1,
            completed=False,
        )


def _completion_present(job_dir: Path) -> bool:
    execution = job_dir / "execution.json"
    if execution.is_symlink():
        raise FineTuneConfigError("execution receipt must not be a symlink")
    return execution.exists()


def _reject_completed(
    job_dir: Path,
    plan: TrainingPlan,
    job_payload: Mapping[str, object],
    settings: FineTuneSettings,
) -> None:
    execution = job_dir / "execution.json"
    if not execution.is_file():
        raise FineTuneConfigError("execution receipt is not a regular file")
    try:
        execution_bytes = execution.read_bytes()
        receipt = json.loads(execution_bytes.decode("utf-8"), parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise FineTuneConfigError("execution receipt is corrupt") from error
    if not isinstance(receipt, dict):
        raise FineTuneConfigError("execution receipt must be an object")
    attempt_index = receipt.get("attempt_index")
    if isinstance(attempt_index, bool) or not isinstance(attempt_index, int):
        raise FineTuneConfigError("execution receipt has an invalid attempt index")
    try:
        expected_outputs = _verified_outputs(plan, settings)
    except FineTuneRuntimeError as error:
        raise FineTuneConfigError("completed output is missing or invalid") from error
    _validate_receipt(
        receipt,
        job_dir=job_dir,
        plan=plan,
        plan_sha256=cast(str, job_payload["plan_sha256"]),
        attempt_index=attempt_index,
        completed=True,
        expected_outputs=expected_outputs,
    )
    _validate_completed_history(
        job_dir,
        plan,
        cast(str, job_payload["plan_sha256"]),
        attempt_index,
        execution_bytes,
        expected_outputs,
    )
    raise FineTuneConfigError("completed fine-tuning jobs cannot be executed again")


def _validate_completed_history(
    job_dir: Path,
    plan: TrainingPlan,
    plan_sha256: str,
    completed_index: int,
    execution_bytes: bytes,
    expected_outputs: list[dict[str, object]],
) -> None:
    if completed_index < 1:
        raise FineTuneConfigError("completed attempt index must be positive")
    attempts = job_dir / "attempts"
    if attempts.is_symlink() or not attempts.is_dir():
        raise FineTuneConfigError("completed attempt history is missing or unsafe")
    expected_names = [f"{index:04d}" for index in range(1, completed_index + 1)]
    actual = sorted(attempts.iterdir())
    if [path.name for path in actual] != expected_names:
        raise FineTuneConfigError("completed attempt history is non-contiguous or has extras")
    for index, attempt in enumerate(actual, start=1):
        if attempt.is_symlink() or not attempt.is_dir():
            raise FineTuneConfigError("completed attempt history contains an unsafe directory")
        children = {child.name for child in attempt.iterdir()}
        if children != {"receipt.json", "stderr.log", "stdout.log"}:
            raise FineTuneConfigError("completed attempt history contains a partial attempt")
        for name in children:
            child = attempt / name
            if child.is_symlink() or not child.is_file():
                raise FineTuneConfigError("completed attempt history contains an unsafe artifact")
        receipt_path = attempt / "receipt.json"
        try:
            receipt_bytes = receipt_path.read_bytes()
            receipt = json.loads(
                receipt_bytes.decode("utf-8"), parse_constant=_reject_json_constant
            )
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise FineTuneConfigError("completed attempt receipt is corrupt") from error
        is_completed = index == completed_index
        _validate_receipt(
            receipt,
            job_dir=job_dir,
            plan=plan,
            plan_sha256=plan_sha256,
            attempt_index=index,
            completed=is_completed,
            expected_outputs=expected_outputs if is_completed else None,
        )
        if is_completed and receipt_bytes != execution_bytes:
            raise FineTuneConfigError("completion and attempt receipts do not match")


def _receipt(
    job_dir: Path,
    plan: TrainingPlan,
    plan_sha256: str,
    attempt_index: int,
    exit_code: int,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema_version": 1,
        "receipt_version": _RECEIPT_VERSION,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "job_id": plan.job_id,
        "target": plan.target,
        "plan_sha256": plan_sha256,
        "attempt_index": attempt_index,
        "exit_code": exit_code,
        "status": "failed" if exit_code else "completed",
        "stdout": _log_binding(job_dir, stdout_path),
        "stderr": _log_binding(job_dir, stderr_path),
    }
    receipt["receipt_sha256"] = _digest(receipt)
    return receipt


def _log_binding(job_dir: Path, path: Path) -> dict[str, object]:
    relative = path.relative_to(job_dir).as_posix()
    return {"path": relative, "size": path.stat().st_size, "sha256": _file_hash(path)}


def _validate_receipt(
    value: object,
    *,
    job_dir: Path,
    plan: TrainingPlan,
    plan_sha256: str,
    attempt_index: int,
    completed: bool,
    expected_outputs: list[dict[str, object]] | None = None,
) -> None:
    if not isinstance(value, dict):
        raise FineTuneConfigError("attempt receipt must be an object")
    receipt = cast(dict[str, object], value)
    expected_fields = _COMPLETED_RECEIPT_FIELDS if completed else _FAILED_RECEIPT_FIELDS
    if set(receipt) != expected_fields:
        raise FineTuneConfigError("attempt receipt has an invalid exact schema")
    expected_status = "completed" if completed else "failed"
    exit_code = receipt.get("exit_code")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("receipt_version") != _RECEIPT_VERSION
        or receipt.get("job_id") != plan.job_id
        or receipt.get("target") != plan.target
        or receipt.get("plan_sha256") != plan_sha256
        or receipt.get("attempt_index") != attempt_index
        or receipt.get("status") != expected_status
        or isinstance(exit_code, bool)
        or not isinstance(exit_code, int)
        or (exit_code != 0 if completed else exit_code == 0)
    ):
        raise FineTuneConfigError("attempt receipt does not match its job and status")
    timestamp = receipt.get("recorded_at")
    if not isinstance(timestamp, str):
        raise FineTuneConfigError("attempt receipt timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as error:
        raise FineTuneConfigError("attempt receipt timestamp is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise FineTuneConfigError("attempt receipt timestamp must be UTC")
    for stream_name in ("stdout", "stderr"):
        _validate_log_binding(
            receipt.get(stream_name),
            job_dir=job_dir,
            expected=Path("attempts") / f"{attempt_index:04d}" / f"{stream_name}.log",
        )
    receipt_digest = receipt.get("receipt_sha256")
    unsigned = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    if receipt_digest != _digest(unsigned):
        raise FineTuneConfigError("attempt receipt digest is invalid")
    if completed:
        _validate_completed_outputs(receipt.get("outputs"), expected_outputs)


def _validate_log_binding(value: object, *, job_dir: Path, expected: Path) -> None:
    if not isinstance(value, dict) or set(value) != {"path", "size", "sha256"}:
        raise FineTuneConfigError("attempt log binding has an invalid schema")
    path_value = value.get("path")
    size = value.get("size")
    digest = value.get("sha256")
    if path_value != expected.as_posix():
        raise FineTuneConfigError("attempt log path is not canonical and job-owned")
    path = job_dir / expected
    if path.is_symlink() or not path.is_file():
        raise FineTuneConfigError("attempt log is missing, foreign, or symlinked")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise FineTuneConfigError("attempt log size is invalid")
    if size != path.stat().st_size or digest != _file_hash(path):
        raise FineTuneConfigError("attempt log bytes do not match their receipt")


def _validate_completed_outputs(
    value: object, expected_outputs: list[dict[str, object]] | None
) -> None:
    if not isinstance(value, list) or not value:
        raise FineTuneConfigError("completion receipt outputs must be a nonempty list")
    paths: set[Path] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"path", "fingerprint"}:
            raise FineTuneConfigError("completion output binding has an invalid schema")
        path_value = item.get("path")
        if not isinstance(path_value, str) or not Path(path_value).is_absolute():
            raise FineTuneConfigError("completion output path must be absolute")
        path = Path(path_value)
        if path in paths or _fingerprint(path) != item.get("fingerprint"):
            raise FineTuneConfigError("completion output bytes do not match their receipt")
        paths.add(path)
    if expected_outputs is None or value != expected_outputs:
        raise FineTuneConfigError("completion outputs do not exactly match the recorded plan")


def _write_exclusive_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise FineTuneConfigError(f"refusing to overwrite existing artifact {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _reject_symlink_chain(path: Path) -> None:
    current = path.absolute()
    while not current.exists() and current != current.parent:
        current = current.parent
    while current != current.parent:
        if current.is_symlink():
            raise FineTuneConfigError(f"path contains a symlinked ancestor: {current}")
        current = current.parent


def _path(settings: FineTuneSettings, key: str) -> Path:
    value = settings.values.get(key)
    if not isinstance(value, Path):
        raise FineTuneConfigError(f"{key} must be a path")
    return value


def _string(settings: FineTuneSettings, key: str) -> str:
    value = settings.values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise FineTuneConfigError(f"{key} must be a nonblank string")
    return value


def _integer(settings: FineTuneSettings, key: str) -> int:
    value = settings.values.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FineTuneConfigError(f"{key} must be a positive integer")
    return value


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _valid_hash(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.absolute()
    right = right.absolute()
    return left == right or left in right.parents or right in left.parents

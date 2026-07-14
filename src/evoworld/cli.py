"""Command-line composition for validation, execution, and durable resume."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import importlib
import json
from pathlib import Path
import re
import sys
from typing import NoReturn, cast
from uuid import uuid4

from evoworld.adapters.deterministic import build_deterministic_adapters
from evoworld.codec import canonical_json, decode_json_object
from evoworld.config import (
    ConfigError,
    EvoWorldConfig,
    config_digest,
    config_from_resolved_manifest,
    load_config,
    resolved_manifest,
)
from evoworld.contracts import JsonObject
from evoworld.pipeline import EvoWorldPipeline, PipelineAdapters
from evoworld.storage import atomic_write_json
from evoworld.training import (
    TARGETS,
    FineTuneConfigError,
    FineTuneRuntimeError,
    execute_finetune,
    plan_finetune,
)

AdapterFactory = Callable[[EvoWorldConfig, Path], PipelineAdapters]

_FACTORY_ID = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$"
)


class CompositionError(ValueError):
    """Raised when the selected execution mode has no safe adapter composition."""


class FactoryError(CompositionError):
    """An explicitly authorized importable factory cannot be loaded safely."""


class _CliUsageError(ValueError):
    """Raised instead of terminating the embedding Python process."""


class _ConfigLoadError(ConfigError):
    """Raised when config or manifest bytes cannot be accessed or decoded."""


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _CliUsageError(message)


def build_parser() -> argparse.ArgumentParser:
    """Build the stable public argument parser without touching any backend."""
    parser = _ArgumentParser(prog="evoworld")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate a resolved configuration")
    _add_config_arguments(validate)

    run = commands.add_parser("run", help="start a new configured run")
    _add_config_arguments(run)
    run.add_argument("--rounds", type=_positive_integer, default=1)
    run.add_argument("--factory", metavar="MODULE:CALLABLE")

    resume = commands.add_parser("resume", help="resume the latest durable round")
    resume.add_argument("run_dir_path", nargs="?", metavar="RUN_DIR")
    resume.add_argument("--run-dir", dest="run_dir_option", metavar="RUN_DIR")
    resume.add_argument("--factory", metavar="MODULE:CALLABLE")

    finetune = commands.add_parser("finetune", help="plan or execute a fine-tuning job")
    finetune_commands = finetune.add_subparsers(dest="finetune_command", required=True)
    finetune_plan = finetune_commands.add_parser("plan", help="write an offline training plan")
    finetune_plan.add_argument("--config", required=True, metavar="CONFIG")
    finetune_plan.add_argument("--target", required=True, choices=TARGETS)
    finetune_execute = finetune_commands.add_parser(
        "execute", help="execute one immutable training job"
    )
    finetune_execute.add_argument("--job", required=True, metavar="JOB")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a CLI command and return its documented process status code."""
    try:
        arguments = build_parser().parse_args(argv)
        command = cast(str, arguments.command)
        if command == "validate":
            return _validate_command(_config_argument(arguments))
        if command == "run":
            return _run_command(
                _config_argument(arguments),
                cast(int, arguments.rounds),
                cast(str | None, arguments.factory),
            )
        if command == "resume":
            return _resume_command(
                _run_dir_argument(arguments), cast(str | None, arguments.factory)
            )
        if command == "finetune":
            finetune_command = cast(str, arguments.finetune_command)
            if finetune_command == "plan":
                _write_json(
                    plan_finetune(Path(cast(str, arguments.config)), cast(str, arguments.target))
                )
                return 0
            if finetune_command == "execute":
                _write_json(execute_finetune(Path(cast(str, arguments.job))))
                return 0
            raise _CliUsageError("unsupported fine-tuning command")
        raise _CliUsageError("unsupported command")
    except _CliUsageError:
        _write_error("UsageError", "invalid command arguments")
        return 2
    except FactoryError as error:
        _write_error("FactoryError", str(error))
        return 2
    except CompositionError as error:
        _write_error("CompositionError", str(error))
        return 2
    except _ConfigLoadError:
        _write_error("ConfigError", "configuration could not be loaded")
        return 2
    except ConfigError:
        _write_error("ConfigError", "configuration validation failed")
        return 2
    except FineTuneConfigError as error:
        _write_error("FineTuneConfigError", str(error))
        return 2
    except FineTuneRuntimeError as error:
        _write_error("FineTuneRuntimeError", str(error))
        return 1
    except Exception:
        _write_error("RuntimeError", "pipeline execution failed")
        return 1


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("config_path", nargs="?", metavar="CONFIG")
    parser.add_argument("--config", dest="config_option", metavar="CONFIG")


def _config_argument(arguments: argparse.Namespace) -> Path:
    positional = cast(str | None, arguments.config_path)
    option = cast(str | None, arguments.config_option)
    if (positional is None) == (option is None):
        raise _CliUsageError("provide exactly one configuration path")
    return Path(option or cast(str, positional))


def _run_dir_argument(arguments: argparse.Namespace) -> Path:
    positional = cast(str | None, arguments.run_dir_path)
    option = cast(str | None, arguments.run_dir_option)
    if (positional is None) == (option is None):
        raise _CliUsageError("provide exactly one run directory")
    return Path(option or cast(str, positional))


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _validate_command(config_path: Path) -> int:
    config = _load_config(config_path)
    manifest = resolved_manifest(config)
    digest = config_digest(config)
    _write_json(
        {
            "config_digest": digest,
            "expert_ids": [expert.expert_id for expert in config.experts],
            "mode": manifest["mode"],
            "schema_version": config.schema_version,
            "status": "valid",
        }
    )
    return 0


def _run_command(config_path: Path, rounds: int, factory_identity: str | None) -> int:
    config = _load_config(config_path)
    run_dir = config.run_root / f"run-{uuid4().hex}"
    adapters = _compose_adapters(config, run_dir, factory_identity)
    if config.mode == "real":
        assert factory_identity is not None
        _record_factory(
            adapters.store.run_dir,
            factory_identity,
            config,
            _adapter_runtime_digest(adapters),
        )
    results = EvoWorldPipeline(config, adapters).run(rounds)
    _write_json(
        {
            "completed_rounds": len(results),
            "run_dir": str(adapters.store.run_dir.resolve()),
            "status": "complete",
        }
    )
    return 0


def _resume_command(run_dir: Path, factory_identity: str | None) -> int:
    resolved_run_dir, config, recorded_factory, recorded_runtime_digest = _config_from_run(run_dir)
    if factory_identity is not None and recorded_factory is not None:
        if factory_identity != recorded_factory:
            raise FactoryError("factory does not match the durable run manifest")
    selected_factory = factory_identity or recorded_factory
    adapters = _compose_adapters(config, resolved_run_dir, selected_factory)
    if config.mode == "real":
        if recorded_factory is None or recorded_runtime_digest is None:
            raise FactoryError("real run manifest is missing durable composition identity")
        if _adapter_runtime_digest(adapters) != recorded_runtime_digest:
            raise FactoryError("runtime manifest or static asset identity changed")
        (
            fresh_run_dir,
            fresh_config,
            fresh_factory,
            fresh_runtime_digest,
        ) = _config_from_run(resolved_run_dir)
        if (
            fresh_run_dir != resolved_run_dir
            or fresh_config != config
            or fresh_factory != recorded_factory
            or fresh_runtime_digest != recorded_runtime_digest
        ):
            raise FactoryError("factory changed the durable composition identity")
    result = EvoWorldPipeline(config, adapters).resume()
    _write_json(
        {
            "completed_rounds": int(result is not None),
            "resumed_round_id": result.round_id if result is not None else None,
            "run_dir": str(adapters.store.run_dir.resolve()),
            "status": "complete",
        }
    )
    return 0


def _compose_adapters(
    config: EvoWorldConfig, run_dir: Path, factory_identity: str | None
) -> PipelineAdapters:
    if config.mode == "dry_run":
        if factory_identity is not None:
            raise CompositionError("factory is not allowed for dry_run mode")
        return build_deterministic_adapters(config, run_dir)
    if factory_identity is None:
        raise CompositionError("real mode requires an explicit --factory")
    factory = _load_factory(factory_identity)
    adapters = factory(config, run_dir)
    try:
        actual_run_dir = adapters.store.run_dir.resolve(strict=True)
    except (AttributeError, OSError) as error:
        raise FactoryError("factory store has no durable run directory") from error
    if actual_run_dir != run_dir.resolve():
        raise FactoryError("factory store is bound to a different run directory")
    return adapters


def _load_factory(identity: str) -> AdapterFactory:
    if _FACTORY_ID.fullmatch(identity) is None:
        raise FactoryError("factory must use strict module:callable syntax")
    module_name, callable_name = identity.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        raise FactoryError("factory module could not be imported") from error
    candidate = getattr(module, callable_name, None)
    if candidate is None or not callable(candidate):
        raise FactoryError("factory callable is missing or not callable")
    return cast(AdapterFactory, candidate)


def _adapter_runtime_digest(adapters: PipelineAdapters) -> str:
    digest = getattr(adapters, "runtime_manifest_digest", None)
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise FactoryError("real factory did not expose a valid runtime manifest digest")
    return digest


def _record_factory(
    run_dir: Path,
    identity: str,
    config: EvoWorldConfig,
    runtime_manifest_digest: str,
) -> None:
    resolved = run_dir.resolve(strict=True)
    path = resolved / "manifest.json"
    try:
        wrapper = decode_json_object(path.read_text(encoding="utf-8"), source="run manifest")
    except (OSError, UnicodeError, ValueError) as error:
        raise ConfigError("run manifest cannot record factory identity") from error
    if wrapper.get("config_digest") != config_digest(config) or canonical_json(
        wrapper.get("resolved_config")
    ) != canonical_json(resolved_manifest(config)):
        raise ConfigError("factory store manifest does not match requested real config")
    composition: JsonObject = {
        "factory": identity,
        "runtime_manifest_digest": runtime_manifest_digest,
    }
    wrapper["composition"] = composition
    atomic_write_json(path, wrapper)


def _config_from_run(
    run_dir: Path,
) -> tuple[Path, EvoWorldConfig, str | None, str | None]:
    try:
        resolved = run_dir.resolve(strict=True)
        if not resolved.is_dir():
            raise NotADirectoryError(resolved)
        wrapper = decode_json_object(
            (resolved / "manifest.json").read_text(encoding="utf-8"),
            source="run manifest",
        )
    except OSError as error:
        raise _ConfigLoadError("run manifest could not be loaded") from error
    except (UnicodeError, ValueError) as error:
        raise ConfigError("run manifest is not valid JSON") from error
    if set(wrapper) not in (
        {"config_digest", "resolved_config"},
        {"config_digest", "resolved_config", "composition"},
    ):
        raise ConfigError("run manifest fields do not match schema version 1")
    manifest = wrapper.get("resolved_config")
    digest = wrapper.get("config_digest")
    if not isinstance(manifest, dict) or not isinstance(digest, str):
        raise ConfigError("run manifest contains invalid configuration data")
    config = config_from_resolved_manifest(manifest)
    if config_digest(config) != digest:
        raise ConfigError("run manifest configuration digest mismatch")
    composition = wrapper.get("composition")
    factory_identity: str | None = None
    runtime_manifest_digest: str | None = None
    if composition is not None:
        if not isinstance(composition, dict) or set(composition) != {
            "factory",
            "runtime_manifest_digest",
        }:
            raise ConfigError("run manifest composition is invalid")
        factory = composition.get("factory")
        if not isinstance(factory, str) or _FACTORY_ID.fullmatch(factory) is None:
            raise ConfigError("run manifest factory identity is invalid")
        factory_identity = factory
        runtime_digest = composition.get("runtime_manifest_digest")
        if (
            not isinstance(runtime_digest, str)
            or len(runtime_digest) != 64
            or any(character not in "0123456789abcdef" for character in runtime_digest)
        ):
            raise ConfigError("run manifest runtime identity is invalid")
        runtime_manifest_digest = runtime_digest
    return resolved, config, factory_identity, runtime_manifest_digest


def _load_config(config_path: Path) -> EvoWorldConfig:
    try:
        return load_config(config_path)
    except ConfigError:
        raise
    except (OSError, UnicodeError) as error:
        raise _ConfigLoadError("configuration could not be loaded") from error


def _write_json(payload: JsonObject) -> None:
    sys.stdout.write(
        json.dumps(
            payload, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
        + "\n"
    )


def _write_error(error_type: str, message: str) -> None:
    payload: JsonObject = {
        "error": {"message": message, "type": error_type},
        "status": "error",
    }
    sys.stderr.write(
        json.dumps(
            payload, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
        + "\n"
    )

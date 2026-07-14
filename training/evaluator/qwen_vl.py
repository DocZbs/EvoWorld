"""Audited public Qwen3-VL evaluator fine-tuning plan for ms-swift 4.3.1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import tomllib
from typing import Callable, Mapping, cast

from training.common.schema import TrainingPlan, TrainingSchemaError, validate_relative_posix_path
from training.evaluator.schema import EvaluatorExport


MS_SWIFT_REVISION = "332bb8fb1071ee692235a502f4ad8b0473b4a432"
_MS_SWIFT_VERSION = "4.3.1"
MS_SWIFT_REQUIREMENT = f"ms-swift=={_MS_SWIFT_VERSION}"
QWEN_VL_UTILS_REQUIREMENT = "qwen-vl-utils==0.0.14"
DECORD_REQUIREMENT = "decord==0.6.0"
TRANSFORMERS_REQUIREMENT = "transformers>=4.57,<5.13.0"
TORCH_REQUIREMENT = "torch>=2.0"
TORCHVISION_REQUIREMENT = "torchvision>=0.15"
_BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
_FRAME_FACTOR = 2
_COMPLETION_MARKER = ".evoworld-complete.json"
_RUNTIME_PROBE_SCRIPT = """\
import json
from importlib.metadata import version
import decord
import qwen_vl_utils
import swift
import torch
import torchvision
import transformers
print(json.dumps({name: version(name) for name in (
    'ms-swift', 'qwen-vl-utils', 'decord', 'transformers', 'torch', 'torchvision'
)} | {'cuda_available': str(torch.cuda.is_available()).lower()}))
"""
_UVX_RUNTIME_PREFIX = (
    "uvx",
    "--from",
    MS_SWIFT_REQUIREMENT,
    "--with",
    QWEN_VL_UTILS_REQUIREMENT,
    "--with",
    DECORD_REQUIREMENT,
    "--with",
    TRANSFORMERS_REQUIREMENT,
    "--with",
    TORCH_REQUIREMENT,
    "--with",
    TORCHVISION_REQUIREMENT,
    "swift",
    "sft",
)
_TARGET = "qwen_evaluator"
_CONFIG_FIELDS = frozenset(
    {
        "ms_swift_version",
        "qwen_vl_utils_version",
        "decord_version",
        "transformers_requirement",
        "torch_requirement",
        "torchvision_requirement",
        "torch_platform",
        "base_model",
        "tuner_type",
        "learning_rate",
        "lora_rank",
        "lora_alpha",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_length",
        "seed",
        "output_dir",
        "max_pixels",
        "video_max_token_num",
        "video_max_pixels",
        "fps",
        "fps_max_frames",
        "torch_dtype",
        "device_map",
        "save_strategy",
        "save_total_limit",
        "logging_steps",
        "dataset_export_sha256",
        "dataset_revision",
    }
)


class EvaluatorConfigError(ValueError):
    """Raised when the public baseline cannot produce a safe, complete plan."""


def _hash(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise EvaluatorConfigError(f"{name} must be a lowercase 64-character SHA-256 value")


def _string(values: dict[str, object], name: str) -> str:
    value = values[name]
    if not isinstance(value, str) or not value.strip():
        raise EvaluatorConfigError(f"{name} must be a nonblank string")
    if "\0" in value or "\n" in value or "\r" in value:
        raise EvaluatorConfigError(f"{name} contains unsafe control characters")
    return value


def _integer(values: dict[str, object], name: str) -> int:
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluatorConfigError(f"{name} must be an integer")
    return value


def _number(values: dict[str, object], name: str) -> float:
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluatorConfigError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvaluatorConfigError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class QwenTrainConfig:
    """Every explicit knob in the configurable public baseline."""

    ms_swift_version: str
    qwen_vl_utils_version: str
    decord_version: str
    transformers_requirement: str
    torch_requirement: str
    torchvision_requirement: str
    torch_platform: str
    base_model: str
    tuner_type: str
    learning_rate: float
    lora_rank: int
    lora_alpha: int
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_length: int
    seed: int
    output_dir: Path
    max_pixels: int
    video_max_token_num: int
    video_max_pixels: int
    fps: float
    fps_max_frames: int
    torch_dtype: str
    device_map: str
    save_strategy: str
    save_total_limit: int
    logging_steps: int
    dataset_export_sha256: str
    dataset_revision: str

    def __post_init__(self) -> None:
        if self.ms_swift_version != _MS_SWIFT_VERSION:
            raise EvaluatorConfigError("ms_swift_version must pin ms-swift 4.3.1")
        if self.qwen_vl_utils_version != "0.0.14":
            raise EvaluatorConfigError("qwen_vl_utils_version must pin qwen-vl-utils 0.0.14")
        if self.decord_version != "0.6.0":
            raise EvaluatorConfigError("decord_version must pin decord 0.6.0")
        if self.transformers_requirement != ">=4.57,<5.13.0":
            raise EvaluatorConfigError("transformers_requirement must be >=4.57,<5.13.0")
        if self.torch_requirement != ">=2.0":
            raise EvaluatorConfigError("torch_requirement must be >=2.0")
        if self.torchvision_requirement != ">=0.15":
            raise EvaluatorConfigError("torchvision_requirement must be >=0.15")
        if self.torch_platform != "cuda":
            raise EvaluatorConfigError("torch_platform must select the CUDA training runtime")
        if self.base_model != _BASE_MODEL:
            raise EvaluatorConfigError("base_model must be the Qwen3-VL 4B Instruct checkpoint")
        if self.tuner_type != "lora":
            raise EvaluatorConfigError("tuner_type must select LoRA")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(self.learning_rate)
            or not 0 < self.learning_rate <= 1
        ):
            raise EvaluatorConfigError("learning_rate must be finite and in (0, 1]")
        positive_integers = {
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_length": self.max_length,
            "max_pixels": self.max_pixels,
            "video_max_token_num": self.video_max_token_num,
            "video_max_pixels": self.video_max_pixels,
            "fps_max_frames": self.fps_max_frames,
            "save_total_limit": self.save_total_limit,
            "logging_steps": self.logging_steps,
        }
        for name, value in positive_integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise EvaluatorConfigError(f"{name} must be a positive integer")
        if (
            isinstance(self.num_train_epochs, bool)
            or not isinstance(self.num_train_epochs, int)
            or self.num_train_epochs < 1
        ):
            raise EvaluatorConfigError("num_train_epochs must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise EvaluatorConfigError("seed must be a nonnegative integer")
        if (
            isinstance(self.fps, bool)
            or not isinstance(self.fps, (int, float))
            or not math.isfinite(self.fps)
            or not 0 < self.fps <= 240
        ):
            raise EvaluatorConfigError("fps must be finite and in (0, 240]")
        if self.video_max_pixels != self.video_max_token_num * 32 * 32:
            raise EvaluatorConfigError(
                "Qwen3-VL video token and pixel budgets must be consistent at 32x32 pixels/token"
            )
        if self.fps_max_frames < _FRAME_FACTOR or self.fps_max_frames % _FRAME_FACTOR:
            raise EvaluatorConfigError("fps_max_frames must be an even multiple of FRAME_FACTOR=2")
        try:
            validate_relative_posix_path(self.output_dir, "output_dir")
        except TrainingSchemaError as error:
            raise EvaluatorConfigError(str(error)) from error
        if self.torch_dtype not in {"bfloat16", "float16", "float32"}:
            raise EvaluatorConfigError("torch_dtype must be bfloat16, float16, or float32")
        if self.device_map not in {"auto", "cuda", "cpu"}:
            raise EvaluatorConfigError("device_map must be auto, cuda, or cpu")
        if self.save_strategy != "epoch":
            raise EvaluatorConfigError(
                "save_strategy must be epoch to guarantee a final checkpoint"
            )
        _hash(self.dataset_export_sha256, "dataset_export_sha256")
        _hash(self.dataset_revision, "dataset_revision")


def load_qwen_config(path: Path) -> QwenTrainConfig:
    """Load the strict public-baseline table without hidden recipe defaults."""
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
        raw = document["qwen3_vl_4b"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as error:
        raise EvaluatorConfigError(f"unable to load evaluator configuration from {path}") from error
    if set(document) != {"qwen3_vl_4b"} or not isinstance(raw, dict):
        raise EvaluatorConfigError("config must contain only the [qwen3_vl_4b] table")
    values = cast(dict[str, object], raw)
    missing = _CONFIG_FIELDS - set(values)
    unknown = set(values) - _CONFIG_FIELDS
    if missing:
        raise EvaluatorConfigError(f"missing evaluator config fields: {sorted(missing)}")
    if unknown:
        raise EvaluatorConfigError(f"unknown evaluator config fields: {sorted(unknown)}")
    return QwenTrainConfig(
        ms_swift_version=_string(values, "ms_swift_version"),
        qwen_vl_utils_version=_string(values, "qwen_vl_utils_version"),
        decord_version=_string(values, "decord_version"),
        transformers_requirement=_string(values, "transformers_requirement"),
        torch_requirement=_string(values, "torch_requirement"),
        torchvision_requirement=_string(values, "torchvision_requirement"),
        torch_platform=_string(values, "torch_platform"),
        base_model=_string(values, "base_model"),
        tuner_type=_string(values, "tuner_type"),
        learning_rate=_number(values, "learning_rate"),
        lora_rank=_integer(values, "lora_rank"),
        lora_alpha=_integer(values, "lora_alpha"),
        num_train_epochs=_integer(values, "num_train_epochs"),
        per_device_train_batch_size=_integer(values, "per_device_train_batch_size"),
        gradient_accumulation_steps=_integer(values, "gradient_accumulation_steps"),
        max_length=_integer(values, "max_length"),
        seed=_integer(values, "seed"),
        output_dir=Path(_string(values, "output_dir")),
        max_pixels=_integer(values, "max_pixels"),
        video_max_token_num=_integer(values, "video_max_token_num"),
        video_max_pixels=_integer(values, "video_max_pixels"),
        fps=_number(values, "fps"),
        fps_max_frames=_integer(values, "fps_max_frames"),
        torch_dtype=_string(values, "torch_dtype"),
        device_map=_string(values, "device_map"),
        save_strategy=_string(values, "save_strategy"),
        save_total_limit=_integer(values, "save_total_limit"),
        logging_steps=_integer(values, "logging_steps"),
        dataset_export_sha256=_string(values, "dataset_export_sha256"),
        dataset_revision=_string(values, "dataset_revision"),
    )


def _version_tuple(value: str, package: str) -> tuple[int, ...]:
    public = value.partition("+")[0]
    try:
        result = tuple(int(part) for part in public.split("."))
    except ValueError as error:
        raise EvaluatorConfigError(
            f"{package} has an invalid resolved version {value!r}"
        ) from error
    if not result:
        raise EvaluatorConfigError(f"{package} has an invalid resolved version {value!r}")
    return result


def validate_qwen_runtime_versions(installed_versions: Mapping[str, str]) -> None:
    """Validate the uvx-resolved tools before the ``swift sft`` command is accepted."""
    required = {"ms-swift", "qwen-vl-utils", "decord", "transformers", "torch", "torchvision"}
    missing = required - set(installed_versions)
    if missing:
        raise EvaluatorConfigError(f"resolved Qwen runtime is missing {sorted(missing)}")
    exact = {
        "ms-swift": "4.3.1",
        "qwen-vl-utils": "0.0.14",
        "decord": "0.6.0",
    }
    for package, expected in exact.items():
        if installed_versions[package] != expected:
            raise EvaluatorConfigError(f"{package} must resolve to {expected}")
    transformers = _version_tuple(installed_versions["transformers"], "transformers")
    if transformers < (4, 57) or transformers >= (5, 13):
        raise EvaluatorConfigError("transformers must resolve in [4.57, 5.13)")
    if _version_tuple(installed_versions["torch"], "torch") < (2, 0):
        raise EvaluatorConfigError("torch must resolve to version 2.0 or newer")
    if _version_tuple(installed_versions["torchvision"], "torchvision") < (0, 15):
        raise EvaluatorConfigError("torchvision must resolve to version 0.15 or newer")
    if installed_versions.get("cuda_available") != "true":
        raise EvaluatorConfigError("Qwen training requires a CUDA-capable torch runtime")


def _export_revision(export: EvaluatorExport) -> str:
    values: list[dict[str, object]] = []
    try:
        split_counts: dict[str, int] = {}
        for path, expected_split in ((export.path, "train"), (export.validation_path, "eval")):
            split_counts[expected_split] = 0
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError("export row must be an object")
                provenance = row["evoworld_provenance"]
                if not isinstance(provenance, dict):
                    raise TypeError("provenance must be an object")
                example_id = provenance["example_id"]
                content_hash = provenance["content_sha256"]
                revision = provenance["revision"]
                if provenance.get("split") != expected_split:
                    raise TypeError("export split provenance does not match its file")
                if not isinstance(example_id, str) or not example_id.strip():
                    raise TypeError("example_id must be a nonblank string")
                if not isinstance(content_hash, str):
                    raise TypeError("content_sha256 must be a string")
                if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
                    raise TypeError("revision must be a positive integer")
                _hash(content_hash, "export content_sha256")
                values.append(
                    {
                        "content_sha256": content_hash,
                        "example_id": example_id,
                        "revision": revision,
                    }
                )
                split_counts[expected_split] += 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise EvaluatorConfigError("stale export has invalid revision provenance") from error
    if (
        len(values) != export.example_count
        or not all(split_counts.values())
        or len({cast(str, value["example_id"]) for value in values}) != len(values)
    ):
        raise EvaluatorConfigError("stale export has invalid revision provenance")
    encoded = json.dumps(
        sorted(values, key=lambda item: cast(str, item["example_id"])),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _current_export_hash(export: EvaluatorExport) -> str:
    try:
        train_hash = sha256(export.path.read_bytes()).hexdigest()
        validation_hash = sha256(export.validation_path.read_bytes()).hexdigest()
    except OSError as error:
        raise EvaluatorConfigError("stale export is missing") from error
    if train_hash != export.train_sha256 or validation_hash != export.validation_sha256:
        raise EvaluatorConfigError("stale export split hash does not match the planned dataset")
    encoded = json.dumps(
        {
            "dataset_revision": export.dataset_revision,
            "train_sha256": train_hash,
            "validation_sha256": validation_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _validate_execution_media(export: EvaluatorExport) -> None:
    root = export.dataset_root.resolve()
    row_count = 0
    try:
        for path, expected_split in (
            (export.path, "train"),
            (export.validation_path, "eval"),
        ):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                videos = row["videos"]
                provenance = row["evoworld_provenance"]
                if (
                    not isinstance(videos, list)
                    or len(videos) != 1
                    or not isinstance(videos[0], list)
                    or not isinstance(provenance, dict)
                    or provenance.get("split") != expected_split
                ):
                    raise TypeError("invalid exported media row")
                media_hashes = provenance["media_sha256"]
                if not isinstance(media_hashes, list) or len(videos[0]) != len(media_hashes):
                    raise TypeError("exported media paths and hashes differ in length")
                for raw_path, expected_hash in zip(videos[0], media_hashes, strict=True):
                    if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
                        raise TypeError("exported media path and hash must be strings")
                    relative_path = Path(raw_path)
                    validate_relative_posix_path(relative_path, "exported media path")
                    _hash(expected_hash, "exported media sha256")
                    candidate = root / relative_path
                    if candidate.is_symlink():
                        raise EvaluatorConfigError(
                            "dataset media must not be a symlink at execution"
                        )
                    resolved = candidate.resolve()
                    if not resolved.is_relative_to(root) or not resolved.is_file():
                        raise EvaluatorConfigError(
                            "dataset media path is missing or escapes dataset_root"
                        )
                    if sha256(resolved.read_bytes()).hexdigest() != expected_hash:
                        raise EvaluatorConfigError("dataset media hash changed after export")
                row_count += 1
    except EvaluatorConfigError:
        raise
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        TrainingSchemaError,
    ) as error:
        raise EvaluatorConfigError("exported dataset media binding is invalid") from error
    if row_count != export.example_count:
        raise EvaluatorConfigError("exported dataset media row count does not match the manifest")


def _validate_execution_inputs(export: EvaluatorExport, config: QwenTrainConfig) -> None:
    if _export_revision(export) != export.dataset_revision:
        raise EvaluatorConfigError("stale export annotation revision does not match its content")
    if config.dataset_revision != export.dataset_revision:
        raise EvaluatorConfigError("dataset annotation revision does not match the export")
    current_hash = _current_export_hash(export)
    if current_hash != export.sha256 or config.dataset_export_sha256 != export.sha256:
        raise EvaluatorConfigError("stale export hash does not match the planned dataset")
    _validate_export_frame_budget(export, config)
    _validate_execution_media(export)


def _validate_export_frame_budget(export: EvaluatorExport, config: QwenTrainConfig) -> None:
    """Match qwen-vl-utils 0.0.14's direct frame-list ``FRAME_FACTOR=2`` branch."""
    try:
        for path in (export.path, export.validation_path):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                videos = row["videos"]
                if (
                    not isinstance(videos, list)
                    or len(videos) != 1
                    or not isinstance(videos[0], list)
                    or not all(isinstance(frame, str) and frame for frame in videos[0])
                ):
                    raise TypeError("videos must contain exactly one nonempty frame list")
                frame_count = len(videos[0])
                if frame_count < _FRAME_FACTOR:
                    raise EvaluatorConfigError(
                        "each Qwen video frame list must contain at least 2 frames"
                    )
                if frame_count > config.fps_max_frames:
                    raise EvaluatorConfigError("Qwen video frame list exceeds fps_max_frames")
                padded_count = math.ceil(frame_count / _FRAME_FACTOR) * _FRAME_FACTOR
                if padded_count > config.fps_max_frames:
                    raise EvaluatorConfigError("Qwen FRAME_FACTOR=2 padding exceeds fps_max_frames")
    except EvaluatorConfigError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise EvaluatorConfigError("stale export has an invalid Qwen video frame list") from error


def _config_payload(config: QwenTrainConfig) -> dict[str, object]:
    values = cast(dict[str, object], asdict(config))
    values["output_dir"] = config.output_dir.as_posix()
    return values


def _validate_lexical_output_path(
    workspace_root: Path,
    candidate: Path,
    description: str,
) -> None:
    """Reject links and non-directories without following any output component."""
    try:
        relative = candidate.relative_to(workspace_root)
    except ValueError as error:
        raise EvaluatorConfigError(f"{description} escapes working_directory") from error
    current = workspace_root
    for component in relative.parts:
        current /= component
        try:
            status = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise EvaluatorConfigError(f"unable to inspect {description}") from error
        if stat.S_ISLNK(status.st_mode):
            raise EvaluatorConfigError(f"{description} contains a symlink")
        if not stat.S_ISDIR(status.st_mode):
            raise EvaluatorConfigError(f"{description} contains a non-directory component")


def _training_argv(
    config: QwenTrainConfig,
    export: EvaluatorExport,
    *,
    output_dir: str,
) -> tuple[str, ...]:
    model_kwargs = json.dumps(
        {
            "fps": config.fps,
            "fps_max_frames": config.fps_max_frames,
            "video_max_token_num": config.video_max_token_num,
            "video_max_pixels": config.video_max_pixels,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    # Source: https://github.com/modelscope/ms-swift/blob/v4.3.1/docs/source_en/Instruction/Command-line-parameters.md
    return (
        *_UVX_RUNTIME_PREFIX,
        "--model",
        config.base_model,
        "--tuner_type",
        config.tuner_type,
        "--dataset",
        str(export.path),
        "--val_dataset",
        str(export.validation_path),
        "--split_dataset_ratio",
        "0",
        "--strict",
        "true",
        "--learning_rate",
        str(config.learning_rate),
        "--lora_rank",
        str(config.lora_rank),
        "--lora_alpha",
        str(config.lora_alpha),
        "--num_train_epochs",
        str(config.num_train_epochs),
        "--per_device_train_batch_size",
        str(config.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(config.gradient_accumulation_steps),
        "--max_length",
        str(config.max_length),
        "--seed",
        str(config.seed),
        "--output_dir",
        output_dir,
        "--add_version",
        "false",
        "--max_pixels",
        str(config.max_pixels),
        "--model_kwargs",
        model_kwargs,
        "--torch_dtype",
        config.torch_dtype,
        "--device_map",
        config.device_map,
        "--save_strategy",
        config.save_strategy,
        "--save_total_limit",
        str(config.save_total_limit),
        "--logging_steps",
        str(config.logging_steps),
    )


def _read_json_object(path: Path, description: str) -> dict[str, object]:
    try:
        if path.is_symlink():
            raise EvaluatorConfigError(f"completed Qwen output has symlinked {description}")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluatorConfigError(f"completed Qwen output has invalid {description}") from error
    if not isinstance(value, dict):
        raise EvaluatorConfigError(f"completed Qwen output has invalid {description}")
    return cast(dict[str, object], value)


def _expected_saved_args(
    config: QwenTrainConfig, export: EvaluatorExport, output_dir: Path
) -> dict[str, object]:
    return {
        "swift_version": _MS_SWIFT_VERSION,
        "model": config.base_model,
        "tuner_type": config.tuner_type,
        "dataset": [str(export.path)],
        "val_dataset": [str(export.validation_path)],
        "split_dataset_ratio": 0.0,
        "strict": True,
        "learning_rate": config.learning_rate,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_length": config.max_length,
        "seed": config.seed,
        "output_dir": str(output_dir),
        "max_pixels": config.max_pixels,
        "model_kwargs": {
            "fps": config.fps,
            "fps_max_frames": config.fps_max_frames,
            "video_max_token_num": config.video_max_token_num,
            "video_max_pixels": config.video_max_pixels,
        },
        "torch_dtype": config.torch_dtype,
        "device_map": config.device_map,
        "save_strategy": config.save_strategy,
        "save_total_limit": config.save_total_limit,
        "logging_steps": config.logging_steps,
        "add_version": False,
    }


def _validate_saved_recipe(
    args: dict[str, object], config: QwenTrainConfig, export: EvaluatorExport, output_dir: Path
) -> None:
    for key, expected in _expected_saved_args(config, export, output_dir).items():
        if args.get(key) != expected:
            raise EvaluatorConfigError(f"completed Qwen output recipe mismatch for {key}")


def _validate_ms_swift_output(
    plan: TrainingPlan,
    export: EvaluatorExport,
    config: QwenTrainConfig,
    *,
    workspace_root: Path,
) -> Path:
    if not workspace_root.is_absolute() or not workspace_root.is_dir():
        raise EvaluatorConfigError("workspace_root must be an existing absolute directory")
    if len(plan.expected_outputs) != 1:
        raise EvaluatorConfigError("Qwen plan must declare exactly one output root")
    try:
        output_flag = plan.argv.index("--output_dir")
        output_dir = Path(plan.argv[output_flag + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise EvaluatorConfigError("Qwen plan is missing its output directory") from error
    expected_dir = (workspace_root / plan.expected_outputs[0]).resolve()
    if output_dir != expected_dir or not output_dir.is_dir():
        raise EvaluatorConfigError(
            "completed Qwen output does not match the planned output directory"
        )

    root_args_path = output_dir / "args.json"
    root_args = _read_json_object(root_args_path, "args.json")
    _validate_saved_recipe(root_args, config, export, output_dir)

    try:
        log_rows = [
            json.loads(line)
            for line in (output_dir / "logging.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluatorConfigError("completed Qwen output has invalid logging.jsonl") from error
    final_rows = [
        row for row in log_rows if isinstance(row, dict) and "last_model_checkpoint" in row
    ]
    if not final_rows:
        raise EvaluatorConfigError("completed Qwen output lacks the final checkpoint log")
    final = cast(dict[str, object], final_rows[-1])
    global_step = final.get("global_step")
    checkpoint_value = final.get("last_model_checkpoint")
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 1:
        raise EvaluatorConfigError("completed Qwen output has an invalid final global_step")
    if not isinstance(checkpoint_value, str):
        raise EvaluatorConfigError("completed Qwen output has an invalid final checkpoint path")
    checkpoint = Path(checkpoint_value).resolve()
    if checkpoint != output_dir / f"checkpoint-{global_step}" or not checkpoint.is_dir():
        raise EvaluatorConfigError("completed Qwen output final checkpoint is not from this job")

    checkpoint_args_path = checkpoint / "args.json"
    checkpoint_args = _read_json_object(checkpoint_args_path, "checkpoint args.json")
    if checkpoint_args != root_args:
        raise EvaluatorConfigError(
            "completed Qwen checkpoint recipe does not match its output root"
        )
    adapter_config = _read_json_object(checkpoint / "adapter_config.json", "adapter_config.json")
    if (
        adapter_config.get("r") != config.lora_rank
        or adapter_config.get("lora_alpha") != config.lora_alpha
    ):
        raise EvaluatorConfigError("completed Qwen adapter configuration does not match the recipe")
    for artifact in ("adapter_model.safetensors", "training_args.bin"):
        path = checkpoint / artifact
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size < 1:
                raise EvaluatorConfigError(f"completed Qwen adapter artifact {artifact} is empty")
        except OSError as error:
            raise EvaluatorConfigError(
                f"completed Qwen adapter artifact {artifact} is missing"
            ) from error
    return checkpoint


def _completion_payload(
    plan: TrainingPlan, export: EvaluatorExport, checkpoint: Path
) -> dict[str, object]:
    recipe_hash = plan.job_id.removeprefix("qwen-evaluator-")
    _hash(recipe_hash, "Qwen recipe hash")
    argv_hash = sha256(
        json.dumps(list(plan.argv), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    try:
        artifact_hashes = {
            name: sha256((checkpoint / name).read_bytes()).hexdigest()
            for name in (
                "adapter_config.json",
                "adapter_model.safetensors",
                "args.json",
                "training_args.bin",
            )
        }
    except OSError as error:
        raise EvaluatorConfigError("completed Qwen artifacts cannot be hashed") from error
    return {
        "artifact_sha256": artifact_hashes,
        "argv_sha256": argv_hash,
        "checkpoint": checkpoint.name,
        "dataset_export_sha256": export.sha256,
        "job_id": plan.job_id,
        "recipe_sha256": recipe_hash,
        "schema_version": 1,
        "upstream_revision": plan.upstream_revision,
    }


def _runtime_probe_argv() -> tuple[str, ...]:
    return (
        "uvx",
        "--with",
        MS_SWIFT_REQUIREMENT,
        "--with",
        QWEN_VL_UTILS_REQUIREMENT,
        "--with",
        DECORD_REQUIREMENT,
        "--with",
        TRANSFORMERS_REQUIREMENT,
        "--with",
        TORCH_REQUIREMENT,
        "--with",
        TORCHVISION_REQUIREMENT,
        "python",
        "-c",
        _RUNTIME_PROBE_SCRIPT,
    )


def _probe_runtime_versions(plan: TrainingPlan) -> Mapping[str, str]:
    environment = os.environ.copy()
    environment.update(plan.environment)
    process = subprocess.run(
        _runtime_probe_argv(),
        cwd=plan.cwd,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        value = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise EvaluatorConfigError("Qwen runtime probe returned invalid JSON") from error
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(version, str) for key, version in value.items()
    ):
        raise EvaluatorConfigError("Qwen runtime probe returned invalid versions")
    return cast(dict[str, str], value)


def _write_completion_marker(
    plan: TrainingPlan, export: EvaluatorExport, checkpoint: Path, output_dir: Path
) -> None:
    marker = output_dir / _COMPLETION_MARKER
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(_completion_payload(plan, export, checkpoint), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(marker)


def validate_qwen_training_output(
    plan: TrainingPlan,
    export: EvaluatorExport,
    config: QwenTrainConfig,
    *,
    workspace_root: Path,
) -> tuple[Path, ...]:
    """Prove successful wrapper completion plus matching official ms-swift artifacts."""
    _validate_execution_inputs(export, config)
    checkpoint = _validate_ms_swift_output(
        plan,
        export,
        config,
        workspace_root=workspace_root,
    )
    output_dir = checkpoint.parent
    marker = _read_json_object(output_dir / _COMPLETION_MARKER, "completion marker")
    if marker != _completion_payload(plan, export, checkpoint):
        raise EvaluatorConfigError("completed Qwen output marker does not match this job")
    return plan.expected_outputs


def run_qwen_training(
    plan: TrainingPlan,
    export: EvaluatorExport,
    config: QwenTrainConfig,
    *,
    workspace_root: Path,
    command_runner: Callable[[TrainingPlan], None] | None = None,
    version_probe: Callable[[tuple[str, ...]], Mapping[str, str]] | None = None,
) -> tuple[Path, ...]:
    """Execute the audited uvx plan once and validate its official ms-swift artifacts."""
    expected_plan = plan_qwen_training(
        export,
        config,
        working_directory=workspace_root,
    )
    if plan != expected_plan:
        raise EvaluatorConfigError("Qwen runtime plan does not match the audited recipe")
    output_dir = Path(plan.argv[plan.argv.index("--output_dir") + 1]).resolve()
    if output_dir.is_dir() and any(output_dir.iterdir()):
        return validate_qwen_training_output(
            plan,
            export,
            config,
            workspace_root=workspace_root,
        )
    if version_probe is None:
        installed_versions = _probe_runtime_versions(plan)
    else:
        installed_versions = version_probe(_runtime_probe_argv())
    validate_qwen_runtime_versions(installed_versions)
    _validate_execution_inputs(export, config)
    if command_runner is None:
        environment = os.environ.copy()
        environment.update(plan.environment)
        subprocess.run(plan.argv, cwd=plan.cwd, env=environment, check=True)
    else:
        command_runner(plan)
    _validate_execution_inputs(export, config)
    checkpoint = _validate_ms_swift_output(
        plan,
        export,
        config,
        workspace_root=workspace_root,
    )
    _write_completion_marker(plan, export, checkpoint, output_dir)
    return validate_qwen_training_output(
        plan,
        export,
        config,
        workspace_root=workspace_root,
    )


def plan_qwen_training(
    export: EvaluatorExport,
    config: QwenTrainConfig,
    *,
    working_directory: Path,
) -> TrainingPlan:
    """Create an idempotent ms-swift 4.3.1 argv bound to exported bytes and revisions."""
    if not working_directory.is_absolute():
        raise EvaluatorConfigError("working_directory must be absolute")
    workspace_root = working_directory.resolve()
    if not workspace_root.is_dir():
        raise EvaluatorConfigError("working_directory must be an existing directory")
    cwd = export.dataset_root.resolve()
    if not cwd.is_dir():
        raise EvaluatorConfigError("export dataset_root must be an existing directory")
    _validate_execution_inputs(export, config)
    output_root = workspace_root / config.output_dir
    _validate_lexical_output_path(workspace_root, output_root, "output_dir")
    environment: dict[str, str] = {}
    argv_template = _training_argv(
        config,
        export,
        output_dir=str(output_root / "{job_id}"),
    )
    recipe = json.dumps(
        {
            "argv": list(argv_template),
            "config": _config_payload(config),
            "cwd": str(cwd),
            "environment": environment,
            "export": {
                "dataset_revision": export.dataset_revision,
                "dataset_root": str(export.dataset_root.resolve()),
                "path": str(export.path),
                "sha256": export.sha256,
                "train_sha256": export.train_sha256,
                "validation_path": str(export.validation_path),
                "validation_sha256": export.validation_sha256,
            },
            "output_root": str(output_root),
            "runtime_requirements": [
                MS_SWIFT_REQUIREMENT,
                QWEN_VL_UTILS_REQUIREMENT,
                DECORD_REQUIREMENT,
                TRANSFORMERS_REQUIREMENT,
                TORCH_REQUIREMENT,
                TORCHVISION_REQUIREMENT,
            ],
            "schema_version": 1,
            "upstream_revision": MS_SWIFT_REVISION,
            "workspace_root": str(workspace_root),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    recipe_hash = sha256(recipe).hexdigest()
    job_id = f"qwen-evaluator-{recipe_hash}"
    relative_output = config.output_dir / job_id
    job_output = workspace_root / relative_output
    _validate_lexical_output_path(workspace_root, job_output, "job output")
    argv = _training_argv(config, export, output_dir=str(job_output))
    try:
        plan = TrainingPlan(
            schema_version=1,
            job_id=job_id,
            target=_TARGET,
            manifest_sha256=export.sha256,
            upstream_revision=MS_SWIFT_REVISION,
            cwd=cwd,
            argv=argv,
            environment=environment,
            expected_outputs=(relative_output,),
        )
    except TrainingSchemaError as error:
        raise EvaluatorConfigError("unable to create a safe evaluator training plan") from error
    if job_output.exists():
        if not job_output.is_dir():
            raise EvaluatorConfigError("refusing to reuse a non-directory Qwen output")
        if any(job_output.iterdir()):
            try:
                validate_qwen_training_output(
                    plan,
                    export,
                    config,
                    workspace_root=workspace_root,
                )
            except EvaluatorConfigError as error:
                raise EvaluatorConfigError(f"refusing to reuse nonempty output: {error}") from error
    return plan

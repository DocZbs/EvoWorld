"""Audited GR00T-N1.6 fine-tuning and open-loop evaluation plans."""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Any, cast

import imageio_ffmpeg  # type: ignore[import-untyped]

from evoworld.contracts import DiagnosisLabel
from training.common.schema import (
    TrainingManifest,
    TrainingPlan,
    TrainingSample,
    TrainingSchemaError,
)


GROOT_REVISION = "ead52833afbbf4243f8cd5e7664f48a94de03b19"
_BASE_MODEL = "nvidia/GR00T-N1.6-3B"
_TARGET = "groot_n16"
_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
_REQUIRED_UPSTREAM_PATHS = (
    Path("gr00t/experiment/launch_finetune.py"),
    Path("gr00t/eval/open_loop_eval.py"),
    Path("gr00t/configs/data/embodiment_configs.py"),
)
_REQUIRED_METADATA = (
    Path("meta/info.json"),
    Path("meta/tasks.jsonl"),
    Path("meta/episodes.jsonl"),
    Path("meta/stats.json"),
    Path("meta/relative_stats.json"),
    Path("meta/modality.json"),
    Path("meta/evoworld_manifest.json"),
)
_BUILTIN_EMBODIMENT_TAGS = {
    "LIBERO_PANDA": "libero_panda",
}
_RELEASED_EMBODIMENT_TAGS = frozenset({"NEW_EMBODIMENT", *_BUILTIN_EMBODIMENT_TAGS})
_CONFIG_FIELDS = frozenset(
    {
        "base_model_path",
        "embodiment_tag",
        "modality_config_path",
        "num_gpus",
        "global_batch_size",
        "max_steps",
        "save_steps",
        "save_total_limit",
        "dataloader_num_workers",
        "action_horizon",
        "open_loop_steps",
        "trajectory_ids",
    }
)
_STAT_KEYS = frozenset({"mean", "std", "min", "max", "q01", "q99"})
RevisionReader = Callable[[Path], str]


class GrootConfigError(ValueError):
    """Raised when a GR00T input cannot produce a method-aligned plan."""


@dataclass(frozen=True, slots=True)
class GrootTrainConfig:
    """Immutable EvoWorld settings for the pinned GR00T-N1.6 entry points."""

    base_model_path: str
    embodiment_tag: str
    modality_config_path: Path | None
    num_gpus: int
    global_batch_size: int
    max_steps: int
    save_steps: int
    save_total_limit: int
    dataloader_num_workers: int
    action_horizon: int
    open_loop_steps: int
    trajectory_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "trajectory_ids", tuple(self.trajectory_ids))
        if self.base_model_path != _BASE_MODEL:
            raise GrootConfigError(f"base_model_path must be {_BASE_MODEL}")
        if self.embodiment_tag not in _RELEASED_EMBODIMENT_TAGS:
            raise GrootConfigError(f"unsupported GR00T embodiment tag {self.embodiment_tag!r}")
        if self.embodiment_tag == "NEW_EMBODIMENT":
            if self.modality_config_path is None:
                raise GrootConfigError("NEW_EMBODIMENT requires a custom modality config")
            if (
                self.modality_config_path.is_absolute()
                or ".." in self.modality_config_path.parts
                or self.modality_config_path.suffix != ".py"
            ):
                raise GrootConfigError("modality_config_path must be a relative Python module path")
        elif self.modality_config_path is not None:
            raise GrootConfigError("released built-in tags cannot use a custom modality config")
        if (self.num_gpus, self.global_batch_size, self.max_steps) != (8, 64, 5000):
            raise GrootConfigError("GR00T requires 8 GPUs, global batch 64, and 5,000 steps")
        if (self.save_steps, self.save_total_limit, self.dataloader_num_workers) != (1000, 5, 2):
            raise GrootConfigError(
                "GR00T requires explicit pinned defaults: save_steps=1000, "
                "save_total_limit=5, dataloader_num_workers=2"
            )
        if self.action_horizon != 16:
            raise GrootConfigError("GR00T-N1.6 requires a 16-step action horizon")
        if self.open_loop_steps < 1:
            raise GrootConfigError("open_loop_steps must be positive")
        if len(self.trajectory_ids) != 1 or self.trajectory_ids[0] < 0:
            raise GrootConfigError("open-loop plans require exactly one trajectory ID")


@dataclass(frozen=True, slots=True)
class _ModalityRegistration:
    keys: dict[str, tuple[str, ...]]
    delta_indices: dict[str, tuple[int, ...]]
    embodiment_tag: str


def load_groot_config(path: Path) -> GrootTrainConfig:
    """Load the one strict GR00T-N1.6 TOML table without silent defaults."""
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
        values = document["groot_n16"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as error:
        raise GrootConfigError(f"unable to load GR00T configuration from {path}") from error
    if set(document) != {"groot_n16"} or not isinstance(values, dict):
        raise GrootConfigError("GR00T config must contain only the [groot_n16] table")
    unknown = set(values) - _CONFIG_FIELDS
    missing = (_CONFIG_FIELDS - {"modality_config_path"}) - set(values)
    if unknown:
        raise GrootConfigError(f"unknown GR00T config fields: {sorted(unknown)}")
    if missing:
        raise GrootConfigError(f"missing GR00T config fields: {sorted(missing)}")
    try:
        return GrootTrainConfig(
            base_model_path=_string(values, "base_model_path"),
            embodiment_tag=_string(values, "embodiment_tag"),
            modality_config_path=_optional_path(values, "modality_config_path"),
            num_gpus=_integer(values, "num_gpus"),
            global_batch_size=_integer(values, "global_batch_size"),
            max_steps=_integer(values, "max_steps"),
            save_steps=_integer(values, "save_steps"),
            save_total_limit=_integer(values, "save_total_limit"),
            dataloader_num_workers=_integer(values, "dataloader_num_workers"),
            action_horizon=_integer(values, "action_horizon"),
            open_loop_steps=_integer(values, "open_loop_steps"),
            trajectory_ids=_integer_tuple(values, "trajectory_ids"),
        )
    except KeyError as error:
        raise GrootConfigError(f"missing GR00T config field {error.args[0]!r}") from error


def _git_revision(root: Path) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise GrootConfigError(f"unable to read GR00T revision from {root}") from error
    return result.stdout.strip()


def validate_groot_root(
    root: Path,
    *,
    revision_reader: RevisionReader = _git_revision,
) -> Path:
    """Require the official entry points at the exact N1.6 release revision."""
    if not root.is_absolute():
        raise GrootConfigError("groot_root must be absolute")
    resolved = root.resolve()
    for relative in _REQUIRED_UPSTREAM_PATHS:
        if not (resolved / relative).is_file():
            raise GrootConfigError(f"groot_root is missing {relative.as_posix()}")
    revision = revision_reader(resolved)
    if revision != GROOT_REVISION:
        raise GrootConfigError(
            f"GR00T revision must be {GROOT_REVISION}, got {revision or '<blank>'}"
        )
    return resolved


def validate_groot_dataset(
    dataset_root: Path,
    manifest: TrainingManifest,
    config: GrootTrainConfig,
    *,
    groot_root: Path,
) -> Path:
    """Validate a complete, train-only GR00T-flavoured LeRobot v2.1 root."""
    root, _ = _validate_groot_dataset(dataset_root, manifest, config, groot_root=groot_root)
    return root


def _validate_groot_dataset(
    dataset_root: Path,
    manifest: TrainingManifest,
    config: GrootTrainConfig,
    *,
    groot_root: Path,
) -> tuple[Path, _ModalityRegistration]:
    if not dataset_root.is_absolute():
        raise GrootConfigError("GR00T dataset root must be absolute")
    root = dataset_root.resolve()
    if not root.is_dir():
        raise GrootConfigError(f"GR00T dataset root does not exist: {root}")
    for relative in _REQUIRED_METADATA:
        if not (root / relative).is_file():
            raise GrootConfigError(f"GR00T dataset is missing {relative.as_posix()}")

    info = _read_json(root / "meta/info.json")
    modality = _read_json(root / "meta/modality.json")
    tasks = _read_jsonlines(root / "meta/tasks.jsonl")
    episodes = _read_jsonlines(root / "meta/episodes.jsonl")
    stats = _read_json(root / "meta/stats.json")
    relative_stats = _read_json(root / "meta/relative_stats.json")
    if config.embodiment_tag == "NEW_EMBODIMENT":
        if config.modality_config_path is None:  # guarded by GrootTrainConfig
            raise GrootConfigError("NEW_EMBODIMENT requires a registration module")
        registration_path = _contained_file(
            root,
            root / config.modality_config_path,
            "registration module",
        )
        registration = _parse_registration(registration_path)
        if registration.embodiment_tag != config.embodiment_tag:
            raise GrootConfigError("registration embodiment tag does not match the training config")
        _probe_registration_import(groot_root, registration_path, registration)
    else:
        registration = _parse_builtin_registration(groot_root, config.embodiment_tag)

    action_dimension, state_dimension = _validate_info(info, tasks, episodes, modality)
    required_rows = max(registration.delta_indices["action"]) + 1
    if any(cast(int, episode["length"]) < required_rows for episode in episodes):
        raise GrootConfigError("every GR00T episode must cover the full 16-step action horizon")
    _validate_modality(modality, info, registration, action_dimension, state_dimension)
    _validate_statistics(stats, relative_stats, modality, action_dimension, state_dimension, config)
    _validate_episode_files(
        root,
        info,
        modality,
        registration,
        tasks,
        episodes,
        action_dimension,
        state_dimension,
    )
    _validate_manifest(root, manifest, config, tasks, episodes, action_dimension)

    sidecar = _load_manifest(root / "meta/evoworld_manifest.json")
    if sidecar != manifest:
        raise GrootConfigError("GR00T dataset manifest sidecar does not match the planned manifest")
    return root, registration


def plan_groot_training(
    manifest: TrainingManifest,
    config: GrootTrainConfig,
    *,
    groot_root: Path,
    dataset_root: Path,
    output_root: Path,
    manifest_path: Path,
    revision_reader: RevisionReader = _git_revision,
) -> TrainingPlan:
    """Build a safe argv for the pinned official N1.6 fine-tuning entry point."""
    _require_absolute((groot_root, dataset_root, output_root, manifest_path))
    groot_root = validate_groot_root(groot_root, revision_reader=revision_reader)
    dataset_root, _ = _validate_groot_dataset(
        dataset_root,
        manifest,
        config,
        groot_root=groot_root,
    )
    expected_manifest_path = dataset_root / "meta/evoworld_manifest.json"
    if manifest_path.resolve() != expected_manifest_path:
        raise GrootConfigError(
            "manifest_path must be contained at dataset/meta/evoworld_manifest.json"
        )
    if _load_manifest(manifest_path) != manifest:
        raise GrootConfigError("manifest_path does not contain the planned immutable manifest")
    output_root = output_root.resolve()
    job_id = f"groot-n16-{manifest.sha256[:12]}"
    job_output = _contained_path(output_root, output_root / job_id, "training output")
    modality_args: tuple[str, ...] = ()
    if config.modality_config_path is not None:
        modality_path = _contained_file(
            dataset_root,
            dataset_root / config.modality_config_path,
            "registration module",
        )
        modality_args = ("--modality-config-path", str(modality_path))
    argv = (
        "uv",
        "run",
        "python",
        "gr00t/experiment/launch_finetune.py",
        "--base-model-path",
        config.base_model_path,
        "--dataset-path",
        str(dataset_root),
        "--embodiment-tag",
        config.embodiment_tag,
        *modality_args,
        "--num-gpus",
        str(config.num_gpus),
        "--output-dir",
        str(job_output),
        "--save-total-limit",
        str(config.save_total_limit),
        "--save-steps",
        str(config.save_steps),
        "--max-steps",
        str(config.max_steps),
        "--global-batch-size",
        str(config.global_batch_size),
        "--dataloader-num-workers",
        str(config.dataloader_num_workers),
    )
    return _training_plan(
        job_id=job_id,
        target=_TARGET,
        manifest=manifest,
        cwd=groot_root,
        argv=argv,
        expected_outputs=(Path(f"checkpoint-{config.max_steps}"),),
    )


def plan_groot_open_loop_eval(
    manifest: TrainingManifest,
    config: GrootTrainConfig,
    *,
    groot_root: Path,
    dataset_root: Path,
    checkpoint_path: Path,
    output_root: Path,
    revision_reader: RevisionReader = _git_revision,
) -> TrainingPlan:
    """Build the distinct official open-loop evaluation argv for a final checkpoint."""
    _require_absolute((groot_root, dataset_root, checkpoint_path, output_root))
    groot_root = validate_groot_root(groot_root, revision_reader=revision_reader)
    dataset_root, registration = _validate_groot_dataset(
        dataset_root,
        manifest,
        config,
        groot_root=groot_root,
    )
    info = _read_json(dataset_root / "meta/info.json")
    total_episodes = cast(int, info["total_episodes"])
    if max(config.trajectory_ids) >= total_episodes:
        raise GrootConfigError("open-loop trajectory IDs must exist in the validated dataset")
    output_root = output_root.resolve()
    checkpoint_path = _contained_file(
        output_root,
        checkpoint_path,
        "open-loop checkpoint",
        directory=True,
    )
    expected_name = f"checkpoint-{config.max_steps}"
    training_job_id = f"groot-n16-{manifest.sha256[:12]}"
    expected_checkpoint = output_root / training_job_id / expected_name
    if checkpoint_path != expected_checkpoint:
        raise GrootConfigError(
            f"open-loop checkpoint must be the final artifact of training job {training_job_id}"
        )
    job_id = f"groot-open-loop-{manifest.sha256[:12]}"
    job_output = _contained_path(output_root, output_root / job_id, "open-loop output")
    action_keys = registration.keys["action"]
    argv = (
        "uv",
        "run",
        "python",
        "gr00t/eval/open_loop_eval.py",
        "--dataset-path",
        str(dataset_root),
        "--embodiment-tag",
        config.embodiment_tag,
        "--model-path",
        str(checkpoint_path),
        "--traj-ids",
        *(str(index) for index in config.trajectory_ids),
        "--action-horizon",
        str(config.action_horizon),
        "--steps",
        str(config.open_loop_steps),
        "--save-plot-path",
        str(job_output / "open-loop.jpeg"),
        "--modality-keys",
        *action_keys,
    )
    return _training_plan(
        job_id=job_id,
        target=f"{_TARGET}_open_loop",
        manifest=manifest,
        cwd=groot_root,
        argv=argv,
        expected_outputs=(Path("open-loop.jpeg"),),
    )


def _training_plan(
    *,
    job_id: str,
    target: str,
    manifest: TrainingManifest,
    cwd: Path,
    argv: tuple[str, ...],
    expected_outputs: tuple[Path, ...],
) -> TrainingPlan:
    try:
        return TrainingPlan(
            schema_version=1,
            job_id=job_id,
            target=target,
            manifest_sha256=manifest.sha256,
            upstream_revision=GROOT_REVISION,
            cwd=cwd,
            argv=argv,
            environment={},
            expected_outputs=expected_outputs,
        )
    except TrainingSchemaError as error:
        raise GrootConfigError("unable to create a safe GR00T plan") from error


def _validate_info(
    info: dict[str, object],
    tasks: tuple[dict[str, object], ...],
    episodes: tuple[dict[str, object], ...],
    modality: dict[str, object],
) -> tuple[int, int]:
    if (
        info.get("codebase_version") != "v2.1"
        or info.get("data_path") != _DATA_PATH
        or info.get("video_path") != _VIDEO_PATH
    ):
        raise GrootConfigError("GR00T dataset must use pinned LeRobot v2.1 data and video paths")
    if info.get("chunks_size") != 1000:
        raise GrootConfigError("GR00T LeRobot v2.1 chunks_size must be 1000")
    fps = info.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
        raise GrootConfigError("GR00T info.json fps must be positive")
    if info.get("splits") != {"train": f"0:{len(episodes)}"}:
        raise GrootConfigError("GR00T fine-tuning requires a train-only split without eval leakage")
    if not tasks or not episodes:
        raise GrootConfigError("GR00T task and episode metadata must not be empty")
    if [row.get("task_index") for row in tasks] != list(range(len(tasks))):
        raise GrootConfigError("GR00T task indices must be contiguous")
    if [row.get("episode_index") for row in episodes] != list(range(len(episodes))):
        raise GrootConfigError("GR00T episode indices must be contiguous")
    lengths = [row.get("length") for row in episodes]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in lengths):
        raise GrootConfigError("GR00T episode lengths must be positive integers")
    if (
        info.get("total_tasks") != len(tasks)
        or info.get("total_episodes") != len(episodes)
        or info.get("total_frames") != sum(cast(list[int], lengths))
    ):
        raise GrootConfigError("GR00T metadata counts do not match task and episode rows")
    features = info.get("features")
    if not isinstance(features, dict):
        raise GrootConfigError("GR00T info.json features must be an object")
    action_dimension = _feature_width(features, "action", "float32")
    state_dimension = _feature_width(features, "observation.state", "float32")
    for key, dtype in (
        ("task_index", "int64"),
        ("episode_index", "int64"),
        ("frame_index", "int64"),
        ("index", "int64"),
        ("timestamp", "float32"),
    ):
        _validate_scalar_feature(features, key, dtype)
    video = modality.get("video")
    if not isinstance(video, dict):
        raise GrootConfigError("GR00T modality.json must define video modalities")
    if info.get("total_videos") != len(video) * len(episodes):
        raise GrootConfigError("GR00T total_videos does not match video modalities and episodes")
    return action_dimension, state_dimension


def _feature_width(features: dict[str, object], key: str, dtype: str) -> int:
    feature = features.get(key)
    if not isinstance(feature, dict) or feature.get("dtype") != dtype:
        raise GrootConfigError(f"GR00T feature {key!r} must use {dtype}")
    shape = feature.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 1
        or isinstance(shape[0], bool)
        or not isinstance(shape[0], int)
        or shape[0] < 1
    ):
        raise GrootConfigError(f"GR00T feature {key!r} must have a positive vector shape")
    return cast(int, shape[0])


def _validate_scalar_feature(features: dict[str, object], key: str, dtype: str) -> None:
    feature = features.get(key)
    if (
        not isinstance(feature, dict)
        or feature.get("dtype") != dtype
        or feature.get("shape") != [1]
    ):
        raise GrootConfigError(f"GR00T features must include {key!r} as {dtype} shape [1]")


def _validate_modality(
    modality: dict[str, object],
    info: dict[str, object],
    registration: _ModalityRegistration,
    action_dimension: int,
    state_dimension: int,
) -> None:
    required = {"state", "action", "video", "annotation"}
    missing = required - set(modality)
    if missing:
        raise GrootConfigError(f"GR00T modality.json is missing {sorted(missing)}")
    _validate_ranges(modality["state"], registration.keys["state"], state_dimension, "state")
    _validate_ranges(modality["action"], registration.keys["action"], action_dimension, "action")
    features = cast(dict[str, object], info["features"])
    video = modality["video"]
    if not isinstance(video, dict) or not registration.keys["video"]:
        raise GrootConfigError("GR00T video modalities must not be empty")
    if set(video) != set(registration.keys["video"]):
        raise GrootConfigError("GR00T video registration keys must exactly match modality.json")
    for key in registration.keys["video"]:
        mapping = video.get(key)
        if not isinstance(mapping, dict) or not isinstance(mapping.get("original_key"), str):
            raise GrootConfigError(f"GR00T video modality {key!r} has no original_key")
        feature = features.get(cast(str, mapping["original_key"]))
        if (
            not isinstance(feature, dict)
            or feature.get("dtype") != "video"
            or not _valid_video_shape(feature.get("shape"))
        ):
            raise GrootConfigError(f"GR00T video modality {key!r} is not backed by a video feature")
    annotation = modality["annotation"]
    language_keys = registration.keys["language"]
    if not isinstance(annotation, dict) or len(language_keys) != 1:
        raise GrootConfigError("GR00T language modality must contain exactly one key")
    language_key = language_keys[0]
    annotation_key = language_key.removeprefix("annotation.")
    mapping = annotation.get(annotation_key)
    original_key = mapping.get("original_key") if isinstance(mapping, dict) else None
    language_feature = features.get(original_key) if isinstance(original_key, str) else None
    if (
        not isinstance(language_feature, dict)
        or language_feature.get("dtype") != "int64"
        or language_feature.get("shape") != [1]
    ):
        raise GrootConfigError("GR00T language key must map to an int64 task-index feature")


def _valid_video_shape(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value)
        and value[2] == 3
    )


def _validate_ranges(value: object, keys: tuple[str, ...], width: int, name: str) -> None:
    if not isinstance(value, dict) or not keys:
        raise GrootConfigError(f"GR00T {name} modalities must not be empty")
    if set(value) != set(keys):
        raise GrootConfigError(f"GR00T {name} registration keys must exactly match modality.json")
    cursor = 0
    for key in keys:
        item = value.get(key)
        if not isinstance(item, dict):
            raise GrootConfigError(f"GR00T {name} modality {key!r} is missing")
        start, end = item.get("start"), item.get("end")
        if start != cursor or isinstance(end, bool) or not isinstance(end, int) or end <= cursor:
            raise GrootConfigError(f"GR00T {name} modality ranges must be contiguous")
        cursor = end
    if cursor != width:
        raise GrootConfigError(f"GR00T {name} shape {width} does not match modality ranges")


def _validate_statistics(
    stats: dict[str, object],
    relative_stats: dict[str, object],
    modality: dict[str, object],
    action_dimension: int,
    state_dimension: int,
    config: GrootTrainConfig,
) -> None:
    _validate_stat_block(stats.get("action"), action_dimension, "action stats")
    _validate_stat_block(stats.get("observation.state"), state_dimension, "state stats")
    action = modality.get("action")
    if not isinstance(action, dict):
        raise GrootConfigError("GR00T action modalities must be an object")
    for key, item in action.items():
        if not isinstance(item, dict):
            raise GrootConfigError(f"GR00T action modality {key!r} must be an object")
        width = cast(int, item["end"]) - cast(int, item["start"])
        block = relative_stats.get(key)
        if not isinstance(block, dict) or set(block) != _STAT_KEYS:
            raise GrootConfigError(f"GR00T relative stats are missing action modality {key!r}")
        for rows in block.values():
            if (
                not isinstance(rows, list)
                or len(rows) != config.action_horizon
                or any(
                    not isinstance(row, list)
                    or len(row) != width
                    or any(not _finite_number(value) for value in row)
                    for row in rows
                )
            ):
                raise GrootConfigError(
                    "GR00T relative action stats require finite numeric leaves matching deltas"
                )


def _validate_stat_block(value: object, width: int, name: str) -> None:
    if not isinstance(value, dict) or set(value) != _STAT_KEYS:
        raise GrootConfigError(f"GR00T {name} must contain the six official statistic fields")
    if any(
        not isinstance(vector, list)
        or len(vector) != width
        or any(not _finite_number(item) for item in vector)
        for vector in value.values()
    ):
        raise GrootConfigError(f"GR00T {name} must contain finite numeric vectors")


def _finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _validate_episode_files(
    root: Path,
    info: dict[str, object],
    modality: dict[str, object],
    registration: _ModalityRegistration,
    tasks: tuple[dict[str, object], ...],
    episodes: tuple[dict[str, object], ...],
    action_dimension: int,
    state_dimension: int,
) -> None:
    video = cast(dict[str, dict[str, object]], modality["video"])
    annotation = cast(dict[str, dict[str, object]], modality["annotation"])
    language_key = registration.keys["language"][0].removeprefix("annotation.")
    language_column = cast(str, annotation[language_key]["original_key"])
    task_map = {cast(int, row["task_index"]): cast(str, row["task"]) for row in tasks}
    global_index = 0
    for index, episode in enumerate(episodes):
        episode_length = cast(int, episode["length"])
        episode_tasks = cast(list[str], episode["tasks"])
        parquet_path = root / _DATA_PATH.format(episode_chunk=index // 1000, episode_index=index)
        _contained_file(root, parquet_path, "episode parquet")
        if not _has_parquet_magic(parquet_path):
            raise GrootConfigError(
                f"episode parquet is not a valid Parquet container: {parquet_path}"
            )
        _validate_parquet_contents(
            parquet_path,
            expected_rows=episode_length,
            episode_index=index,
            global_index=global_index,
            fps=cast(float, info["fps"]),
            action_dimension=action_dimension,
            state_dimension=state_dimension,
            language_column=language_column,
            episode_tasks=episode_tasks,
            task_map=task_map,
        )
        global_index += episode_length
        if not episode_tasks or not all(isinstance(task, str) for task in episode_tasks):
            raise GrootConfigError("GR00T episodes must bind at least one language task")
        for mapping in video.values():
            original_key = cast(str, mapping["original_key"])
            video_path = root / cast(str, info["video_path"]).format(
                episode_chunk=index // 1000,
                video_key=original_key,
                episode_index=index,
            )
            _contained_file(root, video_path, "episode video")
            feature = cast(
                dict[str, object], cast(dict[str, object], info["features"])[original_key]
            )
            shape = cast(list[int], feature["shape"])
            probe = _probe_mp4(video_path)
            if (probe.height, probe.width) != (shape[0], shape[1]):
                raise GrootConfigError("GR00T video stream dimensions do not match info.json")
            if probe.frame_count != episode_length:
                raise GrootConfigError("GR00T video frames do not match episode rows")
            if not math.isclose(probe.fps, cast(float, info["fps"]), rel_tol=1e-4):
                raise GrootConfigError("GR00T video stream fps does not match info.json")


def _validate_manifest(
    root: Path,
    manifest: TrainingManifest,
    config: GrootTrainConfig,
    tasks: tuple[dict[str, object], ...],
    episodes: tuple[dict[str, object], ...],
    action_dimension: int,
) -> None:
    if len(manifest.samples) != len(episodes):
        raise GrootConfigError("GR00T manifest must contain exactly one sample per train episode")
    task_values = {row.get("task") for row in tasks}
    dataset_sha256 = _dataset_content_sha256(root, config)
    for index, (sample, episode) in enumerate(zip(manifest.samples, episodes, strict=True)):
        if sample.diagnosis is not DiagnosisLabel.PF:
            raise GrootConfigError("GR00T expert fine-tuning accepts only PF samples")
        if sample.split != "train":
            raise GrootConfigError("GR00T expert fine-tuning accepts only train samples")
        if sample.target != _TARGET:
            raise GrootConfigError(f"GR00T manifest samples must target {_TARGET}")
        if sample.dataset_root != Path("data/chunk-000"):
            raise GrootConfigError("GR00T manifest samples must bind data/chunk-000")
        expected_parquet = Path(f"episode_{index:06d}.parquet")
        if (
            sample.observation_paths != (expected_parquet,)
            or sample.action_path != expected_parquet
        ):
            raise GrootConfigError("GR00T manifest paths must bind the matching episode parquet")
        if (
            sample.action_schema != "relative_joint_position"
            or sample.action_dimension != action_dimension
            or sample.action_horizon != config.action_horizon
        ):
            raise GrootConfigError(
                "GR00T action provenance does not match action shape and horizon"
            )
        episode_tasks = episode.get("tasks")
        if sample.instruction not in task_values or sample.instruction not in cast(
            list[object], episode_tasks
        ):
            raise GrootConfigError("GR00T sample language is not bound to episode task metadata")
        if dataset_sha256 != sample.content_sha256:
            raise GrootConfigError(
                f"GR00T sample content provenance changed for {sample.sample_id}"
            )


def _dataset_content_sha256(root: Path, config: GrootTrainConfig) -> str:
    digest = sha256()
    artifacts = [
        ("metadata", Path(f"meta/{name}"))
        for name in (
            "info.json",
            "tasks.jsonl",
            "episodes.jsonl",
            "stats.json",
            "relative_stats.json",
            "modality.json",
        )
    ]
    if config.modality_config_path is not None:
        artifacts.append(("registration", config.modality_config_path))
    artifacts.extend(("parquet", path.relative_to(root)) for path in root.glob("data/**/*.parquet"))
    artifacts.extend(("video", path.relative_to(root)) for path in root.glob("videos/**/*.mp4"))
    seen: set[Path] = set()
    for role, relative in sorted(artifacts, key=lambda value: (value[0], value[1].as_posix())):
        if relative in seen:
            continue
        seen.add(relative)
        artifact = _contained_file(root, root / relative, "provenance artifact")
        digest.update(f"{role}:{relative.as_posix()}\0".encode())
        with artifact.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _parse_registration(path: Path) -> _ModalityRegistration:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise GrootConfigError(f"invalid GR00T registration module {path}") from error
    body = list(tree.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if len(body) != 5:
        raise GrootConfigError("registration module contains unsafe top-level code")
    import_nodes = [node for node in body if isinstance(node, ast.ImportFrom)]
    if len(import_nodes) != 3 or any(
        node.level != 0 or len(node.names) != 1 for node in import_nodes
    ):
        raise GrootConfigError("registration module imports must use three exact official symbols")
    imports = {
        (node.module, alias.name, alias.asname)
        for node in body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    required_imports = {
        ("gr00t.configs.data.embodiment_configs", "register_modality_config", None),
        ("gr00t.data.embodiment_tags", "EmbodimentTag", None),
        ("gr00t.data.types", "ModalityConfig", None),
    }
    if imports != required_imports:
        raise GrootConfigError(
            "registration module must import official GR00T registration symbols"
        )
    assignments = [node for node in body if isinstance(node, ast.Assign)]
    if (
        len(assignments) != 1
        or len(assignments[0].targets) != 1
        or not isinstance(assignments[0].targets[0], ast.Name)
        or assignments[0].targets[0].id != "evoworld_config"
        or not isinstance(assignments[0].value, ast.Dict)
    ):
        raise GrootConfigError(
            "registration module must contain one unshadowed evoworld_config assignment"
        )
    assignment = assignments[0]
    assignment_name = cast(ast.Name, assignment.targets[0]).id
    register_calls = [
        node.value
        for node in body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "register_modality_config"
    ]
    if len(register_calls) != 1 or len(register_calls[0].args) != 1:
        raise GrootConfigError(
            "registration module must call register_modality_config exactly once"
        )
    call = register_calls[0]
    if {keyword.arg for keyword in call.keywords} != {"embodiment_tag"}:
        raise GrootConfigError("registration call permits only the embodiment_tag keyword")
    if not isinstance(call.args[0], ast.Name) or call.args[0].id != assignment_name:
        raise GrootConfigError("registration call must reference a literal modality configuration")
    tag_node = _keyword(call, "embodiment_tag")
    if (
        not isinstance(tag_node, ast.Attribute)
        or not isinstance(tag_node.value, ast.Name)
        or tag_node.value.id != "EmbodimentTag"
    ):
        raise GrootConfigError("registration call must use an explicit EmbodimentTag member")
    config_dict = cast(ast.Dict, assignment.value)
    registration = _registration_from_dict(config_dict, tag_node.attr, custom=True)
    if registration.embodiment_tag != "NEW_EMBODIMENT":
        raise GrootConfigError("custom registration is allowed only for NEW_EMBODIMENT")
    return registration


def _parse_builtin_registration(root: Path, embodiment_tag: str) -> _ModalityRegistration:
    source_path = root / "gr00t/configs/data/embodiment_configs.py"
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, SyntaxError) as error:
        raise GrootConfigError("unable to parse pinned built-in modality configurations") from error
    assignments = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "MODALITY_CONFIGS"
            for target in node.targets
        )
        and isinstance(node.value, ast.Dict)
    ]
    if len(assignments) != 1:
        raise GrootConfigError("pinned built-in modality config registry is missing")
    registry = assignments[0]
    registry_key = _BUILTIN_EMBODIMENT_TAGS[embodiment_tag]
    for key_node, value_node in zip(registry.keys, registry.values, strict=True):
        try:
            key = ast.literal_eval(key_node) if key_node is not None else None
        except (ValueError, TypeError):
            continue
        if key == registry_key and isinstance(value_node, ast.Dict):
            return _registration_from_dict(value_node, embodiment_tag, custom=False)
    raise GrootConfigError(f"pinned built-in config does not define {embodiment_tag}")


def _registration_from_dict(
    config_dict: ast.Dict,
    embodiment_tag: str,
    *,
    custom: bool,
) -> _ModalityRegistration:
    if any(key is None for key in config_dict.keys):
        raise GrootConfigError("registration dictionaries cannot use unpacking")
    entries: dict[str, ast.AST] = {}
    for key_node, value_node in zip(config_dict.keys, config_dict.values, strict=True):
        if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
            raise GrootConfigError("registration modality names must be string literals")
        key = key_node.value
        if key in entries:
            raise GrootConfigError("registration modality names must not be duplicated")
        entries[key] = value_node
    required = {"video", "state", "action", "language"}
    if custom and set(entries) != required:
        raise GrootConfigError("custom registration must contain exactly four modality entries")
    if not custom and not required <= set(entries):
        raise GrootConfigError(
            f"registration module is missing modalities {sorted(required - set(entries))}"
        )
    keys: dict[str, tuple[str, ...]] = {}
    deltas: dict[str, tuple[int, ...]] = {}
    for name in required:
        value = entries[name]
        if (
            not isinstance(value, ast.Call)
            or not isinstance(value.func, ast.Name)
            or value.func.id != "ModalityConfig"
        ):
            raise GrootConfigError(f"registration {name} must construct ModalityConfig")
        if value.args:
            raise GrootConfigError(f"registration {name} must use explicit ModalityConfig keywords")
        keyword_names = [keyword.arg for keyword in value.keywords]
        if None in keyword_names or len(keyword_names) != len(set(keyword_names)):
            raise GrootConfigError(
                "registration ModalityConfig keywords must be explicit and unique"
            )
        if custom and set(keyword_names) != {"delta_indices", "modality_keys"}:
            raise GrootConfigError(
                "custom registration permits only delta_indices and modality_keys"
            )
        keys[name] = _string_ast_tuple(_keyword(value, "modality_keys"), f"{name} modality_keys")
        deltas[name] = _integer_ast_tuple(_keyword(value, "delta_indices"), f"{name} delta_indices")
    for name in ("video", "state", "language"):
        if deltas[name] != (0,):
            raise GrootConfigError(f"GR00T {name} delta indices must be [0]")
    if deltas["action"] != tuple(range(16)):
        raise GrootConfigError("GR00T action delta indices must be contiguous [0, 16)")
    return _ModalityRegistration(keys=keys, delta_indices=deltas, embodiment_tag=embodiment_tag)


def _probe_registration_import(
    root: Path,
    registration_path: Path,
    registration: _ModalityRegistration,
) -> None:
    probe = """
import importlib.util
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
registration = pathlib.Path(sys.argv[2])
expected = json.loads(sys.argv[3])
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location("evoworld_modality_probe", registration)
if spec is None or spec.loader is None:
    raise RuntimeError("registration spec unavailable")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if pathlib.Path(module.__file__).resolve() != registration.resolve():
    raise RuntimeError("registration probe imported a different source")
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
actual = MODALITY_CONFIGS.get(expected["registry_key"])
if actual is None:
    raise RuntimeError("NEW_EMBODIMENT was not registered")
if set(actual) != set(expected["modalities"]):
    raise RuntimeError("registered config modalities differ from the audited source")
for name, expected_config in expected["modalities"].items():
    actual_config = actual[name]
    observed = {
        "delta_indices": list(actual_config.delta_indices),
        "modality_keys": list(actual_config.modality_keys),
    }
    if observed != expected_config:
        raise RuntimeError(f"registered config differs for {name}")
"""
    expected = json.dumps(
        {
            "registry_key": "new_embodiment",
            "modalities": {
                name: {
                    "delta_indices": list(registration.delta_indices[name]),
                    "modality_keys": list(registration.keys[name]),
                }
                for name in ("video", "state", "action", "language")
            },
        },
        sort_keys=True,
    )
    try:
        subprocess.run(
            (sys.executable, "-I", "-c", probe, str(root), str(registration_path), expected),
            cwd=root,
            env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONNOUSERSITE": "1"},
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise GrootConfigError("registration failed isolated import probe") from error


def _keyword(call: ast.Call, name: str) -> ast.AST:
    matches = [keyword.value for keyword in call.keywords if keyword.arg == name]
    if len(matches) != 1:
        raise GrootConfigError(f"registration call requires exactly one {name}")
    return matches[0]


def _string_ast_tuple(node: ast.AST, name: str) -> tuple[str, ...]:
    if not isinstance(node, (ast.List, ast.Tuple)) or not node.elts:
        raise GrootConfigError(f"registration {name} must be a non-empty string list")
    values: list[str] = []
    for item in node.elts:
        if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
            raise GrootConfigError(f"registration {name} must contain literal strings")
        values.append(item.value)
    if len(values) != len(set(values)):
        raise GrootConfigError(f"registration {name} must not contain duplicate keys")
    return tuple(values)


def _integer_ast_tuple(node: ast.AST, name: str) -> tuple[int, ...]:
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_literal_ast_integer(item, name) for item in node.elts]
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "list"
        and not node.keywords
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Name)
        and node.args[0].func.id == "range"
        and not node.args[0].keywords
    ):
        range_values = tuple(_literal_ast_integer(item, name) for item in node.args[0].args)
        if not 1 <= len(range_values) <= 3 or any(abs(value) > 1024 for value in range_values):
            raise GrootConfigError(f"registration {name} range is outside safe bounds")
        try:
            value_range = range(*range_values)
        except ValueError as error:
            raise GrootConfigError(f"registration {name} range is invalid") from error
        if len(value_range) > 256:
            raise GrootConfigError(f"registration {name} range is outside safe bounds")
        values = list(value_range)
    else:
        raise GrootConfigError(f"registration {name} must be a literal list or list(range(...))")
    if not values or any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise GrootConfigError(f"registration {name} must contain integers")
    return tuple(values)


def _literal_ast_integer(node: ast.AST, name: str) -> int:
    if (
        not isinstance(node, ast.Constant)
        or isinstance(node.value, bool)
        or not isinstance(node.value, int)
    ):
        raise GrootConfigError(f"registration {name} must contain literal integers")
    return node.value


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise GrootConfigError(f"invalid JSON metadata {path}") from error
    if not isinstance(value, dict):
        raise GrootConfigError(f"metadata must be a JSON object: {path}")
    return cast(dict[str, object], value)


def _read_jsonlines(path: Path) -> tuple[dict[str, object], ...]:
    try:
        values = tuple(
            json.loads(line, parse_constant=_reject_json_constant)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        )
    except (OSError, json.JSONDecodeError) as error:
        raise GrootConfigError(f"invalid JSONL metadata {path}") from error
    if not all(isinstance(value, dict) for value in values):
        raise GrootConfigError(f"JSONL metadata must contain objects: {path}")
    return cast(tuple[dict[str, object], ...], values)


def _reject_json_constant(value: str) -> object:
    raise GrootConfigError(f"JSON constant {value!r} is not finite")


def _load_manifest(path: Path) -> TrainingManifest:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
        samples = tuple(
            _sample_from_dict(cast(dict[str, object], item)) for item in payload["samples"]
        )
        return TrainingManifest(
            schema_version=cast(int, payload["schema_version"]),
            manifest_id=cast(str, payload["manifest_id"]),
            samples=samples,
        )
    except (OSError, KeyError, TypeError, ValueError, TrainingSchemaError) as error:
        raise GrootConfigError(f"invalid GR00T manifest artifact {path}") from error


def _sample_from_dict(values: dict[str, object]) -> TrainingSample:
    action_path = values["action_path"]
    return TrainingSample(
        schema_version=cast(int, values["schema_version"]),
        sample_id=cast(str, values["sample_id"]),
        target=cast(str, values["target"]),
        task_id=cast(str, values["task_id"]),
        instruction=cast(str, values["instruction"]),
        skill_family=cast(str, values["skill_family"]),
        scene_id=cast(str, values["scene_id"]),
        trajectory_id=cast(str, values["trajectory_id"]),
        dataset_root=Path(cast(str, values["dataset_root"])),
        observation_paths=tuple(
            Path(cast(str, item)) for item in cast(list[object], values["observation_paths"])
        ),
        action_path=Path(cast(str, action_path)) if action_path is not None else None,
        action_schema=cast(str | None, values["action_schema"]),
        action_dimension=cast(int | None, values["action_dimension"]),
        action_horizon=cast(int | None, values["action_horizon"]),
        source_type=cast(str, values["source_type"]),
        source_checkpoint=cast(str, values["source_checkpoint"]),
        diagnosis=DiagnosisLabel(cast(str, values["diagnosis"])),
        split=cast(str, values["split"]),
        content_sha256=cast(str, values["content_sha256"]),
    )


def _contained_file(root: Path, candidate: Path, name: str, *, directory: bool = False) -> Path:
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise GrootConfigError(f"{name} must be contained within {resolved_root}")
    exists = resolved.is_dir() if directory else resolved.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise GrootConfigError(f"{name} {kind} does not exist: {resolved}")
    return resolved


def _contained_path(root: Path, candidate: Path, name: str) -> Path:
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise GrootConfigError(f"{name} must be a child of {resolved_root}")
    return resolved


def _require_absolute(paths: Iterable[Path]) -> None:
    if not all(path.is_absolute() for path in paths):
        raise GrootConfigError(
            "GR00T root, dataset, output, manifest, and checkpoint paths must be absolute"
        )


def _has_parquet_magic(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            first = stream.read(4)
            stream.seek(-4, 2)
            last = stream.read(4)
    except (OSError, ValueError):
        return False
    return first == b"PAR1" and last == b"PAR1"


def _validate_parquet_contents(
    path: Path,
    *,
    expected_rows: int,
    episode_index: int,
    global_index: int,
    fps: float,
    action_dimension: int,
    state_dimension: int,
    language_column: str,
    episode_tasks: list[str],
    task_map: dict[int, str],
) -> None:
    try:
        import pyarrow as arrow  # type: ignore[import-untyped]
        import pyarrow.parquet as parquet  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - exercised in a runtime-only install
        raise GrootConfigError("pyarrow is required to audit GR00T Parquet metadata") from error
    try:
        parquet_file = parquet.ParquetFile(path)
        schema = parquet_file.schema_arrow
        columns = set(schema.names)
        row_count = parquet_file.metadata.num_rows
        table = parquet_file.read()
    except Exception as error:
        raise GrootConfigError(f"unable to read GR00T Parquet metadata: {path}") from error
    required_columns = {
        "observation.state",
        "action",
        "task_index",
        "episode_index",
        "frame_index",
        "index",
        "timestamp",
        language_column,
    }
    missing = required_columns - columns
    if missing:
        raise GrootConfigError(f"GR00T Parquet columns are missing {sorted(missing)}")
    if row_count != expected_rows:
        raise GrootConfigError(
            f"GR00T Parquet rows {row_count} do not match episode length {expected_rows}"
        )
    vector_types = {
        "action": arrow.list_(arrow.float32(), action_dimension),
        "observation.state": arrow.list_(arrow.float32(), state_dimension),
    }
    for column, expected_type in vector_types.items():
        actual_type = schema.field(column).type
        if actual_type != expected_type:
            label = "state" if column == "observation.state" else column
            raise GrootConfigError(
                f"GR00T Parquet {label} fixed shape/float32 type does not match info.features"
            )
        vectors = table[column].to_pylist()
        if table[column].null_count or any(
            not isinstance(vector, list)
            or any(
                not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector
            )
            for vector in vectors
        ):
            label = "state" if column == "observation.state" else column
            raise GrootConfigError(f"GR00T Parquet {label} values must be finite and non-null")
    scalar_types = {
        "task_index": arrow.int64(),
        "episode_index": arrow.int64(),
        "frame_index": arrow.int64(),
        "index": arrow.int64(),
        "timestamp": arrow.float32(),
        language_column: arrow.int64(),
    }
    for column, expected_type in scalar_types.items():
        if schema.field(column).type != expected_type or table[column].null_count:
            label = "language" if column == language_column else column
            raise GrootConfigError(
                f"GR00T Parquet {label} column has the wrong type or null values"
            )
    _require_column_values(
        table["episode_index"].to_pylist(), [episode_index] * expected_rows, "episode_index"
    )
    _require_column_values(
        table["frame_index"].to_pylist(), list(range(expected_rows)), "frame_index"
    )
    _require_column_values(
        table["index"].to_pylist(),
        list(range(global_index, global_index + expected_rows)),
        "index",
    )
    task_indices = table["task_index"].to_pylist()
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in task_map
        or task_map[value] not in episode_tasks
        for value in task_indices
    ):
        raise GrootConfigError("GR00T Parquet task_index values do not match episode task metadata")
    timestamps = table["timestamp"].to_pylist()
    if any(
        not isinstance(value, (int, float))
        or not math.isclose(value, frame / fps, rel_tol=1e-5, abs_tol=1e-6)
        for frame, value in enumerate(timestamps)
    ):
        raise GrootConfigError("GR00T Parquet timestamp values must be contiguous at info.json fps")
    languages = table[language_column].to_pylist()
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in task_map
        or task_map[value] not in episode_tasks
        for value in languages
    ):
        raise GrootConfigError("GR00T Parquet language task indices must match episode metadata")


def _require_column_values(actual: list[object], expected: list[int], name: str) -> None:
    if actual != expected:
        raise GrootConfigError(f"GR00T Parquet {name} values must be contiguous and metadata-bound")


@dataclass(frozen=True, slots=True)
class _VideoProbe:
    codec: str
    width: int
    height: int
    frame_count: int
    fps: float


def _probe_mp4(path: Path) -> _VideoProbe:
    try:
        data = _read_mp4_moov(path)
        moov = (b"moov", 0, len(data))
        for track in _mp4_children(data, moov, b"trak"):
            mdia = _one_mp4_box(_mp4_children(data, track), b"mdia")
            hdlr = _one_mp4_box(_mp4_children(data, mdia), b"hdlr")
            if data[hdlr[1] + 8 : hdlr[1] + 12] != b"vide":
                continue
            mdhd = _one_mp4_box(_mp4_children(data, mdia), b"mdhd")
            timescale = _mp4_timescale(data, mdhd)
            minf = _one_mp4_box(_mp4_children(data, mdia), b"minf")
            stbl = _one_mp4_box(_mp4_children(data, minf), b"stbl")
            codec, width, height = _mp4_sample_description(
                data,
                _one_mp4_box(_mp4_children(data, stbl), b"stsd"),
            )
            sample_count = _mp4_sample_count(
                data,
                _one_mp4_box(_mp4_children(data, stbl), b"stsz"),
            )
            timed_samples, duration = _mp4_timing(
                data,
                _one_mp4_box(_mp4_children(data, stbl), b"stts"),
            )
            if sample_count < 1 or timed_samples != sample_count or duration < 1:
                raise ValueError("invalid video sample table")
            probe = _VideoProbe(
                codec=codec,
                width=width,
                height=height,
                frame_count=sample_count,
                fps=sample_count * timescale / duration,
            )
            _decode_video(path)
            return probe
    except GrootConfigError:
        raise
    except (OSError, ValueError, IndexError, ZeroDivisionError) as error:
        raise GrootConfigError(f"invalid MP4 video stream: {path}") from error
    raise GrootConfigError(f"MP4 video stream has no decodable video track: {path}")


def _read_mp4_moov(path: Path) -> bytes:
    file_size = path.stat().st_size
    cursor = 0
    moov_payload: bytes | None = None
    mdat_count = 0
    with path.open("rb") as stream:
        while cursor < file_size:
            if file_size - cursor < 8:
                raise ValueError("trailing bytes outside MP4 boxes")
            stream.seek(cursor)
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError("truncated MP4 box header")
            size = int.from_bytes(header[:4], "big")
            box_type = header[4:]
            header_size = 8
            if size == 1:
                extended = stream.read(8)
                if len(extended) != 8:
                    raise ValueError("truncated extended MP4 box")
                size = int.from_bytes(extended, "big")
                header_size = 16
            elif size == 0:
                size = file_size - cursor
            if size < header_size or cursor + size > file_size:
                raise ValueError("invalid MP4 box size")
            payload_size = size - header_size
            if box_type == b"moov":
                if moov_payload is not None:
                    raise ValueError("expected one moov box")
                if payload_size > 64 * 1024 * 1024:
                    raise ValueError("MP4 metadata exceeds safe bounds")
                moov_payload = stream.read(payload_size)
                if len(moov_payload) != payload_size:
                    raise ValueError("truncated moov box")
            elif box_type == b"mdat":
                if payload_size < 1:
                    raise ValueError("empty media data")
                mdat_count += 1
            cursor += size
    if moov_payload is None or mdat_count != 1:
        raise ValueError("expected one moov and one mdat box")
    return moov_payload


def _decode_video(path: Path) -> None:
    try:
        executable = _packaged_ffmpeg_executable()
        result = subprocess.run(
            (
                executable,
                "-v",
                "error",
                "-xerror",
                "-nostdin",
                "-protocol_whitelist",
                "file,pipe",
                "-i",
                str(path),
                "-f",
                "null",
                "-",
            ),
            check=False,
            capture_output=True,
            env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GrootConfigError(f"GR00T video decode failed: {path}") from error
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        reason = detail[-1] if detail else "ffmpeg returned an error"
        raise GrootConfigError(f"GR00T video decode failed for {path}: {reason}")


def _packaged_ffmpeg_executable() -> str:
    package_root = Path(cast(str, imageio_ffmpeg.__file__)).resolve().parent
    candidates = tuple(
        path
        for path in (package_root / "binaries").iterdir()
        if path.is_file() and path.name.startswith("ffmpeg-")
    )
    if len(candidates) != 1:
        raise GrootConfigError("imageio-ffmpeg must provide exactly one packaged FFmpeg binary")
    return str(candidates[0])


def _mp4_boxes(data: bytes, start: int, end: int) -> tuple[tuple[bytes, int, int], ...]:
    boxes: list[tuple[bytes, int, int]] = []
    cursor = start
    while cursor + 8 <= end:
        size = int.from_bytes(data[cursor : cursor + 4], "big")
        box_type = data[cursor + 4 : cursor + 8]
        header_size = 8
        if size == 1:
            if cursor + 16 > end:
                raise ValueError("truncated extended MP4 box")
            size = int.from_bytes(data[cursor + 8 : cursor + 16], "big")
            header_size = 16
        elif size == 0:
            size = end - cursor
        if size < header_size or cursor + size > end:
            raise ValueError("invalid MP4 box size")
        boxes.append((box_type, cursor + header_size, cursor + size))
        cursor += size
    if cursor != end:
        raise ValueError("trailing bytes outside MP4 boxes")
    return tuple(boxes)


def _mp4_children(
    data: bytes,
    parent: tuple[bytes, int, int],
    box_type: bytes | None = None,
) -> tuple[tuple[bytes, int, int], ...]:
    children = _mp4_boxes(data, parent[1], parent[2])
    return tuple(box for box in children if box_type is None or box[0] == box_type)


def _one_mp4_box(
    boxes: tuple[tuple[bytes, int, int], ...],
    box_type: bytes,
) -> tuple[bytes, int, int]:
    matches = [box for box in boxes if box[0] == box_type]
    if len(matches) != 1:
        raise ValueError(f"expected one {box_type!r} box")
    return matches[0]


def _mp4_timescale(data: bytes, mdhd: tuple[bytes, int, int]) -> int:
    payload = mdhd[1]
    version = data[payload]
    offset = 20 if version == 1 else 12
    timescale = int.from_bytes(data[payload + offset : payload + offset + 4], "big")
    if timescale < 1:
        raise ValueError("invalid MP4 timescale")
    return timescale


def _mp4_sample_description(
    data: bytes,
    stsd: tuple[bytes, int, int],
) -> tuple[str, int, int]:
    entry_count = int.from_bytes(data[stsd[1] + 4 : stsd[1] + 8], "big")
    entry_start = stsd[1] + 8
    if entry_count != 1 or entry_start + 36 > stsd[2]:
        raise ValueError("invalid MP4 video sample description")
    entry_size = int.from_bytes(data[entry_start : entry_start + 4], "big")
    codec_bytes = data[entry_start + 4 : entry_start + 8]
    if entry_size < 36 or entry_start + entry_size > stsd[2]:
        raise ValueError("invalid MP4 video sample entry")
    if codec_bytes not in {b"avc1", b"avc3", b"hvc1", b"hev1", b"mp4v", b"av01"}:
        raise ValueError("unsupported MP4 video codec")
    width = int.from_bytes(data[entry_start + 32 : entry_start + 34], "big")
    height = int.from_bytes(data[entry_start + 34 : entry_start + 36], "big")
    if width < 1 or height < 1:
        raise ValueError("invalid MP4 video dimensions")
    return codec_bytes.decode("ascii"), width, height


def _mp4_sample_count(data: bytes, stsz: tuple[bytes, int, int]) -> int:
    if stsz[1] + 12 > stsz[2]:
        raise ValueError("truncated MP4 sample-size table")
    return int.from_bytes(data[stsz[1] + 8 : stsz[1] + 12], "big")


def _mp4_timing(data: bytes, stts: tuple[bytes, int, int]) -> tuple[int, int]:
    entry_count = int.from_bytes(data[stts[1] + 4 : stts[1] + 8], "big")
    cursor = stts[1] + 8
    samples = 0
    duration = 0
    for _ in range(entry_count):
        if cursor + 8 > stts[2]:
            raise ValueError("truncated MP4 timing table")
        count = int.from_bytes(data[cursor : cursor + 4], "big")
        delta = int.from_bytes(data[cursor + 4 : cursor + 8], "big")
        samples += count
        duration += count * delta
        cursor += 8
    if cursor != stts[2]:
        raise ValueError("invalid MP4 timing table")
    return samples, duration


def _string(values: dict[str, Any], key: str) -> str:
    value = values[key]
    if not isinstance(value, str) or not value.strip():
        raise GrootConfigError(f"{key} must be a non-empty string")
    return value


def _integer(values: dict[str, Any], key: str) -> int:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise GrootConfigError(f"{key} must be an integer")
    return cast(int, value)


def _integer_tuple(values: dict[str, Any], key: str) -> tuple[int, ...]:
    value = values[key]
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise GrootConfigError(f"{key} must be an integer array")
    return tuple(cast(list[int], value))


def _optional_path(values: dict[str, Any], key: str) -> Path | None:
    value = values.get(key)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise GrootConfigError(f"{key} must be a string when provided")
    return Path(value)

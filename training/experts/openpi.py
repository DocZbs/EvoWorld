"""Method-aligned planning for the pinned openpi experts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tomllib
from typing import Any, cast

from evoworld.contracts import DiagnosisLabel
from training.common.schema import (
    TrainingManifest,
    TrainingPlan,
    TrainingSample,
    TrainingSchemaError,
)


OPENPI_REVISION = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
_SUPPORTED_EXPERTS = frozenset({"pi05", "pi0", "pi0_fast"})
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_BRIDGE_PATH = _REPOSITORY_ROOT / "integrations/openpi/train_bridge.py"
_DEFAULT_CONFIG_PATH = _REPOSITORY_ROOT / "configs/experts/openpi.toml"
_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
_REQUIRED_OPENPI_PATHS = (
    Path("scripts/train.py"),
    Path("scripts/compute_norm_stats.py"),
    Path("src/openpi"),
)
_DROID_FEATURES: dict[str, tuple[str, tuple[int, ...]]] = {
    "exterior_image_1_left": ("image", (180, 320, 3)),
    "exterior_image_2_left": ("image", (180, 320, 3)),
    "wrist_image_left": ("image", (180, 320, 3)),
    "joint_position": ("float32", (7,)),
    "gripper_position": ("float32", (1,)),
    "actions": ("float32", (8,)),
}
RevisionReader = Callable[[Path], str]


class OpenPiConfigError(ValueError):
    """Raised when an openpi configuration is not method-aligned."""


@dataclass(frozen=True, slots=True)
class OpenPiTrainConfig:
    """The EvoWorld overrides applied to an otherwise upstream openpi config."""

    name: str
    upstream_config_name: str
    generated_name: str
    action_dimension: int | None
    horizon: int
    batch_size: int
    max_tokens: int | None
    steps: int
    workers: int
    shuffle: bool
    storage_chunk: str
    action_representation: str

    def __post_init__(self) -> None:
        if self.name not in _SUPPORTED_EXPERTS:
            raise OpenPiConfigError(f"unsupported openpi expert {self.name!r}")
        if not self.upstream_config_name or not self.generated_name:
            raise OpenPiConfigError("upstream and generated config names must not be blank")
        if (self.steps, self.workers, self.shuffle) != (5000, 8, False):
            raise OpenPiConfigError("openpi requires 5,000 steps, 8 workers, and shuffle=false")
        if self.storage_chunk != "chunk-000":
            raise OpenPiConfigError("openpi requires chunk-000 storage")
        if self.action_representation != "joint_velocity":
            raise OpenPiConfigError("openpi requires joint_velocity actions")
        if self.horizon < 1 or self.batch_size < 1:
            raise OpenPiConfigError("horizon and batch_size must be positive")
        expected = {
            "pi05": (32, 15, 16, None),
            "pi0": (None, 10, 8, None),
            "pi0_fast": (8, 10, 8, 180),
        }[self.name]
        actual = (self.action_dimension, self.horizon, self.batch_size, self.max_tokens)
        if actual != expected:
            raise OpenPiConfigError(f"{self.name} settings must be {expected}, got {actual}")


def _git_revision(root: Path) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise OpenPiConfigError(f"unable to read openpi revision from {root}") from error
    return result.stdout.strip()


def validate_openpi_root(
    root: Path,
    *,
    revision_reader: RevisionReader = _git_revision,
) -> Path:
    """Validate the explicit pinned checkout before any official module import."""
    if not root.is_absolute():
        raise OpenPiConfigError("openpi_root must be absolute")
    resolved = root.resolve()
    for relative in _REQUIRED_OPENPI_PATHS:
        if not (resolved / relative).exists():
            raise OpenPiConfigError(f"openpi_root is missing {relative.as_posix()}")
    revision = revision_reader(resolved)
    if revision != OPENPI_REVISION:
        raise OpenPiConfigError(
            f"openpi revision must be {OPENPI_REVISION}, got {revision or '<blank>'}"
        )
    return resolved


def compute_sample_content_sha256(dataset_root: Path, sample: TrainingSample) -> str:
    """Hash every manifest-addressed artifact with its role and relative path."""
    if not dataset_root.is_absolute():
        raise OpenPiConfigError("dataset_root must be absolute when hashing sample content")
    root = dataset_root.resolve()
    action_path = sample.action_path
    paths = (*sample.observation_paths, *((action_path,) if action_path else ()))
    digest = sha256()
    for index, path in enumerate(paths):
        relative = sample.dataset_root / path
        resolved = (root / relative).resolve()
        if resolved != root and root not in resolved.parents:
            raise OpenPiConfigError(f"sample path escapes dataset root: {relative.as_posix()}")
        if not resolved.is_file():
            raise OpenPiConfigError(f"sample artifact does not exist: {relative.as_posix()}")
        digest.update(f"{index}:{relative.as_posix()}\0".encode())
        with resolved.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def load_openpi_manifest(path: Path) -> TrainingManifest:
    """Load the canonical JSON artifact consumed by the openpi child process."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        samples = tuple(
            _sample_from_dict(cast(dict[str, object], values))
            for values in cast(list[object], payload["samples"])
        )
        return TrainingManifest(
            schema_version=cast(int, payload["schema_version"]),
            manifest_id=cast(str, payload["manifest_id"]),
            samples=samples,
        )
    except (OSError, KeyError, TypeError, ValueError, TrainingSchemaError) as error:
        raise OpenPiConfigError(f"invalid openpi manifest artifact {path}") from error


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


def validate_lerobot_dataset(
    dataset_root: Path,
    manifest: TrainingManifest,
    config: OpenPiTrainConfig,
) -> Path:
    """Validate a complete, preconverted LeRobot v2.1 train root and sidecar."""
    if not dataset_root.is_absolute():
        raise OpenPiConfigError("LeRobot dataset root must be absolute")
    root = dataset_root.resolve()
    required = (
        Path("meta/info.json"),
        Path("meta/tasks.jsonl"),
        Path("meta/episodes.jsonl"),
        Path("meta/episodes_stats.jsonl"),
        Path("meta/evoworld_manifest.json"),
        Path("data/chunk-000"),
    )
    for relative in required:
        if not (root / relative).exists():
            raise OpenPiConfigError(f"LeRobot dataset root is missing {relative.as_posix()}")

    info = _read_json(root / "meta/info.json")
    if info.get("codebase_version") != "v2.1" or info.get("data_path") != _DATA_PATH:
        raise OpenPiConfigError("dataset must use pinned LeRobot v2.1 data paths")
    features = info.get("features")
    if not isinstance(features, dict):
        raise OpenPiConfigError("LeRobot info.json features must be an object")
    for key, (dtype, shape) in _DROID_FEATURES.items():
        feature = features.get(key)
        if not isinstance(feature, dict):
            raise OpenPiConfigError(f"DROID dataset is missing feature {key!r}")
        actual = (feature.get("dtype"), tuple(feature.get("shape", ())))
        if actual != (dtype, shape):
            if key == "actions":
                raise OpenPiConfigError(
                    "DROID actions must be 8-dimensional joint-velocity plus gripper"
                )
            raise OpenPiConfigError(f"DROID feature {key!r} must be {(dtype, shape)}, got {actual}")

    tasks = _read_jsonlines(root / "meta/tasks.jsonl")
    episodes = _read_jsonlines(root / "meta/episodes.jsonl")
    episode_stats = _read_jsonlines(root / "meta/episodes_stats.jsonl")
    if not tasks or len(episodes) != len(manifest.samples) or len(episode_stats) != len(episodes):
        raise OpenPiConfigError("LeRobot task, episode, and episode-stat metadata are inconsistent")
    if info.get("total_episodes") != len(episodes):
        raise OpenPiConfigError("LeRobot total_episodes does not match episodes.jsonl")
    episode_lengths = [item.get("length") for item in episodes]
    if not all(isinstance(length, int) for length in episode_lengths) or info.get(
        "total_frames"
    ) != sum(cast(list[int], episode_lengths)):
        raise OpenPiConfigError("LeRobot total_frames does not match episode rows")
    if [item.get("episode_index") for item in episode_stats] != list(range(len(episodes))):
        raise OpenPiConfigError("LeRobot episode stats do not match train-only episodes")
    task_values = {item.get("task") for item in tasks}
    for index, (sample, episode) in enumerate(zip(manifest.samples, episodes, strict=True)):
        episode_tasks = episode.get("tasks")
        if not isinstance(episode_tasks, list) or (
            sample.instruction not in task_values or sample.instruction not in episode_tasks
        ):
            raise OpenPiConfigError("manifest instruction is not bound to LeRobot task metadata")
        if (
            episode.get("episode_index") != index
            or cast(int, episode.get("length", 0)) < config.horizon
        ):
            raise OpenPiConfigError("LeRobot episode does not satisfy the expert action horizon")
        episode_path = root / _DATA_PATH.format(episode_chunk=index // 1000, episode_index=index)
        if not episode_path.is_file() or episode_path.parent.name != config.storage_chunk:
            raise OpenPiConfigError("LeRobot episodes must be stored under data/chunk-000")
        if episode.get("sample_id") != sample.sample_id:
            raise OpenPiConfigError("LeRobot episode sample_id does not match the train manifest")

    _validate_manifest(manifest, config)
    sidecar = load_openpi_manifest(root / "meta/evoworld_manifest.json")
    if sidecar != manifest:
        raise OpenPiConfigError(
            "preconverted LeRobot train root does not match its manifest sidecar"
        )
    for sample in manifest.samples:
        if compute_sample_content_sha256(root, sample) != sample.content_sha256:
            raise OpenPiConfigError(f"sample content provenance changed for {sample.sample_id}")
    return root


def _read_json(path: Path) -> dict[str, object]:
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OpenPiConfigError(f"invalid JSON metadata {path}") from error
    if not isinstance(values, dict):
        raise OpenPiConfigError(f"metadata must be an object: {path}")
    return cast(dict[str, object], values)


def _read_jsonlines(path: Path) -> tuple[dict[str, object], ...]:
    try:
        values = tuple(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, json.JSONDecodeError) as error:
        raise OpenPiConfigError(f"invalid JSONL metadata {path}") from error
    if not all(isinstance(item, dict) for item in values):
        raise OpenPiConfigError(f"JSONL metadata must contain objects: {path}")
    return cast(tuple[dict[str, object], ...], values)


def load_openpi_config(path: Path, name: str) -> OpenPiTrainConfig:
    """Load one explicit, method-aligned expert configuration from TOML."""
    try:
        values = tomllib.loads(path.read_text(encoding="utf-8"))[name]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as error:
        raise OpenPiConfigError(
            f"unable to load openpi configuration {name!r} from {path}"
        ) from error
    if not isinstance(values, dict):
        raise OpenPiConfigError(f"openpi configuration {name!r} must be a TOML table")
    try:
        return OpenPiTrainConfig(
            name=name,
            upstream_config_name=_string(values, "upstream_config_name"),
            generated_name=_string(values, "generated_name"),
            action_dimension=_optional_int(values, "action_dimension"),
            horizon=_integer(values, "horizon"),
            batch_size=_integer(values, "batch_size"),
            max_tokens=_optional_int(values, "max_tokens"),
            steps=_integer(values, "steps"),
            workers=_integer(values, "workers"),
            shuffle=_boolean(values, "shuffle"),
            storage_chunk=_string(values, "storage_chunk"),
            action_representation=_string(values, "action_representation"),
        )
    except KeyError as error:
        raise OpenPiConfigError(
            f"openpi configuration {name!r} is missing {error.args[0]!r}"
        ) from error


def plan_openpi_training(
    manifest: TrainingManifest,
    config: OpenPiTrainConfig,
    *,
    openpi_root: Path,
    output_root: Path,
    dataset_root: Path,
    manifest_path: Path,
    config_path: Path = _DEFAULT_CONFIG_PATH,
    revision_reader: RevisionReader = _git_revision,
) -> TrainingPlan:
    """Create a launch-safe bridge plan bound to one immutable curated manifest."""
    paths = (openpi_root, output_root, dataset_root, manifest_path, config_path)
    if not all(path.is_absolute() for path in paths):
        raise OpenPiConfigError(
            "openpi, output, dataset, manifest, and config paths must be absolute"
        )
    openpi_root = validate_openpi_root(openpi_root, revision_reader=revision_reader)
    if load_openpi_manifest(manifest_path) != manifest:
        raise OpenPiConfigError("manifest_path does not contain the planned immutable manifest")
    validate_lerobot_dataset(dataset_root, manifest, config)
    job_id = f"openpi-{config.name}-{manifest.sha256[:12]}"
    job_output = output_root / job_id
    try:
        return TrainingPlan(
            schema_version=1,
            job_id=job_id,
            target=config.name,
            manifest_sha256=manifest.sha256,
            upstream_revision=OPENPI_REVISION,
            cwd=openpi_root,
            argv=(
                "uv",
                "run",
                "--project",
                str(openpi_root),
                "python",
                str(_BRIDGE_PATH),
                "train",
                "--openpi-root",
                str(openpi_root),
                "--config-path",
                str(config_path),
                "--expert",
                config.name,
                "--manifest-path",
                str(manifest_path),
                "--manifest-sha256",
                manifest.sha256,
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(job_output),
                "--job-id",
                job_id,
            ),
            environment={},
            expected_outputs=(
                Path(f"checkpoints/{config.generated_name}/{job_id}/{config.steps - 1}"),
                Path(f"assets/{config.generated_name}/evoworld_chunk_000/norm_stats.json"),
            ),
        )
    except TrainingSchemaError as error:
        raise OpenPiConfigError("unable to create a safe openpi training plan") from error


def _validate_manifest(manifest: TrainingManifest, config: OpenPiTrainConfig) -> None:
    for sample in manifest.samples:
        if sample.diagnosis is not DiagnosisLabel.PF or sample.split != "train":
            raise OpenPiConfigError("openpi manifests must contain only PF train samples")
        if sample.target != config.name:
            raise OpenPiConfigError("openpi manifest samples must target one selected expert")
        if sample.dataset_root != Path("data/chunk-000"):
            raise OpenPiConfigError("openpi samples must use chunk-000 storage")
        if sample.action_schema != config.action_representation:
            raise OpenPiConfigError("openpi samples must use joint_velocity actions")
        if sample.action_dimension != 8:
            raise OpenPiConfigError(
                "openpi samples must contain raw 8D DROID joint-velocity actions"
            )
        if sample.action_horizon != config.horizon:
            raise OpenPiConfigError("sample action horizon does not match selected openpi expert")


def _string(values: dict[str, Any], key: str) -> str:
    value = values[key]
    if not isinstance(value, str):
        raise OpenPiConfigError(f"{key} must be a string")
    return value


def _integer(values: dict[str, Any], key: str) -> int:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise OpenPiConfigError(f"{key} must be an integer")
    return cast(int, value)


def _optional_int(values: dict[str, Any], key: str) -> int | None:
    value = values.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise OpenPiConfigError(f"{key} must be an integer or null")
    return cast(int, value)


def _boolean(values: dict[str, Any], key: str) -> bool:
    value = values[key]
    if not isinstance(value, bool):
        raise OpenPiConfigError(f"{key} must be a boolean")
    return value

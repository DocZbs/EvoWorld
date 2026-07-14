"""Executable openpi fine-tuning bridge with method-aligned data loading."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import functools
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Protocol, cast

# An absolute script launch sets sys.path to ``integrations/openpi``. Add only
# this repository root before importing the lightweight planner package.
if __package__ in {None, ""}:
    _REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from training.common.schema import TrainingManifest
from training.experts.openpi import (
    OPENPI_REVISION,
    OpenPiTrainConfig,
    RevisionReader,
    load_openpi_config,
    load_openpi_manifest,
    validate_lerobot_dataset,
    validate_openpi_root,
)


_NORM_ASSET_ID = "evoworld_chunk_000"


class OpenPiBridgeError(RuntimeError):
    """Raised when the pinned openpi bridge cannot execute safely."""


@dataclass(frozen=True, slots=True)
class OpenPiRunResult:
    """Auditable output locations produced by the two-stage training bridge."""

    normalization_dir: Path
    final_checkpoint: Path


class OpenPiUpstream(Protocol):
    """Official-component boundary used by the process-local bridge."""

    def create_config(
        self,
        config: OpenPiTrainConfig,
        *,
        dataset_root: Path,
        output_dir: Path,
        job_id: str,
    ) -> object: ...

    def validate_dataset(
        self,
        dataset_root: Path,
        manifest: TrainingManifest,
        config: OpenPiTrainConfig,
    ) -> None: ...

    def compute_normalization(self, train_config: object, *, num_workers: int) -> Path: ...

    def run_training(
        self,
        train_config: object,
        *,
        shuffle: bool,
        num_workers: int,
        steps: int,
    ) -> Path: ...


def train(
    config: OpenPiTrainConfig,
    *,
    manifest: TrainingManifest,
    dataset_root: Path,
    output_dir: Path,
    job_id: str,
    upstream: OpenPiUpstream | None = None,
) -> OpenPiRunResult:
    """Compute normalization and run the exact training budget with one config recipe."""
    _validate_runtime_inputs(dataset_root, output_dir, job_id)
    validate_lerobot_dataset(dataset_root, manifest, config)
    bridge = upstream or OfficialOpenPiUpstream()
    bridge.validate_dataset(dataset_root, manifest, config)

    normalization_config = bridge.create_config(
        config,
        dataset_root=dataset_root,
        output_dir=output_dir,
        job_id=job_id,
    )
    normalization_dir = bridge.compute_normalization(
        normalization_config,
        num_workers=config.workers,
    )

    # Recreate the config so the official DataConfigFactory loads the just-written stats.
    train_config = bridge.create_config(
        config,
        dataset_root=dataset_root,
        output_dir=output_dir,
        job_id=job_id,
    )
    final_checkpoint = bridge.run_training(
        train_config,
        shuffle=config.shuffle,
        num_workers=config.workers,
        steps=config.steps,
    )
    return OpenPiRunResult(
        normalization_dir=normalization_dir,
        final_checkpoint=final_checkpoint,
    )


def normalize(
    config: OpenPiTrainConfig,
    *,
    manifest: TrainingManifest,
    dataset_root: Path,
    output_dir: Path,
    job_id: str,
    upstream: OpenPiUpstream | None = None,
) -> Path:
    """Run the independently executable normalization stage for a generated config."""
    _validate_runtime_inputs(dataset_root, output_dir, job_id)
    validate_lerobot_dataset(dataset_root, manifest, config)
    bridge = upstream or OfficialOpenPiUpstream()
    bridge.validate_dataset(dataset_root, manifest, config)
    train_config = bridge.create_config(
        config,
        dataset_root=dataset_root,
        output_dir=output_dir,
        job_id=job_id,
    )
    return bridge.compute_normalization(train_config, num_workers=config.workers)


ModuleLoader = Callable[[str], Any]


class OfficialOpenPiUpstream:
    """Compose the pinned public config, helpers, loader, sharding, and checkpoints.

    Imports stay delayed until this bridge runs inside the pinned openpi environment.
    The loop calls ``scripts.train.init_train_state`` and ``scripts.train.train_step``
    but deliberately never calls the upstream ``main``, whose loader forces shuffle.
    """

    def __init__(self, *, module_loader: ModuleLoader = importlib.import_module) -> None:
        self._module_loader = module_loader

    def verify_imports(self) -> None:
        """Reach every pinned script/config entry point used by this bridge."""
        self._module_loader("openpi.training.config")
        self._module_loader("scripts.train")
        self._module_loader("scripts.compute_norm_stats")
        self._module_loader("lerobot.common.datasets.lerobot_dataset")
        self._module_loader("pyarrow.parquet")

    def validate_dataset(
        self,
        dataset_root: Path,
        manifest: TrainingManifest,
        config: OpenPiTrainConfig,
    ) -> None:
        """Open the local root with pinned LeRobot and inspect real Arrow rows."""
        lerobot_dataset = self._module_loader("lerobot.common.datasets.lerobot_dataset")
        parquet = self._module_loader("pyarrow.parquet")
        numpy = self._module_loader("numpy")

        metadata = lerobot_dataset.LeRobotDatasetMetadata(
            repo_id="evoworld/local",
            root=dataset_root,
            revision="v2.1",
        )
        expected_episodes = len(manifest.samples)
        if (
            metadata.total_episodes != expected_episodes
            or len(metadata.episodes) != expected_episodes
            or len(metadata.episodes_stats) != expected_episodes
            or metadata.info.get("splits") != {"train": f"0:{expected_episodes}"}
            or metadata.info.get("total_tasks") != len(metadata.tasks)
        ):
            raise OpenPiBridgeError(
                "pinned LeRobot metadata is not the exact train-only manifest root"
            )
        for index, sample_spec in enumerate(manifest.samples):
            episode = metadata.episodes.get(index)
            if episode is None or episode.get("sample_id") != sample_spec.sample_id:
                raise OpenPiBridgeError(
                    "pinned LeRobot episodes are not one-to-one with manifest sample IDs"
                )
        episode_paths = [
            dataset_root / metadata.get_data_file_path(index)
            for index in range(metadata.total_episodes)
        ]
        required_columns = {
            "exterior_image_1_left",
            "exterior_image_2_left",
            "wrist_image_left",
            "joint_position",
            "gripper_position",
            "actions",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        }
        total_rows = 0
        global_indices: list[int] = []
        for episode_index, episode_path in enumerate(episode_paths):
            table = parquet.read_table(episode_path)
            if not required_columns.issubset(set(table.column_names)):
                missing = sorted(required_columns - set(table.column_names))
                raise OpenPiBridgeError(f"LeRobot Arrow episode is missing columns: {missing}")
            rows = int(table.num_rows)
            if rows != metadata.episodes[episode_index]["length"]:
                raise OpenPiBridgeError("Arrow row count differs from episodes.jsonl")
            total_rows += rows

            actions = numpy.asarray(table.column("actions").to_pylist())
            joints = numpy.asarray(table.column("joint_position").to_pylist())
            gripper = numpy.asarray(table.column("gripper_position").to_pylist())
            timestamps = numpy.asarray(table.column("timestamp").to_pylist())
            action_type = str(table.schema.field("actions").type)
            joint_type = str(table.schema.field("joint_position").type)
            gripper_type = str(table.schema.field("gripper_position").type)
            timestamp_type = str(table.schema.field("timestamp").type)
            if actions.shape != (rows, 8) or action_type != "fixed_size_list<element: float>[8]":
                raise OpenPiBridgeError("Arrow actions must be float32 rows with dimension 8")
            if joints.shape != (rows, 7) or gripper.shape != (rows, 1):
                raise OpenPiBridgeError("Arrow DROID joint/gripper shapes are invalid")
            if (
                joint_type != "fixed_size_list<element: float>[7]"
                or gripper_type != "fixed_size_list<element: float>[1]"
            ):
                raise OpenPiBridgeError("Arrow DROID state values must be float32")
            if timestamps.shape != (rows,) or timestamp_type != "float":
                raise OpenPiBridgeError("Arrow timestamps must be scalar float32 values")
            for image_key in (
                "exterior_image_1_left",
                "exterior_image_2_left",
                "wrist_image_left",
            ):
                if any(value is None for value in table.column(image_key).to_pylist()):
                    raise OpenPiBridgeError(
                        f"Arrow image reference {image_key!r} contains null rows"
                    )

            frame_indices = table.column("frame_index").to_pylist()
            episode_indices = table.column("episode_index").to_pylist()
            indices = table.column("index").to_pylist()
            task_indices = table.column("task_index").to_pylist()
            if frame_indices != list(range(rows)) or episode_indices != [episode_index] * rows:
                raise OpenPiBridgeError("Arrow frame/episode indices are not canonical")
            if any(index not in metadata.tasks for index in task_indices):
                raise OpenPiBridgeError("Arrow task_index is absent from tasks.jsonl")
            global_indices.extend(cast(list[int], indices))

        if total_rows != metadata.total_frames or global_indices != list(range(total_rows)):
            raise OpenPiBridgeError("Arrow rows do not match LeRobot global metadata")

        dataset = lerobot_dataset.LeRobotDataset(
            repo_id="evoworld/local",
            root=dataset_root,
            revision="v2.1",
            delta_timestamps={
                "actions": [step / metadata.fps for step in range(config.horizon)],
            },
            download_videos=False,
        )
        if len(dataset) != metadata.total_frames:
            raise OpenPiBridgeError("pinned LeRobot dataset length differs from metadata")
        sample = dataset[0]
        if tuple(numpy.asarray(sample["actions"]).shape) != (config.horizon, 8):
            raise OpenPiBridgeError("pinned LeRobot did not produce the configured action horizon")
        if tuple(numpy.asarray(sample["joint_position"]).shape)[-1:] != (7,):
            raise OpenPiBridgeError("pinned LeRobot sample has invalid joint state")
        if tuple(numpy.asarray(sample["gripper_position"]).shape)[-1:] != (1,):
            raise OpenPiBridgeError("pinned LeRobot sample has invalid gripper state")
        for image_key in (
            "exterior_image_1_left",
            "exterior_image_2_left",
            "wrist_image_left",
        ):
            if numpy.asarray(sample[image_key]).ndim != 3:
                raise OpenPiBridgeError(f"pinned LeRobot sample has invalid image {image_key!r}")

    def create_config(
        self,
        config: OpenPiTrainConfig,
        *,
        dataset_root: Path,
        output_dir: Path,
        job_id: str,
    ) -> object:
        training_config = self._module_loader("openpi.training.config")
        base = training_config.get_config(config.upstream_config_name)

        model_updates: dict[str, object] = {"action_horizon": config.horizon}
        if config.action_dimension is not None:
            model_updates["action_dim"] = config.action_dimension
        if config.max_tokens is not None:
            model_updates["max_token_len"] = config.max_tokens
        model = replace(base.model, **model_updates)

        # LeRobotDROIDDataConfig is the pinned official joint-velocity facade. It
        # replaces the Libero factories used by the pi0 and pi0-FAST base recipes.
        data = training_config.LeRobotDROIDDataConfig(
            repo_id=str(dataset_root),
            assets=training_config.AssetsConfig(asset_id=_NORM_ASSET_ID),
            base_config=training_config.DataConfig(prompt_from_task=True),
        )
        return replace(
            base,
            name=config.generated_name,
            exp_name=job_id,
            model=model,
            data=data,
            batch_size=config.batch_size,
            num_workers=config.workers,
            num_train_steps=config.steps,
            assets_base_dir=str(output_dir / "assets"),
            checkpoint_base_dir=str(output_dir / "checkpoints"),
        )

    def compute_normalization(self, train_config: object, *, num_workers: int) -> Path:
        config = cast(Any, train_config)
        norm_script = self._module_loader("scripts.compute_norm_stats")
        normalize_module = self._module_loader("openpi.shared.normalize")
        numpy = self._module_loader("numpy")

        data_config = config.data.create(config.assets_dirs, config.model)
        if data_config.rlds_data_dir is not None:
            data_loader, num_batches = norm_script.create_rlds_dataloader(
                data_config,
                config.model.action_horizon,
                config.batch_size,
            )
        else:
            data_loader, num_batches = norm_script.create_torch_dataloader(
                data_config,
                config.model.action_horizon,
                config.batch_size,
                config.model,
                num_workers,
            )
        del num_batches

        stats = {key: normalize_module.RunningStats() for key in ("state", "actions")}
        for batch in data_loader:
            for key, running in stats.items():
                running.update(numpy.asarray(batch[key]))
        norm_stats = {key: running.get_statistics() for key, running in stats.items()}

        asset_id = cast(str | None, data_config.asset_id)
        if not asset_id:
            raise OpenPiBridgeError("official data config did not expose a normalization asset id")
        output_path = Path(config.assets_dirs) / asset_id
        normalize_module.save(output_path, norm_stats)
        return output_path

    def run_training(
        self,
        train_config: object,
        *,
        shuffle: bool,
        num_workers: int,
        steps: int,
    ) -> Path:
        config = cast(Any, train_config)
        if shuffle:
            raise OpenPiBridgeError("method-aligned openpi training requires shuffle=false")
        if (num_workers, steps) != (8, 5000):
            raise OpenPiBridgeError(
                "method-aligned openpi training requires 8 workers and 5,000 steps"
            )
        if (config.num_workers, config.num_train_steps) != (num_workers, steps):
            raise OpenPiBridgeError(
                "official TrainConfig budget differs from the audited bridge budget"
            )

        jax = self._module_loader("jax")
        train_script = self._module_loader("scripts.train")
        checkpoints = self._module_loader("openpi.training.checkpoints")
        data_loader_module = self._module_loader("openpi.training.data_loader")
        sharding = self._module_loader("openpi.training.sharding")

        if config.batch_size % jax.device_count() != 0:
            raise OpenPiBridgeError("global batch size must be divisible by the JAX device count")
        jax.config.update("jax_compilation_cache_dir", str(Path("~/.cache/jax").expanduser()))

        rng = jax.random.key(config.seed)
        train_rng, init_rng = jax.random.split(rng)
        mesh = sharding.make_mesh(config.fsdp_devices)
        data_sharding = jax.sharding.NamedSharding(
            mesh,
            jax.sharding.PartitionSpec(sharding.DATA_AXIS),
        )
        replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        checkpoint_manager, resuming = checkpoints.initialize_checkpoint_dir(
            config.checkpoint_dir,
            keep_period=config.keep_period,
            overwrite=config.overwrite,
            resume=config.resume,
        )
        train_script.init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

        configured = replace(config, num_workers=num_workers)
        data_loader = data_loader_module.create_data_loader(
            configured,
            sharding=data_sharding,
            shuffle=shuffle,
        )
        data_iterator = iter(data_loader)
        batch = next(data_iterator)

        train_state, state_sharding = train_script.init_train_state(
            configured,
            init_rng,
            mesh,
            resume=resuming,
        )
        train_state = jax.block_until_ready(train_state)
        if resuming:
            train_state = checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

        compiled_train_step = jax.jit(
            functools.partial(train_script.train_step, configured),
            in_shardings=(replicated_sharding, state_sharding, data_sharding),
            out_shardings=(state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )
        start_step = int(train_state.step)
        if not 0 <= start_step <= steps:
            raise OpenPiBridgeError(f"invalid restored openpi step {start_step}")

        for step in range(start_step, steps):
            with sharding.set_mesh(mesh):
                train_state, _ = compiled_train_step(train_rng, train_state, batch)
            batch = next(data_iterator)
            if (step % configured.save_interval == 0 and step > start_step) or step == steps - 1:
                checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

        checkpoint_manager.wait_until_finished()
        return Path(config.checkpoint_dir) / str(steps - 1)


def _validate_runtime_inputs(dataset_root: Path, output_dir: Path, job_id: str) -> None:
    if not dataset_root.is_absolute() or not output_dir.is_absolute():
        raise OpenPiBridgeError("dataset_root and output_dir must be absolute")
    if not job_id or any(character.isspace() or character in "/\\" for character in job_id):
        raise OpenPiBridgeError("job_id must be a safe single path segment")


def _prepare_openpi_imports(openpi_root: Path) -> None:
    for path in (openpi_root / "src", openpi_root):
        value = str(path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for operation in ("train", "normalize"):
        command = commands.add_parser(operation)
        command.add_argument("--openpi-root", type=Path, required=True)
        command.add_argument("--config-path", type=Path, required=True)
        command.add_argument("--expert", choices=("pi05", "pi0", "pi0_fast"), required=True)
        command.add_argument("--manifest-path", type=Path, required=True)
        command.add_argument("--manifest-sha256", required=True)
        command.add_argument("--dataset-root", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--job-id", required=True)
        command.add_argument("--dry-run-imports", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    upstream: OpenPiUpstream | None = None,
    revision_reader: RevisionReader | None = None,
) -> int:
    """Execute a normalization-only stage or the audited normalization+train pipeline."""
    arguments = _parser().parse_args(argv)
    openpi_root = cast(Path, arguments.openpi_root)
    config_path = cast(Path, arguments.config_path)
    manifest_path = cast(Path, arguments.manifest_path)
    dataset_root = cast(Path, arguments.dataset_root)
    output_dir = cast(Path, arguments.output_dir)
    if not config_path.is_absolute() or not manifest_path.is_absolute():
        raise OpenPiBridgeError("config_path and manifest_path must be absolute")
    manifest_sha256 = cast(str, arguments.manifest_sha256)
    if len(manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in manifest_sha256
    ):
        raise OpenPiBridgeError("manifest_sha256 must be a lowercase SHA-256 digest")

    root_kwargs = {} if revision_reader is None else {"revision_reader": revision_reader}
    resolved_openpi_root = validate_openpi_root(openpi_root, **root_kwargs)
    _prepare_openpi_imports(resolved_openpi_root)
    config = load_openpi_config(config_path, cast(str, arguments.expert))
    manifest = load_openpi_manifest(manifest_path)
    if manifest.sha256 != manifest_sha256:
        raise OpenPiBridgeError("manifest artifact does not match manifest_sha256")
    validate_lerobot_dataset(dataset_root, manifest, config)

    if cast(bool, arguments.dry_run_imports):
        if upstream is not None:
            raise OpenPiBridgeError("dry-run imports requires the default official facade")
        OfficialOpenPiUpstream().verify_imports()
        print(
            json.dumps(
                {
                    "openpi_revision": OPENPI_REVISION,
                    "openpi_root": str(resolved_openpi_root),
                    "manifest_sha256": manifest.sha256,
                },
                sort_keys=True,
            )
        )
        return 0

    job_id = cast(str, arguments.job_id)
    if arguments.operation == "normalize":
        normalize(
            config,
            manifest=manifest,
            dataset_root=dataset_root,
            output_dir=output_dir,
            job_id=job_id,
            upstream=upstream,
        )
    else:
        train(
            config,
            manifest=manifest,
            dataset_root=dataset_root,
            output_dir=output_dir,
            job_id=job_id,
            upstream=upstream,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

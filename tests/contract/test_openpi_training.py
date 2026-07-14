from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from integrations.openpi.train_bridge import (
    OfficialOpenPiUpstream,
    main,
    train,
)
from training.common.schema import TrainingManifest, TrainingSample
from training.experts.openpi import (
    OPENPI_REVISION,
    OpenPiConfigError,
    compute_sample_content_sha256,
    load_openpi_config,
    plan_openpi_training,
    validate_lerobot_dataset,
    validate_openpi_root,
)
from evoworld.contracts import DiagnosisLabel


ROOT = Path(__file__).parents[2].resolve()
CONFIG_PATH = ROOT / "configs/experts/openpi.toml"
BRIDGE_PATH = ROOT / "integrations/openpi/train_bridge.py"


@pytest.mark.parametrize(
    ("name", "dimension", "horizon", "batch", "tokens"),
    [
        ("pi05", 32, 15, 16, None),
        ("pi0", None, 10, 8, None),
        ("pi0_fast", 8, 10, 8, 180),
    ],
)
def test_openpi_method_settings(
    name: str,
    dimension: int | None,
    horizon: int,
    batch: int,
    tokens: int | None,
) -> None:
    config = load_openpi_config(CONFIG_PATH, name)

    assert (config.action_dimension, config.horizon, config.batch_size, config.max_tokens) == (
        dimension,
        horizon,
        batch,
        tokens,
    )
    assert (config.steps, config.workers, config.shuffle) == (5000, 8, False)


def test_pipeline_includes_normalization_no_shuffle_and_exact_budget(tmp_path: Path) -> None:
    config = load_openpi_config(CONFIG_PATH, "pi05")
    upstream = RecordingUpstream(tmp_path)
    dataset_root, manifest, _ = _write_lerobot_fixture(tmp_path, "pi05")

    result = train(
        config,
        manifest=manifest,
        dataset_root=dataset_root,
        output_dir=(tmp_path / "output").resolve(),
        job_id="openpi-pi05-test",
        upstream=upstream,
    )

    assert upstream.events == ["validate", "config", "normalize", "config", "train"]
    assert upstream.loader_settings == [(False, 8)]
    assert upstream.steps == [5000]
    assert (
        result.final_checkpoint
        == tmp_path / "output/checkpoints/evoworld_pi05/openpi-pi05-test/4999"
    )


class RecordingUpstream:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.events: list[str] = []
        self.loader_settings: list[tuple[bool, int]] = []
        self.steps: list[int] = []

    def validate_dataset(self, dataset_root: Path, manifest: object, config: object) -> None:
        del dataset_root, manifest, config
        self.events.append("validate")

    def create_config(
        self,
        config: object,
        *,
        dataset_root: Path,
        output_dir: Path,
        job_id: str,
    ) -> object:
        self.events.append("config")
        return SimpleNamespace(
            config=config,
            dataset_root=dataset_root,
            output_dir=output_dir,
            checkpoint_dir=output_dir / "checkpoints" / "evoworld_pi05" / job_id,
        )

    def compute_normalization(self, train_config: object, *, num_workers: int) -> Path:
        self.events.append("normalize")
        assert num_workers == 8
        return self.root / "output/assets/evoworld_pi05/evoworld_chunk_000"

    def run_training(
        self,
        train_config: object,
        *,
        shuffle: bool,
        num_workers: int,
        steps: int,
    ) -> Path:
        self.events.append("train")
        self.loader_settings.append((shuffle, num_workers))
        self.steps.append(steps)
        checkpoint_dir = getattr(train_config, "checkpoint_dir")
        return Path(checkpoint_dir) / str(steps - 1)


@dataclass(frozen=True)
class FakeModel:
    action_dim: int
    action_horizon: int
    max_token_len: int = 48


@dataclass(frozen=True)
class FakeDataConfig:
    prompt_from_task: bool = False


@dataclass(frozen=True)
class FakeAssetsConfig:
    asset_id: str | None = None


@dataclass(frozen=True)
class FakeDroidFactory:
    repo_id: str
    assets: FakeAssetsConfig
    base_config: FakeDataConfig

    def create(self, assets_dirs: Path, model: FakeModel) -> object:
        del model
        return SimpleNamespace(
            repo_id=self.repo_id,
            asset_id=self.assets.asset_id,
            norm_stats={"state": object(), "actions": object()},
            rlds_data_dir=None,
            assets_dirs=assets_dirs,
        )


@dataclass(frozen=True)
class FakeTrainConfig:
    name: str
    model: FakeModel
    data: object
    batch_size: int = 32
    num_workers: int = 2
    num_train_steps: int = 30000
    exp_name: str = "base"
    assets_base_dir: str = "./assets"
    checkpoint_base_dir: str = "./checkpoints"
    fsdp_devices: int = 1
    seed: int = 42
    keep_period: int | None = 5000
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = False
    save_interval: int = 1000

    @property
    def assets_dirs(self) -> Path:
        return Path(self.assets_base_dir) / self.name

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.checkpoint_base_dir) / self.name / self.exp_name


class InfiniteLoader:
    def __init__(self, data_config: object) -> None:
        self._data_config = data_config

    def __iter__(self) -> InfiniteLoader:
        return self

    def __next__(self) -> tuple[str, str]:
        return ("observation", "actions")

    def data_config(self) -> object:
        return self._data_config


class FakeCheckpointManager:
    def __init__(self) -> None:
        self.waited = False

    def wait_until_finished(self) -> None:
        self.waited = True


class FakeRunningStats:
    def __init__(self) -> None:
        self.values: list[object] = []

    def update(self, value: object) -> None:
        self.values.append(value)

    def get_statistics(self) -> tuple[object, ...]:
        return tuple(self.values)


class FakeOfficialModules:
    def __init__(self) -> None:
        self.loader_calls: list[tuple[bool, int, object]] = []
        self.saved_steps: list[int] = []
        self.train_steps = 0
        self.normalization_saves: list[tuple[Path, object]] = []
        self.manager = FakeCheckpointManager()
        self.train_config_module = SimpleNamespace(
            get_config=self._get_config,
            LeRobotDROIDDataConfig=FakeDroidFactory,
            DataConfig=FakeDataConfig,
            AssetsConfig=FakeAssetsConfig,
        )
        self.data_loader = SimpleNamespace(create_data_loader=self._create_data_loader)
        self.checkpoints = SimpleNamespace(
            initialize_checkpoint_dir=lambda *args, **kwargs: (self.manager, False),
            save_state=self._save_state,
            restore_state=lambda manager, state, loader: state,
        )
        self.sharding = SimpleNamespace(
            DATA_AXIS="data",
            make_mesh=lambda devices: "mesh",
            set_mesh=lambda mesh: nullcontext(),
        )
        self.jax = SimpleNamespace(
            device_count=lambda: 1,
            process_index=lambda: 0,
            config=SimpleNamespace(update=lambda *args: None),
            random=SimpleNamespace(key=lambda seed: seed, split=lambda key: (key + 1, key + 2)),
            sharding=SimpleNamespace(
                NamedSharding=lambda mesh, spec: (mesh, spec),
                PartitionSpec=lambda *axes: axes,
            ),
            jit=lambda function, **kwargs: function,
            block_until_ready=lambda value: value,
        )
        self.train_script = SimpleNamespace(
            init_wandb=lambda *args, **kwargs: None,
            init_train_state=lambda config, rng, mesh, resume: (SimpleNamespace(step=0), "state"),
            train_step=self._train_step,
        )
        self.norm_script = SimpleNamespace(
            create_torch_dataloader=lambda *args: (
                [{"state": [1.0], "actions": [2.0]}],
                1,
            ),
            create_rlds_dataloader=lambda *args: (_ for _ in ()).throw(AssertionError("not RLDS")),
        )
        self.normalize = SimpleNamespace(
            RunningStats=FakeRunningStats,
            save=lambda path, stats: self.normalization_saves.append((Path(path), stats)),
        )
        self.numpy = np
        self.arrow_table = FakeArrowTable()
        self.parquet = SimpleNamespace(read_table=lambda path: self.arrow_table)
        self.lerobot = SimpleNamespace(
            LeRobotDatasetMetadata=FakeLeRobotMetadata,
            LeRobotDataset=FakeLeRobotDataset,
        )

    def _get_config(self, name: str) -> FakeTrainConfig:
        dimensions = {"pi05_droid_finetune": 32, "pi0_libero": 32, "pi0_fast_libero": 7}
        return FakeTrainConfig(
            name=name,
            model=FakeModel(action_dim=dimensions[name], action_horizon=16),
            data=object(),
        )

    def _create_data_loader(
        self,
        config: FakeTrainConfig,
        *,
        sharding: object,
        shuffle: bool,
    ) -> InfiniteLoader:
        self.loader_calls.append((shuffle, config.num_workers, sharding))
        return InfiniteLoader(config.data.create(config.assets_dirs, config.model))

    def _train_step(
        self,
        config: FakeTrainConfig,
        rng: object,
        state: object,
        batch: object,
    ) -> tuple[object, dict[str, object]]:
        del config, rng, batch
        self.train_steps += 1
        return SimpleNamespace(step=getattr(state, "step") + 1), {}

    def _save_state(self, manager: object, state: object, loader: object, step: int) -> None:
        del manager, state, loader
        self.saved_steps.append(step)


class FakeArrowColumn:
    def __init__(self, values: list[object]) -> None:
        self.values = values

    def to_pylist(self) -> list[object]:
        return self.values


class FakeArrowTable:
    def __init__(self) -> None:
        self.values: dict[str, list[object]] = {
            "exterior_image_1_left": [b"image-1"] * 15,
            "exterior_image_2_left": [b"image-2"] * 15,
            "wrist_image_left": [b"image-3"] * 15,
            "joint_position": [[0.0] * 7] * 15,
            "gripper_position": [[0.0]] * 15,
            "actions": [[0.0] * 8] * 15,
            "timestamp": [np.float32(index / 15) for index in range(15)],
            "frame_index": list(range(15)),
            "episode_index": [0] * 15,
            "index": list(range(15)),
            "task_index": [0] * 15,
        }
        self.column_names = list(self.values)
        self.num_rows = 15
        self.schema = FakeArrowSchema()

    def column(self, name: str) -> FakeArrowColumn:
        return FakeArrowColumn(self.values[name])


class FakeArrowSchema:
    def field(self, name: str) -> object:
        value = {
            "timestamp": "float",
            "actions": "fixed_size_list<element: float>[8]",
            "joint_position": "fixed_size_list<element: float>[7]",
            "gripper_position": "fixed_size_list<element: float>[1]",
        }[name]
        return SimpleNamespace(type=value)


class FakeLeRobotMetadata:
    def __init__(self, repo_id: str, root: Path, revision: str) -> None:
        del repo_id, root, revision
        self.total_episodes = 1
        self.total_frames = 15
        self.fps = 15
        self.episodes = {0: {"length": 15}}
        self.episodes[0]["sample_id"] = "sample-001"
        self.episodes_stats = {0: {}}
        self.tasks = {0: "place the mug on the tray"}
        self.info = {"splits": {"train": "0:1"}, "total_tasks": 1}

    def get_data_file_path(self, index: int) -> Path:
        return Path(f"data/chunk-000/episode_{index:06d}.parquet")


class FakeLeRobotDataset:
    def __init__(self, **kwargs: object) -> None:
        timestamps = kwargs["delta_timestamps"]
        assert isinstance(timestamps, dict)
        self.horizon = len(timestamps["actions"])

    def __len__(self) -> int:
        return 15

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        del index
        return {
            "actions": np.zeros((self.horizon, 8), dtype=np.float32),
            "joint_position": np.zeros(7, dtype=np.float32),
            "gripper_position": np.zeros(1, dtype=np.float32),
            "exterior_image_1_left": np.zeros((3, 8, 8), dtype=np.uint8),
            "exterior_image_2_left": np.zeros((3, 8, 8), dtype=np.uint8),
            "wrist_image_left": np.zeros((3, 8, 8), dtype=np.uint8),
        }


@pytest.mark.parametrize(
    ("expert", "dimension", "horizon", "tokens"),
    [("pi05", 32, 15, 48), ("pi0", 32, 10, 48), ("pi0_fast", 8, 10, 180)],
)
def test_default_official_facade_uses_joint_velocity_droid_data_for_every_expert(
    tmp_path: Path,
    expert: str,
    dimension: int,
    horizon: int,
    tokens: int,
) -> None:
    modules = FakeOfficialModules()
    facade = OfficialOpenPiUpstream(module_loader=lambda name: _fake_module(modules, name))
    dataset_root = (tmp_path / "dataset").resolve()

    official = facade.create_config(
        load_openpi_config(CONFIG_PATH, expert),
        dataset_root=dataset_root,
        output_dir=(tmp_path / "out").resolve(),
        job_id=f"openpi-{expert}-test",
    )

    assert isinstance(official.data, FakeDroidFactory)
    assert official.data.repo_id == str(dataset_root)
    assert official.data.base_config.prompt_from_task is True
    assert (
        official.model.action_dim,
        official.model.action_horizon,
        official.model.max_token_len,
    ) == (
        dimension,
        horizon,
        tokens,
    )


def test_default_official_facade_builds_droid_config_and_runs_final_boundary(
    tmp_path: Path,
) -> None:
    modules = FakeOfficialModules()
    facade = OfficialOpenPiUpstream(module_loader=lambda name: _fake_module(modules, name))
    method = load_openpi_config(CONFIG_PATH, "pi0_fast")
    dataset_root = (tmp_path / "dataset").resolve()
    output_dir = (tmp_path / "out").resolve()

    official = facade.create_config(
        method,
        dataset_root=dataset_root,
        output_dir=output_dir,
        job_id="openpi-pi0-fast-test",
    )
    final = facade.run_training(official, shuffle=False, num_workers=8, steps=5000)

    assert isinstance(official.data, FakeDroidFactory)
    assert official.data.repo_id == str(dataset_root)
    assert official.data.base_config.prompt_from_task is True
    assert (
        official.model.action_dim,
        official.model.action_horizon,
        official.model.max_token_len,
    ) == (
        8,
        10,
        180,
    )
    assert (official.batch_size, official.num_workers, official.num_train_steps) == (8, 8, 5000)
    assert official.exp_name == "openpi-pi0-fast-test"
    assert (
        official.checkpoint_dir == output_dir / "checkpoints/evoworld_pi0_fast/openpi-pi0-fast-test"
    )
    assert modules.loader_calls == [(False, 8, ("mesh", ("data",)))]
    assert modules.train_steps == 5000
    assert modules.saved_steps == [1000, 2000, 3000, 4000, 4999]
    assert modules.manager.waited is True
    assert final == official.checkpoint_dir / "4999"


def test_default_official_facade_executes_normalization_helpers(tmp_path: Path) -> None:
    modules = FakeOfficialModules()
    facade = OfficialOpenPiUpstream(module_loader=lambda name: _fake_module(modules, name))
    official = facade.create_config(
        load_openpi_config(CONFIG_PATH, "pi05"),
        dataset_root=(tmp_path / "dataset").resolve(),
        output_dir=(tmp_path / "out").resolve(),
        job_id="openpi-pi05-test",
    )

    output = facade.compute_normalization(official, num_workers=8)

    expected = Path(official.assets_dirs) / "evoworld_chunk_000"
    assert output == expected
    assert modules.normalization_saves == [
        (expected, {"state": ([1.0],), "actions": ([2.0],)}),
    ]


def test_default_official_facade_opens_arrow_rows_and_lerobot_sample(tmp_path: Path) -> None:
    modules = FakeOfficialModules()
    facade = OfficialOpenPiUpstream(module_loader=lambda name: _fake_module(modules, name))

    facade.validate_dataset(
        (tmp_path / "dataset").resolve(),
        _basic_manifest("pi05"),
        load_openpi_config(CONFIG_PATH, "pi05"),
    )


def _fake_module(modules: FakeOfficialModules, name: str) -> Any:
    return {
        "openpi.training.config": modules.train_config_module,
        "openpi.training.data_loader": modules.data_loader,
        "openpi.training.checkpoints": modules.checkpoints,
        "openpi.training.sharding": modules.sharding,
        "scripts.train": modules.train_script,
        "scripts.compute_norm_stats": modules.norm_script,
        "openpi.shared.normalize": modules.normalize,
        "lerobot.common.datasets.lerobot_dataset": modules.lerobot,
        "pyarrow.parquet": modules.parquet,
        "numpy": modules.numpy,
        "jax": modules.jax,
    }[name]


def test_openpi_root_gate_requires_pinned_layout_and_revision(tmp_path: Path) -> None:
    openpi_root = _write_fake_openpi_root(tmp_path / "openpi")

    assert (
        validate_openpi_root(openpi_root, revision_reader=lambda root: OPENPI_REVISION)
        == openpi_root
    )
    with pytest.raises(OpenPiConfigError, match="revision"):
        validate_openpi_root(openpi_root, revision_reader=lambda root: "0" * 40)
    (openpi_root / "scripts/train.py").unlink()
    with pytest.raises(OpenPiConfigError, match="scripts/train.py"):
        validate_openpi_root(openpi_root, revision_reader=lambda root: OPENPI_REVISION)


def test_lerobot_v21_fixture_binds_manifest_features_and_chunk(tmp_path: Path) -> None:
    config = load_openpi_config(CONFIG_PATH, "pi05")
    dataset_root, manifest, manifest_path = _write_lerobot_fixture(tmp_path, "pi05")

    validated = validate_lerobot_dataset(dataset_root, manifest, config)

    assert validated == dataset_root
    assert json.loads((dataset_root / "meta/info.json").read_text())["data_path"] == (
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    assert json.loads(manifest_path.read_text()) == manifest.to_dict()
    table = pq.read_table(dataset_root / "data/chunk-000/episode_000000.parquet")
    assert str(table.schema.field("actions").type) == "fixed_size_list<element: float>[8]"
    image = table.column("exterior_image_1_left").to_pylist()[0]
    assert image["bytes"].startswith(b"\x89PNG\r\n\x1a\n")


def test_lerobot_validator_rejects_non_droid_action_shape(tmp_path: Path) -> None:
    config = load_openpi_config(CONFIG_PATH, "pi0_fast")
    dataset_root, manifest, _ = _write_lerobot_fixture(tmp_path, "pi0_fast")
    info_path = dataset_root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["actions"]["shape"] = [7]
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(OpenPiConfigError, match="8-dimensional joint-velocity"):
        validate_lerobot_dataset(dataset_root, manifest, config)


@pytest.mark.parametrize(
    ("diagnosis", "split", "target"),
    [
        (DiagnosisLabel.TSM, "train", "pi05"),
        (DiagnosisLabel.WMF, "train", "pi05"),
        (DiagnosisLabel.NONE, "train", "pi05"),
        (DiagnosisLabel.PF, "eval", "pi05"),
        (DiagnosisLabel.PF, "train", "pi0"),
    ],
)
def test_lerobot_root_rejects_non_pf_train_or_wrong_expert(
    tmp_path: Path,
    diagnosis: DiagnosisLabel,
    split: str,
    target: str,
) -> None:
    dataset_root, manifest, _ = _write_lerobot_fixture(tmp_path, "pi05")
    changed = replace(manifest.samples[0], diagnosis=diagnosis, split=split, target=target)
    rejected = TrainingManifest(manifest.schema_version, manifest.manifest_id, (changed,))

    with pytest.raises(OpenPiConfigError, match="PF train|selected expert"):
        validate_lerobot_dataset(dataset_root, rejected, load_openpi_config(CONFIG_PATH, "pi05"))


def test_lerobot_root_rejects_extra_episode_outside_train_manifest(tmp_path: Path) -> None:
    dataset_root, manifest, _ = _write_lerobot_fixture(tmp_path, "pi05")
    episodes_path = dataset_root / "meta/episodes.jsonl"
    episodes_path.write_text(
        episodes_path.read_text(encoding="utf-8")
        + json.dumps(
            {"episode_index": 1, "sample_id": "eval-only", "tasks": ["eval"], "length": 15}
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(OpenPiConfigError, match="inconsistent"):
        validate_lerobot_dataset(dataset_root, manifest, load_openpi_config(CONFIG_PATH, "pi05"))


def test_plan_uses_absolute_bridge_and_audits_all_inputs(tmp_path: Path) -> None:
    config = load_openpi_config(CONFIG_PATH, "pi05")
    dataset_root, manifest, manifest_path = _write_lerobot_fixture(tmp_path, "pi05")
    openpi_root = _write_fake_openpi_root(tmp_path / "openpi")
    output_root = (tmp_path / "outputs").resolve()

    plan = plan_openpi_training(
        manifest,
        config,
        openpi_root=openpi_root,
        output_root=output_root,
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        config_path=CONFIG_PATH,
        revision_reader=lambda root: OPENPI_REVISION,
    )

    assert plan.manifest_sha256 == manifest.sha256
    assert plan.upstream_revision == OPENPI_REVISION
    assert plan.cwd == openpi_root
    assert plan.argv[:6] == (
        "uv",
        "run",
        "--project",
        str(openpi_root),
        "python",
        str(BRIDGE_PATH),
    )
    assert "--with-editable" not in plan.argv
    assert plan.argv[6] == "train"
    for flag, value in (
        ("--openpi-root", str(openpi_root)),
        ("--config-path", str(CONFIG_PATH)),
        ("--expert", "pi05"),
        ("--manifest-path", str(manifest_path)),
        ("--manifest-sha256", manifest.sha256),
        ("--dataset-root", str(dataset_root)),
        ("--output-dir", str(output_root / plan.job_id)),
        ("--job-id", plan.job_id),
    ):
        position = plan.argv.index(flag)
        assert plan.argv[position + 1] == value
    assert plan.expected_outputs == (
        Path(f"checkpoints/{config.generated_name}/{plan.job_id}/4999"),
        Path(f"assets/{config.generated_name}/evoworld_chunk_000/norm_stats.json"),
    )


def test_full_planned_subprocess_imports_from_explicit_openpi_root(tmp_path: Path) -> None:
    config = load_openpi_config(CONFIG_PATH, "pi05")
    dataset_root, manifest, manifest_path = _write_lerobot_fixture(tmp_path, "pi05")
    openpi_root = _write_fake_openpi_root(tmp_path / "openpi", record_imports=True)
    plan = plan_openpi_training(
        manifest,
        config,
        openpi_root=openpi_root,
        output_root=(tmp_path / "outputs").resolve(),
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        config_path=CONFIG_PATH,
        revision_reader=lambda root: OPENPI_REVISION,
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(f"#!/bin/sh\nprintf '%s\\n' '{OPENPI_REVISION}'\n", encoding="utf-8")
    fake_git.chmod(0o755)
    uv_path = shutil.which("uv") or "/Users/zbs/Library/Python/3.9/bin/uv"
    assert Path(uv_path).is_file()
    record_path = tmp_path / "imports.txt"
    environment = dict(os.environ)
    environment["PATH"] = os.pathsep.join(
        (str(Path(uv_path).parent), str(fake_bin), environment["PATH"])
    )
    environment["EVOWORLD_IMPORT_RECORD"] = str(record_path)

    result = subprocess.run(
        [*plan.argv, "--dry-run-imports"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["openpi_revision"] == OPENPI_REVISION
    assert record_path.read_text(encoding="utf-8").splitlines() == [
        "config",
        "train:numpy=1",
        "norm",
    ]


def test_cli_parses_and_validates_preconverted_pipeline_arguments(tmp_path: Path) -> None:
    dataset_root, manifest, manifest_path = _write_lerobot_fixture(tmp_path, "pi05")
    openpi_root = _write_fake_openpi_root(tmp_path / "openpi")
    upstream = RecordingUpstream(tmp_path)
    exit_code = main(
        [
            "train",
            "--openpi-root",
            str(openpi_root),
            "--config-path",
            str(CONFIG_PATH),
            "--expert",
            "pi05",
            "--manifest-path",
            str(manifest_path),
            "--manifest-sha256",
            manifest.sha256,
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str((tmp_path / "output").resolve()),
            "--job-id",
            "openpi-pi05-test",
        ],
        upstream=upstream,
        revision_reader=lambda root: OPENPI_REVISION,
    )
    assert exit_code == 0
    assert upstream.events == ["validate", "config", "normalize", "config", "train"]


def _write_fake_openpi_root(path: Path, *, record_imports: bool = False) -> Path:
    root = path.resolve()
    (root / "scripts").mkdir(parents=True)
    (root / "src/openpi/training").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "fake-openpi"\nversion = "0.0.0"\nrequires-python = ">=3.11"\n'
        'dependencies = ["numpy<2"]\n'
        "[tool.uv]\npackage = false\n",
        encoding="utf-8",
    )
    (root / ".python-version").write_text("3.11\n", encoding="utf-8")
    recorder = (
        "from pathlib import Path\nimport os\n"
        "p = Path(os.environ['EVOWORLD_IMPORT_RECORD'])\n"
        "with p.open('a', encoding='utf-8') as f: f.write(LABEL + '\\n')\n"
        if record_imports
        else ""
    )
    for relative, label in (
        ("src/openpi/training/config.py", "config"),
        ("scripts/train.py", "train"),
        ("scripts/compute_norm_stats.py", "norm"),
    ):
        target = root / relative
        if label == "train" and record_imports:
            target.write_text(
                "from pathlib import Path\nimport os\nimport numpy\n"
                "p = Path(os.environ['EVOWORLD_IMPORT_RECORD'])\n"
                "with p.open('a', encoding='utf-8') as f: "
                "f.write('train:numpy=' + numpy.__version__.split('.')[0] + '\\n')\n",
                encoding="utf-8",
            )
        else:
            target.write_text(f"LABEL = {label!r}\n{recorder}", encoding="utf-8")
    for relative in (
        "src/lerobot/__init__.py",
        "src/lerobot/common/__init__.py",
        "src/lerobot/common/datasets/__init__.py",
        "src/lerobot/common/datasets/lerobot_dataset.py",
        "src/pyarrow/__init__.py",
        "src/pyarrow/parquet.py",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")
    for relative in (
        "src/openpi/__init__.py",
        "src/openpi/training/__init__.py",
        "scripts/__init__.py",
    ):
        (root / relative).write_text("", encoding="utf-8")
    return root


def _write_lerobot_fixture(
    tmp_path: Path,
    target: str,
) -> tuple[Path, TrainingManifest, Path]:
    root = (tmp_path / f"{target}-dataset").resolve()
    episode_path = root / "data/chunk-000/episode_000000.parquet"
    episode_path.parent.mkdir(parents=True)
    horizon = {"pi05": 15, "pi0": 10, "pi0_fast": 10}[target]
    _write_real_arrow_episode(episode_path, horizon)
    sample = _sample(
        target,
        dataset_root=Path("data/chunk-000"),
        observation_paths=(Path("episode_000000.parquet"),),
        action_path=Path("episode_000000.parquet"),
        action_dimension=8,
        action_horizon=horizon,
        content_sha256="0" * 64,
    )
    sample = replace(sample, content_sha256=compute_sample_content_sha256(root, sample))
    manifest = TrainingManifest(schema_version=1, manifest_id="manifest-001", samples=(sample,))
    _write_lerobot_metadata(root, manifest, horizon)
    manifest_path = root / "meta/evoworld_manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), sort_keys=True), encoding="utf-8")
    return root, manifest, manifest_path


def _write_real_arrow_episode(path: Path, rows: int) -> None:
    image_type = pa.struct((pa.field("bytes", pa.binary()), pa.field("path", pa.string())))
    image = {"bytes": _png_bytes(width=320, height=180), "path": None}
    table = pa.table(
        {
            "exterior_image_1_left": pa.array([image] * rows, type=image_type),
            "exterior_image_2_left": pa.array([image] * rows, type=image_type),
            "wrist_image_left": pa.array([image] * rows, type=image_type),
            "joint_position": pa.array([[0.0] * 7] * rows, type=pa.list_(pa.float32(), 7)),
            "gripper_position": pa.array([[0.0]] * rows, type=pa.list_(pa.float32(), 1)),
            "actions": pa.array([[0.0] * 8] * rows, type=pa.list_(pa.float32(), 8)),
            "timestamp": pa.array([index / 15 for index in range(rows)], type=pa.float32()),
            "frame_index": pa.array(range(rows), type=pa.int64()),
            "episode_index": pa.array([0] * rows, type=pa.int64()),
            "index": pa.array(range(rows), type=pa.int64()),
            "task_index": pa.array([0] * rows, type=pa.int64()),
        }
    )
    pq.write_table(table, path)


def _png_bytes(*, width: int, height: int) -> bytes:
    """Create a dependency-free RGB PNG that matches the declared image feature."""

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanlines = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


def _write_lerobot_metadata(root: Path, manifest: TrainingManifest, horizon: int) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "panda",
        "total_episodes": len(manifest.samples),
        "total_frames": horizon * len(manifest.samples),
        "total_tasks": len(manifest.samples),
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 15,
        "splits": {"train": f"0:{len(manifest.samples)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": {
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": [180, 320, 3],
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": [180, 320, 3],
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": [180, 320, 3],
                "names": ["height", "width", "channel"],
            },
            "joint_position": {"dtype": "float32", "shape": [7], "names": ["joint_position"]},
            "gripper_position": {"dtype": "float32", "shape": [1], "names": ["gripper_position"]},
            "actions": {"dtype": "float32", "shape": [8], "names": ["actions"]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    tasks = [
        json.dumps({"task_index": index, "task": sample.instruction})
        for index, sample in enumerate(manifest.samples)
    ]
    episodes = [
        json.dumps(
            {
                "episode_index": index,
                "sample_id": sample.sample_id,
                "tasks": [sample.instruction],
                "length": horizon,
            }
        )
        for index, sample in enumerate(manifest.samples)
    ]
    episode_stats = [
        json.dumps({"episode_index": index, "stats": {}})
        for index, _ in enumerate(manifest.samples)
    ]
    (meta / "tasks.jsonl").write_text("\n".join(tasks) + "\n", encoding="utf-8")
    (meta / "episodes.jsonl").write_text("\n".join(episodes) + "\n", encoding="utf-8")
    (meta / "episodes_stats.jsonl").write_text("\n".join(episode_stats) + "\n", encoding="utf-8")


def _basic_manifest(target: str) -> TrainingManifest:
    sample = _sample(
        target,
        dataset_root=Path("data/chunk-000"),
        observation_paths=(Path("episode_000000.parquet"),),
        action_path=Path("episode_000000.parquet"),
        action_dimension=8,
        action_horizon={"pi05": 15, "pi0": 10, "pi0_fast": 10}[target],
        content_sha256="a" * 64,
    )
    return TrainingManifest(schema_version=1, manifest_id="manifest-basic", samples=(sample,))


def _sample(
    target: str,
    *,
    dataset_root: Path,
    observation_paths: tuple[Path, ...],
    action_path: Path,
    action_dimension: int,
    action_horizon: int,
    content_sha256: str,
) -> TrainingSample:
    return TrainingSample(
        schema_version=1,
        sample_id="sample-001",
        target=target,
        task_id="task-001",
        instruction="place the mug on the tray",
        skill_family="pick_place",
        scene_id="scene-001",
        trajectory_id="trajectory-001",
        dataset_root=dataset_root,
        observation_paths=observation_paths,
        action_path=action_path,
        action_schema="joint_velocity",
        action_dimension=action_dimension,
        action_horizon=action_horizon,
        source_type="real",
        source_checkpoint="checkpoint-a",
        diagnosis=DiagnosisLabel.PF,
        split="train",
        content_sha256=content_sha256,
    )

from __future__ import annotations

import base64
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from evoworld.contracts import DiagnosisLabel
from training.common.schema import TrainingManifest, TrainingSample
from training.experts import groot as groot_module
from training.experts.groot import (
    GROOT_REVISION,
    GrootConfigError,
    load_groot_config,
    plan_groot_open_loop_eval,
    plan_groot_training,
    validate_groot_dataset,
)


ROOT = Path(__file__).parents[2].resolve()
CONFIG_PATH = ROOT / "configs/experts/groot_n16.toml"
_TINY_MP4 = base64.b64decode(
    "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"
)
_TINY_AV1_MP4 = base64.b64decode(
    "AAAAIGZ0eXBpc29tAAACAGlzb21hdjAxaXNvMm1wNDEAAANWbW9vdgAAAGxtdmhkAAAAAAAA"
    "AAAAAAAAAAAD6AAAAhYAAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAA"
    "AAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAoF0cmFrAAAAXHRraGQAAAAD"
    "AAAAAAAAAAAAAAABAAAAAAAAAhYAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAA"
    "AAAAAAAAAAAAAABAAAAAABAAAAAQAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAAAAEAAAIWAAAAAAAB"
    "AAAAAAH5bWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAAA8AAAAIABVxAAAAAAALWhkbHIAAAAAAAAA"
    "AHZpZGUAAAAAAAAAAAAAAABWaWRlb0hhbmRsZXIAAAABpG1pbmYAAAAUdm1oZAAAAAEAAAAAAAAA"
    "AAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAAWRzdGJsAAAArHN0c2QAAAAA"
    "AAAAAQAAAJxhdjAxAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAABAAEABIAAAASAAAAAAAAAABGExh"
    "dmM2MS4xOS4xMDAgbGliYW9tLWF2MQAAAAAAAAAAGP//AAAAGGF2MUOBAAwACgoAAAABn/m18gCA"
    "AAAACmZpZWwBAAAAABBwYXNwAAAAAQAAAAEAAAAUYnRydAAAAAAAABGyAAARsgAAABhzdHRzAAAA"
    "AAAAAAEAAAAQAAACAAAAABRzdHNzAAAAAAAAAAEAAAABAAAAHHN0c2MAAAAAAAAAAQAAAAEAAAAQ"
    "AAAAAQAAAFRzdHN6AAAAAAAAAAAAAAAQAAAAHAAAABMAAAATAAAAEwAAABIAAAASAAAAEgAAABIA"
    "AAATAAAAEgAAABIAAAASAAAAEgAAABIAAAASAAAAEgAAABRzdGNvAAAAAAAAAAEAAAOGAAAAYXVk"
    "dGEAAABZbWV0YQAAAAAAAAAhaGRscgAAAAAAAAAAbWRpcmFwcGwAAAAAAAAAAAAAAAAsaWxzdAAA"
    "ACSpdG9vAAAAHGRhdGEAAAABAAAAAExhdmY2MS43LjEwMAAAAAhmcmVlAAABNm1kYXQKCgAAAAGf"
    "+bXyAIAyDhAAwgAAAoAAAACpjliUMhEwA8CAAABGtwAAAkAAIAB2kDIRMAQBBAAARrcAAAJAACAA"
    "dUAyETAGAgiAAEa4AAACQAAgAHOAMhAwCAQNEABGuYAAAkAAIABzMhAwCggRoEBGu4AAAkAAIAB"
    "xMhAwDBAWMIhGvQAAAkAAIABuMhAwDiAawNFGvoAAAkAAIABuMhEwEcCfURoG5AAAAsAAIAB5kD"
    "IQMBIBH1IjBsKAAAKAACAAbDIQMBQCC+IsBsUAAAKAACAAbDIQMBYEDXI1BsgAAAKAACAAaDIQMB"
    "gIEaI+BsuAAAKAACAAaDIQMBoQFjIXBtCAAAKAACAAaDIQMBwgGsIaBteAAAKAACAAaDIQMB4BH1"
    "IjBuQAAALAACAAcA=="
)


def _argv_value(argv: tuple[str, ...], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _dataset_sha256(
    root: Path,
    registration_path: Path | None = Path("meta/evoworld_modality.py"),
) -> str:
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
    if registration_path is not None and (root / registration_path).is_file():
        artifacts.append(("registration", registration_path))
    artifacts.extend(("parquet", item.relative_to(root)) for item in root.glob("data/**/*.parquet"))
    artifacts.extend(("video", item.relative_to(root)) for item in root.glob("videos/**/*.mp4"))
    seen: set[Path] = set()
    for role, relative in sorted(artifacts, key=lambda value: (value[0], value[1].as_posix())):
        if relative in seen:
            continue
        seen.add(relative)
        digest.update(f"{role}:{relative.as_posix()}\0".encode())
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def _refresh_manifest(
    root: Path,
    manifest: TrainingManifest,
    registration_path: Path | None = Path("meta/evoworld_modality.py"),
) -> TrainingManifest:
    refreshed = replace(
        manifest,
        samples=tuple(
            replace(sample, content_sha256=_dataset_sha256(root, registration_path))
            for sample in manifest.samples
        ),
    )
    _write_json(root / "meta/evoworld_manifest.json", refreshed.to_dict())
    return refreshed


def _registration_source(
    *,
    language_key: str = "annotation.human.task",
    action_stop: int = 16,
    embodiment_tag: str = "NEW_EMBODIMENT",
) -> str:
    return f'''from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig

evoworld_config = {{
    "video": ModalityConfig(delta_indices=[0], modality_keys=["front"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["arm", "gripper"]),
    "action": ModalityConfig(
        delta_indices=list(range(0, {action_stop})),
        modality_keys=["arm", "gripper"],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["{language_key}"]),
}}

register_modality_config(
    evoworld_config,
    embodiment_tag=EmbodimentTag.{embodiment_tag},
)
'''


def _stats_vector(width: int) -> dict[str, list[float]]:
    return {key: [0.0] * width for key in ("mean", "std", "min", "max", "q01", "q99")}


def _relative_stats(width: int) -> dict[str, list[list[float]]]:
    return {
        key: [[0.0] * width for _ in range(16)]
        for key in ("mean", "std", "min", "max", "q01", "q99")
    }


def _write_groot_fixture(tmp_path: Path) -> tuple[Path, Path, Path, TrainingManifest]:
    dataset_root = (tmp_path / "dataset").resolve()
    meta_root = dataset_root / "meta"
    data_root = dataset_root / "data/chunk-000"
    video_root = dataset_root / "videos/chunk-000/observation.images.front"
    meta_root.mkdir(parents=True)
    data_root.mkdir(parents=True)
    video_root.mkdir(parents=True)

    info = {
        "codebase_version": "v2.1",
        "robot_type": "test_robot",
        "total_episodes": 1,
        "total_frames": 16,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": 30,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "action": {"dtype": "float32", "shape": [3], "names": ["a", "b", "g"]},
            "observation.state": {
                "dtype": "float32",
                "shape": [3],
                "names": ["a", "b", "g"],
            },
            "observation.images.front": {
                "dtype": "video",
                "shape": [16, 16, 3],
                "names": ["height", "width", "channels"],
            },
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        },
        "total_videos": 1,
    }
    modality = {
        "state": {"arm": {"start": 0, "end": 2}, "gripper": {"start": 2, "end": 3}},
        "action": {"arm": {"start": 0, "end": 2}, "gripper": {"start": 2, "end": 3}},
        "video": {"front": {"original_key": "observation.images.front"}},
        "annotation": {"human.task": {"original_key": "task_index"}},
    }
    _write_json(meta_root / "info.json", info)
    _write_json(meta_root / "modality.json", modality)
    _write_json(
        meta_root / "stats.json",
        {
            "action": _stats_vector(3),
            "observation.state": _stats_vector(3),
        },
    )
    _write_json(
        meta_root / "relative_stats.json",
        {
            "arm": _relative_stats(2),
            "gripper": _relative_stats(1),
        },
    )
    _write_jsonl(meta_root / "tasks.jsonl", [{"task_index": 0, "task": "pick the cube"}])
    _write_jsonl(
        meta_root / "episodes.jsonl",
        [{"episode_index": 0, "tasks": ["pick the cube"], "length": 16}],
    )
    registration_path = meta_root / "evoworld_modality.py"
    registration_path.write_text(_registration_source(), encoding="utf-8")

    parquet_path = data_root / "episode_000000.parquet"
    vector_type = pa.list_(pa.float32(), 3)
    table = pa.table(
        {
            "observation.state": pa.array([[0.0, 0.0, 0.0]] * 16, type=vector_type),
            "action": pa.array([[0.0, 0.0, 0.0]] * 16, type=vector_type),
            "task_index": pa.array([0] * 16, type=pa.int64()),
            "episode_index": pa.array([0] * 16, type=pa.int64()),
            "frame_index": pa.array(list(range(16)), type=pa.int64()),
            "index": pa.array(list(range(16)), type=pa.int64()),
            "timestamp": pa.array([index / 30 for index in range(16)], type=pa.float32()),
        }
    )
    pq.write_table(table, parquet_path)
    (video_root / "episode_000000.mp4").write_bytes(_TINY_MP4)

    provisional = TrainingSample(
        schema_version=1,
        sample_id="groot-000",
        target="groot_n16",
        task_id="pick",
        instruction="pick the cube",
        skill_family="pick",
        scene_id="scene-0",
        trajectory_id="episode-0",
        dataset_root=Path("data/chunk-000"),
        observation_paths=(Path("episode_000000.parquet"),),
        action_path=Path("episode_000000.parquet"),
        action_schema="relative_joint_position",
        action_dimension=3,
        action_horizon=16,
        source_type="real",
        source_checkpoint="robot-demo-v1",
        diagnosis=DiagnosisLabel.PF,
        split="train",
        content_sha256="0" * 64,
    )
    sample = replace(provisional, content_sha256=_dataset_sha256(dataset_root))
    manifest = TrainingManifest(schema_version=1, manifest_id="groot-train", samples=(sample,))
    manifest_path = meta_root / "evoworld_manifest.json"
    _write_json(manifest_path, manifest.to_dict())

    groot_root = (tmp_path / "isaac_groot").resolve()
    (groot_root / "gr00t/experiment").mkdir(parents=True)
    (groot_root / "gr00t/eval").mkdir(parents=True)
    (groot_root / "gr00t/configs/data").mkdir(parents=True)
    (groot_root / "gr00t/data").mkdir(parents=True)
    (groot_root / "gr00t/experiment/launch_finetune.py").touch()
    (groot_root / "gr00t/eval/open_loop_eval.py").touch()
    (groot_root / "gr00t/data/embodiment_tags.py").write_text(
        "from enum import Enum\n\n"
        "class EmbodimentTag(Enum):\n"
        "    NEW_EMBODIMENT = 'new_embodiment'\n"
        "    LIBERO_PANDA = 'libero_panda'\n",
        encoding="utf-8",
    )
    (groot_root / "gr00t/data/types.py").write_text(
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class ModalityConfig:\n"
        "    delta_indices: list[int]\n"
        "    modality_keys: list[str]\n",
        encoding="utf-8",
    )
    (groot_root / "gr00t/configs/data/embodiment_configs.py").write_text(
        "MODALITY_CONFIGS = {}\n\n"
        "def register_modality_config(config, embodiment_tag):\n"
        "    if embodiment_tag.value in MODALITY_CONFIGS:\n"
        "        raise ValueError('already registered')\n"
        "    MODALITY_CONFIGS[embodiment_tag.value] = config\n",
        encoding="utf-8",
    )
    return groot_root, dataset_root, manifest_path, manifest


def _plan(tmp_path: Path) -> tuple[Any, Path, Path, Path, TrainingManifest]:
    groot_root, dataset_root, manifest_path, manifest = _write_groot_fixture(tmp_path)
    output_root = (tmp_path / "outputs").resolve()
    output_root.mkdir()
    config = load_groot_config(CONFIG_PATH)
    plan = plan_groot_training(
        manifest,
        config,
        groot_root=groot_root,
        dataset_root=dataset_root,
        output_root=output_root,
        manifest_path=manifest_path,
        revision_reader=lambda _: GROOT_REVISION,
    )
    return plan, groot_root, dataset_root, output_root, manifest


def test_groot_plan_uses_exact_appendix_budget_and_official_flags(tmp_path: Path) -> None:
    plan, groot_root, dataset_root, output_root, manifest = _plan(tmp_path)

    assert plan.argv[:4] == (
        "uv",
        "run",
        "python",
        "gr00t/experiment/launch_finetune.py",
    )
    assert _argv_value(plan.argv, "--base-model-path") == "nvidia/GR00T-N1.6-3B"
    assert _argv_value(plan.argv, "--num-gpus") == "8"
    assert _argv_value(plan.argv, "--global-batch-size") == "64"
    assert _argv_value(plan.argv, "--max-steps") == "5000"
    assert _argv_value(plan.argv, "--save-steps") == "1000"
    assert _argv_value(plan.argv, "--save-total-limit") == "5"
    assert _argv_value(plan.argv, "--dataloader-num-workers") == "2"
    assert _argv_value(plan.argv, "--dataset-path") == str(dataset_root)
    assert _argv_value(plan.argv, "--modality-config-path") == str(
        dataset_root / "meta/evoworld_modality.py"
    )
    assert Path(_argv_value(plan.argv, "--output-dir")).parent == output_root
    assert plan.cwd == groot_root
    assert plan.manifest_sha256 == manifest.sha256
    assert plan.upstream_revision == GROOT_REVISION
    assert plan.expected_outputs == (Path("checkpoint-5000"),)


def test_config_is_frozen_and_loader_is_strict(tmp_path: Path) -> None:
    config = load_groot_config(CONFIG_PATH)
    with pytest.raises(FrozenInstanceError):
        config.max_steps = 1  # type: ignore[misc]

    invalid_path = tmp_path / "invalid.toml"
    invalid_path.write_text(CONFIG_PATH.read_text() + "unknown = 1\n", encoding="utf-8")
    with pytest.raises(GrootConfigError, match="unknown"):
        load_groot_config(invalid_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_model_path", "nvidia/other"),
        ("num_gpus", 4),
        ("global_batch_size", 32),
        ("max_steps", 4999),
        ("action_horizon", 8),
    ],
)
def test_config_rejects_non_method_settings(field: str, value: object) -> None:
    with pytest.raises(GrootConfigError):
        replace(load_groot_config(CONFIG_PATH), **{field: value})


def test_dataset_validation_checks_modality_language_and_delta_indices(tmp_path: Path) -> None:
    groot_root, dataset_root, _, manifest = _write_groot_fixture(tmp_path)
    config = load_groot_config(CONFIG_PATH)

    modality_path = dataset_root / "meta/modality.json"
    modality = json.loads(modality_path.read_text())
    del modality["video"]
    _write_json(modality_path, modality)
    with pytest.raises(GrootConfigError, match="video"):
        validate_groot_dataset(dataset_root, manifest, config, groot_root=groot_root)

    groot_root, dataset_root, _, manifest = _write_groot_fixture(tmp_path / "wrong-language")
    registration = dataset_root / "meta/evoworld_modality.py"
    registration.write_text(
        _registration_source(language_key="annotation.missing"), encoding="utf-8"
    )
    with pytest.raises(GrootConfigError, match="language"):
        validate_groot_dataset(dataset_root, manifest, config, groot_root=groot_root)

    groot_root, dataset_root, _, manifest = _write_groot_fixture(tmp_path / "wrong-delta")
    registration = dataset_root / "meta/evoworld_modality.py"
    registration.write_text(_registration_source(action_stop=8), encoding="utf-8")
    with pytest.raises(GrootConfigError, match="delta"):
        validate_groot_dataset(dataset_root, manifest, config, groot_root=groot_root)


def test_dataset_validation_checks_registration_and_action_shape(tmp_path: Path) -> None:
    groot_root, dataset_root, _, manifest = _write_groot_fixture(tmp_path)
    config = load_groot_config(CONFIG_PATH)
    (dataset_root / "meta/evoworld_modality.py").unlink()
    with pytest.raises(GrootConfigError, match="registration"):
        validate_groot_dataset(dataset_root, manifest, config, groot_root=groot_root)

    groot_root, dataset_root, _, manifest = _write_groot_fixture(tmp_path / "wrong-shape")
    info_path = dataset_root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["action"]["shape"] = [2]
    _write_json(info_path, info)
    with pytest.raises(GrootConfigError, match="action shape"):
        validate_groot_dataset(dataset_root, manifest, config, groot_root=groot_root)


def test_dataset_validation_checks_parquet_rows_columns_and_official_imports(
    tmp_path: Path,
) -> None:
    groot_root, dataset_root, _, manifest = _write_groot_fixture(tmp_path)
    parquet_path = dataset_root / "data/chunk-000/episode_000000.parquet"
    pq.write_table(pa.table({"task_index": list(range(15))}), parquet_path)
    with pytest.raises(GrootConfigError, match="Parquet.*rows|columns"):
        validate_groot_dataset(
            dataset_root,
            manifest,
            load_groot_config(CONFIG_PATH),
            groot_root=groot_root,
        )

    groot_root, dataset_root, _, manifest = _write_groot_fixture(tmp_path / "wrong-import")
    registration = dataset_root / "meta/evoworld_modality.py"
    registration.write_text(
        _registration_source().replace(
            "from gr00t.data.types import ModalityConfig",
            "from unrelated.types import ModalityConfig",
        ),
        encoding="utf-8",
    )
    with pytest.raises(GrootConfigError, match="official GR00T"):
        validate_groot_dataset(
            dataset_root,
            manifest,
            load_groot_config(CONFIG_PATH),
            groot_root=groot_root,
        )


@pytest.mark.parametrize(
    ("column", "values", "message"),
    [
        ("action", pa.array([[0.0, 0.0, 0.0]] * 16), "action.*float32"),
        (
            "action",
            pa.array([[float("nan"), 0.0, 0.0]] * 16, type=pa.list_(pa.float32(), 3)),
            "action.*finite",
        ),
        (
            "observation.state",
            pa.array([[0.0, 0.0]] * 16, type=pa.list_(pa.float32(), 2)),
            "state.*shape",
        ),
        ("episode_index", pa.array([1] * 16, type=pa.int64()), "episode_index"),
        ("frame_index", pa.array([0] * 16, type=pa.int64()), "frame_index"),
        ("index", pa.array(list(range(1, 17)), type=pa.int64()), "index"),
        ("task_index", pa.array([7] * 16, type=pa.int64()), "task_index"),
        ("timestamp", pa.array([0.0] * 16, type=pa.float32()), "timestamp"),
        ("task_index", pa.array(["pick the cube"] * 16, type=pa.string()), "task_index|language"),
    ],
)
def test_parquet_arrow_schema_and_values_are_bound_to_metadata(
    tmp_path: Path,
    column: str,
    values: pa.Array,
    message: str,
) -> None:
    groot_root, dataset_root, _, manifest = _write_groot_fixture(tmp_path)
    parquet_path = dataset_root / "data/chunk-000/episode_000000.parquet"
    table = pq.read_table(parquet_path)
    pq.write_table(table.set_column(table.column_names.index(column), column, values), parquet_path)

    with pytest.raises(GrootConfigError, match=message):
        validate_groot_dataset(
            dataset_root,
            manifest,
            load_groot_config(CONFIG_PATH),
            groot_root=groot_root,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("chunks_size", 10, "chunks_size"),
        ("fps", 0, "fps"),
        ("codebase_version", "v3.0", "v2.1"),
    ],
)
def test_info_requires_pinned_lerobot_fields(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    groot_root, dataset_root, _, manifest = _write_groot_fixture(tmp_path)
    info_path = dataset_root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info[field] = value
    _write_json(info_path, info)
    with pytest.raises(GrootConfigError, match=message):
        validate_groot_dataset(
            dataset_root,
            manifest,
            load_groot_config(CONFIG_PATH),
            groot_root=groot_root,
        )


def test_episode_must_cover_full_action_delta_horizon(tmp_path: Path) -> None:
    groot_root, dataset_root, _, manifest = _write_groot_fixture(tmp_path)
    episodes_path = dataset_root / "meta/episodes.jsonl"
    _write_jsonl(
        episodes_path,
        [{"episode_index": 0, "tasks": ["pick the cube"], "length": 15}],
    )
    info_path = dataset_root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["total_frames"] = 15
    _write_json(info_path, info)

    with pytest.raises(GrootConfigError, match="horizon|16"):
        validate_groot_dataset(
            dataset_root,
            manifest,
            load_groot_config(CONFIG_PATH),
            groot_root=groot_root,
        )


def test_video_probe_requires_real_stream_frames_dimensions_and_fps(tmp_path: Path) -> None:
    groot_root, dataset_root, _, manifest = _write_groot_fixture(tmp_path)
    video_path = dataset_root / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    video_path.write_bytes(_TINY_MP4[:128])
    with pytest.raises(GrootConfigError, match="video.*stream|MP4.*track|frames"):
        validate_groot_dataset(
            dataset_root,
            manifest,
            load_groot_config(CONFIG_PATH),
            groot_root=groot_root,
        )

    fake_root, fake_dataset, _, fake_manifest = _write_groot_fixture(tmp_path / "fake")
    fake_video = fake_dataset / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    fake_video.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + b"\x00" * 165)
    with pytest.raises(GrootConfigError, match="video stream|MP4"):
        validate_groot_dataset(
            fake_dataset,
            fake_manifest,
            load_groot_config(CONFIG_PATH),
            groot_root=fake_root,
        )


def test_video_requires_full_decoder_success_not_only_valid_sample_tables(tmp_path: Path) -> None:
    groot_root, dataset_root, _, manifest = _write_groot_fixture(tmp_path)
    video_path = dataset_root / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    payload = bytearray(video_path.read_bytes())
    marker = payload.index(b"mdat")
    box_start = marker - 4
    box_size = int.from_bytes(payload[box_start:marker], "big")
    header_size = 8
    if box_size == 1:
        box_size = int.from_bytes(payload[marker + 4 : marker + 12], "big")
        header_size = 16
    payload[box_start + header_size : box_start + box_size] = b"\x00" * (box_size - header_size)
    video_path.write_bytes(payload)

    with pytest.raises(GrootConfigError, match="decode"):
        validate_groot_dataset(
            dataset_root,
            manifest,
            load_groot_config(CONFIG_PATH),
            groot_root=groot_root,
        )


def test_video_decode_accepts_real_av1_stream(tmp_path: Path) -> None:
    groot_root, dataset_root, _, manifest = _write_groot_fixture(tmp_path)
    video_path = dataset_root / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    video_path.write_bytes(_TINY_AV1_MP4)
    manifest = _refresh_manifest(dataset_root, manifest)

    validate_groot_dataset(
        dataset_root,
        manifest,
        load_groot_config(CONFIG_PATH),
        groot_root=groot_root,
    )


def test_video_decode_uses_minimal_environment_and_safe_local_protocols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groot_root, dataset_root, _, manifest = _write_groot_fixture(tmp_path)
    report_path = tmp_path / "ffmpeg-report.log"
    monkeypatch.setenv("FFREPORT", f"file={report_path}:level=32")
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", "/usr/bin/false")
    monkeypatch.setenv("LD_PRELOAD", "/untrusted/library.so")
    monkeypatch.setenv("DYLD_INSERT_LIBRARIES", "/untrusted/library.dylib")
    ffmpeg_calls: list[tuple[tuple[str, ...], dict[str, str] | None]] = []
    original_run = groot_module.subprocess.run

    def recording_run(*args: Any, **kwargs: Any) -> Any:
        argv = tuple(args[0])
        if Path(argv[0]).name.startswith("ffmpeg-"):
            ffmpeg_calls.append((argv, kwargs.get("env")))
        return original_run(*args, **kwargs)

    monkeypatch.setattr(groot_module.subprocess, "run", recording_run)

    validate_groot_dataset(
        dataset_root,
        manifest,
        load_groot_config(CONFIG_PATH),
        groot_root=groot_root,
    )
    assert len(ffmpeg_calls) == 1
    argv, environment = ffmpeg_calls[0]
    assert environment == {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    assert _argv_value(argv, "-protocol_whitelist") == "file,pipe"
    assert not report_path.exists()


@pytest.mark.parametrize("artifact", ["video", "registration", "stats"])
def test_dataset_digest_rejects_stale_non_parquet_artifacts(tmp_path: Path, artifact: str) -> None:
    groot_root, dataset_root, _, manifest = _write_groot_fixture(tmp_path)
    artifact_paths = {
        "video": dataset_root / "videos/chunk-000/observation.images.front/episode_000000.mp4",
        "registration": dataset_root / "meta/evoworld_modality.py",
        "stats": dataset_root / "meta/stats.json",
    }
    if artifact == "video":
        payload = bytearray(artifact_paths[artifact].read_bytes())
        payload[20] ^= 1
        artifact_paths[artifact].write_bytes(payload)
    else:
        with artifact_paths[artifact].open("ab") as stream:
            stream.write(b"\n" if artifact == "stats" else b"# provenance change\n")
    with pytest.raises(GrootConfigError, match="provenance"):
        validate_groot_dataset(
            dataset_root,
            manifest,
            load_groot_config(CONFIG_PATH),
            groot_root=groot_root,
        )


def test_registration_rejects_arbitrary_top_level_code(tmp_path: Path) -> None:
    groot_root, dataset_root, _, manifest = _write_groot_fixture(tmp_path)
    registration = dataset_root / "meta/evoworld_modality.py"
    registration.write_text(
        _registration_source() + "raise RuntimeError('unsafe')\n", encoding="utf-8"
    )
    with pytest.raises(GrootConfigError, match="safe|top-level"):
        validate_groot_dataset(
            dataset_root,
            manifest,
            load_groot_config(CONFIG_PATH),
            groot_root=groot_root,
        )


@pytest.mark.parametrize(
    "target",
    [
        "registration_config",
        "register_modality_config",
        "EmbodimentTag",
        "ModalityConfig",
        "__builtins__",
    ],
)
def test_registration_requires_exact_unshadowed_assignment_target(
    tmp_path: Path,
    target: str,
) -> None:
    groot_root, dataset_root, _, manifest = _write_groot_fixture(tmp_path)
    registration = dataset_root / "meta/evoworld_modality.py"
    registration.write_text(
        _registration_source()
        .replace("evoworld_config = {", f"{target} = {{")
        .replace("    evoworld_config,", f"    {target},"),
        encoding="utf-8",
    )

    with pytest.raises(GrootConfigError, match="assignment|evoworld_config|shadow"):
        validate_groot_dataset(
            dataset_root,
            manifest,
            load_groot_config(CONFIG_PATH),
            groot_root=groot_root,
        )


def test_registration_rejects_duplicate_assignment(tmp_path: Path) -> None:
    groot_root, dataset_root, _, manifest = _write_groot_fixture(tmp_path)
    registration = dataset_root / "meta/evoworld_modality.py"
    registration.write_text(
        _registration_source().replace(
            "evoworld_config = {",
            "evoworld_config = {}\nevoworld_config = {",
        ),
        encoding="utf-8",
    )

    with pytest.raises(GrootConfigError, match="assignment|top-level|duplicate"):
        validate_groot_dataset(
            dataset_root,
            manifest,
            load_groot_config(CONFIG_PATH),
            groot_root=groot_root,
        )


def test_registration_marker_exploit_is_rejected_without_execution(tmp_path: Path) -> None:
    groot_root, dataset_root, _, manifest = _write_groot_fixture(tmp_path)
    marker = tmp_path / "registration-executed"
    registration = dataset_root / "meta/evoworld_modality.py"
    source = _registration_source().replace(
        "evoworld_config = {",
        f'evoworld_config = {{\n    **({{}} if not open({str(marker)!r}, "w") else {{}}),',
    )
    registration.write_text(source, encoding="utf-8")

    with pytest.raises(GrootConfigError, match="safe|dictionary|unpack"):
        validate_groot_dataset(
            dataset_root,
            manifest,
            load_groot_config(CONFIG_PATH),
            groot_root=groot_root,
        )
    assert not marker.exists()


@pytest.mark.parametrize(
    ("artifact", "mutation"),
    [
        ("stats.json", lambda value: value["action"]["mean"].__setitem__(0, float("nan"))),
        ("stats.json", lambda value: value["action"]["mean"].__setitem__(0, "bad")),
        ("relative_stats.json", lambda value: value["arm"]["std"][0].__setitem__(0, True)),
        ("relative_stats.json", lambda value: value["arm"]["std"][0].__setitem__(0, float("inf"))),
    ],
)
def test_statistics_require_recursive_finite_numeric_leaves(
    tmp_path: Path,
    artifact: str,
    mutation: Any,
) -> None:
    groot_root, dataset_root, _, manifest = _write_groot_fixture(tmp_path)
    artifact_path = dataset_root / "meta" / artifact
    value = json.loads(artifact_path.read_text())
    mutation(value)
    _write_json(artifact_path, value)
    with pytest.raises(GrootConfigError, match="finite|constant|numeric"):
        validate_groot_dataset(
            dataset_root,
            manifest,
            load_groot_config(CONFIG_PATH),
            groot_root=groot_root,
        )


def test_dataset_digest_uses_actual_registration_config_path(tmp_path: Path) -> None:
    groot_root, dataset_root, _, manifest = _write_groot_fixture(tmp_path)
    default_path = dataset_root / "meta/evoworld_modality.py"
    custom_relative = Path("meta/custom_registration.py")
    custom_path = dataset_root / custom_relative
    default_path.rename(custom_path)
    config = replace(load_groot_config(CONFIG_PATH), modality_config_path=custom_relative)
    manifest = _refresh_manifest(dataset_root, manifest, custom_relative)
    validate_groot_dataset(dataset_root, manifest, config, groot_root=groot_root)

    with custom_path.open("a", encoding="utf-8") as stream:
        stream.write("# changed\n")
    with pytest.raises(GrootConfigError, match="provenance"):
        validate_groot_dataset(dataset_root, manifest, config, groot_root=groot_root)


def test_registration_rejects_failed_isolated_import_probe(tmp_path: Path) -> None:
    groot_root, dataset_root, _, manifest = _write_groot_fixture(tmp_path / "probe")
    upstream_registration = groot_root / "gr00t/configs/data/embodiment_configs.py"
    upstream_registration.write_text(
        upstream_registration.read_text().replace(
            "MODALITY_CONFIGS[embodiment_tag.value] = config",
            "raise RuntimeError('probe failed')",
        ),
        encoding="utf-8",
    )
    with pytest.raises(GrootConfigError, match="import probe"):
        validate_groot_dataset(
            dataset_root,
            manifest,
            load_groot_config(CONFIG_PATH),
            groot_root=groot_root,
        )


def test_registration_probe_imports_exact_planned_original_source(tmp_path: Path) -> None:
    groot_root, dataset_root, manifest_path, manifest = _write_groot_fixture(tmp_path)
    registration = dataset_root / "meta/evoworld_modality.py"
    registration.write_text(_registration_source() + "# original-probe-token\n", encoding="utf-8")
    manifest = _refresh_manifest(dataset_root, manifest)
    upstream_registration = groot_root / "gr00t/configs/data/embodiment_configs.py"
    upstream_registration.write_text(
        upstream_registration.read_text().replace(
            "    if embodiment_tag.value in MODALITY_CONFIGS:",
            "    import inspect\n"
            "    import pathlib\n"
            "    caller = inspect.currentframe().f_back\n"
            "    source = pathlib.Path(caller.f_globals['__file__']).read_text()\n"
            "    if '# original-probe-token' not in source:\n"
            "        raise RuntimeError('probe did not import original source')\n"
            "    if embodiment_tag.value in MODALITY_CONFIGS:",
        ),
        encoding="utf-8",
    )
    output_root = (tmp_path / "outputs").resolve()
    output_root.mkdir()

    plan = plan_groot_training(
        manifest,
        load_groot_config(CONFIG_PATH),
        groot_root=groot_root,
        dataset_root=dataset_root,
        output_root=output_root,
        manifest_path=manifest_path,
        revision_reader=lambda _: GROOT_REVISION,
    )

    assert _argv_value(plan.argv, "--modality-config-path") == str(registration)


def test_registration_probe_verifies_registered_configuration(tmp_path: Path) -> None:
    groot_root, dataset_root, _, manifest = _write_groot_fixture(tmp_path)
    upstream_registration = groot_root / "gr00t/configs/data/embodiment_configs.py"
    upstream_registration.write_text(
        upstream_registration.read_text().replace(
            "    MODALITY_CONFIGS[embodiment_tag.value] = config",
            "    config['action'].delta_indices = [0]\n"
            "    MODALITY_CONFIGS[embodiment_tag.value] = config",
        ),
        encoding="utf-8",
    )

    with pytest.raises(GrootConfigError, match="import probe|registered config"):
        validate_groot_dataset(
            dataset_root,
            manifest,
            load_groot_config(CONFIG_PATH),
            groot_root=groot_root,
        )


def test_released_embodiment_uses_pinned_builtin_config_without_custom_module(
    tmp_path: Path,
) -> None:
    groot_root, dataset_root, manifest_path, manifest = _write_groot_fixture(tmp_path)
    builtin_source = """from gr00t.data.types import ModalityConfig

MODALITY_CONFIGS = {
    "libero_panda": {
        "video": ModalityConfig(delta_indices=[0], modality_keys=["front"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=["arm", "gripper"]),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)), modality_keys=["arm", "gripper"]
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task"]
        ),
    },
}

def register_modality_config(config, embodiment_tag):
    raise RuntimeError("released tags cannot be re-registered")
"""
    (groot_root / "gr00t/configs/data/embodiment_configs.py").write_text(
        builtin_source,
        encoding="utf-8",
    )
    (dataset_root / "meta/evoworld_modality.py").unlink()
    manifest = _refresh_manifest(dataset_root, manifest, None)
    builtin_config_path = tmp_path / "groot_builtin.toml"
    builtin_config_path.write_text(
        CONFIG_PATH.read_text(encoding="utf-8")
        .replace('embodiment_tag = "NEW_EMBODIMENT"', 'embodiment_tag = "LIBERO_PANDA"')
        .replace('modality_config_path = "meta/evoworld_modality.py"\n', ""),
        encoding="utf-8",
    )
    config = load_groot_config(builtin_config_path)
    assert config.modality_config_path is None
    validate_groot_dataset(dataset_root, manifest, config, groot_root=groot_root)
    output_root = (tmp_path / "outputs").resolve()
    output_root.mkdir()
    plan = plan_groot_training(
        manifest,
        config,
        groot_root=groot_root,
        dataset_root=dataset_root,
        output_root=output_root,
        manifest_path=manifest_path,
        revision_reader=lambda _: GROOT_REVISION,
    )
    assert "--modality-config-path" not in plan.argv

    with pytest.raises(GrootConfigError, match="custom|built-in"):
        replace(config, modality_config_path=Path("meta/evoworld_modality.py"))

    with pytest.raises(GrootConfigError, match="LIBERO|unsupported"):
        replace(config, embodiment_tag="OXE_DROID")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("diagnosis", DiagnosisLabel.TSM, "PF"),
        ("split", "eval", "train"),
        ("target", "pi05", "target"),
        ("content_sha256", "f" * 64, "provenance"),
    ],
)
def test_training_rejects_wrong_manifest_samples(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    groot_root, dataset_root, manifest_path, manifest = _write_groot_fixture(tmp_path)
    bad_manifest = replace(manifest, samples=(replace(manifest.samples[0], **{field: value}),))
    _write_json(manifest_path, bad_manifest.to_dict())
    with pytest.raises(GrootConfigError, match=message):
        plan_groot_training(
            bad_manifest,
            load_groot_config(CONFIG_PATH),
            groot_root=groot_root,
            dataset_root=dataset_root,
            output_root=(tmp_path / "outputs").resolve(),
            manifest_path=manifest_path,
            revision_reader=lambda _: GROOT_REVISION,
        )


def test_training_rejects_eval_split_leakage_and_wrong_revision(tmp_path: Path) -> None:
    groot_root, dataset_root, manifest_path, manifest = _write_groot_fixture(tmp_path)
    info_path = dataset_root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["splits"] = {"train": "0:1", "eval": "0:1"}
    _write_json(info_path, info)
    with pytest.raises(GrootConfigError, match="train-only"):
        validate_groot_dataset(
            dataset_root,
            manifest,
            load_groot_config(CONFIG_PATH),
            groot_root=groot_root,
        )

    with pytest.raises(GrootConfigError, match=GROOT_REVISION):
        plan_groot_training(
            manifest,
            load_groot_config(CONFIG_PATH),
            groot_root=groot_root,
            dataset_root=dataset_root,
            output_root=(tmp_path / "outputs").resolve(),
            manifest_path=manifest_path,
            revision_reader=lambda _: "0" * 40,
        )


def test_open_loop_plan_is_distinct_and_binds_checkpoint_dataset_modality(tmp_path: Path) -> None:
    training_plan, groot_root, dataset_root, output_root, manifest = _plan(tmp_path)
    checkpoint = Path(_argv_value(training_plan.argv, "--output-dir")) / "checkpoint-5000"
    checkpoint.mkdir(parents=True)

    eval_plan = plan_groot_open_loop_eval(
        manifest,
        load_groot_config(CONFIG_PATH),
        groot_root=groot_root,
        dataset_root=dataset_root,
        checkpoint_path=checkpoint,
        output_root=output_root,
        revision_reader=lambda _: GROOT_REVISION,
    )

    assert eval_plan.job_id != training_plan.job_id
    assert eval_plan.target == "groot_n16_open_loop"
    assert eval_plan.argv[:4] == (
        "uv",
        "run",
        "python",
        "gr00t/eval/open_loop_eval.py",
    )
    assert _argv_value(eval_plan.argv, "--model-path") == str(checkpoint)
    assert _argv_value(eval_plan.argv, "--dataset-path") == str(dataset_root)
    assert _argv_value(eval_plan.argv, "--embodiment-tag") == "NEW_EMBODIMENT"
    assert _argv_value(eval_plan.argv, "--action-horizon") == "16"
    modality_index = eval_plan.argv.index("--modality-keys")
    assert eval_plan.argv[modality_index + 1 : modality_index + 3] == ("arm", "gripper")
    assert "gr00t/experiment/launch_finetune.py" not in eval_plan.argv

    with pytest.raises(GrootConfigError, match="trajectory"):
        plan_groot_open_loop_eval(
            manifest,
            replace(load_groot_config(CONFIG_PATH), trajectory_ids=(1,)),
            groot_root=groot_root,
            dataset_root=dataset_root,
            checkpoint_path=checkpoint,
            output_root=output_root,
            revision_reader=lambda _: GROOT_REVISION,
        )

    sibling = output_root / "unrelated/checkpoint-5000"
    sibling.mkdir(parents=True)
    with pytest.raises(GrootConfigError, match="training job|checkpoint"):
        plan_groot_open_loop_eval(
            manifest,
            load_groot_config(CONFIG_PATH),
            groot_root=groot_root,
            dataset_root=dataset_root,
            checkpoint_path=sibling,
            output_root=output_root,
            revision_reader=lambda _: GROOT_REVISION,
        )

    with pytest.raises(GrootConfigError, match="one trajectory"):
        replace(load_groot_config(CONFIG_PATH), trajectory_ids=(0, 1))


def test_plans_require_absolute_contained_paths(tmp_path: Path) -> None:
    groot_root, dataset_root, manifest_path, manifest = _write_groot_fixture(tmp_path)
    with pytest.raises(GrootConfigError, match="absolute"):
        plan_groot_training(
            manifest,
            load_groot_config(CONFIG_PATH),
            groot_root=groot_root,
            dataset_root=dataset_root,
            output_root=Path("relative-output"),
            manifest_path=manifest_path,
            revision_reader=lambda _: GROOT_REVISION,
        )

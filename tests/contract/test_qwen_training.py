from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import imageio_ffmpeg  # type: ignore[import-untyped]
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from evoworld.contracts import DiagnosisLabel
from training.common.jobs import execute_job, write_job
from training.common.schema import TrainingPlan
from training.evaluator.qwen_vl import (
    DECORD_REQUIREMENT,
    MS_SWIFT_REVISION,
    MS_SWIFT_REQUIREMENT,
    QWEN_VL_UTILS_REQUIREMENT,
    TORCH_REQUIREMENT,
    TORCHVISION_REQUIREMENT,
    TRANSFORMERS_REQUIREMENT,
    EvaluatorConfigError,
    EvaluatorExport,
    QwenTrainConfig,
    load_qwen_config,
    plan_qwen_training,
    run_qwen_training,
    validate_qwen_runtime_versions,
)
from training.evaluator.schema import (
    EvaluatorExample,
    EvaluatorSchemaError,
    export_ms_swift,
    validate_evaluator_dataset,
    validate_example,
)


FIXTURE = Path(__file__).parents[1] / "fixtures" / "evaluator_examples.jsonl"
RED_PPM = b"P6\n1 1\n255\n\xff\x00\x00"
GREEN_PPM = b"P6\n1 1\n255\n\x00\xff\x00"
BLUE_PPM = b"P6\n1 1\n255\n\x00\x00\xff"
YELLOW_PPM = b"P6\n1 1\n255\n\xff\xff\x00"
REQUIRED_CONFIG_FIELDS = (
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
)


def _materialize_fixture(tmp_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "dataset"
    media = dataset_root / "media"
    media.mkdir(parents=True)
    (media / "real.ppm").write_bytes(RED_PPM)
    (media / "real-next.ppm").write_bytes(BLUE_PPM)
    (media / "synthetic.ppm").write_bytes(GREEN_PPM)
    (media / "synthetic-next.ppm").write_bytes(YELLOW_PPM)
    dataset_path = dataset_root / "examples.jsonl"
    dataset_path.write_bytes(FIXTURE.read_bytes())
    return dataset_root, dataset_path


def _read_records(dataset_path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]


def _write_records(dataset_path: Path, records: list[dict[str, object]]) -> None:
    dataset_path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def _validated(tmp_path: Path) -> tuple[Path, tuple[EvaluatorExample, ...]]:
    root, dataset_path = _materialize_fixture(tmp_path)
    return root, validate_evaluator_dataset(
        dataset_path,
        dataset_root=root,
        reliability_threshold=0.7,
    )


def _config_text(export: EvaluatorExport, **changes: object) -> str:
    values: dict[str, object] = {
        "ms_swift_version": "4.3.1",
        "qwen_vl_utils_version": "0.0.14",
        "decord_version": "0.6.0",
        "transformers_requirement": ">=4.57,<5.13.0",
        "torch_requirement": ">=2.0",
        "torchvision_requirement": ">=0.15",
        "torch_platform": "cuda",
        "base_model": "Qwen/Qwen3-VL-4B-Instruct",
        "tuner_type": "lora",
        "learning_rate": 0.0001,
        "lora_rank": 16,
        "lora_alpha": 32,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_length": 8192,
        "seed": 42,
        "output_dir": "artifacts/evaluator/qwen3-vl-4b",
        "max_pixels": 1003520,
        "video_max_token_num": 256,
        "video_max_pixels": 262144,
        "fps": 2.0,
        "fps_max_frames": 16,
        "torch_dtype": "bfloat16",
        "device_map": "auto",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "logging_steps": 10,
        "dataset_export_sha256": export.sha256,
        "dataset_revision": export.dataset_revision,
    }
    values.update(changes)
    lines = [
        "# Configurable public baseline; not the unpublished experiment recipe.",
        "[qwen3_vl_4b]",
    ]
    for key, value in values.items():
        if value is None:
            continue
        rendered = json.dumps(value)
        lines.append(f"{key} = {rendered}")
    return "\n".join(lines) + "\n"


def _export_and_config(tmp_path: Path) -> tuple[Path, EvaluatorExport, QwenTrainConfig]:
    root, examples = _validated(tmp_path)
    exported = export_ms_swift(
        examples,
        dataset_root=root,
        export_path=tmp_path / "exports" / "qwen.jsonl",
    )
    config_path = tmp_path / "qwen.toml"
    config_path.write_text(_config_text(exported), encoding="utf-8")
    return root, exported, load_qwen_config(config_path)


def _export_with_train_frames(
    tmp_path: Path, frame_count: int
) -> tuple[EvaluatorExport, QwenTrainConfig]:
    root, dataset_path = _materialize_fixture(tmp_path)
    records = _read_records(dataset_path)
    paths: list[str] = []
    hashes: list[str] = []
    for index in range(frame_count):
        payload = b"P6\n1 1\n255\n" + bytes((index, 0, 0))
        relative = f"media/frame-{index:02d}.ppm"
        (root / relative).write_bytes(payload)
        paths.append(relative)
        hashes.append(sha256(payload).hexdigest())
    records[0]["media_paths"] = paths
    records[0]["media_sha256"] = hashes
    _write_records(dataset_path, records)
    examples = validate_evaluator_dataset(
        dataset_path, dataset_root=root, reliability_threshold=0.7
    )
    exported = export_ms_swift(
        examples,
        dataset_root=root,
        export_path=tmp_path / "exports" / "qwen.jsonl",
    )
    config_path = tmp_path / "qwen.toml"
    config_path.write_text(_config_text(exported), encoding="utf-8")
    return exported, load_qwen_config(config_path)


def _write_completed_output(
    plan: TrainingPlan,
    export: EvaluatorExport,
    config: QwenTrainConfig,
    workspace_root: Path,
    *,
    global_step: int = 3,
) -> None:
    output = Path(plan.argv[plan.argv.index("--output_dir") + 1])
    checkpoint = output / f"checkpoint-{global_step}"
    checkpoint.mkdir(parents=True)
    args = {
        "swift_version": "4.3.1",
        "model": config.base_model,
        "tuner_type": config.tuner_type,
        "dataset": [str(export.path)],
        "val_dataset": [str(export.validation_path)],
        "split_dataset_ratio": 0.0,
        "strict": True,
        "output_dir": str(output),
        "learning_rate": config.learning_rate,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_length": config.max_length,
        "seed": config.seed,
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
    (output / "args.json").write_text(json.dumps(args), encoding="utf-8")
    (output / "logging.jsonl").write_text(
        json.dumps({"last_model_checkpoint": str(checkpoint), "global_step": global_step}) + "\n",
        encoding="utf-8",
    )
    (checkpoint / "args.json").write_text(json.dumps(args), encoding="utf-8")
    (checkpoint / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": config.base_model,
                "lora_alpha": config.lora_alpha,
                "r": config.lora_rank,
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint / "training_args.bin").write_bytes(b"training args")
    assert output.is_relative_to(workspace_root)


def _resolved_runtime_versions() -> dict[str, str]:
    return {
        "ms-swift": "4.3.1",
        "qwen-vl-utils": "0.0.14",
        "decord": "0.6.0",
        "transformers": "4.57.3",
        "torch": "2.8.0",
        "torchvision": "0.23.0",
        "cuda_available": "true",
    }


@pytest.mark.parametrize(
    ("score", "label"),
    ((0.8, DiagnosisLabel.PF), (0.6, DiagnosisLabel.NONE)),
)
def test_evaluator_label_matches_threshold(score: float, label: DiagnosisLabel) -> None:
    with pytest.raises(EvaluatorSchemaError, match="threshold"):
        validate_example(
            EvaluatorExample(
                schema_version=1,
                example_id="example-1",
                task_id="task-1",
                instruction="pick the cube",
                scene_id="scene-1",
                trajectory_id="trajectory-1",
                media_paths=(Path("media/frame.ppm"),),
                media_sha256=(sha256(RED_PPM).hexdigest(),),
                source_type="real",
                source_checkpoint="robot-run",
                provenance="teleoperated",
                content_sha256="a" * 64,
                score=score,
                label=label,
                rationale="The grasp failed.",
                annotator="annotator-a",
                revision=1,
                split="train",
            ),
            reliability_threshold=0.7,
        )


def test_example_is_immutable_and_validated_dataset_uses_real_media(tmp_path: Path) -> None:
    _, examples = _validated(tmp_path)
    assert [example.source_type for example in examples] == [
        "real",
        "manually_verified_synthetic",
    ]
    with pytest.raises(FrozenInstanceError):
        examples[0].score = 0.1  # type: ignore[misc]


def test_invalid_utf8_is_rejected(tmp_path: Path) -> None:
    root, dataset_path = _materialize_fixture(tmp_path)
    dataset_path.write_bytes(b"\xff\xfe")
    with pytest.raises(EvaluatorSchemaError, match="UTF-8"):
        validate_evaluator_dataset(dataset_path, dataset_root=root, reliability_threshold=0.7)


@pytest.mark.parametrize("payload", (None, b"", b"not an image"))
def test_missing_empty_or_corrupt_media_is_rejected(tmp_path: Path, payload: bytes | None) -> None:
    root, dataset_path = _materialize_fixture(tmp_path)
    frame = root / "media" / "real.ppm"
    if payload is None:
        frame.unlink()
    else:
        frame.write_bytes(payload)
    with pytest.raises(EvaluatorSchemaError, match="media"):
        validate_evaluator_dataset(dataset_path, dataset_root=root, reliability_threshold=0.7)


def test_video_container_is_rejected_when_schema_declares_ordered_frames(tmp_path: Path) -> None:
    root, dataset_path = _materialize_fixture(tmp_path)
    video = root / "media" / "trajectory.mp4"
    subprocess.run(
        (
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=2x2:d=0.1",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ),
        check=True,
        capture_output=True,
    )
    records = _read_records(dataset_path)
    records[0]["media_paths"] = ["media/trajectory.mp4"]
    records[0]["media_sha256"] = [sha256(video.read_bytes()).hexdigest()]
    _write_records(dataset_path, records)
    with pytest.raises(EvaluatorSchemaError, match="still frame|image"):
        validate_evaluator_dataset(dataset_path, dataset_root=root, reliability_threshold=0.7)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("label", "OTHER", "label"),
        ("source_type", "synthetic", "source"),
        ("source_checkpoint", " ", "source_checkpoint"),
        ("provenance", "", "provenance"),
        ("rationale", " ", "rationale"),
        ("annotator", "", "annotator"),
    ),
)
def test_invalid_annotation_fields_are_rejected(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    root, dataset_path = _materialize_fixture(tmp_path)
    records = _read_records(dataset_path)
    records[0][field] = value
    _write_records(dataset_path, records)
    with pytest.raises(EvaluatorSchemaError, match=message):
        validate_evaluator_dataset(dataset_path, dataset_root=root, reliability_threshold=0.7)


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    root, dataset_path = _materialize_fixture(tmp_path)
    line = dataset_path.read_text(encoding="utf-8").splitlines()[0]
    duplicate = line[:-1] + ',"example_id":"shadow"}\n'
    dataset_path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(EvaluatorSchemaError, match="duplicate JSON key"):
        validate_evaluator_dataset(dataset_path, dataset_root=root, reliability_threshold=0.7)


@pytest.mark.parametrize("field", ("example_id", "trajectory_id", "content_sha256", "media_sha256"))
@pytest.mark.parametrize("same_split", (False, True))
def test_all_leakage_keys_are_rejected_within_or_across_splits(
    tmp_path: Path, field: str, same_split: bool
) -> None:
    root, dataset_path = _materialize_fixture(tmp_path)
    records = _read_records(dataset_path)
    records[1][field] = records[0][field]
    if field == "media_sha256":
        records[1]["media_paths"] = records[0]["media_paths"]
    if same_split:
        records[1]["split"] = records[0]["split"]
    _write_records(dataset_path, records)
    with pytest.raises(EvaluatorSchemaError, match="leakage|duplicate"):
        validate_evaluator_dataset(dataset_path, dataset_root=root, reliability_threshold=0.7)


@pytest.mark.parametrize("unsafe", ("../escape.ppm", "/tmp/escape.ppm", "nested\\escape.ppm"))
def test_media_paths_must_be_contained_relative_posix_paths(tmp_path: Path, unsafe: str) -> None:
    root, dataset_path = _materialize_fixture(tmp_path)
    records = _read_records(dataset_path)
    records[0]["media_paths"] = [unsafe]
    _write_records(dataset_path, records)
    with pytest.raises(EvaluatorSchemaError, match="media_paths"):
        validate_evaluator_dataset(dataset_path, dataset_root=root, reliability_threshold=0.7)


def test_media_symlink_cannot_escape_dataset_root(tmp_path: Path) -> None:
    root, dataset_path = _materialize_fixture(tmp_path)
    outside = tmp_path / "outside.ppm"
    outside.write_bytes(RED_PPM)
    link = root / "media" / "link.ppm"
    link.symlink_to(outside)
    records = _read_records(dataset_path)
    records[0]["media_paths"] = ["media/link.ppm"]
    records[0]["media_sha256"] = [sha256(RED_PPM).hexdigest()]
    _write_records(dataset_path, records)
    with pytest.raises(EvaluatorSchemaError, match="escapes dataset root"):
        validate_evaluator_dataset(dataset_path, dataset_root=root, reliability_threshold=0.7)


def test_export_is_deterministic_canonical_and_preserves_frame_order(tmp_path: Path) -> None:
    root, examples = _validated(tmp_path)
    first = export_ms_swift(
        examples,
        dataset_root=root,
        export_path=tmp_path / "a" / "dataset.jsonl",
    )
    second = export_ms_swift(
        examples,
        dataset_root=root,
        export_path=tmp_path / "b" / "dataset.jsonl",
    )
    assert first.sha256 == second.sha256
    assert first.dataset_revision == second.dataset_revision
    assert first.path.read_bytes() == second.path.read_bytes()
    assert first.validation_path.read_bytes() == second.validation_path.read_bytes()

    rows = [json.loads(line) for line in first.path.read_text(encoding="utf-8").splitlines()]
    validation_rows = [
        json.loads(line) for line in first.validation_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == len(validation_rows) == 1
    assert rows[0]["messages"][0]["role"] == "system"
    assert rows[0]["messages"][1]["content"].startswith("<video>\n")
    assert rows[0]["videos"] == [["media/real.ppm", "media/real-next.ppm"]]
    assert json.loads(rows[0]["messages"][2]["content"]) == {
        "label": "NONE",
        "rationale": "The block reaches the requested tray.",
        "score": 0.9,
    }
    assert rows[0]["evoworld_provenance"]["revision"] == 1
    assert validation_rows[0]["evoworld_provenance"]["split"] == "eval"
    assert str(tmp_path) not in first.path.read_text(encoding="utf-8")


@pytest.mark.parametrize("missing", REQUIRED_CONFIG_FIELDS)
def test_qwen_config_requires_every_public_recipe_value(tmp_path: Path, missing: str) -> None:
    _, examples = _validated(tmp_path)
    exported = export_ms_swift(
        examples, dataset_root=tmp_path / "dataset", export_path=tmp_path / "q.jsonl"
    )
    config_path = tmp_path / "qwen.toml"
    config_path.write_text(_config_text(exported, **{missing: None}), encoding="utf-8")
    with pytest.raises(EvaluatorConfigError, match=missing):
        load_qwen_config(config_path)


def test_qwen_plan_uses_pinned_public_ms_swift_schema_and_safe_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root, exported, config = _export_and_config(tmp_path)
    local_bin = tmp_path / "local-bin"
    local_bin.mkdir()
    local_swift = local_bin / "swift"
    local_swift.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    local_swift.chmod(0o755)
    monkeypatch.setenv("PATH", str(local_bin))
    plan = plan_qwen_training(exported, config, working_directory=tmp_path.resolve())
    assert plan.upstream_revision == MS_SWIFT_REVISION
    assert plan.argv[:21] == (
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
        "--model",
        "Qwen/Qwen3-VL-4B-Instruct",
        "--tuner_type",
        "lora",
        "--dataset",
        str(exported.path),
    )
    for flag in (
        "--learning_rate",
        "--lora_rank",
        "--lora_alpha",
        "--num_train_epochs",
        "--per_device_train_batch_size",
        "--gradient_accumulation_steps",
        "--max_length",
        "--seed",
        "--output_dir",
        "--max_pixels",
        "--model_kwargs",
        "--torch_dtype",
        "--device_map",
        "--save_strategy",
        "--save_total_limit",
        "--logging_steps",
        "--val_dataset",
        "--split_dataset_ratio",
        "--strict",
        "--add_version",
    ):
        assert flag in plan.argv
    assert plan.environment == {}
    assert plan.cwd == dataset_root.resolve()
    assert all(";" not in argument and "\n" not in argument for argument in plan.argv)
    model_kwargs = json.loads(plan.argv[plan.argv.index("--model_kwargs") + 1])
    assert model_kwargs == {
        "fps": 2.0,
        "fps_max_frames": 16,
        "video_max_pixels": 262144,
        "video_max_token_num": 256,
    }
    output = Path(plan.argv[plan.argv.index("--output_dir") + 1])
    assert output == tmp_path / "artifacts/evaluator/qwen3-vl-4b" / plan.job_id
    assert plan.expected_outputs == (Path("artifacts/evaluator/qwen3-vl-4b") / plan.job_id,)
    assert plan.manifest_sha256 == exported.sha256
    assert len(plan.job_id.removeprefix("qwen-evaluator-")) == 64
    assert plan.argv[plan.argv.index("--add_version") + 1] == "false"
    assert plan.argv[plan.argv.index("--save_strategy") + 1] == "epoch"
    assert "--save_steps" not in plan.argv

    monkeypatch.chdir(plan.cwd)
    row = json.loads(exported.path.read_text(encoding="utf-8").splitlines()[0])
    for frame in row["videos"][0]:
        with Path(frame).open("rb") as stream:
            assert stream.read(2) == b"P6"


def test_uvx_runtime_resolution_contract_precedes_sft_and_rejects_bad_versions() -> None:
    validate_qwen_runtime_versions(_resolved_runtime_versions())
    with pytest.raises(EvaluatorConfigError, match="transformers"):
        validate_qwen_runtime_versions(
            {
                "ms-swift": "4.3.1",
                "qwen-vl-utils": "0.0.14",
                "decord": "0.6.0",
                "transformers": "4.56.2",
                "torch": "2.8.0",
                "torchvision": "0.23.0",
                "cuda_available": "true",
            }
        )
    cpu_versions = _resolved_runtime_versions()
    cpu_versions["cuda_available"] = "false"
    with pytest.raises(EvaluatorConfigError, match="CUDA"):
        validate_qwen_runtime_versions(cpu_versions)


def test_runtime_rejects_a_mutated_recipe_before_sft(tmp_path: Path) -> None:
    _, exported, config = _export_and_config(tmp_path)
    plan = plan_qwen_training(exported, config, working_directory=tmp_path.resolve())
    argv = list(plan.argv)
    argv[argv.index("--learning_rate") + 1] = "0.9"
    mutated = replace(plan, argv=tuple(argv))
    called = False

    def command_runner(_: TrainingPlan) -> None:
        nonlocal called
        called = True

    with pytest.raises(EvaluatorConfigError, match="audited recipe"):
        run_qwen_training(
            mutated,
            exported,
            config,
            workspace_root=tmp_path,
            command_runner=command_runner,
        )
    assert not called


@pytest.mark.parametrize("split_path", ("path", "validation_path"))
def test_runtime_rejects_jsonl_changed_by_probe_before_sft(
    tmp_path: Path,
    split_path: str,
) -> None:
    _, exported, config = _export_and_config(tmp_path)
    plan = plan_qwen_training(exported, config, working_directory=tmp_path.resolve())
    output = Path(plan.argv[plan.argv.index("--output_dir") + 1])
    command_called = False

    def version_probe(_: tuple[str, ...]) -> dict[str, str]:
        getattr(exported, split_path).write_text("{}\n", encoding="utf-8")
        return _resolved_runtime_versions()

    def command_runner(_: TrainingPlan) -> None:
        nonlocal command_called
        command_called = True

    with pytest.raises(EvaluatorConfigError, match="stale export|split hash"):
        run_qwen_training(
            plan,
            exported,
            config,
            workspace_root=tmp_path,
            version_probe=version_probe,
            command_runner=command_runner,
        )
    assert not command_called
    assert not (output / ".evoworld-complete.json").exists()


@pytest.mark.parametrize("media_path", ("media/real.ppm", "media/synthetic-next.ppm"))
def test_runtime_rejects_media_changed_during_sft_without_completion_marker(
    tmp_path: Path,
    media_path: str,
) -> None:
    dataset_root, exported, config = _export_and_config(tmp_path)
    plan = plan_qwen_training(exported, config, working_directory=tmp_path.resolve())
    output = Path(plan.argv[plan.argv.index("--output_dir") + 1])

    def command_runner(actual: TrainingPlan) -> None:
        _write_completed_output(actual, exported, config, tmp_path)
        (dataset_root / media_path).write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")

    with pytest.raises(EvaluatorConfigError, match="media.*hash|dataset media"):
        run_qwen_training(
            plan,
            exported,
            config,
            workspace_root=tmp_path,
            version_probe=lambda _: _resolved_runtime_versions(),
            command_runner=command_runner,
        )
    assert not (output / ".evoworld-complete.json").exists()


def test_qwen_frame_lists_reject_single_or_over_budget_examples(tmp_path: Path) -> None:
    for frame_count, message in ((1, "at least 2"), (17, "fps_max_frames")):
        case = tmp_path / f"frames-{frame_count}"
        case.mkdir()
        exported, config = _export_with_train_frames(case, frame_count)
        with pytest.raises(EvaluatorConfigError, match=message):
            plan_qwen_training(exported, config, working_directory=case.resolve())


def test_qwen_frame_lists_use_documented_factor_two_padding(tmp_path: Path) -> None:
    exported, config = _export_with_train_frames(tmp_path, 3)
    plan = plan_qwen_training(exported, config, working_directory=tmp_path.resolve())
    assert plan.cwd == exported.dataset_root


def test_qwen_frame_budget_must_be_even_for_factor_two_padding(tmp_path: Path) -> None:
    _, exported, _ = _export_and_config(tmp_path)
    path = tmp_path / "odd-frame-budget.toml"
    path.write_text(_config_text(exported, fps_max_frames=15), encoding="utf-8")
    with pytest.raises(EvaluatorConfigError, match="fps_max_frames.*2"):
        load_qwen_config(path)


def test_plan_is_idempotent_and_rejects_stale_export_or_revision(tmp_path: Path) -> None:
    _, exported, config = _export_and_config(tmp_path)
    canonical_bytes = exported.path.read_bytes()
    first = plan_qwen_training(exported, config, working_directory=tmp_path.resolve())
    second = plan_qwen_training(exported, config, working_directory=tmp_path.resolve())
    assert first == second

    exported.path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(EvaluatorConfigError, match="stale export"):
        plan_qwen_training(exported, config, working_directory=tmp_path.resolve())

    exported.path.write_bytes(canonical_bytes)
    restored = replace(exported, dataset_revision="f" * 64)
    with pytest.raises(EvaluatorConfigError, match="revision"):
        plan_qwen_training(restored, config, working_directory=tmp_path.resolve())


def test_recipe_change_gets_a_distinct_hash_job_and_output(tmp_path: Path) -> None:
    _, exported, config = _export_and_config(tmp_path)
    first = plan_qwen_training(exported, config, working_directory=tmp_path.resolve())
    changed_path = tmp_path / "changed.toml"
    changed_path.write_text(_config_text(exported, learning_rate=0.0002), encoding="utf-8")
    changed = plan_qwen_training(
        exported,
        load_qwen_config(changed_path),
        working_directory=tmp_path.resolve(),
    )
    assert changed.manifest_sha256 == first.manifest_sha256 == exported.sha256
    assert changed.job_id != first.job_id
    assert changed.expected_outputs != first.expected_outputs

    other_workspace = tmp_path / "other-workspace"
    other_workspace.mkdir()
    relocated = plan_qwen_training(exported, config, working_directory=other_workspace.resolve())
    assert relocated.manifest_sha256 == first.manifest_sha256
    assert relocated.job_id != first.job_id


def test_existing_nonempty_job_output_is_rejected(tmp_path: Path) -> None:
    _, exported, config = _export_and_config(tmp_path)
    plan = plan_qwen_training(exported, config, working_directory=tmp_path.resolve())
    output = tmp_path / plan.expected_outputs[0]
    output.mkdir(parents=True)
    (output / "foreign-checkpoint").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(EvaluatorConfigError, match="nonempty output"):
        plan_qwen_training(exported, config, working_directory=tmp_path.resolve())


def test_hashed_qwen_job_keeps_completed_execution_idempotent(tmp_path: Path) -> None:
    _, exported, config = _export_and_config(tmp_path)
    plan = plan_qwen_training(exported, config, working_directory=tmp_path.resolve())
    job_path = write_job(tmp_path / "jobs", plan)
    assert write_job(tmp_path / "jobs", plan) == job_path
    calls: list[str] = []
    runtime_events: list[str] = []

    def version_probe(argv: tuple[str, ...]) -> dict[str, str]:
        assert argv[0] == "uvx"
        assert argv[-3:-1] == ("python", "-c")
        assert "swift" not in argv
        compile(argv[-1], "<qwen-runtime-probe>", "exec")
        runtime_events.append("probe")
        return _resolved_runtime_versions()

    def command_runner(actual: TrainingPlan) -> None:
        assert runtime_events == ["probe"]
        runtime_events.append("sft")
        _write_completed_output(actual, exported, config, tmp_path)

    def runner(_: TrainingPlan) -> tuple[Path, ...]:
        calls.append("run")
        return run_qwen_training(
            plan,
            exported,
            config,
            workspace_root=tmp_path,
            command_runner=command_runner,
            version_probe=version_probe,
        )

    first = execute_job(
        job_path,
        current_manifest_sha256=plan.manifest_sha256,
        current_upstream_revision=plan.upstream_revision,
        runner=runner,
    )
    second = execute_job(
        job_path,
        current_manifest_sha256=plan.manifest_sha256,
        current_upstream_revision=plan.upstream_revision,
        runner=runner,
    )
    assert first.status == second.status == "completed"
    assert calls == ["run"]
    assert runtime_events == ["probe", "sft"]


def test_completed_official_output_is_reusable_but_partial_or_foreign_output_is_rejected(
    tmp_path: Path,
) -> None:
    _, exported, config = _export_and_config(tmp_path)
    plan = plan_qwen_training(exported, config, working_directory=tmp_path.resolve())
    run_qwen_training(
        plan,
        exported,
        config,
        workspace_root=tmp_path,
        command_runner=lambda actual: _write_completed_output(actual, exported, config, tmp_path),
        version_probe=lambda _: _resolved_runtime_versions(),
    )
    assert plan_qwen_training(exported, config, working_directory=tmp_path.resolve()) == plan

    checkpoint = Path(plan.argv[plan.argv.index("--output_dir") + 1]) / "checkpoint-3"
    (checkpoint / "adapter_model.safetensors").write_bytes(b"foreign adapter")
    with pytest.raises(EvaluatorConfigError, match="marker"):
        plan_qwen_training(exported, config, working_directory=tmp_path.resolve())

    (checkpoint / "adapter_model.safetensors").write_bytes(b"")
    with pytest.raises(EvaluatorConfigError, match="completed|adapter"):
        plan_qwen_training(exported, config, working_directory=tmp_path.resolve())


def test_epoch_save_accepts_final_three_step_checkpoint_below_old_step_interval(
    tmp_path: Path,
) -> None:
    _, exported, _ = _export_and_config(tmp_path)
    config_path = tmp_path / "three-epoch-qwen.toml"
    config_path.write_text(
        _config_text(exported, num_train_epochs=3),
        encoding="utf-8",
    )
    config = load_qwen_config(config_path)
    plan = plan_qwen_training(exported, config, working_directory=tmp_path.resolve())
    outputs = run_qwen_training(
        plan,
        exported,
        config,
        workspace_root=tmp_path,
        version_probe=lambda _: _resolved_runtime_versions(),
        command_runner=lambda actual: _write_completed_output(
            actual,
            exported,
            config,
            tmp_path,
            global_step=3,
        ),
    )
    output = tmp_path / outputs[0]
    final_log = json.loads((output / "logging.jsonl").read_text(encoding="utf-8"))
    assert final_log["global_step"] == 3
    assert Path(final_log["last_model_checkpoint"]).name == "checkpoint-3"
    assert (output / "checkpoint-3" / "adapter_model.safetensors").is_file()


def test_completed_output_must_match_the_planned_recipe(tmp_path: Path) -> None:
    _, exported, config = _export_and_config(tmp_path)
    plan = plan_qwen_training(exported, config, working_directory=tmp_path.resolve())
    run_qwen_training(
        plan,
        exported,
        config,
        workspace_root=tmp_path,
        command_runner=lambda actual: _write_completed_output(actual, exported, config, tmp_path),
        version_probe=lambda _: _resolved_runtime_versions(),
    )
    args_path = Path(plan.argv[plan.argv.index("--output_dir") + 1]) / "args.json"
    args = json.loads(args_path.read_text(encoding="utf-8"))
    args["learning_rate"] = 0.9
    args_path.write_text(json.dumps(args), encoding="utf-8")
    with pytest.raises(EvaluatorConfigError, match="recipe"):
        plan_qwen_training(exported, config, working_directory=tmp_path.resolve())


def test_official_artifacts_without_success_marker_are_not_completed(tmp_path: Path) -> None:
    _, exported, config = _export_and_config(tmp_path)
    plan = plan_qwen_training(exported, config, working_directory=tmp_path.resolve())
    _write_completed_output(plan, exported, config, tmp_path)
    with pytest.raises(EvaluatorConfigError, match="completion marker"):
        plan_qwen_training(exported, config, working_directory=tmp_path.resolve())


@pytest.mark.parametrize("output_dir", ("../escape", "/tmp/escape", "nested\\escape"))
def test_output_path_must_remain_inside_working_directory(tmp_path: Path, output_dir: str) -> None:
    _, exported, _ = _export_and_config(tmp_path)
    config_path = tmp_path / "unsafe.toml"
    config_path.write_text(_config_text(exported, output_dir=output_dir), encoding="utf-8")
    with pytest.raises(EvaluatorConfigError, match="output_dir"):
        load_qwen_config(config_path)


def test_output_symlink_cannot_escape_working_directory(tmp_path: Path) -> None:
    _, exported, _ = _export_and_config(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "linked-output").symlink_to(outside, target_is_directory=True)
    config_path = tmp_path / "symlink.toml"
    config_path.write_text(
        _config_text(exported, output_dir="linked-output/checkpoint"),
        encoding="utf-8",
    )
    config = load_qwen_config(config_path)
    with pytest.raises(EvaluatorConfigError, match="output_dir"):
        plan_qwen_training(exported, config, working_directory=tmp_path.resolve())


@pytest.mark.parametrize(
    ("output_dir", "link_parent", "link_name"),
    (
        ("linked-output/nested", ".", "linked-output"),
        ("outputs/linked/nested", "outputs", "linked"),
    ),
)
def test_output_root_rejects_any_existing_symlink_ancestor_inside_workspace(
    tmp_path: Path,
    output_dir: str,
    link_parent: str,
    link_name: str,
) -> None:
    _, exported, _ = _export_and_config(tmp_path)
    unrelated = tmp_path / "unrelated-empty-output"
    unrelated.mkdir()
    parent = tmp_path / link_parent
    parent.mkdir(parents=True, exist_ok=True)
    (parent / link_name).symlink_to(unrelated, target_is_directory=True)
    config_path = tmp_path / "symlink-ancestor.toml"
    config_path.write_text(
        _config_text(exported, output_dir=output_dir),
        encoding="utf-8",
    )

    with pytest.raises(EvaluatorConfigError, match="output_dir.*symlink"):
        plan_qwen_training(
            exported,
            load_qwen_config(config_path),
            working_directory=tmp_path.resolve(),
        )

    assert list(unrelated.iterdir()) == []


def test_existing_job_output_symlink_inside_workspace_is_never_reused(tmp_path: Path) -> None:
    _, exported, config = _export_and_config(tmp_path)
    plan = plan_qwen_training(exported, config, working_directory=tmp_path.resolve())
    output = tmp_path / plan.expected_outputs[0]
    output.parent.mkdir(parents=True)
    unrelated = tmp_path / "unrelated-empty-job"
    unrelated.mkdir()
    output.symlink_to(unrelated, target_is_directory=True)

    with pytest.raises(EvaluatorConfigError, match="job output.*symlink"):
        plan_qwen_training(exported, config, working_directory=tmp_path.resolve())

    assert list(unrelated.iterdir()) == []


def test_config_rejects_wrong_version_model_and_unknown_fields(tmp_path: Path) -> None:
    _, exported, _ = _export_and_config(tmp_path)
    for changes, message in (
        ({"ms_swift_version": "4.3.0"}, "4.3.1"),
        ({"base_model": "Qwen/Qwen3-VL-8B-Instruct"}, "4B"),
        ({"tuner_type": "full"}, "LoRA"),
    ):
        path = tmp_path / f"invalid-{len(message)}.toml"
        path.write_text(_config_text(exported, **changes), encoding="utf-8")
        with pytest.raises(EvaluatorConfigError, match=message):
            load_qwen_config(path)

    steps_path = tmp_path / "invalid-save-strategy.toml"
    steps_path.write_text(_config_text(exported, save_strategy="steps"), encoding="utf-8")
    with pytest.raises(EvaluatorConfigError, match="save_strategy.*epoch"):
        load_qwen_config(steps_path)

    path = tmp_path / "unknown.toml"
    path.write_text(_config_text(exported) + "unknown = 1\n", encoding="utf-8")
    with pytest.raises(EvaluatorConfigError, match="unknown"):
        load_qwen_config(path)


def test_qwen3_video_token_and_pixel_budgets_must_be_consistent(tmp_path: Path) -> None:
    _, exported, _ = _export_and_config(tmp_path)
    path = tmp_path / "inconsistent-video-budget.toml"
    path.write_text(_config_text(exported, video_max_pixels=200704), encoding="utf-8")
    with pytest.raises(EvaluatorConfigError, match="video.*consistent"):
        load_qwen_config(path)


@pytest.mark.parametrize("field", ("learning_rate", "fps"))
@pytest.mark.parametrize("literal", ("nan", "inf", "-inf", "true"))
def test_float_recipe_values_must_be_finite_numbers(
    tmp_path: Path, field: str, literal: str
) -> None:
    _, exported, _ = _export_and_config(tmp_path)
    path = tmp_path / f"invalid-{field}-{literal}.toml"
    text = _config_text(exported)
    old_line = next(line for line in text.splitlines() if line.startswith(f"{field} = "))
    path.write_text(text.replace(old_line, f"{field} = {literal}"), encoding="utf-8")
    with pytest.raises(EvaluatorConfigError, match=field):
        load_qwen_config(path)


@pytest.mark.parametrize(
    ("field", "literal"),
    (
        ("num_train_epochs", "2.0"),
        ("num_train_epochs", "true"),
        ("num_train_epochs", "0"),
        ("lora_rank", "true"),
        ("max_pixels", "true"),
        ("video_max_token_num", "true"),
    ),
)
def test_integer_recipe_values_reject_float_or_bool(
    tmp_path: Path, field: str, literal: str
) -> None:
    _, exported, _ = _export_and_config(tmp_path)
    path = tmp_path / f"invalid-{field}.toml"
    text = _config_text(exported)
    old_line = next(line for line in text.splitlines() if line.startswith(f"{field} = "))
    path.write_text(text.replace(old_line, f"{field} = {literal}"), encoding="utf-8")
    with pytest.raises(EvaluatorConfigError, match=field):
        load_qwen_config(path)


@pytest.mark.parametrize(
    ("field", "literal"),
    (
        ("learning_rate", "0.0"),
        ("learning_rate", "1.1"),
        ("fps", "0.0"),
        ("fps", "241.0"),
    ),
)
def test_float_recipe_values_must_be_inside_audited_ranges(
    tmp_path: Path, field: str, literal: str
) -> None:
    _, exported, _ = _export_and_config(tmp_path)
    path = tmp_path / f"out-of-range-{field}.toml"
    text = _config_text(exported)
    old_line = next(line for line in text.splitlines() if line.startswith(f"{field} = "))
    path.write_text(text.replace(old_line, f"{field} = {literal}"), encoding="utf-8")
    with pytest.raises(EvaluatorConfigError, match=field):
        load_qwen_config(path)

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import imageio_ffmpeg
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from evoworld.contracts import DiagnosisLabel
import integrations.ctrl_world.train_bridge as ctrl_bridge
from integrations.ctrl_world.train_bridge import (
    CtrlWorldBridgeError,
    OfficialExecutionContext,
    OfficialCtrlWorldComponents,
    build_official_args,
    execute_request,
    launch_training,
    run_preparation,
    run_bounded_training,
)
from training.common.schema import TrainingManifest, TrainingSample
from training.world_model.ctrl_world import (
    CTRL_WORLD_REVISION,
    CTRL_WORLD_SUBMODULE,
    CtrlWorldConfigError,
    CtrlWorldPrepareConfig,
    CtrlWorldTrainConfig,
    fingerprint_ctrl_world_environment,
    inspect_ctrl_world_media,
    load_ctrl_world_config,
    plan_ctrl_world_base_training,
    plan_ctrl_world_prepare,
    plan_ctrl_world_training,
    sha256_file,
    validate_ctrl_world_dataset,
    verify_request_bindings,
)


ROOT = Path(__file__).parents[2].resolve()
CONFIG_PATH = ROOT / "configs/world_model/ctrl_world.toml"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metadata_context(value: OfficialExecutionContext | None) -> OfficialExecutionContext:
    assert value is not None
    return value


def _write_metadata_from_context(context: OfficialExecutionContext) -> None:
    flags = ctrl_bridge._directory_flags()
    staging_parent_fd = os.open("dataset_meta_info", flags, dir_fd=context.working_directory_fd)
    try:
        try:
            os.mkdir("evoworld", mode=0o700, dir_fd=staging_parent_fd)
        except FileExistsError:
            pass
        output_fd = os.open("evoworld", flags, dir_fd=staging_parent_fd)
        try:
            ctrl_bridge._publish_json_exclusive_at(output_fd, "train_sample.json", [])
            ctrl_bridge._publish_json_exclusive_at(output_fd, "val_sample.json", [])
        finally:
            os.close(output_fd)
    finally:
        os.close(staging_parent_fd)


def _write_video(path: Path, frames: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(path), (320, 192), fps=5, codec="libx264")
    writer.send(None)
    frame = bytes([17]) * (320 * 192 * 3)
    for _ in range(frames):
        writer.send(frame)
    writer.close()


def _write_backend_python(tmp_path: Path) -> Path:
    modules = tmp_path / "backend_modules"
    (modules / "accelerate/commands").mkdir(parents=True)
    (modules / "imageio").mkdir(parents=True)
    (modules / "torch/nn").mkdir(parents=True)
    (modules / "torch/utils/data").mkdir(parents=True)
    for relative in (
        "accelerate/__init__.py",
        "accelerate/commands/__init__.py",
        "accelerate/accelerator.py",
    ):
        (modules / relative).write_text("\n", encoding="utf-8")
    (modules / "accelerate/commands/launch.py").write_text(
        """import json, os, sys
capture = os.environ.get('EVOWORLD_ACCELERATE_CAPTURE')
if capture:
    open(capture, 'w', encoding='utf-8').write(json.dumps(sys.argv[1:]))
""",
        encoding="utf-8",
    )
    (modules / "torch/__init__.py").write_text(
        """import json
class Tensor:
    def __init__(self, shape, dtype): self.shape, self.dtype = tuple(shape), dtype
def load(stream, map_location=None):
    value = json.load(stream)
    return Tensor(value['shape'], value['dtype'])
        """,
        encoding="utf-8",
    )
    for relative in (
        "torch/optim.py",
        "torch/serialization.py",
        "torch/hidden_runtime.py",
        "torch/nn/__init__.py",
        "torch/utils/__init__.py",
        "torch/utils/data/__init__.py",
    ):
        (modules / relative).write_text("\n", encoding="utf-8")
    (modules / "imageio/__init__.py").write_text("\n", encoding="utf-8")
    (modules / "imageio/v3.py").write_text(
        """import imageio_ffmpeg
import numpy as np
def imiter(path):
    frames = imageio_ffmpeg.read_frames(path, pix_fmt='rgb24')
    metadata = next(frames)
    width, height = metadata['size']
    for frame in frames:
        yield np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3)
""",
        encoding="utf-8",
    )
    stub_modules = (
        "decord",
        "diffusers",
        "einops",
        "mediapy",
        "omegaconf",
        "pandas",
        "safetensors",
        "scipy",
        "swanlab",
        "torchvision",
        "tqdm",
        "transformers",
        "wandb",
    )
    for name in stub_modules:
        (modules / f"{name}.py").write_text("\n", encoding="utf-8")
    distribution_files = {
        "accelerate": (
            "accelerate/__init__.py",
            "accelerate/accelerator.py",
            "accelerate/commands/__init__.py",
            "accelerate/commands/launch.py",
        ),
        "imageio": ("imageio/__init__.py", "imageio/v3.py"),
        "torch": (
            "torch/__init__.py",
            "torch/optim.py",
            "torch/serialization.py",
            "torch/hidden_runtime.py",
            "torch/nn/__init__.py",
            "torch/utils/__init__.py",
            "torch/utils/data/__init__.py",
        ),
        **{name: (f"{name}.py",) for name in stub_modules},
    }
    distributions = tuple(distribution_files)
    for name in distributions:
        info = modules / f"{name.replace('-', '_')}-0.0.dist-info"
        metadata = info / "METADATA"
        metadata.parent.mkdir(parents=True)
        metadata.write_text(
            f"Metadata-Version: 2.1\nName: {name.replace('_', '-')}\nVersion: 0.0\n",
            encoding="utf-8",
        )
        record_rows = [f"{relative},sha256=AA,1" for relative in distribution_files[name]]
        record_rows.append(f"{info.name}/METADATA,sha256=AA,1")
        record_rows.append(f"{info.name}/RECORD,,")
        (info / "RECORD").write_text("\n".join(record_rows) + "\n", encoding="utf-8")
    executable = tmp_path / "ctrl-python"
    executable.write_text(
        f"#!/bin/sh\nPYTHONPATH='{modules}' exec '{sys.executable}' \"$@\"\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable.resolve()


def _write_source(repository: Path) -> Path:
    source = repository / CTRL_WORLD_SUBMODULE
    for relative in (
        "config.py",
        "scripts/train_wm.py",
        "models/ctrl_world.py",
        "dataset/dataset_droid_exp33.py",
        "dataset_example/extract_latent.py",
        "dataset_meta_info/create_meta_info.py",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# pinned source: {relative}\n", encoding="utf-8")
    return source.resolve()


def _audit(_: Path) -> dict[str, str]:
    return {CTRL_WORLD_SUBMODULE: CTRL_WORLD_REVISION}


def _sample_digest(root: Path, sample: TrainingSample) -> str:
    digest = sha256()
    media = (*sample.observation_paths, sample.action_path)
    for index, relative in enumerate(media):
        assert relative is not None
        digest.update(f"{index}:{sample.dataset_root / relative}\0".encode())
        digest.update((root / sample.dataset_root / relative).read_bytes())
    return digest.hexdigest()


def _write_episode(
    dataset: Path, split: str, episode_id: int, instruction: str
) -> dict[str, object]:
    videos: list[dict[str, str]] = []
    latents: list[dict[str, str]] = []
    for camera in range(3):
        video = Path(f"videos/{split}/{episode_id}/{camera}.mp4")
        latent = Path(f"latent_videos/{split}/{episode_id}/{camera}.pt")
        _write_video(dataset / video)
        _write_json(dataset / latent, {"shape": [3, 4, 24, 40], "dtype": "torch.float32"})
        videos.append({"video_path": video.as_posix()})
        latents.append({"latent_video_path": latent.as_posix()})
    annotation: dict[str, object] = {
        "texts": [instruction],
        "episode_id": episode_id,
        "video_length": 3,
        "state_length": 3,
        "raw_length": 7,
        "videos": videos,
        "latent_videos": latents,
        "states": [[0.0] * 7 for _ in range(3)],
        "observation.state.cartesian_position": [[0.0] * 6 for _ in range(7)],
        "observation.state.joint_position": [[0.0] * 7 for _ in range(7)],
        "observation.state.gripper_position": [0.0] * 7,
    }
    _write_json(dataset / f"annotation/{split}/{episode_id}.json", annotation)
    return annotation


def _fixture(
    tmp_path: Path,
    *,
    diagnosis: DiagnosisLabel = DiagnosisLabel.WMF,
) -> tuple[CtrlWorldTrainConfig, TrainingManifest, Path, Path]:
    repository = (tmp_path / "repo").resolve()
    source = _write_source(repository)
    python = _write_backend_python(tmp_path)
    data_parent = (tmp_path / "datasets").resolve()
    dataset = data_parent / "evoworld_wmf"
    metadata_parent = (tmp_path / "metadata").resolve()
    metadata = metadata_parent / "evoworld_wmf"
    _write_episode(dataset, "train", 1, "pick the cube")
    _write_episode(dataset, "train", 2, "extra trajectory")
    _write_episode(dataset, "val", 99, "place the cube")
    _write_json(
        metadata / "train_sample.json",
        [
            {"episode_id": 1, "frame_ids": [0]},
            {"episode_id": 2, "frame_ids": [0]},
        ],
    )
    _write_json(metadata / "val_sample.json", [{"episode_id": 99, "frame_ids": [0]}])
    _write_json(metadata / "stat.json", {"state_01": [-1.0] * 7, "state_99": [1.0] * 7})
    observations = tuple(Path(f"videos/train/1/{camera}.mp4") for camera in range(3)) + tuple(
        Path(f"latent_videos/train/1/{camera}.pt") for camera in range(3)
    )
    sample = TrainingSample(
        schema_version=1,
        sample_id="wmf-1",
        target="ctrl_world",
        task_id="pick",
        instruction="pick the cube",
        skill_family="pick",
        scene_id="scene-1",
        trajectory_id="1",
        dataset_root=Path("evoworld_wmf"),
        observation_paths=observations,
        action_path=Path("annotation/train/1.json"),
        action_schema="cartesian_pose_gripper",
        action_dimension=7,
        action_horizon=5,
        source_type="synthetic",
        source_checkpoint="ctrl-world-base",
        diagnosis=diagnosis,
        split="train",
        content_sha256="0" * 64,
    )
    sample = replace(sample, content_sha256=_sample_digest(data_parent, sample))
    manifest = TrainingManifest(1, "wmf", (sample,))
    manifest_path = metadata / "evoworld_manifest.json"
    _write_json(manifest_path, manifest.to_dict())
    models = tmp_path / "models"
    for name in ("svd", "clip"):
        path = models / name
        path.mkdir(parents=True)
        (path / "config.json").write_text(name, encoding="utf-8")
    for name in ("pi.pt", "adapter.pt", "ctrl-world.pt"):
        (models / name).write_bytes(name.encode())
    config = CtrlWorldTrainConfig(
        dataset_root_path=data_parent,
        dataset_meta_info_path=metadata_parent,
        manifest_path=manifest_path,
        output_root=(tmp_path / "outputs").resolve(),
        svd_model_path=(models / "svd").resolve(),
        clip_model_path=(models / "clip").resolve(),
        policy_checkpoint_path=(models / "pi.pt").resolve(),
        action_adapter_path=(models / "adapter.pt").resolve(),
        synthetic_output_root=(tmp_path / "synthetic").resolve(),
        starting_checkpoint=(models / "ctrl-world.pt").resolve(),
        ctrl_world_python=python,
        dataset_name="evoworld_wmf",
        annotation_name="annotation",
        tag="evoworld_wmf",
        wandb_run_name="evoworld_wmf",
        wandb_project_name="evoworld",
        max_train_steps=5000,
        world_size=1,
    )
    return config, manifest, repository, source


def _request(plan: object) -> tuple[Path, dict[str, object]]:
    argv = plan.argv
    path = Path(argv[argv.index("--request-path") + 1])
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_prepare_uses_verified_python_absolute_scripts_and_external_workspace(
    tmp_path: Path,
) -> None:
    repository = (tmp_path / "repo").resolve()
    source = _write_source(repository)
    python = _write_backend_python(tmp_path)
    workspace = (tmp_path / "workspace").resolve()
    droid = (tmp_path / "droid").resolve()
    svd = (tmp_path / "svd").resolve()
    for path in (workspace, droid, svd):
        path.mkdir()
    config = CtrlWorldPrepareConfig(
        droid_hf_path=droid,
        droid_output_path=workspace / "datasets/evoworld",
        svd_model_path=svd,
        dataset_name="evoworld",
        ctrl_world_python=python,
        workspace_root=workspace,
        world_size=1,
    )
    extraction, metadata = plan_ctrl_world_prepare(
        config,
        ctrl_world_root=source,
        repository_root=repository,
        revision_auditor=_audit,
    )
    assert extraction.argv[:3] == (
        str(python),
        str(ROOT / "integrations/ctrl_world/train_bridge.py"),
        "prepare-latents",
    )
    assert metadata.argv[:3] == (
        str(python),
        str(ROOT / "integrations/ctrl_world/train_bridge.py"),
        "bind-metadata",
    )
    assert metadata.cwd == workspace
    assert metadata.expected_outputs[0] == Path("dataset_meta_info/evoworld/train_sample.json")
    for plan, operation in (
        (extraction, "prepare-latents"),
        (metadata, "bind-metadata"),
    ):
        request_path, request = _request(plan)
        assert request["operation"] == operation
        assert sha256_file(request_path) == plan.argv[-1]
        assert request["source_blobs"]["dataset_example/extract_latent.py"]


def _materialize_prepared_tree(root: Path) -> None:
    for relative in (
        "annotation/train/1.json",
        "annotation/val/2.json",
        "videos/train/1/0.mp4",
        "latent_videos/train/1/0.pt",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())


def test_preparation_is_two_stage_content_bound_and_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = (tmp_path / "repo").resolve()
    source = _write_source(repository)
    python = _write_backend_python(tmp_path)
    workspace = (tmp_path / "workspace").resolve()
    droid = (tmp_path / "droid").resolve()
    svd = (tmp_path / "svd").resolve()
    for path in (workspace, droid, svd):
        path.mkdir()
    extraction, metadata = plan_ctrl_world_prepare(
        CtrlWorldPrepareConfig(
            droid_hf_path=droid,
            droid_output_path=workspace / "datasets/evoworld",
            svd_model_path=svd,
            dataset_name="evoworld",
            ctrl_world_python=python,
            workspace_root=workspace,
            world_size=1,
        ),
        ctrl_world_root=source,
        repository_root=repository,
        revision_auditor=_audit,
    )
    metadata_path, metadata_seed = _request(metadata)
    with pytest.raises(CtrlWorldBridgeError, match="receipt|latents"):
        run_preparation(
            "bind-metadata",
            metadata_path,
            metadata.argv[-1],
            revision_auditor=_audit,
            command_runner=lambda _, __: pytest.fail("metadata command ran before receipt"),
        )

    extraction_path, extraction_seed = _request(extraction)
    prepared = Path(extraction_seed["output_bindings"]["droid_output_path"])
    commands: list[tuple[str, ...]] = []

    def extract_runner(command: object, context: OfficialExecutionContext | None) -> None:
        assert context is None
        commands.append(tuple(command))  # type: ignore[arg-type]
        _materialize_prepared_tree(prepared)

    run_preparation(
        "prepare-latents",
        extraction_path,
        extraction.argv[-1],
        revision_auditor=_audit,
        command_runner=extract_runner,
    )
    receipt_path = Path(extraction_seed["output_bindings"]["latents_receipt_path"])
    receipt = json.loads(receipt_path.read_text())
    assert receipt["prepared_inventory_sha256"]
    assert receipt["request_sha256"] == extraction.argv[-1]
    assert commands[0][:3] == (str(python), "-m", "accelerate.commands.launch")

    metadata_output = Path(metadata_seed["output_bindings"]["metadata_output_path"])
    metadata_claim = Path(metadata_seed["output_bindings"]["metadata_claim_path"])
    metadata_completion = Path(metadata_seed["output_bindings"]["metadata_completion_path"])
    metadata_quarantine = Path(metadata_seed["output_bindings"]["metadata_quarantine_root"])
    prepared_file = prepared / "annotation/train/1.json"
    original_prepared = prepared_file.read_bytes()
    prepared_file.write_bytes(b"mutated-after-receipt")
    with pytest.raises(CtrlWorldBridgeError, match="receipt|inventory"):
        run_preparation(
            "bind-metadata",
            metadata_path,
            metadata.argv[-1],
            revision_auditor=_audit,
            command_runner=lambda _, __: pytest.fail("metadata used changed prepared input"),
        )
    prepared_file.write_bytes(original_prepared)
    metadata_output.mkdir(parents=True)
    with pytest.raises(CtrlWorldBridgeError, match="existing|overwrite"):
        run_preparation(
            "bind-metadata",
            metadata_path,
            metadata.argv[-1],
            revision_auditor=_audit,
            command_runner=lambda _, __: pytest.fail("metadata overwrote stale output"),
        )
    metadata_output.rmdir()
    metadata_claim.write_text("claimed", encoding="utf-8")
    with pytest.raises(CtrlWorldBridgeError, match="claimed"):
        run_preparation(
            "bind-metadata",
            metadata_path,
            metadata.argv[-1],
            revision_auditor=_audit,
            command_runner=lambda _, __: pytest.fail("metadata ignored claim"),
        )
    metadata_claim.unlink()

    def drifting_metadata_runner(command: object, context: OfficialExecutionContext | None) -> None:
        commands.append(tuple(command))  # type: ignore[arg-type]
        _write_metadata_from_context(_metadata_context(context))
        prepared_file.write_bytes(b"changed-during-official-command")

    with pytest.raises(CtrlWorldBridgeError, match="receipt|inventory|drift"):
        run_preparation(
            "bind-metadata",
            metadata_path,
            metadata.argv[-1],
            revision_auditor=_audit,
            command_runner=drifting_metadata_runner,
        )
    assert not metadata_completion.exists()
    assert not metadata_output.exists()
    quarantined = tuple(metadata_quarantine.glob("*/staging/dataset_meta_info/evoworld"))
    assert len(quarantined) == 1
    assert (quarantined[0] / "train_sample.json").is_file()
    assert tuple(metadata_quarantine.glob("*/invalid.json"))
    prepared_file.write_bytes(original_prepared)

    outside = tmp_path / "outside-metadata"
    outside_output = outside / "evoworld"
    _write_json(outside_output / "train_sample.json", [])
    _write_json(outside_output / "val_sample.json", [])
    outside_sentinel = outside_output / "owner.bin"
    outside_sentinel.write_bytes(b"outside-owner")

    def replaced_ancestor_runner(command: object, context: OfficialExecutionContext | None) -> None:
        working_fd = _metadata_context(context).working_directory_fd
        os.rmdir("dataset_meta_info", dir_fd=working_fd)
        os.symlink(outside, "dataset_meta_info", target_is_directory=True, dir_fd=working_fd)

    with pytest.raises(
        CtrlWorldBridgeError,
        match=r"metadata (?:staging output is no longer job-owned|output is incomplete)",
    ):
        run_preparation(
            "bind-metadata",
            metadata_path,
            metadata.argv[-1],
            revision_auditor=_audit,
            command_runner=replaced_ancestor_runner,
        )
    assert outside_sentinel.read_bytes() == b"outside-owner"
    assert outside_output.is_dir()
    assert not metadata_output.exists()

    moved_quarantine = metadata_quarantine.with_name("quarantine-bound-fd")

    def quarantine_ancestor_race_runner(
        command: object, context: OfficialExecutionContext | None
    ) -> None:
        _write_metadata_from_context(_metadata_context(context))
        metadata_quarantine.rename(moved_quarantine)
        metadata_quarantine.mkdir()
        (metadata_quarantine / "external.bin").write_bytes(b"external-quarantine")
        prepared_file.write_bytes(b"quarantine-race-drift")

    with pytest.raises(CtrlWorldBridgeError, match="receipt|inventory|drift"):
        run_preparation(
            "bind-metadata",
            metadata_path,
            metadata.argv[-1],
            revision_auditor=_audit,
            command_runner=quarantine_ancestor_race_runner,
        )
    assert (metadata_quarantine / "external.bin").read_bytes() == b"external-quarantine"
    assert len(tuple(moved_quarantine.glob("*/invalid.json"))) == 3
    prepared_file.write_bytes(original_prepared)

    sentinel = b"external-owner"

    def concurrent_metadata_runner(
        command: object, context: OfficialExecutionContext | None
    ) -> None:
        commands.append(tuple(command))  # type: ignore[arg-type]
        _write_metadata_from_context(_metadata_context(context))

    ensure_absent_at = ctrl_bridge._ensure_absent_at
    final_checks = 0

    def race_after_publish_precheck(parent_fd: int, name: str) -> None:
        nonlocal final_checks
        ensure_absent_at(parent_fd, name)
        if name == "evoworld":
            final_checks += 1
        if final_checks == 4:
            metadata_output.mkdir()
            (metadata_output / "sentinel.bin").write_bytes(sentinel)

    monkeypatch.setattr(ctrl_bridge, "_ensure_absent_at", race_after_publish_precheck)

    with pytest.raises(CtrlWorldBridgeError, match="existing|concurrently|overwrite"):
        run_preparation(
            "bind-metadata",
            metadata_path,
            metadata.argv[-1],
            revision_auditor=_audit,
            command_runner=concurrent_metadata_runner,
        )
    monkeypatch.setattr(ctrl_bridge, "_ensure_absent_at", ensure_absent_at)
    assert (metadata_output / "sentinel.bin").read_bytes() == sentinel
    assert not metadata_completion.exists()
    assert len(tuple(metadata_quarantine.glob("*/invalid.json"))) == 1
    (metadata_output / "sentinel.bin").unlink()
    metadata_output.rmdir()

    def metadata_runner(command: object, context: OfficialExecutionContext | None) -> None:
        commands.append(tuple(command))  # type: ignore[arg-type]
        _write_metadata_from_context(_metadata_context(context))

    run_preparation(
        "bind-metadata",
        metadata_path,
        metadata.argv[-1],
        revision_auditor=_audit,
        command_runner=metadata_runner,
    )
    completion = json.loads(metadata_completion.read_text())
    bound_path = Path(completion["metadata_request_path"])
    bound = json.loads(bound_path.read_text())
    assert bound["operation"] == "prepare-metadata"
    assert bound["prepared_input"]["inventory_sha256"] == receipt["prepared_inventory_sha256"]
    assert bound["prepared_input"]["receipt_sha256"] == sha256_file(receipt_path)
    assert completion["prepared_inventory_sha256"] == receipt["prepared_inventory_sha256"]
    assert metadata_output.is_dir()
    assert Path(bound["metadata_staging_path"]) != metadata_output


@pytest.mark.parametrize("replacement", ("symlink", "directory"))
def test_metadata_rejects_replaced_workspace_identity_before_official(
    tmp_path: Path, replacement: str
) -> None:
    repository = (tmp_path / "repo").resolve()
    source = _write_source(repository)
    python = _write_backend_python(tmp_path)
    workspace = (tmp_path / "workspace").resolve()
    droid = (tmp_path / "droid").resolve()
    svd = (tmp_path / "svd").resolve()
    for path in (workspace, droid, svd):
        path.mkdir()
    extraction, metadata = plan_ctrl_world_prepare(
        CtrlWorldPrepareConfig(
            droid_hf_path=droid,
            droid_output_path=workspace / "datasets/evoworld",
            svd_model_path=svd,
            dataset_name="evoworld",
            ctrl_world_python=python,
            workspace_root=workspace,
            world_size=1,
        ),
        ctrl_world_root=source,
        repository_root=repository,
        revision_auditor=_audit,
    )
    extraction_path, extraction_seed = _request(extraction)
    prepared = Path(extraction_seed["output_bindings"]["droid_output_path"])
    run_preparation(
        "prepare-latents",
        extraction_path,
        extraction.argv[-1],
        revision_auditor=_audit,
        command_runner=lambda _, __: _materialize_prepared_tree(prepared),
    )
    metadata_path, _ = _request(metadata)
    moved = tmp_path / "workspace-original"

    def replacing_auditor(root: Path) -> Mapping[str, str]:
        result = _audit(root)
        workspace.rename(moved)
        if replacement == "symlink":
            workspace.symlink_to(moved, target_is_directory=True)
        else:
            workspace.mkdir()
        return result

    with pytest.raises(CtrlWorldBridgeError, match="trusted|root|binding"):
        run_preparation(
            "bind-metadata",
            metadata_path,
            metadata.argv[-1],
            revision_auditor=replacing_auditor,
            command_runner=lambda _, __: pytest.fail("official command reached"),
        )


def test_official_executor_keeps_bound_tree_after_workspace_path_replacement(
    tmp_path: Path,
) -> None:
    repository = (tmp_path / "repo").resolve()
    source = _write_source(repository)
    python = _write_backend_python(tmp_path)
    workspace = (tmp_path / "workspace").resolve()
    moved = (tmp_path / "workspace-bound").resolve()
    droid = (tmp_path / "droid").resolve()
    svd = (tmp_path / "svd").resolve()
    for path in (workspace, droid, svd):
        path.mkdir()
    metadata_script = source / "dataset_meta_info/create_meta_info.py"
    metadata_script.write_text(
        f"""import argparse
import json
import os
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--droid_output_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    args = parser.parse_args()
    os.rename({str(workspace)!r}, {str(moved)!r})
    replacement = Path({str(workspace)!r})
    replacement.mkdir()
    (replacement / "owner.bin").write_bytes(b"replacement-owner")
    prepared_bytes = (Path(args.droid_output_path) / "annotation/train/1.json").read_bytes()
    output = Path("dataset_meta_info") / args.dataset_name
    output.mkdir(parents=True)
    (output / "train_sample.json").write_text(json.dumps([]))
    (output / "val_sample.json").write_text(json.dumps([]))
    (output / "prepared.bin").write_bytes(prepared_bytes)
""",
        encoding="utf-8",
    )
    extraction, metadata = plan_ctrl_world_prepare(
        CtrlWorldPrepareConfig(
            droid_hf_path=droid,
            droid_output_path=workspace / "datasets/evoworld",
            svd_model_path=svd,
            dataset_name="evoworld",
            ctrl_world_python=python,
            workspace_root=workspace,
            world_size=1,
        ),
        ctrl_world_root=source,
        repository_root=repository,
        revision_auditor=_audit,
    )
    extraction_path, extraction_seed = _request(extraction)
    prepared = Path(extraction_seed["output_bindings"]["droid_output_path"])
    run_preparation(
        "prepare-latents",
        extraction_path,
        extraction.argv[-1],
        revision_auditor=_audit,
        command_runner=lambda _, __: _materialize_prepared_tree(prepared),
    )
    prepared_bytes = (prepared / "annotation/train/1.json").read_bytes()
    metadata_path, metadata_seed = _request(metadata)
    backend_original = python.with_name("ctrl-python-bound")
    script_original = metadata_script.with_name("create_meta_info-bound.py")
    backend_marker = tmp_path / "replacement-backend-ran"
    script_marker = tmp_path / "replacement-script-ran"

    def racing_runner(command: Sequence[str], context: OfficialExecutionContext | None) -> None:
        python.rename(backend_original)
        python.write_text(f"#!/bin/sh\ntouch '{backend_marker}'\nexit 97\n", encoding="utf-8")
        python.chmod(0o755)
        metadata_script.rename(script_original)
        metadata_script.write_text(
            f"from pathlib import Path\nPath({str(script_marker)!r}).write_text('ran')\n",
            encoding="utf-8",
        )
        ctrl_bridge._run_command(command, _metadata_context(context))

    with pytest.raises(
        CtrlWorldBridgeError,
        match="receipt|prepared|inventory|child|unsupported|trusted workspace root identity changed",
    ):
        run_preparation(
            "bind-metadata",
            metadata_path,
            metadata.argv[-1],
            revision_auditor=_audit,
            command_runner=racing_runner,
        )
    assert not backend_marker.exists()
    assert not script_marker.exists()
    completion = Path(metadata_seed["output_bindings"]["metadata_completion_path"])
    assert not completion.exists()
    if sys.platform.startswith("linux") and os.execve in os.supports_fd:
        assert (workspace / "owner.bin").read_bytes() == b"replacement-owner"
        quarantined = tuple(
            moved.glob(
                ".evoworld/quarantine/evoworld-*/*/staging/dataset_meta_info/evoworld/prepared.bin"
            )
        )
        assert len(quarantined) == 1
        assert quarantined[0].read_bytes() == prepared_bytes
    else:
        assert workspace.is_dir()
        assert not moved.exists()
        assert not (workspace / "owner.bin").exists()


def test_post_official_root_recheck_rejects_workspace_replacement(tmp_path: Path) -> None:
    repository = (tmp_path / "repo").resolve()
    source = _write_source(repository)
    python = _write_backend_python(tmp_path)
    workspace = (tmp_path / "workspace").resolve()
    moved = (tmp_path / "workspace-bound").resolve()
    droid = (tmp_path / "droid").resolve()
    svd = (tmp_path / "svd").resolve()
    for path in (workspace, droid, svd):
        path.mkdir()
    extraction, metadata = plan_ctrl_world_prepare(
        CtrlWorldPrepareConfig(
            droid_hf_path=droid,
            droid_output_path=workspace / "datasets/evoworld",
            svd_model_path=svd,
            dataset_name="evoworld",
            ctrl_world_python=python,
            workspace_root=workspace,
            world_size=1,
        ),
        ctrl_world_root=source,
        repository_root=repository,
        revision_auditor=_audit,
    )
    extraction_path, extraction_seed = _request(extraction)
    prepared = Path(extraction_seed["output_bindings"]["droid_output_path"])
    run_preparation(
        "prepare-latents",
        extraction_path,
        extraction.argv[-1],
        revision_auditor=_audit,
        command_runner=lambda _, __: _materialize_prepared_tree(prepared),
    )
    metadata_path, metadata_seed = _request(metadata)

    def moving_runner(command: Sequence[str], context: OfficialExecutionContext | None) -> None:
        _write_metadata_from_context(_metadata_context(context))
        workspace.rename(moved)
        workspace.mkdir()
        (workspace / "owner.bin").write_bytes(b"replacement-owner")

    with pytest.raises(CtrlWorldBridgeError, match="trusted|root|identity|changed"):
        run_preparation(
            "bind-metadata",
            metadata_path,
            metadata.argv[-1],
            revision_auditor=_audit,
            command_runner=moving_runner,
        )
    assert (workspace / "owner.bin").read_bytes() == b"replacement-owner"
    assert not Path(metadata_seed["output_bindings"]["metadata_output_path"]).exists()
    assert not Path(metadata_seed["output_bindings"]["metadata_completion_path"]).exists()
    quarantined = tuple(
        moved.glob(
            ".evoworld/quarantine/evoworld-*/*/staging/dataset_meta_info/evoworld/train_sample.json"
        )
    )
    assert len(quarantined) == 1


def test_linux_directory_publish_uses_dirfd_noreplace_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    class RenameAt2:
        argtypes: object = None
        restype: object = None

        def __call__(self, *args: object) -> int:
            calls.append(args)
            return 0

    class LibC:
        renameat2 = RenameAt2()

    monkeypatch.setattr(ctrl_bridge.sys, "platform", "linux")
    monkeypatch.setattr(ctrl_bridge.ctypes, "CDLL", lambda *_, **__: LibC())
    ctrl_bridge._rename_directory_no_replace(11, "source", 22, "target", "test")
    assert calls == [(11, b"source", 22, b"target", 1)]


def test_quarantine_target_race_never_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    quarantine = tmp_path / "quarantine"
    (work / "run").mkdir(parents=True)
    target = quarantine / "fixed"
    target.mkdir(parents=True)
    sentinel = target / "owner.bin"
    sentinel.write_bytes(b"owner")

    class FixedUuid:
        hex = "fixed"

    monkeypatch.setattr(ctrl_bridge, "uuid4", lambda: FixedUuid())
    flags = ctrl_bridge._directory_flags()
    work_fd = os.open(work, flags)
    quarantine_fd = os.open(quarantine, flags)
    try:
        with pytest.raises(FileExistsError):
            ctrl_bridge._quarantine_metadata_failure_at(
                work_fd, "run", quarantine_fd, "0" * 64, RuntimeError("failed")
            )
    finally:
        os.close(quarantine_fd)
        os.close(work_fd)
    assert sentinel.read_bytes() == b"owner"
    assert (work / "run").is_dir()


@pytest.mark.parametrize("existing", ("output", "claim"))
def test_latent_preparation_rejects_existing_or_claimed_output_before_command(
    tmp_path: Path, existing: str
) -> None:
    repository = (tmp_path / "repo").resolve()
    source = _write_source(repository)
    python = _write_backend_python(tmp_path)
    workspace = (tmp_path / "workspace").resolve()
    droid = (tmp_path / "droid").resolve()
    svd = (tmp_path / "svd").resolve()
    for path in (workspace, droid, svd):
        path.mkdir()
    extraction, _ = plan_ctrl_world_prepare(
        CtrlWorldPrepareConfig(
            droid_hf_path=droid,
            droid_output_path=workspace / "datasets/evoworld",
            svd_model_path=svd,
            dataset_name="evoworld",
            ctrl_world_python=python,
            workspace_root=workspace,
            world_size=1,
        ),
        ctrl_world_root=source,
        repository_root=repository,
        revision_auditor=_audit,
    )
    request_path, request = _request(extraction)
    output = Path(request["output_bindings"]["droid_output_path"])
    stale = (
        output if existing == "output" else Path(request["output_bindings"]["latents_claim_path"])
    )
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.mkdir() if existing == "output" else stale.write_text("claimed")
    with pytest.raises(CtrlWorldBridgeError, match="existing|claimed|overwrite"):
        run_preparation(
            "prepare-latents",
            request_path,
            extraction.argv[-1],
            revision_auditor=_audit,
            command_runner=lambda _, __: pytest.fail("official command executed"),
        )


@pytest.mark.parametrize(
    ("operation", "source_relative"),
    (
        ("prepare-latents", "dataset_example/extract_latent.py"),
        ("bind-metadata", "dataset_meta_info/create_meta_info.py"),
    ),
)
def test_preparation_script_mutation_is_rejected_before_execution(
    tmp_path: Path, operation: str, source_relative: str
) -> None:
    repository = (tmp_path / "repo").resolve()
    source = _write_source(repository)
    python = _write_backend_python(tmp_path)
    workspace = (tmp_path / "workspace").resolve()
    droid = (tmp_path / "droid").resolve()
    svd = (tmp_path / "svd").resolve()
    for path in (workspace, droid, svd):
        path.mkdir()
    plans = plan_ctrl_world_prepare(
        CtrlWorldPrepareConfig(
            droid_hf_path=droid,
            droid_output_path=workspace / "datasets/evoworld",
            svd_model_path=svd,
            dataset_name="evoworld",
            ctrl_world_python=python,
            workspace_root=workspace,
            world_size=1,
        ),
        ctrl_world_root=source,
        repository_root=repository,
        revision_auditor=_audit,
    )
    plan = plans[0] if operation == "prepare-latents" else plans[1]
    request_path, _ = _request(plan)
    (source / source_relative).write_text("mutated", encoding="utf-8")
    commands: list[tuple[str, ...]] = []
    with pytest.raises(CtrlWorldBridgeError, match="source|verification"):
        run_preparation(
            operation,
            request_path,
            plan.argv[-1],
            revision_auditor=_audit,
            command_runner=lambda command, _: commands.append(tuple(command)),
        )
    assert commands == []


def test_preparation_rejects_mutated_accelerate_launcher_before_execution(
    tmp_path: Path,
) -> None:
    repository = (tmp_path / "repo").resolve()
    source = _write_source(repository)
    python = _write_backend_python(tmp_path)
    workspace = (tmp_path / "workspace").resolve()
    droid = (tmp_path / "droid").resolve()
    svd = (tmp_path / "svd").resolve()
    for path in (workspace, droid, svd):
        path.mkdir()
    extraction, _ = plan_ctrl_world_prepare(
        CtrlWorldPrepareConfig(
            droid_hf_path=droid,
            droid_output_path=workspace / "datasets/evoworld",
            svd_model_path=svd,
            dataset_name="evoworld",
            ctrl_world_python=python,
            workspace_root=workspace,
            world_size=1,
        ),
        ctrl_world_root=source,
        repository_root=repository,
        revision_auditor=_audit,
    )
    request_path, _ = _request(extraction)
    marker = tmp_path / "launcher-executed"
    (tmp_path / "backend_modules/accelerate/commands/launch.py").write_text(
        f"open({str(marker)!r}, 'w').write('executed')", encoding="utf-8"
    )
    commands: list[tuple[str, ...]] = []
    with pytest.raises(CtrlWorldBridgeError, match="environment|verification"):
        run_preparation(
            "prepare-latents",
            request_path,
            extraction.argv[-1],
            revision_auditor=_audit,
            command_runner=lambda command, _: commands.append(tuple(command)),
        )
    assert commands == []
    assert not marker.exists()


def test_real_media_and_source_faithful_latents_validate(tmp_path: Path) -> None:
    config, manifest, _, _ = _fixture(tmp_path)
    annotations = validate_ctrl_world_dataset(config, manifest)
    assert set(annotations) == {1, 2}
    result = inspect_ctrl_world_media(
        config.ctrl_world_python,
        (config.dataset_path / "videos/train/1/0.mp4",),
        (config.dataset_path / "latent_videos/train/1/0.pt",),
    )
    assert result["videos"][str(config.dataset_path / "videos/train/1/0.mp4")]["frames"] == 3


def test_environment_fingerprint_binds_versions_and_module_files(tmp_path: Path) -> None:
    python = _write_backend_python(tmp_path)
    fingerprint = fingerprint_ctrl_world_environment(python)
    distributions = fingerprint["distributions"]
    assert isinstance(distributions, dict)
    for name in ("torch", "accelerate", "diffusers", "pandas", "decord", "scipy"):
        assert set(distributions[name]) == {"version", "record", "files", "modules"}
        assert distributions[name]["files"]
    assert {row["name"] for row in distributions["accelerate"]["modules"]} >= {
        "accelerate.commands.launch",
        "accelerate.accelerator",
    }
    assert {row["name"] for row in distributions["torch"]["modules"]} >= {
        "torch.optim",
        "torch.serialization",
        "torch.nn",
        "torch.utils.data",
    }
    assert {row["name"] for row in distributions["imageio"]["modules"]} >= {"imageio.v3"}
    assert distributions["accelerate"]["record"]["path"].endswith(".dist-info/RECORD")
    assert distributions["accelerate"]["record"]["bytes_base64"]


def test_environment_fingerprint_binds_raw_record_bytes(tmp_path: Path) -> None:
    python = _write_backend_python(tmp_path)
    before = fingerprint_ctrl_world_environment(python)
    record = tmp_path / "backend_modules/accelerate-0.0.dist-info/RECORD"
    record.write_text(record.read_text().replace("sha256=AA", "sha256=AB", 1))
    after = fingerprint_ctrl_world_environment(python)
    assert before != after
    assert (
        before["distributions"]["accelerate"]["record"]
        != after["distributions"]["accelerate"]["record"]
    )


def test_environment_fingerprint_supports_bound_editable_distribution(tmp_path: Path) -> None:
    python = _write_backend_python(tmp_path)
    modules = tmp_path / "backend_modules"
    editable_root = tmp_path / "editable-a"
    editable_root_b = tmp_path / "editable-b"
    editable_root.mkdir()
    editable_root_b.mkdir()
    editable_module = editable_root / "einops.py"
    editable_module_b = editable_root_b / "einops.py"
    editable_module.write_text("VALUE = 1\n", encoding="utf-8")
    editable_module_b.write_bytes(editable_module.read_bytes())
    (modules / "einops.py").unlink()
    info = modules / "einops-0.0.dist-info"
    direct_url = info / "direct_url.json"
    _write_json(direct_url, {"url": editable_root.as_uri(), "dir_info": {"editable": True}})
    record = info / "RECORD"
    rows = [
        row
        for row in record.read_text(encoding="utf-8").splitlines()
        if not row.startswith("einops.py,")
    ]
    rows.insert(0, f"{info.name}/direct_url.json,,")
    record.write_text("\n".join(rows) + "\n", encoding="utf-8")
    python.write_text(
        f"#!/bin/sh\nPYTHONPATH='{modules}:{editable_root}' exec '{sys.executable}' \"$@\"\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    fingerprint_a = fingerprint_ctrl_world_environment(python)
    module = fingerprint_a["distributions"]["einops"]["modules"][0]
    assert module["path"] == "editable/einops.py"
    assert module["sha256"] == sha256(editable_module.read_bytes()).hexdigest()
    assert fingerprint_a["distributions"]["einops"]["editable"]["root"] == str(editable_root)

    _write_json(direct_url, {"url": editable_root_b.as_uri(), "dir_info": {"editable": True}})
    python.write_text(
        f"#!/bin/sh\nPYTHONPATH='{modules}:{editable_root_b}' exec '{sys.executable}' \"$@\"\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    fingerprint_b = fingerprint_ctrl_world_environment(python)
    assert fingerprint_a != fingerprint_b
    assert fingerprint_a["distributions"]["einops"] != fingerprint_b["distributions"]["einops"]
    assert fingerprint_b["distributions"]["einops"]["editable"]["root"] == str(editable_root_b)
    assert fingerprint_b["distributions"]["einops"]["modules"][0]["sha256"] == module["sha256"]


def test_environment_fingerprint_rejects_oversized_direct_url_before_read(
    tmp_path: Path,
) -> None:
    python = _write_backend_python(tmp_path)
    modules = tmp_path / "backend_modules"
    info = modules / "einops-0.0.dist-info"
    direct_url = info / "direct_url.json"
    with direct_url.open("wb") as stream:
        stream.truncate(1_048_577)
    record = info / "RECORD"
    rows = record.read_text(encoding="utf-8").splitlines()
    rows.insert(0, f"{info.name}/direct_url.json,,")
    record.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(CtrlWorldConfigError, match="fingerprint") as caught:
        fingerprint_ctrl_world_environment(python)
    assert isinstance(caught.value.__cause__, subprocess.CalledProcessError)
    assert "byte bound exceeded" in caught.value.__cause__.stderr


def test_environment_fingerprint_accepts_direct_url_at_exact_byte_limit(tmp_path: Path) -> None:
    python = _write_backend_python(tmp_path)
    modules = tmp_path / "backend_modules"
    info = modules / "einops-0.0.dist-info"
    direct_url = info / "direct_url.json"
    payload = b'{"dir_info":{"editable":false},"url":"file:///tmp"}'
    direct_url.write_bytes(payload + b" " * (1_048_576 - len(payload)))
    record = info / "RECORD"
    rows = record.read_text(encoding="utf-8").splitlines()
    rows.insert(0, f"{info.name}/direct_url.json,,")
    record.write_text("\n".join(rows) + "\n", encoding="utf-8")
    fingerprint = fingerprint_ctrl_world_environment(python)
    direct_entry = next(
        row
        for row in fingerprint["distributions"]["einops"]["files"]
        if row["path"].endswith("direct_url.json")
    )
    assert direct_entry["size"] == 1_048_576


@pytest.mark.parametrize(
    "invalid_row",
    (
        "accelerate/__init__.py,sha256=AA,1",
        "/tmp/escape.py,sha256=AA,1",
        "../escape.py,sha256=AA,1",
    ),
)
def test_environment_fingerprint_rejects_duplicate_absolute_and_escape_record_entries(
    tmp_path: Path, invalid_row: str
) -> None:
    python = _write_backend_python(tmp_path)
    record = tmp_path / "backend_modules/accelerate-0.0.dist-info/RECORD"
    record.write_text(record.read_text() + invalid_row + "\n")
    with pytest.raises(CtrlWorldConfigError, match="fingerprint"):
        fingerprint_ctrl_world_environment(python)


def test_preparation_rejects_missing_backend_dependency(tmp_path: Path) -> None:
    repository = (tmp_path / "repo").resolve()
    source = _write_source(repository)
    python = _write_backend_python(tmp_path)
    workspace = (tmp_path / "workspace").resolve()
    droid = (tmp_path / "droid").resolve()
    svd = (tmp_path / "svd").resolve()
    for path in (workspace, droid, svd):
        path.mkdir()
    (tmp_path / "backend_modules/mediapy.py").unlink()
    config = CtrlWorldPrepareConfig(
        droid_hf_path=droid,
        droid_output_path=workspace / "datasets/evoworld",
        svd_model_path=svd,
        dataset_name="evoworld",
        ctrl_world_python=python,
        workspace_root=workspace,
        world_size=1,
    )
    with pytest.raises(CtrlWorldConfigError, match="fingerprint"):
        plan_ctrl_world_prepare(
            config,
            ctrl_world_root=source,
            repository_root=repository,
            revision_auditor=_audit,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("joint_width", "joint positions"),
        ("gripper_nan", "gripper"),
        ("latent_shape", "latent tensor"),
        ("video_count", "decoded MP4"),
        ("metadata_extra", "metadata rows"),
    ],
)
def test_validation_rejects_source_incompatible_artifacts(
    tmp_path: Path, mutation: str, match: str
) -> None:
    config, manifest, _, _ = _fixture(tmp_path)
    annotation_path = config.dataset_path / "annotation/train/1.json"
    annotation = json.loads(annotation_path.read_text())
    if mutation == "joint_width":
        annotation["observation.state.joint_position"][0] = [0.0] * 6
        _write_json(annotation_path, annotation)
    elif mutation == "gripper_nan":
        annotation["observation.state.gripper_position"][0] = float("nan")
        _write_json(annotation_path, annotation)
    elif mutation == "latent_shape":
        _write_json(
            config.dataset_path / "latent_videos/train/1/0.pt",
            {"shape": [3, 4, 23, 40], "dtype": "torch.float32"},
        )
    elif mutation == "video_count":
        _write_video(config.dataset_path / "videos/train/1/0.mp4", frames=2)
    else:
        rows = json.loads((config.metadata_path / "train_sample.json").read_text())
        rows.append({"episode_id": 999, "frame_ids": [0]})
        _write_json(config.metadata_path / "train_sample.json", rows)
    with pytest.raises(CtrlWorldConfigError, match=match):
        validate_ctrl_world_dataset(config, manifest)


@pytest.mark.parametrize(
    "stats",
    (
        {"state_01": [float("nan")] * 7, "state_99": [1.0] * 7},
        {"state_01": [2.0] * 7, "state_99": [1.0] * 7},
    ),
)
def test_validation_rejects_nonfinite_or_reversed_stats(
    tmp_path: Path, stats: dict[str, list[float]]
) -> None:
    config, manifest, _, _ = _fixture(tmp_path)
    _write_json(config.stats_path, stats)
    with pytest.raises(CtrlWorldConfigError, match="stats"):
        validate_ctrl_world_dataset(config, manifest)


@pytest.mark.parametrize("mutation", ["trajectory", "instruction", "media", "hash"])
def test_manifest_must_be_bijective_with_real_annotation(tmp_path: Path, mutation: str) -> None:
    config, manifest, _, _ = _fixture(tmp_path)
    sample = manifest.samples[0]
    if mutation == "trajectory":
        sample = replace(sample, trajectory_id="777")
    elif mutation == "instruction":
        sample = replace(sample, instruction="junk")
    elif mutation == "media":
        sample = replace(sample, observation_paths=tuple(reversed(sample.observation_paths)))
    else:
        sample = replace(sample, content_sha256="f" * 64)
    changed = replace(manifest, samples=(sample,))
    _write_json(config.manifest_path, changed.to_dict())
    with pytest.raises(CtrlWorldConfigError, match="manifest|trajectory|bijective|provenance"):
        validate_ctrl_world_dataset(config, changed)


def test_wmf_plan_materializes_exact_train_projection_and_bound_request(tmp_path: Path) -> None:
    config, manifest, repository, source = _fixture(tmp_path)
    plan = plan_ctrl_world_training(
        manifest,
        config,
        ctrl_world_root=source,
        repository_root=repository,
        revision_auditor=_audit,
    )
    request_path, request = _request(plan)
    projection = Path(request["projection_root"])
    assert plan.argv[:3] == (
        str(config.ctrl_world_python),
        str(ROOT / "integrations/ctrl_world/train_bridge.py"),
        "launch-train",
    )
    assert plan.expected_outputs[0] == Path("checkpoint-5000.pt")
    assert request["max_train_steps"] == 5000
    assert request["revision"] == CTRL_WORLD_REVISION
    assert (projection / "datasets/evoworld_wmf/annotation/train/1.json").is_file()
    assert not (projection / "datasets/evoworld_wmf/annotation/train/2.json").exists()
    assert json.loads(
        (projection / "dataset_meta_info/evoworld_wmf/train_sample.json").read_text()
    ) == [{"episode_id": 1, "frame_ids": [0]}]
    assert sha256_file(request_path) == plan.argv[-1]
    verify_request_bindings(request)
    commands: list[tuple[str, ...]] = []
    launch_training(
        request_path,
        plan.argv[-1],
        revision_auditor=_audit,
        command_runner=lambda command, _: commands.append(tuple(command)),
    )
    assert commands[0][:6] == (
        str(config.ctrl_world_python),
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        "1",
        str(ROOT / "integrations/ctrl_world/train_bridge.py"),
    )
    assert commands[0][6] == "train-worker"


@pytest.mark.parametrize(
    "relative",
    (
        "accelerate/commands/launch.py",
        "imageio/v3.py",
        "torch/optim.py",
        "torch/serialization.py",
        "torch/hidden_runtime.py",
        "torch/nn/__init__.py",
        "torch/utils/data/__init__.py",
    ),
)
def test_runtime_distribution_mutation_is_rejected_before_accelerate(
    tmp_path: Path, relative: str
) -> None:
    config, manifest, repository, source = _fixture(tmp_path)
    plan = plan_ctrl_world_training(
        manifest,
        config,
        ctrl_world_root=source,
        repository_root=repository,
        revision_auditor=_audit,
    )
    request_path, _ = _request(plan)
    (tmp_path / "backend_modules" / relative).write_text("mutated", encoding="utf-8")
    commands: list[tuple[str, ...]] = []
    with pytest.raises(CtrlWorldBridgeError, match="environment|verification"):
        launch_training(
            request_path,
            plan.argv[-1],
            revision_auditor=_audit,
            command_runner=lambda command, _: commands.append(tuple(command)),
        )
    assert commands == []


def test_plan_rejects_non_wmf_and_failed_i2_audit(tmp_path: Path) -> None:
    config, manifest, repository, source = _fixture(tmp_path, diagnosis=DiagnosisLabel.PF)
    with pytest.raises(CtrlWorldConfigError, match="WMF"):
        plan_ctrl_world_training(
            manifest,
            config,
            ctrl_world_root=source,
            repository_root=repository,
            revision_auditor=_audit,
        )
    manifest = replace(
        manifest, samples=(replace(manifest.samples[0], diagnosis=DiagnosisLabel.WMF),)
    )
    _write_json(config.manifest_path, manifest.to_dict())
    with pytest.raises(CtrlWorldConfigError, match="revision"):
        plan_ctrl_world_training(
            manifest,
            config,
            ctrl_world_root=source,
            repository_root=repository,
            revision_auditor=lambda _: {CTRL_WORLD_SUBMODULE: "0" * 40},
        )


def test_wmf_plan_requires_existing_starting_checkpoint(tmp_path: Path) -> None:
    config, manifest, repository, source = _fixture(tmp_path)
    with pytest.raises(CtrlWorldConfigError, match="starting checkpoint"):
        plan_ctrl_world_training(
            manifest,
            replace(config, starting_checkpoint=None),
            ctrl_world_root=source,
            repository_root=repository,
            revision_auditor=_audit,
        )


def test_public_base_plan_requires_none_train_real_source_and_no_warm_start(
    tmp_path: Path,
) -> None:
    config, manifest, repository, source = _fixture(tmp_path)
    sample = replace(
        manifest.samples[0],
        diagnosis=DiagnosisLabel.NONE,
        source_type="real",
        source_checkpoint="public-droid",
    )
    base_manifest = replace(manifest, samples=(sample,))
    _write_json(config.manifest_path, base_manifest.to_dict())
    base_config = replace(config, starting_checkpoint=None)
    plan = plan_ctrl_world_base_training(
        base_manifest,
        base_config,
        ctrl_world_root=source,
        repository_root=repository,
        revision_auditor=_audit,
    )
    assert plan.target == "ctrl_world_base"
    with pytest.raises(CtrlWorldConfigError, match="warm-start"):
        plan_ctrl_world_base_training(
            base_manifest,
            config,
            ctrl_world_root=source,
            repository_root=repository,
            revision_auditor=_audit,
        )
    for invalid in (
        replace(sample, diagnosis=DiagnosisLabel.WMF),
        replace(sample, split="eval"),
        replace(sample, source_type="synthetic"),
    ):
        invalid_manifest = replace(base_manifest, samples=(invalid,))
        _write_json(config.manifest_path, invalid_manifest.to_dict())
        with pytest.raises(CtrlWorldConfigError, match="public base"):
            plan_ctrl_world_base_training(
                invalid_manifest,
                base_config,
                ctrl_world_root=source,
                repository_root=repository,
                revision_auditor=_audit,
            )


def test_plan_rejects_output_overlap_with_dataset_or_source(tmp_path: Path) -> None:
    config, manifest, repository, source = _fixture(tmp_path)
    for output in (config.dataset_root_path, repository):
        with pytest.raises(CtrlWorldConfigError, match="output root"):
            plan_ctrl_world_training(
                manifest,
                replace(config, output_root=output),
                ctrl_world_root=source,
                repository_root=repository,
                revision_auditor=_audit,
            )


def test_bound_request_rejects_projection_checkpoint_source_and_environment_mutation(
    tmp_path: Path,
) -> None:
    for mutation in ("projection", "checkpoint", "source", "environment"):
        case = tmp_path / mutation
        config, manifest, repository, source = _fixture(case)
        plan = plan_ctrl_world_training(
            manifest,
            config,
            ctrl_world_root=source,
            repository_root=repository,
            revision_auditor=_audit,
        )
        _, request = _request(plan)
        if mutation == "projection":
            Path(request["projection_root"], "projection.json").write_text("{}")
        elif mutation == "checkpoint":
            config.starting_checkpoint.write_bytes(b"changed")
        elif mutation == "source":
            (source / "models/ctrl_world.py").write_text("changed")
        else:
            (case / "backend_modules/torch/optim.py").write_text("changed")
        with pytest.raises(CtrlWorldConfigError, match="changed|binding|source|input|environment"):
            verify_request_bindings(request)


def test_bridge_rejects_rehashed_request_with_changed_output_identity(tmp_path: Path) -> None:
    config, manifest, repository, source = _fixture(tmp_path)
    plan = plan_ctrl_world_training(
        manifest,
        config,
        ctrl_world_root=source,
        repository_root=repository,
        revision_auditor=_audit,
    )
    request_path, request = _request(plan)
    request["output_dir"] = str(tmp_path / "other-output")
    _write_json(request_path, request)
    with pytest.raises(CtrlWorldBridgeError, match="identity"):
        execute_request(
            request_path,
            sha256_file(request_path),
            revision_auditor=_audit,
            components=_Components(),
        )


def test_bridge_rejects_rehashed_request_with_changed_bridge_binding(tmp_path: Path) -> None:
    config, manifest, repository, source = _fixture(tmp_path)
    plan = plan_ctrl_world_training(
        manifest,
        config,
        ctrl_world_root=source,
        repository_root=repository,
        revision_auditor=_audit,
    )
    request_path, request = _request(plan)
    request["bridge_sha256"] = "0" * 64
    _write_json(request_path, request)
    with pytest.raises(CtrlWorldBridgeError, match="bridge binding"):
        execute_request(
            request_path,
            sha256_file(request_path),
            revision_auditor=_audit,
            components=_Components(),
        )


def test_first_hop_rejects_changed_local_helper_before_backend_command(tmp_path: Path) -> None:
    config, manifest, repository, source = _fixture(tmp_path)
    plan = plan_ctrl_world_training(
        manifest,
        config,
        ctrl_world_root=source,
        repository_root=repository,
        revision_auditor=_audit,
    )
    request_path, request = _request(plan)
    helper = next(
        row
        for row in request["local_helpers"]
        if row["path"].endswith("training/world_model/ctrl_world.py")
    )
    helper["sha256"] = "0" * 64
    _write_json(request_path, request)
    commands: list[tuple[str, ...]] = []
    with pytest.raises(CtrlWorldBridgeError, match="local helper"):
        launch_training(
            request_path,
            sha256_file(request_path),
            revision_auditor=_audit,
            command_runner=lambda command, _: commands.append(tuple(command)),
        )
    assert commands == []


def test_first_hop_has_no_runtime_local_import_before_request_verification() -> None:
    bridge = ROOT / "integrations/ctrl_world/train_bridge.py"
    tree = ast.parse(bridge.read_text(encoding="utf-8"))
    runtime_imports = {
        alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names
    } | {node.module or "" for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert not any(
        name.startswith(("training", "evoworld", "integrations")) for name in runtime_imports
    )


def test_first_hop_rejects_actual_helper_mutation_without_importing_it(tmp_path: Path) -> None:
    copied_root = tmp_path / "copy"
    relatives = (
        Path("training/__init__.py"),
        Path("training/world_model/__init__.py"),
        Path("training/world_model/ctrl_world.py"),
        Path("training/common/__init__.py"),
        Path("training/common/schema.py"),
        Path("src/evoworld/__init__.py"),
        Path("src/evoworld/contracts.py"),
    )
    bridge_relative = Path("integrations/ctrl_world/train_bridge.py")
    for relative in (*relatives, bridge_relative):
        destination = copied_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    bridge = (copied_root / bridge_relative).resolve()
    request = {
        "bridge_path": str(bridge),
        "bridge_sha256": sha256_file(bridge),
        "local_helpers": [
            {
                "path": str((copied_root / relative).resolve()),
                "sha256": sha256_file(copied_root / relative),
            }
            for relative in relatives
        ],
    }
    request_path = (tmp_path / "request.json").resolve()
    _write_json(request_path, request)
    marker = tmp_path / "helper-imported"
    helper = copied_root / "training/world_model/ctrl_world.py"
    helper.write_text(
        helper.read_text(encoding="utf-8")
        + f"\nfrom pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        (
            sys.executable,
            str(bridge),
            "launch-train",
            "--request-path",
            str(request_path),
            "--request-sha256",
            sha256_file(request_path),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "local helper changed" in completed.stderr
    assert not marker.exists()


class _Args:
    max_train_steps = 5000
    gradient_accumulation_steps = 2
    train_batch_size = 4
    checkpointing_steps = 20000
    max_grad_norm = 1.0
    ckpt_path: str | None = None
    world_size = 8


class _Accelerator:
    def __init__(self, num_processes: int = 8) -> None:
        self.sync_gradients = False
        self.num_processes = num_processes
        self.is_main_process = True
        self.forward = 0

    def accumulate(self, model: object) -> Any:
        self.forward += 1
        self.sync_gradients = self.forward % 2 == 0
        return nullcontext()

    def autocast(self) -> Any:
        return nullcontext()

    def backward(self, loss: object) -> None:
        pass

    def clip_grad_norm_(self, parameters: object, max_norm: float) -> None:
        pass

    def unwrap_model(self, model: object) -> object:
        return model

    def wait_for_everyone(self) -> None:
        pass

    def end_training(self) -> None:
        pass


class _Model:
    def parameters(self) -> tuple[()]:
        return ()

    def state_dict(self) -> dict[str, int]:
        return {"weight": 1}


class _Optimizer:
    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        pass


class _Components:
    args_factory = _Args

    def __init__(self, num_processes: int = 8) -> None:
        self.accelerator = _Accelerator(num_processes)
        self.saved: list[Path] = []

    def create_accelerator(self, args: object, output_dir: Path) -> _Accelerator:
        return self.accelerator

    def create_model(self, args: object) -> _Model:
        return _Model()

    def load_checkpoint(self, model: object, checkpoint: Path) -> None:
        pass

    def create_train_dataset(self, args: object) -> list[int]:
        return [1]

    def create_dataloader(self, dataset: object, args: object) -> list[int]:
        return [1]

    def create_optimizer(self, model: object, args: object) -> _Optimizer:
        return _Optimizer()

    def prepare(
        self, accelerator: object, model: object, optimizer: object, dataloader: object
    ) -> tuple[object, object, object]:
        return model, optimizer, dataloader

    def loss(self, model: object, batch: object) -> float:
        return 1.0

    def parameters(self, model: object) -> tuple[()]:
        return ()

    def save_checkpoint(self, accelerator: object, model: object, path: Path) -> None:
        path.write_bytes(b"state")
        self.saved.append(path)

    def validate_checkpoint(self, accelerator: object, model: object, path: Path) -> None:
        if path.read_bytes() != b"state":
            raise CtrlWorldBridgeError("checkpoint round-trip validation failed")


def test_bridge_owned_loop_runs_exactly_5000_optimizer_steps_and_forces_final_checkpoint(
    tmp_path: Path,
) -> None:
    components = _Components()
    result = run_bounded_training(
        _Args(), output_dir=tmp_path.resolve(), components=components, binding={"job": "test"}
    )
    assert result.optimizer_steps == 5000
    assert result.forward_steps == 10000
    assert len(components.saved) == 1
    assert components.saved[0].parent == tmp_path
    assert components.saved[0].name.startswith(".checkpoint-5000.pt.")
    assert result.final_checkpoint.read_bytes() == b"state"
    state = json.loads(result.state_path.read_text())
    assert state["total_batch_size"] == 64
    assert state["optimizer_steps"] == 5000


def test_execute_request_rechecks_every_binding_and_produces_final_artifacts(
    tmp_path: Path,
) -> None:
    config, manifest, repository, source = _fixture(tmp_path)
    plan = plan_ctrl_world_training(
        manifest,
        config,
        ctrl_world_root=source,
        repository_root=repository,
        revision_auditor=_audit,
    )
    request_path, _ = _request(plan)
    components = _Components(num_processes=1)
    result = execute_request(
        request_path,
        plan.argv[-1],
        revision_auditor=_audit,
        components=components,
    )
    assert result.optimizer_steps == 5000
    assert result.final_checkpoint.is_file()
    metadata = json.loads(result.metadata_path.read_text())
    assert metadata["request_sha256"] == plan.argv[-1]
    assert metadata["revision"] == CTRL_WORLD_REVISION


def test_execute_request_rejects_existing_output_before_components(tmp_path: Path) -> None:
    config, manifest, repository, source = _fixture(tmp_path)
    plan = plan_ctrl_world_training(
        manifest,
        config,
        ctrl_world_root=source,
        repository_root=repository,
        revision_auditor=_audit,
    )
    request_path, request = _request(plan)
    existing = Path(request["output_dir"]) / "training_state.json"
    existing.write_text("existing", encoding="utf-8")
    with pytest.raises(CtrlWorldBridgeError, match="overwrite"):
        execute_request(
            request_path,
            plan.argv[-1],
            revision_auditor=_audit,
            components=_Components(num_processes=1),
        )
    assert existing.read_text(encoding="utf-8") == "existing"


def test_loop_rejects_world_size_mismatch(tmp_path: Path) -> None:
    with pytest.raises(CtrlWorldBridgeError, match="world size"):
        run_bounded_training(
            _Args(),
            output_dir=tmp_path.resolve(),
            components=_Components(num_processes=1),
            binding={},
        )


def test_checkpoint_round_trip_failure_never_publishes_final_artifacts(tmp_path: Path) -> None:
    class InvalidComponents(_Components):
        def save_checkpoint(self, accelerator: object, model: object, path: Path) -> None:
            path.write_bytes(b"invalid")

    with pytest.raises(CtrlWorldBridgeError, match="round-trip"):
        run_bounded_training(
            _Args(),
            output_dir=tmp_path.resolve(),
            components=InvalidComponents(),
            binding={},
        )
    assert not (tmp_path / "checkpoint-5000.pt").exists()
    assert not (tmp_path / "training_state.json").exists()
    assert not (tmp_path / "job_metadata.json").exists()


def test_official_checkpoint_validation_uses_weights_only_and_strict_reload(
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    class Torch:
        @staticmethod
        def load(path: Path, map_location: str, weights_only: bool = False) -> dict[str, int]:
            calls["weights_only"] = weights_only
            calls["map_location"] = map_location
            return {"weight": 1}

    class Model:
        def load_state_dict(self, state: object, strict: bool) -> None:
            calls["state"] = state
            calls["strict"] = strict

    components = object.__new__(OfficialCtrlWorldComponents)
    components.torch = Torch()
    components.validate_checkpoint(_Accelerator(), Model(), tmp_path / "checkpoint.tmp")
    assert calls == {
        "weights_only": True,
        "map_location": "cpu",
        "state": {"weight": 1},
        "strict": True,
    }


def test_existing_or_racing_outputs_are_never_overwritten(tmp_path: Path) -> None:
    existing = tmp_path / "checkpoint-5000.pt"
    existing.write_bytes(b"existing")
    with pytest.raises(CtrlWorldBridgeError, match="overwrite"):
        run_bounded_training(
            _Args(),
            output_dir=tmp_path.resolve(),
            components=_Components(),
            binding={},
        )
    assert existing.read_bytes() == b"existing"

    race_root = tmp_path / "race"

    class RacingComponents(_Components):
        def validate_checkpoint(self, accelerator: object, model: object, path: Path) -> None:
            super().validate_checkpoint(accelerator, model, path)
            (race_root / "checkpoint-5000.pt").write_bytes(b"racer")

    with pytest.raises(CtrlWorldBridgeError, match="concurrently"):
        run_bounded_training(
            _Args(),
            output_dir=race_root.resolve(),
            components=RacingComponents(),
            binding={},
        )
    assert (race_root / "checkpoint-5000.pt").read_bytes() == b"racer"


def test_official_args_override_all_cephfs_paths_without_changing_training_defaults(
    tmp_path: Path,
) -> None:
    config, _, _, _ = _fixture(tmp_path)

    class Args(_Args):
        svd_model_path = clip_model_path = ckpt_path = pi_ckpt = "/cephfs/default"
        dataset_root_path = dataset_meta_info_path = data_stat_path = "/cephfs/default"
        val_model_path = val_dataset_dir = action_adapter = "/cephfs/default"
        dataset_names = dataset_cfgs = "default"
        annotation_name = tag = output_dir = "default"
        wandb_run_name = wandb_project_name = save_dir = task_name = "default"
        learning_rate = 1e-5
        shuffle = True

    args = build_official_args(
        config,
        output_dir=(tmp_path / "job").resolve(),
        args_factory=Args,
    )
    assert args.max_train_steps == 5000
    assert args.learning_rate == 1e-5
    assert args.shuffle is True
    assert "/cephfs" not in str(vars(args))


def test_checked_in_config_is_only_the_pinned_contract() -> None:
    assert load_ctrl_world_config(CONFIG_PATH) == {
        "revision": CTRL_WORLD_REVISION,
        "max_train_steps": 5000,
        "world_size": 1,
        "annotation_name": "annotation",
    }

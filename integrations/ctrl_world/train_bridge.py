"""Bounded Ctrl-World loop built from pinned official components."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
import ctypes
from dataclasses import dataclass
import errno
import fcntl
from hashlib import sha256
import inspect
import importlib
import json
import os
from pathlib import Path
import runpy
import stat
import subprocess
import sys
from typing import TYPE_CHECKING, Any, Iterable, Protocol, cast
from uuid import uuid4

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))
    sys.path.insert(0, str(_REPOSITORY_ROOT))

CTRL_WORLD_REVISION = "99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d"
_LOCAL_HELPER_RELATIVES = (
    Path("training/__init__.py"),
    Path("training/world_model/__init__.py"),
    Path("training/world_model/ctrl_world.py"),
    Path("training/common/__init__.py"),
    Path("training/common/schema.py"),
    Path("src/evoworld/__init__.py"),
    Path("src/evoworld/contracts.py"),
)
_OPTIONAL_LOCAL_HELPER_RELATIVES = (
    Path("src/evoworld/adapters/__init__.py"),
    Path("src/evoworld/adapters/revisions.py"),
)
_HELPER_MODULE: Any | None = None

if TYPE_CHECKING:
    from training.world_model.ctrl_world import CtrlWorldTrainConfig, RevisionAuditor


class CtrlWorldBridgeError(RuntimeError):
    """Raised when a bound request or official training component is unsafe."""


class ArgsFactory(Protocol):
    def __call__(self) -> object: ...


class AcceleratorLike(Protocol):
    sync_gradients: bool
    num_processes: int
    is_main_process: bool

    def accumulate(self, model: object) -> AbstractContextManager[object]: ...
    def autocast(self) -> AbstractContextManager[object]: ...
    def backward(self, loss: object) -> None: ...
    def clip_grad_norm_(self, parameters: object, max_norm: float) -> None: ...
    def unwrap_model(self, model: object) -> object: ...
    def wait_for_everyone(self) -> None: ...
    def end_training(self) -> None: ...
    def prepare(self, *values: object) -> tuple[object, ...]: ...


class OptimizerLike(Protocol):
    def step(self) -> None: ...
    def zero_grad(self) -> None: ...


class TrainingComponents(Protocol):
    def create_accelerator(self, args: object, output_dir: Path) -> AcceleratorLike: ...
    def create_model(self, args: object) -> object: ...
    def load_checkpoint(self, model: object, checkpoint: Path) -> None: ...
    def create_train_dataset(self, args: object) -> object: ...
    def create_dataloader(self, dataset: object, args: object) -> object: ...
    def create_optimizer(self, model: object, args: object) -> OptimizerLike: ...
    def prepare(
        self,
        accelerator: AcceleratorLike,
        model: object,
        optimizer: OptimizerLike,
        dataloader: object,
    ) -> tuple[object, OptimizerLike, object]: ...
    def loss(self, model: object, batch: object) -> object: ...
    def parameters(self, model: object) -> object: ...
    def save_checkpoint(
        self,
        accelerator: AcceleratorLike,
        model: object,
        path: Path,
    ) -> None: ...
    def validate_checkpoint(
        self,
        accelerator: AcceleratorLike,
        model: object,
        path: Path,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class BoundedTrainingResult:
    optimizer_steps: int
    forward_steps: int
    final_checkpoint: Path
    state_path: Path
    metadata_path: Path


@dataclass(frozen=True, slots=True)
class OfficialExecutionContext:
    workspace_root_fd: int
    working_directory_fd: int
    workspace_identity: tuple[int, int]
    working_directory_identity: tuple[int, int]
    prepared_relative: str
    backend_fd: int
    script_fd: int
    backend_binding: tuple[int, int, int, str]
    script_binding: tuple[int, int, int, str]


@dataclass(frozen=True, slots=True)
class BoundOfficialFiles:
    backend_fd: int
    script_fd: int
    backend_binding: tuple[int, int, int, str]
    script_binding: tuple[int, int, int, str]


ModuleLoader = Callable[[str], Any]
CommandRunner = Callable[[Sequence[str], OfficialExecutionContext | None], None]


def sha256_file(path: Path) -> str:
    """Hash a regular file without importing mutable repository helpers."""
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _helpers() -> Any:
    global _HELPER_MODULE
    if _HELPER_MODULE is None:
        _HELPER_MODULE = importlib.import_module("training.world_model.ctrl_world")
    return _HELPER_MODULE


class OfficialCtrlWorldComponents:
    """Load exact pinned model, dataset, Torch, and Accelerate components lazily."""

    def __init__(
        self,
        source: Path,
        *,
        module_loader: ModuleLoader = importlib.import_module,
    ) -> None:
        self.source = source.resolve()
        if str(self.source) in sys.path:
            sys.path.remove(str(self.source))
        sys.path.insert(0, str(self.source))
        self.torch = module_loader("torch")
        self.accelerate = module_loader("accelerate")
        self.model_module = module_loader("models.ctrl_world")
        self.dataset_module = module_loader("dataset.dataset_droid_exp33")
        for name, module in (
            ("models.ctrl_world", self.model_module),
            ("dataset.dataset_droid_exp33", self.dataset_module),
        ):
            _require_module_containment(module, self.source, name)

    def create_accelerator(self, args: object, output_dir: Path) -> AcceleratorLike:
        values = cast(Any, args)
        return cast(
            AcceleratorLike,
            self.accelerate.Accelerator(
                gradient_accumulation_steps=values.gradient_accumulation_steps,
                mixed_precision=values.mixed_precision,
                log_with="wandb",
                project_dir=str(output_dir),
            ),
        )

    def create_model(self, args: object) -> object:
        model = self.model_module.CrtlWorld(args)
        model.train()
        return model

    def load_checkpoint(self, model: object, checkpoint: Path) -> None:
        state = self.torch.load(checkpoint, map_location="cpu")
        cast(Any, model).load_state_dict(state, strict=True)

    def create_train_dataset(self, args: object) -> object:
        return self.dataset_module.Dataset_mix(args, mode="train")

    def create_dataloader(self, dataset: object, args: object) -> object:
        values = cast(Any, args)
        return self.torch.utils.data.DataLoader(
            dataset,
            batch_size=values.train_batch_size,
            shuffle=values.shuffle,
        )

    def create_optimizer(self, model: object, args: object) -> OptimizerLike:
        values = cast(Any, args)
        return cast(
            OptimizerLike,
            self.torch.optim.AdamW(cast(Any, model).parameters(), lr=values.learning_rate),
        )

    def prepare(
        self,
        accelerator: AcceleratorLike,
        model: object,
        optimizer: OptimizerLike,
        dataloader: object,
    ) -> tuple[object, OptimizerLike, object]:
        prepared = accelerator.prepare(model, optimizer, dataloader)
        return cast(tuple[object, OptimizerLike, object], prepared)

    def loss(self, model: object, batch: object) -> object:
        result = model(batch)  # type: ignore[operator]
        if not isinstance(result, tuple) or len(result) != 2:
            raise CtrlWorldBridgeError("pinned CrtlWorld must return (loss, diagnostics)")
        return result[0]

    def parameters(self, model: object) -> object:
        return model.parameters()  # type: ignore[attr-defined]

    def save_checkpoint(
        self,
        accelerator: AcceleratorLike,
        model: object,
        path: Path,
    ) -> None:
        state = accelerator.unwrap_model(model).state_dict()  # type: ignore[attr-defined]
        self.torch.save(state, path)

    def validate_checkpoint(
        self,
        accelerator: AcceleratorLike,
        model: object,
        path: Path,
    ) -> None:
        load_kwargs: dict[str, object] = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(self.torch.load).parameters:
            load_kwargs["weights_only"] = True
        state = self.torch.load(path, **load_kwargs)
        accelerator.unwrap_model(model).load_state_dict(state, strict=True)  # type: ignore[attr-defined]


def build_official_args(
    config: CtrlWorldTrainConfig,
    *,
    output_dir: Path,
    args_factory: ArgsFactory,
) -> object:
    """Instantiate official ``wm_args`` and replace every path/name global."""
    if not output_dir.is_absolute():
        raise CtrlWorldBridgeError("output_dir must be absolute")
    args = args_factory()
    checkpoint = str(config.starting_checkpoint) if config.starting_checkpoint else None
    overrides: dict[str, object] = {
        "svd_model_path": str(config.svd_model_path),
        "clip_model_path": str(config.clip_model_path),
        "ckpt_path": checkpoint,
        "pi_ckpt": str(config.policy_checkpoint_path),
        "dataset_root_path": str(config.dataset_root_path),
        "dataset_names": config.dataset_name,
        "dataset_meta_info_path": str(config.dataset_meta_info_path),
        "dataset_cfgs": config.dataset_name,
        "prob": [1.0],
        "annotation_name": config.annotation_name,
        "tag": config.tag,
        "output_dir": str(output_dir),
        "wandb_run_name": config.wandb_run_name,
        "wandb_project_name": config.wandb_project_name,
        "max_train_steps": config.max_train_steps,
        "world_size": config.world_size,
        "data_stat_path": str(config.stats_path),
        "val_model_path": checkpoint,
        "save_dir": str(config.synthetic_output_root),
        "action_adapter": str(config.action_adapter_path),
        "val_dataset_dir": str(config.dataset_path),
        "task_name": config.tag,
    }
    for name, value in overrides.items():
        setattr(args, name, value)
    _reject_cephfs(args)
    return args


def run_bounded_training(
    args: object,
    *,
    output_dir: Path,
    components: TrainingComponents,
    binding: Mapping[str, object],
) -> BoundedTrainingResult:
    """Run exactly ``max_train_steps`` synchronized optimizer updates and save final state."""
    maximum = getattr(args, "max_train_steps", None)
    if maximum != 5000:
        raise CtrlWorldBridgeError("bounded loop requires exactly 5,000 optimizer steps")
    output_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint = output_dir / "checkpoint-5000.pt"
    state_path = output_dir / "training_state.json"
    metadata_path = output_dir / "job_metadata.json"
    _ensure_outputs_absent((final_checkpoint, state_path, metadata_path))
    accelerator = components.create_accelerator(args, output_dir)
    world_size = cast(int, getattr(args, "world_size", 0))
    if accelerator.num_processes != world_size:
        raise CtrlWorldBridgeError(
            f"Accelerator world size {accelerator.num_processes} does not match request {world_size}"
        )
    model = components.create_model(args)
    checkpoint_value = getattr(args, "ckpt_path", None)
    if checkpoint_value is not None:
        components.load_checkpoint(model, Path(cast(str, checkpoint_value)))
    dataset = components.create_train_dataset(args)
    try:
        if len(dataset) < 1:  # type: ignore[arg-type]
            raise CtrlWorldBridgeError("projected Ctrl-World train dataset is empty")
    except TypeError as error:
        raise CtrlWorldBridgeError("official train dataset must define length") from error
    dataloader = components.create_dataloader(dataset, args)
    optimizer = components.create_optimizer(model, args)
    model, optimizer, dataloader = components.prepare(accelerator, model, optimizer, dataloader)
    optimizer_steps = 0
    forward_steps = 0
    checkpointing_steps = cast(int, getattr(args, "checkpointing_steps"))
    max_grad_norm = cast(float, getattr(args, "max_grad_norm"))
    while optimizer_steps < maximum:
        batches = 0
        for batch in cast(Iterable[object], dataloader):
            batches += 1
            forward_steps += 1
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss = components.loss(model, batch)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(components.parameters(model), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                optimizer_steps += 1
                if (
                    checkpointing_steps > 0
                    and optimizer_steps % checkpointing_steps == 0
                    and optimizer_steps < maximum
                    and accelerator.is_main_process
                ):
                    components.save_checkpoint(
                        accelerator,
                        model,
                        output_dir / f"checkpoint-{optimizer_steps}.pt",
                    )
                if optimizer_steps == maximum:
                    break
                if optimizer_steps > maximum:
                    raise CtrlWorldBridgeError("optimizer-step counter exceeded the bound")
        if batches == 0:
            raise CtrlWorldBridgeError("official dataloader yielded no batches")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint_temporary = _temporary_path(final_checkpoint)
        components.save_checkpoint(accelerator, model, checkpoint_temporary)
        try:
            components.validate_checkpoint(accelerator, model, checkpoint_temporary)
            _exclusive_publish(checkpoint_temporary, final_checkpoint)
        finally:
            checkpoint_temporary.unlink(missing_ok=True)
        total_batch_size = (
            cast(int, getattr(args, "train_batch_size"))
            * accelerator.num_processes
            * cast(int, getattr(args, "gradient_accumulation_steps"))
        )
        _publish_json_exclusive(
            state_path,
            {
                "optimizer_steps": optimizer_steps,
                "forward_steps": forward_steps,
                "gradient_accumulation_steps": getattr(args, "gradient_accumulation_steps"),
                "num_processes": accelerator.num_processes,
                "total_batch_size": total_batch_size,
                "final_checkpoint": final_checkpoint.name,
            },
        )
        _publish_json_exclusive(metadata_path, dict(binding))
    accelerator.wait_for_everyone()
    accelerator.end_training()
    return BoundedTrainingResult(
        optimizer_steps=optimizer_steps,
        forward_steps=forward_steps,
        final_checkpoint=final_checkpoint,
        state_path=state_path,
        metadata_path=metadata_path,
    )


def execute_request(
    request_path: Path,
    request_sha256: str,
    *,
    revision_auditor: RevisionAuditor | None = None,
    module_loader: ModuleLoader = importlib.import_module,
    components: TrainingComponents | None = None,
) -> BoundedTrainingResult:
    """Verify the immutable job request, then run only pinned official components."""
    request = _load_bound_request(request_path, request_sha256)
    source, config = _verify_training_request(request, request_path, revision_auditor)
    output_dir = Path(cast(str, request["output_dir"]))
    if components is None:
        _bootstrap_source(source)
        config_module = module_loader("config")
        _require_module_containment(config_module, source, "config")
        args_factory = cast(ArgsFactory, config_module.wm_args)
    else:
        args_factory = cast(ArgsFactory, getattr(components, "args_factory"))
    args = build_official_args(config, output_dir=output_dir, args_factory=args_factory)
    runtime = components or OfficialCtrlWorldComponents(source, module_loader=module_loader)
    return run_bounded_training(
        args,
        output_dir=output_dir,
        components=runtime,
        binding={
            "request_sha256": request_sha256,
            "job_id": request["job_id"],
            "target": request["target"],
            "manifest_sha256": request["manifest_sha256"],
            "revision": request["revision"],
            "source_blobs": request["source_blobs"],
            "projection_sha256": request["projection_sha256"],
            "environment_sha256": request["environment_sha256"],
            "world_size": request["world_size"],
            "inputs": request["inputs"],
            "output_dir": request["output_dir"],
        },
    )


def launch_training(
    request_path: Path,
    request_sha256: str,
    *,
    revision_auditor: RevisionAuditor | None = None,
    command_runner: CommandRunner | None = None,
) -> None:
    """Verify once in the bootstrap, then launch the verified distributed worker."""
    request = _load_bound_request(request_path, request_sha256)
    _verify_training_request(request, request_path, revision_auditor)
    command = (
        cast(str, request["ctrl_world_python"]),
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(request["world_size"]),
        str(Path(__file__).resolve()),
        "train-worker",
        "--request-path",
        str(request_path.resolve()),
        "--request-sha256",
        request_sha256,
    )
    (command_runner or _run_command)(command, None)


def run_preparation(
    operation: str,
    request_path: Path,
    request_sha256: str,
    *,
    revision_auditor: RevisionAuditor | None = None,
    command_runner: CommandRunner | None = None,
) -> None:
    """Verify preparation bindings before importing or executing official entry points."""
    request = _load_bound_request(request_path, request_sha256)
    _validate_preparation_request_identity(request, request_path, operation)
    if operation == "bind-metadata":
        with _open_trusted_roots(request), _open_official_files(request) as official_files:
            source = _verify_source_checkout(request, revision_auditor)
            try:
                _helpers().verify_preparation_request_bindings(request)
            except Exception as error:
                raise CtrlWorldBridgeError(
                    "bound preparation request verification failed"
                ) from error
            _bind_and_run_metadata(
                request,
                request_sha256,
                source,
                command_runner or _run_command,
                official_files,
            )
        return
    source = _verify_source_checkout(request, revision_auditor)
    try:
        _helpers().verify_preparation_request_bindings(request)
    except Exception as error:
        raise CtrlWorldBridgeError("bound preparation request verification failed") from error
    _run_latent_preparation(
        request,
        request_sha256,
        source,
        command_runner or _run_command,
    )


def _run_latent_preparation(
    request: Mapping[str, object],
    request_sha256: str,
    source: Path,
    command_runner: CommandRunner,
) -> None:
    inputs = cast(dict[str, dict[str, str]], request["inputs"])
    outputs = cast(dict[str, str], request["output_bindings"])
    output = Path(outputs["droid_output_path"])
    receipt_path = Path(outputs["latents_receipt_path"])
    claim_path = Path(outputs["latents_claim_path"])
    _ensure_outputs_absent((output, receipt_path))
    with _exclusive_claim(claim_path, request_sha256):
        _ensure_outputs_absent((output, receipt_path))
        _preflight_official_preparation(source)
        command_runner(
            (
                cast(str, request["ctrl_world_python"]),
                "-m",
                "accelerate.commands.launch",
                "--num_processes",
                str(request["world_size"]),
                str(source / "dataset_example/extract_latent.py"),
                "--droid_hf_path",
                inputs["droid_hf_path"]["path"],
                "--droid_output_path",
                outputs["droid_output_path"],
                "--svd_path",
                inputs["svd_model_path"]["path"],
            ),
            None,
        )
        inventory = _validate_prepared_droid_tree(output)
        inventory_sha = _helpers().canonical_sha256(inventory)
        _publish_json_exclusive(
            receipt_path,
            {
                "schema_version": 1,
                "request_id": request["request_id"],
                "request_sha256": request_sha256,
                "prepared_root": str(output),
                "prepared_inventory": inventory,
                "prepared_inventory_sha256": inventory_sha,
            },
        )


def _bind_and_run_metadata(
    request: Mapping[str, object],
    request_sha256: str,
    source: Path,
    command_runner: CommandRunner,
    official_files: BoundOfficialFiles,
) -> None:
    outputs = cast(dict[str, str], request["output_bindings"])
    trusted_roots = cast(dict[str, object], request["trusted_roots"])
    workspace_binding = cast(dict[str, object], trusted_roots["workspace"])
    workspace_path = Path(cast(str, workspace_binding["path"]))
    prepared_identity = Path(outputs["droid_output_path"])
    names = cast(dict[str, str], request["names"])
    name = names["dataset_name"]
    request_id = cast(str, request["request_id"])
    with _open_trusted_roots(request) as (workspace_fd, request_fd):
        prepared_relative = prepared_identity.relative_to(workspace_path)
        prepared_root = _fd_path(workspace_fd).joinpath(*prepared_relative.parts)
        receipt_path = _fd_path(request_fd) / Path(outputs["latents_receipt_path"]).name
        completion_name = Path(outputs["metadata_completion_path"]).name
        final_parent_fd = -1
        work_root_fd = -1
        quarantine_root_fd = -1
        try:
            final_parent_fd = _open_directory_tree(
                workspace_fd, ("dataset_meta_info",), create=True
            )
            work_root_fd = _open_directory_tree(
                request_fd, ("metadata-work", f"{name}-{request_id[:12]}"), create=True
            )
            quarantine_root_fd = _open_directory_tree(
                request_fd, ("quarantine", f"{name}-{request_id[:12]}"), create=True
            )
            _ensure_absent_at(final_parent_fd, name)
            _ensure_absent_at(request_fd, completion_name)
            receipt, inventory = _load_latents_receipt(
                request,
                request_sha256,
                receipt_path,
                prepared_root,
                prepared_identity,
            )
            claim_name = Path(outputs["metadata_claim_path"]).name
            with _exclusive_claim_at(request_fd, claim_name, request_sha256):
                _ensure_absent_at(final_parent_fd, name)
                _ensure_absent_at(request_fd, completion_name)
                run_name = uuid4().hex
                run_fd, staging_parent_fd = _mkdir_run_tree(work_root_fd, run_name)
                staging_output = _fd_path(staging_parent_fd) / name
                run_root_identity = Path(outputs["metadata_work_root"]) / run_name
                bound_request = {
                    **dict(request),
                    "operation": "prepare-metadata",
                    "parent_request_sha256": request_sha256,
                    "metadata_staging_path": str(run_root_identity / "dataset_meta_info" / name),
                    "metadata_final_path": outputs["metadata_output_path"],
                    "metadata_working_directory": str(run_root_identity),
                    "prepared_input": {
                        "root": str(prepared_identity),
                        "inventory": inventory,
                        "inventory_sha256": receipt["prepared_inventory_sha256"],
                        "receipt_path": outputs["latents_receipt_path"],
                        "receipt_sha256": sha256_file(receipt_path),
                    },
                }
                try:
                    _publish_json_exclusive_at(run_fd, "metadata-request.json", bound_request)
                    persisted_request = _read_json_at(run_fd, "metadata-request.json")
                    if persisted_request != bound_request:
                        raise CtrlWorldBridgeError(
                            "bound metadata request changed before execution"
                        )
                    _load_latents_receipt(
                        request,
                        request_sha256,
                        receipt_path,
                        prepared_root,
                        prepared_identity,
                    )
                    _preflight_official_preparation(source)
                    command_runner(
                        (
                            cast(str, request["ctrl_world_python"]),
                            str(source / "dataset_meta_info/create_meta_info.py"),
                            "--droid_output_path",
                            os.path.relpath(prepared_identity, start=run_root_identity),
                            "--dataset_name",
                            name,
                        ),
                        OfficialExecutionContext(
                            workspace_root_fd=workspace_fd,
                            working_directory_fd=run_fd,
                            workspace_identity=_fd_identity(workspace_fd),
                            working_directory_identity=_fd_identity(run_fd),
                            prepared_relative=os.path.relpath(
                                prepared_identity, start=run_root_identity
                            ),
                            backend_fd=official_files.backend_fd,
                            script_fd=official_files.script_fd,
                            backend_binding=official_files.backend_binding,
                            script_binding=official_files.script_binding,
                        ),
                    )
                    with _open_trusted_roots(request):
                        pass
                    post_receipt, post_inventory = _load_latents_receipt(
                        request,
                        request_sha256,
                        receipt_path,
                        prepared_root,
                        prepared_identity,
                    )
                    prepared_input = cast(dict[str, object], bound_request["prepared_input"])
                    _assert_prepared_input_matches(
                        prepared_input, post_receipt, post_inventory, receipt_path
                    )
                    _validate_metadata_output(staging_output)
                    metadata_inventory = cast(
                        list[dict[str, str]], _helpers().artifact_inventory(staging_output)
                    )
                    final_receipt, final_inventory = _load_latents_receipt(
                        request,
                        request_sha256,
                        receipt_path,
                        prepared_root,
                        prepared_identity,
                    )
                    _assert_prepared_input_matches(
                        prepared_input, final_receipt, final_inventory, receipt_path
                    )
                    publish_claim_name = Path(outputs["metadata_publish_claim_path"]).name
                    with _exclusive_claim_at(final_parent_fd, publish_claim_name, request_sha256):
                        _ensure_absent_at(final_parent_fd, name)
                        _ensure_absent_at(request_fd, completion_name)
                        _publish_directory_exclusive(name, name, staging_parent_fd, final_parent_fd)
                    metadata_output = _fd_path(final_parent_fd) / name
                    published_inventory = cast(
                        list[dict[str, str]], _helpers().artifact_inventory(metadata_output)
                    )
                    if published_inventory != metadata_inventory:
                        raise CtrlWorldBridgeError("published metadata inventory changed")
                    _publish_json_exclusive_at(
                        request_fd,
                        completion_name,
                        {
                            "schema_version": 1,
                            "request_id": request["request_id"],
                            "metadata_request_sha256": _sha256_file_at(
                                run_fd, "metadata-request.json"
                            ),
                            "prepared_inventory_sha256": prepared_input["inventory_sha256"],
                            "metadata_request_path": str(
                                run_root_identity / "metadata-request.json"
                            ),
                            "metadata_inventory": published_inventory,
                            "metadata_inventory_sha256": _helpers().canonical_sha256(
                                published_inventory
                            ),
                        },
                    )
                except Exception as error:
                    _quarantine_metadata_failure_at(
                        work_root_fd,
                        run_name,
                        quarantine_root_fd,
                        request_sha256,
                        error,
                    )
                    raise
                finally:
                    os.close(staging_parent_fd)
                    os.close(run_fd)
        finally:
            for descriptor in (quarantine_root_fd, work_root_fd, final_parent_fd):
                if descriptor >= 0:
                    os.close(descriptor)


def _assert_prepared_input_matches(
    prepared_input: Mapping[str, object],
    receipt: Mapping[str, object],
    inventory: list[dict[str, str]],
    receipt_path: Path,
) -> None:
    if (
        inventory != prepared_input["inventory"]
        or receipt["prepared_inventory_sha256"] != prepared_input["inventory_sha256"]
        or sha256_file(receipt_path) != prepared_input["receipt_sha256"]
    ):
        raise CtrlWorldBridgeError("prepared inventory drifted during metadata execution")


def _quarantine_metadata_failure_at(
    work_root_fd: int,
    run_name: str,
    quarantine_root_fd: int,
    request_sha256: str,
    error: Exception,
) -> None:
    quarantine_fd = _mkdir_open_exclusive(quarantine_root_fd, uuid4().hex)
    try:
        _rename_directory_no_replace(
            work_root_fd, run_name, quarantine_fd, "staging", "metadata quarantine"
        )
        _publish_json_exclusive_at(
            quarantine_fd,
            "invalid.json",
            {
                "schema_version": 1,
                "request_sha256": request_sha256,
                "error_type": type(error).__name__,
                "reason": str(error),
            },
        )
    finally:
        os.close(quarantine_fd)


def _load_latents_receipt(
    request: Mapping[str, object],
    metadata_request_sha256: str,
    receipt_path: Path,
    prepared_root: Path,
    prepared_identity: Path,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise CtrlWorldBridgeError("latents completion receipt is missing")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CtrlWorldBridgeError("latents completion receipt is invalid") from error
    inventory = _validate_prepared_droid_tree(prepared_root)
    if (
        not isinstance(receipt, dict)
        or set(receipt)
        != {
            "schema_version",
            "request_id",
            "request_sha256",
            "prepared_root",
            "prepared_inventory",
            "prepared_inventory_sha256",
        }
        or receipt.get("schema_version") != 1
        or receipt.get("request_id") != request["request_id"]
        or receipt.get("prepared_root") != str(prepared_identity)
        or receipt.get("prepared_inventory") != inventory
        or receipt.get("prepared_inventory_sha256") != _helpers().canonical_sha256(inventory)
    ):
        raise CtrlWorldBridgeError("latents completion receipt or inventory changed")
    if (
        receipt.get("request_sha256") != request.get("latents_request_sha256")
        or receipt.get("request_sha256") == metadata_request_sha256
    ):
        raise CtrlWorldBridgeError("metadata seed is not bound to the latents receipt")
    return cast(dict[str, object], receipt), inventory


def _validate_prepared_droid_tree(root: Path) -> list[dict[str, str]]:
    required = (
        (root / "annotation/train", "*.json"),
        (root / "annotation/val", "*.json"),
        (root / "videos", "*.mp4"),
        (root / "latent_videos", "*.pt"),
    )
    if any(
        not path.is_dir()
        or path.is_symlink()
        or not any(item.is_file() and not item.is_symlink() for item in path.rglob(pattern))
        for path, pattern in required
    ):
        raise CtrlWorldBridgeError("prepared DROID tree is incomplete")
    inventory = cast(list[dict[str, str]], _helpers().artifact_inventory(root))
    if not inventory:
        raise CtrlWorldBridgeError("prepared DROID tree is empty")
    return inventory


def _validate_metadata_output(root: Path) -> None:
    required = (root / "train_sample.json", root / "val_sample.json")
    if any(not path.is_file() or path.is_symlink() for path in required):
        raise CtrlWorldBridgeError("metadata output is incomplete")
    try:
        values = [json.loads(path.read_text(encoding="utf-8")) for path in required]
    except (OSError, json.JSONDecodeError) as error:
        raise CtrlWorldBridgeError("metadata output is invalid JSON") from error
    if any(not isinstance(value, list) for value in values):
        raise CtrlWorldBridgeError("metadata output must contain train/val arrays")


@contextmanager
def _exclusive_claim(path: Path, request_sha256: str) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise CtrlWorldBridgeError(f"preparation output is already claimed: {path}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(request_sha256 + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        path.unlink(missing_ok=True)


def _load_bound_request(request_path: Path, request_sha256: str) -> dict[str, object]:
    if sha256_file(request_path) != request_sha256:
        raise CtrlWorldBridgeError("Ctrl-World request hash changed after planning")
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CtrlWorldBridgeError("Ctrl-World request is invalid JSON") from error
    if not isinstance(request, dict):
        raise CtrlWorldBridgeError("Ctrl-World request must be an object")
    decoded = cast(dict[str, object], request)
    bridge = decoded.get("bridge_path")
    bridge_sha = decoded.get("bridge_sha256")
    this_bridge = Path(__file__).resolve()
    if (
        not isinstance(bridge, str)
        or Path(bridge).resolve() != this_bridge
        or not isinstance(bridge_sha, str)
        or sha256_file(this_bridge) != bridge_sha
    ):
        raise CtrlWorldBridgeError("checked-in Ctrl-World bridge binding changed")
    helper_rows = decoded.get("local_helpers")
    relatives = (
        *_LOCAL_HELPER_RELATIVES,
        *(
            relative
            for relative in _OPTIONAL_LOCAL_HELPER_RELATIVES
            if (_REPOSITORY_ROOT / relative).is_file()
        ),
    )
    expected_paths = [(_REPOSITORY_ROOT / relative).resolve() for relative in relatives]
    if not isinstance(helper_rows, list) or len(helper_rows) != len(expected_paths):
        raise CtrlWorldBridgeError("bound local helper inventory is not exact")
    for row, expected in zip(helper_rows, expected_paths, strict=True):
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "sha256"}
            or row.get("path") != str(expected)
            or not isinstance(row.get("sha256"), str)
            or sha256_file(expected) != row["sha256"]
        ):
            raise CtrlWorldBridgeError("bound local helper changed before bootstrap")
    return decoded


def _verify_training_request(
    request: Mapping[str, object],
    request_path: Path,
    revision_auditor: RevisionAuditor | None,
) -> tuple[Path, CtrlWorldTrainConfig]:
    _validate_request_identity(request, request_path)
    output_dir = Path(cast(str, request["output_dir"]))
    _ensure_outputs_absent(
        (
            output_dir / "checkpoint-5000.pt",
            output_dir / "training_state.json",
            output_dir / "job_metadata.json",
        )
    )
    source = _verify_source_checkout(request, revision_auditor)
    try:
        _helpers().verify_request_bindings(request)
        config = _config_from_request(request)
        manifest = _helpers().load_ctrl_world_manifest(config.manifest_path)
    except Exception as error:
        raise CtrlWorldBridgeError("bound Ctrl-World request verification failed") from error
    if manifest.sha256 != request["manifest_sha256"]:
        raise CtrlWorldBridgeError("projected manifest does not match the planned hash")
    return source, config


def _verify_source_checkout(
    request: Mapping[str, object], revision_auditor: RevisionAuditor | None
) -> Path:
    try:
        kwargs = {"revision_auditor": revision_auditor} if revision_auditor else {}
        source, source_blobs = _helpers().validate_ctrl_world_root(
            Path(cast(str, request["ctrl_world_root"])),
            repository_root=Path(cast(str, request["repository_root"])),
            **kwargs,
        )
    except Exception as error:
        raise CtrlWorldBridgeError("I2 Ctrl-World source verification failed") from error
    if source_blobs != request["source_blobs"]:
        raise CtrlWorldBridgeError("source blob binding changed")
    return cast(Path, source)


def _preflight_official_preparation(source: Path) -> None:
    _bootstrap_source(source)
    try:
        runpy.run_path(
            str(source / "dataset_example/extract_latent.py"),
            run_name="evoworld_extract_preflight",
        )
        runpy.run_path(
            str(source / "dataset_meta_info/create_meta_info.py"),
            run_name="evoworld_metadata_preflight",
        )
    except Exception as error:
        raise CtrlWorldBridgeError("official preparation preflight failed") from error


def _run_command(command: Sequence[str], context: OfficialExecutionContext | None) -> None:
    try:
        if context is None:
            subprocess.run(tuple(command), check=True)
            return
        if (
            not sys.platform.startswith("linux")
            or os.execve not in os.supports_fd
            or not Path("/proc/self/exe").exists()
        ):
            raise CtrlWorldBridgeError(
                "fd-backed official execution is unsupported on this platform"
            )
        executor = (
            "/proc/self/exe",
            str(Path(__file__).resolve()),
            "exec-official",
            "--workspace-fd",
            str(context.workspace_root_fd),
            "--working-fd",
            str(context.working_directory_fd),
            "--workspace-dev",
            str(context.workspace_identity[0]),
            "--workspace-ino",
            str(context.workspace_identity[1]),
            "--working-dev",
            str(context.working_directory_identity[0]),
            "--working-ino",
            str(context.working_directory_identity[1]),
            "--prepared-relative",
            context.prepared_relative,
            "--backend-fd",
            str(context.backend_fd),
            "--script-fd",
            str(context.script_fd),
            "--backend-binding",
            json.dumps(context.backend_binding),
            "--script-binding",
            json.dumps(context.script_binding),
            "--",
            *command,
        )
        subprocess.run(
            executor,
            check=True,
            pass_fds=(
                context.workspace_root_fd,
                context.working_directory_fd,
                context.backend_fd,
                context.script_fd,
            ),
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise CtrlWorldBridgeError("verified Ctrl-World child command failed") from error


def _config_from_request(request: Mapping[str, object]) -> CtrlWorldTrainConfig:
    paths = cast(dict[str, object], request["paths"])
    names = cast(dict[str, object], request["names"])
    output = Path(cast(str, request["output_dir"]))
    starting = paths["starting_checkpoint"]
    return cast(
        "CtrlWorldTrainConfig",
        _helpers().CtrlWorldTrainConfig(
            dataset_root_path=Path(cast(str, paths["dataset_root_path"])),
            dataset_meta_info_path=Path(cast(str, paths["dataset_meta_info_path"])),
            manifest_path=Path(cast(str, paths["manifest_path"])),
            output_root=output.parent,
            svd_model_path=Path(cast(str, paths["svd_model_path"])),
            clip_model_path=Path(cast(str, paths["clip_model_path"])),
            policy_checkpoint_path=Path(cast(str, paths["policy_checkpoint_path"])),
            action_adapter_path=Path(cast(str, paths["action_adapter_path"])),
            synthetic_output_root=Path(cast(str, paths["synthetic_output_root"])),
            starting_checkpoint=Path(cast(str, starting)) if starting else None,
            ctrl_world_python=Path(cast(str, request["ctrl_world_python"])),
            dataset_name=cast(str, names["dataset_name"]),
            annotation_name=cast(str, names["annotation_name"]),
            tag=cast(str, names["tag"]),
            wandb_run_name=cast(str, names["wandb_run_name"]),
            wandb_project_name=cast(str, names["wandb_project_name"]),
            max_train_steps=cast(int, request["max_train_steps"]),
            world_size=cast(int, request["world_size"]),
        ),
    )


def _validate_request_identity(request: Mapping[str, object], request_path: Path) -> None:
    required = {
        "schema_version",
        "job_id",
        "target",
        "manifest_sha256",
        "revision",
        "repository_root",
        "ctrl_world_root",
        "ctrl_world_python",
        "bridge_path",
        "bridge_sha256",
        "local_helpers",
        "environment_fingerprint",
        "environment_sha256",
        "source_blobs",
        "projection_root",
        "projection_bindings",
        "projection_sha256",
        "output_dir",
        "max_train_steps",
        "world_size",
        "inputs",
        "paths",
        "names",
    }
    if set(request) != required:
        raise CtrlWorldBridgeError("Ctrl-World request fields do not match the contract")
    job_id = request["job_id"]
    output = Path(cast(str, request["output_dir"])).resolve()
    if (
        request["schema_version"] != 1
        or request["revision"] != CTRL_WORLD_REVISION
        or request["target"] not in {"ctrl_world", "ctrl_world_base"}
        or request["max_train_steps"] != 5000
        or isinstance(request["world_size"], bool)
        or not isinstance(request["world_size"], int)
        or request["world_size"] < 1
        or not isinstance(job_id, str)
        or output.name != job_id
        or request_path.resolve() != output / "ctrl_world_request.json"
    ):
        raise CtrlWorldBridgeError("job/revision/target/output identity is not exact")


def _validate_preparation_request_identity(
    request: Mapping[str, object], request_path: Path, operation: str
) -> None:
    required = {
        "schema_version",
        "operation",
        "request_id",
        "revision",
        "repository_root",
        "ctrl_world_root",
        "ctrl_world_python",
        "bridge_path",
        "bridge_sha256",
        "local_helpers",
        "environment_fingerprint",
        "environment_sha256",
        "source_blobs",
        "trusted_roots",
        "official_execution",
        "world_size",
        "inputs",
        "output_bindings",
        "names",
    }
    if operation == "bind-metadata":
        required.add("latents_request_sha256")
    inputs = request.get("inputs")
    outputs = request.get("output_bindings")
    names = request.get("names")
    trusted_roots = request.get("trusted_roots")
    official_execution = request.get("official_execution")
    request_id = request.get("request_id")
    if (
        set(request) != required
        or operation not in {"prepare-latents", "bind-metadata"}
        or request.get("operation") != operation
        or request.get("schema_version") != 1
        or request.get("revision") != CTRL_WORLD_REVISION
        or not isinstance(request_id, str)
        or len(request_id) != 64
        or any(character not in "0123456789abcdef" for character in request_id)
        or request_path.name != f"ctrl-world-{operation}-{request_id[:12]}.json"
        or isinstance(request.get("world_size"), bool)
        or not isinstance(request.get("world_size"), int)
        or cast(int, request["world_size"]) < 1
        or request_path.resolve().parent.name != ".evoworld"
        or not isinstance(trusted_roots, dict)
        or set(trusted_roots) != {"workspace", "request"}
        or not isinstance(official_execution, dict)
        or set(official_execution) != {"backend", "metadata_script"}
        or not isinstance(inputs, dict)
        or set(inputs) != {"droid_hf_path", "svd_model_path"}
        or not isinstance(outputs, dict)
        or set(outputs)
        != {
            "droid_output_path",
            "metadata_output_path",
            "latents_claim_path",
            "latents_receipt_path",
            "metadata_claim_path",
            "metadata_work_root",
            "metadata_publish_claim_path",
            "metadata_completion_path",
            "metadata_quarantine_root",
        }
        or not isinstance(names, dict)
        or set(names) != {"dataset_name"}
        or not isinstance(names.get("dataset_name"), str)
        or not cast(str, names["dataset_name"]).replace("_", "").replace("-", "").isalnum()
    ):
        raise CtrlWorldBridgeError("preparation request identity is not exact")
    workspace_binding = _validate_root_binding(trusted_roots["workspace"], "workspace")
    request_binding = _validate_root_binding(trusted_roots["request"], "request")
    workspace = Path(cast(str, workspace_binding["path"]))
    if (
        Path(cast(str, request_binding["path"])) != workspace / ".evoworld"
        or request_path != workspace / ".evoworld" / request_path.name
    ):
        raise CtrlWorldBridgeError("preparation trusted roots are not exact")
    backend_binding = _validate_execution_file_binding(official_execution["backend"], "backend")
    script_binding = _validate_execution_file_binding(
        official_execution["metadata_script"], "metadata script"
    )
    environment = cast(dict[str, object], request["environment_fingerprint"])
    source_blobs = cast(dict[str, str], request["source_blobs"])
    if (
        backend_binding["path"] != request["ctrl_world_python"]
        or backend_binding["sha256"] != environment.get("launcher_sha256")
        or script_binding["path"]
        != str(
            Path(cast(str, request["ctrl_world_root"])) / "dataset_meta_info/create_meta_info.py"
        )
        or script_binding["sha256"] != source_blobs.get("dataset_meta_info/create_meta_info.py")
    ):
        raise CtrlWorldBridgeError("official execution binding is not exact")
    if operation == "bind-metadata" and (
        not isinstance(request.get("latents_request_sha256"), str)
        or len(cast(str, request["latents_request_sha256"])) != 64
    ):
        raise CtrlWorldBridgeError("metadata seed does not bind the latents request")
    for binding in cast(dict[str, object], inputs).values():
        if (
            not isinstance(binding, dict)
            or not isinstance(binding.get("path"), str)
            or not Path(cast(str, binding["path"])).is_absolute()
            or not isinstance(binding.get("sha256"), str)
        ):
            raise CtrlWorldBridgeError("preparation input binding is not exact")
    if not all(isinstance(value, str) and Path(value).is_absolute() for value in outputs.values()):
        raise CtrlWorldBridgeError("preparation output binding is not exact")
    droid_output = Path(cast(str, outputs["droid_output_path"]))
    metadata_output = Path(cast(str, outputs["metadata_output_path"]))
    name = cast(str, names["dataset_name"])
    internal = workspace / ".evoworld"
    if (
        workspace not in droid_output.parents
        or metadata_output != workspace / "dataset_meta_info" / name
        or Path(cast(str, outputs["latents_claim_path"])) != internal / f"{name}-latents.claim"
        or Path(cast(str, outputs["latents_receipt_path"]))
        != internal / f"{name}-latents-receipt.json"
        or Path(cast(str, outputs["metadata_claim_path"])) != internal / f"{name}-metadata.claim"
        or Path(cast(str, outputs["metadata_work_root"]))
        != internal / "metadata-work" / f"{name}-{request_id[:12]}"
        or Path(cast(str, outputs["metadata_publish_claim_path"]))
        != workspace / "dataset_meta_info" / f".{name}.publish.claim"
        or Path(cast(str, outputs["metadata_completion_path"]))
        != internal / f"{name}-metadata-receipt.json"
        or Path(cast(str, outputs["metadata_quarantine_root"]))
        != internal / "quarantine" / f"{name}-{request_id[:12]}"
    ):
        raise CtrlWorldBridgeError("preparation output binding escaped its workspace")


def _validate_root_binding(value: object, name: str) -> dict[str, object]:
    if (
        not isinstance(value, dict)
        or set(value) != {"path", "st_dev", "st_ino"}
        or not isinstance(value.get("path"), str)
        or not Path(cast(str, value["path"])).is_absolute()
        or isinstance(value.get("st_dev"), bool)
        or not isinstance(value.get("st_dev"), int)
        or isinstance(value.get("st_ino"), bool)
        or not isinstance(value.get("st_ino"), int)
    ):
        raise CtrlWorldBridgeError(f"{name} root binding is not exact")
    return cast(dict[str, object], value)


def _validate_execution_file_binding(value: object, name: str) -> dict[str, object]:
    if (
        not isinstance(value, dict)
        or set(value) != {"path", "st_dev", "st_ino", "size", "sha256"}
        or not isinstance(value.get("path"), str)
        or not Path(cast(str, value["path"])).is_absolute()
        or any(isinstance(value.get(key), bool) for key in ("st_dev", "st_ino", "size"))
        or not all(isinstance(value.get(key), int) for key in ("st_dev", "st_ino", "size"))
        or not isinstance(value.get("sha256"), str)
    ):
        raise CtrlWorldBridgeError(f"official {name} binding is not exact")
    return cast(dict[str, object], value)


def _bootstrap_source(source: Path) -> None:
    value = str(source.resolve())
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


def _require_module_containment(module: object, root: Path, name: str) -> None:
    module_file = getattr(module, "__file__", None)
    if (
        not isinstance(module_file, str)
        or root.resolve() not in Path(module_file).resolve().parents
    ):
        raise CtrlWorldBridgeError(
            f"official module {name} was imported outside the pinned checkout"
        )


def _reject_cephfs(args: object) -> None:
    for name in dir(args):
        if name.startswith("_"):
            continue
        try:
            value = getattr(args, name)
        except Exception as error:
            raise CtrlWorldBridgeError(f"unable to audit wm_args.{name}") from error
        if _contains_cephfs(value):
            raise CtrlWorldBridgeError(f"wm_args.{name} still contains a /cephfs default")


def _contains_cephfs(value: object) -> bool:
    if isinstance(value, str):
        return "/cephfs" in value
    if isinstance(value, dict):
        return any(_contains_cephfs(item) for item in (*value.keys(), *value.values()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_cephfs(item) for item in value)
    return False


def _ensure_outputs_absent(paths: tuple[Path, ...]) -> None:
    existing = [path.name for path in paths if path.exists() or path.is_symlink()]
    if existing:
        raise CtrlWorldBridgeError(
            f"refusing to overwrite existing job outputs: {', '.join(existing)}"
        )


def _ensure_safe_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise CtrlWorldBridgeError(f"unsafe job directory: {path}")


@contextmanager
def _open_trusted_roots(request: Mapping[str, object]) -> Iterator[tuple[int, int]]:
    roots = cast(dict[str, object], request["trusted_roots"])
    workspace = _validate_root_binding(roots["workspace"], "workspace")
    request_root = _validate_root_binding(roots["request"], "request")
    workspace_fd = -1
    request_fd = -1
    try:
        workspace_fd = _open_absolute_directory(Path(cast(str, workspace["path"])))
        _assert_fd_identity(workspace_fd, workspace, "workspace")
        request_fd = os.open(".evoworld", _directory_flags(), dir_fd=workspace_fd)
        _assert_fd_identity(request_fd, request_root, "request")
        yield workspace_fd, request_fd
    except OSError as error:
        raise CtrlWorldBridgeError("trusted preparation root changed") from error
    finally:
        if request_fd >= 0:
            os.close(request_fd)
        if workspace_fd >= 0:
            os.close(workspace_fd)


@contextmanager
def _open_official_files(request: Mapping[str, object]) -> Iterator[BoundOfficialFiles]:
    values = cast(dict[str, object], request["official_execution"])
    backend = _validate_execution_file_binding(values["backend"], "backend")
    script = _validate_execution_file_binding(values["metadata_script"], "metadata script")
    backend_fd = -1
    script_fd = -1
    try:
        backend_fd = _open_bound_regular_file(backend, "backend")
        script_fd = _open_bound_regular_file(script, "metadata source script")
        yield BoundOfficialFiles(
            backend_fd=backend_fd,
            script_fd=script_fd,
            backend_binding=_execution_binding_tuple(backend),
            script_binding=_execution_binding_tuple(script),
        )
    finally:
        if script_fd >= 0:
            os.close(script_fd)
        if backend_fd >= 0:
            os.close(backend_fd)


def _open_bound_regular_file(binding: Mapping[str, object], name: str) -> int:
    path = Path(cast(str, binding["path"]))
    parent_fd = _open_absolute_directory(path.parent)
    try:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    try:
        _verify_execution_file_fd(descriptor, _execution_binding_tuple(binding), name)
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _execution_binding_tuple(binding: Mapping[str, object]) -> tuple[int, int, int, str]:
    return (
        cast(int, binding["st_dev"]),
        cast(int, binding["st_ino"]),
        cast(int, binding["size"]),
        cast(str, binding["sha256"]),
    )


def _sha256_fd(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = sha256()
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _directory_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _open_absolute_directory(path: Path) -> int:
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise CtrlWorldBridgeError("trusted root path is not canonical")
    descriptor = os.open("/", _directory_flags())
    try:
        for part in path.parts[1:]:
            child = os.open(part, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _assert_fd_identity(descriptor: int, binding: Mapping[str, object], name: str) -> None:
    if _fd_identity(descriptor) != (binding["st_dev"], binding["st_ino"]):
        raise CtrlWorldBridgeError(f"trusted {name} root identity changed")


def _fd_identity(descriptor: int) -> tuple[int, int]:
    value = os.fstat(descriptor)
    return value.st_dev, value.st_ino


def _open_directory_tree(parent_fd: int, parts: tuple[str, ...], *, create: bool) -> int:
    descriptor = os.dup(parent_fd)
    try:
        for part in parts:
            if not part or part in {".", ".."} or "/" in part:
                raise CtrlWorldBridgeError("unsafe trusted-root directory component")
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            child = os.open(part, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _mkdir_open_exclusive(parent_fd: int, name: str) -> int:
    os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    return os.open(name, _directory_flags(), dir_fd=parent_fd)


def _mkdir_run_tree(work_root_fd: int, run_name: str) -> tuple[int, int]:
    run_fd = _mkdir_open_exclusive(work_root_fd, run_name)
    try:
        return run_fd, _mkdir_open_exclusive(run_fd, "dataset_meta_info")
    except Exception:
        os.close(run_fd)
        raise


def _fd_path(descriptor: int) -> Path:
    if sys.platform == "darwin":
        raw = fcntl.fcntl(descriptor, 50, b"\0" * 1024)
        return Path(raw.split(b"\0", 1)[0].decode())
    if sys.platform.startswith("linux"):
        return Path("/proc/self/fd") / str(descriptor)
    raise CtrlWorldBridgeError("trusted-root descriptor paths are unsupported")


def _ensure_absent_at(parent_fd: int, name: str) -> None:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise CtrlWorldBridgeError(f"refusing to overwrite existing job output: {name}")


def _publish_json_exclusive_at(parent_fd: int, name: str, value: Mapping[str, object]) -> None:
    temporary = f".{name}.{uuid4().hex}.tmp"
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=parent_fd,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise CtrlWorldBridgeError(f"job output appeared concurrently: {name}") from error
    finally:
        os.close(descriptor)
        os.unlink(temporary, dir_fd=parent_fd)


def _read_json_at(parent_fd: int, name: str) -> object:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
    with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _sha256_file_at(parent_fd: int, name: str) -> str:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
    digest = sha256()
    with os.fdopen(descriptor, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _exclusive_claim_at(parent_fd: int, name: str, request_sha256: str) -> Iterator[None]:
    try:
        descriptor = os.open(
            name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent_fd
        )
    except FileExistsError as error:
        raise CtrlWorldBridgeError(f"preparation output is already claimed: {name}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(request_sha256 + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        try:
            os.unlink(name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass


def _publish_directory_exclusive(
    staging_name: str,
    destination_name: str,
    staging_parent_fd: int,
    destination_parent_fd: int,
) -> None:
    try:
        staging_stat = os.stat(staging_name, dir_fd=staging_parent_fd, follow_symlinks=False)
    except OSError as error:
        raise CtrlWorldBridgeError("metadata staging output is no longer job-owned") from error
    if not stat.S_ISDIR(staging_stat.st_mode):
        raise CtrlWorldBridgeError("metadata staging output is unsafe")
    if staging_stat.st_dev != os.fstat(destination_parent_fd).st_dev:
        raise CtrlWorldBridgeError("metadata staging and destination are on different filesystems")
    _ensure_absent_at(destination_parent_fd, destination_name)
    _rename_directory_no_replace(
        staging_parent_fd,
        staging_name,
        destination_parent_fd,
        destination_name,
        "metadata output",
    )


def _rename_directory_no_replace(
    source_parent_fd: int,
    source_name: str,
    destination_parent_fd: int,
    destination_name: str,
    label: str,
) -> None:
    source_bytes = os.fsencode(source_name)
    destination_bytes = os.fsencode(destination_name)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = libc.renameatx_np
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            source_parent_fd,
            source_bytes,
            destination_parent_fd,
            destination_bytes,
            0x00000004,
        )
    elif sys.platform.startswith("linux"):
        rename = libc.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            source_parent_fd,
            source_bytes,
            destination_parent_fd,
            destination_bytes,
            1,
        )
    else:
        raise CtrlWorldBridgeError("atomic no-replace directory publication is unsupported")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise CtrlWorldBridgeError(f"{label} appeared concurrently: {destination_name}")
    raise CtrlWorldBridgeError(f"unable to publish {label}: {os.strerror(error_number)}")


def _temporary_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")


def _exclusive_publish(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        raise CtrlWorldBridgeError(f"job output appeared concurrently: {destination}") from error
    except OSError as error:
        raise CtrlWorldBridgeError(f"unable to publish job output: {destination}") from error
    temporary.unlink()


def _publish_json_exclusive(path: Path, value: Mapping[str, object]) -> None:
    temporary = _temporary_path(path)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _exclusive_publish(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _execute_official_with_fds(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workspace-fd", type=int, required=True)
    parser.add_argument("--working-fd", type=int, required=True)
    parser.add_argument("--workspace-dev", type=int, required=True)
    parser.add_argument("--workspace-ino", type=int, required=True)
    parser.add_argument("--working-dev", type=int, required=True)
    parser.add_argument("--working-ino", type=int, required=True)
    parser.add_argument("--prepared-relative", required=True)
    parser.add_argument("--backend-fd", type=int, required=True)
    parser.add_argument("--script-fd", type=int, required=True)
    parser.add_argument("--backend-binding", type=json.loads, required=True)
    parser.add_argument("--script-binding", type=json.loads, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = tuple(args.command)
    if command and command[0] == "--":
        command = command[1:]
    prepared = Path(args.prepared_relative)
    if (
        len(command) < 2
        or not Path(command[0]).is_absolute()
        or not Path(command[1]).is_absolute()
        or prepared.is_absolute()
        or os.path.normpath(args.prepared_relative) != args.prepared_relative
        or command.count("--droid_output_path") != 1
    ):
        raise CtrlWorldBridgeError("official inherited-fd command is invalid")
    prepared_index = command.index("--droid_output_path") + 1
    if prepared_index >= len(command) or command[prepared_index] != args.prepared_relative:
        raise CtrlWorldBridgeError("official prepared path is not fd-relative")
    if _fd_identity(args.workspace_fd) != (args.workspace_dev, args.workspace_ino):
        raise CtrlWorldBridgeError("inherited workspace root identity changed")
    if _fd_identity(args.working_fd) != (args.working_dev, args.working_ino):
        raise CtrlWorldBridgeError("inherited working directory identity changed")
    backend_binding = _decode_execution_file_binding(args.backend_binding, "backend")
    script_binding = _decode_execution_file_binding(args.script_binding, "script")
    _verify_execution_file_fd(args.backend_fd, backend_binding, "backend")
    _verify_execution_file_fd(args.script_fd, script_binding, "script")
    os.fchdir(args.working_fd)
    prepared_stat = os.lstat(args.prepared_relative)
    if stat.S_ISLNK(prepared_stat.st_mode) or not stat.S_ISDIR(prepared_stat.st_mode):
        raise CtrlWorldBridgeError("official prepared input is not a regular directory")
    os.set_inheritable(args.workspace_fd, True)
    os.set_inheritable(args.working_fd, True)
    os.set_inheritable(args.backend_fd, True)
    os.set_inheritable(args.script_fd, True)
    if os.execve not in os.supports_fd or not sys.platform.startswith("linux"):
        raise CtrlWorldBridgeError("fd-backed official execution is unsupported")
    official_command = (
        command[0],
        f"/proc/self/fd/{args.script_fd}",
        *command[2:],
    )
    os.execve(args.backend_fd, official_command, os.environ.copy())
    raise AssertionError("execvpe returned unexpectedly")


def _decode_execution_file_binding(value: object, name: str) -> tuple[int, int, int, str]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) for item in value[:3])
        or not all(isinstance(item, int) for item in value[:3])
        or not isinstance(value[3], str)
    ):
        raise CtrlWorldBridgeError(f"inherited {name} binding is invalid")
    return cast(tuple[int, int, int, str], tuple(value))


def _verify_execution_file_fd(
    descriptor: int, binding: tuple[int, int, int, str], name: str
) -> None:
    value = os.fstat(descriptor)
    if (
        not stat.S_ISREG(value.st_mode)
        or (value.st_dev, value.st_ino, value.st_size) != binding[:3]
        or _sha256_fd(descriptor) != binding[3]
    ):
        raise CtrlWorldBridgeError(f"inherited {name} file identity changed")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=("prepare-latents", "bind-metadata", "launch-train", "train-worker"),
    )
    parser.add_argument("--request-path", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    values = tuple(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "exec-official":
        return _execute_official_with_fds(values[1:])
    args = _parser().parse_args(values)
    request_path = args.request_path.resolve()
    if args.operation in {"prepare-latents", "bind-metadata"}:
        run_preparation(args.operation, request_path, args.request_sha256)
    elif args.operation == "launch-train":
        launch_training(request_path, args.request_sha256)
    else:
        execute_request(request_path, args.request_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

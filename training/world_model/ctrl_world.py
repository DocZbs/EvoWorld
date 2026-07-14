"""Pinned Ctrl-World preparation, validation, projection, and launch plans."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tomllib
from typing import cast

from evoworld.contracts import DiagnosisLabel
from training.common.schema import (
    TrainingManifest,
    TrainingPlan,
    TrainingSample,
    TrainingSchemaError,
)


CTRL_WORLD_REVISION = "99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d"
CTRL_WORLD_SUBMODULE = "third_party/ctrl_world"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_BRIDGE_PATH = _REPOSITORY_ROOT / "integrations/ctrl_world/train_bridge.py"
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
_SOURCE_FILES = (
    Path("config.py"),
    Path("scripts/train_wm.py"),
    Path("models/ctrl_world.py"),
    Path("dataset/dataset_droid_exp33.py"),
    Path("dataset_example/extract_latent.py"),
    Path("dataset_meta_info/create_meta_info.py"),
)
_ENVIRONMENT_DISTRIBUTIONS = (
    ("accelerate", ("accelerate", "accelerate.commands.launch", "accelerate.accelerator")),
    ("decord", ("decord",)),
    ("diffusers", ("diffusers",)),
    ("einops", ("einops",)),
    ("imageio", ("imageio", "imageio.v3")),
    ("imageio-ffmpeg", ("imageio_ffmpeg",)),
    ("mediapy", ("mediapy",)),
    ("numpy", ("numpy",)),
    ("omegaconf", ("omegaconf",)),
    ("pandas", ("pandas",)),
    ("safetensors", ("safetensors",)),
    ("scipy", ("scipy",)),
    ("swanlab", ("swanlab",)),
    (
        "torch",
        ("torch", "torch.optim", "torch.serialization", "torch.nn", "torch.utils.data"),
    ),
    ("torchvision", ("torchvision",)),
    ("tqdm", ("tqdm",)),
    ("transformers", ("transformers",)),
    ("wandb", ("wandb",)),
)
RevisionAuditor = Callable[[Path], Mapping[str, str]]
MediaInspector = Callable[[Path, tuple[Path, ...], tuple[Path, ...]], Mapping[str, object]]


class CtrlWorldConfigError(ValueError):
    """Raised when a Ctrl-World source, dataset, or job binding is unsafe."""


@dataclass(frozen=True, slots=True)
class CtrlWorldPrepareConfig:
    droid_hf_path: Path
    droid_output_path: Path
    svd_model_path: Path
    dataset_name: str
    ctrl_world_python: Path
    workspace_root: Path
    world_size: int

    def __post_init__(self) -> None:
        _require_absolute(
            (
                self.droid_hf_path,
                self.droid_output_path,
                self.svd_model_path,
                self.ctrl_world_python,
                self.workspace_root,
            )
        )
        _safe_name(self.dataset_name, "dataset_name")
        _require_contained(self.workspace_root, self.droid_output_path, "prepared dataset")
        if _paths_overlap(self.droid_hf_path, self.droid_output_path):
            raise CtrlWorldConfigError("DROID input and output paths must not overlap")
        if isinstance(self.world_size, bool) or self.world_size < 1:
            raise CtrlWorldConfigError("world_size must be positive")


@dataclass(frozen=True, slots=True)
class CtrlWorldTrainConfig:
    dataset_root_path: Path
    dataset_meta_info_path: Path
    manifest_path: Path
    output_root: Path
    svd_model_path: Path
    clip_model_path: Path
    policy_checkpoint_path: Path
    action_adapter_path: Path
    synthetic_output_root: Path
    starting_checkpoint: Path | None
    ctrl_world_python: Path
    dataset_name: str
    annotation_name: str
    tag: str
    wandb_run_name: str
    wandb_project_name: str
    max_train_steps: int
    world_size: int

    def __post_init__(self) -> None:
        paths = (
            self.dataset_root_path,
            self.dataset_meta_info_path,
            self.manifest_path,
            self.output_root,
            self.svd_model_path,
            self.clip_model_path,
            self.policy_checkpoint_path,
            self.action_adapter_path,
            self.synthetic_output_root,
            self.ctrl_world_python,
        )
        _require_absolute(paths)
        if self.starting_checkpoint is not None:
            _require_absolute((self.starting_checkpoint,))
        for name, value in (
            ("dataset_name", self.dataset_name),
            ("annotation_name", self.annotation_name),
            ("tag", self.tag),
            ("wandb_run_name", self.wandb_run_name),
            ("wandb_project_name", self.wandb_project_name),
        ):
            _safe_name(value, name)
        if self.annotation_name != "annotation":
            raise CtrlWorldConfigError("Ctrl-World requires annotation_name='annotation'")
        if self.max_train_steps != 5000:
            raise CtrlWorldConfigError("Ctrl-World training requires exactly 5,000 optimizer steps")
        if isinstance(self.world_size, bool) or self.world_size < 1:
            raise CtrlWorldConfigError("world_size must be positive")
        path_values = (*paths, *((self.starting_checkpoint,) if self.starting_checkpoint else ()))
        if any("/cephfs" in str(path) for path in path_values):
            raise CtrlWorldConfigError("Ctrl-World paths must not contain /cephfs defaults")

    @property
    def dataset_path(self) -> Path:
        return self.dataset_root_path / self.dataset_name

    @property
    def metadata_path(self) -> Path:
        return self.dataset_meta_info_path / self.dataset_name

    @property
    def stats_path(self) -> Path:
        return self.metadata_path / "stat.json"


def _default_revision_audit(repository_root: Path) -> Mapping[str, str]:
    try:
        module = importlib.import_module("evoworld.adapters.revisions")
        inspect_submodules = module.inspect_submodules
    except (ImportError, AttributeError) as error:
        raise CtrlWorldConfigError("I2 recursive revision audit is unavailable") from error
    try:
        result = inspect_submodules(repository_root)
    except Exception as error:
        raise CtrlWorldConfigError("I2 recursive revision audit rejected the checkout") from error
    if not isinstance(result, Mapping):
        raise CtrlWorldConfigError("I2 revision audit returned an invalid result")
    return cast(Mapping[str, str], result)


def validate_ctrl_world_root(
    ctrl_world_root: Path,
    *,
    repository_root: Path,
    revision_auditor: RevisionAuditor = _default_revision_audit,
) -> tuple[Path, dict[str, str]]:
    """Use I2 to require the exact clean recursive checkout, then hash used source."""
    _require_absolute((ctrl_world_root, repository_root))
    repository = repository_root.resolve()
    source = ctrl_world_root.resolve()
    expected = (repository / CTRL_WORLD_SUBMODULE).resolve()
    if source != expected:
        raise CtrlWorldConfigError("Ctrl-World root must be repository/third_party/ctrl_world")
    revisions = revision_auditor(repository)
    if revisions.get(CTRL_WORLD_SUBMODULE) != CTRL_WORLD_REVISION:
        raise CtrlWorldConfigError("audited Ctrl-World revision does not match the pin")
    blobs: dict[str, str] = {}
    for relative in _SOURCE_FILES:
        path = source / relative
        if not path.is_file():
            raise CtrlWorldConfigError(f"Ctrl-World source is missing {relative.as_posix()}")
        blobs[relative.as_posix()] = sha256_file(path)
    return source, blobs


def fingerprint_ctrl_world_environment(python: Path) -> dict[str, object]:
    """Re-hash every recorded distribution file and required runtime module."""
    if not python.is_file() or not os.access(python, os.X_OK):
        raise CtrlWorldConfigError("ctrl_world_python must be an executable file")
    script = r"""import base64, csv, hashlib, importlib.machinery, importlib.metadata, io, json, pathlib, stat, sys, urllib.parse
MAX_DISTRIBUTION_FILES = 100000
MAX_DISTRIBUTION_BYTES = 4294967296
MAX_TOTAL_FILES = 250000
MAX_FILE_BYTES = 1073741824
MAX_DIRECT_URL_BYTES = 1048576
MAX_TOTAL_BYTES = 8589934592
MAX_RECORD_BYTES = 16777216

def contained(path, roots):
    return any(path == root or root in path.parents for root in roots)

def stream_hash(path, distribution_bytes, total_bytes):
    before = path.stat()
    if (
        before.st_size > MAX_FILE_BYTES
        or distribution_bytes + before.st_size > MAX_DISTRIBUTION_BYTES
        or total_bytes + before.st_size > MAX_TOTAL_BYTES
    ):
        raise RuntimeError(f"environment inventory byte bound exceeded: {path}")
    digest = hashlib.sha256()
    read_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1048576):
            read_bytes += len(chunk)
            digest.update(chunk)
    after = path.stat()
    if read_bytes != before.st_size or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"environment file changed while hashing: {path}")
    return digest.hexdigest(), read_bytes

def stream_hash_and_read(path, distribution_bytes, total_bytes):
    before = path.stat()
    if (
        before.st_size > MAX_DIRECT_URL_BYTES
        or before.st_size > MAX_FILE_BYTES
        or distribution_bytes + before.st_size > MAX_DISTRIBUTION_BYTES
        or total_bytes + before.st_size > MAX_TOTAL_BYTES
    ):
        raise RuntimeError(f"environment inventory byte bound exceeded: {path}")
    digest = hashlib.sha256()
    content = bytearray()
    read_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1048576):
            read_bytes += len(chunk)
            digest.update(chunk)
            content.extend(chunk)
    after = path.stat()
    if read_bytes != before.st_size or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"environment file changed while hashing: {path}")
    return digest.hexdigest(), read_bytes, content

def locate_without_import(module_name):
    search = None
    fullname = ""
    spec = None
    for part in module_name.split("."):
        fullname = part if not fullname else fullname + "." + part
        spec = importlib.machinery.PathFinder.find_spec(fullname, search)
        if spec is None:
            raise RuntimeError(f"module cannot be located without import: {module_name}")
        search = list(spec.submodule_search_locations or ())
    origin = None if spec is None else spec.origin
    if not isinstance(origin, str) or origin in {"built-in", "frozen"}:
        raise RuntimeError(f"module has no filesystem origin: {module_name}")
    return pathlib.Path(origin).resolve(strict=True)

specifications = json.loads(sys.stdin.read())
distributions = {}
total_files = 0
total_entries = 0
total_bytes = 0
for distribution_name, module_names in specifications:
    if distribution_name in distributions:
        raise RuntimeError(f"duplicate distribution specification: {distribution_name}")
    distribution = importlib.metadata.distribution(distribution_name)
    distribution_info = getattr(distribution, "_path", None)
    if not isinstance(distribution_info, pathlib.Path) or not distribution_info.is_dir():
        raise RuntimeError(f"distribution has no filesystem metadata root: {distribution_name}")
    distribution_info = distribution_info.resolve(strict=True)
    distribution_root = distribution_info.parent
    record_path = distribution_info / "RECORD"
    if not record_path.is_file() or record_path.is_symlink():
        raise RuntimeError(f"distribution has no regular RECORD: {distribution_name}")
    record_size = record_path.stat().st_size
    if record_size > MAX_RECORD_BYTES:
        raise RuntimeError(f"distribution RECORD is too large: {distribution_name}")
    record_bytes = record_path.read_bytes()
    if len(record_bytes) != record_size:
        raise RuntimeError(f"distribution RECORD changed while reading: {distribution_name}")
    if total_bytes + len(record_bytes) > MAX_TOTAL_BYTES:
        raise RuntimeError(f"environment inventory byte bound exceeded: {distribution_name}")
    total_bytes += len(record_bytes)
    total_files += 1
    distribution_bytes = len(record_bytes)
    distribution_files = 1
    try:
        record_text = record_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"distribution RECORD is not UTF-8: {distribution_name}") from error
    rows = list(csv.reader(io.StringIO(record_text, newline="")))
    if len(rows) > MAX_DISTRIBUTION_FILES or total_entries + len(rows) > MAX_TOTAL_FILES:
        raise RuntimeError(f"environment inventory file bound exceeded: {distribution_name}")
    total_entries += len(rows)
    seen = set()
    seen_resolved = set()
    selected = []
    prehashed = {}
    allowed_roots = {
        distribution_root,
        pathlib.Path(sys.prefix).resolve(strict=True),
        pathlib.Path(sys.base_prefix).resolve(strict=True),
    }
    parsed = []
    for row in rows:
        if len(row) != 3:
            raise RuntimeError(f"invalid RECORD row width: {distribution_name}")
        relative, recorded_hash, recorded_size = row
        pure = pathlib.PurePosixPath(relative)
        if (
            not relative
            or "\\" in relative
            or pure.is_absolute()
            or pure.as_posix() != relative
            or any(part in {"", "."} for part in pure.parts)
            or ":" in pure.parts[0]
            or any(ord(character) < 32 for character in relative)
            or relative in seen
        ):
            raise RuntimeError(f"unsafe or duplicate RECORD entry: {distribution_name}: {relative}")
        if (recorded_hash and "=" not in recorded_hash) or (
            recorded_size and not recorded_size.isdigit()
        ):
            raise RuntimeError(f"malformed RECORD metadata: {distribution_name}: {relative}")
        seen.add(relative)
        lexical_path = distribution_root / pathlib.Path(*pure.parts)
        entry_stat = lexical_path.lstat()
        if stat.S_ISLNK(entry_stat.st_mode) or not stat.S_ISREG(entry_stat.st_mode):
            raise RuntimeError(f"RECORD entry is not a regular non-symlink file: {relative}")
        path = lexical_path.resolve(strict=True)
        if not contained(path, allowed_roots) or path in seen_resolved:
            raise RuntimeError(f"RECORD entry escaped allowed roots: {distribution_name}: {relative}")
        seen_resolved.add(path)
        parsed.append((relative, recorded_hash, recorded_size, path))
    editable_roots = set()
    editable_binding = None
    direct_entries = [row for row in parsed if row[0].endswith(".dist-info/direct_url.json")]
    if len(direct_entries) > 1:
        raise RuntimeError(f"distribution has duplicate direct_url metadata: {distribution_name}")
    runtime_roots = set()
    for value in sys.path:
        if not value:
            continue
        try:
            runtime_path = pathlib.Path(value).resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if runtime_path.is_dir():
            runtime_roots.add(runtime_path)
    for relative, _, _, path in direct_entries:
        if distribution_files + 1 > MAX_DISTRIBUTION_FILES:
            raise RuntimeError(f"distribution inventory file bound exceeded: {distribution_name}")
        direct_sha, direct_size, direct_bytes = stream_hash_and_read(
            path, distribution_bytes, total_bytes
        )
        distribution_bytes += direct_size
        distribution_files += 1
        total_bytes += direct_size
        total_files += 1
        prehashed[path] = (direct_sha, direct_size)
        try:
            direct = json.loads(direct_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"distribution direct_url is invalid: {distribution_name}") from error
        if relative.endswith(".dist-info/direct_url.json"):
            if direct.get("dir_info", {}).get("editable") is True:
                url = urllib.parse.urlparse(direct.get("url", ""))
                if url.scheme != "file" or url.netloc not in {"", "localhost"} or url.query or url.fragment:
                    raise RuntimeError(f"editable distribution URL is not local: {distribution_name}")
                editable_lexical = pathlib.Path(urllib.parse.unquote(url.path))
                if not editable_lexical.is_absolute():
                    raise RuntimeError(f"editable distribution root is not absolute: {distribution_name}")
                editable_stat = editable_lexical.lstat()
                editable = editable_lexical.resolve(strict=True)
                if (
                    editable_lexical != editable
                    or stat.S_ISLNK(editable_stat.st_mode)
                    or not stat.S_ISDIR(editable_stat.st_mode)
                    or editable not in runtime_roots
                ):
                    raise RuntimeError(f"editable distribution root is invalid: {distribution_name}")
                editable_roots.add(editable)
                editable_binding = {
                    "direct_url_path": relative,
                    "direct_url_sha256": direct_sha,
                    "direct_url_size": direct_size,
                    "root": str(editable),
                }
    allowed_roots.update(editable_roots)
    for relative, recorded_hash, _, path in parsed:
        if not path.is_file():
            continue
        if (
            not recorded_hash
            and path not in prehashed
            and path.suffix.lower() not in {".py", ".so", ".dylib", ".pyd"}
        ):
            continue
        if path in prehashed:
            digest, size = prehashed[path]
        else:
            if distribution_files + 1 > MAX_DISTRIBUTION_FILES:
                raise RuntimeError(
                    f"distribution inventory file bound exceeded: {distribution_name}"
                )
            digest, size = stream_hash(path, distribution_bytes, total_bytes)
            distribution_bytes += size
            distribution_files += 1
            total_bytes += size
            total_files += 1
        selected.append({"path": relative, "sha256": digest, "size": size})
    if not selected:
        raise RuntimeError(f"distribution has no integrity files: {distribution_name}")
    modules = []
    for module_name in module_names:
        prefix = module_name.replace(".", "/")
        exact = {prefix + ".py", prefix + "/__init__.py"}
        candidates = [
            row for row in selected
            if row["path"] in exact
            or (
                pathlib.PurePosixPath(row["path"]).parent.as_posix()
                == pathlib.PurePosixPath(prefix).parent.as_posix()
                and pathlib.PurePosixPath(row["path"]).name.startswith(
                    pathlib.PurePosixPath(prefix).name + "."
                )
                and pathlib.PurePosixPath(row["path"]).suffix.lower()
                in {".so", ".dylib", ".pyd"}
            )
        ]
        if len(candidates) > 1:
            raise RuntimeError(f"module has duplicate RECORD files: {module_name}")
        if candidates:
            relative = candidates[0]["path"]
            module_sha = candidates[0]["sha256"]
        else:
            module_path = locate_without_import(module_name)
            if not editable_roots or not contained(module_path, editable_roots):
                raise RuntimeError(f"module is absent from RECORD and editable root: {module_name}")
            if distribution_files + 1 > MAX_DISTRIBUTION_FILES:
                raise RuntimeError(
                    f"distribution inventory file bound exceeded: {distribution_name}"
                )
            module_sha, module_size = stream_hash(
                module_path, distribution_bytes, total_bytes
            )
            distribution_bytes += module_size
            distribution_files += 1
            total_bytes += module_size
            total_files += 1
            editable_root = next(root for root in editable_roots if contained(module_path, {root}))
            relative = "editable/" + module_path.relative_to(editable_root).as_posix()
        modules.append({
            "name": module_name,
            "path": relative,
            "sha256": module_sha,
        })
    distribution_value = {
        "version": distribution.version,
        "record": {
            "path": distribution_info.name + "/RECORD",
            "sha256": hashlib.sha256(record_bytes).hexdigest(),
            "size": len(record_bytes),
            "bytes_base64": base64.b64encode(record_bytes).decode("ascii"),
        },
        "files": sorted(selected, key=lambda row: row["path"]),
        "modules": modules,
    }
    if editable_binding is not None:
        distribution_value["editable"] = editable_binding
    distributions[distribution_name] = distribution_value
value = {
    "python_executable": str(pathlib.Path(sys.executable).resolve(strict=True)),
    "python_version": list(sys.version_info[:3]),
    "distributions": distributions,
}
print(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
"""
    try:
        completed = subprocess.run(
            (str(python), "-c", script),
            input=json.dumps(_ENVIRONMENT_DISTRIBUTIONS),
            check=True,
            capture_output=True,
            text=True,
        )
        decoded = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise CtrlWorldConfigError("Ctrl-World environment fingerprint failed") from error
    if not isinstance(decoded, dict):
        raise CtrlWorldConfigError("Ctrl-World environment fingerprint is invalid")
    fingerprint = cast(dict[str, object], decoded)
    fingerprint["launcher_path"] = str(python.resolve())
    fingerprint["launcher_sha256"] = sha256_file(python)
    return fingerprint


def plan_ctrl_world_prepare(
    config: CtrlWorldPrepareConfig,
    *,
    ctrl_world_root: Path,
    repository_root: Path,
    revision_auditor: RevisionAuditor = _default_revision_audit,
) -> tuple[TrainingPlan, TrainingPlan]:
    source, source_blobs = validate_ctrl_world_root(
        ctrl_world_root,
        repository_root=repository_root,
        revision_auditor=revision_auditor,
    )
    environment = fingerprint_ctrl_world_environment(config.ctrl_world_python)
    if not config.workspace_root.is_dir() or not config.droid_hf_path.is_dir():
        raise CtrlWorldConfigError("preparation workspace and DROID source must exist")
    if not config.svd_model_path.exists():
        raise CtrlWorldConfigError("SVD model path must exist")
    workspace = config.workspace_root.resolve(strict=True)
    if workspace != config.workspace_root or config.workspace_root.is_symlink():
        raise CtrlWorldConfigError("preparation workspace must be canonical and non-symlink")
    if config.droid_output_path.resolve() != config.droid_output_path:
        raise CtrlWorldConfigError("prepared output path must be canonical")
    request_directory = workspace / ".evoworld"
    request_directory.mkdir(exist_ok=True)
    if request_directory.is_symlink() or not request_directory.is_dir():
        raise CtrlWorldConfigError("preparation request root is unsafe")
    workspace_stat = workspace.stat()
    request_root_stat = request_directory.stat()
    trusted_roots = {
        "workspace": {
            "path": str(workspace),
            "st_dev": workspace_stat.st_dev,
            "st_ino": workspace_stat.st_ino,
        },
        "request": {
            "path": str(request_directory),
            "st_dev": request_root_stat.st_dev,
            "st_ino": request_root_stat.st_ino,
        },
    }
    official_execution = {
        "backend": _regular_file_binding(config.ctrl_world_python),
        "metadata_script": _regular_file_binding(source / "dataset_meta_info/create_meta_info.py"),
    }
    digest = _canonical_sha(
        {
            "config": {
                "droid_hf_path": str(config.droid_hf_path),
                "droid_output_path": str(config.droid_output_path),
                "svd_model_path": str(config.svd_model_path),
                "dataset_name": config.dataset_name,
                "world_size": config.world_size,
            },
            "revision": CTRL_WORLD_REVISION,
            "environment": environment,
            "source_blobs": source_blobs,
            "trusted_roots": trusted_roots,
            "official_execution": official_execution,
        }
    )
    output_relative = config.droid_output_path.relative_to(config.workspace_root)
    request_common: dict[str, object] = {
        "schema_version": 1,
        "revision": CTRL_WORLD_REVISION,
        "repository_root": str(repository_root.resolve()),
        "ctrl_world_root": str(source),
        "ctrl_world_python": str(config.ctrl_world_python),
        "bridge_path": str(_BRIDGE_PATH),
        "bridge_sha256": sha256_file(_BRIDGE_PATH),
        "local_helpers": _local_helper_bindings(),
        "environment_fingerprint": environment,
        "environment_sha256": _canonical_sha(environment),
        "source_blobs": source_blobs,
        "trusted_roots": trusted_roots,
        "official_execution": official_execution,
        "world_size": config.world_size,
        "inputs": {
            "droid_hf_path": {
                "path": str(config.droid_hf_path),
                "sha256": sha256_path(config.droid_hf_path),
            },
            "svd_model_path": {
                "path": str(config.svd_model_path),
                "sha256": sha256_path(config.svd_model_path),
            },
        },
        "output_bindings": {
            "droid_output_path": str(config.droid_output_path),
            "metadata_output_path": str(
                config.workspace_root / "dataset_meta_info" / config.dataset_name
            ),
            "latents_claim_path": str(request_directory / f"{config.dataset_name}-latents.claim"),
            "latents_receipt_path": str(
                request_directory / f"{config.dataset_name}-latents-receipt.json"
            ),
            "metadata_claim_path": str(request_directory / f"{config.dataset_name}-metadata.claim"),
            "metadata_work_root": str(
                request_directory / "metadata-work" / f"{config.dataset_name}-{digest[:12]}"
            ),
            "metadata_publish_claim_path": str(
                config.workspace_root
                / "dataset_meta_info"
                / f".{config.dataset_name}.publish.claim"
            ),
            "metadata_completion_path": str(
                request_directory / f"{config.dataset_name}-metadata-receipt.json"
            ),
            "metadata_quarantine_root": str(
                request_directory / "quarantine" / f"{config.dataset_name}-{digest[:12]}"
            ),
        },
        "names": {"dataset_name": config.dataset_name},
    }
    extraction_payload = {
        **request_common,
        "operation": "prepare-latents",
        "request_id": digest,
    }
    extraction_request = request_directory / f"ctrl-world-prepare-latents-{digest[:12]}.json"
    _write_exact_json(extraction_request, extraction_payload)
    extraction_sha = sha256_file(extraction_request)
    metadata_payload = {
        **request_common,
        "operation": "bind-metadata",
        "request_id": digest,
        "latents_request_sha256": extraction_sha,
    }
    metadata_request = request_directory / f"ctrl-world-bind-metadata-{digest[:12]}.json"
    _write_exact_json(metadata_request, metadata_payload)
    metadata_sha = sha256_file(metadata_request)
    try:
        extraction = TrainingPlan(
            schema_version=1,
            job_id=f"ctrl-world-extract-{digest[:12]}",
            target="ctrl_world_prepare",
            manifest_sha256=digest,
            upstream_revision=CTRL_WORLD_REVISION,
            cwd=config.workspace_root.resolve(),
            argv=(
                str(config.ctrl_world_python),
                str(_BRIDGE_PATH),
                "prepare-latents",
                "--request-path",
                str(extraction_request),
                "--request-sha256",
                extraction_sha,
            ),
            environment={},
            expected_outputs=(
                output_relative / "annotation/train",
                output_relative / "annotation/val",
                output_relative / "videos",
                output_relative / "latent_videos",
            ),
        )
        metadata = TrainingPlan(
            schema_version=1,
            job_id=f"ctrl-world-metadata-{digest[:12]}",
            target="ctrl_world_metadata",
            manifest_sha256=digest,
            upstream_revision=CTRL_WORLD_REVISION,
            cwd=config.workspace_root.resolve(),
            argv=(
                str(config.ctrl_world_python),
                str(_BRIDGE_PATH),
                "bind-metadata",
                "--request-path",
                str(metadata_request),
                "--request-sha256",
                metadata_sha,
            ),
            environment={},
            expected_outputs=(
                Path(f"dataset_meta_info/{config.dataset_name}/train_sample.json"),
                Path(f"dataset_meta_info/{config.dataset_name}/val_sample.json"),
            ),
        )
    except TrainingSchemaError as error:
        raise CtrlWorldConfigError("unable to build safe Ctrl-World preparation plans") from error
    return extraction, metadata


def load_ctrl_world_config(path: Path) -> dict[str, str | int]:
    try:
        values = tomllib.loads(path.read_text(encoding="utf-8"))["ctrl_world"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as error:
        raise CtrlWorldConfigError(f"unable to load Ctrl-World configuration {path}") from error
    expected: dict[str, str | int] = {
        "revision": CTRL_WORLD_REVISION,
        "max_train_steps": 5000,
        "world_size": 1,
        "annotation_name": "annotation",
    }
    if values != expected:
        raise CtrlWorldConfigError("Ctrl-World TOML must contain only the pinned contract")
    return expected


def compute_ctrl_world_sample_sha256(root: Path, sample: TrainingSample) -> str:
    if not root.is_absolute():
        raise CtrlWorldConfigError("dataset root must be absolute")
    digest = sha256()
    media = (*sample.observation_paths, *((sample.action_path,) if sample.action_path else ()))
    for index, relative in enumerate(media):
        path = _contained_file(root, root / sample.dataset_root / relative, "sample artifact")
        digest.update(f"{index}:{sample.dataset_root / relative}\0".encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def inspect_ctrl_world_media(
    python: Path,
    videos: tuple[Path, ...],
    latents: tuple[Path, ...],
) -> Mapping[str, object]:
    """Decode videos and load tensors inside the configured Ctrl-World environment."""
    script = r"""import json, sys
import imageio.v3 as iio
import torch
request = json.load(sys.stdin)
result = {"videos": {}, "latents": {}}
for path in request["videos"]:
    frames = list(iio.imiter(path))
    if not frames:
        raise ValueError(f"empty video: {path}")
    result["videos"][path] = {
        "frames": len(frames), "shape": list(frames[0].shape),
    }
for path in request["latents"]:
    with open(path, "rb") as stream:
        tensor = torch.load(stream, map_location="cpu")
    result["latents"][path] = {
        "shape": list(tensor.shape), "dtype": str(tensor.dtype),
    }
print(json.dumps(result, allow_nan=False))
"""
    try:
        completed = subprocess.run(
            (str(python), "-c", script),
            input=json.dumps(
                {
                    "videos": [str(path) for path in videos],
                    "latents": [str(path) for path in latents],
                }
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        decoded = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise CtrlWorldConfigError("Ctrl-World environment could not decode media") from error
    if not isinstance(decoded, dict):
        raise CtrlWorldConfigError("Ctrl-World media inspector returned invalid output")
    return cast(Mapping[str, object], decoded)


def validate_ctrl_world_dataset(
    config: CtrlWorldTrainConfig,
    manifest: TrainingManifest,
    *,
    media_inspector: MediaInspector = inspect_ctrl_world_media,
) -> dict[int, dict[str, object]]:
    dataset = config.dataset_path.resolve()
    metadata = config.metadata_path.resolve()
    annotations_by_split: dict[str, dict[int, dict[str, object]]] = {}
    all_videos: list[Path] = []
    all_latents: list[Path] = []
    for split in ("train", "val"):
        directory = dataset / config.annotation_name / split
        if not directory.is_dir():
            raise CtrlWorldConfigError(f"Ctrl-World dataset is missing annotation/{split}")
        annotations: dict[int, dict[str, object]] = {}
        for path in sorted(directory.glob("*.json")):
            value = _read_object(path)
            episode_id, videos, latents = _validate_annotation(dataset, split, value)
            if episode_id in annotations or path.stem != str(episode_id):
                raise CtrlWorldConfigError(
                    "annotation filenames and episode IDs must be one-to-one"
                )
            annotations[episode_id] = value
            all_videos.extend(videos)
            all_latents.extend(latents)
        if not annotations:
            raise CtrlWorldConfigError(f"annotation/{split} must not be empty")
        _validate_metadata_rows(metadata, split, annotations)
        annotations_by_split[split] = annotations
    _validate_media_inspection(
        media_inspector(config.ctrl_world_python, tuple(all_videos), tuple(all_latents)),
        annotations_by_split,
        dataset,
    )
    _validate_stats(config.stats_path)
    expected_sidecar = metadata / "evoworld_manifest.json"
    if config.manifest_path.resolve() != expected_sidecar:
        raise CtrlWorldConfigError("manifest path must be the real metadata sidecar")
    if load_ctrl_world_manifest(config.manifest_path) != manifest:
        raise CtrlWorldConfigError("manifest sidecar does not match the planned manifest")
    _validate_manifest_bijection(config, manifest, annotations_by_split["train"])
    return annotations_by_split["train"]


def plan_ctrl_world_training(
    manifest: TrainingManifest,
    config: CtrlWorldTrainConfig,
    *,
    ctrl_world_root: Path,
    repository_root: Path,
    revision_auditor: RevisionAuditor = _default_revision_audit,
    media_inspector: MediaInspector = inspect_ctrl_world_media,
) -> TrainingPlan:
    if config.starting_checkpoint is None or not config.starting_checkpoint.is_file():
        raise CtrlWorldConfigError("WMF post-training requires an explicit starting checkpoint")
    if any(sample.diagnosis is not DiagnosisLabel.WMF for sample in manifest.samples):
        raise CtrlWorldConfigError("WMF jobs accept only WMF diagnosis records")
    return _plan_training(
        manifest,
        config,
        ctrl_world_root=ctrl_world_root,
        repository_root=repository_root,
        target="ctrl_world",
        revision_auditor=revision_auditor,
        media_inspector=media_inspector,
    )


def plan_ctrl_world_base_training(
    manifest: TrainingManifest,
    config: CtrlWorldTrainConfig,
    *,
    ctrl_world_root: Path,
    repository_root: Path,
    revision_auditor: RevisionAuditor = _default_revision_audit,
    media_inspector: MediaInspector = inspect_ctrl_world_media,
) -> TrainingPlan:
    if config.starting_checkpoint is not None:
        raise CtrlWorldConfigError("public base training must not warm-start a checkpoint")
    if any(
        sample.diagnosis is not DiagnosisLabel.NONE
        or sample.split != "train"
        or sample.source_type != "real"
        for sample in manifest.samples
    ):
        raise CtrlWorldConfigError("public base training requires real-source NONE/train records")
    return _plan_training(
        manifest,
        config,
        ctrl_world_root=ctrl_world_root,
        repository_root=repository_root,
        target="ctrl_world_base",
        revision_auditor=revision_auditor,
        media_inspector=media_inspector,
    )


def _plan_training(
    manifest: TrainingManifest,
    config: CtrlWorldTrainConfig,
    *,
    ctrl_world_root: Path,
    repository_root: Path,
    target: str,
    revision_auditor: RevisionAuditor,
    media_inspector: MediaInspector,
) -> TrainingPlan:
    source, source_blobs = validate_ctrl_world_root(
        ctrl_world_root,
        repository_root=repository_root,
        revision_auditor=revision_auditor,
    )
    environment = fingerprint_ctrl_world_environment(config.ctrl_world_python)
    _validate_runtime_paths(config)
    if _paths_overlap(config.output_root, source) or _paths_overlap(
        config.output_root, repository_root
    ):
        raise CtrlWorldConfigError("output root must not overlap the repository or submodule")
    train_annotations = validate_ctrl_world_dataset(
        config, manifest, media_inspector=media_inspector
    )
    job_id = f"{target.replace('_', '-')}-{manifest.sha256[:12]}"
    job_root = (config.output_root / job_id).resolve()
    for output_name in ("checkpoint-5000.pt", "training_state.json", "job_metadata.json"):
        if (job_root / output_name).exists():
            raise CtrlWorldConfigError("final job output already exists; refusing to overwrite")
    projected = _materialize_projection(job_root, config, manifest, train_annotations)
    projection_bindings = _artifact_bindings(projected)
    projection_sha = _canonical_sha(projection_bindings)
    output_dir = job_root
    inputs = {
        "manifest": sha256_file(
            projected / "dataset_meta_info" / config.dataset_name / "evoworld_manifest.json"
        ),
        "checkpoint": sha256_file(config.starting_checkpoint)
        if config.starting_checkpoint
        else None,
        "svd_model": sha256_path(config.svd_model_path),
        "clip_model": sha256_path(config.clip_model_path),
        "policy_checkpoint": sha256_path(config.policy_checkpoint_path),
        "action_adapter": sha256_path(config.action_adapter_path),
    }
    request: dict[str, object] = {
        "schema_version": 1,
        "job_id": job_id,
        "target": target,
        "manifest_sha256": manifest.sha256,
        "revision": CTRL_WORLD_REVISION,
        "repository_root": str(repository_root.resolve()),
        "ctrl_world_root": str(source),
        "ctrl_world_python": str(config.ctrl_world_python),
        "bridge_path": str(_BRIDGE_PATH),
        "bridge_sha256": sha256_file(_BRIDGE_PATH),
        "local_helpers": _local_helper_bindings(),
        "environment_fingerprint": environment,
        "environment_sha256": _canonical_sha(environment),
        "source_blobs": source_blobs,
        "projection_root": str(projected),
        "projection_bindings": projection_bindings,
        "projection_sha256": projection_sha,
        "output_dir": str(output_dir),
        "max_train_steps": config.max_train_steps,
        "world_size": config.world_size,
        "inputs": inputs,
        "paths": {
            "dataset_root_path": str(projected / "datasets"),
            "dataset_meta_info_path": str(projected / "dataset_meta_info"),
            "manifest_path": str(
                projected / "dataset_meta_info" / config.dataset_name / "evoworld_manifest.json"
            ),
            "svd_model_path": str(config.svd_model_path),
            "clip_model_path": str(config.clip_model_path),
            "policy_checkpoint_path": str(config.policy_checkpoint_path),
            "action_adapter_path": str(config.action_adapter_path),
            "synthetic_output_root": str(config.synthetic_output_root),
            "starting_checkpoint": str(config.starting_checkpoint)
            if config.starting_checkpoint
            else None,
        },
        "names": {
            "dataset_name": config.dataset_name,
            "annotation_name": config.annotation_name,
            "tag": config.tag,
            "wandb_run_name": config.wandb_run_name,
            "wandb_project_name": config.wandb_project_name,
        },
    }
    request_path = job_root / "ctrl_world_request.json"
    _write_exact_json(request_path, request)
    request_sha = sha256_file(request_path)
    argv = (
        str(config.ctrl_world_python),
        str(_BRIDGE_PATH),
        "launch-train",
        "--request-path",
        str(request_path),
        "--request-sha256",
        request_sha,
    )
    try:
        return TrainingPlan(
            schema_version=1,
            job_id=job_id,
            target=target,
            manifest_sha256=manifest.sha256,
            upstream_revision=CTRL_WORLD_REVISION,
            cwd=source,
            argv=argv,
            environment={},
            expected_outputs=(
                Path("checkpoint-5000.pt"),
                Path("training_state.json"),
                Path("job_metadata.json"),
            ),
        )
    except TrainingSchemaError as error:
        raise CtrlWorldConfigError("unable to create a safe Ctrl-World plan") from error


def _materialize_projection(
    job_root: Path,
    config: CtrlWorldTrainConfig,
    manifest: TrainingManifest,
    annotations: Mapping[int, dict[str, object]],
) -> Path:
    projection = job_root / "input"
    dataset_target = projection / "datasets" / config.dataset_name
    metadata_target = projection / "dataset_meta_info" / config.dataset_name
    selected_ids = {int(sample.trajectory_id) for sample in manifest.samples}
    metadata_rows = _read_array(config.metadata_path / "train_sample.json")
    selected_rows = [
        row
        for row in metadata_rows
        if isinstance(row, dict) and row.get("episode_id") in selected_ids
    ]
    if {
        cast(int, row["episode_id"]) for row in cast(list[dict[str, object]], selected_rows)
    } != selected_ids:
        raise CtrlWorldConfigError("projected metadata does not cover every manifest episode")
    for episode_id in sorted(selected_ids):
        annotation = annotations[episode_id]
        annotation_source = config.dataset_path / "annotation/train" / f"{episode_id}.json"
        _copy_exact(annotation_source, dataset_target / "annotation/train" / f"{episode_id}.json")
        for field, key in (("videos", "video_path"), ("latent_videos", "latent_video_path")):
            for row in cast(list[dict[str, str]], annotation[field]):
                relative = Path(row[key])
                _copy_exact(config.dataset_path / relative, dataset_target / relative)
    _write_exact_json(metadata_target / "train_sample.json", selected_rows)
    _copy_exact(config.stats_path, metadata_target / "stat.json")
    _write_exact_json(metadata_target / "evoworld_manifest.json", manifest.to_dict())
    _write_exact_json(
        projection / "projection.json",
        {
            "manifest_sha256": manifest.sha256,
            "dataset_name": config.dataset_name,
            "episode_ids": sorted(selected_ids),
            "metadata_row_count": len(selected_rows),
        },
    )
    return projection


def load_ctrl_world_manifest(path: Path) -> TrainingManifest:
    try:
        payload = _read_object(path)
        rows = payload["samples"]
        if not isinstance(rows, list):
            raise TypeError("samples must be a list")
        return TrainingManifest(
            schema_version=cast(int, payload["schema_version"]),
            manifest_id=cast(str, payload["manifest_id"]),
            samples=tuple(_sample_from_dict(cast(dict[str, object], row)) for row in rows),
        )
    except (KeyError, TypeError, ValueError, TrainingSchemaError) as error:
        raise CtrlWorldConfigError(f"invalid manifest sidecar {path}") from error


def _validate_annotation(
    dataset: Path, split: str, annotation: dict[str, object]
) -> tuple[int, tuple[Path, ...], tuple[Path, ...]]:
    episode_id = annotation.get("episode_id")
    video_length = annotation.get("video_length")
    state_length = annotation.get("state_length")
    raw_length = annotation.get("raw_length")
    if (
        isinstance(episode_id, bool)
        or not isinstance(episode_id, int)
        or isinstance(video_length, bool)
        or not isinstance(video_length, int)
        or video_length < 1
        or state_length != video_length
        or isinstance(raw_length, bool)
        or not isinstance(raw_length, int)
        or raw_length < (video_length - 1) * 3 + 1
    ):
        raise CtrlWorldConfigError("annotation episode/video/raw counts are invalid")
    texts = annotation.get("texts")
    if (
        not isinstance(texts, list)
        or len(texts) != 1
        or not isinstance(texts[0], str)
        or not texts[0].strip()
    ):
        raise CtrlWorldConfigError("annotation must contain exactly one instruction")
    _finite_rows(annotation.get("states"), video_length, 7, "states")
    _finite_rows(
        annotation.get("observation.state.cartesian_position"), raw_length, 6, "cartesian positions"
    )
    _finite_rows(
        annotation.get("observation.state.joint_position"), raw_length, 7, "joint positions"
    )
    gripper = annotation.get("observation.state.gripper_position")
    if (
        not isinstance(gripper, list)
        or len(gripper) != raw_length
        or any(not _finite_number(value) for value in gripper)
    ):
        raise CtrlWorldConfigError("gripper positions must be finite scalars matching raw_length")
    videos = _camera_paths(dataset, split, episode_id, annotation, "videos", "video_path", ".mp4")
    latents = _camera_paths(
        dataset, split, episode_id, annotation, "latent_videos", "latent_video_path", ".pt"
    )
    return episode_id, videos, latents


def _camera_paths(
    dataset: Path,
    split: str,
    episode_id: int,
    annotation: Mapping[str, object],
    field: str,
    key: str,
    suffix: str,
) -> tuple[Path, ...]:
    rows = annotation.get(field)
    if not isinstance(rows, list) or len(rows) != 3:
        raise CtrlWorldConfigError("every episode requires exactly three camera artifacts")
    paths: list[Path] = []
    for camera, row in enumerate(rows):
        value = row.get(key) if isinstance(row, dict) else None
        expected = Path(field) / split / str(episode_id) / f"{camera}{suffix}"
        if value != expected.as_posix():
            raise CtrlWorldConfigError("camera paths must exactly match pinned extraction layout")
        paths.append(_contained_file(dataset, dataset / expected, "camera artifact"))
    return tuple(paths)


def _validate_metadata_rows(
    metadata: Path, split: str, annotations: Mapping[int, Mapping[str, object]]
) -> None:
    rows = _read_array(metadata / f"{split}_sample.json")
    expected = [
        {"episode_id": episode_id, "frame_ids": [frame]}
        for episode_id, annotation in sorted(annotations.items())
        for frame in range(max(1, cast(int, annotation["video_length"]) - 4))
    ]

    def key(row: object) -> str:
        return json.dumps(row, sort_keys=True, separators=(",", ":"))

    if sorted(rows, key=key) != sorted(expected, key=key):
        raise CtrlWorldConfigError("metadata rows/counts/types do not match pinned generation")


def _validate_media_inspection(
    result: Mapping[str, object],
    annotations_by_split: Mapping[str, Mapping[int, Mapping[str, object]]],
    dataset: Path,
) -> None:
    videos = result.get("videos")
    latents = result.get("latents")
    if not isinstance(videos, dict) or not isinstance(latents, dict):
        raise CtrlWorldConfigError("media inspection must report videos and latents")
    for split, split_annotations in annotations_by_split.items():
        for episode_id, annotation in split_annotations.items():
            length = annotation["video_length"]
            for camera in range(3):
                video = str(dataset / f"videos/{split}/{episode_id}/{camera}.mp4")
                latent = str(dataset / f"latent_videos/{split}/{episode_id}/{camera}.pt")
                video_info = videos.get(video)
                latent_info = latents.get(latent)
                if not isinstance(video_info, dict) or video_info != {
                    "frames": length,
                    "shape": [192, 320, 3],
                }:
                    raise CtrlWorldConfigError("decoded MP4 shape/count does not match annotation")
                if not isinstance(latent_info, dict):
                    raise CtrlWorldConfigError("latent inspection result is missing")
                if latent_info.get("shape") != [length, 4, 24, 40] or latent_info.get(
                    "dtype"
                ) not in {"torch.float16", "torch.float32", "torch.bfloat16"}:
                    raise CtrlWorldConfigError("latent tensor dtype/shape/count is invalid")


def _validate_manifest_bijection(
    config: CtrlWorldTrainConfig,
    manifest: TrainingManifest,
    train_annotations: Mapping[int, Mapping[str, object]],
) -> None:
    episode_ids: set[int] = set()
    for sample in manifest.samples:
        if sample.target != "ctrl_world" or sample.split != "train":
            raise CtrlWorldConfigError(
                "Ctrl-World manifest must contain only ctrl_world/train records"
            )
        try:
            episode_id = int(sample.trajectory_id)
        except ValueError as error:
            raise CtrlWorldConfigError(
                "trajectory_id must be the numeric annotation episode_id"
            ) from error
        if episode_id in episode_ids or episode_id not in train_annotations:
            raise CtrlWorldConfigError("manifest and annotation episodes must be one-to-one")
        episode_ids.add(episode_id)
        annotation = train_annotations[episode_id]
        expected_media = tuple(
            Path(cast(str, row[key]))
            for field, key in (("videos", "video_path"), ("latent_videos", "latent_video_path"))
            for row in cast(list[dict[str, object]], annotation[field])
        )
        if (
            sample.dataset_root != Path(config.dataset_name)
            or sample.action_path != Path(f"annotation/train/{episode_id}.json")
            or sample.instruction != cast(list[str], annotation["texts"])[0]
            or sample.observation_paths != expected_media
        ):
            raise CtrlWorldConfigError("manifest is not bijective with annotation/media identity")
        if (
            compute_ctrl_world_sample_sha256(config.dataset_root_path, sample)
            != sample.content_sha256
        ):
            raise CtrlWorldConfigError("manifest sample provenance hash is stale")


def _validate_stats(path: Path) -> None:
    values = _read_object(path)
    if set(values) != {"state_01", "state_99"}:
        raise CtrlWorldConfigError("stats require exact state_01/state_99 fields")
    low, high = values["state_01"], values["state_99"]
    if (
        not isinstance(low, list)
        or not isinstance(high, list)
        or len(low) != 7
        or len(high) != 7
        or any(not _finite_number(value) for value in (*low, *high))
        or any(cast(float, a) >= cast(float, b) for a, b in zip(low, high, strict=True))
    ):
        raise CtrlWorldConfigError("p01/p99 stats must be finite, 7-D, and strictly ordered")


def _validate_runtime_paths(config: CtrlWorldTrainConfig) -> None:
    required = (
        config.manifest_path,
        config.svd_model_path,
        config.clip_model_path,
        config.policy_checkpoint_path,
        config.action_adapter_path,
        config.ctrl_world_python,
        *((config.starting_checkpoint,) if config.starting_checkpoint else ()),
    )
    if any(not path.exists() for path in required):
        raise CtrlWorldConfigError("a required Ctrl-World input path is missing")
    protected = (
        config.dataset_root_path,
        config.dataset_meta_info_path,
        config.synthetic_output_root,
        *required,
    )
    if any(_paths_overlap(config.output_root, path) for path in protected):
        raise CtrlWorldConfigError("output root must not overlap any input/model/checkpoint")


def verify_request_bindings(request: Mapping[str, object]) -> None:
    """Re-hash every projected artifact and immutable input before backend imports."""
    projection_root = Path(cast(str, request["projection_root"]))
    bindings = request["projection_bindings"]
    if not isinstance(bindings, list) or _canonical_sha(bindings) != request["projection_sha256"]:
        raise CtrlWorldConfigError("projection binding digest changed")
    if _artifact_bindings(projection_root) != bindings:
        raise CtrlWorldConfigError("projected dataset changed after planning")
    source = Path(cast(str, request["ctrl_world_root"]))
    source_blobs = request["source_blobs"]
    if (
        not isinstance(source_blobs, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in source_blobs.items()
        )
        or any(
            sha256_file(source / cast(str, relative)) != digest
            for relative, digest in source_blobs.items()
        )
    ):
        raise CtrlWorldConfigError("pinned source blob changed after planning")
    environment = fingerprint_ctrl_world_environment(Path(cast(str, request["ctrl_world_python"])))
    if (
        environment != request["environment_fingerprint"]
        or _canonical_sha(environment) != request["environment_sha256"]
    ):
        raise CtrlWorldConfigError("Ctrl-World environment changed after planning")
    paths = cast(dict[str, object], request["paths"])
    inputs = cast(dict[str, object], request["inputs"])
    checks = {
        "manifest": Path(cast(str, paths["manifest_path"])),
        "checkpoint": Path(cast(str, paths["starting_checkpoint"]))
        if paths["starting_checkpoint"]
        else None,
        "svd_model": Path(cast(str, paths["svd_model_path"])),
        "clip_model": Path(cast(str, paths["clip_model_path"])),
        "policy_checkpoint": Path(cast(str, paths["policy_checkpoint_path"])),
        "action_adapter": Path(cast(str, paths["action_adapter_path"])),
    }
    for name, path in checks.items():
        expected = inputs[name]
        actual = sha256_path(path) if path is not None else None
        if actual != expected:
            raise CtrlWorldConfigError(f"bound input changed after planning: {name}")


def verify_preparation_request_bindings(request: Mapping[str, object]) -> None:
    """Re-hash the backend, pinned source, and preparation inputs."""
    source = Path(cast(str, request["ctrl_world_root"]))
    source_blobs = request["source_blobs"]
    if (
        not isinstance(source_blobs, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in source_blobs.items()
        )
        or any(
            sha256_file(source / cast(str, relative)) != digest
            for relative, digest in source_blobs.items()
        )
    ):
        raise CtrlWorldConfigError("pinned preparation source changed after planning")
    environment = fingerprint_ctrl_world_environment(Path(cast(str, request["ctrl_world_python"])))
    if (
        environment != request["environment_fingerprint"]
        or _canonical_sha(environment) != request["environment_sha256"]
    ):
        raise CtrlWorldConfigError("Ctrl-World environment changed after planning")
    inputs = request["inputs"]
    if not isinstance(inputs, dict):
        raise CtrlWorldConfigError("preparation input bindings are invalid")
    for name in ("droid_hf_path", "svd_model_path"):
        binding = inputs.get(name)
        if (
            not isinstance(binding, dict)
            or not isinstance(binding.get("path"), str)
            or not isinstance(binding.get("sha256"), str)
            or sha256_path(Path(cast(str, binding["path"]))) != binding["sha256"]
        ):
            raise CtrlWorldConfigError(f"bound preparation input changed: {name}")


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file_binding(path: Path) -> dict[str, object]:
    if path.resolve(strict=True) != path or path.is_symlink():
        raise CtrlWorldConfigError(f"official executable path is not canonical: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        value = os.fstat(descriptor)
        if not stat.S_ISREG(value.st_mode):
            raise CtrlWorldConfigError(f"official executable is not regular: {path}")
        digest = sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        return {
            "path": str(path),
            "st_dev": value.st_dev,
            "st_ino": value.st_ino,
            "size": value.st_size,
            "sha256": digest.hexdigest(),
        }
    finally:
        os.close(descriptor)


def _local_helper_bindings() -> list[dict[str, str]]:
    relatives = (
        *_LOCAL_HELPER_RELATIVES,
        *(
            relative
            for relative in _OPTIONAL_LOCAL_HELPER_RELATIVES
            if (_REPOSITORY_ROOT / relative).is_file()
        ),
    )
    return [
        {
            "path": str((_REPOSITORY_ROOT / relative).resolve()),
            "sha256": sha256_file(_REPOSITORY_ROOT / relative),
        }
        for relative in relatives
    ]


def sha256_path(path: Path) -> str:
    if path.is_symlink():
        raise CtrlWorldConfigError(f"bound path must not be a symlink: {path}")
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise CtrlWorldConfigError(f"bound path does not exist: {path}")
    return _canonical_sha(_artifact_bindings(path))


def artifact_inventory(root: Path) -> list[dict[str, str]]:
    """Return the canonical path/content inventory for a completed artifact tree."""
    if not root.is_dir():
        raise CtrlWorldConfigError(f"artifact tree does not exist: {root}")
    return _artifact_bindings(root)


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible value with the repository canonical encoding."""
    return _canonical_sha(value)


def _artifact_bindings(root: Path) -> list[dict[str, str]]:
    if root.is_symlink() or any(item.is_symlink() for item in root.rglob("*")):
        raise CtrlWorldConfigError(f"artifact tree must not contain symlinks: {root}")
    return [
        {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


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


def _finite_rows(value: object, length: int, width: int, name: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(
            not isinstance(row, list)
            or len(row) != width
            or any(not _finite_number(item) for item in row)
            for row in value
        )
    ):
        raise CtrlWorldConfigError(f"{name} must be finite {width}-D rows matching length")


def _read_object(path: Path) -> dict[str, object]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise CtrlWorldConfigError(f"JSON artifact must be an object: {path}")
    return cast(dict[str, object], value)


def _read_array(path: Path) -> list[object]:
    value = _read_json(path)
    if not isinstance(value, list):
        raise CtrlWorldConfigError(f"JSON artifact must be an array: {path}")
    return cast(list[object], value)


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CtrlWorldConfigError(f"invalid JSON artifact {path}") from error


def _write_exact_json(path: Path, value: object) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise CtrlWorldConfigError(f"existing job artifact conflicts with this plan: {path}")
    path.write_text(encoded, encoding="utf-8")


def _copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise CtrlWorldConfigError(f"existing projection conflicts with source: {destination}")
        return
    shutil.copy2(source, destination)


def _canonical_sha(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _contained_file(root: Path, path: Path, name: str) -> Path:
    if path.is_symlink():
        raise CtrlWorldConfigError(f"{name} must not be a symlink")
    resolved = _require_contained(root, path, name)
    if not resolved.is_file():
        raise CtrlWorldConfigError(f"{name} is missing: {resolved}")
    return resolved


def _require_contained(root: Path, path: Path, name: str) -> Path:
    resolved_root, resolved = root.resolve(), path.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise CtrlWorldConfigError(f"{name} must be contained in {root}")
    return resolved


def _paths_overlap(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left in right.parents or right in left.parents


def _require_absolute(paths: tuple[Path, ...]) -> None:
    if not all(path.is_absolute() for path in paths):
        raise CtrlWorldConfigError("all Ctrl-World paths must be absolute")


def _safe_name(value: str, name: str) -> None:
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or any(char.isspace() for char in value)
    ):
        raise CtrlWorldConfigError(f"{name} must be a safe non-empty name")


def _finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)

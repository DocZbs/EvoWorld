# Real runtime composition

Real runs use an importable factory instead of an in-process registry:

```bash
export EVOWORLD_RUNTIME_MANIFEST=/absolute/path/runtime.json
python -m evoworld run \
  --config /absolute/path/real.toml \
  --factory evoworld.real_composition:build_real_adapters
```

The exact factory identity and canonical runtime digest are stored in `manifest.json`.
`resume` loads the factory in a fresh process, rebuilds the digest from the strict
runtime manifest and local asset bytes, and rejects any drift before a backend is
called. An explicit, different `--factory` is also rejected. `validate` only parses
the TOML configuration and never imports a factory or touches a backend.

`evoworld.real_composition:build_real_adapters` constructs the pinned OpenPi,
GR00T, Ctrl-World, and Qwen3-VL process adapters. It never substitutes a
deterministic adapter. Deployment-private task sourcing, scene anchoring, routing,
rollout scoring, robot execution, training, memory, and durable storage are supplied
by the required `pipeline_provider`. Missing assets, a missing provider or private
executor, an invalid backend checkout, or an incomplete provider fails closed before
a pipeline round starts.

## Runtime manifest

The manifest is strict JSON: it must be an absolute, non-symlink regular file, every
path must be absolute, and unknown or missing fields are rejected. Schema 2 has this
shape:

```json
{
  "schema_version": 2,
  "project_root": "/opt/evoworld",
  "media_root": "/srv/evoworld/media",
  "asset_root": "/srv/evoworld/assets",
  "pipeline_provider": "robot_cell.runtime:build_ports",
  "action_projectors": {
    "pi05": {
      "identity": "pi05-cartesian-v1",
      "implementation": "affine_v1",
      "source_schema": "joint_velocity",
      "source_width": 8,
      "target_schema": "cartesian_pose_7d",
      "config_path": "/srv/evoworld/assets/projectors/pi05/config.json",
      "config_digest": "4444444444444444444444444444444444444444444444444444444444444444",
      "source_digest": "5555555555555555555555555555555555555555555555555555555555555555",
      "weights_path": "/srv/evoworld/assets/projectors/pi05/affine-weights.json",
      "weights_digest": "6666666666666666666666666666666666666666666666666666666666666666",
      "content_digest": "7777777777777777777777777777777777777777777777777777777777777777"
    },
    "groot": {
      "identity": "groot-cartesian-v1",
      "implementation": "affine_v1",
      "source_schema": "groot:action.arm+action.gripper",
      "source_width": 8,
      "target_schema": "cartesian_pose_7d",
      "config_path": "/srv/evoworld/assets/projectors/groot/config.json",
      "config_digest": "8888888888888888888888888888888888888888888888888888888888888888",
      "source_digest": "5555555555555555555555555555555555555555555555555555555555555555",
      "weights_path": "/srv/evoworld/assets/projectors/groot/affine-weights.json",
      "weights_digest": "9999999999999999999999999999999999999999999999999999999999999999",
      "content_digest": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    }
  },
  "openpi": {
    "expert_id": "pi05",
    "python": "/opt/envs/openpi/bin/python",
    "model": "pi05",
    "host": "127.0.0.1",
    "port": 8000,
    "timeout_seconds": 5.0,
    "checkpoint_identity": "/srv/evoworld/checkpoints/pi05_droid",
    "checkpoint_digest": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  },
  "groot": {
    "expert_id": "groot",
    "python": "/opt/envs/groot/bin/python",
    "host": "127.0.0.1",
    "port": 5555,
    "timeout_seconds": 20.0,
    "checkpoint_identity": "/srv/evoworld/checkpoints/groot_n16_droid",
    "checkpoint_digest": "123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0"
  },
  "ctrl_world": {
    "python": "/opt/envs/ctrl-world/bin/python",
    "model_root": "/srv/evoworld/assets/svd",
    "text_encoder_root": "/srv/evoworld/assets/clip",
    "dataset_root": "/srv/evoworld/media/droid",
    "checkpoint_path": "/srv/evoworld/assets/ctrl-world/model.ckpt",
    "stats_path": "/srv/evoworld/assets/ctrl-world/stats.json",
    "timeout_seconds": 300.0,
    "fps": 4.0
  },
  "qwen_vl": {
    "python": "/opt/envs/qwen/bin/python",
    "model_id": "Qwen/Qwen3-VL-4B-Instruct",
    "model_digest": "23456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef01",
    "model_revision": "3456789abcdef0123456789abcdef0123456789a",
    "adapter_checkpoint": "/srv/evoworld/assets/qwen/evaluator-lora",
    "checkpoint_root": "/srv/evoworld/assets/qwen",
    "device": "cuda",
    "dtype": "bfloat16",
    "max_new_tokens": 256,
    "timeout_seconds": 300.0
  }
}
```

EvoWorld does not ship deployment-specific projection assets. Both entries are
mandatory: a checkout without the private affine configuration and calibration
weights fails before any real backend starts. There is no projector factory or
deployment-owned source module. `affine_v1` is the fixed, audited implementation in
`src/evoworld/adapters/projection.py`, and `source_digest` binds those repository
bytes. Each config is strict JSON
`{"schema_version": 1, "implementation": "affine_v1", "source_width": N,
"target_width": 7}`. Each weights file is strict JSON
`{"schema_version": 1, "matrix": <7 x N finite numbers>, "bias": <7 finite
numbers>}`. Both files must be regular, symlink-free children of `asset_root`; they
are hashed and parsed from the same no-follow file-descriptor snapshot into immutable
values. GR00T must declare its exact raw action keys; wildcard schemas are rejected.
Its `source_width` must equal the deployed service's actual final action dimension;
the manifest, config, and matrix must agree. The pinned openpi response mapper emits
width eight. `content_digest` binds the identity, implementation, schemas, widths,
and config/source/weights digests. The complete runtime-manifest digest separately
binds the configured manifest asset paths.

Every repeated-character digest in the example is illustrative. Replace projector
asset digests with the hashes of the deployed bytes, `source_digest` with the hash of
the checked-out library implementation, and `content_digest` with the canonical
combined identity from `projector_content_digest(...)`. Replace each policy
checkpoint identity with the wrapper's resolved path and its digest with the exact
combined manifest SHA-256 reported by that service.

Set both Qwen checkpoint fields to `null` to use the base evaluator. A fine-tuned
adapter is a directory contained by `checkpoint_root` with `adapter_config.json`;
the isolated bridge loads it with PEFT over the configured Qwen3-VL base model.
`model_revision` must be the full 40-character Hugging Face Git commit and is passed
to both processor and model loading; `model_digest` records the separately verified
model artifact identity used by the deployment.

The provider callable receives `(config, run_dir, backend_metadata)`. The third
argument is immutable `RuntimeBackendMetadata`: pure runtime/projector digests and
expert-identity summaries only. It exposes no policy, projector, world-model, or
evaluator object. The provider must return exactly a
`CoreRuntimePorts` value containing only task sourcing, scene anchoring, routing,
rollout scoring, robot execution, training, memory, and durable storage. The public
composition injects the four audited backends itself, so the provider cannot replace
them. Its store must be bound to the exact requested run directory, and its durable
manifest must match the requested real configuration after canonical JSON
normalization.

Each library-owned wrapper calls the official policy under a derived `:raw`
operation ID and the fixed projector under `:project`. Only a finite, provenance-matched,
timestamp-preserving `cartesian_pose_7d` chunk reaches Ctrl-World or physical
execution. Config, public implementation source, weights, and canonical paths are
included in the runtime digest. Assets are loaded once from verified snapshots into
immutable tuples; resume revalidates the manifest before composition. There is no
dynamic projector import or deterministic projection fallback.

## Durable adapter operations

Each process-backed operation writes a canonical intent before invoking its backend
and a validated result receipt afterward. Files live under digest-only directories
inside that adapter's artifact directory. A completed retry returns the verified
receipt without another backend call. Input media, local model assets, checkpoints,
and selected dataset records are bound by byte manifests. OpenPi and GR00T also
verify the configured checkpoint identity and digest through independent server
metadata or an identity endpoint before inference.
Ctrl-World receipts bind every output frame and video and re-hash them on replay.
Reusing the ID after any bound bytes have changed raises an operation conflict;
finding an intent without a receipt raises an in-doubt error. Cross-process file
locks and atomic, fsynced writes serialize claims. Process logs are separately
isolated under a digest of operation kind and request ID.

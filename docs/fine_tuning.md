# Fine-tuning

All EvoWorld fine-tuning is plan-first. A plan binds a canonical dataset manifest,
the exact upstream commit, absolute working directory, argv array, allowlisted
environment, and expected outputs in an immutable `job.json`. Planning performs no
training. Execution revalidates those bindings and refuses stale or already completed
jobs.

```bash
cp configs/finetune/pi05.example.toml /absolute/path/pi05.toml
# Edit every /replace/with/... path in /absolute/path/pi05.toml.
uv run evoworld finetune plan --config /absolute/path/pi05.toml --target pi05
uv run evoworld finetune execute --job <job-from-plan-json>
```

The valid targets are `pi05`, `pi0`, `pi0_fast`, `groot_n16`, `ctrl_world`, and
`qwen_evaluator`. A fine-tuning config must contain exactly one `[finetune]` table;
[`configs/finetune/pi05.example.toml`](../configs/finetune/pi05.example.toml) is the
openpi pi0.5 field-complete template. Copy it and provide real manifest, dataset,
output, backend, recipe, and job paths. `configs/reproduction.toml` only pins the
three upstream revisions and is not a fine-tuning config. Planning prints the
absolute job path; by contract it is
`<configured-job_root>/<job-id>/job.json`, not a fixed `runs/...` location.

## Shared data and job rules

Use the canonical manifests in [data.md](data.md). All expert and world-model
adaptations are bounded to exactly 5,000 optimizer steps. Plans never edit a
submodule. Bridges compose pinned upstream APIs only where the official CLI cannot
express a method-required setting, and the resulting override remains visible in
`job.json`.

Run GPU commands in each upstream project's isolated Linux CUDA environment. The
core uv environment is only for validation, planning, and tests.

## openpi experts

Source: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi),
revision `15a9616a00943ada6c20a0f158e3adb39df2ccac`. Settings are declared in
`configs/experts/openpi.toml`.

| Target | Upstream base config | Action dimension | Horizon | Batch | Max tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| `pi05` | `pi05_droid_finetune` | 32 | 15 | 16 | upstream default |
| `pi0` | `pi0_libero` | upstream default | 10 | 8 | upstream default |
| `pi0_fast` | `pi0_fast_libero` | 8 | 10 | 8 | 180 |

Every target uses joint-velocity supervision, exactly 5,000 steps, eight loader
workers, `shuffle=false`, and LeRobot v2.1 data in `data/chunk-000`. The checked-in
bridge exposes an independently runnable `normalize` operation using the same
validated inputs as the planned training argv:

```bash
uv run --project /absolute/path/EvoWorld/third_party/openpi \
  python /absolute/path/EvoWorld/integrations/openpi/train_bridge.py normalize \
  --openpi-root /absolute/path/EvoWorld/third_party/openpi \
  --config-path /absolute/path/EvoWorld/configs/experts/openpi.toml \
  --expert pi05 \
  --manifest-path /absolute/path/data/training-manifest.json \
  --manifest-sha256 <manifest-sha256> \
  --dataset-root /absolute/path/data/lerobot-v2.1 \
  --output-dir /absolute/path/output/<job-id> \
  --job-id <job-id>
```

The bridge calls the pinned official normalization implementation internally. The
pinned `scripts/train.py` forces shuffle, so method-aligned training also uses this
bridge around the official JAX configuration, loader, optimizer, schedule, loss,
checkpoint, and train-state components. It validates the pinned checkout and public
upstream config before launch.

## GR00T-N1.6

Source: [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), revision
`ead52833afbbf4243f8cd5e7664f48a94de03b19` (`n1.6-release`). The strict settings in
`configs/experts/groot_n16.toml` are:

- base model `nvidia/GR00T-N1.6-3B`;
- `NEW_EMBODIMENT` with `meta/evoworld_modality.py`;
- eight GPUs and global batch size 64;
- exactly 5,000 steps, save every 1,000, keep five checkpoints;
- two data-loader workers and a 16-step action horizon.

The plan invokes the official source entry point from the pinned checkout:

```bash
uv run python gr00t/experiment/launch_finetune.py --base-model-path nvidia/GR00T-N1.6-3B --num-gpus 8 --global-batch-size 64 --max-steps 5000 ...
```

Open-loop evaluation is a separate plan using
`gr00t/eval/open_loop_eval.py`. It binds exactly `checkpoint-5000` and one trajectory
ID per output, so evaluation plots cannot overwrite each other.

## Ctrl-World

Source: [Robert-gyj/Ctrl-World](https://github.com/Robert-gyj/Ctrl-World), revision
`99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d`. The settings live in
`configs/world_model/ctrl_world.toml`: 5,000 synchronized optimizer updates,
`annotation` metadata, and an explicit world size (the public baseline is one).

Preparation has two receipt-bound operations: `prepare-latents` runs the pinned
latent extractor into an exclusive destination, then `bind-metadata` consumes that
verified tree and publishes metadata atomically. Both enter the checked-in bootstrap
under the absolute Ctrl-World Python; the bootstrap verifies the clean recursive
checkout, official scripts, inputs, outputs, and relevant Python distributions before
calling upstream code.

Base training accepts real `NONE/train` records and no starting checkpoint. WMF
post-training accepts only `WMF/train` records and requires a starting checkpoint.
There is no invented upstream WMF executable. The bridge composes the pinned
`wm_args`, `CrtlWorld`, `Dataset_mix`, loss, AdamW, Accelerate, and save components,
counts only synchronized optimizer updates, and publishes a strictly reloaded
`checkpoint-5000.pt` without overwriting an existing output.

Ctrl-World optimizer, precision, accumulation, batching, loss, and periodic-save
defaults remain upstream values because no different public exact values are
available. The available description also does not establish an eight-GPU
Ctrl-World recipe, so this repository does not claim one.

## Qwen3-VL evaluator baseline

The public base is
[`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).
`configs/evaluator/qwen3_vl_4b.toml` defines a public LoRA baseline executed with
`ms-swift==4.3.1`, `qwen-vl-utils==0.0.14`, `decord==0.6.0`, Transformers
`>=4.57,<5.13.0`, Torch `>=2.0`, and torchvision `>=0.15` on CUDA.

The complete public settings are base model `Qwen/Qwen3-VL-4B-Instruct`, LoRA rank
16 and alpha 32, learning rate `1e-4`, two epochs, per-device batch one, gradient
accumulation eight, maximum length 8192, seed 42, bfloat16, epoch saves, and at most
two retained checkpoints. Image/video pixel, token, FPS, and frame budgets are also
explicit in the config. Training is launched through exact `uvx` package pins and
the official `swift sft` command, with `--add_version false` so the expected output
path stays deterministic.

The launcher first probes the isolated runtime and CUDA availability, validates both
exported splits and every frame, then accepts success only when ms-swift produced
saved arguments, logs, an epoch-final checkpoint, and LoRA adapter artifacts. The
exact evaluator examples, trained checkpoint, prompt, and hyperparameters used for
the reported experiments are private. These values are a configurable public
baseline and must not be described as the exact private recipe.

## What must be supplied

Execution requires legal access to DROID-derived data, prepared manifests, model
weights, base and starting checkpoints, sufficient Linux CUDA hardware, and writable
output roots. Real closed-loop collection additionally requires the robot,
calibration, controllers, safety interlocks, and deployment-specific executor. None
of those private assets are downloaded or emulated by `finetune execute`.

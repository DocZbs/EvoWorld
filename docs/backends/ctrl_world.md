# Ctrl-World rollout backend

EvoWorld targets Ctrl-World commit
`99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d`. The checkout remains an
unmodified Git submodule at `third_party/ctrl_world`; the repository inventory
test verifies that gitlink. Before a rollout command is built, the I2 recursive
revision gate checks every registered submodule, requires the Ctrl-World HEAD to
equal this commit, and rejects modified or uninitialized top-level and nested
checkouts. The verified revision is passed to the isolated backend and returned
unchanged in every successful rollout's provenance.

## Required local assets

The adapter requires explicit absolute paths for all heavyweight inputs:

- the pinned Ctrl-World checkout and its exact `config.py`;
- a Stable Video Diffusion model directory;
- a CLIP text-encoder directory;
- a Ctrl-World checkpoint file;
- the DROID normalization statistics JSON;
- a DROID episode root;
- one conditioning video belonging to the selected episode; and
- an existing output directory contained by the configured artifact root.

Model, checkpoint, and statistics paths must remain under one configured asset
root. The DROID root and conditioning video must remain under one configured
media root. Both the core adapter and backend resolve paths strictly, so missing
files and symlinks that escape either root are rejected. Paths copied from the
upstream `/cephfs` examples are rejected rather than treated as defaults.

The selected DROID episode must have
`annotation/val/<trajectory_id>.json`. It must contain `texts`, `states`,
`joints`, and a non-empty `videos` list, plus either a non-empty `action` array
or a positive `video_length`. The declared length must agree with the exact
number of 7-D finite Cartesian states and 8-D finite joint states. The selected
start index plus requested action steps must fit without upstream end-frame
clamping. Each video entry contains exactly one relative `video_path`; the
scene's first observation path must resolve to one of those files. The first
annotation instruction must match the task instruction. All of these checks run
before the GPU model is constructed. `ctrl_world_trajectory_id` and a non-negative
`ctrl_world_start_index` are supplied in `SceneRecord.context`.

## Exact upstream seam

`build_ctrl_world_process_bridge` launches an absolute bridge script with an
absolute backend Python executable, uses the pinned checkout as its fixed
working directory, and supplies only `PYTHONIOENCODING` and `PYTHONPATH`. The
bridge is short lived and communicates through the version-one JSONL protocol;
Ctrl-World, Torch, and NumPy are not imported by the core adapter.

Inside the backend environment, the bridge loads the exact configured
`config.py` and `scripts/rollout_replay_traj.py` files lazily. It then follows
the pinned replay implementation:

1. construct `wm_args(task_type="replay")`;
2. set explicit `svd_model_path`, `clip_model_path`, `ckpt_path`,
   `val_model_path`, `val_dataset_dir`, `data_stat_path`, and `save_dir` values;
3. clear the unused replay-time `pi_ckpt` and `action_adapter` defaults, and
   reject the configuration if any `/cephfs` path remains;
4. construct the official `agent(args)`;
5. call `agent.get_traj_info(trajectory_id, start_idx=..., steps=...)` to use
   the official DROID decoding and VAE path; and
6. call the official
   `agent.forward_wm(action_cond, video_latent_true, current_latent,
   his_cond=..., text=...)` signature.

The pinned method returns `(videos_cat, true_video, videos, latents)`. EvoWorld
uses element 2, `videos`, which is the predicted-only `(V,F,H,W,3)` uint8 tensor;
it never publishes element 0's diagnostic prediction/ground-truth composite or
element 1's ground truth. Multiple predicted camera views are joined left to
right into `(F,H,V*W,3)` frames without dropping a view.

No upstream source is patched, and this bridge does not invent a generalized
Ctrl-World API. If that pinned class, method, or return signature changes, the
bridge fails explicitly.

## Action and artifact boundary

Ctrl-World conditioning is a non-empty, timestamped sequence of finite 7-D
Cartesian poses with schema `cartesian_pose_7d`. The expert identifier in the
`ExpertSpec`, `ActionChunk`, request, and response must agree. Policy-native
joint or joint-velocity actions are not sliced or silently truncated. A caller
must apply and identify a separate, explicit kinematic or learned projection
before invoking this adapter. This is especially important for openpi service
outputs, whose public DROID action width is eight.

The backend writes sequential `frame_000000.png` artifacts and `rollout.mp4`
into a temporary directory, then atomically renames that directory into the
request output directory. The core validates response fields exactly, checks
request/task/scene/expert/action/trajectory/start-index identity, and requires
one output frame per requested action step. Each artifact must be a regular file
inside one direct, real rollout directory under the configured output root.
Every relative path component is checked with `lstat`; file and directory
symlinks are rejected before resolution. The core fully decodes every PNG and
every MP4 frame, then requires the video frame count, dimensions, and FPS to
match the PNG sequence and metrics. Pillow and `imageio-ffmpeg` are runtime
dependencies, so validation does not depend on a system FFmpeg installation.
Backend, verified revision, trajectory, and start index are retained as
provenance. Process diagnostics go to stderr and retained logs, never into
protocol stdout.

The public integration describes only the behavior exposed by the pinned
source and explicit local assets. It does not claim access to private data,
checkpoints, annotations, or undisclosed training settings.
## Training and preparation

EvoWorld uses the public Ctrl-World repository at revision
`99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d`. The upstream checkout is read-only:
the adapter validates the revision, builds explicit plans, and imports pinned
training components only inside the Ctrl-World environment.

## Public preparation and base training

Preparation preserves the two pinned public entry points and their argument names,
but every plan enters the checked-in bootstrap first:

```text
<ctrl-world-python> <absolute-bridge> prepare-latents \
  --request-path <absolute-request> --request-sha256 <sha256>

<ctrl-world-python> <absolute-bridge> bind-metadata \
  --request-path <absolute-request> --request-sha256 <sha256>
```

Only after request, I2 checkout, source, environment, and input/output bindings pass
does the bootstrap invoke the absolute pinned script. Latent extraction exclusively
claims an absent destination, validates the completed DROID tree, and atomically
publishes an inventory-hashed completion receipt. `bind-metadata` is a second phase:
it refuses to run before that receipt exists, re-hashes the prepared tree, writes a
new metadata request binding the receipt and exact inventory, exclusively claims an
absent metadata destination, creates a unique same-filesystem job staging directory,
and invokes the metadata script with that directory as its working directory. The
official script therefore cannot write the shared final path. Stale, partial,
claimed, or unexpected outputs are never reused or overwritten.
After the official command, the bridge re-hashes the entire prepared tree and receipt
bytes against the bound request, validates the staging inventory, and atomically
publishes the directory with an operating-system no-replace rename. Only then can it
publish a metadata completion receipt. Failure moves only the job staging directory
and bound request into a unique same-workspace quarantine with an invalid marker;
external output that appears at any time is rejected and left byte-for-byte intact.
The preparation request also binds the canonical workspace and `.evoworld` request
root by device and inode. Before any mutable checkout callback or official preflight,
the bridge opens each root component from `/` with dirfd-relative `O_NOFOLLOW` and
keeps the verified descriptors alive through execution. Staging, final-parent, claim,
receipt, completion, and quarantine traversal and creation are relative to those
descriptors. Quarantine uses the same fd-relative no-replace rename as publication,
so moving or replacing a lexical workspace or quarantine path cannot redirect output.
The official metadata process is launched through the checked-in stdlib-only bridge
executor. The parent passes the verified workspace and job-directory descriptors via
`pass_fds`; the child rechecks their device/inode identity, calls `fchdir` on the job
descriptor, and delegates only to the fd-backed backend/script executor with a
relative prepared-data path. No F_GETPATH-derived cwd or input path crosses the official
execution boundary. Inherited descriptors are explicitly marked for the final exec,
and every parent/error path closes its local descriptors.
The plan also binds the backend launcher and official metadata script by canonical
path, device, inode, size, and streamed SHA-256. Both regular files are opened with
`O_NOFOLLOW` before revision callbacks and their descriptors remain live through the
child. On Linux the child re-hashes both descriptors and uses fd-backed `execve` for
the backend with `/proc/self/fd/<script-fd>` as the script argument. Platforms without
reliable executable-fd support, including current macOS Python, fail closed instead
of reopening either pathname. After the child returns, the parent again walks the
canonical workspace and request-root paths from `/` and compares their bound
identities before any post-audit, final publication, or completion receipt.

Before base training, EvoWorld requires non-empty real `annotation/train` and
`annotation/val` splits, exactly three resized videos and three latent tensors per
episode, typed metadata rows covering every episode, and finite ordered 1st/99th
percentile state statistics. Every media path must remain inside its dataset and
match the immutable manifest hash.

The statistics file uses the names present in the pinned public source,
`state_01` and `state_99`. The metadata script's percentile calculation is commented
out upstream, so callers must generate and audit `stat.json` before planning
training.

Base training is a distinct public-data policy: its manifest must contain only
real-source `NONE`/`train` records and it must not have a starting checkpoint. WMF
requires only `WMF`/`train` records and an explicit starting checkpoint.

## Bounded WMF updates

`WMF` does not select a separate or invented executable. The pinned
`scripts/train_wm.py` loop is audited, but its `main(args)` is not called: its epoch
loop and 20,000-step checkpoint default cannot guarantee a final artifact at the
required 5,000-step boundary. The bridge instead composes the same official
`config.wm_args`, `CrtlWorld`, `Dataset_mix`, model loss, Torch `AdamW`, Accelerate
accumulation, clipping, and state-dict save components in a bridge-owned loop.

The optimizer-step counter advances only when `Accelerator.sync_gradients` is true,
so gradient accumulation and process count cannot multiply or shorten the budget.
The loop stops at exactly 5,000 synchronized optimizer updates and always writes
`checkpoint-5000.pt`, `training_state.json`, and `job_metadata.json`, even though the
unchanged periodic checkpoint interval is larger than the run.

World size is explicit in configuration. A plan starts `<absolute-bridge>
launch-train`; after verification that bootstrap invokes the bound Python's
`accelerate.commands.launch --num_processes` with `<absolute-bridge> train-worker`.
The worker repeats the complete verification before importing official training
components and also requires `Accelerator.num_processes` to match. The checked-in
public baseline uses one process.
The available appendix does not establish an eight-GPU Ctrl-World recipe, so this
backend does not claim one.

Before launch, the planner materializes an independent job input directory. Its
`train_sample.json` contains only episodes selected one-to-one by the immutable WMF
manifest. Annotation identity, instruction, trajectory ID, all three videos, all
three latent tensors, and content hashes must agree; unrelated metadata episodes are
not visible to official `Dataset_mix`. The projection sidecar, every projected file,
model/checkpoint inputs, backend Python executable, and every imported pinned source
blob are SHA-bound in one request. The child process rechecks those hashes plus the
I2 clean recursive submodule audit before importing GPU libraries.

The request also fingerprints the resolved Python version and executable. For every
relevant distribution it records the normalized distribution name, version, raw
base64-encoded RECORD bytes and identity, and a sorted relative-path/size/SHA-256
inventory recomputed with streaming reads (plus Python and native-library files
without recorded hashes). Duplicate, absolute, malformed, or escaping RECORD entries
fail closed; explicit per-file, per-distribution, total-file, and total-byte bounds
limit the scan. Local editable installs are accepted only when a bound local
`direct_url.json` identifies a canonical root already present on the interpreter's
runtime path. The direct-URL file is lstat-checked, streamed, and counted even when
its RECORD hash field is empty, with an exact 1 MiB maximum accepted and the next byte
rejected before opening the file. One bounded byte buffer supplies both hashing and
JSON decoding; its relative identity, raw digest, size, and resolved root are bound.
Explicit module bindings include
`accelerate.commands.launch`, `accelerate.accelerator`, `imageio.v3`, and Torch's
optimizer, serialization, `nn`, and data-loader modules. Verification deliberately
re-reads these files without a cache, so cost is linear in the installed distribution
bytes and environment drift cannot be hidden behind stale process state. Preparation
preflights both official scripts only after this verification.

Before the bridge imports any mutable repository helper, its stdlib-only first hop
checks the request hash, its own absolute path/hash, and the exact path/hash inventory
of every local module needed by verification. Only that verified first hop may lazily
import dataset/schema helpers. The same rule applies to preparation, launch, and the
distributed worker.

The maximum optimization budget is exactly 5,000 steps. Optimizer, learning-rate,
precision, batching, loss, and periodic-save defaults remain those of the pinned
public implementation because the appendix does not fix different values. The
mandatory final save makes the fixed budget auditable. The bridge rejects surviving
`/cephfs` defaults, a changed or dirty source revision, stale source/input/projection
hashes, output/input overlap, a missing checkpoint, or evaluation and non-WMF
leakage.

Final state is first written to a same-directory temporary checkpoint, reloaded with
the official Torch loader, and applied to the unwrapped model with `strict=True`.
Only a successful round trip is published with an exclusive no-overwrite link;
checkpoint, state, and metadata publication refuse existing or racing paths.

The repository supplies the public preparation, validation, and launch contract.
It does not include private annotations, private checkpoints, or unpublished
hyperparameters, and does not claim those unavailable artifacts are reproduced.

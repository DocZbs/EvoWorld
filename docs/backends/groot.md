# GR00T-N1.6 backend

EvoWorld targets NVIDIA Isaac GR00T commit
`ead52833afbbf4243f8cd5e7664f48a94de03b19` (`n1.6-release`). The upstream
checkout is an unmodified submodule at `third_party/isaac_groot`.

Start the service with the argv returned by
`build_groot_server_command`. Its required top-level `project_root` binds the repo
to the exact audited gitlink. Fake, symlinked, dirty, uninitialized, and
wrong-revision recursive trees are rejected, and the process bridge repeats the
audit immediately before each child launch. EvoWorld deliberately binds it to `127.0.0.1`;
the argv uses absolute `--project` and checked-in wrapper paths, so it stays bound to
the validated submodule checkout regardless of the caller's current directory.
Manage this long-running GPU service separately from the short-lived JSONL
bridge, and stop it explicitly when the run ends. The bridge does not launch or
terminate the service: it verifies the existing service with official
`PolicyClient.ping`, queries the wrapper's independent `evoworld_identity` endpoint,
queries `get_modality_config`, then calls `get_action`.

The wrapper inventories and hashes every file in the canonical, symlink-free checkpoint
directory before constructing the official `Gr00tPolicy`, then verifies both the
checkpoint manifest and pinned checkout again before constructing the official
`PolicyServer`. The identity endpoint returns the resolved checkpoint path as
`checkpoint_identity` and the combined manifest SHA-256 as `checkpoint_digest`; it
re-verifies the checkpoint whenever queried. Runtime configuration must contain those
exact values. Identity is never trusted from `get_action`'s auxiliary info, and the
inference result remains action-only.

Scene context must contain nested `video` and `state` mappings whose keys and
temporal horizons match the checkpoint modality configuration. Video values are
absolute paths contained by the configured media root. They are decoded to RGB
inside the GR00T environment. State values cross the boundary as finite JSON
numbers and become float32 arrays there. The task instruction supplies each
configured language key.

The backend accepts only genuine float32 `(B,T,D)` arrays from every configured
action key, concatenates their final dimensions in modality order, and retains
batch zero as an immutable `ActionChunk`. Timestamps are the checkpoint action
`delta_indices`; the schema records the ordered action keys.

This adapter documents and tests the public N1.6 interfaces only. It does not
claim access to unpublished checkpoints, private data, or undisclosed recipes.
## Fine-tuning and open-loop evaluation

EvoWorld keeps NVIDIA Isaac GR00T unmodified at revision
`ead52833afbbf4243f8cd5e7664f48a94de03b19` (`n1.6-release`). The training
planner validates the checkout, the curated manifest, and the complete GR00T
LeRobot v2.1 dataset before it emits an argument vector. It never invokes a
shell.

## Fine-tuning

The method settings reproduced from the appendix are:

- base checkpoint `nvidia/GR00T-N1.6-3B`;
- 8 GPUs and global batch size 64;
- 5,000 optimization steps;
- PF-only expert samples in the train split.

The command also makes three values explicit that are inherited from the
pinned N1.6 `FinetuneConfig`, rather than reported as method-specific choices:
save every 1,000 steps, keep 5 checkpoints, and use 2 data-loader workers.
These values are recorded in `configs/experts/groot_n16.toml` so an upstream
default change cannot silently alter a run.

The emitted command is the official entry point:

```text
uv run python gr00t/experiment/launch_finetune.py ...
```

It includes the base model, absolute dataset and output paths, embodiment tag,
registration module, GPU count, global batch size, maximum steps, save cadence,
and worker count using the pinned Tyro signatures. The expected final artifact
is `checkpoint-5000` under the job output directory.

The dataset audit requires consistent `info.json`, `tasks.jsonl`,
`episodes.jsonl`, `stats.json`, `relative_stats.json`, and `modality.json`
metadata. It checks LeRobot v2.1 chunk and frame-rate fields, Arrow dtypes and
fixed vector widths, contiguous episode/frame/global indices, task and language
task indices, timestamps, and the 16-row minimum imposed by the action deltas.
The language annotation follows the pinned loader by mapping to the integer
`task_index` feature; free-form string columns are rejected. Video validation
streams the MP4 metadata instead of loading the whole media file, accepts the
pinned H.264/H.265/MPEG-4 and AV1 sample entries, checks dimensions, sample
tables, frame count, and frame rate, then fully decodes every frame with the
bundled FFmpeg runtime. The decoder receives only an explicit UTF-8 locale
environment and a `file,pipe` protocol whitelist, so host reporting, loader,
and executable override variables cannot affect it.

Dataset provenance is one deterministic, deduplicated digest over every
Parquet and video plus the shared metadata and the registration source at the
configured relative path. Statistics and relative statistics must have finite
numeric leaves. A change to media, statistics, modality metadata, or
registration code therefore invalidates the manifest before launch. For
`NEW_EMBODIMENT`, the source is parsed with an exact recursive AST whitelist;
the only assignment target is exactly `evoworld_config`, so registration
symbols and special names cannot be shadowed. After the complete static proof,
an isolated process imports the exact same original file supplied to the
training command and verifies the registered tag and every modality value
against the audited AST. `LIBERO_PANDA` is the only verified released built-in
tag, uses its pinned 16-step configuration, and cannot supply a custom
registration module.

## Open-loop evaluation

Open-loop evaluation is a separate audited plan using the official
`gr00t/eval/open_loop_eval.py` entry point. It binds exactly the final
`groot-n16-<manifest-hash>/checkpoint-5000` artifact, rather than any sibling
directory with the same basename. Each plan accepts one trajectory ID so plots
cannot overwrite one another. The validated dataset, embodiment tag, 16-step
action horizon, and action modality keys are carried into the command. Its
output is a trajectory plot; it is not treated as another fine-tuning
invocation.

## Policy service adapter

Online inference runs in the GR00T environment through the EvoWorld policy
service adapter. The core orchestrator exchanges normalized requests and
batched `(B, T, D)` action chunks with that adapter; it does not import GR00T
from the core process. Service transport, health checks, and modality mapping
are documented with the adapter implementation, while this section covers the
training and open-loop plans only.

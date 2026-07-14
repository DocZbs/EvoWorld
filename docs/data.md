# Data contracts

EvoWorld does not discover training examples by scanning a run directory. Training
consumes an explicit, immutable `TrainingManifest`; evaluator tuning consumes a
separate strict annotation JSONL that is exported deterministically for ms-swift.
Paths remain relative inside manifests, while launchers require an absolute dataset
root and re-hash files before execution.

## Canonical training manifest

A manifest has `schema_version`, `manifest_id`, and a non-empty ordered `samples`
array. Every `TrainingSample` contains:

- `sample_id`, `target`, `task_id`, instruction, and `skill_family`;
- `scene_id` and `trajectory_id`;
- relative `dataset_root`, ordered `observation_paths`, and optional `action_path`;
- complete action provenance when actions exist: schema, dimension, and horizon;
- explicit `source_type` and `source_checkpoint`;
- diagnosis `TSM`, `PF`, `WMF`, or `NONE` and split `train` or `eval`; and
- `content_sha256`, computed from the bound sample artifacts.

Canonical JSON with sorted keys defines `manifest_sha256`. A persisted training plan
records that digest and the 40-character upstream revision. Execution rejects a
changed manifest, checkout, source file, dataset artifact, target-declared runtime
binding, or completed job.

Environment binding is target-specific. Ctrl-World records and revalidates the
resolved Python executable/version plus a complete fingerprint of every relevant
installed distribution. Current openpi and GR00T plans intentionally have an empty
`environment` mapping; their immutable plan instead binds the explicit argv,
checkout revision, recipe, manifest, prepared dataset, and expected outputs. Do not
interpret those plans as a full host-environment fingerprint.

Diagnosis-aware curation ignores a caller-supplied target. `PF` requires an explicit
selected expert and is routed only to that expert; `WMF` routes to `ctrl_world`;
`TSM` is handled by grounding/router memory rather than a model fine-tune; verified
`NONE` examples may be included only when evaluator curation is explicitly enabled.

## DROID-derived layouts

Operators must obtain and prepare [DROID](https://droid-dataset.github.io/) under
its own terms. This repository contains validators and conversion contracts, not the
dataset.

### openpi

The pi0.5/pi0/pi0-FAST launchers require a preconverted LeRobot v2.1 root with:

```text
meta/info.json
meta/tasks.jsonl
meta/episodes.jsonl
meta/episodes_stats.jsonl
meta/evoworld_manifest.json
data/chunk-000/episode_*.parquet
```

The sidecar must equal the planned manifest. Episodes are train-only and map one to
one to samples. DROID actions are eight-dimensional joint velocity plus gripper;
task text, episode lengths, feature shapes, content hashes, and `chunk-000` placement
are checked before normalization or training.

### GR00T-N1.6

GR00T also uses LeRobot v2.1, with its video and modality metadata:

```text
meta/info.json
meta/tasks.jsonl
meta/episodes.jsonl
meta/stats.json
meta/relative_stats.json
meta/modality.json
meta/evoworld_manifest.json
meta/evoworld_modality.py
data/chunk-000/episode_*.parquet
videos/chunk-000/<camera-key>/episode_*.mp4
```

`info.json` must declare one train-only split, contiguous indices, consistent counts,
and the pinned data/video patterns. The modality registration must bind language,
state, action, cameras, and the 16-step action horizon. Statistics must be finite and
shape-compatible. The custom `NEW_EMBODIMENT` module is parsed and probed before the
official launcher is called.

### Ctrl-World

Ctrl-World preparation starts from explicit DROID input and produces immutable
latent/media inventories before metadata binding. Training requires non-empty
`annotation/train` and `annotation/val`, typed rows for every episode, exactly three
resized camera videos and three latent tensors per episode, plus finite ordered
`state_01` and `state_99` statistics. WMF data is an independently materialized
projection containing only manifest-selected `WMF/train` episodes; base training
accepts only real-source `NONE/train` records.

Preparation output, metadata output, and training output are exclusive destinations.
Partial or stale outputs are not overwritten. See
[backends/ctrl_world.md](backends/ctrl_world.md) for the complete receipt and
environment-binding protocol.

## Evaluator annotations

Each Qwen3-VL `EvaluatorExample` binds task and instruction, scene, trajectory,
ordered relative frames and per-frame SHA-256 values, `source_type`, source
checkpoint, provenance, content hash, numeric score, diagnosis label, rationale,
annotator, revision, and split. `source_type` must be either `real` or
`manually_verified_synthetic`; it is never inferred from a path.

Every frame is contained under the absolute dataset root, decoded, and byte-checked.
The label partition is strict: a score at or above the reliability threshold requires
`NONE`; a score below it requires `TSM`, `PF`, or `WMF`. Export requires non-empty
train and eval splits and writes separate canonical JSONL files with ordered media
and strict assistant JSON.

## Leakage rules

General training splits are deterministic connected groups: any samples sharing a
`trajectory_id` or `scene_id` stay together. The seeded group rank is independent of
input order. Validation rejects duplicate `sample_id`, `trajectory_id`, or
`content_sha256` within a split and across train/eval.

Evaluator validation is stricter: `example_id`, `trajectory_id`, `content_sha256`,
and every `media_sha256` must be globally unique across both splits. The exporter
revalidates media immediately before writing, and the Qwen launcher revalidates both
JSONL splits and all media before and after SFT. Synthetic samples must retain the
generator checkpoint and manual-verification provenance so they cannot be mistaken
for real robot observations.

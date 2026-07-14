# EvoWorld Public Reproduction Design

## Status

Approved for specification on 2026-07-17. This document defines the public,
runnable reproduction of EvoWorld. It does not claim that unpublished datasets,
annotations, checkpoints, or robot hardware are included.

## Assumptions

1. The release must implement the complete four-stage EvoWorld loop rather than
   only provide training scripts or a collection of upstream links.
2. Users will provide large datasets, model weights, API credentials, and robot
   endpoints. The repository will validate these inputs and explain how to obtain
   public assets, but will not download them implicitly.
3. Ctrl-World, openpi, and Isaac-GR00T must remain unmodified upstream Git
   submodules pinned to reproducible revisions.
4. GPU-heavy backends run in isolated environments. The lightweight EvoWorld core
   communicates with them through explicit process contracts.
5. A deterministic dry-run must exercise the same orchestration state machine as a
   real run without requiring a GPU, network access, or model weights.
6. Existing project-page content remains available after the repository is
   reorganized.
7. No remote branch or push is part of this work. Local commits are used only as
   recoverable save points.

## Objective

Build a concise, testable implementation of the method described in EvoWorld:

1. generate or load open-vocabulary tasks;
2. anchor each task to an offline real-robot scene;
3. route among modular VLA experts using semantic memory and world-model rollouts;
4. execute the selected expert through a pluggable executor;
5. evaluate the trajectory with a VLM;
6. store verified trajectories or trigger a targeted grounding, policy, or
   world-model update.

The implementation is successful when a user can validate configuration, run a
complete deterministic loop locally, inspect every intermediate artifact, and
replace the deterministic adapters with the pinned official backends for real
inference and fine-tuning.

## Fidelity to the Method

The implementation follows the notation and control flow in the article.

### Contextual scene anchoring

For task instruction `l` and indexed observations `o_i`, the grounding component
computes aligned language and observation embeddings and either selects the nearest
neighbor or samples using the configured temperature:

```text
p(i | l) = softmax(cosine(e_l, e_o_i) / temperature)
```

Deterministic nearest-neighbor mode is intended for evaluation and tests. Softmax
sampling is intended for exploration during self-evolution.

### Strategic routing and imagined rollouts

The router produces a semantic compatibility score conditioned on the instruction,
grounded observation, expert metadata, and experience memory. Every eligible expert
then receives the same grounded scene and produces an action-conditioned imagined
rollout through the world-model adapter. Final selection uses:

```text
joint_score = lambda * router_score + (1 - lambda) * world_model_score
```

The run manifest records both component scores, the configured `lambda`, and the
selected expert so that the decision is auditable.

### VLM diagnosis and targeted evolution

The evaluator returns a continuous score in `[0, 1]` and one diagnostic label:

- `TSM`: task-scene mismatch;
- `PF`: policy failure;
- `WMF`: world-model failure;
- `NONE`: no failure for a verified trajectory.

When the score reaches the reliability threshold, the trajectory is appended to the
verified buffer. Otherwise:

- `TSM` updates grounding evidence and router memory;
- `PF` creates or executes a fine-tuning job for only the selected expert;
- `WMF` creates or executes a Ctrl-World post-training job.

Stable modules are not globally retrained.

## Architecture

### Dependency direction

```text
CLI and configuration
        |
        v
EvoWorld application pipeline
        |
        v
Domain contracts and ports
        ^
        |
Deterministic adapters / upstream process adapters / storage adapters
```

Domain code never imports any submodule. Integrations implement domain ports and
may invoke bridge programs inside backend-specific environments.

### Repository structure

```text
src/evoworld/
  cli.py                    command-line entry point
  config.py                 TOML loading and validation
  contracts.py              immutable domain values and enums
  pipeline.py               four-stage loop and resume logic
  grounding.py              retrieval and sampling
  routing.py                memory-aware and rollout-aware selection
  evolution.py              targeted update decisions
  storage.py                run manifests, memory, and verified buffer
  ports.py                  adapter protocols
  adapters/
    deterministic.py        no-GPU end-to-end backend
    process.py              JSONL process transport
    ctrl_world.py           world-model command and response mapping
    openpi.py               pi0/pi0.5/pi0-FAST command mapping
    groot.py                GR00T-N1.6 command mapping
    qwen_vl.py              evaluator command mapping
integrations/
  ctrl_world/               bridge entry points run in Ctrl-World environment
  openpi/                   bridge entry points run in openpi environment
  groot/                    bridge entry points run in GR00T environment
  qwen_vl/                  evaluator inference bridge
training/
  experts/                  expert dataset preparation and fine-tuning launchers
  world_model/              Ctrl-World preparation and post-training launcher
  evaluator/                evaluator schema, SFT/LoRA launcher, and validation
configs/
  reproduction.toml         method-aligned real-backend configuration
  dry_run.toml              deterministic local configuration
  experts/                  per-backbone settings
scripts/                    environment setup and dataset indexing helpers
tests/
  unit/                     formulas and domain behavior
  contract/                 backend command and protocol compatibility
  integration/              full loop and diagnosis branches
third_party/
  ctrl_world/               pinned Git submodule
  openpi/                   pinned Git submodule
  isaac_groot/              pinned Git submodule
site/                       existing project page
docs/                       architecture, data, training, and reproduction guides
```

## Domain Contracts

The domain layer uses frozen dataclasses, enums, `pathlib.Path`, and typed protocols.
Large media are passed by path, never embedded in JSON.

### Core values

- `TaskSpec`: task ID, instruction, optional skill hints, metadata.
- `SceneRecord`: scene ID, observation paths, context, embedding reference.
- `ExpertSpec`: expert ID, backend, task families, checkpoint, action schema.
- `ActionChunk`: expert ID, ordered numeric actions, timestamps, schema name.
- `Rollout`: rollout ID, expert ID, scene ID, frame paths, actions, metrics.
- `CandidateScore`: router score, world-model score, joint score, validity.
- `Diagnosis`: score, label, reason, evaluator metadata.
- `ExperienceRecord`: task, scene, expert, score, diagnosis, artifact references.
- `UpdateDecision`: target module, training dataset, command plan, execution mode.
- `RoundState`: stage, inputs, outputs, timestamps, deterministic seed.

### Adapter ports

- `TaskSource.next_task()`
- `SceneAnchor.anchor(task)`
- `Router.shortlist(task, scene, memory)`
- `ExpertPolicy.act(task, observation)`
- `WorldModel.imagine(task, scene, expert, actions)`
- `RolloutScorer.score(task, rollout)`
- `Executor.execute(task, scene, expert)`
- `TrajectoryEvaluator.evaluate(task, scene, trajectory, memory)`
- `Trainer.plan(update)` and `Trainer.execute(plan)`
- `RunStore` and `ExperienceMemory`

## Backend Isolation Protocol

Backends use newline-delimited JSON over standard input and output. Each request has:

```json
{
  "protocol_version": 1,
  "request_id": "round-000001-openpi",
  "operation": "act",
  "artifact_dir": "/absolute/run/path",
  "payload": {}
}
```

Each response has `request_id`, `status`, `payload`, `artifacts`, `metrics`, and an
optional structured `error`. Only configured executables and argument arrays are
launched. The core does not concatenate shell strings.

Long-lived expert or evaluator services may use the same logical request/response
schema through their official client transport. Process startup, health checks, and
shutdown remain the responsibility of the adapter.

The pinned upstream command-line interfaces do not expose every setting required by
the method. EvoWorld bridge programs may therefore import official upstream Python
components, construct their public configuration objects, and invoke their training
loops with explicit values. These bridges must not patch or edit submodule files,
must record every override in `job.json`, and must be covered by contract tests.

## Pipeline State and Artifacts

Each invocation creates `runs/<run_id>/` with:

```text
manifest.json               immutable resolved configuration and Git revisions
state.json                  latest resumable state
tasks.jsonl                 generated or loaded task specifications
memory.jsonl                append-only experience records
rounds/<round_id>/
  grounding.json
  candidates.json
  diagnosis.json
  update.json
  rollouts/<expert_id>/      media and rollout metadata
verified/<trajectory_id>/    accepted trajectories and provenance
training/<job_id>/           command, logs, metrics, and output references
```

State files are written atomically. Resume refuses a changed resolved configuration
unless the user explicitly starts a new run.

## Upstream Submodules

The first public release pins the following official revisions:

| Path | Repository | Revision | Purpose |
| --- | --- | --- | --- |
| `third_party/ctrl_world` | `Robert-gyj/Ctrl-World` | `99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d` | action-conditioned rollout and world-model training |
| `third_party/openpi` | `Physical-Intelligence/openpi` | `15a9616a00943ada6c20a0f158e3adb39df2ccac` | pi0, pi0.5, and pi0-FAST inference/fine-tuning |
| `third_party/isaac_groot` | `NVIDIA/Isaac-GR00T` | `ead52833afbbf4243f8cd5e7664f48a94de03b19` (`n1.6-release`) | GR00T-N1.6 inference/fine-tuning |

Recursive submodule initialization is documented because openpi and GR00T may
contain their own submodules. A verification command compares every checked-out
revision against this table.

## Fine-Tuning Design

### Shared behavior

Training code consumes curated manifests rather than reading run directories by
convention. Every manifest includes task family, scene, trajectory, provenance,
diagnosis, action schema, source checkpoint, and split. Launchers support `plan` and
`execute` modes. `plan` is the default and writes the exact command without running
it.

All expert adaptations use a 5,000-step budget. Upstream optimizer, schedule,
preprocessing, action representation, batch construction, and loss definitions are
preserved unless an explicitly documented EvoWorld setting overrides them.

### openpi experts

The launcher:

1. validates or converts the matched subset to LeRobot/DROID form;
2. checks the `chunk-000` split;
3. computes normalization statistics through the official entry point;
4. generates EvoWorld-specific upstream training configs without editing the
   submodule;
5. invokes an EvoWorld bridge around the official JAX data loader and training
   components so `shuffle=False` can be enforced.

Method-aligned settings are:

| Expert | Action dimension | Horizon | Batch size | Maximum tokens |
| --- | ---: | ---: | ---: | ---: |
| pi0.5 | 32 | 15 | 16 | upstream default |
| pi0 | upstream default | 10 | 8 | upstream default |
| pi0-FAST | 8 | 10 | 8 | 180 |

The data loader uses eight workers, no shuffle, and joint-velocity supervision.
The pinned upstream `scripts/train.py` hard-codes `shuffle=True`, so it is not used
directly for method-aligned training. Normalization still uses the official
`scripts/compute_norm_stats.py` entry point. The bridge records that it is an adapter
around upstream APIs rather than an unmodified official CLI invocation.

### GR00T-N1.6 expert

The launcher invokes the official N1.6 fine-tuning entry point with:

- base checkpoint `nvidia/GR00T-N1.6-3B`;
- eight GPUs;
- global batch size 64;
- 5,000 maximum steps;
- an explicit LeRobot modality configuration;
- dataset and output paths supplied by configuration.

### Ctrl-World

The world-model launcher exposes three operations:

1. DROID latent extraction and metadata preparation;
2. base training through bridge-owned composition of pinned official components;
3. diagnosis-selected downstream post-training for `WMF` examples.

Rollout adapters map EvoWorld tasks, grounded observations, and expert action chunks
to official Ctrl-World replay or policy-in-the-loop entry points.

The pinned Ctrl-World training CLI does not expose its output directory or maximum
step budget and defaults to 500,000 steps. Its epoch loop and periodic-save interval
also do not guarantee a final artifact at 5,000 steps. EvoWorld therefore audits that
loop and uses a bridge-owned orchestration of the same pinned `wm_args`, model,
dataset, loss, optimizer, Accelerate accumulation, and checkpoint components. It
increments only synchronized optimizer updates, stops at exactly 5,000, and forces a
final checkpoint. There is no separate upstream post-training executable; `WMF`
uses a manifest-exact staged train projection and explicit starting checkpoint.
World size is explicit and runtime-checked; the public baseline is one process and
does not claim an unpublished eight-GPU Ctrl-World recipe. Every preparation and
training plan first enters the absolute checked-in bridge under the bound Ctrl-World
Python. The bootstrap verifies I2 revision/dirty state, pinned source blobs, inputs,
outputs, and a complete RECORD-based distribution fingerprint before it can execute
Accelerate or an official script. Distributed training re-enters the bridge as a
`train-worker` and repeats verification. Explicit module bindings include the
Accelerate launcher/accelerator, `imageio.v3`, and Torch optimizer, serialization,
neural-network, and data-loader code. The first hop uses only stdlib until it verifies
the bridge and every local helper path/hash. Preparation is explicitly two-stage:
latent output is absent and exclusively claimed, validated, and receipted before a
new content-bound metadata request can be created; metadata has its own absent-output
claim, runs the official script only in a unique same-filesystem staging directory,
and repeats the receipt plus complete prepared-byte audit before an atomic no-replace
publication. Failure quarantines only job-owned staging/request artifacts and leaves
external output untouched. Canonical workspace and request roots are device/inode
bound, opened component-by-component with `openat(O_NOFOLLOW)` before mutable
callbacks, and held for fd-relative staging, final publication, and quarantine.
The official metadata child inherits those verified descriptors, rechecks identity,
uses `fchdir` plus a root-relative prepared path, and execs the pinned absolute script;
descriptor-to-absolute-path conversion is not part of the official access boundary.
Backend and script descriptors bind regular-file device, inode, size, and SHA-256
before mutable callbacks. Linux performs fd-backed backend exec with an inherited
script fd; unsupported platforms fail closed. The parent reopens canonical trusted
roots after child return and permits no post-audit or publication before identity
revalidation succeeds.
RECORD raw bytes and relative identity are bound, unsafe/duplicate/escaping
entries and bounded-inventory violations fail closed, and verified editable installs
remain supported. Editable direct-URL bytes are streamed into one bounded buffer and
bound independently of the optional RECORD hash, with an exact 1 MiB cap, together
with their canonical runtime-path root. The final
checkpoint is published only after a strict reload
succeeds and an exclusive no-overwrite publish wins.

### Qwen3-VL evaluator

The evaluator training directory defines a JSONL schema for real trajectories and
manually annotated synthetic trajectories. Each example includes task text, ordered
frame paths, scalar score, diagnostic label, and rationale. Validation rejects scores
outside `[0, 1]`, unknown labels, missing frames, and train/evaluation leakage.

The public baseline launcher resolves `ms-swift==4.3.1`,
`qwen-vl-utils==0.0.14`, `decord==0.6.0`, `transformers>=4.57,<5.13.0`, and a
CUDA-capable `torch>=2.0` plus `torchvision>=0.15` runtime with `uvx` before
executing `swift sft`. Training
runs from the dataset root so ordered relative frame paths follow the direct
frame-list branch in qwen-vl-utils. That branch requires at least two frames and
pads odd counts to its `FRAME_FACTOR=2`; the planner rejects lists or padding that
exceed the configured frame budget.

The launcher performs parameter-efficient supervised fine-tuning of
Qwen3-VL-4B-Instruct and exports a bridge-compatible checkpoint. All recipe values,
including learning rate, LoRA rank, epochs or steps, batch size, accumulation,
maximum length, video token budget, and seed, must be explicit in configuration.
An existing output is reusable only when ms-swift's saved `args.json`, final
checkpoint log, LoRA adapter configuration, and non-empty adapter artifacts prove
that it completed the same immutable job and export. Because ms-swift writes its
final log even from a `finally` path, the EvoWorld runner adds an atomic completion
marker only after the SFT process exits successfully and the official artifacts pass
validation.

The checked-in runner revalidates the immutable job recipe, both exported JSONL
files, and every referenced media path and SHA-256 immediately after the CUDA runtime
probe and again after SFT returns. A changed split or media byte therefore prevents
SFT launch or completion-marker publication. Training uses
`--save_strategy epoch`, for which the official Trainer lifecycle saves the final
epoch as `checkpoint-<global_step>` even when the total step count is below or not
divisible by a step-based interval.
Exact annotations, prompt, hyperparameters, framework revision, and checkpoint used
for reported results are not published and must not be represented as included or
exactly reproduced by the public baseline.

## Configuration

Configuration uses TOML so the core can parse it with Python 3.11's standard
library. Paths are resolved relative to the configuration file, then recorded as
absolute paths in the run manifest. Environment variables may supply secrets or
machine-specific locations, but resolved secret values are never serialized.

Validation checks:

- supported protocol and schema versions;
- unique expert IDs and at least one enabled expert;
- `lambda` and reliability threshold in `[0, 1]`;
- positive rollout horizon and training budgets;
- compatible expert action schemas;
- required files, executable commands, submodule revisions, and backend health;
- explicit executor selection for non-dry runs.

## Commands

The intended public commands are:

```bash
# Clone and initialize pinned upstream code.
git clone --recurse-submodules https://github.com/DocZbs/EvoWorld.git
cd EvoWorld

# Install the lightweight core and development tools.
uv sync --extra dev

# Validate local configuration and upstream revisions.
uv run evoworld validate --config configs/dry_run.toml

# Run one deterministic end-to-end round without GPUs or weights.
uv run evoworld run --config configs/dry_run.toml --rounds 1

# Run or resume the configured reproduction loop.
uv run evoworld run --config configs/reproduction.toml --rounds 100
uv run evoworld resume --run-dir runs/<run-id>

# Plan first, then explicitly execute a fine-tuning job.
uv run evoworld finetune plan --config configs/reproduction.toml --target pi05
uv run evoworld finetune execute --job runs/<run-id>/training/<job-id>/job.json

# Quality gates.
uv run pytest
uv run ruff check .
uv run mypy src
```

Backend environment setup is intentionally separate and documented under
`docs/backends/`.

## Code Style

Use small, typed functions with explicit domain values and early validation. Avoid
backend checks throughout the pipeline; select an adapter once during composition.

```python
@dataclass(frozen=True, slots=True)
class CandidateScore:
    expert_id: str
    router_score: float
    world_model_score: float

    def joint(self, weight: float) -> float:
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be between 0 and 1")
        return weight * self.router_score + (1.0 - weight) * self.world_model_score
```

Naming uses `snake_case` for modules and functions, `PascalCase` for types, and
domain terminology from the method. Public functions and types have docstrings.

## Error Handling

- Configuration and schema errors fail before GPU processes start.
- A failed candidate rollout is recorded as invalid and does not prevent other
  experts from being evaluated.
- A round fails only when no candidate remains valid or a required execution stage
  cannot complete.
- Every external process has a timeout, captures stdout/stderr, records its exit
  status, and raises a stage-specific exception.
- Interrupted runs preserve the last completed stage and may be resumed.
- Missing private assets produce actionable errors; they never cause an implicit
  switch to deterministic adapters.
- Training execution is idempotent by job ID and refuses to overwrite a completed
  output unless explicitly requested.

## Testing Strategy

### Unit tests

- cosine similarity, softmax sampling, and nearest-neighbor grounding;
- joint routing score and deterministic tie-breaking;
- experience-memory lookup and exploration/exploitation behavior;
- reliability threshold handling and `TSM`/`PF`/`WMF` decisions;
- atomic state transitions and resume validation;
- training-manifest curation and leakage detection.

### Contract tests

- JSONL protocol round trips and structured backend errors;
- Ctrl-World rollout and post-training command construction;
- openpi normalization, fine-tuning, and inference command construction;
- GR00T-N1.6 fine-tuning and evaluation command construction;
- Qwen3-VL dataset and evaluator response schemas;
- pinned submodule revisions.

### Integration tests

- a full successful dry-run round;
- one round for each diagnostic update branch;
- multi-expert routing with one failed candidate;
- interrupted run followed by resume;
- verified-buffer provenance from task through diagnosis.

Tests must not require network access, CUDA, external weights, or initialized
submodule contents unless explicitly marked as backend smoke tests.

## Project Page

Existing HTML, CSS, JavaScript, images, and PDF assets move under `site/`. A GitHub
Pages workflow deploys only that directory. The disabled code button becomes a link
to the repository root. Moving the site is a mechanical change kept separate from
pipeline behavior commits.

## Boundaries

### Always

- pin and verify every upstream revision;
- preserve upstream source without local edits;
- record full provenance for generated data and training jobs;
- run targeted tests, lint, and type checks before each implementation commit;
- keep dry-run and real execution on the same domain pipeline;
- use absolute artifact paths in backend requests;
- keep all changes and commits local unless the user separately authorizes a push.

### Ask first

- change a pinned upstream revision;
- add a new large runtime dependency to the core;
- change published artifact or protocol schemas incompatibly;
- execute expensive real training or download large model/data assets;
- interact with real robot hardware;
- push, create a remote branch, or open a pull request.

### Never

- commit credentials, checkpoints, datasets, generated videos, or run outputs;
- edit files inside a Git submodule from the parent repository;
- silently substitute fake outputs in a real run;
- claim reproduction of reported metrics without the required private artifacts;
- globally retrain stable modules when a diagnosis identifies one update target;
- build shell commands by concatenating untrusted task or path values.

## Success Criteria

1. The repository installs as a Python 3.11 project and exposes the documented CLI.
2. Three official upstream repositories are pinned as Git submodules at the approved
   paths and revisions.
3. `configs/dry_run.toml` completes at least one full four-stage loop and produces an
   inspectable run manifest, memory entry, rollout, diagnosis, and update result.
4. Tests prove the successful path and all three diagnosis branches.
5. Real adapters construct and validate the official upstream inference and training
   entry points without modifying submodule code.
6. Fine-tuning launchers encode all method-aligned expert budgets and dimensions
   listed in this document.
7. Ctrl-World and evaluator post-training paths are present and selected by the
   correct diagnostic labels.
8. Resume, external-process failure, and missing-asset behavior are tested.
9. Documentation covers setup, architecture, data layout, backend environments,
   fine-tuning, dry-run, real runs, and limitations.
10. The project page remains deployable and links to the released code.
11. `pytest`, Ruff, mypy, and the end-to-end dry-run complete successfully in a clean
    lightweight environment.
12. A final audit maps every method stage and appendix training setting to code,
    configuration, tests, and documentation.

## Source Alignment

Implementation-specific behavior follows these official sources:

- Ctrl-World: <https://github.com/Robert-gyj/Ctrl-World>
- openpi: <https://github.com/Physical-Intelligence/openpi>
- Isaac-GR00T N1.6: <https://github.com/NVIDIA/Isaac-GR00T/tree/n1.6-release>

The supplied EvoWorld PDF is the authority for pipeline stages, equations,
diagnostic labels, expert settings, and evaluation protocol. When an upstream API
changes, the pinned revision and this specification take precedence for this
release.

## Open Questions

None. Any newly discovered conflict between the supplied PDF and a pinned upstream
entry point must be surfaced before changing the design.

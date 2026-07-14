# EvoWorld Training and Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver diagnosis-aware curation and plan-first fine-tuning for pi0/pi0.5/pi0-FAST, GR00T-N1.6, Ctrl-World, and Qwen3-VL-4B-Instruct.

**Architecture:** A canonical manifest and immutable job format separate EvoWorld data decisions from backend launchers. Each launcher validates inputs and emits an auditable argv plan before any expensive execution.

**Tech Stack:** Python 3.11, JSONL/TOML, official pinned upstream APIs, ms-swift 4.3.1, pytest.

## Global Constraints

- Every VLA adaptation uses exactly 5,000 optimization steps.
- GR00T-N1.6 uses 8 GPUs and global batch size 64.
- pi0.5 uses action dimension 32, horizon 15, batch size 16.
- pi0 uses horizon 10, batch size 8, and pinned upstream action dimension.
- pi0-FAST uses action dimension 8, horizon 10, max token length 180, batch size 8.
- openpi loading uses eight workers, `shuffle=False`, and joint-velocity supervision.
- TSM creates no GPU-training manifest; PF targets only the selected expert; WMF targets only Ctrl-World.
- Training execution is explicit, idempotent, and refuses stale manifests/revisions.
- The Qwen public baseline must not be described as the unpublished exact recipe.
- New filenames must not contain `paper`; do not push.

---

### Task T1: Canonical training manifest and job plans

**Files:**
- Create: `training/common/schema.py`
- Create: `training/common/jobs.py`
- Test: `tests/unit/test_training_schema.py`
- Test: `tests/unit/test_training_jobs.py`

**Interfaces:**
- Produces: `TrainingSample`, `TrainingManifest`, `TrainingPlan`, `write_job`, `execute_job`.

- [ ] **Step 1: Write failing schema and job safety tests**

```python
def test_training_sample_rejects_parent_media_path() -> None:
    with pytest.raises(TrainingSchemaError, match="relative"):
        sample(observation_paths=(Path("../frame.png"),))


def test_execute_rejects_changed_manifest(tmp_path: Path) -> None:
    job = write_job(tmp_path, plan(manifest_sha256="original"))
    with pytest.raises(StaleJobError, match="manifest"):
        execute_job(job, current_manifest_sha256="changed", runner=FakeRunner())
```

- [ ] **Step 2: Verify tests fail before schemas exist**

Run: `uv run pytest tests/unit/test_training_schema.py tests/unit/test_training_jobs.py -v`

Expected: FAIL importing `training.common`.

- [ ] **Step 3: Implement canonical immutable values**

```python
@dataclass(frozen=True, slots=True)
class TrainingSample:
    schema_version: int
    sample_id: str
    target: str
    task_id: str
    instruction: str
    skill_family: str
    scene_id: str
    trajectory_id: str
    dataset_root: Path
    observation_paths: tuple[Path, ...]
    action_path: Path | None
    action_schema: str | None
    action_dimension: int | None
    action_horizon: int | None
    source_type: str
    source_checkpoint: str
    diagnosis: DiagnosisLabel
    split: str
    content_sha256: str


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    schema_version: int
    job_id: str
    target: str
    manifest_sha256: str
    upstream_revision: str
    cwd: Path
    argv: tuple[str, ...]
    environment: Mapping[str, str]
    expected_outputs: tuple[Path, ...]
```

Require manifest-relative POSIX media paths without `..`; resolve only at launch.
Jobs store argv arrays, allowlisted nonsecret environment values, logs, status, and
output references. Completed jobs are immutable unless explicit overwrite is added
in a future approved design.

- [ ] **Step 4: Verify schema, job, lint, and types**

Run: `uv run pytest tests/unit/test_training_schema.py tests/unit/test_training_jobs.py -v && uv run ruff check training/common tests/unit/test_training_schema.py tests/unit/test_training_jobs.py && uv run mypy training`

Expected: validation, hashing, stale-revision, failure-log, and idempotency tests pass.

- [ ] **Step 5: Commit training foundations**

```bash
git add training/common/schema.py training/common/jobs.py tests/unit/test_training_schema.py tests/unit/test_training_jobs.py
git commit -m "feat: define auditable training jobs"
```

### Task T2: Diagnosis-aware curation and leakage-safe splits

**Files:**
- Create: `training/common/curate.py`
- Test: `tests/unit/test_training_curation.py`
- Create: `tests/fixtures/training_records.jsonl`

**Interfaces:**
- Produces: `curate_records`, `group_split`, `validate_no_leakage`.

- [ ] **Step 1: Write failing routing and leakage tests**

```python
def test_diagnoses_route_to_one_training_target() -> None:
    assert curate_records([record("TSM")]) == ()
    assert {row.target for row in curate_records([record("PF", expert_id="pi05")])} == {"pi05"}
    assert {row.target for row in curate_records([record("WMF")])} == {"ctrl_world"}


def test_split_rejects_trajectory_or_hash_leakage() -> None:
    with pytest.raises(DataLeakageError):
        validate_no_leakage([sample("train", "traj-1", "hash")],
                            [sample("eval", "traj-1", "hash")])
```

- [ ] **Step 2: Verify tests fail before curation exists**

Run: `uv run pytest tests/unit/test_training_curation.py -v`

Expected: FAIL importing curator functions.

- [ ] **Step 3: Implement deterministic group curation**

Group by trajectory ID, then scene ID when available, before seed-based split.
Reject sample-ID, trajectory-ID, or content-hash overlap. Never infer target from
directory names. Verified `NONE` examples may enter evaluator positives only when an
explicit evaluator flag is set.

- [ ] **Step 4: Verify curation**

Run: `uv run pytest tests/unit/test_training_curation.py -v && uv run ruff check training/common/curate.py tests/unit/test_training_curation.py && uv run mypy training`

Expected: PF/WMF/TSM, deterministic split, and leakage tests pass.

- [ ] **Step 5: Commit curation**

```bash
git add training/common/curate.py tests/unit/test_training_curation.py tests/fixtures/training_records.jsonl
git commit -m "feat: curate diagnosis-targeted training data"
```

### Task T3: openpi expert configs and no-shuffle bridge

**Files:**
- Create: `training/experts/openpi.py`
- Create: `integrations/openpi/train_bridge.py`
- Create: `configs/experts/openpi.toml`
- Test: `tests/contract/test_openpi_training.py`
- Modify: `docs/backends/openpi.md`

**Interfaces:**
- Produces: `OpenPiTrainConfig`, `plan_openpi_training`, bridge operation `train`.

- [ ] **Step 1: Write failing exact-setting tests**

```python
@pytest.mark.parametrize(
    ("name", "dimension", "horizon", "batch", "tokens"),
    [("pi05", 32, 15, 16, None), ("pi0", None, 10, 8, None),
     ("pi0_fast", 8, 10, 8, 180)],
)
def test_openpi_method_settings(name: str, dimension: int | None, horizon: int,
                                batch: int, tokens: int | None) -> None:
    config = load_openpi_config(CONFIG_PATH, name)
    assert (config.action_dimension, config.horizon, config.batch_size, config.max_tokens) == (dimension, horizon, batch, tokens)
    assert (config.steps, config.workers, config.shuffle) == (5000, 8, False)
```

Add a contract test whose fake upstream `create_data_loader` records
`shuffle=False` and `num_workers=8` when the bridge runs.

- [ ] **Step 2: Verify tests fail before the launcher exists**

Run: `uv run pytest tests/contract/test_openpi_training.py -v`

Expected: FAIL importing openpi training code.

- [ ] **Step 3: Implement custom config construction and bridge**

Normalization plan uses:

```python
["uv", "run", "scripts/compute_norm_stats.py", "--config-name", generated_name]
```

The bridge imports the pinned openpi config/model/data/training components, builds
pi05/pi0/pi0-FAST configs from `openpi.toml`, constructs its loader with exactly
`shuffle=False` and eight workers, and runs exactly 5,000 steps. It preserves
upstream optimizer, schedule, preprocessing, losses, and checkpoints. It does not
call pinned `scripts/train.py`, because that file forces `shuffle=True`.

- [ ] **Step 4: Verify all expert plans**

Run: `uv run pytest tests/contract/test_openpi_training.py -v && uv run ruff check training/experts/openpi.py integrations/openpi/train_bridge.py tests/contract/test_openpi_training.py && uv run mypy training integrations`

Expected: all three snapshots, data-layout validation, no-shuffle hook, and job hashes pass.

- [ ] **Step 5: Commit openpi training**

```bash
git add training/experts/openpi.py integrations/openpi/train_bridge.py configs/experts/openpi.toml tests/contract/test_openpi_training.py docs/backends/openpi.md
git commit -m "feat: plan method-aligned openpi fine-tuning"
```

### Task T4: GR00T-N1.6 fine-tuning launcher

**Files:**
- Create: `training/experts/groot.py`
- Create: `configs/experts/groot_n16.toml`
- Test: `tests/contract/test_groot_training.py`
- Modify: `docs/backends/groot.md`

**Interfaces:**
- Produces: `GrootTrainConfig`, `plan_groot_training`, `plan_groot_open_loop_eval`.

- [ ] **Step 1: Write failing exact-command tests**

```python
def test_groot_plan_uses_appendix_budget(tmp_path: Path) -> None:
    plan = plan_groot_training(valid_groot_manifest(tmp_path), load_config(CONFIG_PATH))
    assert argv_value(plan.argv, "--base-model-path") == "nvidia/GR00T-N1.6-3B"
    assert argv_value(plan.argv, "--num-gpus") == "8"
    assert argv_value(plan.argv, "--global-batch-size") == "64"
    assert argv_value(plan.argv, "--max-steps") == "5000"
```

- [ ] **Step 2: Verify tests fail before launcher exists**

Run: `uv run pytest tests/contract/test_groot_training.py -v`

Expected: FAIL importing GR00T training code.

- [ ] **Step 3: Build official N1.6 argv and input validation**

Use `uv run python gr00t/experiment/launch_finetune.py` with base model, dataset,
`NEW_EMBODIMENT` or explicit released tag, modality config, 8 GPUs, output, 5,000
steps, global batch 64, explicit save steps, and worker count. Validate GR00T-flavored
LeRobot v2, `meta/modality.json`, registration module, language key, and action
delta indices. Open-loop plan uses `gr00t/eval/open_loop_eval.py`.

- [ ] **Step 4: Verify GR00T plans and validation**

Run: `uv run pytest tests/contract/test_groot_training.py -v && uv run ruff check training/experts/groot.py tests/contract/test_groot_training.py && uv run mypy training`

Expected: exact command, modality, delta-index, and missing-path tests pass.

- [ ] **Step 5: Commit GR00T training**

```bash
git add training/experts/groot.py configs/experts/groot_n16.toml tests/contract/test_groot_training.py docs/backends/groot.md
git commit -m "feat: plan GR00T-N1.6 fine-tuning"
```

### Task T5: Ctrl-World preparation and bounded post-training bridge

**Files:**
- Create: `training/world_model/ctrl_world.py`
- Create: `integrations/ctrl_world/train_bridge.py`
- Create: `configs/world_model/ctrl_world.toml`
- Test: `tests/contract/test_ctrl_world_training.py`
- Modify: `docs/backends/ctrl_world.md`

**Interfaces:**
- Produces: preparation, metadata, base training, and WMF post-training job plans.

- [ ] **Step 1: Write failing prep and bounded-training tests**

```python
def test_ctrl_world_prepare_uses_official_entrypoints(tmp_path: Path) -> None:
    plans = plan_ctrl_world_prepare(valid_paths(tmp_path))
    assert plans[0].argv[2] == "prepare-latents"
    assert plans[1].argv[2] == "bind-metadata"


def test_ctrl_world_training_is_bounded_to_five_thousand_steps(tmp_path: Path) -> None:
    plan = plan_ctrl_world_training(wmf_manifest(tmp_path), ctrl_config(tmp_path))
    assert plan.payload["max_train_steps"] == 5000
    assert plan.payload["output_dir"] == str((tmp_path / "output").resolve())
```

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest tests/contract/test_ctrl_world_training.py -v`

Expected: FAIL importing Ctrl-World training code.

- [ ] **Step 3: Implement public preparation and safe bridge**

Preparation builds canonical requests whose plans first execute the checked-in
bootstrap under the absolute Ctrl-World Python. The bootstrap recomputes the I2
checkout audit, source/input/output bindings, and full RECORD-based environment
fingerprint before invoking absolute official latent extraction or metadata scripts.
Latent extraction claims an absent destination and publishes an atomic completion
receipt over the validated inventory. Only then may the second-stage binder generate
a new metadata request containing that receipt and exact prepared-tree inventory;
metadata also claims an absent destination and rejects every stale artifact.
The official command writes only to a unique same-filesystem staging directory. The
bridge repeats the complete receipt/prepared-byte audit, validates staging, and uses
an atomic no-replace directory rename before publishing completion; failure
quarantines only this run's staging and bound request.
The request binds canonical workspace/request-root device and inode identity. The
bridge verifies each component with `openat(O_NOFOLLOW)` before mutable callbacks and
holds trusted-root descriptors across the entire metadata run; all staging, final,
and quarantine operations are dirfd-relative and no-replace.
Official metadata execution uses a stdlib-only inherited-fd executor: `pass_fds`,
child-side identity checks, `fchdir(job_fd)`, relative prepared input, then fd-backed
backend/script execution. Ordinary absolute cwd/input paths are not reintroduced.
Backend and metadata-script regular files are path/device/inode/size/SHA-bound and
opened before callbacks. Linux executes both through inherited fds; unsupported
platforms fail closed. Canonical trusted-root identities are reopened and rechecked
immediately after the child and before post-audit or publication.
Training uses the same first-hop bootstrap; only after verification may it launch
Accelerate, which re-enters `train-worker` and verifies again. Validate
`annotation/{train,val}`, three-camera latent videos, train/val metadata, and p01/p99
stats. Training bridge imports official `wm_args`, model, dataset, loss, optimizer,
and Accelerate components; it does not call the unsafe upstream `main`. It stages a
manifest-exact WMF train projection, counts synchronized optimizer updates to exactly
5,000, and forces a strictly reloaded, exclusively published final checkpoint.
World size and distribution/module integrity are explicit and runtime-checked,
including Accelerate launcher/accelerator, `imageio.v3`, and Torch execution modules.
The first hop is stdlib-only until bridge and local-helper hashes pass. Distribution
fingerprints include raw RECORD bytes/identity, enforce safe canonical entries and
bounded streaming reads, and support explicitly bound editable roots. Editable
`direct_url.json` is lstat-checked, streamed, counted, and bound even with an empty
RECORD hash field, including its canonical runtime-path root and an exact 1 MiB cap.
WMF requires a diagnosis-curated dataset and starting checkpoint; public base
training requires real-source NONE/train records without a checkpoint. There is no
separate upstream post-training executable.

- [ ] **Step 4: Verify world-model plans**

Run: `uv run pytest tests/contract/test_ctrl_world_training.py -v && uv run ruff check training/world_model/ctrl_world.py integrations/ctrl_world/train_bridge.py tests/contract/test_ctrl_world_training.py && uv run mypy training integrations`

Expected: preparation, layout, bounded budget, checkpoint, and path tests pass.

- [ ] **Step 5: Commit Ctrl-World training**

```bash
git add training/world_model/ctrl_world.py integrations/ctrl_world/train_bridge.py configs/world_model/ctrl_world.toml tests/contract/test_ctrl_world_training.py docs/backends/ctrl_world.md
git commit -m "feat: plan bounded Ctrl-World updates"
```

### Task T6: Qwen3-VL evaluator data and public fine-tuning baseline

**Files:**
- Create: `training/evaluator/schema.py`
- Create: `training/evaluator/qwen_vl.py`
- Create: `configs/evaluator/qwen3_vl_4b.toml`
- Test: `tests/contract/test_qwen_training.py`
- Create: `tests/fixtures/evaluator_examples.jsonl`

**Interfaces:**
- Produces: `EvaluatorExample`, `validate_evaluator_dataset`, `export_ms_swift`, `plan_qwen_training`.

- [ ] **Step 1: Write failing annotation and recipe tests**

```python
def test_evaluator_label_matches_threshold() -> None:
    with pytest.raises(EvaluatorSchemaError, match="threshold"):
        validate_example(example(score=0.8, label="PF"), reliability_threshold=0.7)


def test_qwen_plan_requires_every_public_recipe_value(tmp_path: Path) -> None:
    config = qwen_config(tmp_path, learning_rate=None)
    with pytest.raises(EvaluatorConfigError, match="learning_rate"):
        plan_qwen_training(valid_evaluator_dataset(tmp_path), config)
```

Add tests for invalid UTF-8, missing/empty frames, unknown labels, blank rationale,
duplicate IDs, and train/eval trajectory/hash leakage.

- [ ] **Step 2: Verify tests fail before evaluator training exists**

Run: `uv run pytest tests/contract/test_qwen_training.py -v`

Expected: FAIL importing evaluator training code.

- [ ] **Step 3: Implement canonical schema and ms-swift export**

Each example contains task, ordered frames, real/synthetic source, provenance, score,
TSM/PF/WMF/NONE label, rationale, annotator, and revision. Export derived ms-swift
JSONL with fixed system instruction, multimodal user request, ordered video/frame
inputs, and assistant strict JSON `{score,label,rationale}`.

- [ ] **Step 4: Build the explicit public-baseline command**

```python
argv = (
    "uvx", "--from", "ms-swift==4.3.1",
    "--with", "qwen-vl-utils==0.0.14", "--with", "decord==0.6.0",
    "--with", "transformers>=4.57,<5.13.0", "--with", "torch>=2.0",
    "--with", "torchvision>=0.15",
    "swift", "sft", "--model", "Qwen/Qwen3-VL-4B-Instruct",
    "--tuner_type", "lora", "--dataset", str(export_path),
    "--learning_rate", str(config.learning_rate),
    "--lora_rank", str(config.lora_rank), "--lora_alpha", str(config.lora_alpha),
    "--num_train_epochs", str(config.num_train_epochs),
    "--per_device_train_batch_size", str(config.per_device_train_batch_size),
    "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
    "--max_length", str(config.max_length), "--seed", str(config.seed),
    "--output_dir", str(job_output), "--add_version", "false",
    "--save_strategy", "epoch",
)
```

Pin the complete evaluator runtime and CUDA platform; require explicit video
token/pixel and frame budgets plus all recipe values. Run from the dataset root,
revalidate both JSONL splits and every media hash immediately before and after SFT,
and accept completed output only after validating ms-swift's epoch-final LoRA
checkpoint and saved arguments. Documentation labels these values a configurable public
baseline, not the unpublished experiment recipe.

- [ ] **Step 5: Verify and commit evaluator fine-tuning**

Run: `uv run pytest tests/contract/test_qwen_training.py -v && uv run ruff check training/evaluator tests/contract/test_qwen_training.py && uv run mypy training`

Expected: schema, leakage, export, explicit-recipe, version, and argv tests pass.

```bash
git add training/evaluator/schema.py training/evaluator/qwen_vl.py configs/evaluator/qwen3_vl_4b.toml tests/contract/test_qwen_training.py tests/fixtures/evaluator_examples.jsonl
git commit -m "feat: add the Qwen3-VL evaluator baseline"
```

### Task T7: Unified plan-first fine-tuning CLI

**Files:**
- Create: `src/evoworld/training.py`
- Modify: `src/evoworld/cli.py`
- Test: `tests/integration/test_finetune_cli.py`

**Interfaces:**
- Produces: `evoworld finetune plan` and `evoworld finetune execute`.

- [ ] **Step 1: Write failing plan and execute tests**

```python
@pytest.mark.parametrize("target", ["pi05", "pi0", "pi0_fast", "groot_n16", "ctrl_world", "qwen_evaluator"])
def test_finetune_plan_writes_job_without_execution(tmp_path: Path, target: str) -> None:
    result = main(["finetune", "plan", "--config", str(config_for(target, tmp_path)),
                   "--target", target])
    assert result == 0
    assert FakeRunner.calls == []
    assert next(tmp_path.rglob("job.json")).is_file()


def test_finetune_execute_rejects_completed_job(tmp_path: Path) -> None:
    job = completed_job(tmp_path)
    assert main(["finetune", "execute", "--job", str(job)]) == 2
```

- [ ] **Step 2: Verify tests fail before CLI integration**

Run: `uv run pytest tests/integration/test_finetune_cli.py -v`

Expected: FAIL because `finetune` is not registered.

- [ ] **Step 3: Compose target-specific launchers**

Dispatch by explicit target enum, validate manifest and upstream revision before plan,
write `job.json` atomically, and print JSON summary. Execute validates hashes, paths,
revision, status, and environment; captures stdout/stderr and output references.

- [ ] **Step 4: Run the complete training quality gate**

Run: `uv run pytest tests/unit/test_training_schema.py tests/unit/test_training_jobs.py tests/unit/test_training_curation.py tests/contract/test_openpi_training.py tests/contract/test_groot_training.py tests/contract/test_ctrl_world_training.py tests/contract/test_qwen_training.py tests/integration/test_finetune_cli.py -v && uv run ruff check training integrations src tests && uv run mypy training integrations src`

Expected: every target plans offline; all checks exit zero; no backend training runs.

- [ ] **Step 5: Commit the fine-tuning CLI**

```bash
git add src/evoworld/training.py src/evoworld/cli.py tests/integration/test_finetune_cli.py
git commit -m "feat: expose plan-first EvoWorld fine-tuning"
```

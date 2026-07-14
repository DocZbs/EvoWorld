# EvoWorld Core Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the lightweight, deterministic, resumable four-stage EvoWorld orchestration core.

**Architecture:** Frozen domain values and typed ports isolate method logic from all heavy backends. TOML composition selects adapters once, while atomic run storage makes every stage inspectable and resumable.

**Tech Stack:** Python 3.11, standard-library `dataclasses`/`tomllib`/`argparse`, NumPy, pytest, Ruff, mypy, uv.

## Global Constraints

- Core runtime supports Python 3.11 or newer.
- Core modules never import files from `third_party/`.
- Dry-run and real execution use the same `EvoWorldPipeline` state machine.
- Missing real assets must fail; real mode never falls back to deterministic adapters.
- Configuration and artifacts use absolute paths after resolution.
- New filenames must not contain `paper`.
- All commits remain local; do not push or create remote branches.
- Use TDD and commit each task after its complete quality gate passes.

---

### Task C1: Package and quality-tool scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `uv.lock`
- Modify: `.gitignore`
- Create: `src/evoworld/__init__.py`
- Test: `tests/unit/test_package.py`

**Interfaces:**
- Produces: installable `evoworld` package and `evoworld = "evoworld.cli:main"` console entry.

- [ ] **Step 1: Write the failing package test**

```python
from importlib.metadata import version

import evoworld


def test_package_exposes_version() -> None:
    assert evoworld.__version__ == version("evoworld")
```

- [ ] **Step 2: Verify the test fails before the package exists**

Run: `python3.11 -m pytest tests/unit/test_package.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'evoworld'`.

- [ ] **Step 3: Add the package metadata and version export**

```toml
[project]
name = "evoworld"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["numpy>=2.0,<3"]

[project.scripts]
evoworld = "evoworld.cli:main"

[project.optional-dependencies]
dev = ["mypy>=1.17,<2", "pytest>=8.4,<9", "ruff>=0.12,<1"]

[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/evoworld"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["backend: requires initialized heavyweight backend"]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.mypy]
python_version = "3.11"
strict = true
```

```python
from importlib.metadata import version

__version__ = version("evoworld")
```

Extend `.gitignore` with `.venv/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`,
`__pycache__/`, `*.pyc`, `runs/`, `checkpoints/`, `outputs/`, `.env`, and model/media
extensions used by generated artifacts.

- [ ] **Step 4: Install and verify the scaffold**

Run: `uv sync --extra dev && uv run pytest tests/unit/test_package.py -v && uv run ruff check pyproject.toml src tests`

Expected: one passing test and zero Ruff errors.

- [ ] **Step 5: Commit the scaffold**

```bash
git add pyproject.toml .gitignore src/evoworld/__init__.py tests/unit/test_package.py uv.lock
git commit -m "chore: scaffold the EvoWorld Python package"
```

### Task C2: Immutable domain contracts and ports

**Files:**
- Create: `src/evoworld/contracts.py`
- Create: `src/evoworld/ports.py`
- Test: `tests/unit/test_contracts.py`

**Interfaces:**
- Produces: `TaskSpec`, `SceneRecord`, `ExpertSpec`, `ActionChunk`, `Rollout`, `Trajectory`, `CandidateScore`, `Diagnosis`, `ExperienceRecord`, `UpdateDecision`, `RoundState`, `RoundResult`, and adapter protocols.

- [ ] **Step 1: Write contract invariants as failing tests**

```python
import pytest

from evoworld.contracts import CandidateScore, Diagnosis, DiagnosisLabel, TaskSpec


def test_task_rejects_blank_instruction() -> None:
    with pytest.raises(ValueError, match="instruction"):
        TaskSpec(task_id="t1", instruction=" ")


def test_candidate_uses_method_joint_score() -> None:
    score = CandidateScore.from_components("pi05", "r1", 0.8, 0.2, 0.25)
    assert score.joint_score == pytest.approx(0.35)


def test_diagnosis_rejects_out_of_range_score() -> None:
    with pytest.raises(ValueError, match="score"):
        Diagnosis(score=1.1, label=DiagnosisLabel.NONE, reason="invalid")
```

- [ ] **Step 2: Run the tests and confirm missing contracts**

Run: `uv run pytest tests/unit/test_contracts.py -v`

Expected: FAIL importing `evoworld.contracts`.

- [ ] **Step 3: Implement frozen values and typed ports**

Use `@dataclass(frozen=True, slots=True)` and these stable signatures:

```python
JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class DiagnosisLabel(str, Enum):
    TSM = "TSM"
    PF = "PF"
    WMF = "WMF"
    NONE = "NONE"


@dataclass(frozen=True, slots=True)
class CandidateScore:
    expert_id: str
    rollout_id: str | None
    router_score: float
    world_model_score: float
    joint_score: float
    valid: bool = True
    error: str | None = None

    @classmethod
    def from_components(
        cls, expert_id: str, rollout_id: str, router_score: float,
        world_model_score: float, weight: float,
    ) -> "CandidateScore":
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be between 0 and 1")
        joint = weight * router_score + (1.0 - weight) * world_model_score
        return cls(expert_id, rollout_id, router_score, world_model_score, joint)
```

Define protocols with the approved signatures in the design, including
`TaskSource.next_task`, `SceneAnchor.anchor`, `Router.shortlist`,
`ExpertPolicy.act`, `WorldModel.imagine`, `RolloutScorer.score`,
`Executor.execute`, `TrajectoryEvaluator.evaluate`, `Trainer.plan/execute`,
`ExperienceMemory.lookup/append`, and `RunStore` artifact methods.

- [ ] **Step 4: Verify contracts, lint, and types**

Run: `uv run pytest tests/unit/test_contracts.py -v && uv run ruff check src/evoworld/contracts.py src/evoworld/ports.py tests/unit/test_contracts.py && uv run mypy src`

Expected: all contract tests pass and both static checks exit zero.

- [ ] **Step 5: Commit the shared interfaces**

```bash
git add src/evoworld/contracts.py src/evoworld/ports.py tests/unit/test_contracts.py
git commit -m "feat: define EvoWorld domain contracts"
```

### Task C3: TOML configuration and resolved manifests

**Files:**
- Create: `src/evoworld/config.py`
- Create: `configs/dry_run.toml`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: C2 enums and `ExpertSpec`.
- Produces: `EvoWorldConfig`, `load_config`, `validate_config`, `resolved_manifest`, `config_digest`.

- [ ] **Step 1: Write failing path and validation tests**

```python
def test_paths_resolve_relative_to_config(tmp_path: Path) -> None:
    path = write_config(tmp_path, run_root="artifacts/runs")
    config = load_config(path)
    assert config.run_root == (tmp_path / "artifacts/runs").resolve()


def test_real_mode_requires_explicit_executor(tmp_path: Path) -> None:
    path = write_config(tmp_path, mode="real", executor="")
    with pytest.raises(ConfigError, match="executor"):
        load_config(path)
```

- [ ] **Step 2: Confirm tests fail on the absent loader**

Run: `uv run pytest tests/unit/test_config.py -v`

Expected: FAIL importing `load_config`.

- [ ] **Step 3: Implement strict TOML resolution**

```python
@dataclass(frozen=True, slots=True)
class EvoWorldConfig:
    schema_version: int
    seed: int
    run_root: Path
    mode: str
    grounding_mode: GroundingMode
    grounding_temperature: float
    router_weight: float
    reliability_threshold: float
    rollout_horizon: int
    executor: str | None
    experts: tuple[ExpertSpec, ...]


def load_config(path: str | Path) -> EvoWorldConfig:
    config_path = Path(path).resolve(strict=True)
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    config = _parse_config(raw, base_dir=config_path.parent)
    validate_config(config)
    return config
```

Validate schema version 1, unique/enabled experts, bounds, positive horizons,
action-schema compatibility, and explicit real executor. Hash canonical JSON of all
behavior settings while replacing secret-bearing values with `"<redacted>"`.

- [ ] **Step 4: Verify config behavior**

Run: `uv run pytest tests/unit/test_config.py -v && uv run ruff check src/evoworld/config.py tests/unit/test_config.py && uv run mypy src`

Expected: all tests and checks pass.

- [ ] **Step 5: Commit configuration support**

```bash
git add src/evoworld/config.py configs/dry_run.toml tests/unit/test_config.py
git commit -m "feat: validate EvoWorld run configuration"
```

### Task C4: Atomic run storage and experience memory

**Files:**
- Create: `src/evoworld/storage.py`
- Test: `tests/unit/test_storage.py`

**Interfaces:**
- Consumes: C2 serialization values and C3 config digests.
- Produces: `atomic_write_json`, `JsonRunStore`, `JsonlExperienceMemory`.

- [ ] **Step 1: Write failing atomicity and idempotency tests**

```python
def test_state_replace_leaves_no_temporary_file(tmp_path: Path) -> None:
    store = JsonRunStore.create(tmp_path, "run-1", {}, "digest")
    store.write_state(round_state("round-1"))
    assert store.read_state().round_id == "round-1"
    assert list(store.run_dir.glob("*.tmp")) == []


def test_memory_append_is_idempotent(tmp_path: Path) -> None:
    memory = JsonlExperienceMemory(tmp_path / "memory.jsonl")
    record = experience_record(record_id="exp-1")
    memory.append(record)
    memory.append(record)
    assert memory.lookup(record.task, record.scene) == (record,)
```

- [ ] **Step 2: Confirm tests fail before storage exists**

Run: `uv run pytest tests/unit/test_storage.py -v`

Expected: FAIL importing storage classes.

- [ ] **Step 3: Implement atomic and deterministic storage**

```python
def atomic_write_json(path: Path, value: JsonValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
```

`JsonRunStore.create` uses exclusive manifest creation; `open` validates config
digest; append-only stores reject a different record with the same deterministic
ID and ignore an identical retry. Verified artifacts preserve absolute provenance.

- [ ] **Step 4: Verify storage**

Run: `uv run pytest tests/unit/test_storage.py -v && uv run ruff check src/evoworld/storage.py tests/unit/test_storage.py && uv run mypy src`

Expected: storage tests and static checks pass.

- [ ] **Step 5: Commit storage**

```bash
git add src/evoworld/storage.py tests/unit/test_storage.py
git commit -m "feat: persist resumable EvoWorld runs"
```

### Task C5: Contextual scene anchoring

**Files:**
- Create: `src/evoworld/grounding.py`
- Test: `tests/unit/test_grounding.py`

**Interfaces:**
- Produces: `cosine_similarity`, `softmax_probabilities`, `select_scene_id`, `IndexedSceneAnchor`.

- [ ] **Step 1: Write failing equation and sampling tests**

```python
def test_nearest_scene_uses_cosine_similarity() -> None:
    selected = select_scene_id([1.0, 0.0], {"b": [0.0, 1.0], "a": [1.0, 0.0]},
                               mode=GroundingMode.NEAREST, temperature=0.5,
                               rng=random.Random(7))
    assert selected == "a"


def test_softmax_sampling_is_seeded() -> None:
    ids = [select_scene_id([1.0], {"a": [1.0], "b": [0.1]},
                           mode=GroundingMode.SOFTMAX, temperature=0.8,
                           rng=random.Random(11)) for _ in range(2)]
    assert ids[0] == ids[1]
```

- [ ] **Step 2: Verify the tests fail**

Run: `uv run pytest tests/unit/test_grounding.py -v`

Expected: FAIL importing grounding functions.

- [ ] **Step 3: Implement the published grounding formula**

Use NumPy only for vector math, reject empty/zero/mismatched vectors, require
`temperature > 0`, use lexical scene ID tie-breaking, and use only the injected RNG
for sampling. `IndexedSceneAnchor` receives text encoder, indexed scene records, and
embeddings through constructor injection.

- [ ] **Step 4: Verify grounding**

Run: `uv run pytest tests/unit/test_grounding.py -v && uv run ruff check src/evoworld/grounding.py tests/unit/test_grounding.py && uv run mypy src`

Expected: equation, validation, tie, and seed tests pass.

- [ ] **Step 5: Commit grounding**

```bash
git add src/evoworld/grounding.py tests/unit/test_grounding.py
git commit -m "feat: anchor tasks to indexed scenes"
```

### Task C6: Memory-aware routing and rollout selection

**Files:**
- Create: `src/evoworld/routing.py`
- Test: `tests/unit/test_routing.py`

**Interfaces:**
- Produces: `MemoryAwareRouter`, `experience_score`, `select_candidate`.

- [ ] **Step 1: Write failing route and selection tests**

```python
def test_candidate_selection_ignores_invalid_rollouts() -> None:
    valid = candidate("pi05", joint=0.4, valid=True)
    invalid = candidate("groot", joint=0.9, valid=False)
    assert select_candidate((invalid, valid)) == valid


def test_candidate_tie_breaks_by_expert_id() -> None:
    assert select_candidate((candidate("pi05", 0.5), candidate("groot", 0.5))).expert_id == "groot"
```

- [ ] **Step 2: Verify routing tests fail**

Run: `uv run pytest tests/unit/test_routing.py -v`

Expected: FAIL importing routing functions.

- [ ] **Step 3: Implement routing**

`MemoryAwareRouter` combines injected semantic logits with successful historical
scores using a validated memory weight. Unfamiliar contexts sample categorical
logits using an injected seed; recurrent contexts rank deterministically. Candidate
selection sorts by `(-joint_score, expert_id)` and raises `NoValidCandidateError`
when all rollouts are invalid.

- [ ] **Step 4: Verify routing**

Run: `uv run pytest tests/unit/test_routing.py -v && uv run ruff check src/evoworld/routing.py tests/unit/test_routing.py && uv run mypy src`

Expected: exploration, exploitation, disabled-expert, invalid-rollout, and tie tests pass.

- [ ] **Step 5: Commit routing**

```bash
git add src/evoworld/routing.py tests/unit/test_routing.py
git commit -m "feat: route tasks with memory and rollout scores"
```

### Task C7: Diagnosis-driven evolution decisions

**Files:**
- Create: `src/evoworld/evolution.py`
- Test: `tests/unit/test_evolution.py`

**Interfaces:**
- Produces: `decide_update(...) -> UpdateDecision`.

- [ ] **Step 1: Write one failing test per method branch**

```python
@pytest.mark.parametrize(
    ("label", "target"),
    [(DiagnosisLabel.TSM, UpdateTarget.GROUNDING_ROUTER),
     (DiagnosisLabel.PF, UpdateTarget.EXPERT_POLICY),
     (DiagnosisLabel.WMF, UpdateTarget.WORLD_MODEL)],
)
def test_failure_targets_only_diagnosed_module(label: DiagnosisLabel, target: UpdateTarget) -> None:
    decision = decide_update(diagnosis=diagnosis(0.2, label), reliability_threshold=0.7,
                             task=task(), scene=scene(), expert=expert("pi05"),
                             trajectory=trajectory(), execution_mode=ExecutionMode.PLAN)
    assert decision.target is target
```

- [ ] **Step 2: Verify evolution tests fail**

Run: `uv run pytest tests/unit/test_evolution.py -v`

Expected: FAIL importing `decide_update`.

- [ ] **Step 3: Implement threshold and diagnosis rules**

Return `UpdateTarget.NONE` when `score >= threshold`; otherwise map exactly TSM,
PF, and WMF. Reject high-score failure labels and low-score `NONE`. PF carries only
the selected expert ID; TSM emits no GPU-training records; WMF targets Ctrl-World.

- [ ] **Step 4: Verify evolution**

Run: `uv run pytest tests/unit/test_evolution.py -v && uv run ruff check src/evoworld/evolution.py tests/unit/test_evolution.py && uv run mypy src`

Expected: success and all diagnosis branches pass.

- [ ] **Step 5: Commit evolution logic**

```bash
git add src/evoworld/evolution.py tests/unit/test_evolution.py
git commit -m "feat: target updates from trajectory diagnoses"
```

### Task C8: Deterministic adapters

**Files:**
- Create: `src/evoworld/adapters/__init__.py`
- Create: `src/evoworld/adapters/deterministic.py`
- Test: `tests/unit/test_deterministic_adapters.py`

**Interfaces:**
- Consumes: all C2 ports.
- Produces: `DeterministicScenario`, `AdapterSet`, `build_deterministic_adapters`.

- [ ] **Step 1: Write failing repeatability and scripted-failure tests**

```python
def test_same_seed_produces_same_actions(tmp_path: Path) -> None:
    first = build_deterministic_adapters(config(seed=9), tmp_path / "a")
    second = build_deterministic_adapters(config(seed=9), tmp_path / "b")
    assert first.policies["pi05"].act(task(), scene().observation_paths) == second.policies["pi05"].act(task(), scene().observation_paths)


def test_scenario_can_fail_one_rollout(tmp_path: Path) -> None:
    adapters = build_deterministic_adapters(config(), tmp_path,
        DeterministicScenario(failing_rollout_experts=frozenset({"pi05"})))
    with pytest.raises(DeterministicRolloutError):
        adapters.world_model.imagine(task(), scene(), expert("pi05"), actions())
```

- [ ] **Step 2: Verify tests fail before adapters exist**

Run: `uv run pytest tests/unit/test_deterministic_adapters.py -v`

Expected: FAIL importing deterministic adapters.

- [ ] **Step 3: Implement every port without external dependencies**

Generate stable IDs using SHA-256 of seed and domain IDs; write small text frame
artifacts under the absolute run directory; support scripted evaluation score/label
and per-expert rollout failure. Do not inspect GPU or network state.

- [ ] **Step 4: Verify deterministic adapters**

Run: `uv run pytest tests/unit/test_deterministic_adapters.py -v && uv run ruff check src/evoworld/adapters tests/unit/test_deterministic_adapters.py && uv run mypy src`

Expected: all port, repeatability, and failure tests pass.

- [ ] **Step 5: Commit adapters**

```bash
git add src/evoworld/adapters tests/unit/test_deterministic_adapters.py
git commit -m "feat: add deterministic EvoWorld adapters"
```

### Task C9: Successful four-stage pipeline

**Files:**
- Create: `src/evoworld/pipeline.py`
- Test: `tests/integration/test_pipeline_success.py`

**Interfaces:**
- Produces: `EvoWorldPipeline.run`, `run_round`, `resume`.

- [ ] **Step 1: Write the failing successful-round test**

```python
def test_successful_round_persists_every_method_stage(tmp_path: Path) -> None:
    pipeline = make_pipeline(tmp_path, label=DiagnosisLabel.NONE, score=0.9)
    result = pipeline.run(rounds=1)[0]
    assert result.verified is True
    round_dir = pipeline.store.run_dir / "rounds" / result.round_id
    assert {path.name for path in round_dir.iterdir()} >= {
        "grounding.json", "candidates.json", "trajectory.json", "diagnosis.json", "update.json"
    }
```

- [ ] **Step 2: Verify the integration test fails**

Run: `uv run pytest tests/integration/test_pipeline_success.py -v`

Expected: FAIL importing `EvoWorldPipeline`.

- [ ] **Step 3: Implement the four stages**

```python
class EvoWorldPipeline:
    def run(self, rounds: int) -> tuple[RoundResult, ...]:
        results = [self.run_round(f"round-{index:06d}") for index in range(rounds)]
        return tuple(results)

    def run_round(self, round_id: str, state: RoundState | None = None) -> RoundResult:
        current = state or self.new_round_state(round_id)
        current = self.complete_task_and_grounding(current)
        current = self.complete_routing_and_imagination(current)
        current = self.complete_execution(current)
        current = self.complete_evaluation_and_evolution(current)
        return self.round_result(current)

    def resume(self) -> RoundResult | None:
        state = self.store.read_state()
        if state is None or state.stage is RoundStage.COMPLETE:
            return None
        return self.run_round(state.round_id, state)
```

Persist completed state after: task+grounding; routing+all imagined rollouts;
selected-expert execution; evaluator+memory+evolution. Candidate failures become
invalid `CandidateScore` values. A verified trajectory is stored only after the
diagnosis threshold is met.

- [ ] **Step 4: Verify the successful flow**

Run: `uv run pytest tests/integration/test_pipeline_success.py -v && uv run ruff check src/evoworld/pipeline.py tests/integration/test_pipeline_success.py && uv run mypy src`

Expected: the test passes and artifacts match the method stages.

- [ ] **Step 5: Commit the pipeline**

```bash
git add src/evoworld/pipeline.py tests/integration/test_pipeline_success.py
git commit -m "feat: run the EvoWorld four-stage loop"
```

### Task C10: Diagnosis branches and candidate failure isolation

**Files:**
- Test: `tests/integration/test_pipeline_diagnoses.py`
- Modify: `src/evoworld/pipeline.py`

**Interfaces:**
- Consumes: C7 decisions and C8 scenarios.
- Produces: verified behavior for TSM, PF, WMF, partial candidate failure, and total candidate failure.

- [ ] **Step 1: Write failing parameterized branch tests**

```python
@pytest.mark.parametrize(
    ("label", "target"),
    [(DiagnosisLabel.TSM, UpdateTarget.GROUNDING_ROUTER),
     (DiagnosisLabel.PF, UpdateTarget.EXPERT_POLICY),
     (DiagnosisLabel.WMF, UpdateTarget.WORLD_MODEL)],
)
def test_pipeline_persists_targeted_update(tmp_path: Path, label: DiagnosisLabel, target: UpdateTarget) -> None:
    result = make_pipeline(tmp_path, label=label, score=0.2).run(1)[0]
    assert result.update.target is target
    assert result.verified is False
```

Add tests proving one failed candidate is skipped and all failed candidates raise
`NoValidCandidateError` with every candidate error recorded.

- [ ] **Step 2: Run tests to expose missing branch artifacts**

Run: `uv run pytest tests/integration/test_pipeline_diagnoses.py -v`

Expected: at least one FAIL for missing targeted update or invalid-candidate evidence.

- [ ] **Step 3: Add only the missing branch persistence behavior**

Ensure TSM writes grounding evidence and no training job, PF includes only selected
expert, and WMF names only Ctrl-World. Preserve candidate error text without
including backend secrets.

- [ ] **Step 4: Verify all diagnosis paths**

Run: `uv run pytest tests/integration/test_pipeline_diagnoses.py -v`

Expected: all parameterized branches and candidate failures pass.

- [ ] **Step 5: Commit diagnosis integration**

```bash
git add src/evoworld/pipeline.py tests/integration/test_pipeline_diagnoses.py
git commit -m "test: cover EvoWorld diagnosis branches"
```

### Task C11: Interruption and resume

**Files:**
- Modify: `src/evoworld/pipeline.py`
- Modify: `src/evoworld/storage.py`
- Test: `tests/integration/test_resume.py`

**Interfaces:**
- Produces: stage-aware `resume` with exactly-once append and store behavior.

- [ ] **Step 1: Write fail-after-stage resume tests**

```python
@pytest.mark.parametrize("failed_stage", list(RoundStage)[:-1])
def test_resume_does_not_repeat_completed_stage(tmp_path: Path, failed_stage: RoundStage) -> None:
    pipeline, calls = make_fail_once_pipeline(tmp_path, failed_stage)
    with pytest.raises(ScriptedInterruption):
        pipeline.run(1)
    persisted_stage = pipeline.store.read_state().stage
    resumed = pipeline.resume()
    assert resumed is not None
    assert calls.completed_stages.count(persisted_stage) == 1
    assert unique_jsonl_ids(pipeline.store.run_dir / "memory.jsonl")
```

- [ ] **Step 2: Verify resume tests fail**

Run: `uv run pytest tests/integration/test_resume.py -v`

Expected: FAIL due to repeated stages or absent deserialization.

- [ ] **Step 3: Restore completed outputs instead of recomputing**

Deserialize stage artifacts into domain values, persist round seed and config digest,
and check deterministic record IDs before task/memory/verified writes. Completed-run
resume returns without invoking adapters.

- [ ] **Step 4: Verify resume and regression suite**

Run: `uv run pytest tests/integration/test_resume.py tests/integration/test_pipeline_success.py tests/integration/test_pipeline_diagnoses.py -v`

Expected: all resume and pipeline tests pass with no duplicate IDs.

- [ ] **Step 5: Commit resume support**

```bash
git add src/evoworld/pipeline.py src/evoworld/storage.py tests/integration/test_resume.py
git commit -m "feat: resume interrupted EvoWorld rounds"
```

### Task C12: Core CLI composition

**Files:**
- Create: `src/evoworld/cli.py`
- Create: `src/evoworld/__main__.py`
- Test: `tests/integration/test_cli.py`

**Interfaces:**
- Produces: `build_parser()` and `main(argv=None) -> int`; commands `validate`, `run`, `resume`.

- [ ] **Step 1: Write failing CLI tests**

```python
def test_dry_run_cli_completes_one_round(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = materialize_dry_config(tmp_path)
    assert main(["run", "--config", str(config_path), "--rounds", "1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["completed_rounds"] == 1


def test_real_mode_never_falls_back_to_deterministic(tmp_path: Path) -> None:
    config_path = materialize_real_config_without_backend(tmp_path)
    assert main(["run", "--config", str(config_path), "--rounds", "1"]) == 2
```

- [ ] **Step 2: Verify CLI tests fail**

Run: `uv run pytest tests/integration/test_cli.py -v`

Expected: FAIL importing `evoworld.cli`.

- [ ] **Step 3: Implement CLI composition and JSON results**

Use `argparse`, return 0 for success, 2 for validation errors, and 1 for runtime
stage errors. `validate` performs no backend training. `run` composes deterministic
adapters only when `mode = "dry_run"`; otherwise require registered real factories.

- [ ] **Step 4: Run the complete core quality gate**

Run: `uv run pytest tests/unit tests/integration -v && uv run ruff check . && uv run mypy src && uv run evoworld validate --config configs/dry_run.toml && uv run evoworld run --config configs/dry_run.toml --rounds 1`

Expected: all tests and checks exit zero; CLI prints one completed round and an
absolute run directory.

- [ ] **Step 5: Commit the core CLI**

```bash
git add src/evoworld/cli.py src/evoworld/__main__.py tests/integration/test_cli.py
git commit -m "feat: expose the EvoWorld core CLI"
```

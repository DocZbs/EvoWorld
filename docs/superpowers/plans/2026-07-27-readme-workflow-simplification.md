# README and Workflow Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a concise repository landing page and compact GitHub Actions while preserving all documented commands, quality gates, and Pages deployment.

**Architecture:** README remains a short index into focused documentation rather than duplicating backend details. CI and Pages keep their existing security and behavioral contracts but collapse presentation-only YAML. Tests assert user-visible commands, links, triggers, permissions, actions, and compactness.

**Tech Stack:** Markdown, GitHub Actions YAML, pytest, PyYAML, uv, Git.

## Global Constraints

- Keep README at 70 lines or fewer.
- Keep both `.github/workflows/ci.yml` and `.github/workflows/pages.yml`.
- Preserve all existing CI quality commands and site-only Pages deployment.
- Do not modify core pipeline, backend, training, submodule, dependency, or site files.
- Do not add a filename containing `paper`.
- Publish only to remote `main`; do not create a remote backup branch or tag.
- Rewrite remote history only with an exact `--force-with-lease`.

---

### Task 1: Concise README

**Files:**
- Modify: `README.md`
- Modify: `tests/integration/test_cli.py`

**Interfaces:**
- Consumes: Existing CLI commands and focused documents under `docs/`.
- Produces: A README of at most 70 lines that retains all public command and link contracts.

- [ ] **Step 1: Add a failing compactness test**

Add:

```python
def test_readme_is_concise() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert len(readme.splitlines()) <= 70
    for heading in ("## Quick start", "## Fine-tuning", "## Documentation", "## Limitations"):
        assert heading in readme
```

- [ ] **Step 2: Run the README contract and confirm RED**

Run:

```bash
uv run pytest tests/integration/test_cli.py -k readme -v
```

Expected: `test_readme_is_concise` fails because the current README has 152 lines.

- [ ] **Step 3: Replace README with the concise public entry point**

Use this structure and retain the exact strings already required by the parametrized
command/link tests:

````markdown
# EvoWorld

EvoWorld is a world-model-centric framework for routing modular robot policies,
simulating candidate actions, executing a selected policy, and using Qwen3-VL
diagnosis to select targeted updates.

The public repository includes deterministic orchestration, pinned Ctrl-World,
OpenPI, and Isaac GR00T submodules, backend bridges, data contracts, and plan-first
fine-tuning. Real experiments require private data, checkpoints, calibration, and
robot infrastructure.

## Quick start

```bash
git clone --recursive https://github.com/DocZbs/EvoWorld.git
cd EvoWorld
git submodule update --init --recursive
uv sync --python 3.11 --extra dev
uv run python scripts/verify_submodules.py
uv run evoworld validate --config configs/dry_run.toml
uv run evoworld run --config configs/dry_run.toml --rounds 1
uv run evoworld resume --run-dir artifacts/runs/<run-id>
```

The method follows **Task Generation**, **Contextual Scene Anchoring**,
**Strategic Routing and Imaginative Simulation**, and **VLM Diagnosis and
Recursive Evolution**.

## Fine-tuning

Fine-tuning is explicit and plan-first:

```bash
cp configs/finetune/pi05.example.toml /absolute/path/pi05.toml
uv run evoworld finetune plan --config /absolute/path/pi05.toml --target pi05
uv run evoworld finetune execute --job <job-from-plan-json>
```

Supported targets are `pi05`, `pi0`, `pi0_fast`, `groot_n16`, `ctrl_world`, and
`qwen_evaluator`. See [fine-tuning](docs/fine_tuning.md) for data and GPU
requirements.

## Documentation

- [Architecture](docs/architecture.md)
- [Data contracts](docs/data.md)
- [Real runtime](docs/backends/real_composition.md)
- [OpenPI](docs/backends/openpi.md), [GR00T](docs/backends/groot.md),
  [Ctrl-World](docs/backends/ctrl_world.md), and [Qwen3-VL](docs/backends/qwen_vl.md)
- [Project article](site/static/pdfs/evoworld.pdf)

## Limitations

The repository does not ship private trajectories, annotations, trained weights,
projector calibration, or robot control code. Deterministic mode runs locally;
real training and execution require the documented Linux/CUDA environments and
operator-provided safety controls.

## Citation

```bibtex
@inproceedings{zhang2026evoworld,
  title={EvoWorld: A World-Model-Centric Framework for Continuous Self-Evolution of Modular Embodied Skills},
  author={Boshi Zhang and Sen Cui and {BaoHuaYin} and Youyi Kou and Junyu Wu and Zuo Pu and Tao Xue and Zhikang Chen and Shanshan Wei and Min Zhang and Miao Liu and Changshui Zhang and {Zhang Tao}},
  booktitle={European Conference on Computer Vision},
  year={2026}
}
```
````

- [ ] **Step 4: Run README tests and confirm GREEN**

Run:

```bash
uv run pytest tests/integration/test_cli.py -k 'readme or public_markdown' -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the README slice**

```bash
git add README.md tests/integration/test_cli.py
git commit -m "docs: make the repository entry point concise"
```

### Task 2: Compact CI and Pages workflows

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/pages.yml`
- Modify: `tests/integration/test_release_verifier.py`
- Modify: `tests/integration/test_site.py`

**Interfaces:**
- Consumes: Existing lightweight test marker and `site/` artifact.
- Produces: Compact workflows with unchanged CI commands and Pages deployment behavior.

- [ ] **Step 1: Update workflow tests for the approved compact contract**

In `test_ci_is_lightweight_locked_and_explicit_about_submodule_coverage`, remove the
presentation-only assertion:

```python
assert "Local-only backend smoke" in workflow
```

Add:

```python
def test_ci_workflow_is_compact() -> None:
    workflow = (PROJECT_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert len(workflow.splitlines()) <= 32
    assert "concurrency:" not in workflow
```

Change the Pages trigger contract to:

```python
assert set(triggers) == {"push"}
```

Remove the `workflow_dispatch` value assertion and its mutation case. Add:

```python
def test_pages_workflow_is_compact() -> None:
    workflow = Path(".github/workflows/pages.yml").read_text(encoding="utf-8")

    assert len(workflow.splitlines()) <= 30
    assert "workflow_dispatch" not in workflow
    assert "concurrency:" not in workflow
```

- [ ] **Step 2: Run workflow tests and confirm RED**

Run:

```bash
uv run pytest tests/integration/test_release_verifier.py -k 'ci_' -v
uv run pytest tests/integration/test_site.py -k 'pages_workflow' -v
```

Expected: compactness and Pages trigger assertions fail against the current workflows.

- [ ] **Step 3: Replace CI with the compact equivalent**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: false
          persist-credentials: false
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true
      - run: uv sync --locked --extra dev
      - run: |
          uv run pytest -m 'not backend'
          uv run ruff check .
          uv run ruff format --check src training integrations scripts tests
          uv run mypy src training integrations
          uv run evoworld validate --config configs/dry_run.toml
          uv run evoworld run --config configs/dry_run.toml --rounds 1
```

- [ ] **Step 4: Replace Pages with the compact equivalent**

```yaml
name: Pages

on:
  push:
    branches: [main]

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: false
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: site
      - uses: actions/deploy-pages@v4
        id: deployment
```

- [ ] **Step 5: Run workflow tests and confirm GREEN**

Run:

```bash
uv run pytest tests/integration/test_release_verifier.py -k 'ci_' -v
uv run pytest tests/integration/test_site.py -k 'pages_workflow' -v
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit the workflow slice**

```bash
git add .github/workflows/ci.yml .github/workflows/pages.yml \
  tests/integration/test_release_verifier.py tests/integration/test_site.py
git commit -m "ci: simplify public automation"
```

### Task 3: Verify, squash, and publish

**Files:**
- Verify: all tracked files and recursive submodules
- Publish: `refs/heads/main`

**Interfaces:**
- Consumes: The verified final Git tree and remote `main` at `6d49601d2c3ec6c8f8bd95af688dc99756dc7066`.
- Produces: A remote `main` with the original root commit plus one release commit.

- [ ] **Step 1: Run the complete release gate**

```bash
PATH=/Users/zbs/Library/Python/3.9/bin:$PATH \
  uv run --python 3.11 python scripts/verify_release.py
uv run python scripts/audit_fidelity.py
git diff --check
git status --porcelain=v1 --ignore-submodules=none
```

Expected: 952 or more tests pass; Ruff, formatting, mypy, config validation,
submodule verification, fidelity audit, and one deterministic round exit zero; the
worktree is clean after the two implementation commits.

- [ ] **Step 2: Create a local-only pre-squash backup**

```bash
git branch backup/pre-squash-20260727 HEAD
```

Expected: the local branch points to the complete granular history and is not pushed.

- [ ] **Step 3: Fetch and verify the exact remote lease**

```bash
git fetch https://github.com/DocZbs/EvoWorld.git main
test "$(git rev-parse FETCH_HEAD)" = "6d49601d2c3ec6c8f8bd95af688dc99756dc7066"
test "$(git rev-list --max-parents=0 FETCH_HEAD)" = \
  "1d6f107724cb7c54f3776bc996c6db5bba71aea5"
```

Expected: remote `main` has not moved and its original root is unchanged.

- [ ] **Step 4: Create one release commit over the original root**

Create a commit from the verified final tree with parent
`1d6f107724cb7c54f3776bc996c6db5bba71aea5` and message:

```text
feat: release the EvoWorld reproduction pipeline

Add the complete deterministic and real composition pipeline, pinned official
submodules, method-aligned fine-tuning bridges, documentation, tests, and compact
GitHub automation.
```

Verify:

```bash
git rev-list --count 1d6f107724cb7c54f3776bc996c6db5bba71aea5..$SQUASHED_SHA
```

Expected: `1`.

- [ ] **Step 5: Force-with-lease only the remote main**

```bash
git push \
  --force-with-lease=refs/heads/main:6d49601d2c3ec6c8f8bd95af688dc99756dc7066 \
  https://github.com/DocZbs/EvoWorld.git \
  "$SQUASHED_SHA:refs/heads/main"
```

Expected: Git reports a forced update of `main`; no other ref is pushed.

- [ ] **Step 6: Verify and align the local checkout**

```bash
git ls-remote --heads https://github.com/DocZbs/EvoWorld.git
git switch -C main "$SQUASHED_SHA"
git fetch origin main
git branch --set-upstream-to=origin/main main
git status -sb --ignore-submodules=none
```

Expected: the only remote head is `main`, its SHA equals `$SQUASHED_SHA`, remote
history has exactly two commits, and the local `main` worktree is clean.

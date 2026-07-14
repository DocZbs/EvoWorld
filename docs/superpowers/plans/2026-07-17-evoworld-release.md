# EvoWorld Release, Documentation, and Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the project page, document the complete public workflow, automate lightweight quality gates, and prove method-to-code fidelity.

**Architecture:** The static site is deployed independently from the Python package. Documentation links every user-visible command to validated configuration, while a machine-checked fidelity matrix prevents a superficially green but incomplete release.

**Tech Stack:** GitHub Pages artifact workflow, Markdown, Python 3.11, pytest, Ruff, mypy, uv.

## Global Constraints

- Existing site content and PDF assets remain available.
- The disabled code button becomes a working repository link.
- Documentation distinguishes public runnable code from unpublished assets.
- Normal CI does not initialize heavy nested submodules or require CUDA/network.
- Final verification is fresh and covers tests, types, lint, dry-run, submodules, site, and fidelity.
- New filenames must not contain `paper`.
- All changes and commits remain local; do not push.

---

### Task R1: Move and deploy the existing project page

**Files:**
- Move: `index.html` to `site/index.html`
- Move: `static/` to `site/static/`
- Move: `.nojekyll` to `site/.nojekyll`
- Create: `.github/workflows/pages.yml`
- Test: `tests/integration/test_site.py`

**Interfaces:**
- Produces: self-contained `site/` and Pages artifact upload from that directory.

- [ ] **Step 1: Write failing site integrity tests**

```python
def test_site_has_no_missing_local_assets() -> None:
    document = Path("site/index.html").read_text(encoding="utf-8")
    for relative in extract_local_asset_paths(document):
        assert (Path("site") / relative).is_file(), relative


def test_code_button_links_to_repository() -> None:
    document = Path("site/index.html").read_text(encoding="utf-8")
    assert "https://github.com/DocZbs/EvoWorld" in document
    assert "Code Soon" not in document
```

- [ ] **Step 2: Verify tests fail before site relocation**

Run: `uv run pytest tests/integration/test_site.py -v`

Expected: FAIL because `site/index.html` does not exist.

- [ ] **Step 3: Move assets and add Pages deployment**

```bash
mkdir -p site
git mv index.html site/index.html
git mv static site/static
git mv .nojekyll site/.nojekyll
```

Update only the code button to a repository link. Pages workflow checks out without
recursive submodules, configures Pages, uploads `site/`, and deploys the artifact on
authorized main pushes or manual dispatch.

- [ ] **Step 4: Verify site integrity and workflow syntax**

Run: `uv run pytest tests/integration/test_site.py -v && python -c 'import pathlib; assert pathlib.Path(".github/workflows/pages.yml").is_file()'`

Expected: all local assets resolve and the code link is enabled.

- [ ] **Step 5: Commit the project-page move**

```bash
git add site .github/workflows/pages.yml tests/integration/test_site.py
git commit -m "chore: isolate the EvoWorld project page"
```

### Task R2: User and backend documentation

**Files:**
- Create: `README.md`
- Create: `docs/architecture.md`
- Create: `docs/data.md`
- Create: `docs/fine_tuning.md`
- Modify: `tests/integration/test_cli.py`

**Interfaces:**
- Produces: complete setup, dry-run, real-run, data, training, and limitation guide.

- [ ] **Step 1: Write a failing documentation command check**

Add to `tests/integration/test_cli.py`:

```python
@pytest.mark.parametrize("command", [
    "uv run evoworld validate --config configs/dry_run.toml",
    "uv run evoworld run --config configs/dry_run.toml --rounds 1",
    "uv run evoworld finetune plan --config configs/reproduction.toml --target pi05",
])
def test_readme_contains_public_commands(command: str) -> None:
    assert command in Path("README.md").read_text(encoding="utf-8")
```

- [ ] **Step 2: Verify the check fails before README exists**

Run: `uv run pytest tests/integration/test_cli.py -k readme -v`

Expected: FAIL with missing `README.md`.

- [ ] **Step 3: Write concise, source-linked documentation**

README covers clone with recursive submodules, core install, dry-run, architecture,
real backend environment links, data/weight prerequisites, all training targets, and
citation. Architecture maps four method stages and ports. Data specifies canonical
manifests and artifacts. Fine-tuning records exact appendix knobs and source-linked
official commands. A limitations section in README lists every unpublished asset and
states that public Qwen settings are a baseline rather than the exact private recipe.

- [ ] **Step 4: Verify documentation and links**

Run: `uv run pytest tests/integration/test_cli.py -k 'readme or command' -v && rg -n 'Ctrl-World|openpi|GR00T|Qwen3-VL|TSM|PF|WMF|5,000' README.md docs`

Expected: command tests pass and every major component appears in the documentation.

- [ ] **Step 5: Commit user documentation**

```bash
git add README.md docs/architecture.md docs/data.md docs/fine_tuning.md tests/integration/test_cli.py
git commit -m "docs: explain the complete EvoWorld workflow"
```

### Task R3: CI and complete local quality gate

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `scripts/verify_release.py`
- Test: `tests/integration/test_release_verifier.py`

**Interfaces:**
- Produces: lightweight CI and one local release verification entry point.

- [ ] **Step 1: Write failing release-verifier tests**

```python
def test_release_verifier_runs_every_lightweight_gate() -> None:
    runner = RecordingRunner()
    verify_release(runner)
    assert runner.commands == [
        ("uv", "run", "pytest"),
        ("uv", "run", "ruff", "check", "."),
        ("uv", "run", "mypy", "src", "training", "integrations"),
        ("uv", "run", "evoworld", "validate", "--config", "configs/dry_run.toml"),
        ("uv", "run", "evoworld", "run", "--config", "configs/dry_run.toml", "--rounds", "1"),
    ]
```

- [ ] **Step 2: Verify the test fails before the verifier exists**

Run: `uv run pytest tests/integration/test_release_verifier.py -v`

Expected: FAIL importing `verify_release`.

- [ ] **Step 3: Implement fail-fast local and CI gates**

`verify_release.py` executes argv arrays in order and returns the first nonzero exit.
CI uses Python 3.11 and uv, does not recurse submodules, runs normal tests/lint/types,
and runs one deterministic round. Backend smoke tests remain a separate manual job.

- [ ] **Step 4: Verify the verifier and workflow content**

Run: `uv run pytest tests/integration/test_release_verifier.py -v && rg -n 'pytest|ruff|mypy|dry_run' .github/workflows/ci.yml scripts/verify_release.py`

Expected: tests pass and CI contains all lightweight gates.

- [ ] **Step 5: Commit release automation**

```bash
git add .github/workflows/ci.yml scripts/verify_release.py tests/integration/test_release_verifier.py
git commit -m "ci: verify the lightweight EvoWorld release"
```

### Task R4: Method-to-code fidelity audit

**Files:**
- Create: `docs/fidelity.md`
- Create: `docs/limitations.md`
- Create: `scripts/audit_fidelity.py`
- Test: `tests/integration/test_fidelity_audit.py`
- Modify: `tasks/todo.md`

**Interfaces:**
- Produces: machine-readable requirement markers and final human evidence matrix.

- [ ] **Step 1: Write a failing audit-completeness test**

```python
REQUIREMENTS = {
    "scene-anchoring", "joint-routing-score", "vlm-diagnosis", "verified-buffer",
    "tsm-update", "pf-update", "wmf-update", "pi05-config", "pi0-config",
    "pi0-fast-config", "groot-n16-config", "ctrl-world-training", "qwen-evaluator",
    "resume", "submodule-pins", "dry-run", "site",
}


def test_fidelity_matrix_covers_every_requirement() -> None:
    matrix = load_fidelity_matrix(Path("docs/fidelity.md"))
    assert set(matrix) == REQUIREMENTS
    assert all(row.code and row.config and row.test and row.documentation for row in matrix.values())
```

- [ ] **Step 2: Verify the audit fails before the matrix exists**

Run: `uv run pytest tests/integration/test_fidelity_audit.py -v`

Expected: FAIL because `docs/fidelity.md` is missing.

- [ ] **Step 3: Map every method and success criterion to evidence**

Create one table row per marker with exact source section, code path/symbol, config
key, test node, and documentation link. `docs/limitations.md` records every missing
private asset and the public-baseline boundary. `audit_fidelity.py` validates markers
and that every referenced local file exists; it does not treat existence alone as
proof of runtime behavior.

- [ ] **Step 4: Run the fresh full completion audit**

Run: `uv run python scripts/verify_release.py && uv run python scripts/verify_submodules.py && uv run python scripts/audit_fidelity.py && git diff --check && git status --short`

Expected: every command exits zero; no unstaged implementation changes remain; the
only divergence from `origin/main` is the intended local commit series.

- [ ] **Step 5: Commit the fidelity evidence**

```bash
git add docs/fidelity.md docs/limitations.md scripts/audit_fidelity.py tests/integration/test_fidelity_audit.py tasks/todo.md
git commit -m "docs: audit EvoWorld method fidelity"
```

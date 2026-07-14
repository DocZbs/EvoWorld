# Implementation Plan: EvoWorld Public Reproduction

## Overview

Implement the approved EvoWorld design as a lightweight Python 3.11 orchestration
core, three pinned upstream Git submodules, isolated real-backend bridges, complete
expert/world-model/evaluator fine-tuning launchers, deterministic end-to-end tests,
documentation, and a deployable project page.

The detailed task instructions are split by independent subsystem:

1. [Core pipeline](../docs/superpowers/plans/2026-07-17-evoworld-core.md)
2. [Upstream integrations](../docs/superpowers/plans/2026-07-17-evoworld-integrations.md)
3. [Training and evaluator](../docs/superpowers/plans/2026-07-17-evoworld-training.md)
4. [Release, documentation, and audit](../docs/superpowers/plans/2026-07-17-evoworld-release.md)

## Architecture Decisions

- Domain modules depend only on typed ports; no domain file imports a GPU-heavy
  submodule.
- Real backends run in their own environments. EvoWorld adapters use official
  WebSocket/ZeroMQ clients where available and JSONL process bridges elsewhere.
- Ctrl-World, openpi, and Isaac-GR00T remain unmodified and are pinned to the
  revisions approved in the design.
- EvoWorld bridge programs may compose official upstream Python components when a
  pinned CLI does not expose required method settings. Every override is recorded
  and contract-tested.
- Dry-run and real runs use the same pipeline state machine. Missing real assets
  never cause fallback to deterministic adapters.
- Training is plan-first. Execution requires an explicit command and validates
  manifests, submodule revisions, and idempotency.
- Qwen3-VL fine-tuning is delivered as a public `ms-swift==4.3.1` baseline. The
  private annotations, prompt, hyperparameters, and checkpoint used for reported
  results are explicitly not claimed.

## Dependency Graph

```text
Core C1-C2: package + domain interfaces
    |
    +--> Core C3-C8: config/storage/math/evolution/deterministic adapters
    |       |
    |       +--> Core C9-C12: pipeline, failures, resume, CLI
    |
    +--> Integrations I1-I3: submodules, revision audit, process protocol
    |       |
    |       +--> Integrations I4-I7: openpi, GR00T, Ctrl-World, Qwen adapters
    |
    +--> Training T1-T2: job schema + diagnosis-aware curation
            |
            +--> Training T3-T6: four backend training lanes
                    |
                    +--> Training T7: unified fine-tune CLI

Core C12 + Integrations I7 + Training T7
    |
    +--> Release R1-R4: site, documentation, CI, fidelity audit
```

## Parallel Execution Lanes

After C1 and C2 establish shared interfaces, the following lanes may use separate
local worktrees and local-only branches:

- Lane A: C3-C12, the orchestration core.
- Lane B: I1-I7, pinned submodules and runtime adapters.
- Lane C: T1-T7, curation and all fine-tuning paths.

R1 may run independently after the plan is approved. R2-R4 wait for the three code
lanes. No worktree or branch is pushed; `origin` keeps a disabled push URL.

## Verification Checkpoints

### Foundation

- `uv sync --extra dev` succeeds.
- Domain and training schema tests pass without CUDA, network, or initialized
  submodule imports.
- `git submodule status --recursive` matches the approved revisions.

### Core and adapters

- `uv run pytest tests/unit tests/contract` succeeds.
- `uv run ruff check .` succeeds.
- `uv run mypy src training integrations` succeeds.
- Every real adapter builds a command or client request from an offline fixture.

### End-to-end

- `uv run evoworld validate --config configs/dry_run.toml` exits zero.
- `uv run evoworld run --config configs/dry_run.toml --rounds 1` exits zero.
- Successful, TSM, PF, and WMF scenarios produce the expected artifacts.
- Resume does not duplicate tasks, memory records, verified data, or training jobs.

### Release

- The full test, lint, type-check, dry-run, site-link, submodule, and fidelity-audit
  commands run fresh and exit zero.
- Every design success criterion maps to concrete code, configuration, tests, and
  documentation in `docs/fidelity.md`.

## Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| openpi CLI forces `shuffle=True` | Appendix mismatch | Adapter-owned bridge composes pinned upstream loader/trainer with `shuffle=False`; contract test asserts it |
| Ctrl-World CLI hides output and step budget | Unsafe 500k-step run or missing 5k checkpoint | Bridge stages a manifest-exact dataset, composes pinned training components, counts 5k synchronized optimizer updates, and forces the final checkpoint |
| Ctrl-World rollout scripts are preset-oriented | Arbitrary task/scene integration may fail | Define a narrow 7-D action bridge and validate artifact/layout contracts before launch |
| Qwen evaluator recipe is unpublished | Exact result reproduction is impossible | Ship a fully explicit public baseline and label private values as unavailable |
| Nested submodules add clone cost | Setup can fail or become dirty | Recursive init command, exact SHA audit, and offline normal tests |
| Multiple agents edit shared contracts | Merge conflicts and type drift | Complete C1-C2 first; publish signatures; use isolated local worktrees for later lanes |
| Resume duplicates append-only data | Corrupted training evidence | Deterministic IDs, idempotent append/store operations, and fail-after-stage tests |
| Heavy backend imports break CI | Lightweight checks become unusable | Lazy adapters, process boundaries, and separately marked backend smoke tests |

## Open Questions

None. Real training remains plan-only until the user supplies data, checkpoints,
hardware, and explicit evaluator recipe values.

# EvoWorld Implementation Checklist

## Core pipeline

- [x] C1: Package and quality-tool scaffold
- [x] C2: Immutable domain contracts and ports
- [x] C3: TOML configuration and resolved manifests
- [x] C4: Atomic run storage and experience memory
- [x] C5: Contextual scene anchoring
- [x] C6: Memory-aware routing and imagined-rollout selection
- [x] C7: Diagnosis-driven evolution decisions
- [x] C8: Deterministic adapters
- [x] C9: Successful four-stage pipeline
- [x] C10: Diagnosis branches and candidate failure isolation
- [x] C11: Interruption and resume
- [x] C12: Core CLI composition

## Upstream integrations

- [x] I1: Add and pin three recursive Git submodules
- [x] I2: Verify submodule revisions and cleanliness
- [x] I3: JSONL process protocol
- [x] I4: openpi policy service adapter
- [x] I5: GR00T-N1.6 policy service adapter
- [x] I6: Ctrl-World rollout bridge
- [x] I7: Qwen3-VL evaluator bridge

## Training and evaluator

- [x] T1: Canonical training manifest and job plans
- [x] T2: Diagnosis-aware curation and leakage-safe splits
- [x] T3: openpi expert configurations and no-shuffle bridge
- [x] T4: GR00T-N1.6 fine-tuning launcher
- [x] T5: Ctrl-World preparation and bounded post-training bridge
- [x] T6: Qwen3-VL evaluator schema, export, and public fine-tuning baseline
- [x] T7: Unified plan-first fine-tuning CLI

## Release and audit

- [x] R1: Move and deploy the existing project page
- [x] R2: User and backend documentation
- [x] R3: CI and complete local quality gate
- [x] R4: Method-to-code fidelity audit

## Final evidence

- [x] Full tests pass
- [x] Ruff passes
- [x] mypy passes
- [x] End-to-end dry-run passes
- [x] All submodule revisions match
- [x] Site links and deploy workflow validate
- [x] All design success criteria have direct evidence
- [x] Remote repository remains unchanged unless separately authorized

The full-test and recursive-submodule gates were checked only after the final
convergence worktree ran them with every nested checkout initialized. Real robot
and CUDA smoke runs are declared limitations, not silently treated as public
software success gates.

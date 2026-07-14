# Architecture

EvoWorld keeps method control in a small Python core and runs heavyweight backends
through explicit adapters. The core owns ordering, typed contracts, provenance, and
durability; it does not import JAX, Torch, Ctrl-World, GR00T, or Qwen3-VL in a normal
dry run.

## Four article stages

The method names and order are:

1. **Task Generation** creates or selects the embodied task specification.
2. **Contextual Scene Anchoring** binds that task to a concrete scene and context.
3. **Strategic Routing and Imaginative Simulation** shortlists compatible experts,
   obtains candidate action chunks, imagines them with Ctrl-World, and selects by
   the joint routing/world-model score.
4. **VLM Diagnosis and Recursive Evolution** evaluates the observed trajectory and
   directs the next `TSM`, `PF`, `WMF`, or `NONE` response.

Physical execution connects stages 3 and 4 so the VLM can diagnose an observed
trajectory. It is an operational boundary in this implementation, not a fifth stage
or a renamed article stage.

## Durable execution mapping

| Code checkpoint | Article-stage mapping | Primary evidence |
| --- | --- | --- |
| Grounding | Task Generation + Contextual Scene Anchoring | `grounding.json` with the exact `TaskSpec` and `SceneRecord` |
| Candidate selection | Strategic Routing and Imaginative Simulation | `candidates.json` with router score, world-model score, joint score, and failure isolation |
| Physical execution boundary | Connects imaginative simulation to VLM diagnosis | `trajectory.json` with task, scene, expert, projected action, and artifact provenance |
| Diagnosis/update | VLM Diagnosis and Recursive Evolution | `diagnosis.json`, `update.json`, verified artifacts, and `memory.jsonl` |

The routing stage is memory-aware. Semantic preference and successful experience
for the same task and scene produce the expert order. Each candidate then receives
an independent action, imagined rollout, and score. Invalid candidates are retained
as explicit failures; a deterministic expert-ID tie break selects the highest valid
`joint_score`.

At or above the configured reliability threshold, the evaluator must return
`NONE`, no component is trained, and the verified trajectory may enter memory.
Below the threshold the label selects exactly one response:

| Diagnosis | Update target |
| --- | --- |
| `TSM` | task/scene grounding and router memory |
| `PF` | selected expert policy |
| `WMF` | Ctrl-World world model |
| `NONE` | no update |

## Ports

`src/evoworld/ports.py` is the framework boundary:

| Port | Stable operation |
| --- | --- |
| `TaskSource` | `next_task(operation_id=...)` |
| `SceneAnchor` | `anchor(task, operation_id=...)` |
| `Router` | `shortlist(task, scene, memory, operation_id=...)` |
| `ExpertPolicy` | `act(task, observation, operation_id=...)` |
| `WorldModel` | `imagine(task, scene, expert, actions, operation_id=...)` |
| `RolloutScorer` | `score(task, rollout, operation_id=...)` |
| `Executor` | `execute(task, scene, expert, actions, operation_id=...)` |
| `TrajectoryEvaluator` | `evaluate(task, scene, trajectory, memory, operation_id=...)` |
| `Trainer` | `plan(update, operation_id=...)` and `execute(plan, operation_id=...)` |
| `ExperienceMemory` | exact-context lookup and idempotent append |
| `RunStore` | claims, state, evidence artifacts, and verified trajectories |

Dry mode composes deterministic implementations of every port. Real mode injects
the public raw openpi and GR00T policies, library-owned audited projection with
private calibration assets, Ctrl-World, and Qwen3-VL, and requires a private
provider for task sourcing, anchoring, rollout scoring, robot execution, training,
memory, and storage. The provider cannot replace these audited backends or their
library-owned projection wrappers. Raw policy chunks are projected to finite,
timestamp-preserving
`cartesian_pose_7d` before Ctrl-World or physical execution. The projector code is a
fixed repository-owned implementation; deployment configuration, calibration, and
weights are content-bound, and no fallback or dynamically imported projector is
supplied.

## Durable operations and resume

Before an external call, the pipeline persists an `OperationIntent` containing a
stable operation ID, operation kind, canonical finite-JSON input, and input digest.
After a validated result is durably incorporated, it appends the matching
`OperationReceipt`. IDs are derived from round, kind, and expert where applicable,
so retries can bind to the same canonical input.

This is honest at-least-once delivery, not a claim of universal exactly-once
hardware behavior. Adapters may replay only when their own durable receipt proves
the first call completed with identical bound bytes. Reusing an ID with different
input raises an operation conflict. A claimed physical operation with no provable
result raises `OperationInDoubtError` and persists the `in_doubt` stage for operator
resolution.

`JsonRunStore` uses atomic writes, per-round claims, monotonic revisions, and
compare-and-swap state updates. `resume` reads the durable run manifest and latest
incomplete state; it rejects configuration, factory, runtime-manifest, or static-
asset drift before calling a backend.

## Real process boundary

`evoworld.real_composition:build_real_adapters` reads the strict manifest described
in [backends/real_composition.md](backends/real_composition.md). Process adapters use
argv arrays and structured requests rather than shell interpolation. Every adapter
validates revision, checkpoint identity, media containment, input byte manifests,
response schema, and output artifacts. Backend diagnostics are kept out of protocol
stdout.

The upstream code is pinned and isolated:

- [openpi](https://github.com/Physical-Intelligence/openpi) serves pi0.5, pi0, and
  pi0-FAST policies and supplies their fine-tuning APIs.
- [Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) supplies GR00T-N1.6 policy
  service and training entry points.
- [Ctrl-World](https://github.com/Robert-gyj/Ctrl-World) generates action-conditioned
  video rollouts and supplies the world-model components.
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) evaluates ordered trajectory frames
  and returns strict `{score,label,rationale}` JSON.

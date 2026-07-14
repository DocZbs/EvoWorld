# Reproduction boundaries and limitations

The public repository is intentionally fail-closed at every private boundary. It
reproduces orchestration, schemas, provenance checks, deterministic control flow,
configuration contracts, and plan construction. It does not manufacture unavailable
data, weights, calibration, compute, or physical evidence.

## Private inputs and assets

- DROID-derived and robot-collected trajectories are not redistributed. The exact
  robot trajectory set used by the reported experiments remains private.
- Manual Qwen3-VL diagnosis annotations are private. Public evaluator fixtures test
  the schema and leakage rules; they are not the experiment annotation corpus.
- Trained expert, evaluator, and world-model checkpoints are not distributed.
  Ctrl-World base assets, starting weights, and any model-host credentials or access
  approvals must be supplied by the operator.
- Runtime action projection uses the schema-version 2 projector manifest. Each
  deployment must supply its private projector configuration, calibration data, and
  calibration weights with the declared content hashes. The repository-owned
  projector code does not infer or invent those values.

## Real execution and compute

- A robot deployment needs robot calibration, a deployment-specific executor,
  controller integration, collision limits, emergency stops, and an operator-owned
  safety case. The deterministic executor is not a hardware safety controller.
- The backends require separate Linux CUDA environments because their upstream
  dependency stacks conflict. GR00T-N1.6's checked configuration requires eight
  GPUs; this is not silently reduced to fit a workstation.
- Real robot and CUDA smoke tests were not completed on this local macOS machine.
  Contract tests validate commands, bindings, revisions, and outputs, but they are
  not evidence that a GPU kernel ran or a robot moved.

## Evaluator recipe boundary

The checked Qwen3-VL 4B LoRA configuration is a public, reproducible baseline. It is
not the unpublished exact evaluator recipe, manual annotation set, hyperparameter
search, or trained checkpoint used for the reported results. The repository does
not claim numerical equivalence between the public baseline and that private run.

## What the public checks establish

The deterministic loop can execute all four method stages locally, persist
candidate scores and operation receipts, diagnose deterministic trajectories,
select `TSM`, `PF`, `WMF`, or `NONE`, append verified records, and resume without
silently replaying ambiguous physical effects. The plan-first fine-tuning gates can
validate manifests, revisions, source and asset hashes, exact argv arrays,
environment allowlists, output locations, retry receipts, and target-specific
configuration before a backend launch.

Those checks cannot claim the reported trained weights, real-world success rates,
world-model visual quality, evaluator accuracy, GPU throughput, robot safety, or
hardware reproducibility. Passing the public quality gate proves the checked
software contracts against public fixtures; it does not replace private assets or a
real deployment run.

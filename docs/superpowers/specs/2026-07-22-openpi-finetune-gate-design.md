# OpenPI Fine-Tuning Gate Design

**Status:** Approved for implementation on 2026-07-22

## Context

EvoWorld already plans and executes method-aligned OpenPI fine-tuning jobs, records
immutable `job.json` plans, and publishes a successful `execution.json` receipt that
binds the produced checkpoint bytes. Real composition currently accepts only an
OpenPI checkpoint identity and digest. It verifies that identity with the serving
process, but it does not prove that the checkpoint came from a completed EvoWorld
fine-tuning job.

For method-aligned reproduction, OpenPI fine-tuning is a prerequisite rather than an
optional deployment convention. A real run must therefore fail before any backend or
private provider is called unless its OpenPI checkpoint is bound to a valid completed
fine-tuning job.

## Goals

- Make successful OpenPI fine-tuning a hard prerequisite for every real run.
- Bind the real runtime to the exact immutable job, completion receipt, target,
  upstream OpenPI revision, and checkpoint output.
- Revalidate the binding when a run is resumed so provenance drift cannot be hidden.
- Keep deterministic dry runs independent of private datasets, weights, and CUDA.
- Reject a dirty OpenPI checkout before a fine-tuning plan is created or executed.
- Keep the official OpenPI submodule unmodified and keep all work local unless a
  separate push is authorized.

## Non-Goals

- Downloading DROID data or model weights.
- Running an expensive CUDA fine-tuning job during repository verification.
- Treating an arbitrary externally trained checkpoint as method-aligned reproduction.
- Adding a compatibility bypass for legacy real-runtime manifests.

## Decision

### Runtime manifest schema 3

The strict runtime manifest advances from schema 2 to schema 3. The `openpi` object
keeps its service and checkpoint fields and gains one exact `finetune` object:

```json
{
  "schema_version": 3,
  "openpi": {
    "expert_id": "pi05",
    "python": "/opt/envs/openpi/bin/python",
    "model": "pi05",
    "host": "127.0.0.1",
    "port": 8000,
    "timeout_seconds": 5.0,
    "checkpoint_identity": "/srv/checkpoints/evoworld_pi05/.../4999",
    "checkpoint_digest": "<service checkpoint digest>",
    "finetune": {
      "job_path": "/srv/jobs/openpi-pi05-.../job.json",
      "job_file_sha256": "<sha256 of exact job.json bytes>",
      "receipt_path": "/srv/jobs/openpi-pi05-.../execution.json",
      "receipt_file_sha256": "<sha256 of exact execution.json bytes>"
    }
  }
}
```

All four fine-tuning fields are mandatory. Paths must be absolute, canonical,
non-symlink regular files. The receipt must be the canonical `execution.json` next to
the named `job.json`. File digests bind the runtime manifest to the bytes used during
validation and are included in the canonical runtime digest.

Schema 1 and schema 2 manifests fail with an actionable migration error. There is no
boolean or documentation-only escape hatch because either would preserve the current
gap.

### Completion verifier

The fine-tuning module exposes one read-only completion verifier built from the same
strict parsers used by `finetune execute`. It verifies, in order:

1. the job and receipt paths and their manifest-declared file digests;
2. the exact immutable job schema and internal job/plan hashes;
3. a completed, zero-exit receipt with a valid self-digest, logs, and contiguous
   attempt history;
4. the requested target equals the runtime OpenPI expert ID;
5. the plan upstream revision equals the repository's pinned OpenPI revision;
6. the completed output list contains exactly the configured OpenPI checkpoint
   identity; and
7. the current checkpoint bytes still match the output fingerprint in the receipt.

The verifier returns immutable pure-value provenance for composition. Real
composition adds its digests to `RuntimeExpertMetadata` and runs this validation
before constructing adapters, importing the deployment-private provider, or creating
backend artifact directories.

The OpenPI policy service remains an independent second boundary: it must report the
same checkpoint identity and configured service digest after loading the checkpoint.
The receipt gate proves training provenance; the service handshake proves the live
model identity.

### Resume behavior

The schema-3 manifest, job bytes, receipt bytes, and checkpoint fingerprint contribute
to the runtime identity. Resume rebuilds real composition from the same manifest and
repeats completion verification. Changes to any bound artifact fail closed before an
external backend operation.

### OpenPI checkout cleanliness

OpenPI fine-tuning validation continues to require the pinned commit and required
layout. It additionally runs a recursive, porcelain Git status check and rejects
tracked changes, untracked files, or dirty nested submodules. Tests inject a status
reader where a synthetic checkout is needed; production uses Git directly.

## Error Handling

Every missing, malformed, stale, symlinked, mismatched, failed, or tampered provenance
artifact raises a configuration-level error before GPU, service, provider, or robot
code runs. Error messages name the failed binding without printing secrets or backend
environment values.

Legacy schema errors explain that schema 3 and an EvoWorld OpenPI completion receipt
are required. A checkpoint that is valid for inference but lacks this provenance is
explicitly rejected as unsuitable for method-aligned real execution.

## Documentation Changes

- The README real-run instructions point checkpoint and fine-tuning provenance fields
  to the runtime JSON rather than `real.toml`.
- The fine-tuning guide shows the required order: plan, execute, then bind the emitted
  job and receipt to the schema-3 runtime manifest.
- The real-composition guide documents every new field and the schema-2 migration.
- Limitations continue to state that the public repository does not contain trained
  weights, private data, or evidence of completed real hardware experiments.

## Test Strategy

TDD starts with failures for:

- a real manifest with no fine-tuning binding;
- schema 2 migration;
- missing, symlinked, moved, or digest-mismatched job and receipt files;
- failed, incomplete, foreign-target, wrong-revision, or recomputed/tampered receipts;
- a receipt whose output is not the runtime checkpoint;
- checkpoint bytes changed after completion;
- a dirty OpenPI checkout during planning; and
- provider/backend factories being untouched when the gate fails.

Positive integration coverage creates a small synthetic completed OpenPI job and
proves that schema-3 composition reaches the existing trusted adapter boundary. The
complete repository release gate then runs all tests, Ruff, formatting, mypy,
configuration validation, submodule verification, method-fidelity audit, and the
deterministic end-to-end dry run.

## Acceptance Criteria

1. No schema-3 real run can begin without a verified completed OpenPI fine-tuning
   provenance binding.
2. The bound completion target, pinned revision, output path, and current checkpoint
   bytes match the configured live OpenPI checkpoint.
3. Tampering or drift fails before a provider, backend process, or robot executor is
   called, including on resume.
4. Planning rejects dirty recursive OpenPI checkouts.
5. Public documentation describes the executable order and actual runtime fields.
6. Deterministic mode remains runnable without CUDA or private assets.
7. The full release gate passes and the remote repository remains unchanged.

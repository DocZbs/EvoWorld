# README and Workflow Simplification

**Status:** Approved for implementation on 2026-07-27

## Goal

Make the repository landing page and GitHub automation easier to scan without
removing supported commands, quality gates, or project-page deployment.

## README

Reduce `README.md` from 152 lines to roughly 50–60 lines. Keep only:

- a short project description and public/private boundary;
- recursive clone and Python 3.11 installation;
- deterministic validation and one-round smoke commands;
- the required plan-then-execute fine-tuning order;
- links to architecture, data, fine-tuning, real-runtime, and limitation docs; and
- the citation.

Move detailed backend settings, exact submodule revisions, runtime-manifest
explanations, and training caveats behind the existing focused documentation links.
The concise README must retain every command and documentation path enforced by the
public CLI tests.

## GitHub Actions

Keep both workflows because they provide distinct remote-only behavior.

`ci.yml` retains:

- pushes to `main` and pull requests as triggers;
- read-only repository permission;
- checkout without heavyweight submodules;
- Python 3.11, uv, and locked development dependency setup;
- non-backend tests, Ruff, formatting, mypy, config validation, and one dry run.

It removes concurrency configuration, explanatory comments, verbose step names, and
separate command steps where one grouped quality-gate step is equally clear.

`pages.yml` retains:

- pushes to `main` as the only trigger;
- the minimum Pages permissions and environment;
- checkout without submodules;
- configuration, upload of only `site/`, and deployment.

It removes manual dispatch, concurrency configuration, and redundant step names.

## Tests and Boundaries

Update tests only where they assert removed presentation details such as
`workflow_dispatch`, concurrency, or individual step names. Preserve behavioral
assertions for triggers, permissions, commands, submodule exclusion, and the
site-only artifact path.

No core pipeline, backend adapter, training code, dependency, submodule, site asset,
or filename containing `paper` changes. Run targeted README/workflow tests followed
by the complete release verifier. Commit and push only after both pass.

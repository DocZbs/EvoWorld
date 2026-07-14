# Method-to-code fidelity

This matrix is an auditable index, not a claim that a path exists. Each row points
to a Python symbol found through AST parsing, a concrete configuration key or
section, a pytest nodeid that is collectable as written, and a local documentation
anchor. `scripts/audit_fidelity.py` enforces the exact marker set and rejects extra
columns, duplicate rows, unsafe paths, remote links, and unverifiable references.

The audit's collection check proves that a nodeid is addressable. **Runtime evidence
exists only when that test passes** in the quality gate; neither file existence nor
collection alone proves backend execution, GPU execution, or a robot result. The
test column says exactly which executable assertion supplies evidence for each
claim.

`public` means the referenced implementation, configuration, and deterministic or
contract test are in this repository. `private-required` means the public boundary
and tests are present, but reproducing the reported real result still needs the
private assets or hardware listed in [limitations](limitations.md#private-inputs-and-assets).

| Marker | Article/method section | Code file + symbol | Config file + key/section | Runnable test nodeid | Documentation local link | Public/private status |
| --- | --- | --- | --- | --- | --- | --- |
| scene-anchoring | Contextual Scene Anchoring | `src/evoworld/grounding.py::IndexedSceneAnchor` | `configs/dry_run.toml::grounding_mode` | `tests/unit/test_grounding.py::test_indexed_scene_anchor_returns_injected_scene` | [stage contract](architecture.md#four-article-stages) | public |
| joint-routing-score | Strategic Routing and Imaginative Simulation | `src/evoworld/contracts.py::CandidateScore.from_components` | `configs/dry_run.toml::router_weight` | `tests/unit/test_contracts.py::test_candidate_score_combines_router_and_world_model_scores` | [durable candidate evidence](architecture.md#durable-execution-mapping) | public |
| vlm-diagnosis | VLM Diagnosis and Recursive Evolution | `src/evoworld/adapters/qwen_vl.py::QwenVlEvaluator` | `configs/dry_run.toml::reliability_threshold` | `tests/contract/test_qwen_evaluator_adapter.py::test_qwen_response_maps_to_diagnosis` | [diagnosis mapping](architecture.md#durable-execution-mapping) | private-required |
| verified-buffer | VLM Diagnosis and Recursive Evolution | `src/evoworld/storage.py::JsonRunStore.write_verified` | `configs/dry_run.toml::reliability_threshold` | `tests/integration/test_pipeline_success.py::test_successful_round_persists_every_method_stage` | [canonical records](data.md#canonical-training-manifest) | public |
| tsm-update | VLM Diagnosis and Recursive Evolution | `src/evoworld/evolution.py::decide_update` | `configs/dry_run.toml::reliability_threshold` | `tests/unit/test_evolution.py::test_tsm_has_no_training_artifacts` | [update mapping](architecture.md#durable-execution-mapping) | public |
| pf-update | VLM Diagnosis and Recursive Evolution | `src/evoworld/evolution.py::decide_update` | `configs/dry_run.toml::reliability_threshold` | `tests/unit/test_evolution.py::test_policy_failure_keeps_only_the_selected_expert` | [update mapping](architecture.md#durable-execution-mapping) | public |
| wmf-update | VLM Diagnosis and Recursive Evolution | `src/evoworld/evolution.py::decide_update` | `configs/dry_run.toml::reliability_threshold` | `tests/unit/test_evolution.py::test_wmf_targets_the_world_model` | [update mapping](architecture.md#durable-execution-mapping) | public |
| pi05-config | VLM Diagnosis and Recursive Evolution | `training/experts/openpi.py::plan_openpi_training` | `configs/experts/openpi.toml::pi05` | `tests/contract/test_openpi_training.py::test_openpi_method_settings[pi05-32-15-16-None]` | [openpi fine-tuning](fine_tuning.md#openpi-experts) | private-required |
| pi0-config | VLM Diagnosis and Recursive Evolution | `training/experts/openpi.py::plan_openpi_training` | `configs/experts/openpi.toml::pi0` | `tests/contract/test_openpi_training.py::test_openpi_method_settings[pi0-None-10-8-None]` | [openpi fine-tuning](fine_tuning.md#openpi-experts) | private-required |
| pi0-fast-config | VLM Diagnosis and Recursive Evolution | `training/experts/openpi.py::plan_openpi_training` | `configs/experts/openpi.toml::pi0_fast` | `tests/contract/test_openpi_training.py::test_openpi_method_settings[pi0_fast-8-10-8-180]` | [openpi fine-tuning](fine_tuning.md#openpi-experts) | private-required |
| groot-n16-config | VLM Diagnosis and Recursive Evolution | `training/experts/groot.py::plan_groot_training` | `configs/experts/groot_n16.toml::groot_n16` | `tests/contract/test_groot_training.py::test_groot_plan_uses_exact_appendix_budget_and_official_flags` | [GR00T-N1.6 fine-tuning](fine_tuning.md#gr00t-n16) | private-required |
| ctrl-world-training | VLM Diagnosis and Recursive Evolution | `training/world_model/ctrl_world.py::plan_ctrl_world_training` | `configs/world_model/ctrl_world.toml::ctrl_world.max_train_steps` | `tests/contract/test_ctrl_world_training.py::test_wmf_plan_materializes_exact_train_projection_and_bound_request` | [Ctrl-World fine-tuning](fine_tuning.md#ctrl-world) | private-required |
| qwen-evaluator | VLM Diagnosis and Recursive Evolution | `training/evaluator/qwen_vl.py::plan_qwen_training` | `configs/evaluator/qwen3_vl_4b.toml::qwen3_vl_4b` | `tests/contract/test_qwen_training.py::test_qwen_plan_uses_pinned_public_ms_swift_schema_and_safe_argv` | [Qwen3-VL baseline](fine_tuning.md#qwen3-vl-evaluator-baseline) | private-required |
| resume | Task Generation; Contextual Scene Anchoring; Strategic Routing and Imaginative Simulation; VLM Diagnosis and Recursive Evolution | `src/evoworld/pipeline.py::EvoWorldPipeline` | `configs/dry_run.toml::run_root` | `tests/integration/test_pipeline_success.py::test_resume_skips_every_completed_side_effect[grounded]` | [durable resume](architecture.md#durable-operations-and-resume) | public |
| submodule-pins | Strategic Routing and Imaginative Simulation; VLM Diagnosis and Recursive Evolution | `src/evoworld/adapters/revisions.py::inspect_submodules` | `configs/reproduction.toml::revisions` | `tests/contract/test_submodule_revisions.py::test_revision_audit_returns_expected_resolved_revisions` | [upstream process boundary](architecture.md#real-process-boundary) | public |
| dry-run | Task Generation; Contextual Scene Anchoring; Strategic Routing and Imaginative Simulation; VLM Diagnosis and Recursive Evolution | `src/evoworld/cli.py::main` | `configs/dry_run.toml::mode` | `tests/integration/test_cli.py::test_dry_run_cli_completes_one_round[<lambda>0]` | [public ports](architecture.md#ports) | public |
| site | Task Generation; Contextual Scene Anchoring; Strategic Routing and Imaginative Simulation; VLM Diagnosis and Recursive Evolution | `tests/integration/test_site.py::validate_public_site_release` | `.github/workflows/pages.yml::jobs.deploy` | `tests/integration/test_site.py::test_public_site_release_contract` | [site and release evidence](fidelity.md#site-and-public-release) | public |

## Site and public release

The static site mirrors the four-stage terminology and retains its local visual and
PDF assets. Its tests validate local references and the Pages workflow. They do not
turn illustrative media into evidence of a newly executed robot experiment; real
execution remains subject to the [declared limitations](limitations.md#real-execution-and-compute).

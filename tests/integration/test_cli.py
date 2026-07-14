"""Public CLI composition and recovery contracts."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
import re
import subprocess
import sys
from typing import cast

import pytest

from evoworld.adapters.deterministic import build_deterministic_adapters
from evoworld.cli import main
from evoworld.config import load_config
from evoworld.contracts import RoundStage, RoundState
from evoworld.pipeline import EvoWorldPipeline
from evoworld.storage import JsonRunStore
from training.common.schema import TrainingPlan


def _write_config(
    root: Path,
    *,
    mode: str = "dry_run",
    executor: str | None = None,
) -> Path:
    executor_line = f'executor = "{executor}"\n' if executor is not None else ""
    path = root / "config.toml"
    path.write_text(
        f"""\
schema_version = 1
seed = 7
run_root = "runs"
mode = "{mode}"
grounding_mode = "nearest"
grounding_temperature = 0.5
router_weight = 0.25
reliability_threshold = 0.7
rollout_horizon = 3
{executor_line}
[[experts]]
expert_id = "pi05"
backend = "openpi"
task_families = ["manipulation"]
action_schema = "joint_velocity"
enabled = true
""",
        encoding="utf-8",
    )
    return path


def _payload(captured: pytest.CaptureFixture[str]) -> dict[str, object]:
    output = captured.readouterr()
    assert output.err == ""
    return cast(dict[str, object], json.loads(output.out))


def test_validate_has_no_run_or_backend_side_effects(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path)

    def fail_if_composed(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("validate must not compose adapters")

    monkeypatch.setattr(
        "evoworld.cli.build_deterministic_adapters",
        fail_if_composed,
    )

    assert main(["validate", "--config", str(config_path)]) == 0

    payload = _payload(capsys)
    assert payload == {
        "config_digest": payload["config_digest"],
        "expert_ids": ["pi05"],
        "mode": "dry_run",
        "schema_version": 1,
        "status": "valid",
    }
    assert len(cast(str, payload["config_digest"])) == 64
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    "arguments",
    (
        lambda path: ["run", "--config", str(path), "--rounds", "1"],
        lambda path: ["run", "--rounds", "1", str(path)],
    ),
)
def test_dry_run_cli_completes_one_round(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    arguments: Callable[[Path], list[str]],
) -> None:
    config_path = _write_config(tmp_path)

    assert main(arguments(config_path)) == 0

    payload = _payload(capsys)
    assert payload["completed_rounds"] == 1
    assert payload["status"] == "complete"
    run_dir = Path(cast(str, payload["run_dir"]))
    assert run_dir.is_absolute()
    assert run_dir.is_dir()
    assert (run_dir / "manifest.json").is_file()


def test_real_mode_never_falls_back_to_deterministic(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path, mode="real", executor="robot-cell")

    def fail_if_composed(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("real mode must not compose deterministic adapters")

    monkeypatch.setattr(
        "evoworld.cli.build_deterministic_adapters",
        fail_if_composed,
    )

    assert main(["run", "--config", str(config_path), "--rounds", "1"]) == 2

    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err) == {
        "error": {
            "message": "real mode requires an explicit --factory",
            "type": "CompositionError",
        },
        "status": "error",
    }
    assert not (tmp_path / "runs").exists()


def test_console_factory_survives_fresh_process_resume_and_is_manifested(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path, mode="real", executor="robot-cell")
    module = tmp_path / "fixture_factory.py"
    module.write_text(
        "from dataclasses import replace\n"
        "import json\n"
        "import os\n"
        "from evoworld.adapters.deterministic import build_deterministic_adapters\n"
        "from evoworld.config import config_digest, resolved_manifest\n"
        "from evoworld.storage import atomic_write_json\n"
        "class Bundle:\n"
        "    runtime_manifest_digest = os.environ.get('FIXTURE_RUNTIME_DIGEST', 'e' * 64)\n"
        "    def __init__(self, wrapped): self.wrapped = wrapped\n"
        "    def __getattr__(self, name): return getattr(self.wrapped, name)\n"
        "def build(config, run_dir):\n"
        "    dry = replace(config, mode='dry_run', executor=None)\n"
        "    manifest = run_dir / 'manifest.json'\n"
        "    existing = json.loads(manifest.read_text()) if manifest.exists() else {}\n"
        "    if existing:\n"
        "        atomic_write_json(manifest, {'config_digest': config_digest(dry), "
        "'resolved_config': resolved_manifest(dry)})\n"
        "    adapters = build_deterministic_adapters(dry, run_dir)\n"
        "    wrapper = {'config_digest': config_digest(config), "
        "'resolved_config': resolved_manifest(config)}\n"
        "    if 'composition' in existing and not os.environ.get('FIXTURE_DROP_COMPOSITION'):\n"
        "        wrapper['composition'] = existing['composition']\n"
        "    atomic_write_json(adapters.store.run_dir / 'manifest.json', wrapper)\n"
        "    return Bundle(adapters)\n",
        encoding="utf-8",
    )
    project = Path(__file__).parents[2].resolve()
    environment = {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": f"{tmp_path}:{project / 'src'}:{project}",
    }

    started = subprocess.run(
        [
            sys.executable,
            "-m",
            "evoworld",
            "run",
            "--config",
            str(config_path),
            "--rounds",
            "1",
            "--factory",
            "fixture_factory:build",
        ],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert started.returncode == 0, started.stderr
    run_dir = Path(json.loads(started.stdout)["run_dir"])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["composition"] == {
        "factory": "fixture_factory:build",
        "runtime_manifest_digest": "e" * 64,
    }

    resumed = subprocess.run(
        [sys.executable, "-m", "evoworld", "resume", "--run-dir", str(run_dir)],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout)["run_dir"] == str(run_dir)

    drifted_environment = {**environment, "FIXTURE_RUNTIME_DIGEST": "f" * 64}
    drifted = subprocess.run(
        [sys.executable, "-m", "evoworld", "resume", "--run-dir", str(run_dir)],
        cwd=project,
        env=drifted_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert drifted.returncode == 2
    assert json.loads(drifted.stderr)["error"] == {
        "message": "runtime manifest or static asset identity changed",
        "type": "FactoryError",
    }

    dropped_environment = {**environment, "FIXTURE_DROP_COMPOSITION": "1"}
    dropped = subprocess.run(
        [sys.executable, "-m", "evoworld", "resume", "--run-dir", str(run_dir)],
        cwd=project,
        env=dropped_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert dropped.returncode == 2
    assert json.loads(dropped.stderr)["error"] == {
        "message": "factory changed the durable composition identity",
        "type": "FactoryError",
    }


@pytest.mark.parametrize(
    ("identity", "message"),
    [
        ("not a factory", "strict module:callable syntax"),
        ("missing_factory:build", "module could not be imported"),
        ("json:missing", "callable is missing"),
    ],
)
def test_real_factory_errors_are_typed_and_redacted(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    identity: str,
    message: str,
) -> None:
    config_path = _write_config(tmp_path, mode="real", executor="robot-cell")

    assert main(["run", "--config", str(config_path), "--factory", identity]) == 2

    output = capsys.readouterr()
    assert output.out == ""
    error = json.loads(output.err)["error"]
    assert error["type"] == "FactoryError"
    assert message in error["message"]
    assert str(tmp_path) not in error["message"]
    assert not (tmp_path / "runs").exists()


def test_resume_completes_latest_incomplete_round_from_run_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = load_config(_write_config(tmp_path))
    run_dir = tmp_path / "runs" / "interrupted"
    adapters = build_deterministic_adapters(config, run_dir)
    started = RoundState(
        round_id="round-000042",
        stage=RoundStage.STARTED,
        started_at="2026-07-17T00:00:00+00:00",
        seed=config.seed,
        reliability_threshold=config.reliability_threshold,
    )
    adapters.store.claim_round(started)

    assert main(["resume", "--run-dir", str(run_dir)]) == 0

    payload = _payload(capsys)
    assert payload == {
        "completed_rounds": 1,
        "resumed_round_id": "round-000042",
        "run_dir": str(run_dir.resolve()),
        "status": "complete",
    }
    assert cast(JsonRunStore, adapters.store).read_state_required().stage is RoundStage.COMPLETE


def test_resume_does_not_revive_a_completed_latest_round(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = load_config(_write_config(tmp_path))
    run_dir = tmp_path / "runs" / "complete"
    adapters = build_deterministic_adapters(config, run_dir)
    adapters.store.claim_round(
        RoundState(
            round_id="round-orphan",
            stage=RoundStage.STARTED,
            started_at="2026-07-17T00:00:00+00:00",
            seed=config.seed,
            reliability_threshold=config.reliability_threshold,
        )
    )
    pipeline = EvoWorldPipeline(config, adapters)
    pipeline.run_round("round-complete")
    before = (run_dir / "state.json").read_bytes()

    assert main(["resume", "--run-dir", str(run_dir)]) == 0

    payload = _payload(capsys)
    assert payload["completed_rounds"] == 0
    assert payload["resumed_round_id"] is None
    assert (run_dir / "state.json").read_bytes() == before
    assert adapters.store.load_state("round-orphan").stage is RoundStage.STARTED


@pytest.mark.parametrize("error_type", (RuntimeError, OSError))
def test_runtime_error_is_code_one_and_redacts_private_details(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    config = load_config(_write_config(tmp_path))
    run_dir = tmp_path / "runs" / "in-doubt"
    build_deterministic_adapters(config, run_dir)

    def fail_with_private_detail(_pipeline: EvoWorldPipeline) -> None:
        raise error_type(f"secret token at {tmp_path / 'backend' / 'secret.key'}")

    monkeypatch.setattr(EvoWorldPipeline, "resume", fail_with_private_detail)

    assert main(["resume", "--run-dir", str(run_dir)]) == 1

    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err) == {
        "error": {"message": "pipeline execution failed", "type": "RuntimeError"},
        "status": "error",
    }
    assert str(tmp_path) not in output.err
    assert "secret" not in output.err


def test_config_error_is_code_two_and_redacts_absolute_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "private" / "secret-config.toml"

    assert main(["validate", "--config", str(missing)]) == 2

    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err) == {
        "error": {
            "message": "configuration could not be loaded",
            "type": "ConfigError",
        },
        "status": "error",
    }
    assert str(tmp_path) not in output.err


def test_invalid_run_manifest_is_a_redacted_config_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "runs" / "corrupt"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("not-json", encoding="utf-8")

    assert main(["resume", "--run-dir", str(run_dir)]) == 2

    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err)["error"] == {
        "message": "configuration validation failed",
        "type": "ConfigError",
    }
    assert str(tmp_path) not in output.err


@pytest.mark.parametrize(
    ("field", "toml_value"),
    (
        (field, toml_value)
        for field in (
            "grounding_temperature",
            "router_weight",
            "reliability_threshold",
        )
        for toml_value in ("nan", "+inf", "-inf")
    ),
)
def test_nonfinite_config_fails_before_run_composition_or_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    toml_value: str,
) -> None:
    config_path = _write_config(tmp_path)
    text = config_path.read_text(encoding="utf-8")
    prefix = f"{field} = "
    config_path.write_text(
        "\n".join(
            f"{prefix}{toml_value}" if line.startswith(prefix) else line
            for line in text.splitlines()
        )
        + "\n",
        encoding="utf-8",
    )

    def fail_if_composed(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("invalid config must fail before adapter composition")

    monkeypatch.setattr("evoworld.cli.build_deterministic_adapters", fail_if_composed)

    assert main(["validate", "--config", str(config_path)]) == 2
    assert main(["run", "--config", str(config_path), "--rounds", "1"]) == 2

    output = capsys.readouterr()
    assert output.out == ""
    errors = [json.loads(line) for line in output.err.splitlines()]
    assert errors == [
        {
            "error": {
                "message": "configuration validation failed",
                "type": "ConfigError",
            },
            "status": "error",
        },
        {
            "error": {
                "message": "configuration validation failed",
                "type": "ConfigError",
            },
            "status": "error",
        },
    ]
    assert str(tmp_path) not in output.err
    assert not (tmp_path / "runs").exists()


def test_python_module_entrypoint_matches_console_contract(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    completed = subprocess.run(
        [sys.executable, "-m", "evoworld", "validate", "--config", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["status"] == "valid"
    assert not (tmp_path / "runs").exists()


def test_pi05_finetune_example_loads_and_plans_to_its_configured_job_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    template = Path("configs/finetune/pi05.example.toml")
    rendered = template.read_text(encoding="utf-8").replace(
        "/replace/with/job-root", (tmp_path / "jobs").as_posix()
    )
    config = tmp_path / template.name
    config.write_text(rendered, encoding="utf-8")
    plan = TrainingPlan(
        schema_version=1,
        job_id="pi05-example-job",
        target="pi05",
        manifest_sha256="a" * 64,
        upstream_revision="b" * 40,
        cwd=tmp_path,
        argv=("fake-train", "pi05"),
        environment={},
        expected_outputs=(Path("checkpoint"),),
    )
    from evoworld.training import PlannedTarget

    monkeypatch.setattr(
        "evoworld.training._compose_target",
        lambda settings, target: PlannedTarget(plan, None),
    )

    assert main(["finetune", "plan", "--config", str(config), "--target", "pi05"]) == 0

    job = tmp_path / "jobs/pi05-example-job/job.json"
    assert job.is_file()
    assert json.loads(capsys.readouterr().out)["job"] == str(job.resolve())


def test_readme_is_concise() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert len(readme.splitlines()) <= 70
    for heading in ("## Quick start", "## Fine-tuning", "## Documentation", "## Limitations"):
        assert heading in readme


@pytest.mark.parametrize(
    "command",
    (
        "uv run evoworld validate --config configs/dry_run.toml",
        "uv run evoworld run --config configs/dry_run.toml --rounds 1",
        "uv run evoworld resume --run-dir artifacts/runs/<run-id>",
        "cp configs/finetune/pi05.example.toml /absolute/path/pi05.toml",
        "uv run evoworld finetune plan --config /absolute/path/pi05.toml --target pi05",
        "uv run evoworld finetune execute --job <job-from-plan-json>",
    ),
)
def test_readme_contains_public_commands(command: str) -> None:
    assert command in Path("README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "relative_path",
    (
        "docs/architecture.md",
        "docs/data.md",
        "docs/fine_tuning.md",
        "docs/backends/real_composition.md",
        "site/static/pdfs/evoworld.pdf",
    ),
)
def test_readme_links_to_existing_public_guides(relative_path: str) -> None:
    assert relative_path in Path("README.md").read_text(encoding="utf-8")
    assert Path(relative_path).is_file()


def test_public_markdown_local_links_resolve() -> None:
    public_docs = [
        Path("README.md"),
        *(path for path in Path("docs").rglob("*.md") if "superpowers" not in path.parts),
    ]
    for source in public_docs:
        for raw_target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", source.read_text(encoding="utf-8")):
            target = raw_target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            assert (source.parent / target).is_file(), f"broken local link in {source}: {target}"


def test_public_docs_preserve_method_names_and_current_finetune_entrypoints() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    architecture = Path("docs/architecture.md").read_text(encoding="utf-8")
    fine_tuning = Path("docs/fine_tuning.md").read_text(encoding="utf-8")
    rendered_readme = " ".join(readme.split())
    rendered_architecture = " ".join(architecture.split())
    for stage in (
        "Task Generation",
        "Contextual Scene Anchoring",
        "Strategic Routing and Imaginative Simulation",
        "VLM Diagnosis and Recursive Evolution",
    ):
        assert stage in rendered_readme
        assert stage in rendered_architecture
    obsolete = (
        "finetune plan --config configs/reproduction.toml",
        "finetune execute --job runs/<run-id>/training/<job-id>/job.json",
        "scripts/compute_norm_stats.py --config-name",
    )
    assert all(command not in readme + fine_tuning for command in obsolete)
    for argument in (
        "integrations/openpi/train_bridge.py normalize",
        "--openpi-root",
        "--config-path",
        "--expert",
        "--manifest-path",
        "--manifest-sha256",
        "--dataset-root",
        "--output-dir",
        "--job-id",
    ):
        assert argument in fine_tuning

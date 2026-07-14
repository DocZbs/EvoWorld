from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.audit_fidelity import (
    AuditError,
    EXPECTED_COLUMNS,
    EXPECTED_MARKERS,
    MatrixRow,
    _audit_config_reference,
    audit_repository,
    parse_matrix,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = REPOSITORY_ROOT / "docs" / "fidelity.md"


class RecordingCollector:
    def __init__(self, collected: set[str]) -> None:
        self.collected = collected
        self.requested: tuple[str, ...] = ()

    def collect(self, root: Path, nodeids: tuple[str, ...]) -> set[str]:
        assert root == REPOSITORY_ROOT
        self.requested = nodeids
        return self.collected


def _valid_rows() -> tuple[MatrixRow, ...]:
    return parse_matrix(MATRIX_PATH.read_text(encoding="utf-8"))


def _write_matrix(path: Path, rows: tuple[MatrixRow, ...]) -> None:
    lines = [
        "# Test matrix",
        "",
        "| " + " | ".join(EXPECTED_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in EXPECTED_COLUMNS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    row.marker,
                    row.article_section,
                    f"`{row.code_ref}`",
                    f"`{row.config_ref}`",
                    f"`{row.test_nodeid}`",
                    f"[evidence]({row.documentation_link})",
                    row.status,
                )
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_checked_in_matrix_has_exact_contract_and_collectable_evidence() -> None:
    rows = _valid_rows()
    collector = RecordingCollector({row.test_nodeid for row in rows})

    audited = audit_repository(REPOSITORY_ROOT, MATRIX_PATH, collector=collector)

    assert {row.marker for row in audited} == EXPECTED_MARKERS
    assert collector.requested == tuple(row.test_nodeid for row in audited)
    assert all(row.evidence_kind == "runtime-test" for row in audited)


def test_default_cli_audits_real_pytest_collection() -> None:
    completed = subprocess.run(
        (sys.executable, "scripts/audit_fidelity.py"),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "fidelity audit passed: 17 markers"


@pytest.mark.parametrize(
    "mutation, message",
    (
        (lambda rows: rows[:-1], "marker set"),
        (lambda rows: rows + (rows[0],), "duplicate marker"),
    ),
)
def test_matrix_rejects_missing_or_duplicate_markers(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    rows = _valid_rows()
    matrix = tmp_path / "matrix.md"
    _write_matrix(matrix, mutation(rows))  # type: ignore[operator]

    with pytest.raises(AuditError, match=message):
        audit_repository(REPOSITORY_ROOT, matrix, collector=RecordingCollector(set()))


def test_parser_rejects_unknown_column() -> None:
    text = MATRIX_PATH.read_text(encoding="utf-8").replace(
        "| Public/private status |", "| Public/private status | Unknown |", 1
    )

    with pytest.raises(AuditError, match="exact columns"):
        parse_matrix(text)


def test_parser_rejects_a_complete_duplicate_contract_table() -> None:
    document = MATRIX_PATH.read_text(encoding="utf-8")
    table = document[document.index("| Marker |") : document.index("\n\n## Site")]

    with pytest.raises(AuditError, match="exactly one contract table"):
        parse_matrix(f"{document}\n{table}\n")


def test_parser_rejects_an_outerless_duplicate_contract_table() -> None:
    document = MATRIX_PATH.read_text(encoding="utf-8")
    table = document[document.index("| Marker |") : document.index("\n\n## Site")]
    outerless = "\n".join(
        line.strip().removeprefix("|").removesuffix("|") for line in table.splitlines()
    )

    with pytest.raises(AuditError, match="exactly one contract table"):
        parse_matrix(f"{document}\n{outerless}\n")


def test_parser_rejects_any_second_markdown_table() -> None:
    document = MATRIX_PATH.read_text(encoding="utf-8")
    extra = "| Unrelated | Extra |\n| --- | --- |\n| value | value |\n"

    with pytest.raises(AuditError, match="additional Markdown table"):
        parse_matrix(f"{document}\n{extra}")


@pytest.mark.parametrize(
    "document, message",
    (
        ("jobs: |\n  deploy:\n", "configuration key"),
        ("jobs:\n  deploy: {}\njobs:\n  deploy: {}\n", "duplicate YAML key"),
        ("jobs:\n  deploy: {}\n  deploy: {}\n", "duplicate YAML key"),
    ),
)
def test_yaml_config_rejects_scalar_disguise_and_duplicate_keys(
    tmp_path: Path,
    document: str,
    message: str,
) -> None:
    workflow = tmp_path / ".github/workflows/pages.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(document, encoding="utf-8")

    with pytest.raises(AuditError, match=message):
        _audit_config_reference(tmp_path, ".github/workflows/pages.yml::jobs.deploy")


@pytest.mark.parametrize(
    "field, unsafe",
    (
        ("code_ref", "../src/evoworld/pipeline.py::EvoWorldPipeline"),
        ("config_ref", "https://example.test/config.toml::section"),
        ("documentation_link", "architecture.md?source=external#four-article-stages"),
        ("test_nodeid", "tests/integration/test_paper_claim.py::test_claim"),
    ),
)
def test_audit_rejects_traversal_uri_query_and_banned_filename(
    tmp_path: Path,
    field: str,
    unsafe: str,
) -> None:
    rows = _valid_rows()
    matrix = tmp_path / "matrix.md"
    _write_matrix(matrix, (replace(rows[0], **{field: unsafe}), *rows[1:]))

    with pytest.raises(AuditError, match="unsafe|forbidden"):
        audit_repository(REPOSITORY_ROOT, matrix, collector=RecordingCollector(set()))


@pytest.mark.parametrize(
    "field, replacement, message",
    (
        ("code_ref", "src/evoworld/pipeline.py::MissingSymbol", "AST symbol"),
        ("config_ref", "configs/dry_run.toml::missing_key", "configuration key"),
        ("documentation_link", "architecture.md#missing-anchor", "documentation anchor"),
    ),
)
def test_audit_rejects_unverifiable_code_config_and_documentation_refs(
    tmp_path: Path,
    field: str,
    replacement: str,
    message: str,
) -> None:
    rows = _valid_rows()
    matrix = tmp_path / "matrix.md"
    _write_matrix(matrix, (replace(rows[0], **{field: replacement}), *rows[1:]))

    with pytest.raises(AuditError, match=message):
        audit_repository(
            REPOSITORY_ROOT,
            matrix,
            collector=RecordingCollector({row.test_nodeid for row in rows}),
        )


def test_audit_rejects_nodeid_not_returned_by_collector(tmp_path: Path) -> None:
    rows = _valid_rows()
    matrix = tmp_path / "matrix.md"
    _write_matrix(matrix, rows)

    with pytest.raises(AuditError, match="pytest collection"):
        audit_repository(REPOSITORY_ROOT, matrix, collector=RecordingCollector(set()))

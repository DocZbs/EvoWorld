"""Fail-closed method-to-code fidelity matrix audit."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tomllib
from typing import Protocol
from urllib.parse import unquote, urlsplit

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = REPOSITORY_ROOT / "docs" / "fidelity.md"
EXPECTED_COLUMNS = (
    "Marker",
    "Article/method section",
    "Code file + symbol",
    "Config file + key/section",
    "Runnable test nodeid",
    "Documentation local link",
    "Public/private status",
)
EXPECTED_MARKERS = frozenset(
    {
        "scene-anchoring",
        "joint-routing-score",
        "vlm-diagnosis",
        "verified-buffer",
        "tsm-update",
        "pf-update",
        "wmf-update",
        "pi05-config",
        "pi0-config",
        "pi0-fast-config",
        "groot-n16-config",
        "ctrl-world-training",
        "qwen-evaluator",
        "resume",
        "submodule-pins",
        "dry-run",
        "site",
    }
)
ARTICLE_SECTIONS = frozenset(
    {
        "Task Generation",
        "Contextual Scene Anchoring",
        "Strategic Routing and Imaginative Simulation",
        "VLM Diagnosis and Recursive Evolution",
    }
)
STATUSES = frozenset({"public", "private-required"})
_SYMBOL = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$")
_MARKDOWN_LINK = re.compile(r"^\[[^\]]+\]\(([^()]+)\)$")


class _UniqueSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that never permits an overwritten mapping key."""


_BOOL_TAG = "tag:yaml.org,2002:bool"
_UniqueSafeLoader.yaml_implicit_resolvers = {
    key: [entry for entry in entries if entry[0] != _BOOL_TAG]
    for key, entries in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_STRICT_BOOL = re.compile(r"^(?:true|false|True|False|TRUE|FALSE)$")
for _initial in "tTfF":
    _UniqueSafeLoader.yaml_implicit_resolvers.setdefault(_initial, []).append(
        (_BOOL_TAG, _STRICT_BOOL)
    )


def _construct_unique_mapping(
    loader: yaml.SafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "unhashable YAML mapping key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate YAML key: {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


class AuditError(ValueError):
    """The checked-in fidelity claim is incomplete or unverifiable."""


@dataclass(frozen=True)
class MatrixRow:
    marker: str
    article_section: str
    code_ref: str
    config_ref: str
    test_nodeid: str
    documentation_link: str
    status: str

    @property
    def evidence_kind(self) -> str:
        """Distinguish runnable test evidence from a file-existence assertion."""

        return "runtime-test"


class NodeCollector(Protocol):
    """Boundary for proving that every documented nodeid is pytest-addressable."""

    def collect(self, root: Path, nodeids: tuple[str, ...]) -> set[str]:
        """Return the exact nodeids collected by pytest."""


class PytestCollector:
    """Collect exact nodeids in one isolated pytest invocation."""

    def collect(self, root: Path, nodeids: tuple[str, ...]) -> set[str]:
        completed = subprocess.run(
            (
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "--disable-warnings",
                *nodeids,
            ),
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise AuditError(f"pytest collection failed: {detail}")
        return {
            line.strip()
            for line in completed.stdout.splitlines()
            if line.strip().startswith("tests/") and "::" in line
        }


def parse_matrix(document: str) -> tuple[MatrixRow, ...]:
    """Parse the one strict Markdown matrix without accepting extra fields."""

    lines = document.splitlines()
    table_starts = _table_starts(lines)
    header_indices = [
        index for index in table_starts if tuple(_table_cells(lines[index])) == EXPECTED_COLUMNS
    ]
    if not header_indices:
        raise AuditError("matrix must use the exact columns")
    if len(header_indices) != 1:
        raise AuditError("document must contain exactly one contract table")
    header_index = header_indices[0]
    if any(index != header_index for index in table_starts):
        raise AuditError("document contains an additional Markdown table")
    if header_index + 1 >= len(lines):
        raise AuditError("matrix separator is missing")
    separator = _table_cells(lines[header_index + 1])
    if len(separator) != len(EXPECTED_COLUMNS) or not all(
        re.fullmatch(r":?-{3,}:?", cell) for cell in separator
    ):
        raise AuditError("matrix separator does not match the exact columns")

    rows: list[MatrixRow] = []
    for line in lines[header_index + 2 :]:
        cells = _table_cells(line)
        if not cells:
            break
        if len(cells) != len(EXPECTED_COLUMNS):
            raise AuditError("matrix row has unknown fields or missing columns")
        if any(not cell for cell in cells):
            raise AuditError("matrix row fields cannot be empty")
        documentation_match = _MARKDOWN_LINK.fullmatch(cells[5])
        if documentation_match is None:
            raise AuditError("documentation field must be one local Markdown link")
        rows.append(
            MatrixRow(
                marker=cells[0],
                article_section=cells[1],
                code_ref=_inline_code(cells[2], "code reference"),
                config_ref=_inline_code(cells[3], "configuration reference"),
                test_nodeid=_inline_code(cells[4], "test nodeid"),
                documentation_link=documentation_match.group(1),
                status=cells[6],
            )
        )
    markers = [row.marker for row in rows]
    if len(markers) != len(set(markers)):
        raise AuditError("matrix contains a duplicate marker")
    if set(markers) != EXPECTED_MARKERS:
        missing = sorted(EXPECTED_MARKERS - set(markers))
        extra = sorted(set(markers) - EXPECTED_MARKERS)
        raise AuditError(f"matrix marker set mismatch: missing={missing}, extra={extra}")
    for row in rows:
        sections = {section.strip() for section in row.article_section.split(";")}
        if not sections or "" in sections or not sections <= ARTICLE_SECTIONS:
            raise AuditError(f"unknown article/method section for {row.marker}")
        if row.status not in STATUSES:
            raise AuditError(f"unknown public/private status for {row.marker}")
    return tuple(rows)


def audit_repository(
    root: Path,
    matrix_path: Path,
    *,
    collector: NodeCollector | None = None,
) -> tuple[MatrixRow, ...]:
    """Validate every matrix claim against repository structure and pytest collection."""

    repository_root = root.resolve(strict=True)
    rows = parse_matrix(matrix_path.read_text(encoding="utf-8"))
    for row in rows:
        _audit_code_reference(repository_root, row.code_ref)
        _audit_config_reference(repository_root, row.config_ref)
        _audit_documentation_reference(
            repository_root,
            repository_root / "docs",
            row.documentation_link,
        )
        _test_path(repository_root, row.test_nodeid)

    nodeids = tuple(row.test_nodeid for row in rows)
    collected = (PytestCollector() if collector is None else collector).collect(
        repository_root, nodeids
    )
    missing_nodeids = sorted(set(nodeids) - collected)
    if missing_nodeids:
        raise AuditError(f"pytest collection did not return nodeids: {missing_nodeids}")
    return rows


def _table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    if "|" not in stripped:
        return []
    cells = [cell.strip() for cell in stripped.split("|")]
    return cells if len(cells) >= 2 else []


def _table_starts(lines: Sequence[str]) -> tuple[int, ...]:
    starts: list[int] = []
    for index in range(len(lines) - 1):
        header = _table_cells(lines[index])
        separator = _table_cells(lines[index + 1])
        if (
            header
            and len(separator) == len(header)
            and all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator)
        ):
            starts.append(index)
    return tuple(starts)


def _inline_code(value: str, field: str) -> str:
    if len(value) < 3 or not value.startswith("`") or not value.endswith("`"):
        raise AuditError(f"{field} must use one inline-code span")
    unwrapped = value[1:-1]
    if "`" in unwrapped:
        raise AuditError(f"{field} contains an unexpected delimiter")
    return unwrapped


def _safe_relative_path(value: str, field: str, *, allow_fragment: bool = False) -> Path:
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or parsed.query or (parsed.fragment and not allow_fragment):
        raise AuditError(f"unsafe URI in {field}")
    decoded = unquote(parsed.path)
    pure = PurePosixPath(decoded)
    if (
        not decoded
        or decoded.startswith("/")
        or "\\" in decoded
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise AuditError(f"unsafe path traversal in {field}")
    if any("paper" in part.casefold() for part in pure.parts):
        raise AuditError(f"forbidden filename in {field}")
    return Path(*pure.parts)


def _contained_file(root: Path, relative: Path, field: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise AuditError(f"missing or unsafe local file for {field}: {relative.as_posix()}")
    return candidate


def _split_ref(value: str, field: str) -> tuple[str, str]:
    if value.count("::") != 1:
        raise AuditError(f"{field} must contain one file::name reference")
    path, name = value.split("::")
    if not name or not _SYMBOL.fullmatch(name):
        raise AuditError(f"invalid name in {field}")
    return path, name


def _audit_code_reference(root: Path, reference: str) -> None:
    path_text, symbol = _split_ref(reference, "code reference")
    relative = _safe_relative_path(path_text, "code reference")
    if relative.suffix != ".py":
        raise AuditError("code reference must name a Python file")
    path = _contained_file(root, relative, "code reference")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
    symbols = _qualified_symbols(tree)
    if symbol not in symbols:
        raise AuditError(f"AST symbol is missing: {reference}")


def _qualified_symbols(tree: ast.AST) -> set[str]:
    found: set[str] = set()

    def visit(body: Sequence[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{prefix}.{node.name}" if prefix else node.name
                found.add(qualified)
                visit(node.body, qualified)

    if isinstance(tree, ast.Module):
        visit(tree.body)
    return found


def _audit_config_reference(root: Path, reference: str) -> None:
    path_text, key = _split_ref(reference, "configuration reference")
    relative = _safe_relative_path(path_text, "configuration reference")
    path = _contained_file(root, relative, "configuration reference")
    parts = key.split(".")
    if path.suffix == ".toml":
        with path.open("rb") as stream:
            payload: object = tomllib.load(stream)
        if not _mapping_path_exists(payload, parts):
            raise AuditError(f"configuration key is missing: {reference}")
        return
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not _mapping_path_exists(payload, parts):
            raise AuditError(f"configuration key is missing: {reference}")
        return
    if path.suffix in {".yml", ".yaml"}:
        payload = _load_yaml_mapping(path)
        if not _mapping_path_exists(payload, parts):
            raise AuditError(f"configuration key is missing: {reference}")
        return
    raise AuditError("configuration reference must name TOML, JSON, or YAML")


def _mapping_path_exists(value: object, parts: Sequence[str]) -> bool:
    current = value
    for part in parts:
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True


def _load_yaml_mapping(path: Path) -> Mapping[object, object]:
    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueSafeLoader)
    except yaml.YAMLError as error:
        raise AuditError(f"invalid YAML configuration: {error}") from error
    if not isinstance(payload, Mapping):
        raise AuditError("YAML configuration root must be a mapping")
    return payload


def _audit_documentation_reference(root: Path, docs_root: Path, reference: str) -> None:
    parsed = urlsplit(reference)
    relative = _safe_relative_path(reference, "documentation reference", allow_fragment=True)
    path = _contained_file(docs_root, relative, "documentation reference")
    if path.suffix != ".md":
        raise AuditError("documentation link must name a Markdown file")
    if not path.is_relative_to(root):
        raise AuditError("documentation link escapes the repository")
    if not parsed.fragment:
        raise AuditError("documentation link must identify a local anchor")
    anchors = {
        _heading_anchor(match.group(1))
        for line in path.read_text(encoding="utf-8").splitlines()
        if (match := _HEADING.fullmatch(line)) is not None
    }
    if parsed.fragment not in anchors:
        raise AuditError(f"documentation anchor is missing: {reference}")


def _heading_anchor(heading: str) -> str:
    without_markup = re.sub(r"[`*_~]", "", heading).strip().casefold()
    without_punctuation = re.sub(r"[^\w\- ]", "", without_markup)
    return re.sub(r"\s+", "-", without_punctuation)


def _test_path(root: Path, nodeid: str) -> None:
    path_text = nodeid.split("::", 1)[0]
    if nodeid.count("::") < 1:
        raise AuditError("test nodeid must include a test symbol")
    relative = _safe_relative_path(path_text, "test nodeid")
    if relative.suffix != ".py" or not relative.parts or relative.parts[0] != "tests":
        raise AuditError("test nodeid must name a Python file under tests/")
    _contained_file(root, relative, "test nodeid")


def main() -> int:
    """Audit the checked-in matrix and print only a stable summary."""

    try:
        rows = audit_repository(REPOSITORY_ROOT, DEFAULT_MATRIX)
    except (
        AuditError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
        yaml.YAMLError,
    ) as error:
        print(f"fidelity audit failed: {error}", file=sys.stderr)
        return 1
    print(f"fidelity audit passed: {len(rows)} markers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

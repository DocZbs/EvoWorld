from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from scripts.audit_fidelity import _load_yaml_mapping


SITE_ROOT = Path("site")
REMOTE_SCHEMES = {"data", "http", "https", "mailto", "tel", "javascript", "blob"}
CSS_REFERENCE = re.compile(
    r"(?:url\(\s*|@import\s+)(?:[\"']?)([^\"')\s;]+)",
    re.IGNORECASE,
)
JS_REFERENCE = re.compile(
    r"(?:import\s+(?:[^;]*?\s+from\s+)?|export\s+[^;]*?\s+from\s+|import\s*\()"
    r"[\"']([^\"']+)[\"']",
)
SOURCE_MAP_REFERENCE = re.compile(r"[#@]\s*sourceMappingURL=([^\s*]+)")


class _HTMLReferences(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.references: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        for attribute in ("href", "src", "poster", "data"):
            value = values.get(attribute)
            if value:
                self.references.append(value)

        srcset = values.get("srcset") or ""
        for candidate in srcset.split(","):
            reference = candidate.strip().split(maxsplit=1)
            if reference:
                self.references.append(reference[0])


def _local_path(source: Path, reference: str) -> Path | None:
    parsed = urlsplit(reference.strip())
    if parsed.scheme.lower() in REMOTE_SCHEMES or parsed.netloc or not parsed.path:
        return None

    decoded_path = unquote(parsed.path).replace("\\", "/")
    candidate = (
        SITE_ROOT / decoded_path.lstrip("/")
        if decoded_path.startswith("/")
        else source.parent / decoded_path
    )
    resolved_site = SITE_ROOT.resolve()
    resolved_candidate = candidate.resolve()
    assert resolved_candidate.is_relative_to(resolved_site), (
        f"local reference escapes site/: {source} -> {reference}"
    )
    return resolved_candidate


def _references(path: Path) -> Iterable[str]:
    document = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".html", ".htm"}:
        parser = _HTMLReferences()
        parser.feed(document)
        return parser.references
    if path.suffix.lower() == ".css":
        return CSS_REFERENCE.findall(document)
    if path.suffix.lower() in {".js", ".mjs"}:
        return [*JS_REFERENCE.findall(document), *SOURCE_MAP_REFERENCE.findall(document)]
    return []


def _validate_local_assets() -> None:
    pending = [SITE_ROOT / "index.html"]
    visited: set[Path] = set()

    while pending:
        source = pending.pop()
        assert source.is_file(), source
        source = source.resolve()
        if source in visited:
            continue
        visited.add(source)

        for reference in _references(source):
            local_path = _local_path(source, reference)
            if local_path is None:
                continue
            assert local_path.is_file(), f"missing local reference: {source} -> {reference}"
            if local_path.suffix.lower() in {".html", ".htm", ".css", ".js", ".mjs"}:
                pending.append(local_path)


def _validate_code_link() -> None:
    document = (SITE_ROOT / "index.html").read_text(encoding="utf-8")
    assert 'href="https://github.com/DocZbs/EvoWorld"' in document
    assert "Code Soon" not in document


def _mapping(value: object, label: str) -> Mapping[object, object]:
    assert isinstance(value, Mapping), f"{label} must be a mapping"
    return value


def _validate_pages_workflow(path: Path = Path(".github/workflows/pages.yml")) -> None:
    workflow = _load_yaml_mapping(path)
    triggers = _mapping(workflow.get("on"), "on")
    assert set(triggers) == {"push"}
    push = _mapping(triggers.get("push"), "on.push")
    assert push.get("branches") == ["main"]

    permissions = _mapping(workflow.get("permissions"), "permissions")
    assert permissions == {
        "contents": "read",
        "pages": "write",
        "id-token": "write",
    }

    jobs = _mapping(workflow.get("jobs"), "jobs")
    assert set(jobs) == {"deploy"}
    deploy = _mapping(jobs.get("deploy"), "jobs.deploy")
    assert deploy.get("runs-on") == "ubuntu-latest"
    environment = _mapping(deploy.get("environment"), "jobs.deploy.environment")
    assert environment == {
        "name": "github-pages",
        "url": "${{ steps.deployment.outputs.page_url }}",
    }
    raw_steps = deploy.get("steps")
    assert isinstance(raw_steps, list)
    steps = tuple(_mapping(step, "jobs.deploy.steps[]") for step in raw_steps)
    expected_actions = (
        "actions/checkout@v4",
        "actions/configure-pages@v5",
        "actions/upload-pages-artifact@v3",
        "actions/deploy-pages@v4",
    )
    assert tuple(step.get("uses") for step in steps) == expected_actions

    checkout = _mapping(steps[0].get("with"), "checkout.with")
    assert checkout.get("submodules") is False
    upload = _mapping(steps[2].get("with"), "upload-pages-artifact.with")
    assert upload.get("path") == "site"
    for step in steps:
        options = step.get("with")
        if options is None:
            continue
        structured = _mapping(options, "step.with")
        if "submodules" in structured:
            assert structured["submodules"] is False


def validate_public_site_release() -> None:
    """Validate the assets, repository link, and site-only Pages deployment together."""

    _validate_local_assets()
    _validate_code_link()
    _validate_pages_workflow()


def test_site_has_no_missing_local_assets() -> None:
    _validate_local_assets()


def test_code_button_links_to_repository() -> None:
    _validate_code_link()


def test_local_reference_normalization_handles_urls_and_rejects_escape() -> None:
    source = (SITE_ROOT / "index.html").resolve()
    expected = (SITE_ROOT / "static/images/teaser.png").resolve()

    assert _local_path(source, "static/images/%74easer.png?cache=1#preview") == expected
    assert _local_path(source, "#top") is None
    with pytest.raises(AssertionError, match="escapes site"):
        _local_path(source, "%2e%2e/pyproject.toml")


def test_pages_workflow_publishes_only_site_without_submodules() -> None:
    _validate_pages_workflow()


def test_pages_workflow_is_compact() -> None:
    workflow = Path(".github/workflows/pages.yml").read_text(encoding="utf-8")

    assert len(workflow.splitlines()) <= 30
    assert "workflow_dispatch" not in workflow
    assert "concurrency:" not in workflow


@pytest.mark.parametrize(
    "old, new",
    (
        ("branches: [main]", "branches: [develop]"),
        ("actions/checkout@v4", "example/checkout@v4"),
        ("actions/configure-pages@v5", "example/configure-pages@v5"),
        ("actions/upload-pages-artifact@v3", "example/upload-pages-artifact@v3"),
        ("actions/deploy-pages@v4", "example/deploy-pages@v4"),
        ("submodules: false", "submodules: true"),
        ("submodules: false", "submodules: recursive"),
        ("path: site", "path: ."),
        ("  contents: read", "  contents: write"),
        ("  pages: write", "  pages: read"),
        ("  id-token: write", "  id-token: read"),
        ("permissions:\n  contents: read\n  pages: write\n  id-token: write", "permissions: read"),
        ("    runs-on: ubuntu-latest", "    runs-on: macos-latest"),
        ("      name: github-pages", "      name: production"),
        (
            "      url: ${{ steps.deployment.outputs.page_url }}",
            "      url: https://example.test",
        ),
        (
            "    environment:\n      name: github-pages\n"
            "      url: ${{ steps.deployment.outputs.page_url }}",
            "    environment: github-pages",
        ),
    ),
)
def test_pages_workflow_structure_rejects_changed_release_contract(
    tmp_path: Path,
    old: str,
    new: str,
) -> None:
    source = Path(".github/workflows/pages.yml").read_text(encoding="utf-8")
    workflow = tmp_path / "pages.yml"
    workflow.write_text(source.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(AssertionError):
        _validate_pages_workflow(workflow)


def test_public_site_release_contract() -> None:
    validate_public_site_release()

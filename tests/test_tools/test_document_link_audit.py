# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Document-link audit tests.

"""Tests for deterministic local and bounded external link governance."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from tools.document_link_audit import (
    DEFAULT_POLICY,
    ROOT,
    SCHEMA,
    ExternalResult,
    _read_policy,
    _tool_sha256,
    audit_external,
    audit_local,
    audit_site,
    extract_links,
    public_sources,
)


def _git_track(root: Path, *paths: str) -> None:
    """Initialize a real local index and stage only the fixture's named paths."""
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)  # noqa: S603
    subprocess.run(["git", "add", "--", *paths], cwd=root, check=True)  # noqa: S603


def test_live_public_document_links_are_locally_resolvable() -> None:
    """The current tracked public source graph has no broken local reference."""
    policy = _read_policy(DEFAULT_POLICY)

    findings, refs = audit_local(ROOT, policy)

    assert findings == ()
    assert len(refs) >= 800


def test_markdown_extraction_ignores_code_and_preserves_source_lines(tmp_path: Path) -> None:
    """Commands do not become links while prose references retain line provenance."""
    source = tmp_path / "README.md"
    source.write_text(
        "# Demo\n\n```text\n[ignored](missing.md)\n```\n\n[kept](target.md#result)\n",
        encoding="utf-8",
    )

    refs = extract_links(source, tmp_path)

    assert [(ref.line, ref.target, ref.kind) for ref in refs] == [(7, "target.md#result", "markdown")]


def test_tex_extraction_does_not_recrawl_structured_url_labels(tmp_path: Path) -> None:
    """Structured LaTeX links produce only their URL, never ``}{label}`` debris."""
    source = tmp_path / "paper.tex"
    source.write_text(
        "\\href{https://example.test/paper}{Paper label}\n\\url{https://example.test/source}\n",
        encoding="utf-8",
    )

    refs = extract_links(source, tmp_path)

    assert [(ref.line, ref.target, ref.kind) for ref in refs] == [
        (1, "https://example.test/paper", "tex-url"),
        (2, "https://example.test/source", "tex-url"),
    ]


def test_local_audit_rejects_missing_target_anchor_and_secret_query(tmp_path: Path) -> None:
    """Broken files, stale anchors, and credential-shaped URLs all fail closed."""
    readme = tmp_path / "README.md"
    target = tmp_path / "target.md"
    readme.write_text(
        "[missing](absent.md)\n"
        "[stale](target.md#old)\n"
        "[secret](https://example.test/x?token=value)\n"
        "[private](http://127.0.0.1/status)\n",
        encoding="utf-8",
    )
    target.write_text("# Current\n", encoding="utf-8")
    _git_track(tmp_path, "README.md", "target.md")

    findings, _ = audit_local(tmp_path, _read_policy(DEFAULT_POLICY))

    assert [finding.reason for finding in findings] == [
        "relative target does not exist",
        "Markdown anchor does not exist",
        "URL contains secret-bearing query key(s): token",
        "URL targets a non-public IP address",
    ]
    assert findings[2].target == "https://example.test/x?token=%5BREDACTED%5D"


def test_mkdocs_navigation_resolves_from_docs_root(tmp_path: Path) -> None:
    """Only the nav block is interpreted and its pages resolve beneath docs/."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "index.md").write_text("# Home\n", encoding="utf-8")
    (tmp_path / "mkdocs.yml").write_text(
        "theme:\n  palette:\n    - scheme: default\nnav:\n  - Home: index.md\nmarkdown_extensions:\n  - tables\n",
        encoding="utf-8",
    )
    _git_track(tmp_path, "docs/index.md", "mkdocs.yml")

    findings, refs = audit_local(tmp_path, _read_policy(DEFAULT_POLICY))

    assert findings == ()
    assert [ref.target for ref in refs if ref.kind == "mkdocs-nav"] == ["index.md"]


def test_local_audit_rejects_public_mkdocs_orphan(tmp_path: Path) -> None:
    """A tracked public docs page must be intentionally reachable in navigation."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "index.md").write_text("# Home\n", encoding="utf-8")
    (tmp_path / "docs" / "orphan.md").write_text("# Orphan\n", encoding="utf-8")
    (tmp_path / "mkdocs.yml").write_text("nav:\n  - Home: index.md\n", encoding="utf-8")
    _git_track(tmp_path, "docs/index.md", "docs/orphan.md", "mkdocs.yml")

    findings, _ = audit_local(tmp_path, _read_policy(DEFAULT_POLICY))

    assert [(finding.source, finding.reason) for finding in findings] == [
        ("docs/orphan.md", "public page is absent from MkDocs nav")
    ]


def test_rendered_site_audit_rejects_missing_internal_asset(tmp_path: Path) -> None:
    """Generated HTML navigation cannot point to a missing site artifact."""
    index = tmp_path / "index.html"
    index.write_text('<a href="guide/">Guide</a><img src="assets/missing.svg">', encoding="utf-8")

    findings = audit_site(tmp_path)

    assert [(finding.target, finding.reason) for finding in findings] == [
        ("assets/missing.svg", "rendered target does not exist"),
        ("guide/", "rendered target does not exist"),
    ]


def test_rendered_site_audit_maps_configured_public_base_path(tmp_path: Path) -> None:
    """Root-relative MkDocs links resolve after removing the deployment prefix."""
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "main.css").write_text("", encoding="utf-8")
    (tmp_path / "index.html").write_text(
        '<link href="/scpn-control/assets/main.css"><a href="/scpn-control/">Home</a>',
        encoding="utf-8",
    )

    assert audit_site(tmp_path, "/scpn-control/") == ()


def test_external_audit_reuses_fresh_provenanced_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh cached result prevents an unnecessary network request."""
    url = "https://example.test/reference"
    policy = _read_policy(DEFAULT_POLICY)
    cache = tmp_path / "cache.json"
    cache.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA,
                "provenance": {
                    "policy_sha256": policy.source_sha256,
                    "tool_sha256": _tool_sha256(),
                },
                "results": [
                    {
                        "url": url,
                        "classification": "reachable",
                        "status_code": 200,
                        "attempts": 1,
                        "checked_at": datetime.now(UTC).isoformat(),
                        "final_url": url,
                        "detail": "HTTP response",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def unexpected_request(_url: str, _policy: object) -> ExternalResult:
        """Reject any request attempted despite a current matching cache record."""
        raise AssertionError("fresh cache must suppress network access")

    monkeypatch.setattr("tools.document_link_audit._check_external", unexpected_request)

    results = audit_external((url,), policy, cache)

    assert len(results) == 1
    assert results[0].cached is True
    assert results[0].classification == "reachable"


def test_external_audit_retries_transient_response_then_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Transient HTTP responses receive only the configured bounded retry."""
    policy = replace(
        _read_policy(DEFAULT_POLICY),
        retries=1,
        retry_backoff_seconds=0.0,
        per_host_delay_seconds=0.0,
    )
    responses = iter(((503, "https://example.test/reference"), (200, "https://example.test/reference")))

    def request_once(_url: str, _policy: object, _method: str) -> tuple[int, str]:
        """Supply the retained transient/positive sequence to the historical test."""
        return next(responses)

    monkeypatch.setattr("tools.document_link_audit._request_once", request_once)

    results = audit_external(("https://example.test/reference",), policy, tmp_path / "missing-cache.json")

    assert [(result.classification, result.attempts, result.status_code) for result in results] == [
        ("reachable", 2, 200)
    ]


def _audit_cli(root: Path, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run the defining script with explicit fixture root and unchanged policy."""
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/document_link_audit.py"),
            "--root",
            str(root),
            "--policy",
            str(DEFAULT_POLICY),
            *args,
        ],
        cwd=cwd or root,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )


def test_cli_existing_untracked_target_refuses_until_real_index_add(tmp_path: Path) -> None:
    """Cold CLI distinguishes existing bytes from actual Git index membership."""
    (tmp_path / "README.md").write_text("[target](target.md#result)\n", encoding="utf-8")
    (tmp_path / "target.md").write_text("# Result\n", encoding="utf-8")
    _git_track(tmp_path, "README.md")
    report = tmp_path / "report.json"
    first = _audit_cli(tmp_path, "--json-out", str(report))
    assert first.returncode == 1
    assert [f["reason"] for f in json.loads(report.read_text())["findings"]] == ["relative target is not tracked"]
    _git_track(tmp_path, "target.md")
    second = _audit_cli(tmp_path, "--json-out", str(report), cwd=tmp_path.parent)
    assert second.returncode == 0
    assert json.loads(report.read_text())["findings"] == []


def test_cli_lists_screened_deduplicated_urls_without_http_results(tmp_path: Path) -> None:
    """Listing records no HTTP observation for actual prose URLs and fragments."""
    (tmp_path / "README.md").write_text(
        "[one](https://reference.invalid/item#first)\n"
        "[two](https://reference.invalid/item#second)\n"
        "[private](http://127.0.0.1/state)\n",
        encoding="utf-8",
    )
    _git_track(tmp_path, "README.md")
    report = tmp_path / "report.json"
    result = _audit_cli(tmp_path, "--list-external", "--json-out", str(report))
    assert result.returncode == 1
    assert result.stdout.splitlines()[0] == "https://reference.invalid/item"
    payload = json.loads(report.read_text())
    assert payload["results"] == []
    assert len(payload["findings"]) == 1
    assert payload["findings"][0]["reason"] == "URL targets a non-public IP address"


def test_public_source_selection_uses_index_names_without_disk_walk(tmp_path: Path) -> None:
    """Untracked/private sources are omitted while a missing indexed file remains."""
    (tmp_path / "docs/internal").mkdir(parents=True)
    for rel in ("README.md", "missing.md", "untracked.md", "docs/internal/notes.md"):
        (tmp_path / rel).write_text("# Source\n", encoding="utf-8")
    _git_track(tmp_path, "README.md", "missing.md", "docs/internal/notes.md")
    (tmp_path / "missing.md").unlink()
    selected = public_sources(tmp_path, _read_policy(DEFAULT_POLICY))
    assert tuple(p.relative_to(tmp_path).as_posix() for p in selected) == ("README.md", "missing.md")


def test_cli_rendered_site_checks_files_but_not_html_fragments(tmp_path: Path) -> None:
    """An actual site refuses absent assets and accepts existing fragment targets."""
    (tmp_path / "README.md").write_text("# Source\n", encoding="utf-8")
    _git_track(tmp_path, "README.md")
    site = tmp_path / "rendered"
    site.mkdir()
    (site / "index.html").write_text('<a href="guide.html#absent">Guide</a>', encoding="utf-8")
    first = _audit_cli(tmp_path, "--site-dir", str(site))
    assert first.returncode == 1 and "rendered target does not exist" in first.stdout
    (site / "guide.html").write_text("<p>Guide</p>", encoding="utf-8")
    second = _audit_cli(tmp_path, "--site-dir", str(site))
    assert second.returncode == 0
    assert audit_site(tmp_path / "missing-site") == ()


def test_external_cache_cli_retains_original_restricted_observation(tmp_path: Path) -> None:
    """Cold external CLI reuses an actual matching cache without changing its time."""
    url = "https://reference.invalid/item"
    (tmp_path / "README.md").write_text(f"[source]({url})\n", encoding="utf-8")
    _git_track(tmp_path, "README.md")
    checked_at = datetime.now(UTC).isoformat()
    cache = tmp_path / "cache.json"
    cache.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA,
                "provenance": {
                    "policy_sha256": _read_policy(DEFAULT_POLICY).source_sha256,
                    "tool_sha256": _tool_sha256(),
                },
                "results": [
                    {
                        "url": url,
                        "classification": "restricted",
                        "status_code": 403,
                        "attempts": 1,
                        "checked_at": checked_at,
                        "final_url": url,
                        "detail": "HTTP response",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = _audit_cli(tmp_path, "--external", "--cache", str(cache))
    assert result.returncode == 0
    row = json.loads(cache.read_text())["results"][0]
    assert row["cached"] is True and row["checked_at"] == checked_at
    assert row["classification"] == "restricted" and row["status_code"] == 403


def test_real_external_request_failure_uses_authored_report_detail(tmp_path: Path) -> None:
    """A real unsuccessful URL request records no interpreter exception text."""
    policy = replace(
        _read_policy(DEFAULT_POLICY),
        retries=0,
        timeout_seconds=0.5,
        per_host_delay_seconds=0.0,
        retry_backoff_seconds=0.0,
    )
    results = audit_external(("https://scpn-control-unresolvable.invalid/",), policy, tmp_path / "absent-cache.json")
    assert len(results) == 1
    assert results[0].classification == "transient" and results[0].attempts == 1
    assert results[0].status_code is None
    assert results[0].detail == "Public URL request failed."


def test_cli_missing_indexed_source_preserves_existing_report(tmp_path: Path) -> None:
    """A native file-read failure cannot replace an earlier report with success."""
    (tmp_path / "README.md").write_text("# Source\n", encoding="utf-8")
    _git_track(tmp_path, "README.md")
    (tmp_path / "README.md").unlink()
    report = tmp_path / "report.json"
    report.write_bytes(b"retained original report\n")
    result = _audit_cli(tmp_path, "--json-out", str(report))
    assert result.returncode != 0
    assert "Document link audit passed" not in result.stdout
    assert report.read_bytes() == b"retained original report\n"


def test_report_source_set_hash_does_not_certify_source_contents(tmp_path: Path) -> None:
    """Equal filenames retain the set hash across different source text bytes."""
    readme = tmp_path / "README.md"
    readme.write_text("# First source\n", encoding="utf-8")
    _git_track(tmp_path, "README.md")
    report = tmp_path / "report.json"
    assert _audit_cli(tmp_path, "--json-out", str(report)).returncode == 0
    first = json.loads(report.read_text())
    readme.write_text("# Different source\n", encoding="utf-8")
    assert _audit_cli(tmp_path, "--json-out", str(report)).returncode == 0
    second = json.loads(report.read_text())
    assert first["provenance"]["source_set_sha256"] == second["provenance"]["source_set_sha256"]
    assert first["source_count"] == second["source_count"] == 1


def test_repository_loop_refuses_before_network_or_report_work(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refuse an actual root loop with external/report flags before either operation."""
    from tools import document_link_audit as link_tool

    loop = tmp_path / "repository-loop"
    loop.symlink_to(loop.name, target_is_directory=True)
    output = tmp_path / "report.json"
    output.write_bytes(b"retained preexisting report\n")
    arguments = ["--root", str(loop), "--external", "--json-out", str(output)]
    assert link_tool.main(arguments) == 2
    captured = capsys.readouterr()
    assert captured.err == "Document link audit refused: repository root could not be resolved\n"
    assert not captured.out
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/document_link_audit.py"), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2 and result.stderr == captured.err and not result.stdout
    assert output.read_bytes() == b"retained preexisting report\n"


def test_actual_indexed_document_graph_checks_escape_and_url_boundaries(tmp_path: Path) -> None:
    """Inspect altered actual prose through the public graph and cold CLI without HTTP."""
    original = (ROOT / "docs/tglf_flux.md").read_text(encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(
        original
        + "\n[escape](../outside.md)\n[directory](assets/)\n[empty](<>)\n"
        + "[hostless](https:///missing-host)\n[internal](https://node.internal/reference)\n"
        + "[user-info](https://test-user:test-value@reference.invalid:not-a-port/report)\n"
        + "[global-ip](https://1.1.1.1/reference)\n[network-relative](//reference.invalid/item)\n",
        encoding="utf-8",
    )
    (tmp_path / "assets").mkdir()
    _git_track(tmp_path, "README.md")
    findings, refs = audit_local(tmp_path, _read_policy(DEFAULT_POLICY))
    reasons = {finding.reason for finding in findings}
    assert reasons == {
        "relative target escapes repository root",
        "URL has no public host",
        "URL targets a non-public host",
        "URL contains user-info credentials",
    }
    assert not any(ref.target == "" for ref in refs)
    assert not any(finding.target == "assets/" for finding in findings)
    result = _audit_cli(tmp_path)
    assert result.returncode == 1 and "test-value" not in result.stdout
    assert "relative target escapes repository root" in result.stdout


def test_rendered_site_path_boundaries_use_real_files(tmp_path: Path) -> None:
    """Accept present base-root targets and ignore foreign roots while refusing escapes."""
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text(
        '<a href="/docs">Base root</a><a href="/elsewhere/page">Other deployment</a>'
        '<a href="">Empty</a><a href="../outside.html">Escape</a>',
        encoding="utf-8",
    )
    findings = audit_site(site, "/docs/")
    assert [(finding.target, finding.reason) for finding in findings] == [
        ("../outside.html", "rendered target escapes site root")
    ]
    (site / "index.html").write_text('<a href="/index.html">Existing</a>', encoding="utf-8")
    assert audit_site(site) == ()


def test_configured_source_glob_exclusion_precedes_suffix_admission(tmp_path: Path) -> None:
    """A real indexed auxiliary file stays excluded even when its suffix is allowed."""
    policy = _read_policy(DEFAULT_POLICY)
    policy = replace(policy, include_suffixes=(*policy.include_suffixes, ".aux"))
    (tmp_path / "papers").mkdir()
    excluded = tmp_path / "papers" / "document.aux"
    excluded.write_bytes((ROOT / "docs/tglf_flux.md").read_bytes())
    _git_track(tmp_path, "papers/document.aux")
    assert public_sources(tmp_path, policy) == ()


@pytest.mark.parametrize("retry", [-1, True, False, 1.5, "2", None])
def test_public_policy_refuses_non_integer_or_negative_retry_budget(retry: object) -> None:
    """A public policy cannot suppress all HTTP work or coerce another scalar type."""
    with pytest.raises(ValueError, match="retries must be a nonnegative integer"):
        replace(_read_policy(DEFAULT_POLICY), retries=cast(int, retry))


@pytest.mark.parametrize("retry", ["-1", "true", "false", "1.5", '"2"'])
def test_invalid_native_retry_policy_refuses_before_cache_replacement(tmp_path: Path, retry: str) -> None:
    """A malformed actual TOML budget refuses the cold CLI before reports or requests."""
    (tmp_path / "README.md").write_text("[reference](https://reference.invalid/item)\n", encoding="utf-8")
    _git_track(tmp_path, "README.md")
    policy = tmp_path / "policy.toml"
    policy.write_text(
        DEFAULT_POLICY.read_text(encoding="utf-8").replace("retries = 2", f"retries = {retry}"),
        encoding="utf-8",
    )
    cache = tmp_path / "cache.json"
    original = b"retained actual cache carrier\n"
    cache.write_bytes(original)
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/document_link_audit.py"),
            "--root",
            str(tmp_path),
            "--policy",
            str(policy),
            "--external",
            "--cache",
            str(cache),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode != 0 and "retries must be a nonnegative integer" in process.stderr
    assert "Document link audit passed" not in process.stdout and cache.read_bytes() == original


def test_cli_site_url_base_and_same_page_query_are_resolved(tmp_path: Path) -> None:
    """Use actual site_url metadata and generated-file paths through the cold audit CLI."""
    (tmp_path / "README.md").write_text("# Reference\n", encoding="utf-8")
    (tmp_path / "mkdocs.yml").write_text("site_url: https://reference.invalid/docs/\n", encoding="utf-8")
    _git_track(tmp_path, "README.md", "mkdocs.yml")
    site = tmp_path / "rendered"
    site.mkdir()
    (site / "index.html").write_text('<a href="/docs/index.html">Home</a><a href="?filter=all">Filter</a>')
    result = _audit_cli(tmp_path, "--site-dir", str(site))
    assert result.returncode == 0 and not result.stderr

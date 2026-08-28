# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Sotek. All rights reserved.
# © Code 2020–2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Document-link audit tests.

"""Tests for deterministic local and bounded external link governance."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

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
)


def _git_track(root: Path, *paths: str) -> None:
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
                        "checked_at": datetime.now(timezone.utc).isoformat(),
                        "final_url": url,
                        "detail": "HTTP response",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def unexpected_request(_url: str, _policy: object) -> ExternalResult:
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
        return next(responses)

    monkeypatch.setattr("tools.document_link_audit._request_once", request_once)

    results = audit_external(("https://example.test/reference",), policy, tmp_path / "missing-cache.json")

    assert [(result.classification, result.attempts, result.status_code) for result in results] == [
        ("reachable", 2, 200)
    ]

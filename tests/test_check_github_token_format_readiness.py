#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — GitHub Token Format Readiness Guard Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations

from tools.check_github_token_format_readiness import scan_text


def _categories(text: str) -> set[str]:
    return {finding.category for finding in scan_text("sample.py", text)}


def test_rejects_legacy_exact_ghs_regex() -> None:
    text = r'pattern = r"ghs_[A-Za-z0-9]{36}"'

    assert "brittle-ghs-regex" in _categories(text)


def test_rejects_exact_installation_token_length_check() -> None:
    text = "if len(installation_token) == 40:\n    pass\n"

    assert "fixed-token-length" in _categories(text)


def test_rejects_small_token_storage_column() -> None:
    text = "github_installation_token VARCHAR(255) NOT NULL"

    assert "small-token-storage" in _categories(text)


def test_rejects_installation_token_endpoint_without_override_header() -> None:
    text = 'requests.post("https://api.github.com/app/installations/1/access_tokens")'

    assert "missing-stateless-token-override-header" in _categories(text)


def test_accepts_github_recommended_opaque_pattern_and_override_header() -> None:
    text = (
        r'pattern = r"ghs_[A-Za-z0-9\._]{36,}"'
        "\n"
        'requests.post("https://api.github.com/app/installations/1/access_tokens", '
        'headers={"X-GitHub-Stateless-S2S-Token": "enabled"})\n'
        "if installation_token.startswith('ghs_'):\n"
        "    pass\n"
    )

    assert scan_text("sample.py", text) == []

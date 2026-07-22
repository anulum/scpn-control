# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — FAIR-MAST Source Policy Tests
"""Tests for the single-source FAIR-MAST provenance policy."""

from __future__ import annotations

from validation.fair_mast_source_policy import (
    FAIR_MAST_CATALOG_URL,
    FAIR_MAST_CITATION,
    FAIR_MAST_CITATIONS,
    FAIR_MAST_LICENCE,
    FAIR_MAST_LICENCE_URL,
    fair_mast_provenance,
)


def test_fair_mast_policy_matches_official_licence_and_citations() -> None:
    assert FAIR_MAST_LICENCE == "CC-BY-SA-4.0"
    assert FAIR_MAST_LICENCE_URL == "https://creativecommons.org/licenses/by-sa/4.0/"
    assert FAIR_MAST_CATALOG_URL == "https://mastapp.site/"
    assert len(FAIR_MAST_CITATIONS) == 2
    assert "10.1109/TPS.2025.3583419" in FAIR_MAST_CITATIONS[0]
    assert "10.1016/j.softx.2024.101869" in FAIR_MAST_CITATIONS[1]
    assert FAIR_MAST_CITATION == "; ".join(FAIR_MAST_CITATIONS)


def test_fair_mast_provenance_returns_an_independent_json_block() -> None:
    first = fair_mast_provenance()
    second = fair_mast_provenance()
    assert first == {
        "licence": FAIR_MAST_LICENCE,
        "licence_url": FAIR_MAST_LICENCE_URL,
        "citation": FAIR_MAST_CITATION,
        "citations": list(FAIR_MAST_CITATIONS),
        "source_policy_url": FAIR_MAST_CATALOG_URL,
    }
    assert first is not second
    assert first["citations"] is not second["citations"]

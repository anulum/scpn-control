# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Roadmap release-boundary tests.
"""Regression tests for release-status boundaries in the roadmap."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v1_release_boundary_is_historical_not_operational() -> None:
    """The public release record must not turn an untagged version into a plan."""
    roadmap = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    assert "No `v1.0.0` tag exists" in roadmap
    assert "future release date" in roadmap
    assert "## Next" not in roadmap
    assert "- [ ]" not in roadmap


def test_public_release_record_links_claim_authorities() -> None:
    """Current boundaries point to generated public evidence authorities."""
    roadmap = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    assert "docs/_generated/capability_manifest.json" in roadmap
    assert "docs/physics_traceability.md" in roadmap
    assert "validation/physics_traceability.json" in roadmap
    assert "contribution interfaces, not an internal execution order" in roadmap

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


def test_v1_roadmap_target_is_not_listed_as_shipped() -> None:
    """The untagged v1.0.0 target must stay outside shipped history."""

    roadmap = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    shipped, next_targets = roadmap.split("## Next / Future Release Targets", maxsplit=1)

    assert "### v1.0.0" not in shipped
    assert "### v1.0.0 — Production readiness target (not tagged)" in next_targets
    assert "No `v1.0.0` tag exists yet; latest released tag is `v0.23.0`." in next_targets


def test_v1_target_has_no_completed_release_checkmarks() -> None:
    """The future v1.0.0 release target must not mark gates complete."""

    roadmap = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    v1_block = roadmap.split("### v1.0.0 — Production readiness target (not tagged)", maxsplit=1)[1]
    v1_block = v1_block.split("### Remaining production work", maxsplit=1)[0]

    assert "- [x]" not in v1_block
    assert "- [ ] Confirm JOSS paper citation count" in v1_block

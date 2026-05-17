# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Dashboard GK state tests
# © 1998–2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────
"""Tests for Streamlit-independent gyrokinetic dashboard state helpers."""

from __future__ import annotations

import pytest

from dashboard.gk_state import dominant_mode_from_types


def test_dominant_mode_from_types_ignores_stable_entries() -> None:
    assert dominant_mode_from_types(["stable", "ITG", "TEM", "ITG", "stable"]) == "ITG"


def test_dominant_mode_from_types_returns_stable_when_no_unstable_mode() -> None:
    assert dominant_mode_from_types(["stable", "stable"]) == "stable"
    assert dominant_mode_from_types([]) == "stable"


def test_dominant_mode_from_types_tie_breaks_by_first_observed_unstable_mode() -> None:
    assert dominant_mode_from_types(["TEM", "ITG", "ITG", "TEM"]) == "TEM"


@pytest.mark.parametrize("mode_types", [[""], ["ITG", ""], ["stable", "  "]])
def test_dominant_mode_from_types_rejects_blank_mode_names(mode_types: list[str]) -> None:
    with pytest.raises(ValueError, match="mode"):
        dominant_mode_from_types(mode_types)

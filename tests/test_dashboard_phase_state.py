# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Dashboard phase state tests
# © 1998–2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────
"""Tests for Streamlit-independent dashboard phase-sync state helpers."""

from __future__ import annotations

import math

import pytest

from dashboard.phase_state import bridge_coupling_indicators


def test_bridge_coupling_indicators_match_dashboard_formulae() -> None:
    indicators = bridge_coupling_indicators(0.5)

    assert indicators.p0_p1_turb_zonal == pytest.approx(0.5 * (1.0 + 0.5 * math.tanh(0.5 / 0.2)))
    assert indicators.p1_p4_zonal_barrier == pytest.approx(0.5 * (1.0 + 0.3 * 0.5))
    assert indicators.p3_p4_elm_barrier == pytest.approx(0.5 * (1.0 + 0.4 * (0.5 - 0.5)))


def test_bridge_coupling_indicators_clip_only_zonal_barrier_path() -> None:
    high = bridge_coupling_indicators(3.0)
    low = bridge_coupling_indicators(-1.0)

    assert high.p1_p4_zonal_barrier == pytest.approx(0.8)
    assert low.p1_p4_zonal_barrier == pytest.approx(0.5)
    assert high.p3_p4_elm_barrier > low.p3_p4_elm_barrier


@pytest.mark.parametrize("r_global", [math.nan, math.inf, -math.inf])
def test_bridge_coupling_indicators_reject_nonfinite_coherence(r_global: float) -> None:
    with pytest.raises(ValueError, match="R_global"):
        bridge_coupling_indicators(r_global)

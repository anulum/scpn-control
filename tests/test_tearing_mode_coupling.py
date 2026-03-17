# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851  Contact: protoscience@anulum.li
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.tearing_mode_coupling import (
    ChirikovOverlap,
    CoupledTearingModes,
    DisruptionTriggerAssessment,
    SawtoothNTMSeeding,
    TearingModeStabilityMap,
)


def test_chirikov_overlap():
    assert ChirikovOverlap.parameter(0.1, 0.1, 0.2) == 0.5
    assert not ChirikovOverlap.is_stochastic(0.5)

    assert np.isclose(ChirikovOverlap.parameter(0.3, 0.3, 0.2), 1.5)
    assert ChirikovOverlap.is_stochastic(1.5)


def test_sawtooth_seeding():
    st = SawtoothNTMSeeding(None)
    amp = st.seed_amplitude(crash_energy_MJ=4.0, r_s=0.5)
    assert np.isclose(amp, 0.1)

    prob_low = st.seed_probability(1.0, threshold=2.0)
    assert prob_low == 0.0

    prob_high = st.seed_probability(3.0, threshold=2.0)
    assert prob_high > 0.0


def test_coupled_tearing_modes():
    c = CoupledTearingModes((3, 2), (2, 1), 0.5, 0.8, 2.0, 6.2, 5.3)

    res_no_seed = c.evolve(1e-6, 1e-6, j_bs=1e5, j_phi=1e6, eta=1e-7, dt=0.01, n_steps=200)
    assert not res_no_seed.disruption
    assert res_no_seed.w1_trace[-1] < 0.1
    assert res_no_seed.w2_trace[-1] < 0.1

    res_seed = c.evolve(
        1e-6,
        2e-3,
        j_bs=8e5,
        j_phi=1e6,
        eta=1e-5,
        dt=0.01,
        n_steps=1000,
        seed_time=0.1,
        seed_amplitude=0.1,
    )
    assert res_seed.disruption
    assert res_seed.overlap_time >= 0.0
    assert np.any(res_seed.chirikov_trace > 1.0)


def test_disruption_trigger_assessment():
    c = CoupledTearingModes((3, 2), (2, 1), 0.5, 0.8, 2.0, 6.2, 5.3)
    ass = DisruptionTriggerAssessment(c)

    path = ass.run_scenario(j_bs=1e6, j_phi=1e6, omega_phi=1e4, seed_energy=10.0)

    if path.warning_time_ms > 0:
        assert path.avoidable


def test_stability_map():
    smap = TearingModeStabilityMap()
    b_range = np.linspace(1.0, 5.0, 10)
    li_range = np.linspace(0.5, 1.5, 10)

    res = smap.scan_beta_li(b_range, li_range)

    assert res.shape == (10, 10)
    assert res[0, 0] == 1  # low beta × low li → stable
    assert res[-1, -1] == -1  # high beta × high li → unstable


def test_chirikov_overlap_stochastic():
    """σ > 1 when w1 + w2 > 2 Δr; Chirikov 1979, Phys. Rep. 52, 263, Eq. 3.1."""
    delta_r = 0.15

    # Just below threshold
    sigma_sub = ChirikovOverlap.parameter(0.1, 0.1, delta_r)
    assert sigma_sub < 1.0
    assert not ChirikovOverlap.is_stochastic(sigma_sub)

    # Just above threshold: w1 + w2 = 0.31 > 2 × 0.15 = 0.30
    sigma_super = ChirikovOverlap.parameter(0.16, 0.15, delta_r)
    assert sigma_super > 1.0
    assert ChirikovOverlap.is_stochastic(sigma_super)

    # Exact boundary σ = 1
    sigma_exact = ChirikovOverlap.parameter(delta_r, delta_r, delta_r)
    assert np.isclose(sigma_exact, 1.0)
    assert not ChirikovOverlap.is_stochastic(sigma_exact)


def test_coupling_coefficient_bounded():
    """Coupling coefficient < 1 for well-separated islands with a/R₀ < 1.

    La Haye & Buttery 2009, Phys. Plasmas 16, 022107, Eq. 8:
        C₁₂ ≈ 0.5 (a/R₀).
    For all realistic tokamaks ε = a/R₀ < 1, so C₁₂ < 0.5 < 1.
    """
    for a, R0 in [(2.0, 6.2), (0.67, 1.85), (1.0, 3.0), (0.5, 1.7)]:
        c = CoupledTearingModes((3, 2), (2, 1), 0.5, 0.8, a, R0, 5.3)
        coeff = c.coupling_coefficient(3, 2, 2, 1)
        assert coeff < 1.0, f"Coupling must be < 1, got {coeff} for a={a}, R0={R0}"
        assert coeff > 0.0, "Coupling must be positive"


@pytest.mark.parametrize(
    "w1,w2,delta_r,expected_stochastic",
    [
        (0.05, 0.05, 0.20, False),  # σ = 0.5 < 1
        (0.40, 0.40, 0.20, True),  # σ = 2.0 > 1
        (0.10, 0.10, 0.10, True),  # σ = 1.0 — boundary; not > 1
        (0.11, 0.10, 0.10, True),  # σ > 1
    ],
)
def test_chirikov_parametric(w1: float, w2: float, delta_r: float, expected_stochastic: bool) -> None:
    """Parametric Chirikov criterion; Chirikov 1979, Eq. 3.1."""
    sigma = ChirikovOverlap.parameter(w1, w2, delta_r)
    # Boundary σ = 1.0: is_stochastic returns False (strict inequality)
    is_s = ChirikovOverlap.is_stochastic(sigma)
    if w1 == 0.10 and w2 == 0.10 and delta_r == 0.10:
        assert not is_s  # exactly 1.0 is not stochastic
    else:
        assert is_s == expected_stochastic

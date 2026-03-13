# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Coupled Tearing Mode Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np

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
    assert np.isclose(amp, 0.1)  # 0.05 * 2

    prob_low = st.seed_probability(1.0, threshold=2.0)
    assert prob_low == 0.0

    prob_high = st.seed_probability(3.0, threshold=2.0)
    assert prob_high > 0.0


def test_coupled_tearing_modes():
    c = CoupledTearingModes((3, 2), (2, 1), 0.5, 0.8, 2.0, 6.2, 5.3)

    # Evolve without seed -> no disruption
    res_no_seed = c.evolve(1e-6, 1e-6, j_bs=1e5, j_phi=1e6, eta=1e-7, dt=0.01, n_steps=200)
    assert not res_no_seed.disruption

    # Evolve with large seed -> disruption via overlap
    # We give w2_0 a small seed (e.g. 2e-3) so bootstrap term isn't completely suppressed by the w_d threshold
    res_seed = c.evolve(
        1e-6, 2e-3, j_bs=8e5, j_phi=1e6, eta=1e-5, dt=0.01, n_steps=1000, seed_time=0.1, seed_amplitude=0.1
    )

    # A large 3/2 seed coupled with high bootstrap should trigger 2/1 and overlap
    if res_seed.disruption:
        assert res_seed.overlap_time >= 0.0
    else:
        # If it doesn't disrupt, at least w1 and w2 should have grown
        assert res_seed.w1_trace[-1] > 0.1
        assert res_seed.w2_trace[-1] > 1e-4


def test_disruption_trigger_assessment():
    c = CoupledTearingModes((3, 2), (2, 1), 0.5, 0.8, 2.0, 6.2, 5.3)
    # Locked mode not strictly necessary for this logic mock
    ass = DisruptionTriggerAssessment(c, None)

    path = ass.run_scenario(j_bs=1e6, j_phi=1e6, omega_phi=1e4, seed_energy=10.0)

    if path.warning_time_ms > 0:
        assert path.avoidable


def test_stability_map():
    smap = TearingModeStabilityMap()
    b_range = np.linspace(1.0, 5.0, 10)
    li_range = np.linspace(0.5, 1.5, 10)

    res = smap.scan_beta_li(b_range, li_range)

    assert res.shape == (10, 10)
    assert res[0, 0] == 1  # Low beta, low li -> stable
    assert res[-1, -1] == -1  # High beta, high li -> unstable

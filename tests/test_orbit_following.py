# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851  Contact: protoscience@anulum.li
from __future__ import annotations

import math

import numpy as np

from scpn_control.core.orbit_following import (
    GuidingCenterOrbit,
    MonteCarloEnsemble,
    OrbitClassifier,
    SlowingDown,
    banana_orbit_width,
    first_orbit_loss,
)


def mock_b_field(R, Z):
    # Pure toroidal field (no poloidal -> no drifts in passing)
    B0 = 5.0
    R0 = 6.0
    B_phi = B0 * R0 / R
    # Add small poloidal field for trapping
    B_R = -0.1 * Z
    B_Z = 0.1 * (R - R0)
    return B_R, B_Z, B_phi


def test_passing_orbit():
    # Pitch angle 0 -> purely parallel -> passing
    orbit = GuidingCenterOrbit(4.0, 2, 3500.0, 0.0, 6.2, 0.0)

    # Evolve a bit
    for _ in range(10):
        orbit.step(mock_b_field, 1e-6)

    assert orbit.v_par > 0.0  # Never reversed
    assert orbit.R > 0.0


def test_trapped_orbit():
    # Pitch angle pi/2 -> purely perpendicular -> trapped instantly
    orbit = GuidingCenterOrbit(4.0, 2, 3500.0, math.pi / 2 - 0.1, 6.2, 0.0)

    # We must have computed mu
    orbit.step(mock_b_field, 1e-6)
    assert orbit.mu > 0.0

    # In a full simulation it would reverse v_par


def test_orbit_classifier():
    R = np.ones(10) * 6.0
    Z = np.zeros(10)
    v_par_pass = np.ones(10)
    v_par_trap = np.array([1.0, 0.5, -0.5, -1.0, -0.5, 0.5, 1.0, 1.0, 1.0, 1.0])

    assert OrbitClassifier.classify(R, Z, v_par_pass, 10.0, 5.0) == "passing"
    assert OrbitClassifier.classify(R, Z, v_par_trap, 10.0, 5.0) == "trapped"

    # Lost
    R_lost = np.array([6.0, 7.0, 11.0, 12.0])
    v_lost = np.ones(4)
    assert OrbitClassifier.classify(R_lost, Z[:4], v_lost, 10.0, 5.0) == "lost"


def test_first_orbit_loss():
    # ITER: large Ip -> low loss
    iter_loss = first_orbit_loss(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0)
    assert iter_loss < 0.05

    # NSTX: small Ip, small a -> higher loss
    nstx_loss = first_orbit_loss(R0=0.9, a=0.6, B0=1.0, Ip_MA=1.0)
    assert nstx_loss > iter_loss
    assert nstx_loss > 0.5


def test_slowing_down():
    tau = SlowingDown.tau_sd(Te_keV=20.0, ne_20=1.0, Z_eff=1.5)
    # Stix 1972, Eq. 7; Wesson Table 7.1: ~0.2 s for Te=20 keV, ne=1e20, Zeff=1.5
    assert 0.1 < tau < 1.0

    vc = SlowingDown.critical_velocity(20.0, 1.0)

    # 3.5 MeV alpha v ~ 1.3e7 m/s
    v_alpha = 1.3e7
    f_i, f_e = SlowingDown.heating_partition(v_alpha, vc)

    # Fast alpha heats electrons primarily
    assert f_e > f_i


def test_mc_ensemble():
    ens = MonteCarloEnsemble(10, 3500.0, 6.2, 2.0, 5.3)
    ens.initialize(np.ones(10), np.ones(10), np.linspace(0, 1, 10))

    assert len(ens.particles) == 10

    res = ens.follow(mock_b_field)
    assert res.n_passing + res.n_trapped + res.n_lost == 10


# ── New physics tests ────────────────────────────────────────────────────────


def test_stix_slowing_down():
    """dE/dt < 0 at all energies — Stix 1972, Eq. 12."""
    Te_keV = 20.0
    ne_20 = 1.0
    Z_eff = 1.5
    tau_s = SlowingDown.tau_sd(Te_keV, ne_20, Z_eff)
    E_crit = SlowingDown.critical_energy(Te_keV, A_fast=4.0, A_bg=2.5, Z_bg=1, ne_20=ne_20)

    for E_keV in [50.0, 500.0, 3520.0]:
        dE = SlowingDown.dE_dt(E_keV, E_crit, tau_s)
        assert dE < 0.0, f"dE/dt must be negative at E={E_keV} keV (Stix 1972, Eq. 12)"


def test_critical_energy_partition():
    """
    Below E_crit ions absorb >50% of fast-ion power.
    Stix 1972, Eq. 16: f_ion = v_c³ / (v³ + v_c³).
    At v < v_c → f_ion > 0.5.
    """
    Te_keV = 20.0
    ne_20 = 1.0
    v_c = SlowingDown.critical_velocity(Te_keV, ne_20)

    # Speed well below v_c → dominant ion heating
    v_slow = 0.3 * v_c
    f_ion, f_elec = SlowingDown.heating_partition(v_slow, v_c)
    assert f_ion > 0.5, f"Below v_c ion fraction must exceed 0.5 (got {f_ion:.3f})"

    # Speed well above v_c → dominant electron heating
    v_fast = 3.0 * v_c
    f_ion_hi, f_elec_hi = SlowingDown.heating_partition(v_fast, v_c)
    assert f_elec_hi > 0.5, f"Above v_c electron fraction must exceed 0.5 (got {f_elec_hi:.3f})"

    # At v = v_c: equal partition within 5%
    f_ion_eq, f_elec_eq = SlowingDown.heating_partition(v_c, v_c)
    assert abs(f_ion_eq - 0.5) < 0.05


def test_banana_width_scaling():
    """
    Δ_b = q ρ_L / √ε — Wesson 2011, Eq. 5.4.14.
    Verify linear scaling with q and inverse-sqrt scaling with ε.
    """
    rho_L = 0.05  # m, typical alpha Larmor radius

    # Linear in q
    w1 = banana_orbit_width(q=1.5, rho_L=rho_L, epsilon=0.3)
    w2 = banana_orbit_width(q=3.0, rho_L=rho_L, epsilon=0.3)
    assert abs(w2 / w1 - 2.0) < 1e-10, "Δ_b must scale linearly with q (Wesson Eq. 5.4.14)"

    # Inverse sqrt in ε
    w3 = banana_orbit_width(q=2.0, rho_L=rho_L, epsilon=0.1)
    w4 = banana_orbit_width(q=2.0, rho_L=rho_L, epsilon=0.4)
    expected_ratio = math.sqrt(0.4 / 0.1)  # √(ε2/ε1)
    assert abs(w3 / w4 - expected_ratio) < 1e-10, "Δ_b must scale as 1/√ε (Wesson Eq. 5.4.14)"


def test_first_orbit_loss_current_dependence():
    """
    Higher I_p → smaller prompt loss fraction.
    Goldston 1981, J. Comput. Phys. 43, 61, Eq. 15: f_lost ∝ 1/I_p.
    """
    base = dict(R0=6.2, a=2.0, B0=5.3, E_alpha_keV=3520.0)

    f_low_Ip = first_orbit_loss(Ip_MA=5.0, **base)
    f_high_Ip = first_orbit_loss(Ip_MA=15.0, **base)

    assert f_high_Ip < f_low_Ip, "Higher I_p must reduce prompt losses (Goldston 1981, Eq. 15)"

    # Ratio should match 1/I_p scaling within 5% (both in unsaturated regime)
    expected_ratio = 5.0 / 15.0
    actual_ratio = f_high_Ip / f_low_Ip
    assert abs(actual_ratio - expected_ratio) < 0.05, (
        f"f_lost ratio {actual_ratio:.3f} deviates from 1/I_p = {expected_ratio:.3f}"
    )

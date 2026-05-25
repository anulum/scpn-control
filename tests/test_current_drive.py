# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851  Contact: protoscience@anulum.li
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.current_drive import (
    CurrentDriveMix,
    ECCDSource,
    LHCDSource,
    NBISource,
    eccd_efficiency,
    nbi_critical_energy,
    nbi_slowing_down_time,
)


def test_zero_power():
    rho = np.linspace(0, 1, 50)
    ne = np.ones(50)
    Te = np.ones(50)
    Ti = np.ones(50)

    eccd = ECCDSource(P_ec_MW=0.0, rho_dep=0.5, sigma_rho=0.05)
    assert np.allclose(eccd.j_cd(rho, ne, Te), 0.0)

    nbi = NBISource(P_nbi_MW=0.0, E_beam_keV=100.0, rho_tangency=0.0)
    assert np.allclose(nbi.j_cd(rho, ne, Te, Ti), 0.0)


def test_eccd_peaked_deposition():
    rho = np.linspace(0, 1, 100)
    ne = np.ones(100)
    Te = np.ones(100)

    eccd = ECCDSource(P_ec_MW=10.0, rho_dep=0.4, sigma_rho=0.05)
    j_cd = eccd.j_cd(rho, ne, Te)

    # Peak should be near rho=0.4
    peak_idx = np.argmax(j_cd)
    assert 0.35 < rho[peak_idx] < 0.45


def test_eccd_edge_deposition_conserves_requested_absorbed_power():
    rho = np.linspace(0, 1, 400)
    eccd = ECCDSource(P_ec_MW=3.0, rho_dep=0.02, sigma_rho=0.05)
    assert np.trapezoid(eccd.P_absorbed(rho), rho) == pytest.approx(3.0e6, rel=1e-5)


def test_eccd_total_driven_current():
    rho = np.linspace(0, 1, 100)
    ne = np.ones(100)
    Te = np.ones(100)
    Ti = np.ones(100)

    eccd = ECCDSource(P_ec_MW=10.0, rho_dep=0.4, sigma_rho=0.05)

    mix = CurrentDriveMix(a=1.0)
    mix.add_source(eccd)

    I_cd = mix.total_driven_current(rho, ne, Te, Ti)
    assert I_cd > 0.0


def test_nbi_tangential_vs_radial():
    rho = np.linspace(0, 1, 100)
    ne = np.ones(100)
    Te = np.ones(100)
    Ti = np.ones(100)

    # In this radial source model, the tangency radius sets the deposition centre.
    nbi_tangential = NBISource(P_nbi_MW=10.0, E_beam_keV=100.0, rho_tangency=0.2)
    nbi_radial = NBISource(P_nbi_MW=10.0, E_beam_keV=100.0, rho_tangency=0.8)

    j_tang = nbi_tangential.j_cd(rho, ne, Te, Ti)
    j_rad = nbi_radial.j_cd(rho, ne, Te, Ti)

    assert np.argmax(j_tang) < np.argmax(j_rad)


def test_lhcd_off_axis():
    rho = np.linspace(0, 1, 100)
    ne = np.ones(100)
    Te = np.ones(100)

    lhcd = LHCDSource(P_lh_MW=5.0, rho_dep=0.7, sigma_rho=0.1)
    j_cd = lhcd.j_cd(rho, ne, Te)

    # Off-axis peak
    assert rho[np.argmax(j_cd)] > 0.5


def test_lhcd_edge_deposition_conserves_requested_absorbed_power():
    rho = np.linspace(0, 1, 400)
    lhcd = LHCDSource(P_lh_MW=2.0, rho_dep=0.98, sigma_rho=0.05)
    assert np.trapezoid(lhcd.P_absorbed(rho), rho) == pytest.approx(2.0e6, rel=1e-5)


def test_nbi_edge_deposition_conserves_requested_heating_power():
    rho = np.linspace(0, 1, 400)
    nbi = NBISource(P_nbi_MW=4.0, E_beam_keV=100.0, rho_tangency=0.98, sigma_rho=0.05)
    assert np.trapezoid(nbi.P_heating(rho), rho) == pytest.approx(4.0e6, rel=1e-5)


def test_current_drive_mix():
    rho = np.linspace(0, 1, 100)
    ne = np.ones(100)
    Te = np.ones(100)
    Ti = np.ones(100)

    eccd = ECCDSource(10.0, 0.4, 0.05)
    lhcd = LHCDSource(5.0, 0.7, 0.1)

    mix = CurrentDriveMix()
    mix.add_source(eccd)
    mix.add_source(lhcd)

    j_tot = mix.total_j_cd(rho, ne, Te, Ti)
    p_tot = mix.total_heating_power(rho)

    j_eccd = eccd.j_cd(rho, ne, Te)
    j_lhcd = lhcd.j_cd(rho, ne, Te)

    assert np.allclose(j_tot, j_eccd + j_lhcd)
    assert np.allclose(p_tot, eccd.P_absorbed(rho) + lhcd.P_absorbed(rho))


def test_current_drive_efficiency():
    rho = np.linspace(0, 1, 200)
    ne = np.ones(200) * 1.0
    Te = np.ones(200) * 1.0
    Ti = np.ones(200) * 1.0

    eccd = ECCDSource(P_ec_MW=10.0, rho_dep=0.5, sigma_rho=0.05, eta_cd=0.03)
    mix = CurrentDriveMix(a=1.0)
    mix.add_source(eccd)

    I_cd = mix.total_driven_current(rho, ne, Te, Ti)
    assert I_cd > 1000.0


def test_total_driven_current_uses_actual_rho_spacing():
    uniform = np.linspace(0.0, 1.0, 200)
    rho = uniform**1.7
    ne = np.ones_like(rho)
    Te = np.ones_like(rho)
    Ti = np.ones_like(rho)

    eccd = ECCDSource(P_ec_MW=8.0, rho_dep=0.55, sigma_rho=0.08, eta_cd=0.03)
    mix = CurrentDriveMix(a=1.7)
    mix.add_source(eccd)

    j_cd = mix.total_j_cd(rho, ne, Te, Ti)
    expected = np.trapezoid(j_cd * 2.0 * np.pi * rho * mix.a**2, rho)

    assert mix.total_driven_current(rho, ne, Te, Ti) == pytest.approx(expected, rel=1e-12)


def test_total_heating_power_rejects_nonfinite_or_nonmonotonic_grid():
    eccd = ECCDSource(P_ec_MW=1.0, rho_dep=0.5, sigma_rho=0.1)
    mix = CurrentDriveMix(a=1.0)
    mix.add_source(eccd)

    with pytest.raises(ValueError, match="rho grid must be finite"):
        mix.total_heating_power(np.array([0.0, np.nan, 1.0]))

    with pytest.raises(ValueError, match="rho grid must be strictly increasing"):
        mix.total_heating_power(np.array([0.0, 0.8, 0.7, 1.0]))


def test_eccd_efficiency_scaling():
    # Higher T_e → higher η_ECCD — Prater 2004, Phys. Plasmas 11, 2349, Eq. 5
    Z_eff = 1.5
    N_parallel = 0.3
    eta_low = eccd_efficiency(Te_keV=5.0, Z_eff=Z_eff, N_parallel=N_parallel)
    eta_high = eccd_efficiency(Te_keV=20.0, Z_eff=Z_eff, N_parallel=N_parallel)
    assert eta_high > eta_low
    # Scaling must be linear in T_e (all else fixed)
    assert abs(eta_high / eta_low - 20.0 / 5.0) < 1e-9


def test_nbi_slowing_down_positive():
    # τ_s > 0 for physical parameters — Stix 1972, Plasma Physics 14, 367, Eq. 6
    tau_s = nbi_slowing_down_time(Te_keV=10.0, ne_19=5.0)
    assert float(tau_s) > 0.0


def test_critical_energy():
    # E_crit ∝ T_e — Stix 1972, Plasma Physics 14, 367, Eq. 8
    E1 = nbi_critical_energy(Te_keV=5.0)
    E2 = nbi_critical_energy(Te_keV=10.0)
    assert abs(float(E2) / float(E1) - 2.0) < 1e-9


def test_lhcd_efficiency_range():
    # η_LHCD ∈ [0.1, 0.2] for ITER-like conditions — Fisch 1978, PRL 41, 873
    rho = np.array([0.7])
    ne = np.array([5.0])  # 5 × 10^19 m^-3
    Te = np.array([10.0])  # 10 keV

    for eta in [0.10, 0.15, 0.20]:
        lhcd = LHCDSource(P_lh_MW=20.0, rho_dep=0.7, sigma_rho=0.1, eta_cd=eta)
        j = lhcd.j_cd(rho, ne, Te)
        assert j[0] > 0.0

    # Confirm the default sits within the ITER range
    lhcd_default = LHCDSource(P_lh_MW=20.0, rho_dep=0.7, sigma_rho=0.1)
    assert 0.10 <= lhcd_default.eta_cd <= 0.20


def test_eccd_zero_sigma():
    """Zero-width ECCD deposition returns no absorbed-power profile."""
    eccd = ECCDSource(P_ec_MW=10.0, rho_dep=0.5, sigma_rho=0.0)
    rho = np.linspace(0, 1, 50)
    assert np.allclose(eccd.P_absorbed(rho), 0.0)


def test_nbi_zero_sigma():
    """Zero-width NBI deposition returns no heating-power profile."""
    nbi = NBISource(P_nbi_MW=10.0, E_beam_keV=100.0, rho_tangency=0.3, sigma_rho=0.0)
    rho = np.linspace(0, 1, 50)
    assert np.allclose(nbi.P_heating(rho), 0.0)


def test_lhcd_zero_sigma():
    """Zero-width LHCD deposition returns no absorbed-power profile."""
    lhcd = LHCDSource(P_lh_MW=5.0, rho_dep=0.7, sigma_rho=0.0)
    rho = np.linspace(0, 1, 50)
    assert np.allclose(lhcd.P_absorbed(rho), 0.0)


def test_mix_with_nbi():
    """CurrentDriveMix dispatches NBI sources through the NBI heating/current path."""
    rho = np.linspace(0, 1, 50)
    ne = np.ones(50) * 5.0
    Te = np.ones(50) * 10.0
    Ti = np.ones(50) * 10.0
    nbi = NBISource(P_nbi_MW=10.0, E_beam_keV=100.0, rho_tangency=0.3)
    mix = CurrentDriveMix(a=2.0)
    mix.add_source(nbi)
    j_tot = mix.total_j_cd(rho, ne, Te, Ti)
    p_tot = mix.total_heating_power(rho)
    assert np.max(j_tot) > 0.0
    assert np.max(p_tot) > 0.0


def test_current_drive_sources_reject_nonphysical_domains():
    with pytest.raises(ValueError, match="P_ec_MW must be finite and >= 0"):
        ECCDSource(P_ec_MW=-1.0, rho_dep=0.5, sigma_rho=0.05)
    with pytest.raises(ValueError, match="rho_dep must be finite and within"):
        ECCDSource(P_ec_MW=1.0, rho_dep=1.5, sigma_rho=0.05)
    with pytest.raises(ValueError, match="E_beam_keV must be finite and > 0"):
        NBISource(P_nbi_MW=1.0, E_beam_keV=0.0, rho_tangency=0.5)
    with pytest.raises(ValueError, match="eta_cd must be finite and >= 0"):
        LHCDSource(P_lh_MW=1.0, rho_dep=0.5, sigma_rho=0.05, eta_cd=-0.1)


def test_current_drive_efficiency_kernels_reject_invalid_scalars():
    with pytest.raises(ValueError, match="Te_keV must contain finite positive values"):
        eccd_efficiency(Te_keV=-1.0, Z_eff=1.5, N_parallel=0.3)
    with pytest.raises(ValueError, match="ne_19 must contain finite positive values"):
        nbi_slowing_down_time(Te_keV=10.0, ne_19=0.0)
    with pytest.raises(ValueError, match="A_ion must be finite and > 0"):
        nbi_critical_energy(Te_keV=10.0, A_ion=0.0)


def test_current_drive_rejects_nonphysical_profile_inputs():
    rho = np.linspace(0, 1, 8)
    eccd = ECCDSource(P_ec_MW=1.0, rho_dep=0.5, sigma_rho=0.1)
    ne = np.ones(8)
    te = np.ones(8)
    bad_ne = ne.copy()
    bad_ne[2] = 0.0

    with pytest.raises(ValueError, match="ne_19 must contain finite positive values"):
        eccd.j_cd(rho, bad_ne, te)

    nbi = NBISource(P_nbi_MW=1.0, E_beam_keV=100.0, rho_tangency=0.5)
    with pytest.raises(ValueError, match="Ti_keV must contain finite positive values"):
        nbi.j_cd(rho, ne, te, np.zeros(8))

    mix = CurrentDriveMix(a=1.0)
    mix.add_source(eccd)
    with pytest.raises(ValueError, match="rho grid must be strictly increasing"):
        mix.total_driven_current(rho[::-1], ne, te, te)

# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Electromagnetic GK Extension Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.gk_eigenvalue import (
    _classify_mode,
    _em_correction_factor,
    _kbm_drive,
    _mtm_drive,
    solve_eigenvalue_single_ky,
    solve_linear_gk,
)
from scpn_control.core.gk_geometry import circular_geometry
from scpn_control.core.gk_interface import GKOutput
from scpn_control.core.gk_species import VelocityGrid, deuterium_ion, electron


@pytest.fixture
def geom():
    return circular_geometry(R0=2.78, a=1.0, rho=0.5, q=1.4, s_hat=0.78, B0=2.0, n_theta=32, n_period=1)


@pytest.fixture
def species():
    return [deuterium_ion(R_L_T=6.9, R_L_n=2.2), electron(R_L_T=6.9, R_L_n=2.2)]


@pytest.fixture
def vgrid():
    return VelocityGrid(n_energy=4, n_lambda=6)


def test_beta_zero_reproduces_electrostatic(geom, species, vgrid):
    """electromagnetic=True with beta_e=0 gives the same result as electrostatic."""
    es = solve_eigenvalue_single_ky(
        k_y_rho_s=0.3,
        species_list=species,
        geom=geom,
        vgrid=vgrid,
        electromagnetic=False,
    )
    em = solve_eigenvalue_single_ky(
        k_y_rho_s=0.3,
        species_list=species,
        geom=geom,
        vgrid=vgrid,
        electromagnetic=True,
        beta_e=0.0,
        alpha_MHD=0.0,
    )
    assert es.gamma == pytest.approx(em.gamma, abs=1e-12)
    assert es.omega_r == pytest.approx(em.omega_r, abs=1e-12)
    assert em.electromagnetic is False  # beta_e=0 → em_active=False


def test_em_flag_propagates_to_eigenmode(geom, species, vgrid):
    mode = solve_eigenvalue_single_ky(
        k_y_rho_s=0.3,
        species_list=species,
        geom=geom,
        vgrid=vgrid,
        electromagnetic=True,
        beta_e=0.02,
        alpha_MHD=0.5,
    )
    assert mode.electromagnetic is True


def test_em_flag_false_by_default(geom, species, vgrid):
    mode = solve_eigenvalue_single_ky(
        k_y_rho_s=0.3,
        species_list=species,
        geom=geom,
        vgrid=vgrid,
    )
    assert mode.electromagnetic is False


def test_kbm_unstable_above_alpha_crit(geom, species, vgrid):
    """High alpha_MHD (> s_hat) + finite beta should drive KBM."""
    mode = solve_eigenvalue_single_ky(
        k_y_rho_s=0.3,
        species_list=species,
        geom=geom,
        vgrid=vgrid,
        electromagnetic=True,
        beta_e=0.05,
        alpha_MHD=2.0,
        s_hat=0.78,
    )
    assert mode.gamma > 0
    assert mode.mode_type == "KBM"


def test_kbm_stable_below_alpha_crit(geom, species, vgrid):
    """alpha_MHD < s_hat: KBM drive absent, should not classify as KBM."""
    mode = solve_eigenvalue_single_ky(
        k_y_rho_s=0.3,
        species_list=species,
        geom=geom,
        vgrid=vgrid,
        electromagnetic=True,
        beta_e=0.02,
        alpha_MHD=0.3,
        s_hat=0.78,
    )
    assert mode.mode_type != "KBM"


def test_mtm_driven_by_R_L_Te(geom, vgrid):
    """MTM driven at low k_y by electron temperature gradient + collisionality."""
    strong_ete = [deuterium_ion(R_L_T=2.0, R_L_n=2.2), electron(R_L_T=15.0, R_L_n=2.2)]
    mode = solve_eigenvalue_single_ky(
        k_y_rho_s=0.15,
        species_list=strong_ete,
        geom=geom,
        vgrid=vgrid,
        electromagnetic=True,
        beta_e=0.03,
        alpha_MHD=0.2,
        s_hat=0.78,
        nu_star=0.5,
    )
    assert mode.electromagnetic is True
    # MTM classification requires omega_r > 0 and k_y < 0.5
    if mode.omega_r > 0:
        assert mode.mode_type == "MTM"


def test_em_correction_factor_unity_at_zero_beta():
    assert _em_correction_factor(0.0, 1.0, 0.78, 0.3) == 1.0


def test_em_correction_factor_increases_with_beta():
    f1 = _em_correction_factor(0.01, 1.0, 0.78, 0.3)
    f2 = _em_correction_factor(0.05, 1.0, 0.78, 0.3)
    assert f2 > f1 > 1.0


def test_kbm_drive_zero_below_threshold():
    drive = _kbm_drive(beta_e=0.02, alpha_MHD=0.5, s_hat=0.78, k_y=0.3, n_theta=16)
    assert np.allclose(drive, 0.0)


def test_kbm_drive_nonzero_above_threshold():
    drive = _kbm_drive(beta_e=0.02, alpha_MHD=2.0, s_hat=0.78, k_y=0.3, n_theta=16)
    assert np.all(drive.real > 0)


def test_mtm_drive_purely_imaginary(geom):
    drive = _mtm_drive(beta_e=0.02, k_y=0.3, omega_star_T_e=6.9, nu_e=0.1, n_theta=len(geom.theta), geom=geom)
    assert np.allclose(drive.real, 0.0)
    assert np.any(drive.imag != 0.0)


def test_classify_mode_kbm():
    assert _classify_mode(omega_r=0.5, k_y=0.3, electromagnetic=True, alpha_MHD=2.0, s_hat=0.78) == "KBM"


def test_classify_mode_mtm():
    assert _classify_mode(omega_r=0.5, k_y=0.3, electromagnetic=True, alpha_MHD=0.5, s_hat=0.78) == "MTM"


def test_classify_mode_itg_when_not_em():
    assert _classify_mode(omega_r=-0.5, k_y=0.3, electromagnetic=False, alpha_MHD=2.0, s_hat=0.78) == "ITG"


def test_classify_mode_etg_high_ky():
    assert _classify_mode(omega_r=0.5, k_y=5.0, electromagnetic=False, alpha_MHD=0.0, s_hat=0.78) == "ETG"


def test_gkoutput_electromagnetic_field():
    out = GKOutput(chi_i=1.0, chi_e=0.5, D_e=0.3, electromagnetic=True)
    assert out.electromagnetic is True
    out2 = GKOutput(chi_i=1.0, chi_e=0.5, D_e=0.3)
    assert out2.electromagnetic is False


def test_solve_linear_gk_em_scan():
    result = solve_linear_gk(
        n_ky_ion=4,
        n_theta=16,
        n_period=1,
        electromagnetic=True,
        beta_e=0.03,
        alpha_MHD=1.5,
        s_hat=0.78,
    )
    assert len(result.modes) == 4
    assert all(m.electromagnetic for m in result.modes)
    assert np.all(np.isfinite(result.gamma))


def test_solve_linear_gk_em_vs_es_growth_rate():
    """EM correction should increase growth rate when alpha_MHD > 0."""
    es = solve_linear_gk(
        n_ky_ion=4,
        n_theta=16,
        n_period=1,
        electromagnetic=False,
    )
    em = solve_linear_gk(
        n_ky_ion=4,
        n_theta=16,
        n_period=1,
        electromagnetic=True,
        beta_e=0.05,
        alpha_MHD=2.0,
        s_hat=0.78,
    )
    assert em.gamma_max >= es.gamma_max


def test_finite_beta_kbm_in_mode_types():
    result = solve_linear_gk(
        n_ky_ion=4,
        n_theta=16,
        n_period=1,
        electromagnetic=True,
        beta_e=0.05,
        alpha_MHD=2.0,
        s_hat=0.78,
    )
    assert "KBM" in result.mode_type

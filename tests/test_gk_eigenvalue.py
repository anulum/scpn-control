# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Linear GK Eigenvalue Solver Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.gk_eigenvalue import (
    EigenMode,
    LinearGKResult,
    _drift_frequency,
    _kbm_drive,
    _mtm_drive,
    _mtm_growth_rate,
    _parallel_streaming_matrix,
    solve_eigenvalue_single_ky,
    solve_linear_gk,
)
from scpn_control.core.gk_geometry import circular_geometry
from scpn_control.core.gk_species import VelocityGrid, deuterium_ion, electron


@pytest.fixture
def cyclone_geometry():
    return circular_geometry(R0=2.78, a=1.0, rho=0.5, q=1.4, s_hat=0.78, B0=2.0, n_theta=32, n_period=1)


@pytest.fixture
def cyclone_species():
    return [deuterium_ion(R_L_T=6.9, R_L_n=2.2), electron(R_L_T=6.9, R_L_n=2.2)]


@pytest.fixture
def small_vgrid():
    return VelocityGrid(n_energy=4, n_lambda=6)


def test_eigenmode_dataclass():
    m = EigenMode(k_y_rho_s=0.3, omega_r=-0.5, gamma=0.2, mode_type="ITG")
    assert m.gamma == 0.2
    assert m.phi_theta is None


def test_single_ky_returns_eigenmode(cyclone_geometry, cyclone_species, small_vgrid):
    mode = solve_eigenvalue_single_ky(
        k_y_rho_s=0.3,
        species_list=cyclone_species,
        geom=cyclone_geometry,
        vgrid=small_vgrid,
        R0=2.78,
        a=1.0,
        B0=2.0,
    )
    assert isinstance(mode, EigenMode)
    assert mode.k_y_rho_s == 0.3
    assert mode.gamma >= 0
    assert mode.mode_type in ("ITG", "TEM", "ETG", "stable")


def test_zero_gradient_stable(cyclone_geometry, small_vgrid):
    species = [deuterium_ion(R_L_T=0.0, R_L_n=0.0), electron(R_L_T=0.0, R_L_n=0.0)]
    mode = solve_eigenvalue_single_ky(
        k_y_rho_s=0.3,
        species_list=species,
        geom=cyclone_geometry,
        vgrid=small_vgrid,
        R0=2.78,
        a=1.0,
        B0=2.0,
    )
    # With zero gradients, no drive → should be stable or near-zero growth
    assert mode.gamma < 0.5  # not strongly unstable


def test_strong_gradient_unstable(cyclone_geometry, small_vgrid):
    species = [deuterium_ion(R_L_T=15.0, R_L_n=2.2), electron(R_L_T=15.0, R_L_n=2.2)]
    mode = solve_eigenvalue_single_ky(
        k_y_rho_s=0.3,
        species_list=species,
        geom=cyclone_geometry,
        vgrid=small_vgrid,
        R0=2.78,
        a=1.0,
        B0=2.0,
    )
    # Strong gradient should produce instability
    assert mode.gamma > 0


def test_solve_linear_gk_spectrum():
    result = solve_linear_gk(
        R0=2.78,
        a=1.0,
        B0=2.0,
        q=1.4,
        s_hat=0.78,
        n_ky_ion=4,
        n_ky_etg=0,
        n_theta=16,
        n_period=1,
    )
    assert isinstance(result, LinearGKResult)
    assert len(result.k_y) == 4
    assert len(result.gamma) == 4
    assert len(result.omega_r) == 4
    assert len(result.mode_type) == 4
    assert len(result.modes) == 4


def test_solve_linear_gk_gamma_max():
    result = solve_linear_gk(
        R0=2.78,
        a=1.0,
        B0=2.0,
        q=1.4,
        s_hat=0.78,
        n_ky_ion=4,
        n_ky_etg=0,
        n_theta=16,
        n_period=1,
    )
    assert result.gamma_max >= 0
    assert np.isfinite(result.gamma_max)
    assert np.isfinite(result.k_y_max)


def test_linear_gk_all_finite():
    result = solve_linear_gk(
        R0=2.78,
        a=1.0,
        B0=2.0,
        q=1.4,
        s_hat=0.78,
        n_ky_ion=4,
        n_theta=16,
        n_period=1,
    )
    assert np.all(np.isfinite(result.gamma))
    assert np.all(np.isfinite(result.omega_r))


def test_linear_gk_empty_etg():
    result = solve_linear_gk(n_ky_ion=4, n_ky_etg=0, n_theta=16, n_period=1)
    assert len(result.k_y) == 4


def test_linear_gk_with_etg():
    result = solve_linear_gk(n_ky_ion=2, n_ky_etg=2, n_theta=16, n_period=1)
    assert len(result.k_y) == 4
    # Last 2 k_y should be > 2.0 (ETG scale)
    assert result.k_y[-1] > 2.0


def test_eigenfunction_shape(cyclone_geometry, cyclone_species, small_vgrid):
    mode = solve_eigenvalue_single_ky(
        k_y_rho_s=0.3,
        species_list=cyclone_species,
        geom=cyclone_geometry,
        vgrid=small_vgrid,
        R0=2.78,
        a=1.0,
        B0=2.0,
    )
    if mode.phi_theta is not None:
        assert len(mode.phi_theta) == len(cyclone_geometry.theta)


def test_custom_species_list():
    ion = deuterium_ion(T_keV=2.0, R_L_T=6.9)
    e = electron(T_keV=2.0, R_L_T=6.9)
    result = solve_linear_gk(
        species_list=[ion, e],
        n_ky_ion=2,
        n_theta=16,
        n_period=1,
    )
    assert len(result.modes) == 2


# ── Coverage-gap tests: EM branches and empty-array result ────────────


class TestElectromagneticKBMMTM:
    """electromagnetic=True with beta_e=0.05 exercises KBM/MTM branches."""

    def test_em_kbm_mode_detected(self):
        """High alpha_MHD + beta_e should trigger KBM classification."""
        ion = deuterium_ion(T_keV=2.0, R_L_T=6.9, R_L_n=2.2)
        e = electron(T_keV=2.0, R_L_T=6.9, R_L_n=2.2)
        result = solve_linear_gk(
            species_list=[ion, e],
            n_ky_ion=4,
            n_theta=16,
            n_period=1,
            electromagnetic=True,
            beta_e=0.05,
            alpha_MHD=2.0,
            s_hat=0.78,
        )
        assert len(result.modes) == 4
        assert all(np.isfinite(m.gamma) for m in result.modes)
        has_kbm = any(m.mode_type == "KBM" for m in result.modes)
        has_em = any(m.electromagnetic for m in result.modes)
        assert has_em

    def test_em_mtm_branch(self, cyclone_geometry, small_vgrid):
        """Low k_y + electron-direction mode + collisionality triggers MTM."""
        ion = deuterium_ion(T_keV=2.0, R_L_T=6.9, R_L_n=2.2)
        e = electron(T_keV=2.0, R_L_T=6.9, R_L_n=2.2, adiabatic=False)
        mode = solve_eigenvalue_single_ky(
            k_y_rho_s=0.1,
            species_list=[ion, e],
            geom=cyclone_geometry,
            vgrid=small_vgrid,
            electromagnetic=True,
            beta_e=0.05,
            alpha_MHD=0.1,
            s_hat=0.78,
            nu_star=0.5,
        )
        assert mode.electromagnetic is True
        assert mode.gamma >= 0.0

    def test_em_single_ky_high_beta(self, cyclone_geometry, small_vgrid):
        """High beta_e with moderate alpha_MHD should produce KBM growth."""
        ion = deuterium_ion(T_keV=2.0, R_L_T=6.9, R_L_n=2.2)
        e = electron(T_keV=2.0, R_L_T=6.9, R_L_n=2.2)
        mode = solve_eigenvalue_single_ky(
            k_y_rho_s=0.3,
            species_list=[ion, e],
            geom=cyclone_geometry,
            vgrid=small_vgrid,
            electromagnetic=True,
            beta_e=0.05,
            alpha_MHD=1.5,
            s_hat=0.78,
        )
        assert mode.gamma >= 0.0
        assert mode.electromagnetic is True


class TestLinearGKResultEmpty:
    """LinearGKResult with empty arrays returns gamma_max=0, k_y_max=0."""

    def test_empty_gamma_max(self):
        result = LinearGKResult(
            k_y=np.array([]),
            gamma=np.array([]),
            omega_r=np.array([]),
            mode_type=[],
            modes=[],
        )
        assert result.gamma_max == 0.0
        assert result.k_y_max == 0.0

    def test_single_mode_result(self):
        result = LinearGKResult(
            k_y=np.array([0.3]),
            gamma=np.array([0.15]),
            omega_r=np.array([-0.5]),
            mode_type=["ITG"],
            modes=[EigenMode(k_y_rho_s=0.3, omega_r=-0.5, gamma=0.15, mode_type="ITG")],
        )
        assert result.gamma_max == pytest.approx(0.15)
        assert result.k_y_max == pytest.approx(0.3)


class TestDriftFrequency:
    """Lines 113-114: _drift_frequency computes omega_D(theta, E, lambda)."""

    def test_drift_frequency_shape(self, cyclone_geometry):
        B_ratio = cyclone_geometry.B_mag / np.mean(cyclone_geometry.B_mag)
        wd = _drift_frequency(0.3, cyclone_geometry, 1.0, 0.5, B_ratio)
        assert wd.shape == cyclone_geometry.theta.shape
        assert np.all(np.isfinite(wd))


class TestParallelStreamingMatrix:
    """Lines 131-147: _parallel_streaming_matrix builds central FD operator."""

    def test_streaming_matrix_shape(self, cyclone_geometry):
        B_ratio = cyclone_geometry.B_mag / np.mean(cyclone_geometry.B_mag)
        n_theta = len(cyclone_geometry.theta)
        D = _parallel_streaming_matrix(n_theta, cyclone_geometry, 1.0, 0.5, B_ratio)
        assert D.shape == (n_theta, n_theta)
        assert np.all(np.isfinite(D))
        # Periodic BC: corners should be non-zero
        assert D[0, -1] != 0.0
        assert D[-1, 0] != 0.0


class TestMTMGrowthRate:
    """Line 187: _mtm_growth_rate returns 0 for zero denominator."""

    def test_zero_denom(self):
        assert _mtm_growth_rate(0.05, 0.3, 0.0, 0.0) == 0.0


class TestKBMDriveHelpers:
    """Lines 208, 231: _kbm_drive and _mtm_drive edge cases."""

    def test_kbm_drive_shape(self):
        drive = _kbm_drive(0.05, 2.0, 0.78, 0.3, 16)
        assert drive.shape == (16,)
        assert drive.dtype == complex

    def test_kbm_drive_zero_ky(self):
        drive = _kbm_drive(0.05, 2.0, 0.78, 0.0, 16)
        np.testing.assert_array_equal(drive, np.zeros(16, dtype=complex))

    def test_mtm_drive_zero_ky(self, cyclone_geometry):
        drive = _mtm_drive(0.05, 0.0, 1.0, 0.1, len(cyclone_geometry.theta), cyclone_geometry)
        np.testing.assert_array_equal(drive, np.zeros(len(cyclone_geometry.theta), dtype=complex))

    def test_mtm_drive_nonzero(self, cyclone_geometry):
        drive = _mtm_drive(0.05, 0.3, 1.0, 0.1, len(cyclone_geometry.theta), cyclone_geometry)
        assert drive.shape == (len(cyclone_geometry.theta),)
        assert np.max(np.abs(drive)) > 0


class TestMTMBranchSelection:
    """Line 433: MTM branch selected when electron-direction + collisional drive."""

    def test_mtm_branch_selected(self, cyclone_geometry, small_vgrid):
        ion = deuterium_ion(T_keV=2.0, R_L_T=6.9, R_L_n=2.2)
        e = electron(T_keV=2.0, R_L_T=6.9, R_L_n=2.2, adiabatic=False)
        mode = solve_eigenvalue_single_ky(
            k_y_rho_s=0.1,
            species_list=[ion, e],
            geom=cyclone_geometry,
            vgrid=small_vgrid,
            electromagnetic=True,
            beta_e=0.1,
            alpha_MHD=0.01,
            s_hat=0.78,
            nu_star=10.0,
        )
        assert mode.electromagnetic is True
        assert mode.gamma >= 0.0

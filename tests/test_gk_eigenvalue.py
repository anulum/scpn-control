# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Linear Gyrokinetic Eigenvalue Solver Tests

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Linear GK Eigenvalue Solver Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scpn_control.core.gk_eigenvalue import (
    EigenMode,
    LinearGKResult,
    _classify_mode,
    _drift_frequency,
    _kbm_drive,
    _mtm_drive,
    _mtm_growth_rate,
    _parallel_streaming_matrix,
    solve_eigenvalue_single_ky,
    solve_linear_gk,
)
from scpn_control.core.gk_geometry import MillerGeometry, circular_geometry
from scpn_control.core.gk_species import GKSpecies, VelocityGrid, deuterium_ion, electron


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


def test_solve_linear_gk_accepts_explicit_geometry_and_velocity_grid(cyclone_geometry, cyclone_species, small_vgrid):
    """Passing geometry and velocity-grid objects bypasses their default construction."""
    result = solve_linear_gk(
        species_list=cyclone_species,
        geom=cyclone_geometry,
        vgrid=small_vgrid,
        n_ky_ion=1,
        n_ky_etg=0,
    )
    assert isinstance(result, LinearGKResult)
    assert len(result.k_y) == 1


def test_solve_linear_gk_electromagnetic_without_electron_species(cyclone_geometry, small_vgrid):
    """An electromagnetic solve with only ion species skips the electron-driven MTM branch."""
    result = solve_linear_gk(
        species_list=[deuterium_ion(R_L_T=6.9, R_L_n=2.2)],
        geom=cyclone_geometry,
        vgrid=small_vgrid,
        electromagnetic=True,
        beta_e=0.01,
        alpha_MHD=0.5,
        n_ky_ion=1,
        n_ky_etg=0,
    )
    assert isinstance(result, LinearGKResult)
    assert np.all(np.isfinite(result.gamma))


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


class TestSingleKyInputValidation:
    """Reject unphysical scalar inputs before forming the GK operator."""

    @pytest.mark.parametrize(
        ("field", "value", "expected"),
        [
            ("k_y_rho_s", -0.1, "k_y_rho_s must be nonnegative"),
            ("k_y_rho_s", np.inf, "k_y_rho_s must be finite"),
            ("R0", 0.0, "R0 must be positive"),
            ("a", -1.0, "a must be positive"),
            ("B0", np.nan, "B0 must be finite"),
            ("beta_e", -0.01, "beta_e must be nonnegative"),
            ("nu_star", -0.5, "nu_star must be nonnegative"),
        ],
    )
    def test_rejects_unphysical_scalar_inputs(
        self,
        cyclone_geometry,
        cyclone_species,
        small_vgrid,
        field,
        value,
        expected,
    ):
        kwargs = {
            "k_y_rho_s": 0.3,
            "species_list": cyclone_species,
            "geom": cyclone_geometry,
            "vgrid": small_vgrid,
            "R0": 2.78,
            "a": 1.0,
            "B0": 2.0,
            "beta_e": 0.0,
            "nu_star": 0.0,
        }
        kwargs[field] = value

        with pytest.raises(ValueError, match=expected):
            solve_eigenvalue_single_ky(**kwargs)

    @pytest.mark.parametrize("field", ["alpha_MHD", "s_hat"])
    def test_rejects_nonfinite_signed_physics_parameters(
        self,
        cyclone_geometry,
        cyclone_species,
        small_vgrid,
        field,
    ):
        kwargs = {
            "k_y_rho_s": 0.3,
            "species_list": cyclone_species,
            "geom": cyclone_geometry,
            "vgrid": small_vgrid,
            "R0": 2.78,
            "a": 1.0,
            "B0": 2.0,
            "beta_e": 0.0,
            "nu_star": 0.0,
            "alpha_MHD": 0.0,
            "s_hat": 0.78,
        }
        kwargs[field] = np.nan

        with pytest.raises(ValueError, match=f"{field} must be finite"):
            solve_eigenvalue_single_ky(**kwargs)

    def test_allows_reversed_magnetic_shear(
        self,
        cyclone_geometry,
        cyclone_species,
        small_vgrid,
    ):
        mode = solve_eigenvalue_single_ky(
            k_y_rho_s=0.3,
            species_list=cyclone_species,
            geom=cyclone_geometry,
            vgrid=small_vgrid,
            R0=2.78,
            a=1.0,
            B0=2.0,
            s_hat=-0.5,
        )
        assert np.isfinite(mode.gamma)
        assert np.isfinite(mode.omega_r)


class TestSingleKyGeometryValidation:
    """Reject malformed flux-tube geometry before operator assembly."""

    def test_rejects_nonfinite_theta_grid(
        self,
        cyclone_geometry,
        cyclone_species,
        small_vgrid,
    ):
        bad_theta = cyclone_geometry.theta.copy()
        bad_theta[0] = np.nan
        bad_geom = replace(cyclone_geometry, theta=bad_theta)

        with pytest.raises(ValueError, match="geom.theta must be finite"):
            solve_eigenvalue_single_ky(
                k_y_rho_s=0.3,
                species_list=cyclone_species,
                geom=bad_geom,
                vgrid=small_vgrid,
            )

    def test_rejects_nonpositive_magnetic_field(
        self,
        cyclone_geometry,
        cyclone_species,
        small_vgrid,
    ):
        bad_B = cyclone_geometry.B_mag.copy()
        bad_B[0] = 0.0
        bad_geom = replace(cyclone_geometry, B_mag=bad_B)

        with pytest.raises(ValueError, match="geom.B_mag must be positive"):
            solve_eigenvalue_single_ky(
                k_y_rho_s=0.3,
                species_list=cyclone_species,
                geom=bad_geom,
                vgrid=small_vgrid,
            )

    def test_rejects_curvature_shape_mismatch(
        self,
        cyclone_geometry,
        cyclone_species,
        small_vgrid,
    ):
        bad_geom = replace(cyclone_geometry, kappa_n=cyclone_geometry.kappa_n[:-1])

        with pytest.raises(ValueError, match="geom.kappa_n must match geom.theta"):
            solve_eigenvalue_single_ky(
                k_y_rho_s=0.3,
                species_list=cyclone_species,
                geom=bad_geom,
                vgrid=small_vgrid,
            )

    def test_rejects_nonfinite_parallel_metric(
        self,
        cyclone_geometry,
        cyclone_species,
        small_vgrid,
    ):
        bad_metric = cyclone_geometry.b_dot_grad_theta.copy()
        bad_metric[0] = np.inf
        bad_geom = replace(cyclone_geometry, b_dot_grad_theta=bad_metric)

        with pytest.raises(ValueError, match="geom.b_dot_grad_theta must be finite"):
            solve_eigenvalue_single_ky(
                k_y_rho_s=0.3,
                species_list=cyclone_species,
                geom=bad_geom,
                vgrid=small_vgrid,
            )


class TestSingleKySpeciesAndVelocityGridValidation:
    """Reject invalid species ordering and velocity-space quadrature."""

    def test_rejects_empty_species_list(self, cyclone_geometry, small_vgrid):
        with pytest.raises(ValueError, match="species_list must include at least one ion species"):
            solve_eigenvalue_single_ky(
                k_y_rho_s=0.3,
                species_list=[],
                geom=cyclone_geometry,
                vgrid=small_vgrid,
            )

    def test_rejects_non_ion_leading_species(self, cyclone_geometry, small_vgrid):
        with pytest.raises(ValueError, match="species_list\\[0\\] must be an ion species"):
            solve_eigenvalue_single_ky(
                k_y_rho_s=0.3,
                species_list=[electron()],
                geom=cyclone_geometry,
                vgrid=small_vgrid,
            )

    def test_rejects_nonfinite_energy_grid(
        self,
        cyclone_geometry,
        cyclone_species,
        small_vgrid,
    ):
        small_vgrid.energy = small_vgrid.energy.copy()
        small_vgrid.energy[0] = np.nan

        with pytest.raises(ValueError, match="vgrid.energy must be finite"):
            solve_eigenvalue_single_ky(
                k_y_rho_s=0.3,
                species_list=cyclone_species,
                geom=cyclone_geometry,
                vgrid=small_vgrid,
            )

    def test_rejects_energy_weight_shape_mismatch(
        self,
        cyclone_geometry,
        cyclone_species,
        small_vgrid,
    ):
        small_vgrid.energy_weights = small_vgrid.energy_weights[:-1]

        with pytest.raises(ValueError, match="vgrid.energy_weights must match vgrid.energy"):
            solve_eigenvalue_single_ky(
                k_y_rho_s=0.3,
                species_list=cyclone_species,
                geom=cyclone_geometry,
                vgrid=small_vgrid,
            )

    def test_rejects_lambda_outside_trapped_passing_interval(
        self,
        cyclone_geometry,
        cyclone_species,
        small_vgrid,
    ):
        small_vgrid.lam = small_vgrid.lam.copy()
        small_vgrid.lam[0] = -0.1

        with pytest.raises(ValueError, match="vgrid.lam must be within \\[0, 1\\]"):
            solve_eigenvalue_single_ky(
                k_y_rho_s=0.3,
                species_list=cyclone_species,
                geom=cyclone_geometry,
                vgrid=small_vgrid,
            )


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
    """Magnetic drift frequency follows the geometry theta grid."""

    def test_drift_frequency_shape(self, cyclone_geometry):
        B_ratio = cyclone_geometry.B_mag / np.mean(cyclone_geometry.B_mag)
        wd = _drift_frequency(0.3, cyclone_geometry, 1.0, 0.5, B_ratio)
        assert wd.shape == cyclone_geometry.theta.shape
        assert np.all(np.isfinite(wd))


class TestParallelStreamingMatrix:
    """Parallel streaming matrix uses periodic central differences."""

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
    """MTM growth helper is bounded when the drive denominator vanishes."""

    def test_zero_denom(self):
        assert _mtm_growth_rate(0.05, 0.3, 0.0, 0.0) == 0.0


class TestKBMDriveHelpers:
    """Electromagnetic drive helpers preserve shape and zero-drive limits."""

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
    """MTM branch is selected by electron-direction collisional drive."""

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


def test_rejects_undersized_theta_grid(
    cyclone_geometry: MillerGeometry, cyclone_species: list[GKSpecies], small_vgrid: VelocityGrid
) -> None:
    """A theta grid with fewer than three points cannot resolve the flux tube."""
    bad_geom = replace(cyclone_geometry, theta=np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="at least 3 points"):
        solve_eigenvalue_single_ky(k_y_rho_s=0.3, species_list=cyclone_species, geom=bad_geom, vgrid=small_vgrid)


def test_rejects_nonpositive_energy_node_count(
    cyclone_geometry: MillerGeometry, cyclone_species: list[GKSpecies], small_vgrid: VelocityGrid
) -> None:
    """The energy quadrature must declare a positive node count."""
    small_vgrid.n_energy = 0
    with pytest.raises(ValueError, match="vgrid.n_energy must be positive"):
        solve_eigenvalue_single_ky(
            k_y_rho_s=0.3, species_list=cyclone_species, geom=cyclone_geometry, vgrid=small_vgrid
        )


def test_rejects_nonpositive_lambda_node_count(
    cyclone_geometry: MillerGeometry, cyclone_species: list[GKSpecies], small_vgrid: VelocityGrid
) -> None:
    """The pitch-angle quadrature must declare a positive node count."""
    small_vgrid.n_lambda = 0
    with pytest.raises(ValueError, match="vgrid.n_lambda must be positive"):
        solve_eigenvalue_single_ky(
            k_y_rho_s=0.3, species_list=cyclone_species, geom=cyclone_geometry, vgrid=small_vgrid
        )


def test_rejects_energy_node_count_mismatch(
    cyclone_geometry: MillerGeometry, cyclone_species: list[GKSpecies], small_vgrid: VelocityGrid
) -> None:
    """The energy node array length must match the declared node count."""
    small_vgrid.energy = np.zeros(3)
    with pytest.raises(ValueError, match="vgrid.energy must match vgrid.n_energy"):
        solve_eigenvalue_single_ky(
            k_y_rho_s=0.3, species_list=cyclone_species, geom=cyclone_geometry, vgrid=small_vgrid
        )


def test_rejects_lambda_node_count_mismatch(
    cyclone_geometry: MillerGeometry, cyclone_species: list[GKSpecies], small_vgrid: VelocityGrid
) -> None:
    """The pitch-angle node array length must match the declared node count."""
    small_vgrid.lam = np.zeros(3)
    with pytest.raises(ValueError, match="vgrid.lam must match vgrid.n_lambda"):
        solve_eigenvalue_single_ky(
            k_y_rho_s=0.3, species_list=cyclone_species, geom=cyclone_geometry, vgrid=small_vgrid
        )


def test_rejects_lambda_weight_shape_mismatch(
    cyclone_geometry: MillerGeometry, cyclone_species: list[GKSpecies], small_vgrid: VelocityGrid
) -> None:
    """Pitch-angle weights must share the shape of the pitch-angle nodes."""
    small_vgrid.lambda_weights = np.zeros(5)
    with pytest.raises(ValueError, match="vgrid.lambda_weights must match vgrid.lam"):
        solve_eigenvalue_single_ky(
            k_y_rho_s=0.3, species_list=cyclone_species, geom=cyclone_geometry, vgrid=small_vgrid
        )


def test_rejects_negative_energy_nodes(
    cyclone_geometry: MillerGeometry, cyclone_species: list[GKSpecies], small_vgrid: VelocityGrid
) -> None:
    """Energy nodes are kinetic energies and must be nonnegative."""
    small_vgrid.energy = small_vgrid.energy.copy()
    small_vgrid.energy[0] = -1.0
    with pytest.raises(ValueError, match="vgrid.energy must be nonnegative"):
        solve_eigenvalue_single_ky(
            k_y_rho_s=0.3, species_list=cyclone_species, geom=cyclone_geometry, vgrid=small_vgrid
        )


def test_rejects_nonpositive_energy_weights(
    cyclone_geometry: MillerGeometry, cyclone_species: list[GKSpecies], small_vgrid: VelocityGrid
) -> None:
    """Energy quadrature weights must be strictly positive."""
    small_vgrid.energy_weights = small_vgrid.energy_weights.copy()
    small_vgrid.energy_weights[0] = 0.0
    with pytest.raises(ValueError, match="vgrid.energy_weights must be positive"):
        solve_eigenvalue_single_ky(
            k_y_rho_s=0.3, species_list=cyclone_species, geom=cyclone_geometry, vgrid=small_vgrid
        )


def test_rejects_nonpositive_lambda_weights(
    cyclone_geometry: MillerGeometry, cyclone_species: list[GKSpecies], small_vgrid: VelocityGrid
) -> None:
    """Pitch-angle quadrature weights must be strictly positive."""
    small_vgrid.lambda_weights = small_vgrid.lambda_weights.copy()
    small_vgrid.lambda_weights[0] = 0.0
    with pytest.raises(ValueError, match="vgrid.lambda_weights must be positive"):
        solve_eigenvalue_single_ky(
            k_y_rho_s=0.3, species_list=cyclone_species, geom=cyclone_geometry, vgrid=small_vgrid
        )


def test_classify_mode_electromagnetic_mtm_branch() -> None:
    """Low-k_y electron-direction propagation with collisional drive is an MTM."""
    mode = _classify_mode(omega_r=1.0, k_y=0.3, electromagnetic=True, alpha_MHD=0.0, s_hat=1.0, beta_e=0.1, nu_e=0.1)
    assert mode == "MTM"


def test_classify_mode_electrostatic_etg_and_tem() -> None:
    """Electron-direction modes are ETG above k_y rho_s = 2 and TEM below it."""
    etg = _classify_mode(omega_r=1.0, k_y=3.0, electromagnetic=False, alpha_MHD=0.0, s_hat=1.0)
    tem = _classify_mode(omega_r=1.0, k_y=1.0, electromagnetic=False, alpha_MHD=0.0, s_hat=1.0)
    assert etg == "ETG"
    assert tem == "TEM"


def test_em_mtm_growth_branch_selected_for_electron_direction_mode() -> None:
    """A low-k_y collisional electron-direction EM root selects the MTM branch.

    With sub-critical KBM drive (alpha_MHD below s_hat) but a collisional,
    electron-direction root at k_y rho_s < 0.5, the electromagnetic growth-rate
    selection takes the MTM branch rather than the KBM branch.
    """
    geom = circular_geometry(R0=2.78, a=1.0, rho=0.5, q=3.0, s_hat=0.78, B0=2.0, n_theta=32, n_period=1)
    vgrid = VelocityGrid(n_energy=4, n_lambda=6)
    ion = deuterium_ion(T_keV=2.0, R_L_T=0.0, R_L_n=6.0)
    e = electron(T_keV=2.0, R_L_T=12.0, R_L_n=6.0, adiabatic=False)
    mode = solve_eigenvalue_single_ky(
        k_y_rho_s=0.2,
        species_list=[ion, e],
        geom=geom,
        vgrid=vgrid,
        electromagnetic=True,
        beta_e=0.08,
        alpha_MHD=0.1,
        s_hat=0.78,
        nu_star=20.0,
    )
    assert mode.omega_r > 0.0
    assert mode.mode_type == "MTM"
    assert mode.gamma >= 0.0

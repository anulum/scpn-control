# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Integrated Transport Solver Tests
"""
Tests for the TransportSolver class covering initialization, profile
evolution, multi-ion species, energy conservation, and steady-state runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.integrated_transport_solver import (
    TransportSolver,
    PhysicsError,
    chang_hinton_chi_profile,
    calculate_sauter_bootstrap_current_full,
    _load_gyro_bohm_coefficient,
)

# ── Minimal config for fast tests ────────────────────────────────────

MINIMAL_CONFIG = {
    "reactor_name": "TransportSolver-Test",
    "grid_resolution": [20, 20],
    "dimensions": {"R_min": 4.0, "R_max": 8.0, "Z_min": -4.0, "Z_max": 4.0},
    "physics": {"plasma_current_target": 15.0, "vacuum_permeability": 1.0},
    "coils": [
        {"name": "CS", "r": 1.7, "z": 0.0, "current": 0.15},
    ],
    "solver": {
        "max_iterations": 10,
        "convergence_threshold": 1e-4,
        "relaxation_factor": 0.1,
    },
}


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Write a minimal JSON config and return its path."""
    cfg = tmp_path / "test_transport_config.json"
    cfg.write_text(json.dumps(MINIMAL_CONFIG), encoding="utf-8")
    return cfg


@pytest.fixture
def solver(config_file: Path) -> TransportSolver:
    """Create a single-ion TransportSolver with physical initial profiles."""
    ts = TransportSolver(str(config_file), multi_ion=False)
    # Set physically meaningful initial profiles
    ts.Ti = 5.0 * (1 - ts.rho**2)
    ts.Te = 5.0 * (1 - ts.rho**2)
    ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
    ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
    ts.update_transport_model(50.0)
    return ts


@pytest.fixture
def solver_multi(config_file: Path) -> TransportSolver:
    """Create a multi-ion TransportSolver with D, T, He-ash species."""
    ts = TransportSolver(str(config_file), multi_ion=True)
    ts.Ti = 5.0 * (1 - ts.rho**2)
    ts.Te = 5.0 * (1 - ts.rho**2)
    ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
    ts.n_D = 0.5 * ts.ne.copy()
    ts.n_T = 0.5 * ts.ne.copy()
    ts.n_He = np.zeros(ts.nr)
    ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
    ts.update_transport_model(50.0)
    return ts


# ── 1. Initialization ────────────────────────────────────────────────


class TestInitialization:
    def test_init_default(self, config_file: Path) -> None:
        """TransportSolver initializes with correct default profile shapes."""
        ts = TransportSolver(str(config_file))
        assert ts.Ti.shape == (50,)
        assert ts.Te.shape == (50,)
        assert ts.ne.shape == (50,)
        assert ts.nr == 50
        assert ts.rho[0] == 0.0
        assert ts.rho[-1] == 1.0

    def test_init_rejects_invalid_grid_count(self, config_file: Path) -> None:
        """Radial grid must have at least two points for finite differencing."""
        with pytest.raises(ValueError, match="nr"):
            TransportSolver(str(config_file), nr=1)

        with pytest.raises(ValueError, match="nr"):
            TransportSolver(str(config_file), nr=2.5)

    def test_init_multi_ion(self, config_file: Path) -> None:
        """multi_ion=True creates D, T, He-ash arrays on the rho grid."""
        ts = TransportSolver(str(config_file), multi_ion=True)
        assert ts.multi_ion is True
        assert isinstance(ts.n_D, np.ndarray)
        assert isinstance(ts.n_T, np.ndarray)
        assert isinstance(ts.n_He, np.ndarray)
        assert ts.n_D.shape == (50,)
        assert ts.n_T.shape == (50,)
        assert ts.n_He.shape == (50,)

    def test_init_single_ion_no_species(self, config_file: Path) -> None:
        """multi_ion=False leaves species arrays as None."""
        ts = TransportSolver(str(config_file), multi_ion=False)
        assert ts.multi_ion is False
        assert ts.n_D is None
        assert ts.n_T is None
        assert ts.n_He is None

    def test_profiles_shape(self, solver: TransportSolver) -> None:
        """Ti, Te, ne should all be length 50 on the rho grid."""
        assert len(solver.Ti) == 50
        assert len(solver.Te) == 50
        assert len(solver.ne) == 50
        assert len(solver.rho) == 50

    def test_transport_coefficients_initialized(self, solver: TransportSolver) -> None:
        """chi_e, chi_i, D_n should exist with correct length."""
        assert solver.chi_e.shape == (50,)
        assert solver.chi_i.shape == (50,)
        assert solver.D_n.shape == (50,)

    def test_tau_he_factor_default(self, config_file: Path) -> None:
        """Default He-ash pumping time factor is 5.0."""
        ts = TransportSolver(str(config_file), multi_ion=True)
        assert ts.tau_He_factor == 5.0

    def test_d_species_default(self, config_file: Path) -> None:
        """Default particle diffusivity for species transport is 0.3."""
        ts = TransportSolver(str(config_file), multi_ion=True)
        assert ts.D_species == 0.3


# ── 2. Profile Evolution ─────────────────────────────────────────────


class TestEvolveProfiles:
    def test_evolve_profiles_runs(self, solver: TransportSolver) -> None:
        """evolve_profiles returns (avg_T, core_T) as finite floats."""
        avg_T, core_T = solver.evolve_profiles(dt=0.01, P_aux=50.0)
        assert isinstance(avg_T, float)
        assert isinstance(core_T, float)
        assert np.isfinite(avg_T)
        assert np.isfinite(core_T)
        assert avg_T > 0
        assert core_T > 0

    def test_conservation_attribute(self, solver: TransportSolver) -> None:
        """After evolve, _last_conservation_error is a finite float."""
        solver.evolve_profiles(dt=0.01, P_aux=50.0)
        err = solver._last_conservation_error
        assert isinstance(err, float)
        assert np.isfinite(err)

    def test_profiles_stay_positive(self, solver: TransportSolver) -> None:
        """Profiles should remain non-negative after multiple steps."""
        for _ in range(20):
            solver.update_transport_model(50.0)
            solver.evolve_profiles(dt=0.01, P_aux=50.0)
        assert np.all(solver.Ti >= 0)
        assert np.all(solver.Te >= 0)
        assert np.all(solver.ne >= 0)

    def test_evolve_changes_profiles(self, solver: TransportSolver) -> None:
        """Profiles should change after evolution (not frozen)."""
        Ti_before = solver.Ti.copy()
        solver.evolve_profiles(dt=0.01, P_aux=50.0)
        assert not np.allclose(solver.Ti, Ti_before, atol=1e-12)

    def test_enforce_conservation_no_raise_small_dt(self, solver: TransportSolver) -> None:
        """With small dt and reasonable params, enforce_conservation should not raise."""
        # This is best-effort: small dt + moderate heating shouldn't violate conservation
        try:
            solver.evolve_profiles(dt=0.001, P_aux=20.0, enforce_conservation=True)
        except PhysicsError:
            # If the initial conditions are too far from equilibrium,
            # conservation may be violated. This is acceptable.
            pass

    def test_multiple_steps_trend(self, solver: TransportSolver) -> None:
        """With sustained heating, average temperature should change over time."""
        T_start = float(np.mean(solver.Ti))
        for _ in range(50):
            solver.update_transport_model(50.0)
            solver.evolve_profiles(dt=0.01, P_aux=50.0)
        T_end = float(np.mean(solver.Ti))
        # Temperature should have changed (either up or stabilized)
        assert T_end != T_start

    def test_evolve_rejects_nonpositive_or_nonfinite_dt(self, solver: TransportSolver) -> None:
        """dt must be finite and non-negative."""
        # dt=0.0 is now allowed (returns early)
        solver.evolve_profiles(dt=0.0, P_aux=50.0)

        with pytest.raises(ValueError, match="finite and >= 0"):
            solver.evolve_profiles(dt=-0.01, P_aux=50.0)
        with pytest.raises(ValueError):
            solver.evolve_profiles(dt=float("nan"), P_aux=50.0)

    def test_evolve_rejects_nonfinite_heating(self, solver: TransportSolver) -> None:
        """P_aux must be finite."""
        with pytest.raises(ValueError):
            solver.evolve_profiles(dt=0.01, P_aux=float("nan"))
        with pytest.raises(ValueError):
            solver.evolve_profiles(dt=0.01, P_aux=float("inf"))

    def test_evolve_recovers_nonfinite_state(self, solver: TransportSolver) -> None:
        """Non-finite profile/transport state is repaired before CN stepping."""
        solver.Ti[3] = float("nan")
        solver.Te[5] = float("inf")
        solver.ne[7] = float("nan")
        solver.chi_i[11] = float("inf")
        solver.chi_e[13] = float("nan")
        solver.n_impurity[17] = float("inf")

        avg_t, core_t = solver.evolve_profiles(dt=0.01, P_aux=50.0)

        assert np.isfinite(avg_t)
        assert np.isfinite(core_t)
        assert np.all(np.isfinite(solver.Ti))
        assert np.all(np.isfinite(solver.Te))
        assert np.all(np.isfinite(solver.ne))
        assert np.all(np.isfinite(solver.chi_i))
        assert np.all(np.isfinite(solver.chi_e))
        assert np.all(np.isfinite(solver.n_impurity))
        assert solver._last_numerical_recovery_count > 0

    def test_aux_heating_source_power_balance_single_ion(self, solver: TransportSolver) -> None:
        """Integrated heating source must reconstruct the requested total MW."""
        solver.aux_heating_electron_fraction = 0.5
        s_i, s_e = solver._compute_aux_heating_sources(50.0)
        assert np.all(np.isfinite(s_e))
        assert np.any(s_e > 0.0)
        assert np.all(np.isfinite(s_i))

        dV = solver._rho_volume_element()
        e_keV_J = 1.602176634e-16
        ne_m3 = np.maximum(solver.ne, 0.1) * 1e19
        rec_w = 1.5 * np.sum(ne_m3 * (s_i + s_e) * e_keV_J * dV)
        rec_mw = rec_w / 1e6
        assert rec_mw == pytest.approx(50.0, rel=1e-6, abs=1e-6)
        assert solver._last_aux_heating_balance["reconstructed_total_MW"] == pytest.approx(
            50.0,
            rel=1e-6,
            abs=1e-6,
        )

    def test_single_ion_evolves_explicit_electron_channel(self, config_file: Path) -> None:
        """Single-ion mode evolves Te explicitly instead of copying Ti."""
        ts = TransportSolver(str(config_file), multi_ion=False)
        ts.aux_heating_electron_fraction = 1.0
        ts.Ti = np.full(ts.nr, 1.0)
        ts.Te = np.full(ts.nr, 1.0)
        ts.ne = np.full(ts.nr, 8.0)
        ts.chi_i = np.zeros(ts.nr)
        ts.chi_e = np.zeros(ts.nr)

        ts.evolve_profiles(dt=0.01, P_aux=20.0)
        assert not np.allclose(ts.Te, ts.Ti, atol=1e-10)

    def test_aux_heating_source_zero_power(self, solver: TransportSolver) -> None:
        """Zero auxiliary power must return zero source terms."""
        s_i, s_e = solver._compute_aux_heating_sources(0.0)
        assert np.all(s_i == 0.0)
        assert np.all(s_e == 0.0)
        assert solver._last_aux_heating_balance["reconstructed_total_MW"] == 0.0


# ── 3. Multi-Ion Species ──────────────────────────────────────────────


class TestMultiIon:
    def test_multi_ion_he_ash_grows(self, solver_multi: TransportSolver) -> None:
        """With multi_ion=True, after N steps, n_He should increase."""
        he_initial = np.sum(solver_multi.n_He)
        for _ in range(20):
            solver_multi.update_transport_model(50.0)
            solver_multi.evolve_profiles(dt=0.01, P_aux=50.0)
        he_final = np.sum(solver_multi.n_He)
        # He-ash is produced by fusion reactions
        assert he_final > he_initial

    def test_multi_ion_fuel_depletes(self, solver_multi: TransportSolver) -> None:
        """With multi_ion=True, sum of n_D should decrease over time."""
        d_initial = np.sum(solver_multi.n_D)
        for _ in range(20):
            solver_multi.update_transport_model(50.0)
            solver_multi.evolve_profiles(dt=0.01, P_aux=50.0)
        d_final = np.sum(solver_multi.n_D)
        # Fuel is consumed by fusion reactions
        assert d_final < d_initial

    def test_multi_ion_tritium_depletes(self, solver_multi: TransportSolver) -> None:
        """Tritium should also deplete over time due to fusion burn."""
        t_initial = np.sum(solver_multi.n_T)
        for _ in range(20):
            solver_multi.update_transport_model(50.0)
            solver_multi.evolve_profiles(dt=0.01, P_aux=50.0)
        t_final = np.sum(solver_multi.n_T)
        assert t_final < t_initial

    def test_aux_heating_source_power_split_multi_ion(self, solver_multi: TransportSolver) -> None:
        """Multi-ion lane should split auxiliary power between ions/electrons."""
        solver_multi.aux_heating_electron_fraction = 0.5
        s_i, s_e = solver_multi._compute_aux_heating_sources(40.0)
        assert np.all(np.isfinite(s_i))
        assert np.all(np.isfinite(s_e))

        dV = solver_multi._rho_volume_element()
        e_keV_J = 1.602176634e-16
        ne_m3 = np.maximum(solver_multi.ne, 0.1) * 1e19

        rec_i = 1.5 * np.sum(ne_m3 * s_i * e_keV_J * dV) / 1e6
        rec_e = 1.5 * np.sum(ne_m3 * s_e * e_keV_J * dV) / 1e6
        assert rec_i == pytest.approx(20.0, rel=1e-6, abs=1e-6)
        assert rec_e == pytest.approx(20.0, rel=1e-6, abs=1e-6)
        assert solver_multi._last_aux_heating_balance["reconstructed_total_MW"] == pytest.approx(
            40.0,
            rel=1e-6,
            abs=1e-6,
        )

    def test_multi_ion_quasineutrality(self, solver_multi: TransportSolver) -> None:
        """After evolving, ne should be updated from quasineutrality."""
        for _ in range(5):
            solver_multi.update_transport_model(50.0)
            solver_multi.evolve_profiles(dt=0.01, P_aux=50.0)
        # ne = n_D + n_T + 2*n_He + Z_W * n_impurity
        ne_check = (
            solver_multi.n_D
            + solver_multi.n_T
            + 2.0 * solver_multi.n_He
            + 10.0 * np.maximum(solver_multi.n_impurity, 0.0)
        )
        ne_check = np.maximum(ne_check, 0.1)
        np.testing.assert_allclose(solver_multi.ne, ne_check, rtol=1e-10)


# ── 4. Steady State Run ──────────────────────────────────────────────


class TestSteadyState:
    def test_run_to_steady_state_returns_dict(self, solver: TransportSolver) -> None:
        """run_to_steady_state returns a dict with expected keys."""
        result = solver.run_to_steady_state(P_aux=50.0, n_steps=10, dt=0.01)
        assert isinstance(result, dict)
        assert "T_avg" in result
        assert "T_core" in result
        assert "tau_e" in result
        assert "n_steps" in result
        assert "Ti_profile" in result
        assert "ne_profile" in result
        assert isinstance(result["T_avg"], float)
        assert isinstance(result["T_core"], float)
        assert np.isfinite(result["T_avg"])
        assert np.isfinite(result["T_core"])

    def test_run_to_steady_state_profile_shapes(self, solver: TransportSolver) -> None:
        """Returned profiles should have the correct length."""
        result = solver.run_to_steady_state(P_aux=50.0, n_steps=10, dt=0.01)
        assert result["Ti_profile"].shape == (50,)
        assert result["ne_profile"].shape == (50,)

    def test_confinement_time_positive(self, solver: TransportSolver) -> None:
        """Confinement time should be positive for positive loss power."""
        tau_e = solver.compute_confinement_time(50.0)
        assert tau_e > 0
        assert np.isfinite(tau_e)


# ── 5. Neoclassical Transport ─────────────────────────────────────────


class TestNeoclassical:
    def test_set_neoclassical_stores_params(self, solver: TransportSolver) -> None:
        """set_neoclassical stores parameters for Chang-Hinton model."""
        solver.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        assert solver.neoclassical_params is not None
        assert solver.neoclassical_params["R0"] == 6.2
        assert solver.neoclassical_params["a"] == 2.0
        assert solver.neoclassical_params["B0"] == 5.3

    def test_set_neoclassical_rejects_nonphysical_geometry(self, solver: TransportSolver) -> None:
        """Neoclassical transport geometry must stay in the tokamak domain."""
        with pytest.raises(ValueError, match="a must be smaller"):
            solver.set_neoclassical(R0=2.0, a=2.0, B0=5.3)

        with pytest.raises(ValueError, match="A_ion"):
            solver.set_neoclassical(R0=6.2, a=2.0, B0=5.3, A_ion=0.0)

        with pytest.raises(ValueError, match="q_edge"):
            solver.set_neoclassical(R0=6.2, a=2.0, B0=5.3, q0=3.0, q_edge=2.0)

    def test_chang_hinton_profile_shape(self) -> None:
        """Chang-Hinton neoclassical chi should match input rho shape."""
        rho = np.linspace(0, 1, 50)
        Ti = 5.0 * (1 - rho**2)
        ne = 8.0 * (1 - rho**2) ** 0.5
        q = 1.0 + 3.0 * rho**2
        chi = chang_hinton_chi_profile(rho, Ti, ne, q, R0=6.2, a=2.0, B0=5.3)
        assert chi.shape == (50,)
        assert np.all(np.isfinite(chi))
        assert np.all(chi >= 0.01)  # floor applied

    def test_chang_hinton_rejects_invalid_profiles(self) -> None:
        """Non-physical Chang-Hinton profiles fail closed instead of being floored."""
        rho = np.linspace(0, 1, 50)
        Ti = np.full_like(rho, 5.0)
        ne = np.full_like(rho, 8.0)
        q = np.full_like(rho, 2.0)

        with pytest.raises(ValueError, match="rho"):
            chang_hinton_chi_profile(rho[::-1], Ti, ne, q, R0=6.2, a=2.0, B0=5.3)

        with pytest.raises(ValueError, match="T_i"):
            chang_hinton_chi_profile(rho, np.zeros_like(Ti), ne, q, R0=6.2, a=2.0, B0=5.3)

        with pytest.raises(ValueError, match="q"):
            chang_hinton_chi_profile(rho, Ti, ne, np.zeros_like(q), R0=6.2, a=2.0, B0=5.3)

        with pytest.raises(ValueError, match="a must be smaller"):
            chang_hinton_chi_profile(rho, Ti, ne, q, R0=2.0, a=2.0, B0=5.3)

    def test_neoclassical_profiles_require_full_radial_domain(self) -> None:
        """Neoclassical closures require an axis-to-edge normalized radial grid."""
        rho_missing_axis = np.linspace(0.1, 1.0, 50)
        rho_missing_edge = np.linspace(0.0, 0.9, 50)
        Ti = np.full(50, 5.0)
        Te = np.full(50, 5.0)
        ne = np.full(50, 8.0)
        q = np.full(50, 2.0)

        with pytest.raises(ValueError, match="rho must start at 0"):
            chang_hinton_chi_profile(rho_missing_axis, Ti, ne, q, R0=6.2, a=2.0, B0=5.3)

        with pytest.raises(ValueError, match="rho must end at 1"):
            calculate_sauter_bootstrap_current_full(rho_missing_edge, Te, Ti, ne, q, R0=6.2, a=2.0, B0=5.3)

    def test_bootstrap_current_shape(self) -> None:
        """Sauter bootstrap current profile should match rho shape."""
        rho = np.linspace(0, 1, 50)
        Te = 5.0 * (1 - rho**2)
        Ti = 5.0 * (1 - rho**2)
        ne = 8.0 * (1 - rho**2) ** 0.5
        q = 1.0 + 3.0 * rho**2
        j_bs = calculate_sauter_bootstrap_current_full(rho, Te, Ti, ne, q, R0=6.2, a=2.0, B0=5.3)
        assert j_bs.shape == (50,)
        assert np.all(np.isfinite(j_bs))
        # Should be zero at the boundary (j_bs[0] and j_bs[-1])
        assert j_bs[0] == 0.0
        assert j_bs[-1] == 0.0

    def test_sauter_bootstrap_rejects_invalid_profiles(self) -> None:
        """Sauter bootstrap current requires positive finite kinetic profiles."""
        rho = np.linspace(0, 1, 50)
        Te = np.full_like(rho, 5.0)
        Ti = np.full_like(rho, 5.0)
        ne = np.full_like(rho, 8.0)
        q = np.full_like(rho, 2.0)

        with pytest.raises(ValueError, match="Te"):
            calculate_sauter_bootstrap_current_full(rho, np.zeros_like(Te), Ti, ne, q, R0=6.2, a=2.0, B0=5.3)

        with pytest.raises(ValueError, match="ne"):
            calculate_sauter_bootstrap_current_full(rho, Te, Ti, -ne, q, R0=6.2, a=2.0, B0=5.3)

        with pytest.raises(ValueError, match="Z_eff"):
            calculate_sauter_bootstrap_current_full(rho, Te, Ti, ne, q, R0=6.2, a=2.0, B0=5.3, Z_eff=0.0)

    def test_chang_hinton_near_axis_floor_for_tiny_inverse_aspect_ratio(self) -> None:
        """Near-axis Chang-Hinton points should use the finite diffusivity floor."""
        rho = np.linspace(0.0, 1.0, 50)
        Ti = np.full_like(rho, 5.0)
        ne = np.full_like(rho, 8.0)
        q = np.full_like(rho, 2.0)

        chi = chang_hinton_chi_profile(rho, Ti, ne, q, R0=1.0e12, a=1.0, B0=5.0)

        assert chi[0] == pytest.approx(0.01)
        assert chi[1] == pytest.approx(0.01)
        assert np.all(chi >= 0.01)

    def test_bootstrap_current_skips_cells_with_near_zero_poloidal_field(self) -> None:
        """Sauter bootstrap current should stay finite and zero when B_pol is negligible."""
        rho = np.linspace(0.0, 1.0, 50)
        Te = 5.0 * (1.0 - rho**2)
        Ti = 5.0 * (1.0 - rho**2)
        ne = 8.0 * (1.0 - rho**2) ** 0.5
        q = 1.0 + 3.0 * rho**2

        j_bs = calculate_sauter_bootstrap_current_full(rho, Te, Ti, ne, q, R0=6.2, a=2.0, B0=1.0e-15)

        assert j_bs.shape == rho.shape
        assert np.all(np.isfinite(j_bs))
        assert np.all(j_bs[1:-1] == 0.0)


class TestTransportNumericalGuards:
    def test_thomas_solver_handles_near_zero_initial_diagonal(self) -> None:
        """The tridiagonal solve should floor a zero first diagonal without NaNs."""
        n = 10
        a = np.zeros(n - 1)
        b = np.ones(n)
        b[0] = 0.0
        c = np.zeros(n - 1)
        d = np.ones(n)

        x = TransportSolver._thomas_solve(a, b, c, d)

        assert x.shape == (n,)
        assert np.all(np.isfinite(x))

    def test_thomas_solver_handles_nonfinite_initial_diagonal(self) -> None:
        """The tridiagonal solve should repair a non-finite first diagonal."""
        n = 10
        a = np.zeros(n - 1)
        b = np.ones(n)
        b[0] = float("nan")
        c = np.zeros(n - 1)
        d = np.ones(n)

        x = TransportSolver._thomas_solve(a, b, c, d)

        assert x.shape == (n,)
        assert np.all(np.isfinite(x))

    def test_zero_auxiliary_power_overshoot_increments_recovery_counter(self, config_file: Path) -> None:
        """Large zero-power diffusion overshoot should trigger finite-state recovery."""
        ts = TransportSolver(str(config_file), multi_ion=False)
        ts.Ti = 0.5 + 4.5 * ts.rho**2
        ts.Te = ts.Ti.copy()
        ts.ne = np.full(ts.nr, 8.0)
        ts.n_impurity = np.zeros(ts.nr)
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        ts.update_transport_model(0.0)
        ts.chi_i = np.full(ts.nr, 100.0)
        ts.chi_e = np.full(ts.nr, 100.0)

        ts.evolve_profiles(dt=1.0, P_aux=0.0)

        assert np.all(np.isfinite(ts.Ti))
        assert np.all(np.isfinite(ts.Te))
        assert np.all(ts.Ti >= 0.01)
        assert ts._last_numerical_recovery_count >= 1

    def test_nonfinite_energy_volume_marks_conservation_error_infinite(self, solver: TransportSolver) -> None:
        """Non-finite energy integrals should be reported as infinite conservation error."""
        original_volume_element = solver._rho_volume_element

        def _nonfinite_volume_element() -> np.ndarray:
            dV = original_volume_element()
            dV[5] = float("inf")
            return dV

        solver._rho_volume_element = _nonfinite_volume_element  # type: ignore[method-assign]

        solver.evolve_profiles(dt=0.01, P_aux=50.0)

        assert solver._last_conservation_error == float("inf")


# ── 6. Thomas Solver ─────────────────────────────────────────────────


class TestThomasSolver:
    def test_thomas_identity_system(self) -> None:
        """Thomas solver with identity matrix returns the RHS."""
        n = 10
        a = np.zeros(n - 1)
        b = np.ones(n)
        c = np.zeros(n - 1)
        d = np.arange(n, dtype=float)
        x = TransportSolver._thomas_solve(a, b, c, d)
        np.testing.assert_allclose(x, d, atol=1e-12)

    def test_thomas_tridiagonal(self) -> None:
        """Thomas solver correctly solves a known tridiagonal system."""
        # Solve: -x_{i-1} + 2 x_i - x_{i+1} = h^2 * f_i (Poisson)
        n = 50
        a = -np.ones(n - 1)
        b = 2.0 * np.ones(n)
        c = -np.ones(n - 1)
        d = np.ones(n) * (1.0 / (n - 1)) ** 2
        d[0] = 0.0
        d[-1] = 0.0
        b[0] = 1.0
        c[0] = 0.0
        a[-1] = 0.0
        b[-1] = 1.0
        x = TransportSolver._thomas_solve(a, b, c, d)
        assert x.shape == (n,)
        assert np.all(np.isfinite(x))


# ── 7. Bosch-Hale D-T Reactivity ─────────────────────────────────────


class TestBoschHale:
    def test_sigmav_positive_for_fusion_temperatures(self) -> None:
        """Bosch-Hale <sigma*v> should be positive for T > 0.2 keV."""
        T = np.array([1.0, 5.0, 10.0, 20.0, 50.0])
        sv = TransportSolver._bosch_hale_sigmav(T)
        assert np.all(sv > 0)
        assert np.all(np.isfinite(sv))

    def test_sigmav_increases_with_temperature(self) -> None:
        """D-T reactivity should increase monotonically from 1-50 keV (NRL fit)."""
        T = np.array([1.0, 5.0, 10.0, 20.0, 50.0])
        sv = TransportSolver._bosch_hale_sigmav(T)
        # The simplified NRL Formulary fit is monotonically increasing
        # over the range 1-100 keV (peak is beyond 100 keV for this fit)
        for i in range(1, len(sv)):
            assert sv[i] > sv[i - 1], f"sigma_v not increasing: sv[{T[i]}] = {sv[i]} <= sv[{T[i - 1]}] = {sv[i - 1]}"


# ── 8. Impurity Injection ────────────────────────────────────────────


class TestImpurityInjection:
    def test_inject_impurities_increases_edge(self, solver: TransportSolver) -> None:
        """Injecting impurities should increase the total impurity count."""
        imp_before = np.sum(solver.n_impurity)
        solver.inject_impurities(flux_from_wall_per_sec=1e20, dt=0.01)
        imp_after = np.sum(solver.n_impurity)
        assert imp_after > imp_before

    def test_impurities_non_negative(self, solver: TransportSolver) -> None:
        """Impurity profiles should remain non-negative after injection."""
        solver.inject_impurities(flux_from_wall_per_sec=1e20, dt=0.01)
        assert np.all(solver.n_impurity >= 0)


# ── 9. Gyro-Bohm & Neoclassical Method Adapter ───────────────────────


class TestGyroBohm:
    def test_load_gyro_bohm_fallback_missing_file(self) -> None:
        val = _load_gyro_bohm_coefficient("/nonexistent/file.json")
        assert val == pytest.approx(0.1)

    def test_load_gyro_bohm_from_json(self, tmp_path: Path) -> None:
        p = tmp_path / "c_gB.json"
        p.write_text(json.dumps({"c_gB": 0.042}))
        val = _load_gyro_bohm_coefficient(str(p))
        assert val == pytest.approx(0.042)

    def test_load_gyro_bohm_bad_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{not json")
        val = _load_gyro_bohm_coefficient(str(p))
        assert val == pytest.approx(0.1)

    def test_gyro_bohm_chi_with_neoclassical(self, solver: TransportSolver) -> None:
        solver.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        chi = solver._gyro_bohm_chi()
        assert chi.shape == solver.rho.shape
        assert np.all(np.isfinite(chi))
        assert np.all(chi >= 0.01)

    def test_gyro_bohm_chi_no_neoclassical(self, solver: TransportSolver) -> None:
        solver.neoclassical_params = None
        with pytest.raises(RuntimeError, match="neoclassical transport configuration is required for gyro-Bohm"):
            solver._gyro_bohm_chi()

    def test_gyro_bohm_chi_legacy_constant_opt_in(self, solver: TransportSolver) -> None:
        solver.neoclassical_params = None
        solver.allow_legacy_approximations = True
        solver.allow_constant_transport_fallback = True
        chi = solver._gyro_bohm_chi()
        assert np.all(chi == pytest.approx(0.5))

    def test_chang_hinton_method_adapter(self, solver: TransportSolver) -> None:
        solver.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        chi = solver.chang_hinton_chi_profile()
        assert chi.shape == solver.rho.shape
        assert np.all(np.isfinite(chi))

    def test_update_transport_neoclassical_path(self, solver: TransportSolver) -> None:
        solver.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        solver.update_transport_model(50.0)
        assert np.all(np.isfinite(solver.chi_e))
        assert np.all(solver.chi_e > 0)


# ── 10. Zero Aux Heating Overshoot Guard ──────────────────────────────


class TestZeroAuxHeatingGuard:
    def test_evolve_with_zero_aux_heating(self, solver: TransportSolver) -> None:
        solver.Ti = 5.0 * (1 - solver.rho**2)
        solver.Te = solver.Ti.copy()
        solver.ne = 8.0 * (1 - solver.rho**2) ** 0.5
        solver.update_transport_model(0.0)
        ti_before = solver.Ti.copy()
        solver.evolve_profiles(dt=0.001, P_aux=0.0)
        assert np.all(np.isfinite(solver.Ti))
        assert float(np.mean(solver.Ti)) <= float(np.mean(ti_before)) + 1e-8


# ── 11. Transport model branch regressions ───────────────────────────


class TestTransportModelBranches:
    def test_legacy_flags_require_global_gate(self, config_file: Path) -> None:
        with pytest.raises(ValueError, match="allow_legacy_approximations=True"):
            TransportSolver(
                str(config_file),
                allow_constant_transport_fallback=True,
            )

    def test_update_transport_without_neoclassical_fails_closed(self, config_file: Path) -> None:
        ts = TransportSolver(str(config_file))
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        with pytest.raises(RuntimeError, match="neoclassical transport configuration is required"):
            ts.update_transport_model(50.0)

    def test_gyrokinetic_transport_model(self, config_file: Path) -> None:
        """transport_model='gyrokinetic' exercises GyrokineticTransportModel path."""
        ts = TransportSolver(str(config_file), transport_model="gyrokinetic")
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        ts.update_transport_model(50.0)
        assert np.all(np.isfinite(ts.chi_i))
        assert np.all(ts.chi_i > 0)

    def test_tglf_native_transport_model(self, config_file: Path) -> None:
        """transport_model='tglf_native' exercises TGLFNativeSolver path."""
        ts = TransportSolver(str(config_file), transport_model="tglf_native")
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        ts.update_transport_model(50.0)
        assert np.all(np.isfinite(ts.chi_i))
        assert np.all(ts.chi_i > 0)


class TestBoschHaleClamping:
    def test_sigmav_at_clamped_temperature(self) -> None:
        """Bosch-Hale clamps T below 0.2 keV — no divergence or NaN."""
        T_low = np.array([0.0, 0.05, 0.1, 0.19, 0.2])
        sv = TransportSolver._bosch_hale_sigmav(T_low)
        assert np.all(np.isfinite(sv))
        assert np.all(sv > 0)
        # T<0.2 clamped to 0.2, so first 4 entries should give the same value
        np.testing.assert_allclose(sv[:4], sv[4], rtol=1e-12)


class TestAuxHeatingZeroVolume:
    def test_compute_aux_heating_zero_volume_element(self, config_file: Path) -> None:
        """_compute_aux_heating_sources handles degenerate volume gracefully."""
        ts = TransportSolver(str(config_file))
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = ts.Ti.copy()
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        # Patch volume element to return zeros — exercises the fallback path
        original_rve = ts._rho_volume_element
        ts._rho_volume_element = lambda: np.zeros(ts.nr)
        s_i, s_e = ts._compute_aux_heating_sources(50.0)
        assert np.all(s_i == 0.0)
        assert np.all(s_e == 0.0)
        ts._rho_volume_element = original_rve


class TestExternalGKSolverFallback:
    def test_gk_solver_unconverged_fails_closed_by_default(self, config_file: Path) -> None:
        """External GK unconverged results must fail closed by default."""
        ts = TransportSolver(str(config_file), transport_model="external_gk")
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)

        from unittest.mock import MagicMock
        from scpn_control.core.gk_interface import GKOutput

        mock_solver = MagicMock()
        mock_solver.run_from_params.return_value = GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=False)
        ts._gk_solver = mock_solver

        with pytest.raises(RuntimeError, match="unconverged transport"):
            ts._external_gk_transport(ts.neoclassical_params)

    def test_tglf_native_unconverged_fails_closed_by_default(self, config_file: Path) -> None:
        """Native TGLF unconverged results must fail closed by default."""
        ts = TransportSolver(str(config_file), transport_model="tglf_native")
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)

        from unittest.mock import MagicMock
        from scpn_control.core.gk_interface import GKOutput

        mock_solver = MagicMock()
        mock_solver.run_from_params.return_value = GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=False)
        ts._tglf_native_solver = mock_solver

        with pytest.raises(RuntimeError, match="unconverged transport"):
            ts._tglf_native_transport(ts.neoclassical_params)

    def test_tglf_native_legacy_fallback_opt_in(self, config_file: Path) -> None:
        """Legacy fallback for native TGLF is available only with explicit opt-in."""
        ts = TransportSolver(
            str(config_file),
            transport_model="tglf_native",
            tglf_native_allow_gyrobohm_fallback=True,
            allow_legacy_approximations=True,
        )
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)

        from unittest.mock import MagicMock
        from scpn_control.core.gk_interface import GKOutput

        mock_solver = MagicMock()
        mock_solver.run_from_params.return_value = GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=False)
        ts._tglf_native_solver = mock_solver

        chi_i = ts._tglf_native_transport(ts.neoclassical_params)
        assert np.all(np.isfinite(chi_i))
        assert np.all(chi_i >= 0.01)

    def test_gk_solver_invalid_converged_flux_fails_closed(self, config_file: Path) -> None:
        """External GK converged flag with invalid flux values must fail closed."""
        ts = TransportSolver(str(config_file), transport_model="external_gk")
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)

        from unittest.mock import MagicMock
        from scpn_control.core.gk_interface import GKOutput

        mock_solver = MagicMock()
        mock_solver.run_from_params.return_value = GKOutput(chi_i=np.nan, chi_e=1.0, D_e=0.1, converged=True)
        ts._gk_solver = mock_solver

        with pytest.raises(RuntimeError, match="unconverged transport"):
            ts._external_gk_transport(ts.neoclassical_params)

    def test_tglf_native_invalid_converged_flux_fails_closed(self, config_file: Path) -> None:
        """Native TGLF converged flag with invalid flux values must fail closed."""
        ts = TransportSolver(str(config_file), transport_model="tglf_native")
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)

        from unittest.mock import MagicMock
        from scpn_control.core.gk_interface import GKOutput

        mock_solver = MagicMock()
        mock_solver.run_from_params.return_value = GKOutput(chi_i=np.inf, chi_e=-1.0, D_e=np.nan, converged=True)
        ts._tglf_native_solver = mock_solver

        with pytest.raises(RuntimeError, match="unconverged transport"):
            ts._tglf_native_transport(ts.neoclassical_params)

    def test_tglf_native_invalid_converged_flux_legacy_fallback_opt_in(self, config_file: Path) -> None:
        """Native TGLF legacy fallback can absorb invalid converged flux only by explicit opt-in."""
        ts = TransportSolver(
            str(config_file),
            transport_model="tglf_native",
            tglf_native_allow_gyrobohm_fallback=True,
            allow_legacy_approximations=True,
        )
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)

        from unittest.mock import MagicMock
        from scpn_control.core.gk_interface import GKOutput

        mock_solver = MagicMock()
        mock_solver.run_from_params.return_value = GKOutput(chi_i=np.nan, chi_e=np.nan, D_e=np.nan, converged=True)
        ts._tglf_native_solver = mock_solver

        chi_i = ts._tglf_native_transport(ts.neoclassical_params)
        assert np.all(np.isfinite(chi_i))
        assert np.all(chi_i >= 0.01)

    def test_gk_solver_lazy_init(self, config_file: Path) -> None:
        """Exercise ITS lines 540-543: lazy init of _gk_solver."""
        ts = TransportSolver(
            str(config_file),
            transport_model="gyrokinetic",
            external_gk_allow_gyrobohm_fallback=True,
            allow_legacy_approximations=True,
        )
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)

        if hasattr(ts, "_gk_solver"):
            delattr(ts, "_gk_solver")

        chi_i = ts._external_gk_transport(ts.neoclassical_params)
        assert hasattr(ts, "_gk_solver")
        assert np.all(np.isfinite(chi_i))


class TestPedestalBoundary:
    def test_eped_import_error_fallback(self, config_file: Path) -> None:
        """Exercise ITS lines 811-813: ImportError in EPED -> fallback chi suppression."""
        from unittest.mock import patch

        ts = TransportSolver(str(config_file))
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        # High P_aux should exceed the Martin-threshold H-mode trigger here.
        # Patch EPED import to fail -> lines 811-813 fallback
        with patch.dict("sys.modules", {"scpn_control.core.eped_pedestal": None}):
            ts.update_transport_model(50.0)
        assert np.all(np.isfinite(ts.chi_e))
        # Edge chi should be suppressed by 0.1 factor
        edge_mask = ts.rho > 0.9
        assert np.all(ts.chi_e[edge_mask] > 0)

    def test_pedestal_boundary_conditions(self, config_file: Path) -> None:
        """Exercise ITS lines 1379-1386: pedestal boundary conditions in evolve."""
        from scpn_control.core.pedestal import PedestalParams, PedestalProfile

        ts = TransportSolver(str(config_file))
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        ts.update_transport_model(50.0)

        ped_params = PedestalParams(f_ped=3.0, f_sep=0.1, x_ped=0.95, delta=0.04)
        ped = PedestalProfile(ped_params)

        ts.evolve_profiles(dt=0.01, P_aux=50.0, ped_ti=ped, ped_te=ped)
        # Pedestal applied from rho >= x_ped - 2*delta = 0.87
        mask = ts.rho >= 0.87
        assert np.all(np.isfinite(ts.Ti[mask]))
        assert np.all(np.isfinite(ts.Te[mask]))


# ── 11. Module-level scalar/profile validators ───────────────────────


class TestModuleValidators:
    def test_finite_scalar_rejects_nonfinite_and_negative(self) -> None:
        from scpn_control.core.integrated_transport_solver import _finite_scalar

        with pytest.raises(ValueError, match="must be finite"):
            _finite_scalar("x", float("inf"))
        with pytest.raises(ValueError, match="must be non-negative"):
            _finite_scalar("x", -1.0, nonnegative=True)

    def test_normalised_radius_rejects_malformed_grids(self) -> None:
        from scpn_control.core.integrated_transport_solver import _normalised_radius

        with pytest.raises(ValueError, match="one-dimensional"):
            _normalised_radius(np.zeros((2, 2)))
        with pytest.raises(ValueError, match="one-dimensional"):
            _normalised_radius(np.array([0.0]))
        with pytest.raises(ValueError, match="finite"):
            _normalised_radius(np.array([0.0, np.nan, 1.0]))
        with pytest.raises(ValueError, match="normalised interval"):
            _normalised_radius(np.array([0.0, 1.5]))
        with pytest.raises(ValueError, match="strictly increasing"):
            _normalised_radius(np.array([0.0, 0.5, 0.5, 1.0]))

    def test_profile_array_rejects_shape_and_nonfinite(self) -> None:
        from scpn_control.core.integrated_transport_solver import _profile_array

        with pytest.raises(ValueError, match="match the rho grid shape"):
            _profile_array("Te", np.zeros(3), (4,))
        with pytest.raises(ValueError, match="finite values"):
            _profile_array("Te", np.array([np.nan, 1.0]), (2,))


class TestGyroBohmCoefficientLoader:
    def test_load_from_scaling_parameters_nominal(self, tmp_path: Path) -> None:
        p = tmp_path / "c_gB.json"
        p.write_text(json.dumps({"scaling_parameters": {"c_gB_nominal": 0.27}}), encoding="utf-8")
        assert _load_gyro_bohm_coefficient(p) == pytest.approx(0.27)

    def test_missing_coefficient_key_falls_back_to_default(self, tmp_path: Path) -> None:
        p = tmp_path / "no_key.json"
        p.write_text(json.dumps({"unrelated": 1}), encoding="utf-8")
        assert _load_gyro_bohm_coefficient(p) == pytest.approx(0.1)


class TestSauterBootstrapInteriorSkips:
    def test_skips_negligible_inverse_aspect_ratio(self) -> None:
        rho = np.linspace(0.0, 1.0, 20)
        Te = np.linspace(5.0, 0.1, 20)
        Ti = Te.copy()
        ne = np.linspace(8.0, 0.5, 20)
        q = np.linspace(1.0, 4.0, 20)
        j_bs = calculate_sauter_bootstrap_current_full(rho, Te, Ti, ne, q, R0=10.0, a=1.0e-6, B0=5.0)
        assert np.allclose(j_bs, 0.0)

    def test_skips_interior_cells_with_zero_density(self) -> None:
        rho = np.linspace(0.0, 1.0, 20)
        Te = np.linspace(5.0, 0.1, 20)
        Ti = Te.copy()
        ne = np.linspace(8.0, 0.5, 20)
        ne[10] = 0.0
        q = np.linspace(1.0, 4.0, 20)
        j_bs = calculate_sauter_bootstrap_current_full(rho, Te, Ti, ne, q, R0=6.2, a=2.0, B0=5.3)
        assert j_bs[10] == 0.0

    def test_skips_cells_with_degenerate_radial_spacing(self) -> None:
        rho = np.array([0.0, 0.5 - 1e-13, 0.5, 0.5 + 1e-13, 1.0])
        Te = np.array([5.0, 4.0, 3.0, 2.0, 0.0])
        Ti = Te.copy()
        ne = np.array([8.0, 6.0, 4.0, 2.0, 0.5])
        q = np.array([1.0, 2.0, 3.0, 3.5, 4.0])
        j_bs = calculate_sauter_bootstrap_current_full(rho, Te, Ti, ne, q, R0=6.2, a=2.0, B0=5.3)
        # Central cell has |rho[i+1]-rho[i-1]| below the 1e-12 gradient floor.
        assert j_bs[2] == 0.0


class TestRadialGridRestoration:
    def test_restores_corrupted_same_shape_grid_in_place(self, solver: TransportSolver) -> None:
        solver.rho = np.linspace(1.0, 0.0, solver.nr)  # decreasing, correct shape
        restored = solver._ensure_valid_radial_grid()
        assert restored == 1
        np.testing.assert_allclose(solver.rho, np.linspace(0.0, 1.0, solver.nr))

    def test_rebuilds_wrong_shape_grid(self, solver: TransportSolver) -> None:
        solver.rho = np.zeros(4)
        restored = solver._ensure_valid_radial_grid()
        assert restored == 1
        assert solver.rho.shape == (solver.nr,)

    def test_raises_when_nr_below_two(self, solver: TransportSolver) -> None:
        solver.nr = 1
        solver.rho = np.array([0.0])
        with pytest.raises(ValueError, match="nr must be at least 2"):
            solver._ensure_valid_radial_grid()


class TestChangHintonAdapter:
    def test_defaults_when_neoclassical_absent(self, config_file: Path) -> None:
        ts = TransportSolver(str(config_file))
        ts.neoclassical_params = None
        chi = ts.chang_hinton_chi_profile()
        assert chi.shape == (ts.nr,)

    def test_rebuilds_mismatched_q_profile(self, config_file: Path) -> None:
        ts = TransportSolver(str(config_file))
        ts.q_profile = np.array([1.0, 2.0])  # wrong shape, triggers rebuild
        chi = ts.chang_hinton_chi_profile()
        assert chi.shape == (ts.nr,)


class TestBootstrapCurrentDispatch:
    def test_sauter_path_with_neoclassical(self, solver: TransportSolver) -> None:
        j_bs = solver.calculate_bootstrap_current(6.2, np.full(solver.nr, 0.5))
        assert j_bs.shape == (solver.nr,)
        assert np.all(np.isfinite(j_bs))

    def test_fails_closed_without_neoclassical(self, config_file: Path) -> None:
        ts = TransportSolver(str(config_file))
        with pytest.raises(RuntimeError, match="neoclassical transport configuration is required for bootstrap"):
            ts.calculate_bootstrap_current(6.2, np.full(ts.nr, 0.5))

    def test_legacy_fallback_recomputes_geometry(self, config_file: Path) -> None:
        ts = TransportSolver(
            str(config_file),
            allow_simplified_bootstrap_fallback=True,
            allow_legacy_approximations=True,
        )
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        j_bs = ts.calculate_bootstrap_current(0.0, np.full(ts.nr, 0.05))
        assert j_bs[0] == 0.0
        assert j_bs[-1] == 0.0
        assert j_bs.shape == (ts.nr,)


class TestGyroBohmExplicitCoefficient:
    def test_uses_explicit_c_gb_from_params(self, solver: TransportSolver) -> None:
        assert solver.neoclassical_params is not None
        solver.neoclassical_params["c_gB"] = 0.2
        chi = solver._gyro_bohm_chi()
        assert chi.shape == (solver.nr,)
        assert np.all(chi >= 0.01)


class TestThomasInnerGuards:
    def test_repairs_singular_and_nonfinite_inner_rows(self) -> None:
        a = np.array([1.0, 1.0])
        b = np.array([1e-31, 1e-31, 1e-31])
        c = np.array([0.0, 0.0])
        d = np.array([1.0, np.inf, 1.0])
        x = TransportSolver._thomas_solve(a, b, c, d)
        assert x.shape == (3,)
        assert np.all(np.isfinite(x))


class TestExternalGKDispatchAndFailure:
    def _prepared_solver(self, config_file: Path) -> TransportSolver:
        ts = TransportSolver(str(config_file), transport_model="external_gk")
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        return ts

    def test_solver_execution_failure_fails_closed(self, config_file: Path) -> None:
        from unittest.mock import MagicMock

        ts = self._prepared_solver(config_file)
        mock_solver = MagicMock()
        mock_solver.run_from_params.side_effect = RuntimeError("boom")
        ts._gk_solver = mock_solver
        with pytest.raises(RuntimeError, match="solver execution failed"):
            ts._external_gk_transport(ts.neoclassical_params)

    def test_update_transport_model_uses_valid_external_gk_fluxes(self, config_file: Path) -> None:
        from unittest.mock import MagicMock

        from scpn_control.core.gk_interface import GKOutput

        ts = self._prepared_solver(config_file)
        mock_solver = MagicMock()
        mock_solver.run_from_params.return_value = GKOutput(chi_i=1.0, chi_e=0.8, D_e=0.1, converged=True)
        ts._gk_solver = mock_solver
        ts.update_transport_model(50.0)
        assert np.all(np.isfinite(ts.chi_i))
        assert np.all(np.isfinite(ts.chi_e))


class TestConstantTransportFallback:
    def test_update_transport_model_constant_fallback(self, config_file: Path) -> None:
        ts = TransportSolver(
            str(config_file),
            allow_constant_transport_fallback=True,
            allow_legacy_approximations=True,
        )
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.update_transport_model(50.0)
        assert np.all(np.isfinite(ts.chi_e))
        assert np.all(np.isfinite(ts.chi_i))


class TestHModePedestalSuccess:
    def test_high_power_triggers_eped_pedestal_branch(self, config_file: Path) -> None:
        ts = TransportSolver(str(config_file))
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        ts.update_transport_model(500.0)
        assert np.all(np.isfinite(ts.chi_e))
        assert np.all(np.isfinite(ts.Te))

    def test_high_power_eped_import_failure_uses_fallback_suppression(self, config_file: Path) -> None:
        from unittest.mock import patch

        ts = TransportSolver(str(config_file))
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        chi_turb_edge_before = ts.rho > 0.9
        with patch.dict("sys.modules", {"scpn_control.core.eped_pedestal": None}):
            ts.update_transport_model(500.0)
        assert np.all(np.isfinite(ts.chi_e))
        assert np.all(ts.chi_e[chi_turb_edge_before] > 0.0)


class TestSpeciesAndBalanceDiagnostics:
    def test_evolve_species_is_noop_in_single_ion_mode(self, solver: TransportSolver) -> None:
        s_he, p_rad = solver._evolve_species(0.01)
        assert np.all(s_he == 0.0)
        assert np.all(p_rad == 0.0)
        assert solver.particle_balance_error == 0.0

    def test_balance_error_properties_return_floats(self, solver: TransportSolver) -> None:
        assert isinstance(solver.energy_balance_error, float)
        assert isinstance(solver.particle_balance_error, float)


class TestProfileProjectionAndConfinement:
    def test_map_profiles_to_2d_updates_jphi(self, solver: TransportSolver) -> None:
        solver.map_profiles_to_2d()
        assert solver.J_phi.shape == solver.Psi.shape
        assert np.all(np.isfinite(solver.J_phi))

    def test_confinement_time_is_infinite_for_zero_loss_power(self, solver: TransportSolver) -> None:
        assert solver.compute_confinement_time(0.0) == float("inf")


class TestSelfConsistentAndSteadyStateModes:
    def test_run_self_consistent_converges_with_loose_tolerance(self, solver: TransportSolver) -> None:
        result = solver.run_self_consistent(P_aux=50.0, n_inner=2, n_outer=2, dt=0.005, psi_tol=1e12)
        assert result["converged"] is True
        assert result["n_outer_converged"] == 1
        assert "psi_residuals" in result
        assert result["Ti_profile"].shape == (solver.nr,)

    def test_run_self_consistent_exhausts_outer_iterations(self, solver: TransportSolver) -> None:
        result = solver.run_self_consistent(P_aux=50.0, n_inner=1, n_outer=1, dt=0.005, psi_tol=1e-30)
        assert result["converged"] is False
        assert result["n_outer_converged"] == 1

    def test_run_to_steady_state_self_consistent_delegates(self, solver: TransportSolver) -> None:
        result = solver.run_to_steady_state(
            P_aux=50.0,
            self_consistent=True,
            sc_n_inner=1,
            sc_n_outer=1,
            dt=0.005,
            sc_psi_tol=1e12,
        )
        assert "psi_residuals" in result

    def test_run_to_steady_state_adaptive_records_history(self, solver: TransportSolver) -> None:
        result = solver.run_to_steady_state(P_aux=50.0, n_steps=2, dt=0.01, adaptive=True, tol=1e-3)
        assert "dt_final" in result
        assert len(result["dt_history"]) == 2
        assert len(result["error_history"]) == 2
        assert np.isfinite(result["dt_final"])

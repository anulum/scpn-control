# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Transport Orchestration tests
"""Exercise the named transport orchestration surface through the integrated solver."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.integrated_transport_solver import (
    TransportSolver,
)


class TestSteadyState:
    """Check steady-state result structure, profile shapes and confinement diagnostics."""

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

    @pytest.mark.parametrize("adaptive", [False, True])
    @pytest.mark.parametrize("n_steps", [0, -1])
    def test_nonpositive_step_count_rejected_without_mutation(
        self, solver: TransportSolver, adaptive: bool, n_steps: int
    ) -> None:
        """A steady-state run requires an accepted transport step in either mode."""
        initial = solver.capture_evolution_state()
        with pytest.raises(ValueError, match="n_steps must be positive"):
            solver.run_to_steady_state(P_aux=50.0, n_steps=n_steps, adaptive=adaptive)
        np.testing.assert_array_equal(solver.Ti, initial["Ti"])
        np.testing.assert_array_equal(solver.Te, initial["Te"])


class TestMapProfilesZeroCurrent:
    """Check projection when zero density prevents current normalization."""

    def test_zero_density_skips_current_normalisation(self, config_file: Path) -> None:
        """A vanishing toroidal current leaves J_phi un-normalised, avoiding divide-by-zero (arc 1273->exit)."""
        ts = TransportSolver(str(config_file), multi_ion=False)
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 5.0 * (1 - ts.rho**2)
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        ts.ne = np.zeros_like(ts.rho)  # zero density -> zero pressure & bootstrap -> J_phi == 0
        ts.map_profiles_to_2d()
        i_curr = float(np.sum(ts.J_phi) * ts.dR * ts.dZ)
        assert abs(i_curr) <= 1e-9  # degenerate near-zero total current
        assert np.all(np.isfinite(ts.J_phi))  # normalisation skipped -> no I_target / 0 blow-up
        assert np.allclose(ts.J_phi, 0.0)


class TestProfileProjectionAndConfinement:
    """Check equilibrium projection and zero-loss confinement behavior."""

    def test_map_profiles_to_2d_updates_jphi(self, solver: TransportSolver) -> None:
        """Projection produces a finite current array matching the equilibrium grid."""
        solver.map_profiles_to_2d()
        assert solver.J_phi.shape == solver.Psi.shape
        assert np.all(np.isfinite(solver.J_phi))

    def test_map_profiles_preserves_resolved_axis_edge_separation(self, solver: TransportSolver) -> None:
        """Use the resolved X-point flux when its separation from the axis is physical."""
        solver.Psi = np.arange(solver.Psi.size, dtype=np.float64).reshape(solver.Psi.shape)

        solver.map_profiles_to_2d()

        assert solver.J_phi.shape == solver.Psi.shape
        assert np.all(np.isfinite(solver.J_phi))
        integrated_current = float(np.sum(solver.J_phi) * solver.dR * solver.dZ)
        physics = solver.cfg["physics"]
        assert isinstance(physics, dict)
        target = physics["plasma_current_target"]
        assert isinstance(target, (int, float))
        assert integrated_current == pytest.approx(target)

    def test_confinement_time_is_infinite_for_zero_loss_power(self, solver: TransportSolver) -> None:
        """Zero loss power is represented by infinite confinement time."""
        assert solver.compute_confinement_time(0.0) == float("inf")


class TestSelfConsistentAndSteadyStateModes:
    """Check termination paths and adaptive histories without asserting physical convergence."""

    def test_run_self_consistent_converges_with_loose_tolerance(self, solver: TransportSolver) -> None:
        """A deliberately loose residual tolerance exercises early success, not physical convergence."""
        result = solver.run_self_consistent(P_aux=50.0, n_inner=2, n_outer=2, dt=0.005, psi_tol=1e12)
        assert result["converged"] is True
        assert result["n_outer_converged"] == 1
        assert "psi_residuals" in result
        assert result["Ti_profile"].shape == (solver.nr,)

    def test_run_self_consistent_exhausts_outer_iterations(self, solver: TransportSolver) -> None:
        """A deliberately strict tolerance exercises the exhausted-iteration path."""
        result = solver.run_self_consistent(P_aux=50.0, n_inner=1, n_outer=1, dt=0.005, psi_tol=1e-30)
        assert result["converged"] is False
        assert result["n_outer_converged"] == 1

    @pytest.mark.parametrize("n_inner,n_outer", [(0, 1), (1, 0), (-1, 1), (1, -1)])
    def test_nonpositive_coupling_iterations_rejected_without_mutation(
        self, solver: TransportSolver, n_inner: int, n_outer: int
    ) -> None:
        """A coupled run cannot report a solve without transport and equilibrium work."""
        initial = solver.capture_evolution_state()
        with pytest.raises(ValueError, match="n_inner and n_outer must be positive"):
            solver.run_self_consistent(P_aux=50.0, n_inner=n_inner, n_outer=n_outer)
        np.testing.assert_array_equal(solver.Ti, initial["Ti"])
        np.testing.assert_array_equal(solver.Te, initial["Te"])

    def test_run_to_steady_state_self_consistent_delegates(self, solver: TransportSolver) -> None:
        """Self-consistent steady-state mode returns the coupling-loop residual history."""
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
        """Two adaptive steps return finite final dt and two entries in each history."""
        result = solver.run_to_steady_state(P_aux=50.0, n_steps=2, dt=0.01, adaptive=True, tol=1e-3)
        assert "dt_final" in result
        assert len(result["dt_history"]) == 2
        assert len(result["error_history"]) == 2
        assert np.isfinite(result["dt_final"])


def test_pressure_mapping_counts_ions_once() -> None:
    """Actual equilibrium pressure mapping uses nuclei, not charge, for the ion term."""
    config = Path(__file__).parents[1] / "validation/iter_validated_config.json"
    ts = TransportSolver(config, nr=25, multi_ion=True)
    ts.n_D = np.full(25, 2.0)
    ts.n_T = np.full(25, 1.0)
    ts.n_He = np.full(25, 2.0)
    ts.n_impurity = np.full(25, 0.01)
    ts.ne = np.full(25, 7.1)
    ts.Ti = np.full(25, 5.0)
    ts.Te = np.full(25, 2.0)
    ts.set_neoclassical(R0=6.0, a=4.0, B0=5.3)
    ts.map_profiles_to_2d()
    np.testing.assert_allclose(ts.Pressure_2D, 5.01 * 5.0 + 7.1 * 2.0, rtol=1e-13)
    density = ts.ion_density
    density[:] = 0.0
    np.testing.assert_array_equal(ts.ion_density, np.full(25, 5.01))


def test_self_consistent_residual_uses_actual_nonzero_equilibrium(solver: TransportSolver) -> None:
    """A real previous equilibrium supplies the residual denominator without the zero-start fallback."""
    solver.solve_equilibrium()
    initial = solver.Psi.copy()
    initial_norm = float(np.linalg.norm(initial))
    assert initial_norm > 1e-30
    result = solver.run_self_consistent(P_aux=50.0, n_inner=1, n_outer=1, dt=0.005, psi_tol=1e-30)
    expected = float(np.linalg.norm(solver.Psi - initial) / initial_norm)
    assert np.isfinite(expected)
    assert result["psi_residuals"] == pytest.approx([expected], rel=1e-14, abs=0)
    assert result["converged"] == (expected < 1e-30)
    assert result["n_outer_converged"] == 1
    np.testing.assert_array_equal(result["Ti_profile"], solver.Ti)
    np.testing.assert_array_equal(result["ne_profile"], solver.ne)

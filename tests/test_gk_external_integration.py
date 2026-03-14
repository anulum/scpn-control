# ──────────────────────────────────────────────────────────────────────
# SCPN Control — External GK Transport Integration Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from scpn_control.core.gk_interface import GKOutput


@pytest.fixture
def solver_config(tmp_path):
    """Minimal ITER-like JSON config for TransportSolver."""
    import json

    config = {
        "reactor_name": "GK-Test",
        "grid_resolution": [33, 33],
        "dimensions": {"R_min": 2.0, "R_max": 10.0, "Z_min": -6.0, "Z_max": 6.0},
        "physics": {"plasma_current_target": 15.0, "vacuum_permeability": 1.0},
        "coils": [
            {"name": "PF1", "r": 3.0, "z": 7.6, "current": 8.0},
            {"name": "PF2", "r": 8.2, "z": 6.7, "current": -1.3},
        ],
        "R0": 6.2,
        "a": 2.0,
        "B0": 5.3,
        "kappa": 1.7,
        "Ip_MA": 15.0,
        "n_bar_19": 10.0,
    }
    path = tmp_path / "test_config.json"
    path.write_text(json.dumps(config))
    return str(path)


@pytest.fixture
def neo_params():
    """Standard neoclassical parameter dict."""
    return {
        "R0": 6.2,
        "a": 2.0,
        "B0": 5.3,
        "q_profile": np.linspace(1.0, 3.0, 50),
        "A_ion": 2.0,
        "Z_eff": 1.5,
        "kappa": 1.7,
        "delta": 0.33,
    }


def test_external_gk_mode_with_mock_solver(solver_config, neo_params):
    """TransportSolver with external_gk mode uses the GK solver."""
    from scpn_control.core.integrated_transport_solver import TransportSolver

    ts = TransportSolver(solver_config, transport_model="external_gk")
    ts.neoclassical_params = neo_params
    ts.Te = 10.0 * (1 - ts.rho**2)
    ts.Ti = 9.0 * (1 - ts.rho**2)
    ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5

    # Mock the GK solver to return fixed fluxes
    mock_solver = MagicMock()
    mock_solver.run_from_params.return_value = GKOutput(
        chi_i=2.5,
        chi_e=1.8,
        D_e=0.4,
        converged=True,
    )
    ts._gk_solver = mock_solver

    ts.update_transport_model(P_aux=0.0)

    # chi_i should include neoclassical + GK contributions
    assert np.all(np.isfinite(ts.chi_i))
    assert np.all(ts.chi_i > 0)
    assert np.all(np.isfinite(ts.chi_e))
    assert np.all(ts.chi_e > 0)
    # GK solver should have been called for non-axis points
    assert mock_solver.run_from_params.call_count > 0


def test_external_gk_fallback_on_unconverged(solver_config, neo_params):
    """Unconverged GK results fall back to gyro-Bohm."""
    from scpn_control.core.integrated_transport_solver import TransportSolver

    ts = TransportSolver(solver_config, transport_model="external_gk")
    ts.neoclassical_params = neo_params
    ts.Te = 10.0 * (1 - ts.rho**2)
    ts.Ti = 9.0 * (1 - ts.rho**2)
    ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5

    mock_solver = MagicMock()
    mock_solver.run_from_params.return_value = GKOutput(
        chi_i=0.0,
        chi_e=0.0,
        D_e=0.0,
        converged=False,
    )
    ts._gk_solver = mock_solver

    ts.update_transport_model(P_aux=0.0)

    # Should still produce finite positive chi from fallback
    assert np.all(np.isfinite(ts.chi_i))
    assert np.all(ts.chi_i > 0)


def test_external_gk_fallback_on_exception(solver_config, neo_params):
    """Exception in GK solver falls back to gyro-Bohm."""
    from scpn_control.core.integrated_transport_solver import TransportSolver

    ts = TransportSolver(solver_config, transport_model="external_gk")
    ts.neoclassical_params = neo_params
    ts.Te = 10.0 * (1 - ts.rho**2)
    ts.Ti = 9.0 * (1 - ts.rho**2)
    ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5

    mock_solver = MagicMock()
    mock_solver.run_from_params.side_effect = RuntimeError("solver crashed")
    ts._gk_solver = mock_solver

    ts.update_transport_model(P_aux=0.0)

    assert np.all(np.isfinite(ts.chi_i))
    assert np.all(ts.chi_i > 0)


def test_external_gk_axis_boundary(solver_config, neo_params):
    """Axis points (rho <= 0.05) bypass the GK solver."""
    from scpn_control.core.integrated_transport_solver import TransportSolver

    ts = TransportSolver(solver_config, nr=10, transport_model="external_gk")
    ts.neoclassical_params = neo_params
    neo_params["q_profile"] = np.linspace(1.0, 3.0, 10)
    ts.Te = 10.0 * (1 - ts.rho**2)
    ts.Ti = 9.0 * (1 - ts.rho**2)
    ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5

    mock_solver = MagicMock()
    mock_solver.run_from_params.return_value = GKOutput(
        chi_i=5.0,
        chi_e=4.0,
        D_e=1.0,
        converged=True,
    )
    ts._gk_solver = mock_solver

    ts.update_transport_model(P_aux=0.0)

    # First point is rho=0.0 which is <= 0.05 → axis floor (0.01)
    # After adding neoclassical, chi_i[0] >= 0.01
    assert ts.chi_e[0] >= 0.01


def test_external_gk_profiles_evolve(solver_config, neo_params):
    """Full evolve_profiles step with external_gk produces valid output."""
    from scpn_control.core.integrated_transport_solver import TransportSolver

    ts = TransportSolver(solver_config, transport_model="external_gk")
    ts.neoclassical_params = neo_params
    ts.Te = 10.0 * (1 - ts.rho**2)
    ts.Ti = 9.0 * (1 - ts.rho**2)
    ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5

    mock_solver = MagicMock()
    mock_solver.run_from_params.return_value = GKOutput(
        chi_i=2.0,
        chi_e=1.5,
        D_e=0.3,
        converged=True,
    )
    ts._gk_solver = mock_solver

    Te_before = ts.Te.copy()
    for _ in range(3):
        ts.evolve_profiles(dt=0.01, P_aux=20.0)

    # Profiles should have changed
    assert not np.allclose(ts.Te, Te_before)
    assert np.all(np.isfinite(ts.Te))
    assert np.all(np.isfinite(ts.Ti))

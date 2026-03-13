# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Edge Case Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""
Edge case and boundary condition tests for scpn-control.

Tests minimal grid resolutions, zero-step evolution, empty inputs, and
other pathological but technically valid scenarios to ensure robustness.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_control.control.disruption_predictor import predict_disruption_risk
from scpn_control.control.free_boundary_tracking import FreeBoundaryTrackingController
from scpn_control.core.fusion_kernel import CoilSet, FusionKernel
from scpn_control.core.integrated_transport_solver import IntegratedTransportSolver


@pytest.fixture
def minimal_config(tmp_path):
    cfg = {
        "reactor_name": "EdgeCase-Reactor",
        "grid_resolution": [3, 3],
        "dimensions": {"R_min": 1.0, "R_max": 2.0, "Z_min": -1.0, "Z_max": 1.0},
        "physics": {"plasma_current_target": 0.1, "vacuum_permeability": 1.0},
        "coils": [{"name": "PF1", "r": 1.5, "z": 2.0, "current": 1.0}],
        "solver": {
            "boundary_variant": "fixed_boundary",
            "max_iterations": 10,
            "convergence_threshold": 1e-4,
            "omega": 1.0,
        },
        "free_boundary": {
            "target_flux_points": [[1.2, 0.0], [1.8, 0.0]],
            "target_flux_values": [0.0, 0.0],
        },
        "free_boundary_tracking": {
            "control_dt_s": 0.1,
        },
    }
    cfg_path = tmp_path / "edge_case_config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    return str(cfg_path)


def test_gs_solver_min_grid(minimal_config):
    """1. GS solver with grid_resolution: [3, 3] (minimum grid) — must not crash."""
    kernel = FusionKernel(minimal_config)
    # Fixed-boundary solve
    result = kernel.solve(boundary_variant="fixed_boundary")
    assert "success" in result or "converged" in result
    assert kernel.Psi.shape == (3, 3)


def test_transport_solver_zero_chi(minimal_config):
    """2. Transport solver with chi = 0.0 (no diffusion) — profile unchanged."""
    # Use multi_ion=False (default) to keep Te = Ti and avoid equilibration drives.
    solver = IntegratedTransportSolver(minimal_config, multi_ion=False)
    nr = solver.nr
    solver.chi_i = np.zeros(nr)
    solver.chi_e = np.zeros(nr)

    # Use initial profiles that match the solver's hardcoded boundary conditions
    # to ensure "unchanged" status is not tripped by BC enforcement.
    # Ti edge = 0.1.
    solver.Ti = np.full(nr, 0.1)
    solver.Te = np.full(nr, 0.1)

    ti_before = solver.Ti.copy()
    te_before = solver.Te.copy()

    # Evolve with zero heating and zero chi
    solver.evolve_profiles(dt=0.1, P_aux=0.0)

    np.testing.assert_allclose(solver.Ti, ti_before)
    np.testing.assert_allclose(solver.Te, te_before)


def test_transport_solver_zero_dt(minimal_config):
    """3. Transport solver with dt = 0.0 — unchanged state returned."""
    solver = IntegratedTransportSolver(minimal_config)
    ti_before = solver.Ti.copy()

    # This should now return early due to our fix in integrated_transport_solver.py
    solver.evolve_profiles(dt=0.0, P_aux=1.0)

    np.testing.assert_allclose(solver.Ti, ti_before)


def test_coilset_zero_coils():
    """4. CoilSet with zero coils — handle gracefully."""
    # Should be able to instantiate
    coils = CoilSet()
    assert len(coils.positions) == 0


def test_free_boundary_tracking_one_iteration(minimal_config):
    """5. FreeBoundaryTrackingController with 1 iteration — must not crash."""
    controller = FreeBoundaryTrackingController(minimal_config, solve_max_outer_iter=1)
    assert controller.n_coils == 1


def test_fusion_kernel_zero_iter(minimal_config):
    """6. FusionKernel with n_iter=0 — return initial state."""
    with open(minimal_config, "r") as f:
        cfg = json.load(f)
    cfg["solver"]["max_iterations"] = 0
    with open(minimal_config, "w") as f:
        json.dump(cfg, f)

    kernel_zero = FusionKernel(minimal_config)
    # At this point, kernel_zero.Psi is all zeros from initialize_grid()
    psi_init = kernel_zero.Psi.copy()
    result = kernel_zero.solve()

    assert result["iterations"] == 0
    # Our fix in fusion_kernel.py ensures it returns early before any seeding
    np.testing.assert_allclose(kernel_zero.Psi, psi_init)


def test_disruption_predictor_empty_series():
    """7. Disruption predictor with empty time series — handle without crash."""
    with pytest.raises(ValueError, match="signal must contain at least one sample"):
        predict_disruption_risk(np.array([]))

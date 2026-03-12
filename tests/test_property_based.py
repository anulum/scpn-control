# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Hypothesis Property-Based Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from scpn_control.core.fusion_kernel import FusionKernel
from scpn_control.core.integrated_transport_solver import TransportSolver
from scpn_control.phase.kuramoto import kuramoto_sakaguchi_step, order_parameter


# ── GS Solver Properties ─────────────────────────────────────────────

@settings(deadline=None, max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(omega=st.floats(1.0, 1.95))
def test_gs_sor_convergence_property(tmp_path, omega):
    """Verify that SOR solver remains stable for a range of relaxation factors."""
    config = tmp_path / "cfg.json"
    cfg = {
        "reactor_name": "Test",
        "grid_resolution": [16, 16],
        "dimensions": {"R_min": 2.0, "R_max": 6.0, "Z_min": -3.0, "Z_max": 3.0},
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        "coils": [
            {"name": "PF1", "r": 3.0, "z": 4.0, "current": 2.0},
        ],
        "solver": {
            "max_iterations": 10,
            "convergence_threshold": 1e-4,
            "relaxation_factor": omega,
            "solver_method": "sor",
        },
    }
    config.write_text(json.dumps(cfg))
    
    fk = FusionKernel(config)
    # Perform a few steps
    fk.solve_equilibrium()
    
    assert np.all(np.isfinite(fk.Psi))


# ── Transport Properties ─────────────────────────────────────────────

@settings(deadline=None, max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    chi=st.floats(0.0, 10.0),
    dt=st.floats(0.001, 0.1)
)
def test_transport_non_negative_property(tmp_path, chi, dt):
    """Verify that temperature stays non-negative for any valid chi and dt."""
    config = tmp_path / "cfg.json"
    cfg = {
        "reactor_name": "Test",
        "grid_resolution": [33, 33],
        "dimensions": {"R_min": 4.2, "R_max": 8.2, "Z_min": -4.0, "Z_max": 4.0},
        "physics": {"plasma_current_target": 15.0e6},
        "coils": [],
    }
    config.write_text(json.dumps(cfg))
    
    solver = TransportSolver(config)
    solver.chi_i.fill(chi)
    solver.chi_e.fill(chi)
    
    # Reset to a physical state
    solver.Ti.fill(5.0)
    solver.Te.fill(5.0)
    
    solver.evolve_profiles(dt=dt, P_aux=50.0)
    
    assert np.all(solver.Ti >= 0.0)
    assert np.all(solver.Te >= 0.0)
    assert np.all(np.isfinite(solver.Ti))


# ── Kuramoto Properties ──────────────────────────────────────────────

@settings(max_examples=50)
@given(
    theta=st.lists(st.floats(0, 2*np.pi), min_size=2, max_size=10),
    K=st.floats(0.0, 10.0),
    zeta=st.floats(0.0, 1.0)
)
def test_kuramoto_order_parameter_bounds(theta, K, zeta):
    """Verify that order parameter R is always in [0, 1]."""
    th = np.array(theta)
    R, _ = order_parameter(th)
    assert 0.0 <= R <= 1.0 + 1e-9
    
    # Step properties
    omega = np.ones_like(th)
    result = kuramoto_sakaguchi_step(th, omega, dt=0.01, K=K, zeta=zeta, psi_driver=0.0)
    
    assert 0.0 <= result["R"] <= 1.0 + 1e-9
    assert np.all(np.isfinite(result["theta1"]))

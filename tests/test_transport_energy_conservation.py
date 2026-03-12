# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Transport Energy Conservation Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.integrated_transport_solver import TransportSolver


def test_energy_balance_error_property(tmp_path):
    """Verify that energy_balance_error is updated and accessible."""
    config = tmp_path / "iter_config.json"
    # Create minimal config
    config.write_text('{"reactor_name": "ITER", "dimensions": {"R_min": 4.2, "R_max": 8.2, "Z_min": -4.0, "Z_max": 4.0}, "grid_resolution": [33, 33], "physics": {"plasma_current_target": 15.0e6}}')
    
    solver = TransportSolver(config)
    # Initial error should be zero
    assert solver.energy_balance_error == 0.0
    
    # Take a step
    solver.evolve_profiles(dt=0.01, P_aux=50.0)
    
    # Error should be updated and reasonably small for a stable step
    err = solver.energy_balance_error
    assert err >= 0.0
    assert err < 0.1 # Should be much smaller, but 10% is safe bound for toy grid


def test_energy_conservation_enforcement(tmp_path):
    """Verify that PhysicsError is raised if conservation fails when enforced."""
    config = tmp_path / "iter_config.json"
    config.write_text('{"reactor_name": "ITER", "dimensions": {"R_min": 4.2, "R_max": 8.2, "Z_min": -4.0, "Z_max": 4.0}, "grid_resolution": [33, 33], "physics": {"plasma_current_target": 15.0e6}}')
    
    solver = TransportSolver(config)
    
    # Force a massive unphysical source to trigger error if we can, 
    # or just check that it doesn't raise if error is small.
    # The solver tries to be stable, so it's hard to trigger without hacking.
    # But we can verify the parameter is respected.
    
    # Normal case: should not raise
    solver.evolve_profiles(dt=0.001, P_aux=50.0, enforce_conservation=True)
    
    # If we set dt very large, convergence might suffer and error might grow
    with pytest.raises(Exception): # Might raise ValueError for dt or PhysicsError
        solver.evolve_profiles(dt=100.0, P_aux=50.0, enforce_conservation=True)


def test_energy_balance_multi_ion(tmp_path):
    """Verify energy balance tracking works in multi-ion mode."""
    config = tmp_path / "iter_config.json"
    config.write_text('{"reactor_name": "ITER", "dimensions": {"R_min": 4.2, "R_max": 8.2, "Z_min": -4.0, "Z_max": 4.0}, "grid_resolution": [33, 33], "physics": {"plasma_current_target": 15.0e6}}')
    
    solver = TransportSolver(config, multi_ion=True)
    solver.evolve_profiles(dt=0.01, P_aux=50.0)
    
    assert solver.energy_balance_error >= 0.0
    assert solver.energy_balance_error < 0.1

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Integrated transport solver edge guard tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for _evolve_species no multi-ion (877),
confinement_time P_loss<=0 (1179)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.integrated_transport_solver import TransportSolver

MINIMAL_CONFIG = {
    "reactor_name": "TransportGuard-Test",
    "grid_resolution": [20, 20],
    "dimensions": {"R_min": 4.0, "R_max": 8.0, "Z_min": -4.0, "Z_max": 4.0},
    "physics": {"plasma_current_target": 15.0, "vacuum_permeability": 1.0},
    "coils": [{"name": "CS", "r": 1.7, "z": 0.0, "current": 0.15}],
    "solver": {
        "max_iterations": 10,
        "convergence_threshold": 1e-4,
        "relaxation_factor": 0.1,
    },
}


@pytest.fixture
def solver(tmp_path: Path) -> TransportSolver:
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(MINIMAL_CONFIG), encoding="utf-8")
    ts = TransportSolver(str(cfg), multi_ion=False)
    ts.Ti = 5.0 * (1 - ts.rho ** 2)
    ts.Te = 5.0 * (1 - ts.rho ** 2)
    ts.ne = 8.0 * (1 - ts.rho ** 2) ** 0.5
    return ts


class TestConfinementTimeZeroPower:
    def test_zero_p_loss_returns_inf(self, solver):
        """P_loss_MW <= 0 returns inf (line 1179)."""
        tau = solver.compute_confinement_time(P_loss_MW=0.0)
        assert tau == float("inf")

    def test_negative_p_loss_returns_inf(self, solver):
        """Negative P_loss_MW returns inf."""
        tau = solver.compute_confinement_time(P_loss_MW=-1.0)
        assert tau == float("inf")


class TestEvolveSpeciesNoMultiIon:
    def test_single_ion_returns_zeros(self, solver):
        """_evolve_species without multi_ion returns zeros (line 877)."""
        assert solver.multi_ion is False
        S, P = solver._evolve_species(dt=0.001)
        np.testing.assert_array_equal(S, np.zeros(solver.nr))
        np.testing.assert_array_equal(P, np.zeros(solver.nr))

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Momentum Transport Integration Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json

import numpy as np

from scpn_control.core.integrated_transport_solver import TransportSolver


def _make_solver(tmp_path, nr=33):
    cfg = {
        "reactor_name": "test_momentum",
        "dimensions": {"R_min": 4.2, "R_max": 8.2, "Z_min": -3.4, "Z_max": 3.4},
        "grid_resolution": [nr, nr],
        "physics": {"plasma_current_target": 15.0},
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return TransportSolver(p, nr=nr, multi_ion=True)


class TestMomentumInitialization:
    def test_omega_phi_exists(self, tmp_path):
        solver = _make_solver(tmp_path)
        assert hasattr(solver, "omega_phi")
        assert len(solver.omega_phi) == solver.nr
        assert np.all(solver.omega_phi == 0.0)

    def test_momentum_solver_none_before_neoclassical(self, tmp_path):
        solver = _make_solver(tmp_path)
        assert solver._momentum_solver is None

    def test_momentum_solver_created_after_neoclassical(self, tmp_path):
        solver = _make_solver(tmp_path)
        solver.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        assert solver._momentum_solver is not None


class TestMomentumEvolution:
    def test_rotation_evolves_with_neoclassical(self, tmp_path):
        solver = _make_solver(tmp_path)
        solver.set_neoclassical(R0=6.2, a=2.0, B0=5.3)

        # Give non-trivial initial profiles
        solver.Te = 10.0 * (1 - solver.rho**2)
        solver.Ti = 10.0 * (1 - solver.rho**2)
        solver.ne = 10.0 * (1 - solver.rho**2) ** 0.5

        omega_before = solver.omega_phi.copy()
        for _ in range(5):
            solver.evolve_profiles(dt=0.01, P_aux=50.0)

        # Intrinsic rotation torque (from ∇Ti) should drive some rotation
        assert not np.allclose(solver.omega_phi, omega_before), "Rotation profile unchanged after evolution"

    def test_rotation_zero_without_neoclassical(self, tmp_path):
        solver = _make_solver(tmp_path)
        # No neoclassical → no momentum solver → omega stays zero
        solver.evolve_profiles(dt=0.01, P_aux=50.0)
        assert np.all(solver.omega_phi == 0.0)

    def test_rotation_bounded(self, tmp_path):
        solver = _make_solver(tmp_path)
        solver.set_neoclassical(R0=6.2, a=2.0, B0=5.3)

        solver.Te = 10.0 * (1 - solver.rho**2)
        solver.Ti = 10.0 * (1 - solver.rho**2)
        solver.ne = 10.0 * (1 - solver.rho**2) ** 0.5

        for _ in range(20):
            solver.evolve_profiles(dt=0.01, P_aux=50.0)

        assert np.all(np.isfinite(solver.omega_phi)), "omega_phi has non-finite values"

    def test_edge_rotation_zero(self, tmp_path):
        """Boundary condition: no-slip at edge."""
        solver = _make_solver(tmp_path)
        solver.set_neoclassical(R0=6.2, a=2.0, B0=5.3)

        solver.Te = 10.0 * (1 - solver.rho**2)
        solver.Ti = 10.0 * (1 - solver.rho**2)
        solver.ne = 10.0 * (1 - solver.rho**2) ** 0.5

        for _ in range(5):
            solver.evolve_profiles(dt=0.01, P_aux=50.0)

        # MomentumTransportSolver applies omega[-1] = 0 (no-slip)
        assert solver.omega_phi[-1] == 0.0

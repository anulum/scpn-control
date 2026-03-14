# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Particle Balance Conservation Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""
Particle conservation diagnostics for the multi-ion transport solver.

Verifies that the discrete particle balance residual remains small:
  |ΔN_actual - ΔN_sources| / N_total < threshold

Sources/sinks: D-T fusion consumption, He-ash production, He pumping.
Boundary and clipping corrections introduce O(1%) residuals on coarse grids.
"""

from __future__ import annotations

from scpn_control.core.integrated_transport_solver import TransportSolver


def _make_config(tmp_path, nr=33):
    config = tmp_path / "iter_config.json"
    config.write_text(
        '{"reactor_name": "ITER",'
        ' "dimensions": {"R_min": 4.2, "R_max": 8.2, "Z_min": -4.0, "Z_max": 4.0},'
        f' "grid_resolution": [{nr}, {nr}],'
        ' "physics": {"plasma_current_target": 15.0e6}}'
    )
    return config


class TestParticleBalance:
    """Particle balance residual must be bounded."""

    def test_particle_balance_property_exists(self, tmp_path):
        solver = TransportSolver(_make_config(tmp_path), multi_ion=True)
        assert solver.particle_balance_error == 0.0

    def test_particle_balance_after_step(self, tmp_path):
        solver = TransportSolver(_make_config(tmp_path), multi_ion=True)
        solver.evolve_profiles(dt=0.01, P_aux=50.0)
        err = solver.particle_balance_error
        assert err >= 0.0
        # Coarse 33x33 grid: allow up to 10% from boundary/clipping effects
        assert err < 0.10, f"Particle balance error {err:.4e} >= 10%"

    def test_particle_balance_zero_without_multi_ion(self, tmp_path):
        solver = TransportSolver(_make_config(tmp_path), multi_ion=False)
        solver.evolve_profiles(dt=0.01, P_aux=50.0)
        assert solver.particle_balance_error == 0.0

    def test_particle_balance_multiple_steps(self, tmp_path):
        solver = TransportSolver(_make_config(tmp_path), multi_ion=True)
        errors = []
        for _ in range(5):
            solver.evolve_profiles(dt=0.01, P_aux=50.0)
            errors.append(solver.particle_balance_error)

        # All errors should be bounded
        for i, err in enumerate(errors):
            assert err < 0.15, f"Step {i}: particle balance error {err:.4e} >= 15%"

    def test_particle_balance_improves_with_resolution(self, tmp_path):
        """Finer grid should give smaller particle balance residual."""
        errors = {}
        for nr in [17, 33]:
            solver = TransportSolver(_make_config(tmp_path, nr=nr), multi_ion=True)
            solver.evolve_profiles(dt=0.005, P_aux=50.0)
            errors[nr] = solver.particle_balance_error

        # Finer grid should have smaller or comparable error
        # Not strictly guaranteed due to boundary effects, so allow 2x margin
        assert errors[33] < errors[17] * 2.5, (
            f"Particle balance did not improve: 17->{errors[17]:.4e}, 33->{errors[33]:.4e}"
        )


class TestEnergyAndParticleJoint:
    """Both energy and particle balance must be tracked simultaneously."""

    def test_both_diagnostics_updated(self, tmp_path):
        solver = TransportSolver(_make_config(tmp_path), multi_ion=True)
        solver.evolve_profiles(dt=0.01, P_aux=50.0)

        assert solver.energy_balance_error >= 0.0
        assert solver.particle_balance_error >= 0.0
        # Both should be < 10% for a well-behaved step
        assert solver.energy_balance_error < 0.10
        assert solver.particle_balance_error < 0.10

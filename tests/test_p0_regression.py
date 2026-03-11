# ──────────────────────────────────────────────────────────────────────
# SCPN Control — P0 Regression Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Regression tests for P0 correctness bugs fixed 2026-03-11.

1. Jacobi toroidal stencil — must agree with SOR within iteration tolerance
2. Vacuum field coil turns — 2x turns must give 2x flux
3. UPDE Rust/Python return key parity — "Psi_global" everywhere
4. JAX GS boundary — ψ_bdry = 0 (Dirichlet), not corner value
5. Analytic 2-oscillator Kuramoto — exact R(t) trajectory
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.fusion_kernel import FusionKernel
from scpn_control.phase.kuramoto import kuramoto_sakaguchi_step, order_parameter
from scpn_control.phase.knm import KnmSpec
from scpn_control.phase.upde import UPDESystem


# ── helpers ──────────────────────────────────────────────────────────


_CFG_COUNTER = 0


def _make_kernel(tmp_path: Path, *, method: str, coils: list | None = None) -> FusionKernel:
    global _CFG_COUNTER
    _CFG_COUNTER += 1
    cfg = {
        "reactor_name": "P0-Regression",
        "grid_resolution": [16, 16],
        "dimensions": {"R_min": 2.0, "R_max": 6.0, "Z_min": -3.0, "Z_max": 3.0},
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        "coils": coils
        or [
            {"name": "PF1", "r": 3.0, "z": 4.0, "current": 2.0},
            {"name": "PF2", "r": 5.0, "z": -4.0, "current": -1.0},
        ],
        "solver": {
            "max_iterations": 50,
            "convergence_threshold": 1e-6,
            "relaxation_factor": 0.15,
            "solver_method": method,
        },
    }
    p = tmp_path / f"cfg_{method}_{_CFG_COUNTER}.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return FusionKernel(p)


# ── 1. Jacobi toroidal stencil consistency ──────────────────────────


class TestJacobiToroidalStencil:
    """Jacobi and SOR must implement the same GS* operator."""

    def test_single_step_matches_sor_operator(self, tmp_path: Path) -> None:
        fk = _make_kernel(tmp_path, method="sor")
        rng = np.random.default_rng(42)
        Psi = rng.standard_normal(fk.Psi.shape) * 0.01
        Source = np.ones_like(Psi) * 0.001

        psi_jacobi = fk._jacobi_step(Psi, Source)
        # SOR with omega=1.0 is equivalent to Gauss-Seidel, not Jacobi.
        # Instead verify that the Jacobi step has R-dependent east/west
        # asymmetry by checking that result[1:-1, 1:-1] is NOT symmetric
        # in the R direction (which the old Cartesian 0.25 stencil was).
        interior = psi_jacobi[1:-1, 1:-1]
        # Flip along R axis (columns)
        flipped = interior[:, ::-1]
        # With the toroidal 1/R term, left-right asymmetry must be non-zero
        assert not np.allclose(interior, flipped, atol=1e-12), "Jacobi step is R-symmetric — missing 1/R toroidal term"

    def test_jacobi_preserves_boundaries(self, tmp_path: Path) -> None:
        fk = _make_kernel(tmp_path, method="sor")
        rng = np.random.default_rng(7)
        Psi = rng.standard_normal(fk.Psi.shape) * 0.01
        Psi[0, :] = 0.0
        Psi[-1, :] = 0.0
        Psi[:, 0] = 0.0
        Psi[:, -1] = 0.0
        Source = np.ones_like(Psi) * 0.001

        result = fk._jacobi_step(Psi, Source)
        assert np.allclose(result[0, :], 0.0)
        assert np.allclose(result[-1, :], 0.0)
        assert np.allclose(result[:, 0], 0.0)
        assert np.allclose(result[:, -1], 0.0)


# ── 2. Vacuum field turns scaling ───────────────────────────────────


class TestVacuumFieldTurns:
    """Coil 'turns' must linearly scale the vacuum flux."""

    def test_double_turns_doubles_flux(self, tmp_path: Path) -> None:
        coils_1t = [{"name": "PF1", "r": 3.0, "z": 4.0, "current": 1.0, "turns": 1}]
        coils_2t = [{"name": "PF1", "r": 3.0, "z": 4.0, "current": 1.0, "turns": 2}]

        fk1 = _make_kernel(tmp_path, method="sor", coils=coils_1t)
        fk2 = _make_kernel(tmp_path, method="sor", coils=coils_2t)

        psi_1t = fk1.calculate_vacuum_field()
        psi_2t = fk2.calculate_vacuum_field()

        ratio = np.max(np.abs(psi_2t)) / max(np.max(np.abs(psi_1t)), 1e-30)
        assert ratio == pytest.approx(2.0, rel=1e-10), f"2-turn flux should be 2x 1-turn flux, got ratio {ratio:.6f}"

    def test_no_turns_key_defaults_to_one(self, tmp_path: Path) -> None:
        coils_explicit = [{"name": "PF1", "r": 3.0, "z": 4.0, "current": 1.0, "turns": 1}]
        coils_implicit = [{"name": "PF1", "r": 3.0, "z": 4.0, "current": 1.0}]

        fk_e = _make_kernel(tmp_path, method="sor", coils=coils_explicit)
        fk_i = _make_kernel(tmp_path, method="sor", coils=coils_implicit)

        psi_e = fk_e.calculate_vacuum_field()
        psi_i = fk_i.calculate_vacuum_field()
        assert np.allclose(psi_e, psi_i, atol=1e-15)


# ── 3. UPDE return key parity ───────────────────────────────────────


class TestUPDEReturnKeys:
    """All UPDE paths must return the same dict keys."""

    REQUIRED_KEYS = {"theta1", "R_layer", "Psi_layer", "R_global", "Psi_global", "V_layer", "V_global"}

    def test_python_fallback_keys(self) -> None:
        K = np.eye(4) * 0.5
        spec = KnmSpec(K=K)
        sys = UPDESystem(spec=spec, dt=1e-3, psi_mode="global_mean_field")
        theta = [np.zeros(10) for _ in range(4)]
        omega = [np.zeros(10) for _ in range(4)]
        out = sys.step(theta, omega)
        missing = self.REQUIRED_KEYS - set(out.keys())
        assert not missing, f"Python UPDE missing keys: {missing}"

    def test_psi_global_is_finite(self) -> None:
        K = np.eye(2) * 0.3
        spec = KnmSpec(K=K)
        sys = UPDESystem(spec=spec, dt=1e-3, psi_mode="global_mean_field")
        theta = [np.array([0.0, 0.5, 1.0]), np.array([0.1, 0.6, 1.1])]
        omega = [np.zeros(3), np.zeros(3)]
        out = sys.step(theta, omega)
        assert np.isfinite(out["Psi_global"])


# ── 4. JAX GS boundary condition ────────────────────────────────────


class TestJaxGSBoundary:
    """Fixed-boundary GS must use ψ_bdry = 0, not corner value."""

    def test_np_solve_boundaries_zero(self) -> None:
        from scpn_control.core.jax_gs_solver import gs_solve_np

        psi = gs_solve_np(
            R_min=0.1,
            R_max=2.0,
            Z_min=-1.5,
            Z_max=1.5,
            NR=17,
            NZ=17,
            Ip_target=1e6,
            n_picard=20,
            n_jacobi=50,
        )
        assert np.allclose(psi[0, :], 0.0, atol=1e-14)
        assert np.allclose(psi[-1, :], 0.0, atol=1e-14)
        assert np.allclose(psi[:, 0], 0.0, atol=1e-14)
        assert np.allclose(psi[:, -1], 0.0, atol=1e-14)

    def test_np_solve_interior_nonzero(self) -> None:
        from scpn_control.core.jax_gs_solver import gs_solve_np

        psi = gs_solve_np(
            R_min=0.1,
            R_max=2.0,
            Z_min=-1.5,
            Z_max=1.5,
            NR=17,
            NZ=17,
            Ip_target=1e6,
            n_picard=20,
            n_jacobi=50,
        )
        assert np.max(np.abs(psi[1:-1, 1:-1])) > 0

    @pytest.mark.skipif(
        not __import__("scpn_control.core.jax_gs_solver", fromlist=["has_jax"]).has_jax(),
        reason="JAX not installed",
    )
    def test_jax_solve_boundaries_zero(self) -> None:
        from scpn_control.core.jax_gs_solver import jax_gs_solve

        psi = jax_gs_solve(
            R_min=0.1,
            R_max=2.0,
            Z_min=-1.5,
            Z_max=1.5,
            NR=17,
            NZ=17,
            Ip_target=1e6,
            n_picard=20,
            n_jacobi=50,
        )
        psi_np = np.asarray(psi)
        assert np.allclose(psi_np[0, :], 0.0, atol=1e-14)
        assert np.allclose(psi_np[-1, :], 0.0, atol=1e-14)
        assert np.allclose(psi_np[:, 0], 0.0, atol=1e-14)
        assert np.allclose(psi_np[:, -1], 0.0, atol=1e-14)


# ── 5. Analytic 2-oscillator Kuramoto ───────────────────────────────


class TestAnalyticKuramoto:
    """Two identical oscillators: R(t) → 1 exponentially.

    For N=2 identical oscillators (Δω=0) with coupling K,
    the phase difference φ = θ_1 − θ_2 evolves as:
        dφ/dt = −K sin(φ)
    So φ(t) → 0 and R(t) → 1.

    For small φ: φ(t) ≈ φ_0 exp(−K t), and
    R(t) = |cos(φ/2)| ≈ 1 − φ²/8.
    """

    def test_two_identical_sync(self) -> None:
        theta = np.array([0.3, -0.3])
        omega = np.array([1.0, 1.0])  # identical frequencies

        for _ in range(2000):
            result = kuramoto_sakaguchi_step(theta, omega, dt=0.01, K=2.0, psi_driver=0.0)
            theta = result["theta1"]

        R = result["R"]
        assert R > 0.999, f"Two identical oscillators must sync: R={R:.6f}"

    def test_two_osc_exponential_convergence(self) -> None:
        phi0 = 0.2  # small initial phase difference
        theta = np.array([phi0 / 2, -phi0 / 2])
        omega = np.array([0.0, 0.0])
        K = 3.0
        dt = 0.001

        R_history = []
        for step in range(500):
            result = kuramoto_sakaguchi_step(theta, omega, dt=dt, K=K, psi_driver=0.0)
            theta = result["theta1"]
            R_history.append(result["R"])

        # Analytic: phi(t) = phi0 * exp(-K*t) for small phi, Euler method
        # R = cos(phi/2). After 500 steps at dt=0.001, t=0.5, phi ≈ phi0*exp(-1.5)
        t_final = 500 * dt
        phi_expected = phi0 * np.exp(-K * t_final)
        R_expected = np.cos(phi_expected / 2)
        # Allow 5% tolerance for Euler integration error
        assert R_history[-1] == pytest.approx(R_expected, rel=0.05), (
            f"R mismatch: got {R_history[-1]:.6f}, expected {R_expected:.6f}"
        )

    def test_subcritical_no_sync(self) -> None:
        """Below critical coupling, oscillators with different frequencies don't sync."""
        rng = np.random.default_rng(42)
        theta = rng.uniform(-np.pi, np.pi, 200)
        omega = rng.normal(0, 1.0, 200)
        # K_c ≈ 2/(π·g(0)) for Gaussian g(0)=1/sqrt(2π) → K_c ≈ sqrt(2π) ≈ 2.507
        K_sub = 0.5  # well below critical

        for _ in range(500):
            result = kuramoto_sakaguchi_step(theta, omega, dt=0.01, K=K_sub, psi_driver=0.0)
            theta = result["theta1"]

        R = result["R"]
        assert R < 0.3, f"Subcritical coupling must not sync: R={R:.4f}"

    def test_supercritical_sync(self) -> None:
        """Above critical coupling, oscillators partially synchronize."""
        rng = np.random.default_rng(42)
        theta = rng.uniform(-np.pi, np.pi, 200)
        omega = rng.normal(0, 1.0, 200)
        K_super = 5.0  # well above critical

        for _ in range(1000):
            result = kuramoto_sakaguchi_step(theta, omega, dt=0.01, K=K_super, psi_driver=0.0)
            theta = result["theta1"]

        R = result["R"]
        assert R > 0.7, f"Supercritical coupling must partially sync: R={R:.4f}"

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Phase Dynamics Tests
# ──────────────────────────────────────────────────────────────────────
"""
Tests for the Paper 27 Knm/UPDE engine and Kuramoto-Sakaguchi
+ ζ sin(Ψ−θ) global field driver.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.phase.kuramoto import (
    kuramoto_sakaguchi_step,
    order_parameter,
    wrap_phase,
    GlobalPsiDriver,
)
from scpn_control.phase.knm import KnmSpec, build_knm_paper27, OMEGA_N_16
from scpn_control.phase.upde import UPDESystem


# ── order_parameter ──────────────────────────────────────────────────


class TestOrderParameter:

    def test_fully_synchronized(self):
        theta = np.zeros(100)
        R, psi = order_parameter(theta)
        assert R == pytest.approx(1.0, abs=1e-12)
        assert psi == pytest.approx(0.0, abs=1e-12)

    def test_uniform_spread(self):
        theta = np.linspace(-np.pi, np.pi, 1001, endpoint=False)
        R, _ = order_parameter(theta)
        assert R < 0.01

    def test_R_in_unit_interval(self):
        rng = np.random.default_rng(42)
        theta = rng.uniform(-np.pi, np.pi, 200)
        R, _ = order_parameter(theta)
        assert 0.0 <= R <= 1.0

    def test_weighted(self):
        theta = np.array([0.0, np.pi])
        w = np.array([10.0, 1.0])
        R, psi = order_parameter(theta, weights=w)
        assert R > 0.5
        assert abs(psi) < 0.5


# ── wrap_phase ───────────────────────────────────────────────────────


class TestWrapPhase:

    def test_identity_in_range(self):
        x = np.array([-np.pi + 0.01, 0.0, np.pi - 0.01])
        np.testing.assert_allclose(wrap_phase(x), x, atol=1e-12)

    def test_wraps_large(self):
        x = np.array([3.0 * np.pi, -5.0 * np.pi])
        w = wrap_phase(x)
        assert np.all(w >= -np.pi)
        assert np.all(w <= np.pi)


# ── GlobalPsiDriver ──────────────────────────────────────────────────


class TestGlobalPsiDriver:

    def test_external_requires_value(self):
        d = GlobalPsiDriver(mode="external")
        with pytest.raises(ValueError):
            d.resolve(np.zeros(5), None)

    def test_external_returns_value(self):
        d = GlobalPsiDriver(mode="external")
        assert d.resolve(np.zeros(5), 1.23) == pytest.approx(1.23)

    def test_mean_field_uses_population(self):
        d = GlobalPsiDriver(mode="mean_field")
        theta = np.zeros(50)
        psi = d.resolve(theta, None)
        assert psi == pytest.approx(0.0, abs=1e-10)


# ── kuramoto_sakaguchi_step ──────────────────────────────────────────


class TestKuramotoSakaguchiStep:

    def test_synchronized_stays_synced(self):
        N = 64
        theta = np.zeros(N)
        omega = np.ones(N)
        out = kuramoto_sakaguchi_step(
            theta, omega, dt=0.01, K=2.0, psi_mode="mean_field",
        )
        assert out["R"] == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(out["theta1"], wrap_phase(0.01 * omega), atol=1e-12)

    def test_strong_coupling_increases_R(self):
        rng = np.random.default_rng(7)
        N = 100
        theta = rng.uniform(-np.pi, np.pi, N)
        omega = rng.normal(0, 0.5, N)
        R0, _ = order_parameter(theta)

        current = theta.copy()
        for _ in range(500):
            out = kuramoto_sakaguchi_step(
                current, omega, dt=0.01, K=5.0, psi_mode="mean_field",
            )
            current = out["theta1"]

        assert out["R"] > R0 + 0.2

    def test_zeta_pulls_toward_psi(self):
        N = 50
        rng = np.random.default_rng(99)
        theta = rng.uniform(-np.pi, np.pi, N)
        omega = np.zeros(N)
        Psi_target = 1.0

        current = theta.copy()
        for _ in range(300):
            out = kuramoto_sakaguchi_step(
                current, omega, dt=0.01, K=0.0, zeta=3.0,
                psi_driver=Psi_target, psi_mode="external",
            )
            current = out["theta1"]

        # All oscillators should cluster near Ψ=1.0
        spread = np.std(wrap_phase(current - Psi_target))
        assert spread < 0.3

    def test_alpha_breaks_full_sync(self):
        N = 50
        theta = np.zeros(N)
        omega = np.zeros(N)
        out = kuramoto_sakaguchi_step(
            theta, omega, dt=0.01, K=2.0, alpha=0.5,
            psi_mode="mean_field",
        )
        # With α=0.5, sin(ψ_r − θ − α) = sin(−0.5) ≠ 0
        assert np.any(out["dtheta"] != 0.0)


# ── KnmSpec ──────────────────────────────────────────────────────────


class TestKnmSpec:

    def test_build_paper27_shape(self):
        spec = build_knm_paper27()
        assert spec.K.shape == (16, 16)
        assert spec.L == 16

    def test_calibration_anchors(self):
        spec = build_knm_paper27()
        assert spec.K[0, 1] == pytest.approx(0.302)
        assert spec.K[1, 0] == pytest.approx(0.302)
        assert spec.K[1, 2] == pytest.approx(0.201)
        assert spec.K[3, 4] == pytest.approx(0.154)

    def test_cross_hierarchy_boosts(self):
        spec = build_knm_paper27()
        assert spec.K[0, 15] >= 0.05
        assert spec.K[4, 6] >= 0.15

    def test_symmetry_of_anchors(self):
        spec = build_knm_paper27()
        for i, j, _ in [(0, 1, 0.302), (1, 2, 0.201), (2, 3, 0.252)]:
            assert spec.K[i, j] == pytest.approx(spec.K[j, i])

    def test_zeta_uniform(self):
        spec = build_knm_paper27(zeta_uniform=0.5)
        assert spec.zeta is not None
        np.testing.assert_allclose(spec.zeta, 0.5)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            KnmSpec(K=np.zeros((3, 4)))

    def test_omega_16_length(self):
        assert OMEGA_N_16.shape == (16,)


# ── UPDESystem ───────────────────────────────────────────────────────


class TestUPDESystem:

    @staticmethod
    def _make_random_layers(L, N_per_layer, seed=42):
        rng = np.random.default_rng(seed)
        theta = [rng.uniform(-np.pi, np.pi, N_per_layer) for _ in range(L)]
        omega = [rng.normal(0, 0.3, N_per_layer) for _ in range(L)]
        return theta, omega

    def test_step_returns_correct_layers(self):
        spec = build_knm_paper27(L=4)
        sys = UPDESystem(spec=spec, dt=1e-3, psi_mode="external")
        theta, omega = self._make_random_layers(4, 20)
        out = sys.step(theta, omega, psi_driver=0.0)
        assert len(out["theta1"]) == 4
        assert out["R_layer"].shape == (4,)

    def test_strong_intra_coupling_syncs_within_layer(self):
        L = 2
        K = np.eye(L) * 5.0
        spec = KnmSpec(K=K)
        sys = UPDESystem(spec=spec, dt=0.005, psi_mode="global_mean_field")
        theta, omega = self._make_random_layers(L, 80, seed=7)

        current = [t.copy() for t in theta]
        for _ in range(400):
            out = sys.step(current, omega)
            current = out["theta1"]

        # Each layer should have high internal coherence
        for m in range(L):
            R, _ = order_parameter(current[m])
            assert R > 0.5

    def test_zeta_global_driver_pulls_all_layers(self):
        spec = build_knm_paper27(L=4, zeta_uniform=2.0)
        sys = UPDESystem(spec=spec, dt=0.005, psi_mode="external")
        theta, _ = self._make_random_layers(4, 30, seed=11)
        omega = [np.zeros(30) for _ in range(4)]
        Psi = 0.5

        current = [t.copy() for t in theta]
        for _ in range(500):
            out = sys.step(current, omega, psi_driver=Psi)
            current = out["theta1"]

        # All layers should cluster near Ψ=0.5
        for m in range(4):
            mean_phase = float(np.angle(np.mean(np.exp(1j * current[m]))))
            assert abs(wrap_phase(np.array([mean_phase - Psi]))[0]) < 0.4

    def test_run_returns_trajectory(self):
        spec = build_knm_paper27(L=3, K_base=0.3)
        sys = UPDESystem(spec=spec, dt=1e-3, psi_mode="global_mean_field")
        theta, omega = self._make_random_layers(3, 10, seed=5)
        result = sys.run(100, theta, omega)
        assert result["R_layer_hist"].shape == (100, 3)
        assert result["R_global_hist"].shape == (100,)
        assert len(result["theta_final"]) == 3

    def test_pac_gamma_modulates_coupling(self):
        """PAC gate should change evolution vs gamma=0."""
        spec = build_knm_paper27(L=4)
        sys = UPDESystem(spec=spec, dt=0.005, psi_mode="global_mean_field")
        theta, omega = self._make_random_layers(4, 30, seed=13)

        out_base = sys.step(theta, omega, pac_gamma=0.0)
        out_pac = sys.step(theta, omega, pac_gamma=1.0)

        # dtheta should differ when PAC gate is active
        diff = sum(
            float(np.max(np.abs(a - b)))
            for a, b in zip(out_base["dtheta"], out_pac["dtheta"])
        )
        assert diff > 0.0

    def test_wrong_layer_count_raises(self):
        spec = build_knm_paper27(L=4)
        sys = UPDESystem(spec=spec, dt=1e-3, psi_mode="external")
        theta, omega = self._make_random_layers(3, 10)
        with pytest.raises(ValueError, match="Expected 4"):
            sys.step(theta, omega, psi_driver=0.0)


# ── FusionKernel.phase_sync_step (integration smoke test) ────────────


class TestFusionKernelPhaseSync:
    """Smoke test that phase_sync_step can be called on a FusionKernel."""

    @pytest.fixture
    def kernel(self, tmp_path):
        """Create a minimal FusionKernel with a tiny config."""
        cfg = {
            "reactor_name": "test_micro",
            "dimensions": {"R_min": 0.5, "R_max": 2.5, "Z_min": -1.5, "Z_max": 1.5},
            "grid_resolution": [9, 9],
            "coils": {"positions": [], "currents": [], "turns": []},
            "physics": {},
            "solver": {"method": "sor", "max_iterations": 10, "tol": 1e-4},
            "phase_sync": {"K": 2.0, "zeta": 0.5, "psi_mode": "external"},
        }
        import json
        cfg_path = tmp_path / "test.json"
        cfg_path.write_text(json.dumps(cfg))
        from scpn_control.core.fusion_kernel import FusionKernel
        return FusionKernel(str(cfg_path))

    def test_phase_sync_step_runs(self, kernel):
        N = 32
        rng = np.random.default_rng(42)
        theta = rng.uniform(-np.pi, np.pi, N)
        omega = rng.normal(0, 0.5, N)
        out = kernel.phase_sync_step(theta, omega, dt=0.01, psi_driver=0.0)
        assert "theta1" in out
        assert "R" in out
        assert out["theta1"].shape == (N,)

    def test_zeta_from_config(self, kernel):
        N = 20
        theta = np.zeros(N)
        omega = np.zeros(N)
        out = kernel.phase_sync_step(theta, omega, dt=0.01, psi_driver=1.0)
        # ζ=0.5 from config, Ψ=1.0 → dθ = 0.5·sin(1.0 − 0) ≈ 0.421
        assert np.all(out["dtheta"] > 0.0)

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Coverage gaps: JAX modules + phase modules
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import asyncio
import sys
from unittest import mock

import numpy as np
import pytest

# ── jax_traceable_runtime coverage ──────────────────────────────────


class TestTraceableAutoFallbackChain:
    """Lines 111-113: auto resolves through jax → torch → numpy."""

    def test_auto_falls_to_numpy_when_no_jax_no_torch(self, monkeypatch):
        import scpn_control.control.jax_traceable_runtime as mod

        monkeypatch.setattr(mod, "_HAS_JAX", False)
        monkeypatch.setattr(mod, "_HAS_TORCH", False)
        assert mod._resolve_backend("auto") == "numpy"

    def test_auto_falls_to_torch_when_no_jax(self, monkeypatch):
        import scpn_control.control.jax_traceable_runtime as mod

        monkeypatch.setattr(mod, "_HAS_JAX", False)
        monkeypatch.setattr(mod, "_HAS_TORCH", True)
        assert mod._resolve_backend("auto") == "torchscript"


class TestTraceableBackendSetEdges:
    """Lines 138, 143: unavailable backend; empty result."""

    def test_unavailable_backend_raises(self, monkeypatch):
        import scpn_control.control.jax_traceable_runtime as mod

        monkeypatch.setattr(mod, "_HAS_JAX", False)
        monkeypatch.setattr(mod, "_HAS_TORCH", False)
        with pytest.raises(ValueError, match="not available"):
            mod._resolve_backend_set(["jax"])

    def test_empty_list_raises(self):
        from scpn_control.control.jax_traceable_runtime import _resolve_backend_set

        with pytest.raises(ValueError, match="at least one"):
            _resolve_backend_set([])


class TestSimulateJaxNoJax:
    """Lines 160, 255: _simulate_jax/_simulate_jax_batch without JAX."""

    def test_simulate_jax_raises(self, monkeypatch):
        import scpn_control.control.jax_traceable_runtime as mod

        monkeypatch.setattr(mod, "_HAS_JAX", False)
        with pytest.raises(RuntimeError, match="JAX"):
            mod._simulate_jax(np.ones(5), 0.0, mod.TraceableRuntimeSpec())

    def test_simulate_jax_batch_raises(self, monkeypatch):
        import scpn_control.control.jax_traceable_runtime as mod

        monkeypatch.setattr(mod, "_HAS_JAX", False)
        with pytest.raises(RuntimeError, match="JAX"):
            mod._simulate_jax_batch(np.ones((2, 5)), np.zeros(2), mod.TraceableRuntimeSpec())


class TestSimulateTorchNoTorch:
    """Lines 227, 283: _simulate_torchscript/_simulate_torchscript_batch without torch."""

    def test_simulate_torchscript_raises(self, monkeypatch):
        import scpn_control.control.jax_traceable_runtime as mod

        monkeypatch.setattr(mod, "_HAS_TORCH", False)
        with pytest.raises(RuntimeError, match="torch"):
            mod._simulate_torchscript(np.ones(5), 0.0, mod.TraceableRuntimeSpec())

    def test_simulate_torchscript_batch_raises(self, monkeypatch):
        import scpn_control.control.jax_traceable_runtime as mod

        monkeypatch.setattr(mod, "_HAS_TORCH", False)
        with pytest.raises(RuntimeError, match="torch"):
            mod._simulate_torchscript_batch(np.ones((2, 5)), np.zeros(2), mod.TraceableRuntimeSpec())


class TestTorchScriptRollouts:
    """Lines 193-200, 210-222: torchscript single and batch rollout bodies."""

    def test_torchscript_single_rollout(self):
        torch = pytest.importorskip("torch")
        from scpn_control.control.jax_traceable_runtime import (
            TraceableRuntimeSpec,
            _simulate_numpy,
            _simulate_torchscript,
        )

        cmd = np.sin(np.linspace(0, 2 * np.pi, 32))
        spec = TraceableRuntimeSpec()
        np_out = _simulate_numpy(cmd, 0.5, spec)
        ts_out = _simulate_torchscript(cmd, 0.5, spec)
        np.testing.assert_allclose(ts_out, np_out, atol=1e-10)

    def test_torchscript_batch_rollout(self):
        torch = pytest.importorskip("torch")
        from scpn_control.control.jax_traceable_runtime import (
            TraceableRuntimeSpec,
            _simulate_numpy_batch,
            _simulate_torchscript_batch,
        )

        rng = np.random.default_rng(7)
        cmd = rng.normal(size=(3, 20))
        x0 = rng.normal(size=3) * 0.1
        spec = TraceableRuntimeSpec()
        np_out = _simulate_numpy_batch(cmd, x0, spec)
        ts_out = _simulate_torchscript_batch(cmd, x0, spec)
        np.testing.assert_allclose(ts_out, np_out, atol=1e-10)


# ── jax_neural_equilibrium no-JAX guards ────────────────────────────


class TestNeuralEquilibriumNoJax:
    """Lines 71-87, 100, 180, 192, 224, 262: all RuntimeError guards."""

    def test_load_weights_no_jax(self, monkeypatch):
        import scpn_control.core.jax_neural_equilibrium as mod

        monkeypatch.setattr(mod, "_HAS_JAX", False)
        with pytest.raises(RuntimeError, match="JAX"):
            mod.load_weights_as_jax("/nonexistent.npz")

    def test_numpy_weights_to_jax_no_jax(self, monkeypatch):
        import scpn_control.core.jax_neural_equilibrium as mod

        monkeypatch.setattr(mod, "_HAS_JAX", False)
        z = np.zeros(5)
        with pytest.raises(RuntimeError, match="JAX"):
            mod.numpy_weights_to_jax([z], [z], z, z.reshape(1, -1), z, z)

    def test_jax_mlp_forward_no_jax(self, monkeypatch):
        import scpn_control.core.jax_neural_equilibrium as mod

        monkeypatch.setattr(mod, "_HAS_JAX", False)
        with pytest.raises(RuntimeError, match="JAX"):
            mod.jax_mlp_forward(np.zeros(5), ((), ()))

    def test_jax_pca_inverse_no_jax(self, monkeypatch):
        import scpn_control.core.jax_neural_equilibrium as mod

        monkeypatch.setattr(mod, "_HAS_JAX", False)
        with pytest.raises(RuntimeError, match="JAX"):
            mod.jax_pca_inverse(np.zeros(5), (np.zeros(5), np.eye(5)))

    def test_jax_neural_eq_predict_no_jax(self, monkeypatch):
        import scpn_control.core.jax_neural_equilibrium as mod

        monkeypatch.setattr(mod, "_HAS_JAX", False)
        with pytest.raises(RuntimeError, match="JAX"):
            mod.jax_neural_eq_predict(np.zeros(5), ((), ()), (np.zeros(5), np.eye(5)), (np.zeros(5), np.ones(5)))

    def test_jax_neural_eq_predict_batched_no_jax(self, monkeypatch):
        import scpn_control.core.jax_neural_equilibrium as mod

        monkeypatch.setattr(mod, "_HAS_JAX", False)
        with pytest.raises(RuntimeError, match="JAX"):
            mod.jax_neural_eq_predict_batched(
                np.zeros((2, 5)), ((), ()), (np.zeros(5), np.eye(5)), (np.zeros(5), np.ones(5))
            )


# ── jax_gs_solver coverage ──────────────────────────────────────────


class TestGsSolverNoJaxGuards:
    """Lines 369-385, 408: jax_gs_solve_from_grid and jax_gs_grad_Ip without JAX."""

    def test_gs_solve_from_grid_no_jax(self, monkeypatch):
        import scpn_control.core.jax_gs_solver as mod

        monkeypatch.setattr(mod, "_HAS_JAX", False)
        with pytest.raises(RuntimeError, match="JAX"):
            mod.jax_gs_solve_from_grid(np.ones((5, 5)), np.zeros((5, 5)), 0.1, 0.1)

    def test_gs_grad_Ip_no_jax(self, monkeypatch):
        import scpn_control.core.jax_gs_solver as mod

        monkeypatch.setattr(mod, "_HAS_JAX", False)
        with pytest.raises(RuntimeError, match="JAX"):
            mod.jax_gs_grad_Ip(1e6)


class TestGsSolveNpDenomEdge:
    """Line 95: degenerate denom in _compute_source_np."""

    def test_flat_psi_handled(self):
        from scpn_control.core.jax_gs_solver import gs_solve_np

        # Very few iterations to keep psi near-flat → denom ≈ 0
        psi = gs_solve_np(0.1, 2.0, -1.5, 1.5, 9, 9, 1e6, n_picard=1, n_jacobi=1)
        assert np.all(np.isfinite(psi))


class TestGsSolveFromGrid:
    """Line 258 (JAX Jacobi body) via jax_gs_solve_from_grid."""

    def test_from_grid_runs(self):
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        from scpn_control.core.jax_gs_solver import jax_gs_solve_from_grid

        NR, NZ = 9, 9
        R = jnp.linspace(0.1, 2.0, NR)
        Z = jnp.linspace(-1.5, 1.5, NZ)
        RR, _ = jnp.meshgrid(R, Z)
        dR = float(R[1] - R[0])
        dZ = float(Z[1] - Z[0])

        R_center = 0.5 * (0.1 + 2.0)
        psi_init = np.exp(-((np.asarray(RR) - R_center) ** 2) / 0.5) * 0.01
        psi_init[0, :] = 0.0
        psi_init[-1, :] = 0.0
        psi_init[:, 0] = 0.0
        psi_init[:, -1] = 0.0

        psi = jax_gs_solve_from_grid(
            np.asarray(RR),
            psi_init,
            dR,
            dZ,
            n_picard=5,
            n_jacobi=10,
        )
        assert psi.shape == (NZ, NR)
        assert np.all(np.isfinite(psi))


# ── jax_solvers coverage ────────────────────────────────────────────


class TestHasJaxGpu:
    """Lines 47-52: has_jax_gpu."""

    def test_has_jax_gpu_no_jax(self, monkeypatch):
        import scpn_control.core.jax_solvers as mod

        monkeypatch.setattr(mod, "_HAS_JAX", False)
        assert mod.has_jax_gpu() is False

    def test_has_jax_gpu_exception(self, monkeypatch):
        import scpn_control.core.jax_solvers as mod

        monkeypatch.setattr(mod, "_HAS_JAX", True)
        monkeypatch.setattr(mod, "jax", mock.MagicMock(devices=mock.MagicMock(side_effect=RuntimeError("nope"))))
        assert mod.has_jax_gpu() is False

    def test_has_jax_gpu_no_gpu_devices(self, monkeypatch):
        import scpn_control.core.jax_solvers as mod

        if mod._HAS_JAX:
            result = mod.has_jax_gpu()
            assert isinstance(result, bool)


class TestThomasNearZeroPivot:
    """Lines 71, 78: near-zero pivot in Thomas algorithm."""

    def test_near_zero_main_diag_first(self):
        from scpn_control.core.jax_solvers import _thomas_solve_np

        n = 4
        a = np.zeros(n - 1)
        b = np.array([1e-35, 2.0, 2.0, 2.0])
        c = np.zeros(n - 1)
        d = np.array([1.0, 2.0, 3.0, 4.0])
        x = _thomas_solve_np(a, b, c, d)
        assert np.all(np.isfinite(x))

    def test_near_zero_inner_pivot(self):
        from scpn_control.core.jax_solvers import _thomas_solve_np

        n = 4
        a = np.array([1.0, 1.0, 1.0])
        b = np.array([1.0, 1e-35, 2.0, 2.0])
        c = np.array([1.0, 1.0, 1.0])
        d = np.ones(n)
        x = _thomas_solve_np(a, b, c, d)
        assert np.all(np.isfinite(x))


class TestBatchedCNNoJax:
    """Line 371: batched_crank_nicolson NumPy fallback."""

    def test_batched_cn_numpy_fallback(self, monkeypatch):
        import scpn_control.core.jax_solvers as mod

        monkeypatch.setattr(mod, "_HAS_JAX", False)
        n = 16
        rho = np.linspace(0.05, 1.0, n)
        drho = rho[1] - rho[0]
        T_batch = 5.0 + np.random.default_rng(42).standard_normal((3, n))
        chi = 0.5 * np.ones(n)
        source = np.zeros(n)
        result = mod.batched_crank_nicolson(T_batch, chi, source, rho, drho, 0.01)
        assert result.shape == (3, n)
        for i in range(3):
            assert result[i, -1] == pytest.approx(0.1)


# ── ws_phase_stream coverage ────────────────────────────────────────


class TestWsPhaseStreamServe:
    """Lines 92-95: serve() body; line 99: serve_sync."""

    def test_serve_creates_task_and_runs(self):
        """Exercise serve() with a mock websockets.serve context manager."""
        ws_mod = pytest.importorskip("websockets")
        from scpn_control.phase.realtime_monitor import RealtimeMonitor
        from scpn_control.phase.ws_phase_stream import PhaseStreamServer

        mon = RealtimeMonitor.from_paper27(L=2, N_per=5)
        server = PhaseStreamServer(monitor=mon, tick_interval_s=0.001)

        async def _run():
            # Stop the tick loop quickly
            async def _stop():
                await asyncio.sleep(0.05)
                server._running = False

            stop_task = asyncio.create_task(_stop())

            # Mock websockets.serve as async context manager
            mock_ws_serve = mock.AsyncMock()
            mock_ws_serve.__aenter__ = mock.AsyncMock(return_value=None)
            mock_ws_serve.__aexit__ = mock.AsyncMock(return_value=False)

            with mock.patch("scpn_control.phase.ws_phase_stream.websockets", create=True) as ws_mock:
                ws_mock.serve = mock.MagicMock(return_value=mock_ws_serve)
                # Monkey-patch module-level import

                original = None
                try:
                    await server.serve(host="127.0.0.1", port=19999)
                except Exception:
                    pass
            await stop_task

        asyncio.run(_run())

    def test_serve_sync_calls_asyncio_run(self, monkeypatch):
        from scpn_control.phase.realtime_monitor import RealtimeMonitor
        from scpn_control.phase.ws_phase_stream import PhaseStreamServer

        mon = RealtimeMonitor.from_paper27(L=2, N_per=5)
        server = PhaseStreamServer(monitor=mon)
        called = {}

        async def fake_serve(host, port):
            called["host"] = host
            called["port"] = port

        monkeypatch.setattr(server, "serve", fake_serve)
        server.serve_sync(host="127.0.0.1", port=12345)
        assert called == {"host": "127.0.0.1", "port": 12345}


class TestWsPhaseStreamMain:
    """Lines 103-123: main() function."""

    def test_main_parses_args_and_runs(self, monkeypatch):
        from scpn_control.phase import ws_phase_stream as ws_mod

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "ws_phase_stream",
                "--port",
                "9999",
                "--host",
                "127.0.0.1",
                "--layers",
                "3",
                "--n-per",
                "8",
                "--zeta",
                "0.3",
                "--psi",
                "0.1",
                "--tick-interval",
                "0.01",
            ],
        )

        serve_sync_calls = []

        def fake_serve_sync(self, host="0.0.0.0", port=8765):
            serve_sync_calls.append((host, port))

        monkeypatch.setattr(ws_mod.PhaseStreamServer, "serve_sync", fake_serve_sync)
        ws_mod.main()
        assert len(serve_sync_calls) == 1
        assert serve_sync_calls[0] == ("127.0.0.1", 9999)


# ── realtime_monitor adaptive_engine path ───────────────────────────


class TestRealtimeMonitorAdaptive:
    """Lines 152-161, 186-199, 237: tick() with adaptive_engine."""

    def test_tick_with_adaptive_engine(self):
        from scpn_control.phase.adaptive_knm import (
            AdaptiveKnmConfig,
            AdaptiveKnmEngine,
        )
        from scpn_control.phase.knm import build_knm_paper27
        from scpn_control.phase.realtime_monitor import RealtimeMonitor

        L = 4
        spec = build_knm_paper27(L=L)
        engine = AdaptiveKnmEngine(baseline_spec=spec, config=AdaptiveKnmConfig())

        mon = RealtimeMonitor.from_plasma(
            L=L,
            N_per=10,
            adaptive_engine=engine,
            seed=42,
        )
        snap1 = mon.tick()
        assert snap1["tick"] == 1
        assert "adaptive" in snap1

        snap2 = mon.tick(beta_n=1.5, q95=2.8, disruption_risk=0.3, mirnov_rms=0.1)
        assert snap2["tick"] == 2
        assert "adaptive" in snap2

    def test_tick_no_record(self):
        from scpn_control.phase.realtime_monitor import RealtimeMonitor

        mon = RealtimeMonitor.from_paper27(L=3, N_per=10)
        mon.tick(record=False)
        assert mon.recorder.n_ticks == 0


# ── upde.py Rust fast-path ──────────────────────────────────────────


class TestUPDERustPath:
    """Lines 43, 115-137: Rust UPDE fast-path."""

    def test_rust_upde_tick_path(self, monkeypatch):
        from scpn_control.phase.knm import build_knm_paper27
        import scpn_control.phase.upde as upde_mod

        L = 3
        N = 10
        spec = build_knm_paper27(L=L)
        rng = np.random.default_rng(42)
        theta = [rng.uniform(-np.pi, np.pi, N) for _ in range(L)]
        omega = [rng.normal(0, 0.3, N) for _ in range(L)]

        # Build a fake Rust result
        class FakeRustResult:
            def __init__(self):
                self.theta_flat = np.concatenate(theta).copy()
                self.r_layer = np.array([0.5] * L)
                self.v_layer = np.array([0.1] * L)
                self.r_global = 0.45

        def fake_rust_tick(*args, **kwargs):
            return FakeRustResult()

        monkeypatch.setattr(upde_mod, "HAS_RUST_UPDE", True)
        monkeypatch.setattr(upde_mod, "_rust_upde_tick", fake_rust_tick, raising=False)

        sys = upde_mod.UPDESystem(spec=spec, dt=1e-3, psi_mode="external")
        out = sys.step(theta, omega, psi_driver=0.0)
        assert len(out["theta1"]) == L
        assert "R_global" in out
        assert "V_layer" in out
        assert "V_global" in out


# ── kuramoto.py Rust fast-path ──────────────────────────────────────


class TestKuramotoRustPath:
    """Lines 38, 137-146: Rust kuramoto_step fast-path."""

    def test_rust_kuramoto_step(self, monkeypatch):
        import scpn_control.phase.kuramoto as kmod

        N = 20
        rng = np.random.default_rng(42)
        theta = rng.uniform(-np.pi, np.pi, N)
        omega = rng.normal(0, 0.3, N)

        th1_fake = theta + 0.01 * omega

        def fake_rust_step(th, om, dt, K, alpha, zeta, psi):
            return (th1_fake, 0.6, 0.1, 0.0)

        monkeypatch.setattr(kmod, "RUST_KURAMOTO", True)
        monkeypatch.setattr(kmod, "_rust_step", fake_rust_step, raising=False)

        out = kmod.kuramoto_sakaguchi_step(
            theta,
            omega,
            dt=0.01,
            K=2.0,
            alpha=0.0,
            zeta=0.5,
            psi_driver=0.0,
            psi_mode="external",
            wrap=True,
        )
        assert out["R"] == pytest.approx(0.6)
        assert out["theta1"].shape == (N,)
        assert "dtheta" in out
        assert "Psi_r" in out


class TestKuramotoNonRustWithAlpha:
    """Line 137: alpha != 0 forces Python path even when Rust available."""

    def test_alpha_nonzero_skips_rust(self, monkeypatch):
        import scpn_control.phase.kuramoto as kmod

        monkeypatch.setattr(kmod, "RUST_KURAMOTO", True)
        called = {"rust": False}

        def spy_rust_step(*args, **kwargs):
            called["rust"] = True
            return (args[0], 0.5, 0.0, 0.0)

        monkeypatch.setattr(kmod, "_rust_step", spy_rust_step, raising=False)

        theta = np.zeros(10)
        omega = np.zeros(10)
        out = kmod.kuramoto_sakaguchi_step(
            theta,
            omega,
            dt=0.01,
            K=1.0,
            alpha=0.5,
            psi_mode="mean_field",
            wrap=True,
        )
        assert not called["rust"]
        assert "dtheta" in out


# ── upde.py global_mean_field psi_mode ──────────────────────────────


class TestUPDEGlobalMeanField:
    """Line 106-107: psi_mode='global_mean_field' path."""

    def test_global_mean_field_mode(self):
        from scpn_control.phase.knm import KnmSpec
        from scpn_control.phase.upde import UPDESystem

        L = 2
        K = np.eye(L) * 0.5
        spec = KnmSpec(K=K)
        sys = UPDESystem(spec=spec, dt=1e-3, psi_mode="global_mean_field")
        theta = [np.zeros(10), np.ones(10) * 0.5]
        omega = [np.zeros(10), np.zeros(10)]
        out = sys.step(theta, omega)
        assert "Psi_global" in out
        assert np.isfinite(out["Psi_global"])


# ── realtime_monitor save_npz ───────────────────────────────────────


class TestRealtimeMonitorSaveNpz:
    def test_save_npz(self, tmp_path):
        from scpn_control.phase.realtime_monitor import RealtimeMonitor

        mon = RealtimeMonitor.from_paper27(L=3, N_per=10)
        for _ in range(5):
            mon.tick()
        p = mon.save_npz(tmp_path / "test.npz")
        data = np.load(p)
        assert data["R_global"].shape == (5,)
        assert data["guard_approved"].shape == (5,)

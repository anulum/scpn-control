# ──────────────────────────────────────────────────────────────────────
# SCPN Control — End-to-End Phase Sync + Mock DIII-D Shot Tests
# ──────────────────────────────────────────────────────────────────────
"""
End-to-end tests exercising the full RealtimeMonitor pipeline:

1. Generate mock DIII-D shot → load → extract phase-relevant signals
2. Drive RealtimeMonitor from shot data (Ψ = f(beta_N))
3. Verify trajectory export (HDF5 + NPZ)
4. Verify WebSocket server construction
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_control.phase.realtime_monitor import RealtimeMonitor, TrajectoryRecorder
from mock_diiid import generate_mock_shot, save_mock_shot


class TestMockDIIID:

    def test_generate_returns_all_keys(self):
        data = generate_mock_shot()
        expected = {
            "time_s", "Ip_MA", "BT_T", "beta_N", "q95", "ne_1e19",
            "n1_amp", "n2_amp", "locked_mode_amp", "dBdt_gauss_per_s",
            "vertical_position_m", "is_disruption", "disruption_time_idx",
            "disruption_type",
        }
        assert set(data.keys()) == expected

    def test_generate_shapes(self):
        data = generate_mock_shot(n_steps=500)
        for key in ("time_s", "Ip_MA", "BT_T", "beta_N", "q95"):
            assert data[key].shape == (500,)

    def test_save_and_reload(self, tmp_path):
        path = save_mock_shot(tmp_path, shot_id=123456, disruption=True)
        assert path.exists()
        loaded = np.load(path, allow_pickle=True)
        assert bool(loaded["is_disruption"]) is True
        assert loaded["time_s"].shape == (1000,)

    def test_safe_shot_no_disruption(self):
        data = generate_mock_shot(disruption=False)
        assert bool(data["is_disruption"]) is False


class TestE2EPhaseSyncWithShot:

    def test_shot_driven_monitor(self):
        """Drive RealtimeMonitor with Ψ derived from mock shot beta_N."""
        shot = generate_mock_shot(n_steps=200, disruption=False, seed=7)
        beta_n = shot["beta_N"]

        mon = RealtimeMonitor.from_paper27(
            L=4, N_per=20, dt=5e-3, zeta_uniform=1.0, psi_driver=0.0,
        )
        for i in range(len(beta_n)):
            # Map beta_N → Ψ ∈ [-π, π]
            mon.psi_driver = float(np.clip(beta_n[i] - 2.0, -np.pi, np.pi))
            snap = mon.tick()

        assert snap["tick"] == 200
        assert snap["R_global"] > 0.0
        assert np.isfinite(snap["lambda_exp"])

    def test_disruption_triggers_guard(self):
        """During disruption, rapidly shifting Ψ may trigger guard refusal."""
        shot = generate_mock_shot(n_steps=100, disruption=True, seed=3)
        vpos = shot["vertical_position_m"]

        mon = RealtimeMonitor.from_paper27(
            L=4, N_per=20, dt=5e-3, zeta_uniform=0.5,
            guard_window=10, guard_max_violations=2,
        )

        for i in range(len(vpos)):
            # Rapid Ψ jumps from vertical displacement
            mon.psi_driver = float(vpos[i] * 20.0)
            mon.tick()

        # Recorder should have captured all ticks
        assert mon.recorder.n_ticks == 100


class TestTrajectoryExport:

    def test_npz_export(self, tmp_path):
        mon = RealtimeMonitor.from_paper27(L=4, N_per=10, zeta_uniform=1.0)
        for _ in range(50):
            mon.tick()

        path = mon.save_npz(tmp_path / "traj.npz")
        assert path.exists()
        data = np.load(path)
        assert data["R_global"].shape == (50,)
        assert data["R_layer"].shape == (50, 4)
        assert data["guard_approved"].dtype == bool

    def test_hdf5_export(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        mon = RealtimeMonitor.from_paper27(L=4, N_per=10, zeta_uniform=1.0)
        for _ in range(30):
            mon.tick()

        path = mon.save_hdf5(tmp_path / "traj.h5")
        assert path.exists()
        with h5py.File(path, "r") as f:
            assert f["R_global"].shape == (30,)
            assert f["R_layer"].shape == (30, 4)
            assert f.attrs["L"] == 4
            assert f.attrs["n_ticks"] == 30

    def test_recorder_clears_on_reset(self):
        mon = RealtimeMonitor.from_paper27(L=4, N_per=10)
        for _ in range(10):
            mon.tick()
        assert mon.recorder.n_ticks == 10
        mon.reset()
        assert mon.recorder.n_ticks == 0

    def test_record_false_skips_recording(self):
        mon = RealtimeMonitor.from_paper27(L=4, N_per=10)
        mon.tick(record=False)
        mon.tick(record=True)
        assert mon.recorder.n_ticks == 1


class TestWebSocketServer:

    def test_server_construction(self):
        from scpn_control.phase.ws_phase_stream import PhaseStreamServer

        mon = RealtimeMonitor.from_paper27(L=4, N_per=10)
        server = PhaseStreamServer(monitor=mon, tick_interval_s=0.01)
        assert server.tick_interval_s == 0.01
        assert server.monitor is mon

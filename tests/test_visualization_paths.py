# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Visualization Path Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Tests exercising every save_plot / save_animation / save_report matplotlib path."""
from __future__ import annotations

import numpy as np
import pytest

# ── Skip entire module if matplotlib is absent ──
mpl = pytest.importorskip("matplotlib")
import matplotlib
matplotlib.use("Agg")

from scpn_control.control.fusion_control_room import (
    _render_outputs,
    run_control_room,
)


# ── helpers ──────────────────────────────────────────────────────────


def _make_frames(n: int = 4) -> tuple:
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(n):
        frames.append({
            "density": rng.random((10, 10)),
            "psi": rng.random((10, 10)),
            "r_min": 3.0, "r_max": 9.0,
            "z_min": -5.0, "z_max": 5.0,
        })
    h_z = [rng.uniform(-0.5, 0.5) for _ in range(n)]
    h_top = [rng.uniform(0.0, 1.0) for _ in range(n)]
    h_bot = [rng.uniform(0.0, 1.0) for _ in range(n)]
    return frames, h_z, h_top, h_bot


# ── _render_outputs direct tests ─────────────────────────────────────


class TestRenderOutputs:
    def test_save_report_only(self, tmp_path):
        frames, h_z, h_top, h_bot = _make_frames()
        report = str(tmp_path / "report.png")
        anim_ok, anim_err, rep_ok, rep_err = _render_outputs(
            frames, h_z, h_top, h_bot,
            save_animation=False, save_report=True,
            output_gif="unused.gif", output_report=report,
        )
        assert rep_ok is True
        assert rep_err is None
        assert anim_ok is False

    def test_save_animation_only(self, tmp_path):
        frames, h_z, h_top, h_bot = _make_frames(3)
        gif = str(tmp_path / "anim.gif")
        anim_ok, anim_err, rep_ok, rep_err = _render_outputs(
            frames, h_z, h_top, h_bot,
            save_animation=True, save_report=False,
            output_gif=gif, output_report="unused.png",
        )
        assert anim_ok is True
        assert anim_err is None
        assert rep_ok is False

    def test_save_both(self, tmp_path):
        frames, h_z, h_top, h_bot = _make_frames(3)
        gif = str(tmp_path / "both.gif")
        report = str(tmp_path / "both.png")
        anim_ok, anim_err, rep_ok, rep_err = _render_outputs(
            frames, h_z, h_top, h_bot,
            save_animation=True, save_report=True,
            output_gif=gif, output_report=report,
        )
        assert anim_ok is True
        assert rep_ok is True

    def test_bad_output_gif_path(self, tmp_path):
        frames, h_z, h_top, h_bot = _make_frames(3)
        bad_path = str(tmp_path / "nonexistent_dir" / "anim.gif")
        anim_ok, anim_err, _, _ = _render_outputs(
            frames, h_z, h_top, h_bot,
            save_animation=True, save_report=False,
            output_gif=bad_path, output_report="unused.png",
        )
        assert anim_ok is False
        assert anim_err is not None

    def test_bad_output_report_path(self, tmp_path):
        frames, h_z, h_top, h_bot = _make_frames(3)
        bad_path = str(tmp_path / "nonexistent_dir" / "report.png")
        _, _, rep_ok, rep_err = _render_outputs(
            frames, h_z, h_top, h_bot,
            save_animation=False, save_report=True,
            output_gif="unused.gif", output_report=bad_path,
        )
        assert rep_ok is False
        assert rep_err is not None


# ── run_control_room with save_animation / save_report ────────────────


class TestControlRoomVisualization:
    def test_run_with_save_report(self, tmp_path):
        report = str(tmp_path / "cr_report.png")
        s = run_control_room(
            sim_duration=5, seed=0,
            save_animation=False, save_report=True,
            output_report=report,
        )
        assert s["report_saved"] is True
        assert s["report_error"] is None

    def test_run_with_save_animation(self, tmp_path):
        gif = str(tmp_path / "cr_anim.gif")
        s = run_control_room(
            sim_duration=3, seed=0,
            save_animation=True, save_report=False,
            output_gif=gif,
        )
        assert s["animation_saved"] is True
        assert s["animation_error"] is None


# ── tokamak_digital_twin save_plot=True ──────────────────────────────

from scpn_control.control.tokamak_digital_twin import run_digital_twin


class TestDigitalTwinVisualization:
    def test_save_plot_true(self, tmp_path):
        out = str(tmp_path / "twin.png")
        s = run_digital_twin(
            time_steps=20, seed=0,
            save_plot=True, output_path=out, verbose=False,
        )
        assert s["plot_saved"] is True
        assert s["plot_error"] is None

    def test_save_plot_bad_path(self, tmp_path):
        bad = str(tmp_path / "nonexistent" / "twin.png")
        s = run_digital_twin(
            time_steps=20, seed=0,
            save_plot=True, output_path=bad, verbose=True,
        )
        assert s["plot_saved"] is False
        assert s["plot_error"] is not None


# ── tokamak_flight_sim: visualize_flight + save_plot ─────────────────

from scpn_control.control.tokamak_flight_sim import run_flight_sim


class _FlightKernel:
    def __init__(self, _cfg: str) -> None:
        self.cfg = {
            "physics": {"plasma_current_target": 5.0},
            "coils": [{"current": 0.0} for _ in range(5)],
        }
        self.R = np.linspace(5.8, 6.4, 13)
        self.Z = np.linspace(-0.3, 0.3, 13)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros((len(self.Z), len(self.R)), dtype=np.float64)
        self._ticks = 0
        self.solve_equilibrium()

    def solve_equilibrium(self) -> None:
        self._ticks += 1
        radial_drive = float(self.cfg["coils"][2]["current"])
        vertical_drive = float(self.cfg["coils"][4]["current"]) - float(
            self.cfg["coils"][0]["current"]
        )
        center_r = 6.1 + 0.05 * np.tanh(radial_drive / 10.0)
        center_z = 0.0 + 0.04 * np.tanh(vertical_drive / 10.0)
        ir = int(np.argmin(np.abs(self.R - center_r)))
        iz = int(np.argmin(np.abs(self.Z - center_z)))
        self.Psi.fill(-1.0)
        self.Psi[iz, ir] = 1.0 + 0.001 * float(
            self.cfg["physics"]["plasma_current_target"]
        )

    def find_x_point(self, _psi):
        return (float(self.R[-2]), float(self.Z[1])), 0.0


class TestFlightSimVisualization:
    def test_save_plot_true(self, tmp_path):
        out = str(tmp_path / "flight.png")
        s = run_flight_sim(
            shot_duration=10, seed=42,
            save_plot=True, output_path=out, verbose=False,
            kernel_factory=_FlightKernel,
        )
        assert s["plot_saved"] is True
        assert s["plot_error"] is None

    def test_save_plot_bad_path(self, tmp_path):
        bad = str(tmp_path / "nope" / "flight.png")
        s = run_flight_sim(
            shot_duration=10, seed=42,
            save_plot=True, output_path=bad, verbose=False,
            kernel_factory=_FlightKernel,
        )
        assert s["plot_saved"] is False
        assert s["plot_error"] is not None

    def test_visualize_flight_mpl_unavailable(self, monkeypatch, tmp_path):
        """Cover the early-return path when matplotlib import fails."""
        from scpn_control.control.tokamak_flight_sim import IsoFluxController
        ctrl = IsoFluxController.__new__(IsoFluxController)
        ctrl.verbose = False
        ctrl.kernel = _FlightKernel("dummy")
        ctrl.history = {
            "t": [0, 1], "Ip": [5.0, 5.1],
            "R_axis": [6.2, 6.2], "Z_axis": [0.0, 0.0],
            "X_point": [(0.0, 0.0), (0.0, 0.0)],
            "ctrl_R_cmd": [0.0, 0.0], "ctrl_R_applied": [0.0, 0.0],
            "ctrl_Z_cmd": [0.0, 0.0], "ctrl_Z_applied": [0.0, 0.0],
            "beta_cmd": [1.0, 1.0], "beta_applied": [1.0, 1.0],
        }
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "matplotlib.pyplot":
                raise ImportError("mocked mpl missing")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        ok, err = ctrl.visualize_flight(output_path=str(tmp_path / "nope.png"))
        assert ok is False
        assert "mocked mpl missing" in err


# ── advanced_soc_fusion_learning: _plot_learning + save_plot ─────────

from scpn_control.control.advanced_soc_fusion_learning import (
    _plot_learning,
    run_advanced_learning_sim,
)


class TestAdvancedSOCVisualization:
    def test_plot_learning_direct(self, tmp_path):
        rng = np.random.default_rng(0)
        n = 200
        ok, err = _plot_learning(
            rng.random(n), rng.random(n), rng.random(n), rng.random(n),
            rng.random((10, 10, 3)),
            str(tmp_path / "learning.png"),
        )
        assert ok is True
        assert err is None

    def test_plot_learning_bad_path(self, tmp_path):
        rng = np.random.default_rng(0)
        n = 50
        ok, err = _plot_learning(
            rng.random(n), rng.random(n), rng.random(n), rng.random(n),
            rng.random((10, 10, 3)),
            str(tmp_path / "nope" / "learning.png"),
        )
        assert ok is False
        assert err is not None

    def test_run_with_save_plot(self, tmp_path):
        out = str(tmp_path / "soc.png")
        s = run_advanced_learning_sim(
            size=8, time_steps=20, seed=0,
            save_plot=True, output_path=out, verbose=False,
        )
        assert s["plot_saved"] is True
        assert s["plot_error"] is None


# ── spi_mitigation: save_plot=True ───────────────────────────────────

from scpn_control.control.spi_mitigation import run_spi_mitigation


class TestSPIVisualization:
    def test_save_plot_true(self, tmp_path):
        out = str(tmp_path / "spi.png")
        s = run_spi_mitigation(
            save_plot=True, output_path=out, verbose=False,
        )
        assert s["plot_saved"] is True
        assert s["plot_error"] is None

    def test_save_plot_bad_path(self, tmp_path):
        bad = str(tmp_path / "nope" / "spi.png")
        s = run_spi_mitigation(
            save_plot=True, output_path=bad, verbose=True,
        )
        assert s["plot_saved"] is False
        assert s["plot_error"] is not None


# ── fusion_optimal_control: plot_telemetry + save_plot ───────────────

from scpn_control.control.fusion_optimal_control import (
    OptimalController,
    run_optimal_control,
)


class _OptKernel:
    def __init__(self, _cfg: str) -> None:
        self.cfg = {
            "physics": {"plasma_current_target": 8.0},
            "coils": [
                {"name": f"PF{i+1}", "r": 5.7 + 0.2 * i, "z": -0.25 + 0.125 * i, "current": 0.0}
                for i in range(4)
            ],
        }
        self.R = np.linspace(5.8, 6.3, 21)
        self.Z = np.linspace(-0.4, 0.4, 21)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros((len(self.Z), len(self.R)), dtype=np.float64)
        self.J_phi = np.zeros_like(self.Psi)
        self.solve_equilibrium()

    def solve_equilibrium(self) -> None:
        i = [float(c["current"]) for c in self.cfg["coils"]]
        center_r = 6.02 + 0.09 * np.tanh(sum(i) / 8.0)
        center_z = 0.0
        ir = int(np.argmin(np.abs(self.R - center_r)))
        iz = int(np.argmin(np.abs(self.Z - center_z)))
        self.Psi.fill(-1.0)
        self.Psi[iz, ir] = 1.0 + 0.001 * float(
            self.cfg["physics"]["plasma_current_target"]
        )
        self.J_phi = np.exp(
            -((self.RR - center_r) ** 2 + ((self.ZZ - center_z) / 1.5) ** 2)
        )


class TestOptimalControlVisualization:
    def test_plot_telemetry_direct(self, tmp_path):
        oc = OptimalController(
            "dummy.json", kernel_factory=_OptKernel, verbose=False,
        )
        oc.identify_system()
        oc.run_optimal_shot(shot_steps=5, save_plot=False)
        ok, err = oc.plot_telemetry(output_path=str(tmp_path / "opt.png"))
        assert ok is True
        assert err is None

    def test_run_with_save_plot(self, tmp_path):
        out = str(tmp_path / "opt_full.png")
        s = run_optimal_control(
            shot_steps=5, seed=0,
            save_plot=True, output_path=out, verbose=False,
            kernel_factory=_OptKernel,
        )
        assert s["plot_saved"] is True
        assert s["plot_error"] is None

    def test_plot_telemetry_bad_path(self, tmp_path):
        oc = OptimalController(
            "dummy.json", kernel_factory=_OptKernel, verbose=False,
        )
        oc.identify_system()
        oc.run_optimal_shot(shot_steps=5, save_plot=False)
        ok, err = oc.plot_telemetry(output_path=str(tmp_path / "nope" / "x.png"))
        assert ok is False
        assert err is not None


# ── fusion_sota_mpc: _plot_telemetry + save_plot ─────────────────────

from scpn_control.control.fusion_sota_mpc import run_sota_simulation


class _MPCKernel:
    def __init__(self, _cfg: str) -> None:
        self.cfg = {
            "physics": {"plasma_current_target": 7.0},
            "coils": [
                {"name": f"PF{i+1}", "current": 0.0} for i in range(4)
            ],
        }
        self.R = np.linspace(5.8, 6.3, 25)
        self.Z = np.linspace(-0.4, 0.4, 25)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros((len(self.Z), len(self.R)), dtype=np.float64)
        self._xp = (5.0, -3.5)
        self.solve_equilibrium()

    def solve_equilibrium(self) -> None:
        i = [float(c["current"]) for c in self.cfg["coils"]]
        center_r = 6.0 + 0.08 * np.tanh(sum(i) / 7.0)
        ir = int(np.argmin(np.abs(self.R - center_r)))
        iz = len(self.Z) // 2
        self.Psi.fill(-1.0)
        self.Psi[iz, ir] = 1.0 + 0.001 * float(
            self.cfg["physics"]["plasma_current_target"]
        )

    def find_x_point(self, _psi):
        return self._xp, 0.0


class TestSOTAMPCVisualization:
    def test_save_plot_true(self, tmp_path):
        out = str(tmp_path / "mpc.png")
        s = run_sota_simulation(
            config_file="dummy.json",
            shot_length=10, prediction_horizon=3,
            save_plot=True, output_path=out, verbose=False,
            kernel_factory=_MPCKernel,
        )
        assert s["plot_saved"] is True
        assert s["plot_error"] is None

    def test_save_plot_bad_path(self, tmp_path):
        bad = str(tmp_path / "nope" / "mpc.png")
        s = run_sota_simulation(
            config_file="dummy.json",
            shot_length=10, prediction_horizon=3,
            save_plot=True, output_path=bad, verbose=False,
            kernel_factory=_MPCKernel,
        )
        assert s["plot_saved"] is False
        assert s["plot_error"] is not None


# ── director_interface: visualize + save_plot ────────────────────────

from scpn_control.control.director_interface import DirectorInterface
import scpn_control.control.neuro_cybernetic_controller as nc_mod


class _DirKernel:
    def __init__(self, _cfg: str) -> None:
        self.cfg = {
            "physics": {"plasma_current_target": 5.0},
            "coils": [{"current": 0.0} for _ in range(5)],
        }
        self.R = np.linspace(5.9, 6.5, 25)
        self.Z = np.linspace(-0.3, 0.3, 25)
        RR, ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = 1.0 - ((RR - 6.2) ** 2 + ((ZZ - 0.0) / 1.4) ** 2)

    def solve_equilibrium(self) -> None:
        pass


class _DirNCC:
    """Minimal NeuroCyberneticController stand-in for DirectorInterface tests."""
    def __init__(self, _cfg: str) -> None:
        from scpn_control.control.neuro_cybernetic_controller import SpikingControllerPool
        self.kernel = _DirKernel(_cfg)
        self.brain_R = SpikingControllerPool(n_neurons=5, seed=0)
        self.brain_Z = SpikingControllerPool(n_neurons=5, seed=1)

    def initialize_brains(self, use_quantum: bool = False) -> None:
        pass


class TestDirectorInterfaceVisualization:
    @pytest.fixture(autouse=True)
    def _no_neurocore(self, monkeypatch):
        monkeypatch.setattr(nc_mod, "SC_NEUROCORE_AVAILABLE", False)

    def test_save_plot_true(self, tmp_path):
        out = str(tmp_path / "dir.png")
        di = DirectorInterface(
            "dummy.json",
            controller_factory=_DirNCC,
        )
        s = di.run_directed_mission(
            duration=10, save_plot=True, output_path=out, verbose=False,
        )
        assert s["plot_saved"] is True
        assert s["plot_error"] is None

    def test_save_plot_bad_path(self, tmp_path):
        bad = str(tmp_path / "nope" / "dir.png")
        di = DirectorInterface(
            "dummy.json",
            controller_factory=_DirNCC,
        )
        s = di.run_directed_mission(
            duration=10, save_plot=True, output_path=bad, verbose=False,
        )
        assert s["plot_saved"] is False
        assert s["plot_error"] is not None

    def test_visualize_returns_path(self, tmp_path):
        out = str(tmp_path / "viz.png")
        di = DirectorInterface(
            "dummy.json",
            controller_factory=_DirNCC,
        )
        di.run_directed_mission(duration=5, save_plot=False, verbose=False)
        result = di.visualize(output_path=out)
        assert result == out

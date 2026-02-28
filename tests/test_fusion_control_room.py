# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Fusion Control Room Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Deterministic smoke tests for fusion_control_room runtime entry point."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.control.fusion_control_room import TokamakPhysicsEngine, run_control_room


class _DummyKernel:
    """Small deterministic kernel stand-in exposing Psi/R/Z fields."""

    def __init__(self, _config_path: str) -> None:
        self.cfg = {"coils": [{"current": 0.0} for _ in range(5)]}
        self.R = np.linspace(1.0, 5.0, 40)
        self.Z = np.linspace(-3.0, 3.0, 40)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros((40, 40), dtype=np.float64)
        self._ticks = 0
        self.solve_equilibrium()

    def solve_equilibrium(self) -> None:
        self._ticks += 1
        radial_drive = float(self.cfg["coils"][2]["current"])
        vertical_drive = float(self.cfg["coils"][4]["current"]) - float(
            self.cfg["coils"][0]["current"]
        )
        center_r = 3.0 + 0.2 * np.tanh(radial_drive / 25.0)
        center_z = 0.0 + 0.35 * np.tanh(vertical_drive / 25.0)
        self.Psi = (self.RR - center_r) ** 2 + ((self.ZZ - center_z) / 1.7) ** 2


def test_run_control_room_returns_finite_summary_without_outputs() -> None:
    summary = run_control_room(
        sim_duration=18,
        seed=123,
        save_animation=False,
        save_report=False,
        verbose=False,
    )
    for key in (
        "seed",
        "steps",
        "psi_source",
        "final_z",
        "mean_abs_z",
        "max_abs_z",
        "mean_top_action",
        "mean_bottom_action",
        "animation_saved",
        "report_saved",
    ):
        assert key in summary
    assert summary["steps"] == 18
    assert summary["psi_source"] == "analytic"
    assert summary["animation_saved"] is False
    assert summary["report_saved"] is False
    assert np.isfinite(summary["final_z"])
    assert np.isfinite(summary["mean_abs_z"])
    assert np.isfinite(summary["max_abs_z"])


def test_run_control_room_is_deterministic_for_fixed_seed() -> None:
    kwargs = dict(
        sim_duration=14,
        seed=77,
        save_animation=False,
        save_report=False,
        verbose=False,
    )
    a = run_control_room(**kwargs)
    b = run_control_room(**kwargs)
    assert a["final_z"] == b["final_z"]
    assert a["mean_abs_z"] == b["mean_abs_z"]
    assert a["max_abs_z"] == b["max_abs_z"]
    assert a["mean_top_action"] == b["mean_top_action"]
    assert a["mean_bottom_action"] == b["mean_bottom_action"]


def test_run_control_room_supports_kernel_backed_psi_source() -> None:
    summary = run_control_room(
        sim_duration=10,
        seed=9,
        save_animation=False,
        save_report=False,
        verbose=False,
        kernel_factory=_DummyKernel,
        config_file="dummy.json",
    )
    assert summary["psi_source"] == "kernel"
    assert summary["kernel_error"] is None


def test_run_control_room_rejects_invalid_sim_duration() -> None:
    with pytest.raises(ValueError, match="sim_duration"):
        run_control_room(
            sim_duration=0,
            seed=1,
            save_animation=False,
            save_report=False,
            verbose=False,
        )


def test_tokamak_physics_engine_rejects_invalid_size() -> None:
    with pytest.raises(ValueError, match="size"):
        TokamakPhysicsEngine(size=8, seed=1)


# ── TokamakPhysicsEngine ──────────────────────────────────────────

class TestTokamakPhysicsEngine:
    def test_grid_shape(self):
        eng = TokamakPhysicsEngine(size=20, seed=0)
        assert eng.RR.shape == (20, 20)
        assert eng.ZZ.shape == (20, 20)

    def test_initial_state(self):
        eng = TokamakPhysicsEngine(size=16, seed=0)
        assert eng.z_pos == 0.0
        assert eng.v_drift == 0.0

    def test_solve_flux_surfaces_analytic(self):
        eng = TokamakPhysicsEngine(size=20, seed=0)
        density, psi = eng.solve_flux_surfaces()
        assert density.shape == (20, 20)
        assert psi.shape == (20, 20)
        assert np.all(density >= 0.0)

    def test_solve_flux_surfaces_with_kernel(self):
        kernel = _DummyKernel("test")
        eng = TokamakPhysicsEngine(size=40, seed=0, kernel=kernel)
        density, psi = eng.solve_flux_surfaces()
        assert density.shape == (40, 40)
        assert np.all(density >= 0.0)

    def test_kernel_psi_rejects_small_array(self):
        class _TinyKernel:
            Psi = np.zeros((4, 4))
        eng = TokamakPhysicsEngine(size=16, seed=0, kernel=_TinyKernel())
        density, psi = eng.solve_flux_surfaces()
        assert density.shape == (16, 16)

    def test_kernel_psi_rejects_1d(self):
        class _FlatKernel:
            Psi = np.zeros(100)
        eng = TokamakPhysicsEngine(size=16, seed=0, kernel=_FlatKernel())
        density, _ = eng.solve_flux_surfaces()
        assert density.shape == (16, 16)

    def test_step_dynamics_returns_float(self):
        eng = TokamakPhysicsEngine(size=16, seed=0)
        z = eng.step_dynamics(0.5, 0.3)
        assert isinstance(z, float)

    def test_step_dynamics_deterministic(self):
        e1 = TokamakPhysicsEngine(size=16, seed=99)
        e2 = TokamakPhysicsEngine(size=16, seed=99)
        z1 = [e1.step_dynamics(0.1, 0.2) for _ in range(10)]
        z2 = [e2.step_dynamics(0.1, 0.2) for _ in range(10)]
        np.testing.assert_array_equal(z1, z2)

    def test_density_nonzero_in_core(self):
        eng = TokamakPhysicsEngine(size=40, seed=0)
        density, _ = eng.solve_flux_surfaces()
        center = density[20, 20]
        assert center > 0.0


# ── DiagnosticSystem ───────────────────────────────────────────────

class TestDiagnosticSystem:
    def test_measure_near_true(self):
        from scpn_control.control.fusion_control_room import DiagnosticSystem
        rng = np.random.default_rng(0)
        diag = DiagnosticSystem(rng)
        measurements = [diag.measure_position(1.0) for _ in range(200)]
        assert abs(np.mean(measurements) - 1.0) < 0.05

    def test_measure_returns_float(self):
        from scpn_control.control.fusion_control_room import DiagnosticSystem
        rng = np.random.default_rng(0)
        diag = DiagnosticSystem(rng)
        assert isinstance(diag.measure_position(0.0), float)


# ── NeuralController ──────────────────────────────────────────────

class TestNeuralController:
    def test_returns_pair(self):
        from scpn_control.control.fusion_control_room import NeuralController
        ctrl = NeuralController()
        top, bot = ctrl.compute_action(0.5)
        assert isinstance(top, float)
        assert isinstance(bot, float)

    def test_actions_bounded_by_one(self):
        from scpn_control.control.fusion_control_room import NeuralController
        ctrl = NeuralController()
        for z in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            top, bot = ctrl.compute_action(z)
            assert 0.0 <= top <= 1.0
            assert 0.0 <= bot <= 1.0

    def test_positive_z_activates_top_coil(self):
        from scpn_control.control.fusion_control_room import NeuralController
        ctrl = NeuralController()
        top, bot = ctrl.compute_action(1.0)
        assert top > 0.0
        assert bot == 0.0

    def test_negative_z_activates_bottom_coil(self):
        from scpn_control.control.fusion_control_room import NeuralController
        ctrl = NeuralController()
        top, bot = ctrl.compute_action(-1.0)
        assert top == 0.0
        assert bot > 0.0


# ── run_control_room extended ──────────────────────────────────────

class TestRunControlRoomExtended:
    def test_kernel_factory_error_falls_back(self):
        def _bad_factory(_cfg):
            raise RuntimeError("boom")
        summary = run_control_room(
            sim_duration=5, seed=0,
            save_animation=False, save_report=False, verbose=False,
            kernel_factory=_bad_factory, config_file="x.json",
        )
        assert summary["psi_source"] == "analytic"
        assert summary["kernel_error"] == "boom"

    def test_mean_abs_z_nonnegative(self):
        summary = run_control_room(
            sim_duration=20, seed=0,
            save_animation=False, save_report=False, verbose=False,
        )
        assert summary["mean_abs_z"] >= 0.0

    def test_actions_nonnegative(self):
        summary = run_control_room(
            sim_duration=15, seed=0,
            save_animation=False, save_report=False, verbose=False,
        )
        assert summary["mean_top_action"] >= 0.0
        assert summary["mean_bottom_action"] >= 0.0

    def test_kernel_backed_run(self):
        summary = run_control_room(
            sim_duration=8, seed=0,
            save_animation=False, save_report=False, verbose=False,
            kernel_factory=_DummyKernel, config_file="test.json",
        )
        assert summary["psi_source"] == "kernel"
        assert summary["kernel_error"] is None
        assert np.isfinite(summary["final_z"])


# ── Kernel coil/equilibrium edge cases ───────────────────────────

class _KernelFewCoils:
    """Kernel with only 3 coils: coil update branch at line 345 is skipped."""

    def __init__(self, _cfg: str) -> None:
        self.cfg = {"coils": [{"current": 0.0} for _ in range(3)]}
        self.R = np.linspace(1.0, 5.0, 20)
        self.Z = np.linspace(-3.0, 3.0, 20)
        RR, ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = 1.0 - ((RR - 3.0) ** 2 + (ZZ / 1.7) ** 2)

    def solve_equilibrium(self) -> None:
        pass


class _KernelBadEquilibrium:
    """Kernel whose solve_equilibrium raises to cover the warning path."""

    def __init__(self, _cfg: str) -> None:
        self.cfg = {"coils": [{"current": 0.0} for _ in range(5)]}
        self.R = np.linspace(1.0, 5.0, 20)
        self.Z = np.linspace(-3.0, 3.0, 20)
        RR, ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = 1.0 - ((RR - 3.0) ** 2 + (ZZ / 1.7) ** 2)
        self._calls = 0

    def solve_equilibrium(self) -> None:
        self._calls += 1
        if self._calls > 1:
            raise RuntimeError("singular matrix")


class TestKernelEdgeCases:
    def test_few_coils_skips_update(self):
        summary = run_control_room(
            sim_duration=5, seed=0,
            save_animation=False, save_report=False, verbose=False,
            kernel_factory=_KernelFewCoils, config_file="x.json",
        )
        assert summary["psi_source"] == "kernel"
        assert np.isfinite(summary["final_z"])

    def test_bad_equilibrium_logs_warning(self):
        summary = run_control_room(
            sim_duration=5, seed=0,
            save_animation=False, save_report=False, verbose=False,
            kernel_factory=_KernelBadEquilibrium, config_file="x.json",
        )
        assert summary["psi_source"] == "kernel"
        assert np.isfinite(summary["final_z"])

    def test_verbose_kernel_path(self, capsys):
        run_control_room(
            sim_duration=3, seed=0,
            save_animation=False, save_report=False, verbose=True,
            kernel_factory=_DummyKernel, config_file="x.json",
        )
        out = capsys.readouterr().out
        assert "kernel" in out.lower()

    def test_verbose_analytic_fallback_on_error(self, capsys):
        def _bad_factory(_cfg):
            raise RuntimeError("nope")
        run_control_room(
            sim_duration=3, seed=0,
            save_animation=False, save_report=False, verbose=True,
            kernel_factory=_bad_factory, config_file="x.json",
        )
        out = capsys.readouterr().out
        assert "nope" in out

    def test_auto_kernel_factory_from_config_file(self):
        """When kernel_factory=None but config_file set, FusionKernel is used automatically."""
        import scpn_control.control.fusion_control_room as fcr_mod

        original_fk = fcr_mod.FusionKernel
        fcr_mod.FusionKernel = _DummyKernel
        try:
            summary = run_control_room(
                sim_duration=3, seed=0,
                save_animation=False, save_report=False, verbose=False,
                kernel_factory=None, config_file="x.json",
            )
            assert summary["psi_source"] == "kernel"
        finally:
            fcr_mod.FusionKernel = original_fk

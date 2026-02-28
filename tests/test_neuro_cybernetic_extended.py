# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Extended Neuro-Cybernetic Controller Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Tests for _resolve_fusion_kernel, save_plot error path, and edge cases."""
from __future__ import annotations

import numpy as np
import pytest

import scpn_control.control.neuro_cybernetic_controller as nc_mod
from scpn_control.control.neuro_cybernetic_controller import (
    NeuroCyberneticController,
    SpikingControllerPool,
    _resolve_fusion_kernel,
    run_neuro_cybernetic_control,
)


class _DummyKernel:
    def __init__(self, _config_file: str) -> None:
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


@pytest.fixture(autouse=True)
def _force_numpy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nc_mod, "SC_NEUROCORE_AVAILABLE", False)


class TestResolveFusionKernel:
    def test_resolves_successfully(self):
        cls = _resolve_fusion_kernel()
        assert cls is not None

    def test_returns_callable(self):
        cls = _resolve_fusion_kernel()
        assert callable(cls)


class TestSpikingPoolEdgeCases:
    def test_single_neuron_pool(self):
        pool = SpikingControllerPool(n_neurons=1, seed=42)
        output = pool.step(1.0)
        assert isinstance(output, float)

    def test_zero_error_no_drift(self):
        pool = SpikingControllerPool(n_neurons=20, seed=42, noise_std=0.0)
        outputs = [pool.step(0.0) for _ in range(50)]
        assert all(isinstance(o, float) for o in outputs)

    def test_large_error_produces_nonzero(self):
        pool = SpikingControllerPool(n_neurons=20, seed=42, tau_window=5)
        for _ in range(20):
            pool.step(10.0)
        assert pool.last_rate_pos > 0.0

    def test_negative_error_drives_neg_population(self):
        pool = SpikingControllerPool(n_neurons=20, seed=42, tau_window=5)
        for _ in range(20):
            pool.step(-10.0)
        assert pool.last_rate_neg > 0.0


class TestNeuroCyberneticControllerSavePlotError:
    def test_save_plot_error_captured_in_summary(self, monkeypatch):
        nc = NeuroCyberneticController(
            "dummy.json", seed=42, shot_duration=5, kernel_factory=_DummyKernel,
        )
        nc.initialize_brains(use_quantum=False)

        def _bad_visualize(*args, **kwargs):
            raise RuntimeError("matplotlib not available")

        monkeypatch.setattr(nc, "visualize", _bad_visualize)
        summary = nc._execute_simulation(
            "Test", mode="classical", save_plot=True, verbose=False, output_path=None,
        )
        assert summary["plot_saved"] is False
        assert "matplotlib not available" in summary["plot_error"]

    def test_save_plot_error_with_verbose(self, monkeypatch, capsys):
        nc = NeuroCyberneticController(
            "dummy.json", seed=42, shot_duration=3, kernel_factory=_DummyKernel,
        )
        nc.initialize_brains(use_quantum=False)

        def _bad_visualize(*args, **kwargs):
            raise ValueError("bad shape")

        monkeypatch.setattr(nc, "visualize", _bad_visualize)
        nc._execute_simulation(
            "Test", mode="classical", save_plot=True, verbose=True, output_path=None,
        )
        out = capsys.readouterr().out
        assert "Plot export skipped" in out


class TestRunFunctionOutputPath:
    def test_output_path_kwarg_passed(self):
        summary = run_neuro_cybernetic_control(
            config_file="dummy.json",
            shot_duration=5,
            seed=42,
            save_plot=False,
            verbose=False,
            output_path="/tmp/test.png",
            kernel_factory=_DummyKernel,
        )
        assert summary["plot_saved"] is False

    def test_quantum_mode_backend_name(self):
        summary = run_neuro_cybernetic_control(
            config_file="dummy.json",
            shot_duration=5,
            seed=42,
            quantum=True,
            save_plot=False,
            verbose=False,
            kernel_factory=_DummyKernel,
        )
        assert summary["backend_r"] == "numpy_lif"
        assert summary["backend_z"] == "numpy_lif"
        assert summary["mode"] == "quantum"


class TestControllerResetHistory:
    def test_reset_history_called_per_shot(self):
        nc = NeuroCyberneticController(
            "dummy.json", seed=42, shot_duration=10, kernel_factory=_DummyKernel,
        )
        nc.run_shot(save_plot=False, verbose=False)
        first_len = len(nc.history["t"])
        nc.run_shot(save_plot=False, verbose=False)
        assert len(nc.history["t"]) == first_len

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Neuro Cybernetic Controller Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Deterministic tests for reduced spiking-controller pool behavior."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_control.control.neuro_cybernetic_controller as controller_mod
from scpn_control.control.neuro_cybernetic_controller import (
    SpikingControllerPool,
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
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros((25, 25), dtype=np.float64)
        self.solve_equilibrium()

    def solve_equilibrium(self) -> None:
        radial_drive = float(self.cfg["coils"][2]["current"])
        vertical_drive = float(self.cfg["coils"][4]["current"]) - float(
            self.cfg["coils"][0]["current"]
        )
        center_r = 6.2 + 0.07 * np.tanh(radial_drive / 20.0)
        center_z = 0.0 + 0.05 * np.tanh(vertical_drive / 20.0)
        self.Psi = 1.0 - (
            (self.RR - center_r) ** 2 + ((self.ZZ - center_z) / 1.4) ** 2
        )


@pytest.fixture(autouse=True)
def _force_numpy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    # CI py3.11 lane does not bootstrap sc-neurocore; force same path locally.
    monkeypatch.setattr(controller_mod, "SC_NEUROCORE_AVAILABLE", False)


def test_spiking_pool_is_deterministic_for_same_seed() -> None:
    kwargs = dict(
        n_neurons=24,
        gain=2.0,
        tau_window=8,
        seed=19,
        use_quantum=False,
    )
    p1 = SpikingControllerPool(**kwargs)
    p2 = SpikingControllerPool(**kwargs)
    o1 = np.asarray([p1.step(0.15) for _ in range(24)], dtype=np.float64)
    o2 = np.asarray([p2.step(0.15) for _ in range(24)], dtype=np.float64)
    np.testing.assert_allclose(o1, o2, atol=0.0, rtol=0.0)


def test_spiking_pool_push_pull_sign_response() -> None:
    pos_pool = SpikingControllerPool(
        n_neurons=20,
        gain=3.0,
        tau_window=6,
        seed=31,
        use_quantum=False,
    )
    neg_pool = SpikingControllerPool(
        n_neurons=20,
        gain=3.0,
        tau_window=6,
        seed=31,
        use_quantum=False,
    )

    pos = [pos_pool.step(0.2) for _ in range(32)]
    neg = [neg_pool.step(-0.2) for _ in range(32)]
    assert float(np.mean(pos[-8:])) > 0.0
    assert float(np.mean(neg[-8:])) < 0.0


def test_spiking_pool_exposes_backend_name() -> None:
    pool = SpikingControllerPool(n_neurons=8, gain=1.0, tau_window=4, seed=11)
    assert pool.backend == "numpy_lif"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_neurons": 0}, "n_neurons"),
        ({"tau_window": 0}, "tau_window"),
        ({"gain": float("nan")}, "gain"),
        ({"dt_s": 0.0}, "dt_s"),
        ({"tau_mem_s": 0.0}, "tau_mem_s"),
        ({"noise_std": -1.0e-3}, "noise_std"),
    ],
)
def test_spiking_pool_rejects_invalid_constructor_inputs(
    kwargs: dict[str, float | int], match: str
) -> None:
    params: dict[str, float | int | bool] = {
        "n_neurons": 8,
        "gain": 1.0,
        "tau_window": 4,
        "seed": 11,
        "use_quantum": False,
    }
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        SpikingControllerPool(**params)


def test_run_neuro_cybernetic_control_returns_finite_summary_without_plot() -> None:
    summary = run_neuro_cybernetic_control(
        config_file="dummy.json",
        shot_duration=14,
        seed=101,
        quantum=False,
        save_plot=False,
        verbose=False,
        kernel_factory=_DummyKernel,
    )
    for key in (
        "seed",
        "steps",
        "mode",
        "backend_r",
        "backend_z",
        "final_r",
        "final_z",
        "mean_abs_err_r",
        "mean_abs_err_z",
        "max_abs_control_r",
        "max_abs_control_z",
        "mean_spike_imbalance",
        "plot_saved",
    ):
        assert key in summary
    assert summary["seed"] == 101
    assert summary["steps"] == 14
    assert summary["mode"] == "classical"
    assert summary["backend_r"] == "numpy_lif"
    assert summary["backend_z"] == "numpy_lif"
    assert summary["plot_saved"] is False
    assert np.isfinite(summary["final_r"])
    assert np.isfinite(summary["final_z"])


def test_run_neuro_cybernetic_control_is_deterministic_for_seed() -> None:
    kwargs = dict(
        config_file="dummy.json",
        shot_duration=12,
        seed=77,
        quantum=False,
        save_plot=False,
        verbose=False,
        kernel_factory=_DummyKernel,
    )
    a = run_neuro_cybernetic_control(**kwargs)
    b = run_neuro_cybernetic_control(**kwargs)
    for key in (
        "final_r",
        "final_z",
        "mean_abs_err_r",
        "mean_abs_err_z",
        "max_abs_control_r",
        "max_abs_control_z",
        "mean_spike_imbalance",
    ):
        assert a[key] == pytest.approx(b[key], rel=0.0, abs=0.0)


def test_run_neuro_cybernetic_control_rejects_nonpositive_duration() -> None:
    with pytest.raises(ValueError, match="shot_duration"):
        run_neuro_cybernetic_control(
            config_file="dummy.json",
            shot_duration=0,
            save_plot=False,
            verbose=False,
            kernel_factory=_DummyKernel,
        )


def test_run_neuro_cybernetic_control_quantum_mode() -> None:
    summary = run_neuro_cybernetic_control(
        config_file="dummy.json",
        shot_duration=10,
        seed=42,
        quantum=True,
        save_plot=False,
        verbose=False,
        kernel_factory=_DummyKernel,
    )
    assert summary["mode"] == "quantum"
    assert summary["steps"] == 10


def test_spiking_pool_numpy_population_fires_above_threshold() -> None:
    pool = SpikingControllerPool(n_neurons=20, seed=42)
    spikes = 0
    for _ in range(100):
        out = pool.step(2.0)
        if out > 0.0:
            spikes += 1
    assert spikes > 0


def test_spiking_pool_no_numpy_fallback_raises() -> None:
    with pytest.raises(RuntimeError, match="allow_numpy_fallback=False"):
        SpikingControllerPool(
            n_neurons=8, allow_numpy_fallback=False,
        )


def test_spiking_pool_last_rates_nonnegative() -> None:
    pool = SpikingControllerPool(n_neurons=10, seed=42)
    pool.step(0.5)
    assert pool.last_rate_pos >= 0.0
    assert pool.last_rate_neg >= 0.0


def test_controller_history_populated() -> None:
    from scpn_control.control.neuro_cybernetic_controller import NeuroCyberneticController
    nc = NeuroCyberneticController(
        "dummy.json", seed=42, shot_duration=20, kernel_factory=_DummyKernel,
    )
    nc.run_shot(save_plot=False, verbose=False)
    assert len(nc.history["t"]) == 20
    assert len(nc.history["Err_R"]) == 20
    assert len(nc.history["Control_R"]) == 20
    assert len(nc.history["Spike_Rates"]) == 20


def test_controller_coils_updated() -> None:
    """An offset kernel should produce nonzero errors → coil adjustments."""
    from scpn_control.control.neuro_cybernetic_controller import NeuroCyberneticController

    class _OffsetKernel:
        def __init__(self, _config_file: str) -> None:
            self.cfg = {
                "physics": {"plasma_current_target": 5.0},
                "coils": [{"current": 0.0} for _ in range(5)],
            }
            self.R = np.linspace(5.0, 7.0, 25)
            self.Z = np.linspace(-1.0, 1.0, 25)
            RR, ZZ = np.meshgrid(self.R, self.Z)
            # Peak at (5.5, 0.3) — offset from TARGET_R=6.2, TARGET_Z=0.0
            self.Psi = 1.0 - ((RR - 5.5) ** 2 + (ZZ - 0.3) ** 2)

        def solve_equilibrium(self) -> None:
            pass

    nc = NeuroCyberneticController(
        "dummy.json", seed=42, shot_duration=30, kernel_factory=_OffsetKernel,
    )
    nc.run_shot(save_plot=False, verbose=False)
    any_nonzero = any(
        abs(float(c["current"])) > 0.0 for c in nc.kernel.cfg["coils"]
    )
    assert any_nonzero


def test_spiking_pool_rejects_inf_dt() -> None:
    with pytest.raises(ValueError, match="dt_s must be finite"):
        SpikingControllerPool(dt_s=float("inf"))


def test_spiking_pool_rejects_nan_noise() -> None:
    with pytest.raises(ValueError, match="noise_std must be finite"):
        SpikingControllerPool(noise_std=float("nan"))

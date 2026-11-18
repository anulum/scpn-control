# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neuro-cybernetic controller tests
"""Deterministic tests for reduced spiking-controller pool behavior."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import scpn_control.control.neuro_cybernetic_controller as controller_mod
from scpn_control.control.neuro_cybernetic_controller import (
    SpikingControllerPool,
    run_neuro_cybernetic_control,
)


class _DummyKernel:
    """Kernel stand-in whose O-point drifts around (6.2, 0) m with the coil currents."""

    def __init__(self, _config_file: str) -> None:
        """Build a 25 x 25 grid around the target with five coils and solve once."""
        self.cfg: dict[str, Any] = {
            "physics": {"plasma_current_target": 5.0},
            "coils": [{"current": 0.0} for _ in range(5)],
        }
        # The grid brackets the drifting peak, so the solved flux has an interior O-point.
        self.R = np.linspace(5.8, 6.6, 25)
        self.Z = np.linspace(-0.3, 0.3, 25)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros((25, 25), dtype=np.float64)
        self.solve_equilibrium()

    def solve_equilibrium(self) -> None:
        """Place a smooth peak whose centre moves with the radial and vertical coil drives."""
        radial_drive = float(self.cfg["coils"][2]["current"])
        vertical_drive = float(self.cfg["coils"][4]["current"]) - float(self.cfg["coils"][0]["current"])
        center_r = 6.2 + 0.07 * np.tanh(radial_drive / 20.0)
        center_z = 0.0 + 0.05 * np.tanh(vertical_drive / 20.0)
        self.Psi = 1.0 - ((self.RR - center_r) ** 2 + ((self.ZZ - center_z) / 1.4) ** 2)


class _FarOffKernel:
    """Kernel stand-in with a fixed interior axis at R = 100 m, far from the target."""

    def __init__(self, _config_file: str) -> None:
        """Build a 25 x 25 grid over R = 90..110 m with five coils and solve once."""
        self.cfg: dict[str, Any] = {
            "physics": {"plasma_current_target": 5.0},
            "coils": [{"current": 0.0} for _ in range(5)],
        }
        # The axis at R = 100 m is interior to this grid and far from TARGET_R (6.2).
        self.R = np.linspace(90.0, 110.0, 25)
        self.Z = np.linspace(-0.3, 0.3, 25)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros((25, 25), dtype=np.float64)
        self.solve_equilibrium()

    def solve_equilibrium(self) -> None:
        """Place the peak at (100, 0) m regardless of the coils."""
        # Keep the plasma axis far from TARGET_R (6.2) for deterministic low-confidence.
        self.Psi = 1.0 - ((self.RR - 100.0) ** 2 + ((self.ZZ - 0.0) / 1.4) ** 2)


class _SafeStatePool:
    """Deterministic pool used to drive safety FSM transitions in tests."""

    def __init__(
        self,
        n_neurons: int = 20,
        gain: float = 1.0,
        tau_window: int = 10,
        use_quantum: bool = False,
        *,
        seed: int = 42,
        allow_numpy_fallback: bool = True,
        allow_legacy_numpy_fallback: bool = True,
        dt_s: float = 1.0e-3,
        tau_mem_s: float = 15.0e-3,
        noise_std: float = 0.02,
        command_value: float = 1.0,
    ) -> None:
        """Accept the pool signature, ignore the SNN settings and return ``command_value`` on every step."""
        del (
            n_neurons,
            gain,
            tau_window,
            use_quantum,
            seed,
            allow_numpy_fallback,
            allow_legacy_numpy_fallback,
            dt_s,
            tau_mem_s,
            noise_std,
        )
        self.backend = "safe-state-fallback"
        self.command_value = float(command_value)
        self.last_rate_pos = 0.0
        self.last_rate_neg = 0.0
        self.confidence = 1.0

    def step(self, error_signal: float) -> float:
        """Lower the confidence with the error magnitude and return the fixed command."""
        self.confidence = max(0.0, 1.0 - abs(float(error_signal)) / 20.0)
        return float(self.command_value)


class _OverflowPool(_SafeStatePool):
    """Safety-test pool whose command overflows to infinity."""

    def step(self, _error_signal: float) -> float:
        """Report full confidence and return an infinite command."""
        self.confidence = 1.0
        return float("inf")


@pytest.fixture(autouse=True)
def _force_numpy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the NumPy LIF lane, as on CI where sc-neurocore is not installed."""
    monkeypatch.setattr(controller_mod, "SC_NEUROCORE_AVAILABLE", False)


def test_spiking_pool_is_deterministic_for_same_seed() -> None:
    """Two pools with one seed emit identical command sequences."""
    kwargs: dict[str, Any] = dict(
        n_neurons=24,
        gain=2.0,
        tau_window=8,
        seed=19,
        use_quantum=False,
        allow_numpy_fallback=True,
        allow_legacy_numpy_fallback=True,
    )
    p1 = SpikingControllerPool(**kwargs)
    p2 = SpikingControllerPool(**kwargs)
    o1 = np.asarray([p1.step(0.15) for _ in range(24)], dtype=np.float64)
    o2 = np.asarray([p2.step(0.15) for _ in range(24)], dtype=np.float64)
    np.testing.assert_allclose(o1, o2, atol=0.0, rtol=0.0)


def test_spiking_pool_push_pull_sign_response() -> None:
    """A positive error yields positive commands and a negative error negative ones."""
    pos_pool = SpikingControllerPool(
        n_neurons=20,
        gain=3.0,
        tau_window=6,
        seed=31,
        use_quantum=False,
        allow_numpy_fallback=True,
        allow_legacy_numpy_fallback=True,
    )
    neg_pool = SpikingControllerPool(
        n_neurons=20,
        gain=3.0,
        tau_window=6,
        seed=31,
        use_quantum=False,
        allow_numpy_fallback=True,
        allow_legacy_numpy_fallback=True,
    )

    pos = [pos_pool.step(0.2) for _ in range(32)]
    neg = [neg_pool.step(-0.2) for _ in range(32)]
    assert float(np.mean(pos[-8:])) > 0.0
    assert float(np.mean(neg[-8:])) < 0.0


def test_spiking_pool_exposes_backend_name() -> None:
    """Without sc-neurocore the pool reports the NumPy LIF backend."""
    pool = SpikingControllerPool(
        n_neurons=8,
        gain=1.0,
        tau_window=4,
        seed=11,
        allow_numpy_fallback=True,
        allow_legacy_numpy_fallback=True,
    )
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
def test_spiking_pool_rejects_invalid_constructor_inputs(kwargs: dict[str, float | int], match: str) -> None:
    """Each out-of-range pool setting is refused by name."""
    params: dict[str, Any] = {
        "n_neurons": 8,
        "gain": 1.0,
        "tau_window": 4,
        "seed": 11,
        "use_quantum": False,
        "allow_numpy_fallback": True,
        "allow_legacy_numpy_fallback": True,
    }
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        SpikingControllerPool(**params)


def test_run_neuro_cybernetic_control_returns_finite_summary_without_plot() -> None:
    """A classical run returns the documented summary with finite final axis values."""
    summary = run_neuro_cybernetic_control(
        config_file="dummy.json",
        shot_duration=14,
        seed=101,
        quantum=False,
        save_plot=False,
        verbose=False,
        allow_numpy_fallback=True,
        allow_legacy_numpy_fallback=True,
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
    """Two runs with one seed give identical summaries."""
    kwargs: dict[str, Any] = dict(
        config_file="dummy.json",
        shot_duration=12,
        seed=77,
        quantum=False,
        save_plot=False,
        verbose=False,
        allow_numpy_fallback=True,
        allow_legacy_numpy_fallback=True,
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
    """A zero shot duration is refused."""
    with pytest.raises(ValueError, match="shot_duration"):
        run_neuro_cybernetic_control(
            config_file="dummy.json",
            shot_duration=0,
            save_plot=False,
            verbose=False,
            allow_numpy_fallback=True,
            allow_legacy_numpy_fallback=True,
            kernel_factory=_DummyKernel,
        )


def test_run_neuro_cybernetic_control_quantum_mode() -> None:
    """The quantum mode flag is reported and the shot runs to length."""
    summary = run_neuro_cybernetic_control(
        config_file="dummy.json",
        shot_duration=10,
        seed=42,
        quantum=True,
        save_plot=False,
        verbose=False,
        allow_numpy_fallback=True,
        allow_legacy_numpy_fallback=True,
        kernel_factory=_DummyKernel,
    )
    assert summary["mode"] == "quantum"
    assert summary["steps"] == 10


def test_spiking_pool_numpy_population_fires_above_threshold() -> None:
    """A strong drive makes the NumPy population fire."""
    pool = SpikingControllerPool(
        n_neurons=20,
        seed=42,
        allow_numpy_fallback=True,
        allow_legacy_numpy_fallback=True,
    )
    spikes = 0
    for _ in range(100):
        out = pool.step(2.0)
        if out > 0.0:
            spikes += 1
    assert spikes > 0


def test_spiking_pool_no_numpy_fallback_raises() -> None:
    """Without sc-neurocore and without the fallback opt-in the pool refuses to start."""
    with pytest.raises(RuntimeError, match="allow_numpy_fallback=False"):
        SpikingControllerPool(
            n_neurons=8,
            allow_numpy_fallback=False,
        )


def test_spiking_pool_legacy_fallback_requires_explicit_opt_in() -> None:
    """The NumPy fallback also needs the legacy opt-in."""
    with pytest.raises(ValueError, match="allow_legacy_numpy_fallback=True"):
        SpikingControllerPool(
            n_neurons=8,
            allow_numpy_fallback=True,
        )


def test_spiking_pool_last_rates_nonnegative() -> None:
    """Recorded population rates are non-negative."""
    pool = SpikingControllerPool(
        n_neurons=10,
        seed=42,
        allow_numpy_fallback=True,
        allow_legacy_numpy_fallback=True,
    )
    pool.step(0.5)
    assert pool.last_rate_pos >= 0.0
    assert pool.last_rate_neg >= 0.0


def test_controller_history_populated() -> None:
    """A shot records one history entry per step."""
    from scpn_control.control.neuro_cybernetic_controller import NeuroCyberneticController

    nc = NeuroCyberneticController(
        "dummy.json",
        seed=42,
        shot_duration=20,
        allow_numpy_fallback=True,
        allow_legacy_numpy_fallback=True,
        kernel_factory=_DummyKernel,
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
        """Kernel stand-in with a fixed axis at (5.5, 0.3) m, off the target."""

        def __init__(self, _config_file: str) -> None:
            """Build a 25 x 25 grid and a fixed off-target peak."""
            self.cfg: dict[str, Any] = {
                "physics": {"plasma_current_target": 5.0},
                "coils": [{"current": 0.0} for _ in range(5)],
            }
            self.R = np.linspace(5.0, 7.0, 25)
            self.Z = np.linspace(-1.0, 1.0, 25)
            RR, ZZ = np.meshgrid(self.R, self.Z)
            # Peak at (5.5, 0.3) — offset from TARGET_R=6.2, TARGET_Z=0.0
            self.Psi = 1.0 - ((RR - 5.5) ** 2 + (ZZ - 0.3) ** 2)

        def solve_equilibrium(self) -> None:
            """Keep the fixed flux."""
            pass

    nc = NeuroCyberneticController(
        "dummy.json",
        seed=42,
        shot_duration=30,
        allow_numpy_fallback=True,
        allow_legacy_numpy_fallback=True,
        kernel_factory=_OffsetKernel,
    )
    nc.run_shot(save_plot=False, verbose=False)
    any_nonzero = any(abs(float(c["current"])) > 0.0 for c in nc.kernel.cfg["coils"])
    assert any_nonzero


def test_safe_shutdown_reduces_commands_with_single_ramp_step_per_tick() -> None:
    """Low confidence ramps the commands down one step per tick, then latches safe shutdown."""
    nc = controller_mod.NeuroCyberneticController(
        "dummy.json",
        seed=42,
        shot_duration=4,
        allow_numpy_fallback=True,
        allow_legacy_numpy_fallback=True,
        kernel_factory=_FarOffKernel,
        safe_shutdown_ramp_steps=3,
    )
    nc.brain_R = cast(SpikingControllerPool, _SafeStatePool(command_value=1.0))
    nc.brain_Z = cast(SpikingControllerPool, _SafeStatePool(command_value=2.0))

    nc._execute_simulation("safe-shutdown test", mode="classical", save_plot=False, verbose=False, output_path=None)

    assert nc.safety_state == "safe_shutdown"
    assert cast(list[str], nc.history["Safety_State"]) == [
        "safe_shutdown_ramp",
        "safe_shutdown_ramp",
        "safe_shutdown",
        "safe_shutdown",
    ]
    assert nc.safety_reason == "low_confidence"
    assert nc._safe_shutdown_counter == 0
    assert nc.safety_trigger_count == 1
    assert nc.history["Control_R"] == [1.0, pytest.approx(2 / 3), pytest.approx(1 / 3), 0.0]
    assert nc.history["Control_Z"] == [2.0, pytest.approx(4 / 3), pytest.approx(2 / 3), 0.0]


def test_safe_shutdown_overflow_trap_overrides_commands_and_tracks_overflow_events() -> None:
    """An overflowing command trips the trap, zeroes the commands and counts each trap."""
    nc = controller_mod.NeuroCyberneticController(
        "dummy.json",
        seed=42,
        shot_duration=3,
        allow_numpy_fallback=True,
        allow_legacy_numpy_fallback=True,
        kernel_factory=_DummyKernel,
        safe_shutdown_ramp_steps=2,
    )
    nc.brain_R = cast(SpikingControllerPool, _OverflowPool())
    nc.brain_Z = cast(SpikingControllerPool, _SafeStatePool(command_value=0.25))

    nc._execute_simulation(
        "safe-shutdown overflow test", mode="classical", save_plot=False, verbose=False, output_path=None
    )

    assert nc.safety_state == "safe_shutdown"
    assert cast(list[str], nc.history["Safety_State"]) == [
        "safe_shutdown_ramp",
        "safe_shutdown",
        "safe_shutdown",
    ]
    assert nc.safety_reason == "overflow_trap"
    assert nc._overflow_trap_count == 3
    assert all(abs(float(v)) == pytest.approx(0.0) for v in nc.history["Control_R"])
    assert all(abs(float(v)) == pytest.approx(0.0) for v in nc.history["Control_Z"])


def test_spiking_pool_rejects_inf_dt() -> None:
    """An infinite time step is refused."""
    with pytest.raises(ValueError, match="dt_s must be finite"):
        SpikingControllerPool(
            dt_s=float("inf"),
            allow_numpy_fallback=True,
            allow_legacy_numpy_fallback=True,
        )


def test_spiking_pool_rejects_nan_noise() -> None:
    """A NaN noise level is refused."""
    with pytest.raises(ValueError, match="noise_std must be finite"):
        SpikingControllerPool(
            noise_std=float("nan"),
            allow_numpy_fallback=True,
            allow_legacy_numpy_fallback=True,
        )


# ── Verbose & coil-padding paths ─────────────────────────────────


def test_run_neuro_cybernetic_control_verbose_output(caplog: pytest.LogCaptureFixture) -> None:
    """A verbose run logs the SNN or plasma banner."""
    with caplog.at_level(logging.INFO, logger="scpn_control.control.neuro_cybernetic_controller"):
        run_neuro_cybernetic_control(
            config_file="dummy.json",
            shot_duration=3,
            seed=42,
            quantum=False,
            save_plot=False,
            verbose=True,
            allow_numpy_fallback=True,
            allow_legacy_numpy_fallback=True,
            kernel_factory=_DummyKernel,
        )
    assert "SNN" in caplog.text or "PLASMA" in caplog.text


class _KernelFewCoils:
    """Kernel with only 2 coils: forces the coil-padding path (line 313-314)."""

    def __init__(self, _config_file: str) -> None:
        """Build a grid around the target with only two coils and solve once."""
        self.cfg: dict[str, Any] = {
            "physics": {"plasma_current_target": 5.0},
            "coils": [{"current": 0.0}, {"current": 0.0}],
        }
        self.R = np.linspace(5.9, 6.5, 25)
        self.Z = np.linspace(-0.3, 0.3, 25)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros((25, 25), dtype=np.float64)
        self.solve_equilibrium()

    def solve_equilibrium(self) -> None:
        """Place the peak at the target (6.2, 0) m."""
        self.Psi = 1.0 - ((self.RR - 6.2) ** 2 + ((self.ZZ - 0.0) / 1.4) ** 2)


def test_controller_pads_coils_when_fewer_than_five() -> None:
    """A kernel with two coils is padded to five before the shot."""
    from scpn_control.control.neuro_cybernetic_controller import NeuroCyberneticController

    nc = NeuroCyberneticController(
        "dummy.json",
        seed=42,
        shot_duration=5,
        allow_numpy_fallback=True,
        allow_legacy_numpy_fallback=True,
        kernel_factory=_KernelFewCoils,
    )
    nc.run_shot(save_plot=False, verbose=False)
    assert len(nc.kernel.cfg["coils"]) >= 5
    assert len(nc.history["t"]) == 5


class _FakeAxis:
    """Axes stand-in that accepts every plotting call."""

    def set_title(self, *_args: object, **_kwargs: object) -> None:
        """Ignore a title."""
        pass

    def plot(self, *_args: object, **_kwargs: object) -> None:
        """Ignore a line."""
        pass

    def axhline(self, *_args: object, **_kwargs: object) -> None:
        """Ignore a reference line."""
        pass

    def set_ylabel(self, *_args: object, **_kwargs: object) -> None:
        """Ignore a y label."""
        pass

    def set_xlabel(self, *_args: object, **_kwargs: object) -> None:
        """Ignore an x label."""
        pass

    def legend(self, *_args: object, **_kwargs: object) -> None:
        """Ignore a legend."""
        pass

    def grid(self, *_args: object, **_kwargs: object) -> None:
        """Ignore a grid."""
        pass


class _FakePyplot:
    """pyplot stand-in that writes a small file instead of a figure."""

    def subplots(self, *_args: object, **_kwargs: object) -> tuple[object, tuple[_FakeAxis, _FakeAxis]]:
        """Return a placeholder figure and two fake axes."""
        return object(), (_FakeAxis(), _FakeAxis())

    def tight_layout(self) -> None:
        """Ignore the layout call."""
        pass

    def savefig(self, filename: str) -> None:
        """Write a few bytes to ``filename``."""
        Path(filename).write_bytes(b"fake neuro-control plot")

    def close(self, _fig: object) -> None:
        """Ignore the close call."""
        pass


def test_controller_visualize_writes_declared_plot_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Visualisation writes the declared plot file."""
    from scpn_control.control.neuro_cybernetic_controller import NeuroCyberneticController

    monkeypatch.setattr(controller_mod, "HAS_MPL", True)
    monkeypatch.setattr(controller_mod, "plt", _FakePyplot())
    nc = NeuroCyberneticController(
        "dummy.json",
        seed=42,
        shot_duration=4,
        allow_numpy_fallback=True,
        allow_legacy_numpy_fallback=True,
        kernel_factory=_DummyKernel,
    )
    nc.run_shot(save_plot=False, verbose=False)
    plot_path = tmp_path / "neuro_control.png"

    returned = nc.visualize("Neuro control regression", output_path=str(plot_path), verbose=False)

    assert returned == str(plot_path)
    assert plot_path.is_file()
    assert plot_path.stat().st_size > 0


def test_controller_visualize_fails_closed_without_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    """Visualisation without matplotlib raises instead of skipping."""
    from scpn_control.control.neuro_cybernetic_controller import NeuroCyberneticController

    monkeypatch.setattr(controller_mod, "HAS_MPL", False)
    monkeypatch.setattr(controller_mod, "plt", None)
    nc = NeuroCyberneticController(
        "dummy.json",
        seed=42,
        shot_duration=4,
        allow_numpy_fallback=True,
        allow_legacy_numpy_fallback=True,
        kernel_factory=_DummyKernel,
    )
    nc.run_shot(save_plot=False, verbose=False)

    with pytest.raises(RuntimeError, match="matplotlib is required"):
        nc.visualize("Neuro control regression", verbose=False)


@pytest.mark.parametrize("bad_threshold", [1.5, -0.1, float("nan"), float("inf")])
def test_controller_rejects_out_of_range_safety_threshold(bad_threshold: float) -> None:
    """A confidence threshold outside [0, 1] or non-finite is refused."""
    with pytest.raises(ValueError, match=r"safety_confidence_threshold must be finite in \[0.0, 1.0\]"):
        controller_mod.NeuroCyberneticController(
            "dummy.json",
            kernel_factory=_DummyKernel,
            safety_confidence_threshold=bad_threshold,
        )


@pytest.mark.parametrize("bad_steps", [0, -3])
def test_controller_rejects_nonpositive_ramp_steps(bad_steps: int) -> None:
    """A shutdown ramp shorter than one step is refused."""
    with pytest.raises(ValueError, match="safe_shutdown_ramp_steps must be >= 1"):
        controller_mod.NeuroCyberneticController(
            "dummy.json",
            kernel_factory=_DummyKernel,
            safe_shutdown_ramp_steps=bad_steps,
        )


def test_controller_overflow_trap_count_property_tracks_traps() -> None:
    """The overflow-trap property reports the internal count."""
    nc = controller_mod.NeuroCyberneticController(
        "dummy.json",
        kernel_factory=_DummyKernel,
        allow_numpy_fallback=True,
        allow_legacy_numpy_fallback=True,
    )
    assert nc.overflow_trap_count == 0
    nc._overflow_trap_count = 4
    assert nc.overflow_trap_count == 4


def test_controller_safety_is_active_reflects_fsm_state() -> None:
    """Entering safe shutdown makes the safety state machine active."""
    nc = controller_mod.NeuroCyberneticController(
        "dummy.json",
        kernel_factory=_DummyKernel,
    )
    assert nc._safety_is_active() is False
    nc._enter_safe_shutdown(reason="low_confidence")
    assert nc._safety_is_active() is True


def test_safe_command_latches_safe_shutdown_when_ramp_counter_exhausted() -> None:
    """An exhausted ramp counter zeroes the command and latches safe shutdown."""
    nc = controller_mod.NeuroCyberneticController(
        "dummy.json",
        kernel_factory=_DummyKernel,
    )
    nc.safety_state = controller_mod._SAFETY_STATE_SHUTDOWN_RAMP
    nc._safe_shutdown_counter = 0

    shaped = nc._safe_command(5.0)

    assert shaped == 0.0
    assert nc.safety_state == controller_mod._SAFETY_STATE_SAFE_SHUTDOWN


def test_run_shot_reports_plot_saved_with_fake_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    """A shot with plotting reports the saved plot and no error."""
    monkeypatch.setattr(controller_mod, "HAS_MPL", True)
    monkeypatch.setattr(controller_mod, "plt", _FakePyplot())
    nc = controller_mod.NeuroCyberneticController(
        "dummy.json",
        seed=42,
        shot_duration=4,
        allow_numpy_fallback=True,
        allow_legacy_numpy_fallback=True,
        kernel_factory=_DummyKernel,
    )

    summary = nc.run_shot(save_plot=True, verbose=False)

    assert summary["plot_saved"] is True
    assert summary["plot_error"] is None


def test_controller_visualize_logs_save_location_when_verbose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verbose visualisation logs where the plot was saved."""
    monkeypatch.setattr(controller_mod, "HAS_MPL", True)
    monkeypatch.setattr(controller_mod, "plt", _FakePyplot())
    nc = controller_mod.NeuroCyberneticController(
        "dummy.json",
        seed=42,
        shot_duration=4,
        allow_numpy_fallback=True,
        allow_legacy_numpy_fallback=True,
        kernel_factory=_DummyKernel,
    )
    nc.run_shot(save_plot=False, verbose=False)
    plot_path = tmp_path / "neuro_control_verbose.png"

    with caplog.at_level(logging.INFO, logger=controller_mod.logger.name):
        returned = nc.visualize("Neuro control regression", output_path=str(plot_path), verbose=True)

    assert returned == str(plot_path)
    assert any("Analysis saved" in record.message for record in caplog.records)


class _EdgeAxisKernel(_DummyKernel):
    """Kernel whose solved flux rises to the outer box edge, so it has no interior O-point."""

    def solve_equilibrium(self) -> None:
        """Replace the drifting peak by a flux that is largest on the outer edge."""
        self.Psi = np.asarray(self.RR, dtype=np.float64).copy()


def test_run_refuses_a_solved_flux_without_an_interior_axis() -> None:
    """The controller does not steer against an edge cell of a degenerate equilibrium."""
    from scpn_control.core.magnetic_axis import MagneticAxisError

    with pytest.raises(MagneticAxisError, match="not an interior O-point"):
        run_neuro_cybernetic_control(
            config_file="dummy.json",
            shot_duration=4,
            seed=7,
            quantum=False,
            save_plot=False,
            verbose=False,
            allow_numpy_fallback=True,
            allow_legacy_numpy_fallback=True,
            kernel_factory=_EdgeAxisKernel,
        )

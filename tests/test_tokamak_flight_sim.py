# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Tokamak Flight Sim Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Deterministic smoke tests for tokamak_flight_sim runtime entry points."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.control.tokamak_flight_sim import (
    FirstOrderActuator,
    IsoFluxController,
    run_flight_sim,
)


class _DummyKernel:
    """Lightweight deterministic stand-in for FusionKernel in CI tests."""

    def __init__(self, _config_file: str) -> None:
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

    def find_x_point(self, _psi: np.ndarray) -> tuple[tuple[float, float], float]:
        return (float(self.R[-2]), float(self.Z[1])), 0.0


def test_run_flight_sim_returns_finite_summary_without_plot() -> None:
    summary = run_flight_sim(
        config_file="dummy.json",
        shot_duration=18,
        seed=123,
        save_plot=False,
        verbose=False,
        kernel_factory=_DummyKernel,
    )
    for key in (
        "seed",
        "config_path",
        "steps",
        "final_ip_ma",
        "final_axis_r",
        "final_axis_z",
        "final_beta_scale",
        "mean_abs_r_error",
        "mean_abs_z_error",
        "mean_abs_radial_actuator_lag",
        "mean_abs_vertical_actuator_lag",
        "mean_abs_heating_actuator_lag",
        "plot_saved",
    ):
        assert key in summary
    assert summary["config_path"] == "dummy.json"
    assert summary["steps"] == 18
    assert summary["plot_saved"] is False
    assert summary["plot_error"] is None
    assert np.isfinite(summary["final_ip_ma"])
    assert np.isfinite(summary["final_axis_r"])
    assert np.isfinite(summary["final_axis_z"])
    assert np.isfinite(summary["final_beta_scale"])
    assert np.isfinite(summary["mean_abs_r_error"])
    assert np.isfinite(summary["mean_abs_z_error"])
    assert np.isfinite(summary["mean_abs_radial_actuator_lag"])
    assert np.isfinite(summary["mean_abs_vertical_actuator_lag"])
    assert np.isfinite(summary["mean_abs_heating_actuator_lag"])


def test_run_flight_sim_is_deterministic_for_fixed_seed() -> None:
    kwargs = dict(
        config_file="dummy.json",
        shot_duration=14,
        seed=77,
        save_plot=False,
        verbose=False,
        kernel_factory=_DummyKernel,
    )
    a = run_flight_sim(**kwargs)
    b = run_flight_sim(**kwargs)
    assert a["final_ip_ma"] == b["final_ip_ma"]
    assert a["final_axis_r"] == b["final_axis_r"]
    assert a["final_axis_z"] == b["final_axis_z"]
    assert a["mean_abs_r_error"] == b["mean_abs_r_error"]
    assert a["mean_abs_z_error"] == b["mean_abs_z_error"]
    assert a["mean_abs_radial_actuator_lag"] == b["mean_abs_radial_actuator_lag"]
    assert a["mean_abs_vertical_actuator_lag"] == b["mean_abs_vertical_actuator_lag"]
    assert a["final_beta_scale"] == b["final_beta_scale"]
    assert a["mean_abs_heating_actuator_lag"] == b["mean_abs_heating_actuator_lag"]


def test_run_flight_sim_does_not_mutate_global_numpy_rng_state() -> None:
    np.random.seed(2468)
    state = np.random.get_state()

    run_flight_sim(
        config_file="dummy.json",
        shot_duration=10,
        seed=55,
        save_plot=False,
        verbose=False,
        kernel_factory=_DummyKernel,
    )

    observed = float(np.random.random())
    np.random.set_state(state)
    expected = float(np.random.random())
    assert observed == expected


def test_run_flight_sim_rejects_invalid_shot_duration() -> None:
    with pytest.raises(ValueError, match="shot_duration"):
        run_flight_sim(
            config_file="dummy.json",
            shot_duration=0,
            seed=1,
            save_plot=False,
            verbose=False,
            kernel_factory=_DummyKernel,
        )


def test_first_order_actuator_rejects_invalid_params() -> None:
    with pytest.raises(ValueError, match="tau_s"):
        FirstOrderActuator(tau_s=0.0, dt_s=0.05)
    with pytest.raises(ValueError, match="dt_s"):
        FirstOrderActuator(tau_s=0.05, dt_s=0.0)


def test_isoflux_controller_rejects_invalid_control_dt() -> None:
    with pytest.raises(ValueError, match="control_dt_s"):
        IsoFluxController(
            config_file="dummy.json",
            kernel_factory=_DummyKernel,
            verbose=False,
            control_dt_s=0.0,
        )


def test_isoflux_controller_rejects_invalid_heating_and_limit_controls() -> None:
    with pytest.raises(ValueError, match="heating_actuator_tau_s"):
        IsoFluxController(
            config_file="dummy.json",
            kernel_factory=_DummyKernel,
            verbose=False,
            heating_actuator_tau_s=0.0,
        )
    with pytest.raises(ValueError, match="actuator_current_delta_limit"):
        IsoFluxController(
            config_file="dummy.json",
            kernel_factory=_DummyKernel,
            verbose=False,
            actuator_current_delta_limit=0.0,
        )
    with pytest.raises(ValueError, match="heating_beta_max"):
        IsoFluxController(
            config_file="dummy.json",
            kernel_factory=_DummyKernel,
            verbose=False,
            heating_beta_max=1.0,
        )


def test_run_flight_sim_heating_tau_controls_actuator_lag() -> None:
    fast = run_flight_sim(
        config_file="dummy.json",
        shot_duration=18,
        seed=10,
        save_plot=False,
        verbose=False,
        heating_actuator_tau_s=0.002,
        kernel_factory=_DummyKernel,
    )
    slow = run_flight_sim(
        config_file="dummy.json",
        shot_duration=18,
        seed=10,
        save_plot=False,
        verbose=False,
        heating_actuator_tau_s=0.5,
        kernel_factory=_DummyKernel,
    )
    assert fast["mean_abs_heating_actuator_lag"] < slow["mean_abs_heating_actuator_lag"]


# ── FirstOrderActuator: rate limiting ────────────────────────────

class TestActuatorRateLimiting:
    def test_rate_limit_clips_large_step(self):
        act = FirstOrderActuator(tau_s=0.01, dt_s=0.01, rate_limit=1.0)
        act.step(1000.0)
        # max_du = rate_limit * dt_s = 0.01; state was 0 → clipped to 0.01
        assert act.state == pytest.approx(0.01, abs=1e-12)

    def test_no_rate_limit_when_small_step(self):
        act = FirstOrderActuator(tau_s=0.01, dt_s=0.01, rate_limit=1e9)
        act.step(0.5)
        # alpha = dt/(tau+dt) = 0.5; 0 + 0.5*0.5 = 0.25
        assert act.state == pytest.approx(0.25, abs=1e-6)

    def test_negative_rate_limit_clip(self):
        act = FirstOrderActuator(tau_s=0.01, dt_s=0.01, rate_limit=1.0)
        act.state = 1.0
        act.step(-1000.0)
        assert act.state == pytest.approx(0.99, abs=1e-12)


# ── FirstOrderActuator: measurement delay & noise ────────────────

class TestActuatorMeasurement:
    def test_delay_returns_past_value(self):
        act = FirstOrderActuator(
            tau_s=0.01, dt_s=0.01, delay_steps=3, rate_limit=1e9,
        )
        values = []
        for i in range(6):
            act.step(float(i + 1))
            values.append(act.state)
        meas = act.get_measurement()
        # delay_steps=3 → measurement lags by 3 steps
        # buffer has 7 entries (initial 0.0 + 6 steps)
        # idx = len(buffer) - 1 - 3 = 3 → buffer[3] = values[2] (3rd step)
        assert meas == pytest.approx(values[2], abs=1e-6)

    def test_noise_adds_gaussian(self):
        act = FirstOrderActuator(
            tau_s=0.01, dt_s=0.01,
            sensor_noise_std=1.0, rng_seed=42,
        )
        act.step(5.0)
        measurements = [act.get_measurement() for _ in range(200)]
        # Mean should be close to act.state, but not exactly (noise)
        mean_m = float(np.mean(measurements))
        # With noise_std=1.0, mean of 200 samples should be within ~0.2 of state
        assert abs(mean_m - act.state) < 0.5

    def test_zero_noise_exact(self):
        act = FirstOrderActuator(tau_s=0.01, dt_s=0.01, sensor_noise_std=0.0)
        act.step(3.0)
        assert act.get_measurement() == act.state


# ── IsoFluxController: verbose path ──────────────────────────────

class TestIsoFluxControllerVerbose:
    def test_verbose_output(self, capsys):
        sim = IsoFluxController(
            config_file="dummy.json",
            kernel_factory=_DummyKernel,
            verbose=True,
        )
        sim.run_shot(shot_duration=3, save_plot=False)
        out = capsys.readouterr().out
        assert "TOKAMAK FLIGHT SIMULATOR" in out

    def test_nonverbose_no_output(self, capsys):
        sim = IsoFluxController(
            config_file="dummy.json",
            kernel_factory=_DummyKernel,
            verbose=False,
        )
        sim.run_shot(shot_duration=3, save_plot=False)
        out = capsys.readouterr().out
        assert out == ""


# ── run_flight_sim: config_file=None default path ────────────────

class TestRunFlightSimConfigDefault:
    def test_config_none_uses_default_path(self):
        summary = run_flight_sim(
            config_file=None,
            shot_duration=5,
            save_plot=False,
            verbose=False,
            kernel_factory=_DummyKernel,
        )
        assert "iter_config.json" in summary["config_path"]
        assert summary["steps"] == 5


# ── FirstOrderActuator: saturation limits ────────────────────────

class TestActuatorSaturation:
    def test_u_max_clamp(self):
        act = FirstOrderActuator(tau_s=0.01, dt_s=0.01, u_max=0.5, rate_limit=1e9)
        for _ in range(100):
            act.step(1000.0)
        assert act.state <= 0.5

    def test_u_min_clamp(self):
        act = FirstOrderActuator(tau_s=0.01, dt_s=0.01, u_min=-0.5, rate_limit=1e9)
        for _ in range(100):
            act.step(-1000.0)
        assert act.state >= -0.5

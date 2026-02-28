# Tests for ice-pellet fueling mode controller.

import numpy as np
import pytest

from scpn_control.control.fueling_mode import (
    FuelingSimResult,
    IcePelletFuelingController,
    run_fueling_mode,
    simulate_iter_density_control,
)


class TestIcePelletFuelingController:
    def test_init_default(self):
        ctrl = IcePelletFuelingController(target_density=1.0)
        assert ctrl.target_density == 1.0
        assert ctrl.integrator == 0.0

    def test_init_rejects_nonfinite(self):
        with pytest.raises(ValueError, match="finite"):
            IcePelletFuelingController(target_density=float("nan"))

    def test_init_rejects_nonpositive(self):
        with pytest.raises(ValueError, match="finite and > 0"):
            IcePelletFuelingController(target_density=0.0)

    def test_step_returns_command_and_error(self):
        ctrl = IcePelletFuelingController(target_density=1.0)
        cmd, err = ctrl.step(0.8, k=0, dt_s=0.001)
        assert np.isfinite(cmd)
        assert np.isfinite(err)
        assert err == pytest.approx(0.2, abs=1e-6)

    def test_step_near_target_low_command(self):
        ctrl = IcePelletFuelingController(target_density=1.0)
        cmd, err = ctrl.step(0.999, k=0, dt_s=0.001)
        assert abs(cmd) < 1.0


class TestSimulation:
    def test_default_converges(self):
        result = simulate_iter_density_control(steps=500, dt_s=0.001)
        assert isinstance(result, FuelingSimResult)
        assert result.final_abs_error < 0.05
        assert result.rmse > 0.0
        assert len(result.history_density) == result.steps

    def test_custom_target(self):
        result = simulate_iter_density_control(
            target_density=2.0, initial_density=1.5, steps=300
        )
        assert result.final_abs_error < 0.1

    def test_rejects_too_few_steps(self):
        with pytest.raises(ValueError, match="steps must be >= 8"):
            simulate_iter_density_control(steps=3)

    def test_rejects_bad_dt(self):
        with pytest.raises(ValueError, match="dt_s must be finite"):
            simulate_iter_density_control(dt_s=float("nan"))

    def test_rejects_tiny_dt(self):
        with pytest.raises(ValueError, match="dt_s must be >= 1e-5"):
            simulate_iter_density_control(dt_s=1e-8)

    def test_rejects_negative_density(self):
        with pytest.raises(ValueError, match="initial_density must be finite"):
            simulate_iter_density_control(initial_density=-1.0)


class TestRunFuelingMode:
    def test_returns_dict(self):
        out = run_fueling_mode(steps=100)
        assert isinstance(out, dict)
        assert "rmse" in out
        assert "passes_thresholds" in out
        assert "final_density" in out
        assert "max_abs_command" in out

    def test_passes_thresholds_long_run(self):
        out = run_fueling_mode(steps=3000)
        assert out["final_abs_error"] < 0.01

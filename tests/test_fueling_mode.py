# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Test Fueling Mode.

"""Exercise the ice-pellet fueling controller and closed-loop simulation."""

import numpy as np
import pytest

from scpn_control.control.fueling_mode import (
    FuelingSimResult,
    IcePelletFuelingController,
    _build_fueling_controller,
    run_fueling_mode,
    simulate_iter_density_control,
)


def test_fueling_runtime_uses_descriptive_artifact_identity() -> None:
    """The runtime artifact identifies the ice-pellet controller directly."""
    assert _build_fueling_controller().artifact.meta.name == "ice-pellet-fueling-controller"


class TestIcePelletFuelingController:
    """Exercise ice-pellet controller construction and one-step control."""

    def test_init_default(self):
        """Construct the controller with its documented default parameters."""
        ctrl = IcePelletFuelingController(target_density=1.0)
        assert ctrl.target_density == 1.0
        assert ctrl.integrator == 0.0

    def test_init_rejects_nonfinite(self):
        """Reject a non-finite target density during construction."""
        with pytest.raises(ValueError, match="finite"):
            IcePelletFuelingController(target_density=float("nan"))

    def test_init_rejects_nonpositive(self):
        """Reject a non-positive target density during construction."""
        with pytest.raises(ValueError, match="finite and > 0"):
            IcePelletFuelingController(target_density=0.0)

    def test_step_returns_command_and_error(self):
        """Check the command and error produced by step."""
        ctrl = IcePelletFuelingController(target_density=1.0)
        cmd, err = ctrl.step(0.8, k=0, dt_s=0.001)
        assert np.isfinite(cmd)
        assert np.isfinite(err)
        assert err == pytest.approx(0.2, abs=1e-6)

    def test_step_near_target_low_command(self):
        """Limit the fueling command when density is already near target."""
        ctrl = IcePelletFuelingController(target_density=1.0)
        cmd, err = ctrl.step(0.999, k=0, dt_s=0.001)
        assert abs(cmd) < 1.0


class TestSimulation:
    """Exercise closed-loop fueling simulation boundaries."""

    def test_default_converges(self):
        """Confirm convergence under the default fueling simulation parameters."""
        result = simulate_iter_density_control(steps=500, dt_s=0.001)
        assert isinstance(result, FuelingSimResult)
        assert result.final_abs_error < 0.05
        assert result.rmse > 0.0
        assert len(result.history_density) == result.steps

    def test_custom_target(self):
        """Drive the fueling simulation towards a caller-selected target density."""
        result = simulate_iter_density_control(target_density=2.0, initial_density=1.5, steps=300)
        assert result.final_abs_error < 0.1

    def test_rejects_too_few_steps(self):
        """Require rejection of too few steps."""
        with pytest.raises(ValueError, match="steps must be >= 8"):
            simulate_iter_density_control(steps=3)

    def test_rejects_bad_dt(self):
        """Require rejection of bad time step."""
        with pytest.raises(ValueError, match="dt_s must be finite"):
            simulate_iter_density_control(dt_s=float("nan"))

    def test_rejects_tiny_dt(self):
        """Require rejection of tiny time step."""
        with pytest.raises(ValueError, match="dt_s must be >= 1e-5"):
            simulate_iter_density_control(dt_s=1e-8)

    def test_rejects_negative_density(self):
        """Require rejection of negative density."""
        with pytest.raises(ValueError, match="initial_density must be finite"):
            simulate_iter_density_control(initial_density=-1.0)

    def test_rejects_nonfinite_target_density(self):
        """Require rejection of non-finite target density."""
        with pytest.raises(ValueError, match="target_density"):
            simulate_iter_density_control(target_density=float("nan"))

    def test_rejects_nonpositive_target_density(self):
        """Require rejection of nonpositive target density."""
        with pytest.raises(ValueError, match="target_density"):
            simulate_iter_density_control(target_density=0.0)


class TestRunFuelingMode:
    """Exercise the public fueling-mode summary interface."""

    def test_returns_dict(self):
        """Return all declared fueling summary fields in a mapping."""
        out = run_fueling_mode(steps=100)
        assert isinstance(out, dict)
        assert "rmse" in out
        assert "passes_thresholds" in out
        assert "final_density" in out
        assert "max_abs_command" in out

    def test_passes_thresholds_long_run(self):
        """Meet the declared fueling thresholds over a long simulation."""
        out = run_fueling_mode(steps=3000)
        assert out["final_abs_error"] < 0.01

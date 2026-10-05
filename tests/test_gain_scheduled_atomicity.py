# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Gain-scheduled controller atomicity tests.

"""Exercise public regime detection and control-cycle refusal paths."""

import numpy as np
import pytest

from scpn_control.control.gain_scheduled_controller import (
    GainScheduledController,
    OperatingRegime,
    RegimeController,
    RegimeDetector,
)


def _controller(gain: float = 1.0) -> RegimeController:
    return RegimeController(
        OperatingRegime.RAMP_UP,
        Kp=np.array([gain]),
        Ki=np.array([1.0]),
        Kd=np.array([0.0]),
        x_ref=np.array([1.0]),
        constraints={},
    )


def _state(controller: GainScheduledController) -> tuple[object, ...]:
    return (
        controller.current_regime,
        controller.prev_regime,
        controller.switch_time,
        controller.Kp.copy(),
        controller.Ki.copy(),
        controller.Kd.copy(),
        controller.integral_error.copy(),
        controller.prev_error.copy(),
    )


def _assert_state_equal(before: tuple[object, ...], after: tuple[object, ...]) -> None:
    for old, new in zip(before, after, strict=True):
        if isinstance(old, np.ndarray):
            assert isinstance(new, np.ndarray)
            np.testing.assert_array_equal(new, old)
        else:
            assert new == old


def test_detector_rejects_nan_before_history_advance() -> None:
    """An unknown disruption probability cannot become a normal regime."""
    detector = RegimeDetector()
    with pytest.raises(ValueError, match="p_disrupt"):
        detector.detect(np.array([1.0]), np.array([0.0]), 1.0, float("nan"))
    assert detector.history == []


def test_detector_rejects_out_of_range_probability_and_mismatched_state() -> None:
    """Malformed diagnostics cannot enter the regime hysteresis window."""
    detector = RegimeDetector()
    with pytest.raises(ValueError, match="p_disrupt"):
        detector.detect(np.array([1.0]), np.array([0.0]), 1.0, 1.1)
    with pytest.raises(ValueError, match="dstate_dt"):
        detector.detect(np.array([1.0, 2.0]), np.array([0.0]), 1.0, 0.0)
    assert detector.history == []


def test_controller_requires_initial_regime() -> None:
    """A controller without its initial operating point cannot start."""
    with pytest.raises(ValueError, match="RAMP_UP"):
        GainScheduledController({})


def test_step_rejects_nan_measurement_without_state_change() -> None:
    """Invalid measurements must leave the PI state and schedule unchanged."""
    controller = GainScheduledController({OperatingRegime.RAMP_UP: _controller()})
    before = _state(controller)
    with pytest.raises(ValueError, match="x"):
        controller.step(np.array([float("nan")]), 0.0, 0.1, OperatingRegime.RAMP_UP)
    _assert_state_equal(before, _state(controller))


def test_step_rejects_zero_time_step_without_state_change() -> None:
    """A zero time step cannot update the integral or derivative estimate."""
    controller = GainScheduledController({OperatingRegime.RAMP_UP: _controller()})
    before = _state(controller)
    with pytest.raises(ValueError, match="dt"):
        controller.step(np.zeros(1), 0.0, 0.0, OperatingRegime.RAMP_UP)
    _assert_state_equal(before, _state(controller))


def test_step_rejects_finite_input_overflow_without_state_change() -> None:
    """An unrepresentable command must not advance the integral."""
    controller = GainScheduledController({OperatingRegime.RAMP_UP: _controller(1e308)})
    before = _state(controller)
    with pytest.raises(ValueError, match="finite"):
        controller.step(np.array([-1e308]), 0.0, 0.1, OperatingRegime.RAMP_UP)
    _assert_state_equal(before, _state(controller))


def test_missing_switch_regime_does_not_publish_switch() -> None:
    """An absent regime controller cannot leave a half-completed switch."""
    controller = GainScheduledController({OperatingRegime.RAMP_UP: _controller()})
    before = _state(controller)
    with pytest.raises(ValueError, match="controller"):
        controller.step(np.zeros(1), 0.0, 0.1, OperatingRegime.H_MODE_FLAT)
    _assert_state_equal(before, _state(controller))


def test_retrograde_switch_time_does_not_publish_interpolation() -> None:
    """A stale timestamp cannot extrapolate gains outside the switch interval."""
    controller = GainScheduledController(
        {
            OperatingRegime.RAMP_UP: _controller(),
            OperatingRegime.H_MODE_FLAT: _controller(2.0),
        }
    )
    controller.step(np.zeros(1), 1.0, 0.1, OperatingRegime.H_MODE_FLAT)
    before = _state(controller)
    with pytest.raises(ValueError, match="switch fraction"):
        controller.step(np.zeros(1), 0.9, 0.1, OperatingRegime.H_MODE_FLAT)
    _assert_state_equal(before, _state(controller))

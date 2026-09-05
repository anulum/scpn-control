# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — EKF rejected-step state preservation.

"""Exercise rejected EKF operations and recovery through the public API."""

from typing import Literal

import numpy as np
import pytest

from scpn_control._typing import FloatArray
from scpn_control.control.state_estimator import ExtendedKalmanFilter


def _snapshot(ekf: ExtendedKalmanFilter) -> tuple[FloatArray, ...]:
    return tuple(value.copy() for value in (ekf.x, ekf.P, ekf.H, ekf.Q, ekf.R))


def _assert_state(ekf: ExtendedKalmanFilter, expected: tuple[FloatArray, ...]) -> None:
    for actual, previous in zip((ekf.x, ekf.P, ekf.H, ekf.Q, ekf.R), expected, strict=True):
        np.testing.assert_array_equal(actual, previous)


@pytest.mark.parametrize("floating_errors", ["ignore", "raise"])
def test_prediction_overflow_preserves_state_and_retry(floating_errors: Literal["ignore", "raise"]) -> None:
    """Rejected covariance prediction must not advance the state estimate."""
    factor = [1e200]

    def make_filter() -> ExtendedKalmanFilter:
        return ExtendedKalmanFilter(
            np.array([1.0]),
            np.eye(1),
            np.zeros((1, 1)),
            np.eye(1),
            process_model=lambda x, u, dt: x + dt,
            process_jacobian=lambda x, u, dt: np.array([[factor[0]]]),
            measurement_model=lambda x: x.copy(),
            measurement_jacobian=lambda x: np.eye(1),
        )

    ekf, reference = make_filter(), make_filter()
    before = _snapshot(ekf)
    error = ValueError if floating_errors == "ignore" else FloatingPointError
    with np.errstate(over=floating_errors, invalid=floating_errors), pytest.raises(error):
        ekf.predict(1.0)
    _assert_state(ekf, before)
    factor[0] = 1.0
    np.testing.assert_array_equal(ekf.predict(0.5), reference.predict(0.5))
    _assert_state(ekf, _snapshot(reference))


def test_posterior_arithmetic_failure_preserves_state_and_retry() -> None:
    """A Joseph-form arithmetic exception must leave x, P and H together."""

    def make_filter() -> ExtendedKalmanFilter:
        return ExtendedKalmanFilter(
            np.array([1.0]),
            np.eye(1),
            np.zeros((1, 1)),
            np.array([[1e-200]]),
            process_model=lambda x, u, dt: x.copy(),
            process_jacobian=lambda x, u, dt: np.eye(1),
            measurement_model=lambda x: x * 1e100,
            measurement_jacobian=lambda x: np.array([[1e100]]),
        )

    ekf, reference = make_filter(), make_filter()
    before = _snapshot(ekf)
    observation = np.array([2e100])
    with np.errstate(under="raise"), pytest.raises(FloatingPointError, match="underflow"):
        ekf.update(observation)
    _assert_state(ekf, before)
    with np.errstate(under="ignore"):
        np.testing.assert_array_equal(ekf.update(observation), reference.update(observation))
    _assert_state(ekf, _snapshot(reference))
    np.testing.assert_array_equal(observation, [2e100])


@pytest.mark.parametrize("operation", ["predict", "update"])
@pytest.mark.parametrize("failure", ["exception", "invalid_output"])
def test_callback_failure_preserves_state_and_retry(operation: str, failure: str) -> None:
    """Callback argument mutation and rejection must not leak into EKF state."""
    reject = [True]

    def callback(x: FloatArray) -> FloatArray:
        x += 0.25
        if reject[0]:
            if failure == "exception":
                raise ValueError("callback rejected input")
            return np.full_like(x, np.nan)
        return x

    def make_filter() -> ExtendedKalmanFilter:
        return ExtendedKalmanFilter(
            np.array([1.0]),
            np.eye(1),
            np.zeros((1, 1)),
            np.eye(1),
            process_model=lambda x, u, dt: callback(x) if operation == "predict" else x,
            process_jacobian=lambda x, u, dt: np.eye(1),
            measurement_model=lambda x: x.copy(),
            measurement_jacobian=lambda x: np.eye(1),
            state_addition=lambda x, dx: callback(x + dx),
        )

    ekf, reference = make_filter(), make_filter()
    before = _snapshot(ekf)
    with pytest.raises(ValueError):
        if operation == "predict":
            ekf.predict(1.0)
        else:
            ekf.update(np.array([2.0]))
    _assert_state(ekf, before)
    reject[0] = False
    if operation == "predict":
        np.testing.assert_array_equal(ekf.predict(1.0), reference.predict(1.0))
    else:
        np.testing.assert_array_equal(ekf.update(np.array([2.0])), reference.update(np.array([2.0])))
    _assert_state(ekf, _snapshot(reference))

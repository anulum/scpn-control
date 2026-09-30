# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Fault injector tests.

"""Exercise the fault injector that drives controller failover scenarios."""

import numpy as np
import pytest

from scpn_control.control.fault_injector import FaultInjector, FaultType


def test_signals_pass_through_unchanged_before_the_fault_time() -> None:
    """Before the trigger the injector returns the caller's own vector."""
    signals = np.array([1.0, 2.0, 3.0])
    injector = FaultInjector(fault_time=5.0, component_index=1, fault_type=FaultType.SENSOR_DROPOUT)
    assert injector.inject(4.999, signals) is signals


def test_dropout_zeroes_only_the_target_channel_on_a_copy() -> None:
    """A dropout silences one channel and leaves the input vector untouched."""
    signals = np.array([1.0, 2.0, 3.0])
    injector = FaultInjector(fault_time=5.0, component_index=1, fault_type=FaultType.SENSOR_DROPOUT)
    corrupted = injector.inject(5.0, signals)
    np.testing.assert_array_equal(corrupted, [1.0, 0.0, 3.0])
    np.testing.assert_array_equal(signals, [1.0, 2.0, 3.0])


def test_drift_grows_linearly_with_elapsed_time_and_severity() -> None:
    """Sensor drift adds ``severity * (t - fault_time)`` to the target channel."""
    signals = np.array([1.0, 2.0])
    injector = FaultInjector(fault_time=2.0, component_index=0, fault_type=FaultType.SENSOR_DRIFT, severity=0.5)
    np.testing.assert_allclose(injector.inject(2.0, signals), [1.0, 2.0])
    np.testing.assert_allclose(injector.inject(6.0, signals), [3.0, 2.0])


@pytest.mark.parametrize("component_index", [-1, 3])
def test_an_active_fault_outside_the_signal_range_is_refused(component_index: int) -> None:
    """Out-of-range channels raise once the fault is active and pass before it."""
    signals = np.array([1.0, 2.0, 3.0])
    injector = FaultInjector(fault_time=1.0, component_index=component_index, fault_type=FaultType.SENSOR_DRIFT)
    assert injector.inject(0.5, signals) is signals
    with pytest.raises(IndexError, match="component_index outside signal range"):
        injector.inject(1.0, signals)


def test_open_circuit_actuator_delivers_no_output() -> None:
    """An open-circuit actuator drives its channel to zero on a copy."""
    commands = np.array([4.0, -2.0])
    injector = FaultInjector(fault_time=1.0, component_index=1, fault_type=FaultType.OPEN_CIRCUIT_ACTUATOR)
    np.testing.assert_array_equal(injector.inject(1.5, commands), [4.0, 0.0])
    np.testing.assert_array_equal(commands, [4.0, -2.0])


def test_stuck_actuator_locks_at_the_first_active_value() -> None:
    """A stuck actuator holds the value it carried when the fault became active."""
    injector = FaultInjector(fault_time=2.0, component_index=0, fault_type=FaultType.STUCK_ACTUATOR)
    np.testing.assert_array_equal(injector.inject(1.0, np.array([9.0, 1.0])), [9.0, 1.0])
    np.testing.assert_array_equal(injector.inject(2.0, np.array([3.0, 1.0])), [3.0, 1.0])
    np.testing.assert_array_equal(injector.inject(3.0, np.array([7.0, 5.0])), [3.0, 5.0])
    assert injector.stuck_value == 3.0


def test_sensor_noise_adds_seeded_gaussian_noise_of_the_given_deviation() -> None:
    """Noise uses the supplied generator, so a seeded run is reproducible."""
    injector = FaultInjector(
        fault_time=0.0,
        component_index=0,
        fault_type=FaultType.SENSOR_NOISE_INCREASE,
        severity=0.2,
        rng=np.random.default_rng(7),
    )
    draws = np.random.default_rng(7).normal(0.0, 0.2, size=2000)
    readings = np.array([injector.inject(1.0, np.array([5.0, 1.0]))[0] for _ in range(2000)])
    np.testing.assert_array_equal(readings, 5.0 + draws)
    assert abs(readings.std() - 0.2) < 0.01
    np.testing.assert_array_equal(injector.inject(1.0, np.array([5.0, 1.0]))[1:], [1.0])


def test_zero_noise_severity_leaves_the_reading_unchanged() -> None:
    """A zero deviation is admissible and adds nothing."""
    injector = FaultInjector(0.0, 0, FaultType.SENSOR_NOISE_INCREASE, severity=0.0, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(injector.inject(1.0, np.array([2.5])), [2.5])


def test_default_generator_is_created_when_none_is_given() -> None:
    """Omitting ``rng`` still yields a working noise source."""
    injector = FaultInjector(0.0, 0, FaultType.SENSOR_NOISE_INCREASE, severity=1.0)
    assert isinstance(injector.rng, np.random.Generator)
    assert np.isfinite(injector.inject(1.0, np.array([0.0]))[0])


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((0.0, 0, "SENSOR_DROPOUT"), "FaultType"),
        ((np.nan, 0, FaultType.SENSOR_DROPOUT), "fault_time"),
        ((0.0, 0, FaultType.SENSOR_DRIFT, np.inf), "severity"),
        ((0.0, 0, FaultType.SENSOR_NOISE_INCREASE, -0.1), "severity"),
    ],
)
def test_invalid_fault_configuration_is_refused(arguments: tuple[object, ...], message: str) -> None:
    """Unknown categories and non-finite or negative-noise settings fail closed."""
    with pytest.raises(ValueError, match=message):
        FaultInjector(*arguments)

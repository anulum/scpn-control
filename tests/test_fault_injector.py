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

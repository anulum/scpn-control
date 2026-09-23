# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Fault tolerant control contract tests.

"""Exercise fail-closed FDIR inputs through the public controller methods."""

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_control.control.fault_tolerant_control import FaultInjector, FaultType, FDIMonitor, ReconfigurableController


def test_nonfinite_measurement_reports_dropout_without_poisoning_other_sensor() -> None:
    """An invalid measured channel raises one fault while healthy channels still work."""
    monitor = FDIMonitor(n_sensors=2, n_actuators=2, n_alert=2)
    reports = monitor.update(np.array([np.nan, 2.0]), np.array([1.0, 2.0]), 1.0)
    assert [(report.component_index, report.fault_type) for report in reports] == [(0, FaultType.SENSOR_DROPOUT)]
    assert monitor.faulted_sensors == {0}
    assert not monitor.update(np.array([np.inf, 2.0]), np.array([1.0, 2.0]), 2.0)


@pytest.mark.parametrize(
    ("measured", "predicted", "time"),
    [
        (np.array([1.0, 2.0]), np.array([1.0]), 1.0),
        (np.array([1.0]), np.array([np.nan]), 1.0),
        (np.array([1.0]), np.array([1.0]), np.nan),
    ],
)
def test_invalid_fdi_inputs_leave_monitor_unchanged(
    measured: NDArray[np.float64], predicted: NDArray[np.float64], time: float
) -> None:
    """Rejected inputs do not consume a detection-window sample."""
    monitor = FDIMonitor(n_sensors=1, n_actuators=1, n_alert=2)
    with pytest.raises(ValueError):
        monitor.update(measured, predicted, time)
    assert monitor.innovation_idx == 0
    np.testing.assert_array_equal(monitor.innovation_history, np.zeros((2, 1)))
    assert not monitor.detected_faults


def test_bad_variance_is_rejected_before_monitor_mutation() -> None:
    """A zero innovation variance cannot silently declare arbitrary faults."""
    monitor = FDIMonitor(n_sensors=1, n_actuators=1)
    monitor.S_diag[0] = 0.0
    with pytest.raises(ValueError, match="innovation variances"):
        monitor.update(np.array([1.0]), np.array([0.0]), 1.0)
    assert monitor.innovation_idx == 0


def test_negative_coil_index_does_not_fault_last_coil() -> None:
    """Python negative indexing must not reconfigure a different actuator."""
    controller = ReconfigurableController(None, np.eye(2), 2, 2)
    with pytest.raises(IndexError, match="coil_index"):
        controller.handle_actuator_fault(-1, FaultType.OPEN_CIRCUIT_ACTUATOR)
    assert not controller.faulted_coils
    np.testing.assert_array_equal(controller.current_jacobian, np.eye(2))
    np.testing.assert_allclose(controller.step(np.ones(2), 0.1), np.ones(2), atol=1e-5)


def test_invalid_actuator_fault_does_not_mutate_allocation() -> None:
    """Reject category and held value errors before publishing reconfiguration."""
    controller = ReconfigurableController(None, np.eye(2), 2, 2)
    with pytest.raises(ValueError, match="actuator fault"):
        controller.handle_actuator_fault(0, FaultType.SENSOR_DROPOUT)
    with pytest.raises(ValueError, match="stuck_val"):
        controller.handle_actuator_fault(0, FaultType.STUCK_ACTUATOR, np.nan)
    assert not controller.faulted_coils
    assert not controller.stuck_values
    np.testing.assert_array_equal(controller.current_jacobian, np.eye(2))


def test_invalid_allocation_inputs_are_rejected() -> None:
    """A malformed plant or error vector cannot produce a plausible command."""
    with pytest.raises(ValueError, match="sensor-by-coil"):
        ReconfigurableController(None, np.ones((1, 2)), 2, 2)
    controller = ReconfigurableController(None, np.eye(2), 2, 2)
    with pytest.raises(ValueError, match="finite sensor vector"):
        controller.step(np.array([1.0, np.nan]), 0.1)


def test_failed_gain_recalculation_does_not_publish_fault_state() -> None:
    """A failed solve leaves gain, matrix and fault inventory at the prior state."""
    controller = ReconfigurableController(None, np.eye(2), 2, 2)
    old_gain = controller.K.copy()
    controller.W[0, 0] = np.nan
    with pytest.raises(ValueError, match="allocation matrices"):
        controller.handle_actuator_fault(0, FaultType.OPEN_CIRCUIT_ACTUATOR)
    assert not controller.faulted_coils
    np.testing.assert_array_equal(controller.current_jacobian, np.eye(2))
    np.testing.assert_array_equal(controller.K, old_gain)


def test_negative_injector_index_does_not_corrupt_last_signal() -> None:
    """Fault simulation also rejects Python negative indexing."""
    injector = FaultInjector(0.0, -1, FaultType.SENSOR_DROPOUT)
    signals = np.array([1.0, 2.0])
    with pytest.raises(IndexError, match="component_index"):
        injector.inject(1.0, signals)
    np.testing.assert_array_equal(signals, [1.0, 2.0])

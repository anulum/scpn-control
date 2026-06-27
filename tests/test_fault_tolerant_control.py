# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Fault Tolerant Control
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Fault-Tolerant Control Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_control.control.fault_tolerant_control import (
    FDIMonitor,
    FaultReport,
    FaultInjector,
    FaultType,
    ReconfigurableController,
)


def test_fdi_no_fault() -> None:
    fdi = FDIMonitor(n_sensors=3, n_actuators=2, threshold_sigma=3.0, n_alert=5)

    y_meas = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    for t in range(10):
        faults = fdi.update(y_meas, y_pred, float(t))
        assert len(faults) == 0


def test_fdi_sensor_dropout() -> None:
    fdi = FDIMonitor(n_sensors=3, n_actuators=2, threshold_sigma=3.0, n_alert=3)

    y_pred = np.array([1.0, 2.0, 3.0])

    # Normal operation
    fdi.update(np.array([1.0, 2.0, 3.0]), y_pred, 0.0)

    # Dropout on sensor 1 (expected 2.0)
    y_meas = np.array([1.0, 0.0, 3.0])

    # We must also ensure S_diag allows this 2.0 error to be > 3*sigma
    fdi.S_diag = np.array([1.0, 0.1, 1.0])  # Reduce sigma so 2.0 is highly anomalous

    all_faults: list[FaultReport] = []
    # Fill the alert window (n_alert=3) with the dropout reading
    for t in range(1, 5):
        faults = fdi.update(y_meas, y_pred, float(t))
        all_faults.extend(faults)

    assert len(all_faults) == 1
    assert all_faults[0].component_index == 1
    assert all_faults[0].fault_type == FaultType.SENSOR_DROPOUT


def test_fdi_sensor_drift() -> None:
    fdi = FDIMonitor(n_sensors=3, n_actuators=2, threshold_sigma=3.0, n_alert=3)

    y_pred = np.array([1.0, 2.0, 3.0])
    y_meas = np.array([1.0, 2.0, 3.0])

    injector = FaultInjector(fault_time=5.0, component_index=2, fault_type=FaultType.SENSOR_DRIFT, severity=2.0)

    for t in range(10):
        corrupted = injector.inject(float(t), y_meas)
        faults = fdi.update(corrupted, y_pred, float(t))
        if len(faults) > 0:
            assert faults[0].fault_type == FaultType.SENSOR_DRIFT
            assert faults[0].component_index == 2
            break

    assert len(fdi.detected_faults) > 0


def test_reconfigurable_controller_actuator_loss() -> None:
    # Mock square plant for simple exact pseudo-inverse
    J = np.eye(3)
    ctrl = ReconfigurableController(None, J, 3, 3)

    error = np.array([1.0, 1.0, 1.0])
    u_nom = ctrl.step(error, 0.1)

    # With J=I, delta_u should be ~ error
    assert np.allclose(u_nom, [1.0, 1.0, 1.0], atol=1e-3)

    # Inject fault on coil 1
    ctrl.handle_actuator_fault(1, FaultType.OPEN_CIRCUIT_ACTUATOR)

    u_fault = ctrl.step(error, 0.1)

    # Coil 1 should now be 0.0
    assert np.isclose(u_fault[1], 0.0)
    # The others should still respond
    assert np.isclose(u_fault[0], 1.0, atol=1e-3)
    assert np.isclose(u_fault[2], 1.0, atol=1e-3)


def test_controllability_check() -> None:
    J = np.eye(5)
    ctrl = ReconfigurableController(None, J, 5, 5)

    assert ctrl.controllability_check()

    # Fail 3 coils out of 5
    ctrl.handle_actuator_fault(0, FaultType.OPEN_CIRCUIT_ACTUATOR)
    ctrl.handle_actuator_fault(1, FaultType.OPEN_CIRCUIT_ACTUATOR)
    ctrl.handle_actuator_fault(2, FaultType.OPEN_CIRCUIT_ACTUATOR)

    # Should lose controllability
    assert not ctrl.controllability_check()

    # Check graceful shutdown
    sd = ctrl.graceful_shutdown()
    assert np.all(sd == 0.0)


def test_controllability_check_rejects_empty_actuator_space() -> None:
    """A controller with no actuator columns cannot span safety-critical targets."""
    ctrl = ReconfigurableController(None, np.zeros((0, 0)), 0, 0)
    assert not ctrl.controllability_check()
    assert ctrl.graceful_shutdown().shape == (0,)


# --- New citation-backed tests ---


def test_fdir_detects_fault() -> None:
    # Blanke et al. 2006, Ch. 3 — FDI raises fault flag when actuator fails.
    fdi = FDIMonitor(n_sensors=2, n_actuators=1, threshold_sigma=3.0, n_alert=3)
    fdi.S_diag = np.array([0.01, 1.0])  # tight variance on sensor 0

    y_pred = np.array([0.0, 1.0])
    y_meas_fault = np.array([5.0, 1.0])  # sensor 0 stuck far from prediction

    detected: list[FaultReport] = []
    for t in range(5):
        detected.extend(fdi.update(y_meas_fault, y_pred, float(t)))

    assert any(r.component_index == 0 for r in detected)


def test_control_allocation_feasible() -> None:
    # Bodson 2002, J. Guidance 25, 307 — u = B^+ v gives finite output.
    J = np.array([[1.0, 0.5], [0.5, 1.0]])
    ctrl = ReconfigurableController(None, J, 2, 2)

    error = np.array([1.0, 1.0])
    u = ctrl.step(error, 0.01)

    assert np.all(np.isfinite(u)), "pseudo-inverse must yield finite commands"


def test_compute_gain_regularises_singular_jacobian() -> None:
    """Regularized allocation keeps a singular Jacobian finite."""
    J = np.zeros((2, 2))  # singular Jacobian
    ctrl = ReconfigurableController(None, J, 2, 2)
    assert np.all(np.isfinite(ctrl.K))


def test_handle_actuator_fault_stuck() -> None:
    """Lines 163, 168: handle_actuator_fault with STUCK_ACTUATOR stores stuck value."""
    J = np.eye(3)
    ctrl = ReconfigurableController(None, J, 3, 3)
    ctrl.handle_actuator_fault(0, FaultType.STUCK_ACTUATOR, stuck_val=5.0)
    assert 0 in ctrl.stuck_values
    assert ctrl.stuck_values[0] == 5.0

    # Calling again on same coil is a no-op (early return)
    ctrl.handle_actuator_fault(0, FaultType.STUCK_ACTUATOR, stuck_val=10.0)
    assert ctrl.stuck_values[0] == 5.0


def test_handle_sensor_fault() -> None:
    """Sensor fault isolation removes the bad residual from allocation."""
    J = np.eye(3)
    ctrl = ReconfigurableController(None, J, 3, 3)

    nominal = ctrl.step(np.array([1.0, 5.0, 1.0]), 0.1)
    assert np.isclose(nominal[1], 5.0, atol=1e-3)

    ctrl.handle_sensor_fault(1, FaultType.SENSOR_DROPOUT)

    assert 1 in ctrl.faulted_sensors
    assert ctrl.sensor_fault_types[1] is FaultType.SENSOR_DROPOUT
    assert np.allclose(ctrl.W[1, :], 0.0)
    assert np.allclose(ctrl.W[:, 1], 0.0)

    reconfigured = ctrl.step(np.array([1.0, 5.0, 1.0]), 0.1)
    assert np.isclose(reconfigured[1], 0.0, atol=1e-9)
    assert np.isclose(reconfigured[0], 1.0, atol=1e-3)
    assert np.isclose(reconfigured[2], 1.0, atol=1e-3)


def test_handle_sensor_fault_rejects_invalid_index() -> None:
    J = np.eye(2)
    ctrl = ReconfigurableController(None, J, 2, 2)
    try:
        ctrl.handle_sensor_fault(2, FaultType.SENSOR_DRIFT)
    except IndexError as exc:
        assert "sensor_index" in str(exc)
    else:
        raise AssertionError("invalid sensor index must fail fast")


def test_step_compensates_stuck_coil() -> None:
    """Lines 180, 221: step subtracts stuck-coil offset from error."""
    J = np.eye(3)
    ctrl = ReconfigurableController(None, J, 3, 3)
    ctrl.handle_actuator_fault(1, FaultType.STUCK_ACTUATOR, stuck_val=2.0)
    error = np.array([1.0, 1.0, 1.0])
    u = ctrl.step(error, 0.1)
    assert np.isclose(u[1], 0.0)  # faulted coil zeroed


def test_fault_injector_before_fault_time() -> None:
    """Line 215: inject returns unmodified signals before fault_time."""
    inj = FaultInjector(fault_time=5.0, component_index=0, fault_type=FaultType.SENSOR_DROPOUT)
    signals: NDArray[np.float64] = np.array([1.0, 2.0, 3.0])
    result = inj.inject(t=3.0, signals=signals)
    np.testing.assert_array_equal(result, signals)


def test_handle_sensor_fault_rejects_actuator_fault_type() -> None:
    """handle_sensor_fault rejects a fault category that is not a sensor fault."""
    ctrl = ReconfigurableController(None, np.eye(3), 3, 3)
    with pytest.raises(ValueError, match="must describe a sensor fault"):
        ctrl.handle_sensor_fault(0, FaultType.STUCK_ACTUATOR)


def test_handle_sensor_fault_is_idempotent() -> None:
    """Re-flagging an already-faulted sensor is a no-op."""
    ctrl = ReconfigurableController(None, np.eye(3), 3, 3)
    ctrl.handle_sensor_fault(1, FaultType.SENSOR_DROPOUT)
    W_after_first = ctrl.W.copy()
    ctrl.handle_sensor_fault(1, FaultType.SENSOR_DRIFT)
    assert ctrl.sensor_fault_types[1] is FaultType.SENSOR_DROPOUT
    np.testing.assert_array_equal(ctrl.W, W_after_first)


def test_fault_injector_applies_sensor_dropout_after_fault_time() -> None:
    """A sensor-dropout injection zeroes the targeted channel once the fault is active."""
    inj = FaultInjector(fault_time=5.0, component_index=1, fault_type=FaultType.SENSOR_DROPOUT)
    corrupted = inj.inject(t=6.0, signals=np.array([1.0, 2.0, 3.0]))
    assert corrupted[1] == 0.0
    assert corrupted[0] == 1.0 and corrupted[2] == 3.0

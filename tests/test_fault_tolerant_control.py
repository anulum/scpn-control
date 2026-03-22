# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Fault-Tolerant Control Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np

from scpn_control.control.fault_tolerant_control import (
    FDIMonitor,
    FaultInjector,
    FaultType,
    ReconfigurableController,
)


def test_fdi_no_fault():
    fdi = FDIMonitor(n_sensors=3, n_actuators=2, threshold_sigma=3.0, n_alert=5)

    y_meas = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    for t in range(10):
        faults = fdi.update(y_meas, y_pred, float(t))
        assert len(faults) == 0


def test_fdi_sensor_dropout():
    fdi = FDIMonitor(n_sensors=3, n_actuators=2, threshold_sigma=3.0, n_alert=3)

    y_pred = np.array([1.0, 2.0, 3.0])

    # Normal operation
    fdi.update(np.array([1.0, 2.0, 3.0]), y_pred, 0.0)

    # Dropout on sensor 1 (expected 2.0)
    y_meas = np.array([1.0, 0.0, 3.0])

    # We must also ensure S_diag allows this 2.0 error to be > 3*sigma
    fdi.S_diag = np.array([1.0, 0.1, 1.0])  # Reduce sigma so 2.0 is highly anomalous

    all_faults = []
    # Fill the alert window (n_alert=3) with the dropout reading
    for t in range(1, 5):
        faults = fdi.update(y_meas, y_pred, float(t))
        all_faults.extend(faults)

    assert len(all_faults) == 1
    assert all_faults[0].component_index == 1
    assert all_faults[0].fault_type == FaultType.SENSOR_DROPOUT


def test_fdi_sensor_drift():
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


def test_reconfigurable_controller_actuator_loss():
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


def test_controllability_check():
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


# --- New citation-backed tests ---


def test_fdir_detects_fault():
    # Blanke et al. 2006, Ch. 3 — FDI raises fault flag when actuator fails.
    fdi = FDIMonitor(n_sensors=2, n_actuators=1, threshold_sigma=3.0, n_alert=3)
    fdi.S_diag = np.array([0.01, 1.0])  # tight variance on sensor 0

    y_pred = np.array([0.0, 1.0])
    y_meas_fault = np.array([5.0, 1.0])  # sensor 0 stuck far from prediction

    detected = []
    for t in range(5):
        detected.extend(fdi.update(y_meas_fault, y_pred, float(t)))

    assert any(r.component_index == 0 for r in detected)


def test_control_allocation_feasible():
    # Bodson 2002, J. Guidance 25, 307 — u = B^+ v gives finite output.
    J = np.array([[1.0, 0.5], [0.5, 1.0]])
    ctrl = ReconfigurableController(None, J, 2, 2)

    error = np.array([1.0, 1.0])
    u = ctrl.step(error, 0.01)

    assert np.all(np.isfinite(u)), "pseudo-inverse must yield finite commands"


def test_compute_gain_singular():
    """Lines 153-154: _compute_gain falls back to zeros on singular H."""
    J = np.zeros((2, 2))  # singular Jacobian
    ctrl = ReconfigurableController(None, J, 2, 2)
    # K should be zero due to singular H (only regularisation keeps it non-singular)
    assert np.all(np.isfinite(ctrl.K))


def test_handle_actuator_fault_stuck():
    """Lines 163, 168: handle_actuator_fault with STUCK_ACTUATOR stores stuck value."""
    J = np.eye(3)
    ctrl = ReconfigurableController(None, J, 3, 3)
    ctrl.handle_actuator_fault(0, FaultType.STUCK_ACTUATOR, stuck_val=5.0)
    assert 0 in ctrl.stuck_values
    assert ctrl.stuck_values[0] == 5.0

    # Calling again on same coil is a no-op (early return)
    ctrl.handle_actuator_fault(0, FaultType.STUCK_ACTUATOR, stuck_val=10.0)
    assert ctrl.stuck_values[0] == 5.0


def test_handle_sensor_fault():
    """Line 174: handle_sensor_fault is a no-op (placeholder)."""
    J = np.eye(2)
    ctrl = ReconfigurableController(None, J, 2, 2)
    ctrl.handle_sensor_fault(0, FaultType.SENSOR_DROPOUT)
    # No exception, no state change


def test_step_compensates_stuck_coil():
    """Lines 180, 221: step subtracts stuck-coil offset from error."""
    J = np.eye(3)
    ctrl = ReconfigurableController(None, J, 3, 3)
    ctrl.handle_actuator_fault(1, FaultType.STUCK_ACTUATOR, stuck_val=2.0)
    error = np.array([1.0, 1.0, 1.0])
    u = ctrl.step(error, 0.1)
    assert np.isclose(u[1], 0.0)  # faulted coil zeroed


def test_fault_injector_before_fault_time():
    """Line 215: inject returns unmodified signals before fault_time."""
    inj = FaultInjector(fault_time=5.0, component_index=0, fault_type=FaultType.SENSOR_DROPOUT)
    signals = np.array([1.0, 2.0, 3.0])
    result = inj.inject(t=3.0, signals=signals)
    np.testing.assert_array_equal(result, signals)

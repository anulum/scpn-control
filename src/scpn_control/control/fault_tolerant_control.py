# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Fault tolerant control.

"""Fault-detection, isolation, and reconfiguration utilities for controller failover tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.control.fault_injector import FaultInjector as FaultInjector
from scpn_control.control.fault_injector import FaultType as FaultType

# FDIR terminology: Blanke et al. 2006, "Diagnosis and Fault-Tolerant Control",
# Springer, Ch. 1 — fault detection, isolation, reconfiguration.
#
# Tokamak actuator redundancy: multiple PF coils enable reconfiguration after
# individual coil failures; Ambrosino et al. 2008, Fusion Eng. Des. 83, 1485.
#
# Control allocation for over-actuated systems: u = B^+ v where B^+ is the
# Moore-Penrose pseudo-inverse; Bodson 2002, J. Guidance 25, 307.

# Minimum rank required to control Ip and vertical position (two critical targets).
MIN_REQUIRED_RANK: int = 2

# Innovation threshold in units of sigma: z-score above which a sensor is flagged.
# Blanke et al. 2006, Ch. 3 — 3σ threshold is standard for GLR/CUSUM detectors.
DEFAULT_THRESHOLD_SIGMA: float = 3.0

# Small regularisation on the weighted least-squares gain to prevent singularity.
# Bodson 2002, J. Guidance 25, 307, Eq. 12 — Tikhonov regularisation λI.
GAIN_REGULARISATION: float = 1e-6


@dataclass
class FaultReport:
    """A detected fault flagged by the FDI monitor.

    Attributes
    ----------
    component_index
        Index of the faulted actuator or sensor.
    is_sensor
        ``True`` if the faulted component is a sensor, ``False`` for an actuator.
    fault_type
        The detected fault category.
    confidence
        Detection confidence in [0, 1].
    time_detected
        Time of detection in seconds.
    """

    component_index: int
    is_sensor: bool
    fault_type: FaultType
    confidence: float
    time_detected: float


class FDIMonitor:
    """Innovation-based fault detection and isolation.

    Implements the sequential test from Blanke et al. 2006, Ch. 3:
    if every sample in a sliding window of length n_alert satisfies
    |ν_i| > threshold_sigma · σ_i, sensor i is declared faulty.
    """

    def __init__(
        self,
        n_sensors: int,
        n_actuators: int,
        threshold_sigma: float = DEFAULT_THRESHOLD_SIGMA,
        n_alert: int = 5,
    ):
        if isinstance(n_sensors, bool) or not isinstance(n_sensors, int) or n_sensors < 0:
            raise ValueError("n_sensors must be a non-negative integer")
        if isinstance(n_actuators, bool) or not isinstance(n_actuators, int) or n_actuators < 0:
            raise ValueError("n_actuators must be a non-negative integer")
        if isinstance(n_alert, bool) or not isinstance(n_alert, int) or n_alert < 1:
            raise ValueError("n_alert must be a positive integer")
        if not np.isfinite(threshold_sigma) or threshold_sigma <= 0:
            raise ValueError("threshold_sigma must be positive and finite")
        self.n_sensors = n_sensors
        self.n_actuators = n_actuators
        self.threshold_sigma = threshold_sigma
        self.n_alert = n_alert

        self.innovation_history: FloatArray = np.zeros((n_alert, n_sensors))
        self.innovation_idx = 0

        # Normalised innovation variance; set to 1 for pre-whitened inputs.
        self.S_diag: FloatArray = np.ones(n_sensors)

        self.detected_faults: list[FaultReport] = []
        self.faulted_sensors: set[int] = set()

    def update(self, y_measured: AnyFloatArray, y_predicted: AnyFloatArray, t: float) -> list[FaultReport]:
        """Flag invalid measurements immediately and persistent finite innovations.

        Invalid predictions, variance, time, or vector shape raise before the
        monitor consumes a sample. Non-finite measured channels are isolated as
        dropouts while healthy channels continue through the innovation window.
        """
        measured = np.asarray(y_measured, dtype=np.float64)
        predicted = np.asarray(y_predicted, dtype=np.float64)
        if measured.shape != (self.n_sensors,) or predicted.shape != (self.n_sensors,):
            raise ValueError("measurements and predictions must match the sensor vector shape")
        if not np.all(np.isfinite(predicted)) or not np.isfinite(t):
            raise ValueError("predictions and detection time must be finite")
        if self.S_diag.shape != (self.n_sensors,) or not np.all(np.isfinite(self.S_diag)) or np.any(self.S_diag <= 0):
            raise ValueError("innovation variances must be positive and finite")
        invalid = ~np.isfinite(measured)
        with np.errstate(over="ignore", invalid="ignore"):
            nu = np.where(invalid, 0.0, measured - predicted)
        if not np.all(np.isfinite(nu)):
            raise ValueError("finite measurements produced a non-finite innovation")

        self.innovation_history[self.innovation_idx] = nu
        self.innovation_idx = (self.innovation_idx + 1) % self.n_alert

        new_faults = []

        for i in range(self.n_sensors):
            if i in self.faulted_sensors:
                continue

            if invalid[i]:
                report = FaultReport(i, True, FaultType.SENSOR_DROPOUT, 1.0, t)
                new_faults.append(report)
                self.detected_faults.append(report)
                self.faulted_sensors.add(i)
                continue

            hist = self.innovation_history[:, i]
            sigma = np.sqrt(self.S_diag[i])

            if np.all(np.abs(hist) > self.threshold_sigma * sigma):
                if abs(measured[i]) < 1e-6:
                    ftype = FaultType.SENSOR_DROPOUT
                else:
                    ftype = FaultType.SENSOR_DRIFT

                report = FaultReport(
                    component_index=i,
                    is_sensor=True,
                    fault_type=ftype,
                    confidence=1.0,
                    time_detected=t,
                )
                new_faults.append(report)
                self.detected_faults.append(report)
                self.faulted_sensors.add(i)

        return new_faults


class ReconfigurableController:
    """Control-allocation reconfiguration after actuator or sensor faults.

    Weighted pseudo-inverse gain: u = (J^T W J + λI)^{-1} J^T W v.
    Reference: Bodson 2002, J. Guidance 25, 307, Eq. 12.

    Faulted coil columns are zeroed in J before the gain is recomputed,
    matching the reconfiguration procedure in Ambrosino et al. 2008,
    Fusion Eng. Des. 83, 1485 for the ITER VS system.
    """

    def __init__(self, base_controller: Any, jacobian: AnyFloatArray, n_coils: int, n_sensors: int):
        if isinstance(n_coils, bool) or not isinstance(n_coils, int) or n_coils < 0:
            raise ValueError("n_coils must be a non-negative integer")
        if isinstance(n_sensors, bool) or not isinstance(n_sensors, int) or n_sensors < 0:
            raise ValueError("n_sensors must be a non-negative integer")
        self.base_controller = base_controller
        jacobian_arr = np.asarray(jacobian, dtype=np.float64)
        if jacobian_arr.shape != (n_sensors, n_coils) or not np.all(np.isfinite(jacobian_arr)):
            raise ValueError("jacobian must be a finite sensor-by-coil matrix")
        self.nominal_jacobian: FloatArray = jacobian_arr.copy()
        self.current_jacobian: FloatArray = jacobian_arr.copy()
        self.n_coils = n_coils
        self.n_sensors = n_sensors

        self.faulted_coils: set[int] = set()
        self.faulted_sensors: set[int] = set()
        self.stuck_values: dict[int, float] = {}
        self.sensor_fault_types: dict[int, FaultType] = {}

        self.W: FloatArray = np.eye(n_sensors)
        self.lambda_reg = GAIN_REGULARISATION

        self.K = self._compute_gain()

    def _compute_gain(
        self,
        jacobian: FloatArray | None = None,
        weight: FloatArray | None = None,
        faulted_coils: set[int] | None = None,
    ) -> FloatArray:
        """Solve finite weighted allocation for the supplied candidate state."""
        # u = B^+ v  where B^+ = (J^T W J + λI)^{-1} J^T W
        # Bodson 2002, J. Guidance 25, 307, Eq. 12
        J = self.current_jacobian if jacobian is None else jacobian
        W = self.W if weight is None else weight
        if not np.all(np.isfinite(J)) or not np.all(np.isfinite(W)):
            raise ValueError("allocation matrices must be finite")
        # The Tikhonov term λI (λ = 1e-6 > 0) makes H = JᵀWJ + λI strictly
        # positive definite for finite J and W, so solve avoids an explicit inverse.
        # JᵀWJ can still overflow to infinity for a finite J, and the solve then
        # returns a silently zero gain, so the normal matrix is checked as well.
        with np.errstate(over="ignore", invalid="ignore"):
            J_T_W = J.T @ W
            H = J_T_W @ J + self.lambda_reg * np.eye(self.n_coils)
            K = np.linalg.solve(H, J_T_W)
        if not np.all(np.isfinite(H)) or not np.all(np.isfinite(K)):
            raise ValueError("allocation gain must be finite")

        for i in self.faulted_coils if faulted_coils is None else faulted_coils:
            K[i, :] = 0.0

        return np.asarray(K)

    def handle_actuator_fault(self, coil_index: int, fault_type: FaultType, stuck_val: float = 0.0) -> None:
        """Reconfigure the controller around a faulted actuator coil.

        Zeroes the coil's Jacobian column and recomputes the control gain;
        records the held value for a stuck actuator.

        Parameters
        ----------
        coil_index
            Index of the faulted coil.
        fault_type
            The actuator fault category.
        stuck_val
            Held output value for a stuck actuator.

        Raises
        ------
        IndexError
            If the coil index is outside the configured actuator range.
        ValueError
            If the category or held value is invalid, or allocation fails.
        """
        if isinstance(coil_index, bool) or not isinstance(coil_index, int) or not 0 <= coil_index < self.n_coils:
            raise IndexError("coil_index outside configured coil range")
        if fault_type not in {FaultType.STUCK_ACTUATOR, FaultType.OPEN_CIRCUIT_ACTUATOR}:
            raise ValueError("fault_type must describe an actuator fault")
        if fault_type is FaultType.STUCK_ACTUATOR and not np.isfinite(stuck_val):
            raise ValueError("stuck_val must be finite")
        if coil_index in self.faulted_coils:
            return

        candidate_faults = self.faulted_coils | {coil_index}
        candidate_jacobian = self.current_jacobian.copy()
        candidate_jacobian[:, coil_index] = 0.0
        candidate_gain = self._compute_gain(candidate_jacobian, faulted_coils=candidate_faults)
        self.current_jacobian = candidate_jacobian
        self.K = candidate_gain
        self.faulted_coils = candidate_faults
        if fault_type is FaultType.STUCK_ACTUATOR:
            self.stuck_values[coil_index] = float(stuck_val)

    def handle_sensor_fault(self, sensor_index: int, fault_type: FaultType) -> None:
        """Reconfigure the controller around a faulted sensor.

        Zeroes the sensor's weighting and recomputes the control gain.

        Parameters
        ----------
        sensor_index
            Index of the faulted sensor.
        fault_type
            The sensor fault category (dropout, drift, or noise increase).

        Raises
        ------
        IndexError
            If ``sensor_index`` is outside the configured sensor range.
        ValueError
            If ``fault_type`` is not a sensor fault.
        """
        if (
            isinstance(sensor_index, bool)
            or not isinstance(sensor_index, int)
            or not 0 <= sensor_index < self.n_sensors
        ):
            raise IndexError("sensor_index outside configured sensor range")
        if fault_type not in {
            FaultType.SENSOR_DROPOUT,
            FaultType.SENSOR_DRIFT,
            FaultType.SENSOR_NOISE_INCREASE,
        }:
            raise ValueError("fault_type must describe a sensor fault")
        if sensor_index in self.faulted_sensors:
            return

        candidate_weight = self.W.copy()
        candidate_weight[sensor_index, :] = 0.0
        candidate_weight[:, sensor_index] = 0.0
        candidate_gain = self._compute_gain(weight=candidate_weight)
        self.W = candidate_weight
        self.K = candidate_gain
        self.faulted_sensors.add(sensor_index)
        self.sensor_fault_types[sensor_index] = fault_type

    def step(self, error: AnyFloatArray, dt: float) -> FloatArray:
        """Allocate a finite sensor error; compensate for stuck-coil offsets.

        ``dt`` is validated as finite and non-negative. The allocation itself
        is an instantaneous gain operation and does not integrate over ``dt``.
        """
        error_array = np.asarray(error, dtype=np.float64)
        if error_array.shape != (self.n_sensors,) or not np.all(np.isfinite(error_array)):
            raise ValueError("error must be a finite sensor vector")
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("dt must be non-negative and finite")
        adjusted_error = error_array.copy()
        for sensor_idx in self.faulted_sensors:
            adjusted_error[sensor_idx] = 0.0

        with np.errstate(over="ignore", invalid="ignore"):
            for c_idx, val in self.stuck_values.items():
                adjusted_error -= self.nominal_jacobian[:, c_idx] * val
            delta_u = self.K @ adjusted_error
        if not np.all(np.isfinite(delta_u)):
            raise ValueError("allocation produced a non-finite command")

        for c_idx in self.faulted_coils:
            delta_u[c_idx] = 0.0

        return np.asarray(delta_u)

    def controllability_check(self) -> bool:
        """Return True if the remaining actuators span the minimum required target space.

        MIN_REQUIRED_RANK = 2 covers control of Ip and vertical position,
        the two safety-critical outputs identified by Ambrosino et al. 2008.
        """
        if len(self.faulted_coils) > self.n_coils // 2:
            return False

        singular_values = np.linalg.svd(self.current_jacobian, compute_uv=False)
        if singular_values.size == 0:
            return False
        largest = max(float(value) for value in singular_values)
        tolerance = max(self.current_jacobian.shape) * np.finfo(float).eps * largest
        rank = sum(1 for value in singular_values if float(value) > tolerance)
        return bool(rank >= MIN_REQUIRED_RANK)

    def graceful_shutdown(self) -> FloatArray:
        """Return zero ramp-down command for all coils."""
        return np.zeros(self.n_coils)

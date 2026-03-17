# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

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


class FaultType(Enum):
    STUCK_ACTUATOR = auto()
    OPEN_CIRCUIT_ACTUATOR = auto()
    SENSOR_DROPOUT = auto()
    SENSOR_DRIFT = auto()
    SENSOR_NOISE_INCREASE = auto()


@dataclass
class FaultReport:
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
        self.n_sensors = n_sensors
        self.n_actuators = n_actuators
        self.threshold_sigma = threshold_sigma
        self.n_alert = n_alert

        self.innovation_history = np.zeros((n_alert, n_sensors))
        self.innovation_idx = 0

        # Normalised innovation variance; set to 1 for pre-whitened inputs.
        self.S_diag = np.ones(n_sensors)

        self.detected_faults: list[FaultReport] = []
        self.faulted_sensors: set[int] = set()

    def update(self, y_measured: np.ndarray, y_predicted: np.ndarray, t: float) -> list[FaultReport]:
        """Check for sensor faults using the prediction residual (innovation)."""
        nu = y_measured - y_predicted

        self.innovation_history[self.innovation_idx] = nu
        self.innovation_idx = (self.innovation_idx + 1) % self.n_alert

        new_faults = []

        for i in range(self.n_sensors):
            if i in self.faulted_sensors:
                continue

            hist = self.innovation_history[:, i]
            sigma = np.sqrt(self.S_diag[i])

            if np.all(np.abs(hist) > self.threshold_sigma * sigma):
                if np.isnan(y_measured[i]) or abs(y_measured[i]) < 1e-6:
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

    def __init__(self, base_controller: Any, jacobian: np.ndarray, n_coils: int, n_sensors: int):
        self.base_controller = base_controller
        self.nominal_jacobian = jacobian.copy()
        self.current_jacobian = jacobian.copy()
        self.n_coils = n_coils
        self.n_sensors = n_sensors

        self.faulted_coils: set[int] = set()
        self.stuck_values: dict[int, float] = {}

        self.W = np.eye(jacobian.shape[0])
        self.lambda_reg = GAIN_REGULARISATION

        self.K = self._compute_gain()

    def _compute_gain(self) -> np.ndarray:
        # u = B^+ v  where B^+ = (J^T W J + λI)^{-1} J^T W
        # Bodson 2002, J. Guidance 25, 307, Eq. 12
        J = self.current_jacobian
        J_T_W = J.T @ self.W
        H = J_T_W @ J + self.lambda_reg * np.eye(self.n_coils)

        try:
            H_inv = np.linalg.inv(H)
            K = H_inv @ J_T_W
        except np.linalg.LinAlgError:
            K = np.zeros_like(J.T)

        for i in self.faulted_coils:
            K[i, :] = 0.0

        return np.asarray(K)

    def handle_actuator_fault(self, coil_index: int, fault_type: FaultType, stuck_val: float = 0.0) -> None:
        if coil_index in self.faulted_coils:
            return

        self.faulted_coils.add(coil_index)

        if fault_type == FaultType.STUCK_ACTUATOR:
            self.stuck_values[coil_index] = stuck_val

        self.current_jacobian[:, coil_index] = 0.0
        self.K = self._compute_gain()

    def handle_sensor_fault(self, sensor_index: int, fault_type: FaultType) -> None:
        pass

    def step(self, error: np.ndarray, dt: float) -> np.ndarray:
        """Apply reconfigured gain; compensate for stuck-coil offsets."""
        adjusted_error = error.copy()
        for c_idx, val in self.stuck_values.items():
            adjusted_error -= self.nominal_jacobian[:, c_idx] * val

        delta_u = self.K @ adjusted_error

        for c_idx in self.faulted_coils:
            delta_u[c_idx] = 0.0

        return np.asarray(delta_u)

    def controllability_check(self) -> bool:
        """True if remaining actuators span the minimum required target space.

        MIN_REQUIRED_RANK = 2 covers control of Ip and vertical position,
        the two safety-critical outputs identified by Ambrosino et al. 2008.
        """
        if len(self.faulted_coils) > self.n_coils // 2:
            return False

        J = self.current_jacobian
        rank = np.linalg.matrix_rank(J)
        return bool(rank >= MIN_REQUIRED_RANK)

    def graceful_shutdown(self) -> np.ndarray:
        """Return zero ramp-down command for all coils."""
        return np.zeros(self.n_coils)


class FaultInjector:
    def __init__(self, fault_time: float, component_index: int, fault_type: FaultType, severity: float = 1.0):
        self.fault_time = fault_time
        self.component_index = component_index
        self.fault_type = fault_type
        self.severity = severity

    def inject(self, t: float, signals: np.ndarray) -> np.ndarray:
        if t < self.fault_time:
            return signals

        corrupted = signals.copy()

        if self.fault_type == FaultType.SENSOR_DROPOUT:
            corrupted[self.component_index] = 0.0
        elif self.fault_type == FaultType.SENSOR_DRIFT:
            corrupted[self.component_index] += self.severity * (t - self.fault_time)

        return corrupted

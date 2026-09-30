# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Fault injection simulation.

"""Fault injection simulation for controller failover tests."""

from __future__ import annotations

from enum import Enum, auto

import numpy as np

from scpn_control._typing import AnyFloatArray


class FaultType(Enum):
    """Actuator and sensor fault categories."""

    STUCK_ACTUATOR = auto()
    OPEN_CIRCUIT_ACTUATOR = auto()
    SENSOR_DROPOUT = auto()
    SENSOR_DRIFT = auto()
    SENSOR_NOISE_INCREASE = auto()


class FaultInjector:
    """Inject a fault into actuator/sensor signals after a trigger time.

    Every fault category changes the targeted channel once ``t >= fault_time``
    (fault models after Blanke et al. 2006, *Diagnosis and Fault-Tolerant
    Control*, Springer, Ch. 2):

    - ``SENSOR_DROPOUT``: the reading is lost and reported as ``0.0``.
    - ``SENSOR_DRIFT``: an additive bias grows as ``severity * (t - fault_time)``.
    - ``SENSOR_NOISE_INCREASE``: zero-mean Gaussian noise with standard
      deviation ``severity`` is added to the reading.
    - ``OPEN_CIRCUIT_ACTUATOR``: the actuator delivers no output (``0.0``).
    - ``STUCK_ACTUATOR``: the output locks at the value the channel carries on
      the first call at or after ``fault_time`` and stays there.

    Parameters
    ----------
    fault_time
        Time at which the fault begins, in seconds.
    component_index
        Index of the component to corrupt.
    fault_type
        The fault category to inject.
    severity
        Drift rate for ``SENSOR_DRIFT`` or noise standard deviation for
        ``SENSOR_NOISE_INCREASE``; finite, and non-negative for noise.
    rng
        Random generator for ``SENSOR_NOISE_INCREASE``; pass a seeded generator
        for reproducible runs. A fresh unseeded generator is used when omitted.

    Raises
    ------
    ValueError
        If ``fault_type`` is not a :class:`FaultType`, ``fault_time`` or
        ``severity`` is not finite, or a noise severity is negative.
    """

    def __init__(
        self,
        fault_time: float,
        component_index: int,
        fault_type: FaultType,
        severity: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        if not isinstance(fault_type, FaultType):
            raise ValueError("fault_type must be a FaultType")
        if not np.isfinite(fault_time):
            raise ValueError("fault_time must be finite")
        if not np.isfinite(severity) or (fault_type is FaultType.SENSOR_NOISE_INCREASE and severity < 0):
            raise ValueError("severity must be finite, and non-negative for sensor noise")
        self.fault_time = fault_time
        self.component_index = component_index
        self.fault_type = fault_type
        self.severity = severity
        self.rng = rng if rng is not None else np.random.default_rng()
        self.stuck_value: float | None = None

    def inject(self, t: float, signals: AnyFloatArray) -> AnyFloatArray:
        """Return the signal vector with the configured fault applied.

        Parameters
        ----------
        t
            Current time in seconds; the fault applies once ``t >= fault_time``.
        signals
            The clean signal vector.

        Returns
        -------
        AnyFloatArray
            A corrupted copy once the fault is active; the input object itself
            before the fault time.

        Raises
        ------
        IndexError
            If an active fault targets a channel outside the signal vector.
        """
        if t < self.fault_time:
            return signals

        if self.component_index < 0 or self.component_index >= len(signals):
            raise IndexError("component_index outside signal range")

        corrupted = signals.copy()
        index = self.component_index

        if self.fault_type is FaultType.SENSOR_DROPOUT or self.fault_type is FaultType.OPEN_CIRCUIT_ACTUATOR:
            corrupted[index] = 0.0
        elif self.fault_type is FaultType.SENSOR_DRIFT:
            corrupted[index] += self.severity * (t - self.fault_time)
        elif self.fault_type is FaultType.SENSOR_NOISE_INCREASE:
            corrupted[index] += self.rng.normal(0.0, self.severity)
        else:  # FaultType.STUCK_ACTUATOR, the remaining category
            if self.stuck_value is None:
                self.stuck_value = float(signals[index])
            corrupted[index] = self.stuck_value

        return corrupted

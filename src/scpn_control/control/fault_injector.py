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

    Parameters
    ----------
    fault_time
        Time at which the fault begins, in seconds.
    component_index
        Index of the component to corrupt.
    fault_type
        The fault category to inject.
    severity
        Fault severity scale (e.g. drift rate).
    """

    def __init__(self, fault_time: float, component_index: int, fault_type: FaultType, severity: float = 1.0):
        self.fault_time = fault_time
        self.component_index = component_index
        self.fault_type = fault_type
        self.severity = severity

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
            The signals with the fault applied (unchanged before the fault time).

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

        if self.fault_type == FaultType.SENSOR_DROPOUT:
            corrupted[self.component_index] = 0.0
        elif self.fault_type == FaultType.SENSOR_DRIFT:
            corrupted[self.component_index] += self.severity * (t - self.fault_time)

        return corrupted

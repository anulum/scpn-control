# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Optional native PID compatibility
"""Select the native PID binding when present, with the canonical Python fallback."""

from __future__ import annotations

from typing import Any

from scpn_control.control.pid_controller import PIDController


class RustPIDController:
    """Expose one PID interface across optional native and Python backends.

    Parameters
    ----------
    kp, ki, kd : float
        Finite proportional, integral and derivative gains.

    Notes
    -----
    The fallback delegates to the same Python law used by the flight simulator.
    A missing native ``PyPIDController`` therefore cannot weaken finite-value
    refusal or failure atomicity. The selected backend is visible in ``repr``.
    """

    _PurePythonPID = PIDController

    def __init__(self, kp: float, ki: float, kd: float) -> None:
        self._inner: Any
        try:
            from scpn_control_rs import PyPIDController

            self._inner = PyPIDController(kp, ki, kd)
            self._mode = "rust"
        except (ImportError, AttributeError):
            self._inner = self._PurePythonPID(kp, ki, kd)
            self._mode = "fallback"

    @classmethod
    def radial(cls) -> RustPIDController:
        """Create the historical radial PID gains with the available backend."""
        obj = cls.__new__(cls)
        try:
            from scpn_control_rs import PyPIDController

            obj._inner = PyPIDController.radial()
            obj._mode = "rust"
        except (ImportError, AttributeError):
            obj._inner = cls._PurePythonPID(2.0, 0.1, 0.5)
            obj._mode = "fallback"
        return obj

    @classmethod
    def vertical(cls) -> RustPIDController:
        """Create the historical vertical PID gains with the available backend."""
        obj = cls.__new__(cls)
        try:
            from scpn_control_rs import PyPIDController

            obj._inner = PyPIDController.vertical()
            obj._mode = "rust"
        except (ImportError, AttributeError):
            obj._inner = cls._PurePythonPID(5.0, 0.2, 2.0)
            obj._mode = "fallback"
        return obj

    def step(self, error: float) -> float:
        """Advance one finite PID step without publishing a failed update."""
        return float(self._inner.step(error))

    def reset(self) -> None:
        """Clear the integral, derivative and slew state."""
        self._inner.reset()

    @property
    def kp(self) -> float:
        """Return the proportional gain."""
        return float(self._inner.kp)

    @property
    def ki(self) -> float:
        """Return the integral gain."""
        return float(self._inner.ki)

    @property
    def kd(self) -> float:
        """Return the derivative gain."""
        return float(self._inner.kd)

    def __repr__(self) -> str:
        """Show the selected backend and gains for diagnosis."""
        return f"RustPIDController(mode={self._mode}, kp={self.kp}, ki={self.ki}, kd={self.kd})"

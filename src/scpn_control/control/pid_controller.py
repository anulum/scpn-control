# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Anti-Windup PID Controller
"""Position PID law with an optional saturation/slew envelope and anti-windup.

This is the Python counterpart of ``scpn-control-rs`` ``control_control::pid`` and
shares its numerics: with no envelope configured the law is the ideal
``kp*e + ki*Σe + kd*Δe``; with an envelope the output is saturated and
slew-limited and the integrator is frozen for any step whose raw output is
clamped in the same direction as the error (conditional-integration anti-windup),
so a sustained saturating error can never wind the integral up without bound.
"""

from __future__ import annotations

import math


class PIDController:
    """Single-axis PID controller with an optional anti-windup output envelope.

    The envelope is OFF by default so the controller reproduces the ideal PID law
    exactly. Enable it with :meth:`with_output_limits` and/or
    :meth:`with_slew_limit`; both return ``self`` for chaining.
    """

    def __init__(self, kp: float, ki: float, kd: float) -> None:
        kp = float(kp)
        ki = float(ki)
        kd = float(kd)
        if not (math.isfinite(kp) and math.isfinite(ki) and math.isfinite(kd)):
            raise ValueError("pid gains must be finite")
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._err_sum = 0.0
        self._last_err = 0.0
        self._output_min = -math.inf
        self._output_max = math.inf
        self._slew_max = math.inf
        self._last_output = 0.0

    def with_output_limits(self, output_min: float, output_max: float) -> PIDController:
        """Configure the saturation envelope ``[output_min, output_max]``.

        Both bounds must be finite and ordered. Enabling saturation also enables
        the conditional-integration anti-windup behaviour.
        """
        output_min = float(output_min)
        output_max = float(output_max)
        if not (math.isfinite(output_min) and math.isfinite(output_max)):
            raise ValueError("pid output limits must be finite")
        if output_min > output_max:
            raise ValueError("pid output_min must not exceed output_max")
        self._output_min = output_min
        self._output_max = output_max
        return self

    def with_slew_limit(self, slew_max: float) -> PIDController:
        """Configure the maximum per-step output change ``|Δoutput|``.

        Must be finite and strictly positive. A slew-limited step that clamps the
        output against the error direction also freezes the integrator.
        """
        slew_max = float(slew_max)
        if not math.isfinite(slew_max) or slew_max <= 0.0:
            raise ValueError("pid slew_max must be finite and > 0")
        self._slew_max = slew_max
        return self

    def _saturate_and_slew(self, raw: float) -> float:
        """Clamp ``raw`` to the saturation envelope and per-step slew limit."""
        saturated = min(max(raw, self._output_min), self._output_max)
        delta = saturated - self._last_output
        if abs(delta) > self._slew_max:
            return self._last_output + math.copysign(self._slew_max, delta)
        return saturated

    def step(self, error: float) -> float:
        """Advance one step and return the applied control output.

        Raises
        ------
        ValueError
            If ``error`` is not finite.
        """
        error = float(error)
        if not math.isfinite(error):
            raise ValueError("pid error input must be finite")
        d_err = error - self._last_err
        candidate_sum = self._err_sum + error
        raw = self.kp * error + self.ki * candidate_sum + self.kd * d_err
        applied = self._saturate_and_slew(raw)

        # Anti-windup: ``(raw - applied) * error > 0`` means the output was clamped
        # in the same direction the error is pushing, so committing the integral
        # would only wind it further into the limit. Freeze it in that case.
        winding = (raw - applied) * error > 0.0
        if not winding:
            self._err_sum = candidate_sum
        self._last_err = error
        self._last_output = applied
        return applied

    def reset(self) -> None:
        """Clear the integral, derivative, and slew state."""
        self._err_sum = 0.0
        self._last_err = 0.0
        self._last_output = 0.0

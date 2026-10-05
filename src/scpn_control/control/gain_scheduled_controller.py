# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Gain scheduled controller.

"""Gain-scheduled controller selection across declared plasma operating regimes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.control.gain_scheduled_scenario import (
    ScenarioSchedule as ScenarioSchedule,
)
from scpn_control.control.gain_scheduled_scenario import (
    ScenarioWaveform as ScenarioWaveform,
)
from scpn_control.control.gain_scheduled_scenario import (
    iter_baseline_schedule as iter_baseline_schedule,
)

# Gain scheduling: operating-point-indexed PID gains.
# Theoretical basis:
#   Rugh & Shamma 2000, Automatica 36, 1401 — survey of gain-scheduling.
#   Packard 1994, Systems & Control Letters 22, 79 — LPV gain-scheduling.
#   Walker et al. 2006, Fusion Eng. Des. 81, 1927 — DIII-D PCS,
#     operating-point-dependent gains indexed by I_p, β_N, l_i.

# Transition time for bumpless gain interpolation [s].
# Shorter than any expected L→H dwell; avoids step transients.
# Walker et al. 2006, §3.2: inter-regime transitions ≲ 0.5 s on DIII-D PCS.
_TAU_SWITCH: float = 0.5  # s

# Minimum denominator guard for derivative term.
_DT_EPS: float = 1e-6  # s


def _finite_vector(name: str, value: AnyFloatArray, size: int | None = None) -> FloatArray:
    """Return a finite vector with the expected state dimension."""
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0 or (size is not None and array.size != size):
        raise ValueError(f"{name} must be a non-empty vector of the expected size")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array.copy()


def _finite_scalar(name: str, value: float, *, positive: bool = False) -> float:
    """Return a finite scalar within its required domain."""
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


class OperatingRegime(Enum):
    """Discharge operating regimes for gain scheduling."""

    RAMP_UP = auto()
    L_MODE_FLAT = auto()
    LH_TRANSITION = auto()
    H_MODE_FLAT = auto()
    RAMP_DOWN = auto()
    DISRUPTION_MITIGATION = auto()


@dataclass
class RegimeController:
    """PID gains and reference for one operating regime.

    Gain matrix design follows Packard 1994 LPV framework:
    the scheduling variable θ = (I_p, β_N, l_i) parameterises
    a family of locally-stabilising gains.
    """

    regime: OperatingRegime
    Kp: AnyFloatArray
    Ki: AnyFloatArray
    Kd: AnyFloatArray
    x_ref: AnyFloatArray
    constraints: dict[str, Any]


class RegimeDetector:
    """Classify the current tokamak operating regime.

    Hysteresis filter (window length 5) prevents spurious switching.
    Walker et al. 2006, §3.1: regime detection on DIII-D PCS uses
    dI_p/dt thresholds and confinement-factor jumps.

    Thresholds (defaults):
        ramp_rate     0.1 MA/s   — ITER ramp specification (Doyle et al. 2007,
                                   Nucl. Fusion 47, S18, Table IV)
        tau_e_jump    1.5×        — H-mode enhancement factor H_98 ≥ 1.5
                                   (ITER Physics Basis 1999, Nucl. Fusion 39, 2175)
        disruption_prob  0.8      — conservative threshold before mitigation
    """

    # History length for hysteresis filter (steps)
    _HISTORY_LEN: int = 5

    def __init__(self, thresholds: dict[str, float] | None = None) -> None:
        self.thresholds = thresholds or {
            "ramp_rate": 0.1,  # MA/s
            "tau_e_L_mode": 1.0,  # s
            "tau_e_jump": 1.5,  # dimensionless H-mode factor
            "disruption_prob": 0.8,  # dimensionless
        }
        self.history: list[OperatingRegime] = []

    def detect(
        self,
        state: AnyFloatArray,
        dstate_dt: AnyFloatArray,
        tau_E: float,
        p_disrupt: float,
    ) -> OperatingRegime:
        """Classify regime from (state, dstate/dt, τ_E, p_disrupt).

        ``state`` and ``dstate_dt`` must be finite, equal-length vectors;
        ``tau_E`` must be positive and ``p_disrupt`` must be in [0, 1].
        Invalid diagnostics do not advance the hysteresis history.
        """
        state = _finite_vector("state", state)
        rate = _finite_vector("dstate_dt", dstate_dt, state.size)
        tau_E = _finite_scalar("tau_E", tau_E, positive=True)
        p_disrupt = _finite_scalar("p_disrupt", p_disrupt)
        if not 0.0 <= p_disrupt <= 1.0:
            raise ValueError("p_disrupt must be within [0, 1]")
        dIp_dt = rate[0]

        if p_disrupt > self.thresholds["disruption_prob"]:
            new_reg = OperatingRegime.DISRUPTION_MITIGATION
        elif dIp_dt > self.thresholds["ramp_rate"]:
            new_reg = OperatingRegime.RAMP_UP
        elif dIp_dt < -self.thresholds["ramp_rate"]:
            new_reg = OperatingRegime.RAMP_DOWN
        else:
            if tau_E > self.thresholds["tau_e_jump"] * self.thresholds["tau_e_L_mode"]:
                new_reg = OperatingRegime.H_MODE_FLAT
            else:
                new_reg = OperatingRegime.L_MODE_FLAT

        self.history.append(new_reg)
        if len(self.history) > self._HISTORY_LEN:
            self.history.pop(0)

        if self.history.count(new_reg) == self._HISTORY_LEN:
            return new_reg
        if len(set(self.history)) == 1:
            return self.history[0]
        return self.history[0] if self.history else new_reg


class GainScheduledController:
    """Multi-regime PID controller with bumpless gain interpolation.

    Scheduling approach: Rugh & Shamma 2000, Automatica 36, 1401, §3.
    Bumpless transfer via linear interpolation over _TAU_SWITCH seconds
    avoids step transients at regime boundaries (Walker et al. 2006, §3.2).

    LPV interpretation: gains are piecewise-affine in the scheduling vector
    (I_p, β_N, l_i) per Packard 1994, Systems & Control Letters 22, 79.
    """

    def __init__(self, controllers: dict[OperatingRegime, RegimeController]) -> None:
        if OperatingRegime.RAMP_UP not in controllers:
            raise ValueError("RAMP_UP controller is required")
        self.controllers = dict(controllers)
        self.current_regime = OperatingRegime.RAMP_UP
        self.prev_regime = OperatingRegime.RAMP_UP

        initial = self.controllers[self.current_regime]
        x_ref = _finite_vector("x_ref", initial.x_ref)
        self.Kp = _finite_vector("Kp", initial.Kp, x_ref.size)
        self.Ki = _finite_vector("Ki", initial.Ki, x_ref.size)
        self.Kd = _finite_vector("Kd", initial.Kd, x_ref.size)

        self.integral_error = np.zeros_like(x_ref)
        self.prev_error = np.zeros_like(self.integral_error)

        self.switch_time = -1.0
        self.tau_switch = _TAU_SWITCH

    def step(
        self,
        x: AnyFloatArray,
        t: float,
        dt: float,
        detected_regime: OperatingRegime,
    ) -> FloatArray:
        """Compute a finite PID candidate with atomic state publication.

        On regime switch: α = (t - t_switch) / τ_switch ∈ [0,1].
        Gains interpolated linearly: K(α) = (1-α) K_old + α K_new.
        Walker et al. 2006, §3.2, Eq. (4). Gain interpolation alone does
        not establish a bumpless actuator output or facility qualification.
        """
        x = _finite_vector("x", x, self.integral_error.size)
        t = _finite_scalar("t", t)
        dt = _finite_scalar("dt", dt, positive=True)
        tau_switch = _finite_scalar("tau_switch", self.tau_switch, positive=True)
        integral = _finite_vector("integral_error", self.integral_error, x.size)
        prev_error = _finite_vector("prev_error", self.prev_error, x.size)
        if detected_regime not in self.controllers:
            raise ValueError("detected regime has no controller")

        switching = detected_regime != self.current_regime
        current_regime = detected_regime if switching else self.current_regime
        prev_regime = self.current_regime if switching else self.prev_regime
        switch_time = t if switching else self.switch_time
        if switching and detected_regime == OperatingRegime.DISRUPTION_MITIGATION:
            integral.fill(0.0)

        ctrl_new = self.controllers[current_regime]
        x_ref_new = _finite_vector("x_ref", ctrl_new.x_ref, x.size)
        kp_new = _finite_vector("Kp", ctrl_new.Kp, x.size)
        ki_new = _finite_vector("Ki", ctrl_new.Ki, x.size)
        kd_new = _finite_vector("Kd", ctrl_new.Kd, x.size)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if switch_time >= 0 and t - switch_time < tau_switch:
                alpha = _finite_scalar("switch fraction", (t - switch_time) / tau_switch)
                if not 0.0 <= alpha <= 1.0:
                    raise ValueError("switch fraction must be within [0, 1]")
                ctrl_old = self.controllers[prev_regime]
                x_ref_old = _finite_vector("x_ref", ctrl_old.x_ref, x.size)
                kp_old = _finite_vector("Kp", ctrl_old.Kp, x.size)
                ki_old = _finite_vector("Ki", ctrl_old.Ki, x.size)
                kd_old = _finite_vector("Kd", ctrl_old.Kd, x.size)
                kp = (1 - alpha) * kp_old + alpha * kp_new
                ki = (1 - alpha) * ki_old + alpha * ki_new
                kd = (1 - alpha) * kd_old + alpha * kd_new
                x_ref = (1 - alpha) * x_ref_old + alpha * x_ref_new
            else:
                kp, ki, kd, x_ref = kp_new, ki_new, kd_new, x_ref_new
            kp = _finite_vector("Kp", kp, x.size)
            ki = _finite_vector("Ki", ki, x.size)
            kd = _finite_vector("Kd", kd, x.size)
            x_ref = _finite_vector("x_ref", x_ref, x.size)
            error = _finite_vector("error", x_ref - x, x.size)
            integral = _finite_vector("integral_error", integral + error * dt, x.size)
            derror = _finite_vector("derivative error", (error - prev_error) / max(dt, _DT_EPS), x.size)
            output = _finite_vector("control output", kp * error + ki * integral + kd * derror, x.size)

        self.current_regime = current_regime
        self.prev_regime = prev_regime
        self.switch_time = switch_time
        self.Kp, self.Ki, self.Kd = kp, ki, kd
        self.integral_error = integral
        self.prev_error = error
        return output

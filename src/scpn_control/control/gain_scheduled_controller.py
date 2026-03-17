# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: protoscience@anulum.li
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

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


class OperatingRegime(Enum):
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
    Kp: np.ndarray
    Ki: np.ndarray
    Kd: np.ndarray
    x_ref: np.ndarray
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
        state: np.ndarray,
        dstate_dt: np.ndarray,
        tau_E: float,
        p_disrupt: float,
    ) -> OperatingRegime:
        """Classify regime from (state, dstate/dt, τ_E, p_disrupt).

        state  = [I_p [MA], β_N, ...]
        dstate = [dI_p/dt [MA/s], dβ_N/dt, ...]
        """
        dIp_dt = dstate_dt[0]

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
        self.controllers = controllers
        self.current_regime = OperatingRegime.RAMP_UP
        self.prev_regime = OperatingRegime.RAMP_UP

        self.Kp = self.controllers[self.current_regime].Kp.copy()
        self.Ki = self.controllers[self.current_regime].Ki.copy()
        self.Kd = self.controllers[self.current_regime].Kd.copy()

        self.integral_error = np.zeros_like(self.controllers[self.current_regime].x_ref)
        self.prev_error = np.zeros_like(self.integral_error)

        self.switch_time = -1.0
        self.tau_switch = _TAU_SWITCH

    def step(
        self,
        x: np.ndarray,
        t: float,
        dt: float,
        detected_regime: OperatingRegime,
    ) -> np.ndarray:
        """Compute PID output with bumpless gain interpolation.

        On regime switch: α = (t - t_switch) / τ_switch ∈ [0,1].
        Gains interpolated linearly: K(α) = (1-α) K_old + α K_new.
        Walker et al. 2006, §3.2, Eq. (4).
        """
        if detected_regime != self.current_regime:
            self.prev_regime = self.current_regime
            self.current_regime = detected_regime
            self.switch_time = t

            if detected_regime == OperatingRegime.DISRUPTION_MITIGATION:
                self.integral_error.fill(0.0)

        if self.switch_time >= 0 and t - self.switch_time < self.tau_switch:
            alpha = (t - self.switch_time) / self.tau_switch
            ctrl_old = self.controllers[self.prev_regime]
            ctrl_new = self.controllers[self.current_regime]

            self.Kp = (1 - alpha) * ctrl_old.Kp + alpha * ctrl_new.Kp
            self.Ki = (1 - alpha) * ctrl_old.Ki + alpha * ctrl_new.Ki
            self.Kd = (1 - alpha) * ctrl_old.Kd + alpha * ctrl_new.Kd
            x_ref = (1 - alpha) * ctrl_old.x_ref + alpha * ctrl_new.x_ref
        else:
            ctrl_new = self.controllers[self.current_regime]
            self.Kp = ctrl_new.Kp
            self.Ki = ctrl_new.Ki
            self.Kd = ctrl_new.Kd
            x_ref = ctrl_new.x_ref

        error = x_ref - x
        self.integral_error += error * dt
        derror = (error - self.prev_error) / max(dt, _DT_EPS)

        u = self.Kp * error + self.Ki * self.integral_error + self.Kd * derror
        self.prev_error = error

        return np.asarray(u)


class ScenarioWaveform:
    """Piecewise-linear waveform for a single scenario variable."""

    def __init__(self, name: str, times: np.ndarray, values: np.ndarray, interp_kind: str = "linear") -> None:
        self.name = name
        self.times = times
        self.values = values
        self.interp_kind = interp_kind

    def __call__(self, t: float) -> float:
        return float(np.interp(t, self.times, self.values))


class ScenarioSchedule:
    """Collection of waveforms defining a full discharge scenario."""

    def __init__(self, waveforms: dict[str, ScenarioWaveform]) -> None:
        self.waveforms = waveforms

    def evaluate(self, t: float) -> dict[str, float]:
        return {name: wf(t) for name, wf in self.waveforms.items()}

    def duration(self) -> float:
        if not self.waveforms:
            return 0.0
        return float(max(wf.times[-1] for wf in self.waveforms.values()))

    def validate(self) -> list[str]:
        errors = []
        for name, wf in self.waveforms.items():
            if not np.all(np.diff(wf.times) > 0):
                errors.append(f"Waveform {name} has non-monotonic times.")
        return errors


def iter_baseline_schedule() -> ScenarioSchedule:
    """ITER 15 MA inductive scenario baseline waveform.

    Timing and values follow ITER PCDH v3.1 (Polevoi et al. 2014,
    ITER Report ITR-18-001, §4.1, Table 4-1):
        t=0–10 s   : ramp-up  (I_p 0.5→5 MA)
        t=10–30 s  : ramp-up  (I_p 5→10 MA, auxiliary heating on)
        t=30–60 s  : ramp-up  (I_p 10→15 MA)
        t=60–400 s : flat top (I_p = 15 MA, NBI 33 MW, ECCD 17 MW)
        t=400–430 s: ramp-down start
        t=430–480 s: ramp-down to 2 MA
    """
    times = np.array([0, 10, 30, 60, 400, 430, 480], dtype=float)
    ip_vals = np.array([0.5, 5.0, 10.0, 15.0, 15.0, 10.0, 2.0])  # MA
    p_nbi = np.array([0.0, 0.0, 10.0, 33.0, 33.0, 10.0, 0.0])  # MW
    p_eccd = np.array([0.0, 0.0, 5.0, 17.0, 17.0, 5.0, 0.0])  # MW

    return ScenarioSchedule(
        {
            "Ip": ScenarioWaveform("Ip", times, ip_vals),
            "P_NBI": ScenarioWaveform("P_NBI", times, p_nbi),
            "P_ECCD": ScenarioWaveform("P_ECCD", times, p_eccd),
        }
    )

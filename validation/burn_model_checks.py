# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Bounded burn-model numerical evidence.
"""Check staggered 0D burn histories against explicit numerical criteria.

These checks establish consistency of the declared uniform, equal-temperature
closure. They do not validate its species/profile assumptions against a facility.
Design beta values never enter the checks or the pressure conversion.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

NUMERICAL_LIMITS = {
    "energy_balance_relative_error": 1e-10,
    "heat_capacity_relative_error": 1e-10,
    "confinement_scaling_relative_error": 1e-10,
    "radiation_relative_error": 1e-10,
    "loss_power_relative_error": 1e-10,
    "time_alignment_error_s": 1e-9,
    "terminal_power_imbalance_relative": 1e-3,
    "terminal_state_relative_drift": 1e-3,
    "timestep_refinement_relative_error": 1e-3,
}
CLOSURE_COUNTS = (
    "temperature_cap_events",
    "temperature_lower_clip_events",
    "q_cap_events",
    "ash_clip_events",
    "energy_floor_events",
)


@dataclass(frozen=True)
class BurnPoint:
    """Uniform DT operating inputs with explicit units and unmeasured defaults.

    Lengths are metres, field tesla, current MA, density 1e20 m^-3 and
    auxiliary power MW. ``mass_amu`` and ``z_eff`` default to 2.5 and 1.65;
    the carrier's H98 and Q are reference annotations, not model inputs.
    """

    r_m: float
    a_m: float
    b_t: float
    ip_ma: float
    kappa: float
    ne20: float
    p_aux_mw: float
    mass_amu: float = 2.5
    z_eff: float = 1.65

    def __post_init__(self) -> None:
        """Refuse nonfinite/nonpositive inputs and negative impurity radiation."""
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.z_eff < 1:
            raise ValueError("z_eff must be at least one")

    @property
    def volume_m3(self) -> float:
        """Return the declared elongated-torus volume in cubic metres."""
        return 2 * math.pi**2 * self.r_m * self.a_m**2 * self.kappa

    def beta_n(self, w_mj: float) -> float:
        """Convert total electron+ion MJ to volume-averaged beta_N without fitting.

        For this equal-temperature quasineutral closure, p_total=2W/(3V).
        beta_N=100*(2*mu0*p_total/B_t^2)*a*B_t/Ip(MA).
        """
        if isinstance(w_mj, bool) or not math.isfinite(w_mj) or w_mj < 0:
            raise ValueError("total stored energy must be finite and nonnegative")
        pressure = 2 * w_mj * 1e6 / (3 * self.volume_m3)
        return 100 * (2 * (4 * math.pi * 1e-7) * pressure / self.b_t**2) * self.a_m * self.b_t / self.ip_ma


def _history(result: Mapping[str, object], name: str, steps: int) -> NDArray[np.float64]:
    """Require one finite, fully sampled provider history without censoring rows."""
    values = np.asarray(result[name], dtype=np.float64)
    if values.ndim != 1 or len(values) != steps or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain {steps} finite samples")
    return values


def check_history(result: Mapping[str, object], point: BurnPoint, duration_s: float, dt_s: float) -> dict[str, Any]:
    """Reconstruct energy balance and active closures with correct time staggering.

    Temperature and powers are pre-step samples; stored energy is post-step.
    The independent radiation/transport reconstruction uses the provider's
    declared 0D closure. The last ten seconds assess terminal power balance.
    Nonfinite/mis-sized histories raise ValueError; failures are never discarded.
    """
    if not math.isfinite(duration_s) or not math.isfinite(dt_s) or duration_s <= 0 or dt_s <= 0:
        raise ValueError("at least two positive finite time steps are required")
    steps = int(duration_s / dt_s)
    if steps < 2:
        raise ValueError("at least two positive finite time steps are required")
    names = ("time_s", "T_keV", "f_he", "W_MJ", "P_alpha_MW", "P_fus_MW", "P_loss_MW", "P_rad_MW", "tau_E_s", "Q")
    h = {name: _history(result, name, steps) for name in names}
    t = h["T_keV"]
    tau = h["tau_E_s"]
    if (t <= 0).any() or (tau <= 0).any():
        raise ValueError("temperature and confinement time must be positive")
    ne = point.ne20 * 1e20
    heat_capacity = 3 * ne * 1e3 * 1.602e-19 * point.volume_m3
    before = heat_capacity * t
    after = h["W_MJ"] * 1e6
    radiation = (
        5.35e-37 * point.z_eff * ne**2 * np.sqrt(np.maximum(t, 0.1)) + 1e-37 * (point.z_eff - 1) * ne**2
    ) * point.volume_m3
    loss = before / np.maximum(tau, 0.01) + radiation
    heating = (h["P_alpha_MW"] + point.p_aux_mw) * 1e6
    expected_tau = np.maximum(
        0.0562
        * point.ip_ma**0.93
        * point.b_t**0.15
        * (point.ne20 * 10) ** 0.41
        * np.maximum(heating / 1e6, 0.1) ** -0.69
        * point.r_m**1.97
        * (point.a_m / point.r_m) ** 0.58
        * point.kappa**0.78
        * point.mass_amu**0.19,
        0.01,
    )
    reconstructed = before + dt_s * (heating - loss)
    scale = np.maximum(np.maximum(np.abs(before), np.abs(after)), 1.0)
    raw_temperature = after / heat_capacity
    raw_q = h["P_fus_MW"] / point.p_aux_mw
    reactions = h["P_fus_MW"] * 1e6 / (17.6e6 * 1.602e-19)
    raw_ash = h["f_he"] + dt_s * (reactions / (ne * point.volume_m3) - 0.8 * h["f_he"] / (5 * tau))
    window = max(1, min(steps, int(10 / dt_s)))
    checks: dict[str, Any] = {
        "energy_balance_relative_error": float(np.max(np.abs(after - reconstructed) / scale)),
        "heat_capacity_relative_error": float(np.max(np.abs(before[1:] - after[:-1]) / scale[1:])),
        "confinement_scaling_relative_error": float(np.max(np.abs(tau - expected_tau) / expected_tau)),
        "radiation_relative_error": float(
            np.max(np.abs(h["P_rad_MW"] * 1e6 - radiation) / np.maximum(np.abs(radiation), 1.0))
        ),
        "loss_power_relative_error": float(np.max(np.abs(h["P_loss_MW"] * 1e6 - loss) / np.maximum(np.abs(loss), 1.0))),
        "time_alignment_error_s": float(np.max(np.abs(h["time_s"] - np.arange(steps) * dt_s))),
        "terminal_power_imbalance_relative": float(
            np.max((np.abs(heating - loss) / np.maximum(np.maximum(np.abs(heating), np.abs(loss)), 1.0))[-window:])
        ),
        "terminal_state_relative_drift": max(
            float(np.ptp(h[name][-window:]) / max(float(np.max(np.abs(h[name][-window:]))), 1e-12))
            for name in ("T_keV", "W_MJ", "f_he", "P_fus_MW")
        ),
        "temperature_cap_events": int(np.count_nonzero(raw_temperature > 25)),
        "temperature_lower_clip_events": int(np.count_nonzero(raw_temperature < 0.1)),
        "q_cap_events": int(np.count_nonzero(raw_q > 15)),
        "ash_clip_events": int(np.count_nonzero((raw_ash < 0) | (raw_ash > 0.5))),
        "energy_floor_events": int(np.count_nonzero(reconstructed < 1e3)),
        "duration_s": duration_s,
        "dt_s": dt_s,
        "steps": steps,
        "beta_energy_time_s": duration_s,
        "power_sample_time_s": float(h["time_s"][-1]),
    }
    declared = result.get("temperature_cap_events")
    if type(declared) is not int or declared != checks["temperature_cap_events"]:
        raise ValueError("provider temperature-cap count disagrees with reconstructed history")
    return checks


def evidence_failures(checks: object) -> list[str]:
    """Refuse missing, coerced, nonfinite, inaccurate or closure-affected evidence.

    Fixed numerical tolerances concern discretisation/consistency. They are
    independent of the informational ITER/SPARC design beta targets.
    """
    if not isinstance(checks, dict):
        return ["model consistency checks must be an object"]
    failures = []
    for key, limit in NUMERICAL_LIMITS.items():
        value = checks.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            failures.append(f"{key}: finite nonnegative numerical evidence required")
            continue
        try:
            valid = math.isfinite(value) and 0 <= value <= limit
        except OverflowError:
            valid = False
        if not valid:
            failures.append(f"{key}: outside numerical limit {limit}")
    for key in CLOSURE_COUNTS:
        if type(checks.get(key)) is not int or checks[key] != 0:
            failures.append(f"{key}: inactive closure required")
    return failures

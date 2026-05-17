# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Dashboard state preparation
# © 1998–2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────
"""Streamlit-independent state preparation for the control dashboard."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


MachinePreset = dict[str, float | str]
ProfileMap = dict[str, NDArray[np.float64]]


MACHINE_PRESETS: dict[str, MachinePreset] = {
    "DIII-D": {
        "R0": 1.67,
        "a": 0.67,
        "B0": 2.2,
        "Ip": 1.5,
        "kappa": 1.8,
        "delta": 0.35,
        "q95": 3.5,
        "ne_1e19": 5.0,
        "Te0_keV": 3.5,
        "P_aux_MW": 15.0,
        "description": "GA general-purpose tokamak, H-mode reference",
    },
    "SPARC": {
        "R0": 1.85,
        "a": 0.57,
        "B0": 12.2,
        "Ip": 8.7,
        "kappa": 1.97,
        "delta": 0.54,
        "q95": 3.4,
        "ne_1e19": 37.0,
        "Te0_keV": 21.0,
        "P_aux_MW": 25.0,
        "description": "CFS compact high-field, Q > 2 target",
    },
    "ITER": {
        "R0": 6.2,
        "a": 2.0,
        "B0": 5.3,
        "Ip": 15.0,
        "kappa": 1.7,
        "delta": 0.33,
        "q95": 3.0,
        "ne_1e19": 10.1,
        "Te0_keV": 25.0,
        "P_aux_MW": 50.0,
        "description": "ITER 15 MA baseline scenario, Q = 10",
    },
    "NSTX-U": {
        "R0": 0.93,
        "a": 0.63,
        "B0": 1.0,
        "Ip": 2.0,
        "kappa": 2.8,
        "delta": 0.6,
        "q95": 8.0,
        "ne_1e19": 6.0,
        "Te0_keV": 1.5,
        "P_aux_MW": 12.0,
        "description": "PPPL spherical tokamak, high beta, strong shaping",
    },
    "JET": {
        "R0": 2.96,
        "a": 1.25,
        "B0": 3.45,
        "Ip": 3.5,
        "kappa": 1.68,
        "delta": 0.27,
        "q95": 3.3,
        "ne_1e19": 8.0,
        "Te0_keV": 8.0,
        "P_aux_MW": 30.0,
        "description": "UKAEA large tokamak, DT record holder",
    },
}


def _float(machine: MachinePreset, key: str) -> float:
    value = machine[key]
    if isinstance(value, str):
        raise ValueError(f"{key} must be numeric.")
    return float(value)


def shot_phase_label(frac: float) -> str:
    """Map normalised time fraction to shot phase name."""
    if not np.isfinite(frac) or not 0.0 <= frac <= 1.0:
        raise ValueError("frac must be finite and within [0, 1].")
    if frac < 0.10:
        return "STARTUP"
    if frac < 0.25:
        return "RAMP-UP"
    if frac < 0.75:
        return "FLATTOP"
    if frac < 0.90:
        return "RAMP-DOWN"
    return "TERMINATION"


def derived_machine_metrics(machine: MachinePreset) -> dict[str, float]:
    """Compute dashboard derived quantities from a machine preset."""
    r0 = _float(machine, "R0")
    a = _float(machine, "a")
    ip = _float(machine, "Ip")
    if r0 <= 0.0 or a <= 0.0 or ip <= 0.0:
        raise ValueError("R0, a, and Ip must be positive.")
    return {
        "epsilon": a / r0,
        "aspect_ratio": r0 / a,
        "b_pol_t": ip * 0.4 / a,
    }


def _time_amplitude(time_frac: float) -> float:
    if not np.isfinite(time_frac) or not 0.0 <= time_frac <= 1.0:
        raise ValueError("time_frac must be finite and within [0, 1].")
    if time_frac < 0.10:
        amp = 0.2 + 8.0 * time_frac
    elif time_frac < 0.25:
        amp = 0.6 + 2.67 * (time_frac - 0.10)
    elif time_frac < 0.75:
        amp = 1.0
    elif time_frac < 0.90:
        amp = 1.0 - 2.67 * (time_frac - 0.75)
    else:
        amp = 0.6 - 6.0 * (time_frac - 0.90)
    return max(amp, 0.05)


def synthetic_profiles(machine: MachinePreset, n_rho: int = 50, time_frac: float = 0.5) -> ProfileMap:
    """Generate finite pedestal-shaped profiles for dashboard shot replay."""
    if isinstance(n_rho, bool) or int(n_rho) < 2:
        raise ValueError("n_rho must be an integer >= 2.")
    rho = np.linspace(0.0, 1.0, int(n_rho), dtype=np.float64)
    pedestal = 0.5 * (1.0 + np.tanh((0.9 - rho) / 0.05))
    amp = _time_amplitude(float(time_frac))

    te = amp * _float(machine, "Te0_keV") * (0.1 + 0.9 * pedestal)
    ti = te * 0.9
    ne = _float(machine, "ne_1e19") * (0.3 + 0.7 * pedestal) * amp

    return {
        "rho": rho,
        "Te_keV": np.asarray(te, dtype=np.float64),
        "Ti_keV": np.asarray(ti, dtype=np.float64),
        "ne_1e19": np.asarray(ne, dtype=np.float64),
    }


__all__ = [
    "MACHINE_PRESETS",
    "MachinePreset",
    "ProfileMap",
    "derived_machine_metrics",
    "shot_phase_label",
    "synthetic_profiles",
]

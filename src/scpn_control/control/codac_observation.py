# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CODAC controller observation admission.
"""Validate controller-consumed CODAC axis channels before actuation."""

from __future__ import annotations

import math
from typing import Mapping

from scpn_control.scpn.contracts import ControlObservation

_AXIS_LIMITS: dict[str, tuple[float, float]] = {
    "R_axis": (4.0, 8.0),
    "Z_axis": (-2.0, 2.0),
}


def _axis_value(pv_values: Mapping[str, float], key: str) -> float:
    """Resolve one axis from declared names and reject ambiguous or invalid PVs."""
    alias = f"{key}_m"
    names = [name for name in (key, alias) if name in pv_values]
    if not names:
        raise ValueError(f"CODAC {key} observation is missing")
    values: list[float] = []
    low, high = _AXIS_LIMITS[key]
    for name in names:
        raw = pv_values[name]
        if isinstance(raw, bool):
            raise ValueError(f"CODAC {name} observation must be a finite number")
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"CODAC {name} observation must be a finite number") from exc
        if not math.isfinite(value) or value < low or value > high:
            raise ValueError(f"CODAC {name} observation must be within [{low}, {high}]")
        values.append(value)
    if len(values) == 2 and values[0] != values[1]:
        raise ValueError(f"CODAC {key} observation aliases disagree")
    return values[0]


def pack_codac_observation(pv_values: Mapping[str, float]) -> ControlObservation:
    """Build the controller observation from valid CODAC axis measurements."""
    return ControlObservation(
        R_axis_m=_axis_value(pv_values, "R_axis"),
        Z_axis_m=_axis_value(pv_values, "Z_axis"),
    )

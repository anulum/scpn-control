#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Digital twin grid and actuator domains

"""Check original finite grid, actuator and unit declarations without physical replay."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

_REQUIRED_GRID_FIELDS = ("grid_size", "time_steps", "seed", "state_variables", "has_ids_export")
_REQUIRED_ACTUATOR_FIELDS = (
    "actuator_tau_steps",
    "actuator_rate_limit",
    "actuator_bias",
    "sensor_dropout_prob",
    "sensor_noise_std",
)
_REQUIRED_UNITS = {
    "temperature": "keV",
    "density": "m^-3",
    "q_profile": "1",
    "actuator_action": "1",
    "time": "step",
    "ids_pulse": "1",
}


def _valid_grid_metadata(value: object) -> bool:
    """Require uncapped nonboolean grid>=4, positive steps, nonnegative seed, nonempty list of nonblank state names and a boolean IDS flag, admitting duplicates and unknown names."""
    if not isinstance(value, dict) or not all(field in value for field in _REQUIRED_GRID_FIELDS):
        return False
    grid_size = value.get("grid_size")
    time_steps = value.get("time_steps")
    seed = value.get("seed")
    state_variables = value.get("state_variables")
    has_ids_export = value.get("has_ids_export")
    return (
        not isinstance(grid_size, bool)
        and isinstance(grid_size, int)
        and grid_size >= 4
        and not isinstance(time_steps, bool)
        and isinstance(time_steps, int)
        and time_steps > 0
        and not isinstance(seed, bool)
        and isinstance(seed, int)
        and seed >= 0
        and isinstance(state_variables, list)
        and bool(state_variables)
        and all(isinstance(item, str) and item.strip() for item in state_variables)
        and isinstance(has_ids_export, bool)
    )


def _valid_actuator_metadata(value: object) -> bool:
    """Require uncapped nonnegative integer lag, finite signed bias, nonnegative finite rate/noise and dropout in [0,1], without calibration or actuator simulation."""
    if not isinstance(value, dict) or not all(field in value for field in _REQUIRED_ACTUATOR_FIELDS):
        return False
    tau_steps = value.get("actuator_tau_steps")
    if isinstance(tau_steps, bool) or not isinstance(tau_steps, int) or tau_steps < 0:
        return False
    return (
        _is_finite_number(value.get("actuator_bias"))
        and _is_nonnegative_finite(value.get("actuator_rate_limit"))
        and _is_finite_number(value.get("sensor_dropout_prob"))
        and 0.0 <= float(value["sensor_dropout_prob"]) <= 1.0
        and _is_nonnegative_finite(value.get("sensor_noise_std"))
    )


def _valid_units(value: object) -> bool:
    """Require original keV, m^-3, dimensionless q/action/IDS and step labels; admit extras without conversion."""
    return isinstance(value, dict) and all(value.get(field) == unit for field, unit in _REQUIRED_UNITS.items())


def _has_public_reference(payload: dict[str, object]) -> bool:
    """Require nonblank URL or DOI presence without parsing or verification."""
    return any(_has_nonempty_str(payload, field) for field in ("reference_url", "reference_doi"))


def _has_nonempty_str(payload: dict[str, object], field: str) -> bool:
    """Accept original nonblank strings without parsing references."""
    value = payload.get(field)
    return isinstance(value, str) and bool(value.strip())


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    """Admit nonboolean int/float whose binary64 conversion is finite; refuse overflow."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _is_nonnegative_finite(value: object) -> TypeGuard[int | float]:
    """Require finite numbers at least zero."""
    return _is_finite_number(value) and float(value) >= 0.0


def _is_positive_finite(value: object) -> TypeGuard[int | float]:
    """Require finite numbers strictly above zero."""
    return _is_finite_number(value) and float(value) > 0.0

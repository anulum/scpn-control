# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — SOC reference artifact validator

"""Retain SOC lattice, Q-learning metadata and declared numerical domains without executing learning."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

_REQUIRED_LATTICE_FIELDS = (
    "size",
    "time_steps",
    "seed",
    "z_crit_base",
    "flow_generation",
    "flow_damping",
    "shear_efficiency",
    "max_sub_steps",
)

_REQUIRED_LEARNING_FIELDS = (
    "alpha",
    "gamma",
    "epsilon",
    "n_states_turb",
    "n_states_flow",
    "n_actions",
    "reward_definition",
)

_REQUIRED_UNITS = {
    "lattice_gradient": "1",
    "flow": "1",
    "shear": "1",
    "topple_count": "1",
    "q_value": "1",
    "reward": "1",
    "time": "step",
}


def _valid_lattice_metadata(value: object) -> bool:
    """Require original uncapped integer lattice counts, nonnegative declared couplings and damping below one; z_crit_base zero remains metadata-valid but is refused by the separate runtime constructor."""
    if not isinstance(value, dict) or not all(field in value for field in _REQUIRED_LATTICE_FIELDS):
        return False
    size = value.get("size")
    time_steps = value.get("time_steps")
    seed = value.get("seed")
    max_sub_steps = value.get("max_sub_steps")
    if any(isinstance(item, bool) for item in (size, time_steps, seed, max_sub_steps)):
        return False
    if not (isinstance(size, int) and size >= 8):
        return False
    if not (isinstance(time_steps, int) and time_steps > 0):
        return False
    if not (isinstance(seed, int) and seed >= 0):
        return False
    if not (isinstance(max_sub_steps, int) and max_sub_steps > 0):
        return False
    if not all(
        _is_nonnegative_finite(value.get(field)) for field in ("z_crit_base", "flow_generation", "shear_efficiency")
    ):
        return False
    flow_damping = value.get("flow_damping")
    return _is_finite_number(flow_damping) and 0.0 <= float(flow_damping) < 1.0


def _valid_learning_metadata(value: object) -> bool:
    """Require inclusive fractional alpha/gamma/epsilon, positive uncapped nonboolean state/action counts and nonblank reward text without fitting a policy."""
    if not isinstance(value, dict) or not all(field in value for field in _REQUIRED_LEARNING_FIELDS):
        return False
    if not all(_is_fraction(value.get(field)) for field in ("alpha", "gamma", "epsilon")):
        return False
    for field in ("n_states_turb", "n_states_flow", "n_actions"):
        item = value.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            return False
    reward_definition = value.get("reward_definition")
    return isinstance(reward_definition, str) and bool(reward_definition.strip())


def _valid_units(value: object) -> bool:
    """Require six dimensionless SOC labels and time in steps, without converting quantities."""
    return isinstance(value, dict) and all(value.get(field) == unit for field, unit in _REQUIRED_UNITS.items())


def _has_public_reference(payload: dict[str, object]) -> bool:
    """Require nonblank public URL or DOI presence without validating citation syntax."""
    return any(_has_nonempty_str(payload, field) for field in ("reference_url", "reference_doi"))


def _has_nonempty_str(payload: dict[str, object], field: str) -> bool:
    """Require a nonblank string declaration without interpreting its content."""
    value = payload.get(field)
    return isinstance(value, str) and bool(value.strip())


def _is_fraction(value: object) -> bool:
    """Require an admitted finite scalar in the inclusive unit interval."""
    return _is_finite_number(value) and 0.0 <= float(value) <= 1.0


def _is_nonnegative_finite(value: object) -> TypeGuard[int | float]:
    """Require an admitted finite declared error or coupling at least zero."""
    return _is_finite_number(value) and float(value) >= 0.0


def _is_positive_finite(value: object) -> TypeGuard[int | float]:
    """Require an admitted finite declared error bound strictly greater than zero."""
    return _is_finite_number(value) and float(value) > 0.0


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    """Admit nonboolean int/float whose binary64 conversion is finite, refusing overflow."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False

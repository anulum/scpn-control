# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Static structured-mu reference artifact validator

"""Retain static mu unit labels and uncapped positive nonboolean plant dimensions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

_REQUIRED_PLANT_FIELDS = ("state_dimension", "control_dimension", "output_dimension", "uncertainty_total_size")

_REQUIRED_UNITS = {
    "mu": "1",
    "robustness_margin": "1",
    "controller_gain": "1",
    "d_scaling": "1",
    "spectral_abscissa": "s^-1",
}


def _valid_plant_metadata(value: object) -> bool:
    """Require original four plant dimensions as positive uncapped nonboolean integers, without allocating matrices."""
    if not isinstance(value, dict):
        return False
    return all(
        isinstance(v := value.get(field), int) and not isinstance(v, bool) and v > 0 for field in _REQUIRED_PLANT_FIELDS
    )


def _valid_units(value: object) -> bool:
    """Require five original unit labels; historical robustness_margin is the reciprocal static zero-frequency upper bound, not frequency-domain stability."""
    return isinstance(value, dict) and all(value.get(field) == unit for field, unit in _REQUIRED_UNITS.items())


def _has_public_reference(payload: dict[str, object]) -> bool:
    """Require nonblank URL or DOI text without citation syntax or authenticity checks."""
    return any(_has_nonempty_str(payload, field) for field in ("reference_url", "reference_doi"))


def _has_nonempty_str(payload: dict[str, object], field: str) -> bool:
    """Require original nonblank text presence without interpreting content."""
    value = payload.get(field)
    return isinstance(value, str) and bool(value.strip())


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

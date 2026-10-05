# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Current-drive reference artifact validator

"""Preserve declared current-drive unit, scalar, source-grid and inclusive metric/tolerance predicates.

These metadata checks recompute no deposition, current or physical reference.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

_REQUIRED_UNITS = {
    "power": "W",
    "current": "A",
    "current_density": "A/m^2",
    "density": "10^19 m^-3",
    "temperature": "keV",
    "rho": "1",
    "time": "s",
    "energy": "keV",
}


_REQUIRED_SOURCE_FIELDS = ("total_power_W", "rho_points", "rho_min", "rho_max")


_MAXIMUM_ERROR_METRICS = (
    "total_power_relative_error",
    "total_current_relative_error",
    "deposition_centroid_abs_error",
    "peak_current_density_relative_error",
    "nbi_slowing_down_relative_error",
)


def _validate_metric_block(path: Path, metrics: object, tolerances: object, errors: list[dict[str, object]]) -> None:
    """Compare the five declared finite dimensionless errors with inclusive positive tolerances."""
    if not isinstance(metrics, dict):
        errors.append({"path": str(path), "field": "metrics", "error": "metrics must be an object"})
        return
    if not isinstance(tolerances, dict):
        errors.append({"path": str(path), "field": "tolerances", "error": "tolerances must be an object"})
        return
    for field in _MAXIMUM_ERROR_METRICS:
        metric = metrics.get(field)
        tolerance = tolerances.get(field)
        if not _is_nonnegative_finite(metric):
            errors.append({"path": str(path), "field": field, "error": "metric must be finite and non-negative"})
            continue
        if not _is_positive_finite(tolerance):
            errors.append({"path": str(path), "field": field, "error": "tolerance must be finite and positive"})
            continue
        if float(metric) > float(tolerance):
            errors.append({"path": str(path), "field": field, "error": "metric exceeds declared tolerance"})


def _valid_source_metadata(value: object) -> bool:
    """Require positive source scalars and ordered normalised radii inside (0, 1]."""
    if not isinstance(value, dict):
        return False
    if not all(_is_positive_finite(value.get(field)) for field in _REQUIRED_SOURCE_FIELDS):
        return False
    return (
        float(value["rho_min"]) >= 0.0
        and float(value["rho_max"]) <= 1.0
        and float(value["rho_max"]) > float(value["rho_min"])
    )


def _valid_units(value: object) -> bool:
    """Require the eight exact unit labels without conversion or dimension inference."""
    return isinstance(value, dict) and all(value.get(field) == unit for field, unit in _REQUIRED_UNITS.items())


def _has_public_reference(payload: dict[str, object]) -> bool:
    """Accept a nonblank URL or DOI declaration without parsing or retrieval."""
    return any(_has_nonempty_str(payload, field) for field in ("reference_url", "reference_doi"))


def _has_nonempty_str(payload: dict[str, object], field: str) -> bool:
    """Recognise a string containing a non-whitespace character."""
    value = payload.get(field)
    return isinstance(value, str) and bool(value.strip())


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    """Recognise non-boolean real scalars representable as finite Python floats."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _is_nonnegative_finite(value: object) -> TypeGuard[int | float]:
    """Recognise a float-representable non-boolean scalar greater than or equal to zero."""
    return _is_finite_number(value) and float(value) >= 0.0


def _is_positive_finite(value: object) -> TypeGuard[int | float]:
    """Recognise a float-representable non-boolean scalar strictly greater than zero."""
    return _is_finite_number(value) and float(value) > 0.0

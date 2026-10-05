#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — EPED reference artifact validator

"""Check declared EPED grid, shape, domains and tolerances without computing pedestal physics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

_REQUIRED_METRICS = (
    "pedestal_width_relative_error",
    "pedestal_height_relative_error",
    "pressure_limit_relative_error",
    "bootstrap_current_relative_error",
    "collisionality_width_order_error",
)

_REQUIRED_SHAPING_FIELDS = ("kappa", "delta", "R0_m", "a_m")


def _validate_geometry(path: Path, payload: dict[str, object], errors: list[dict[str, object]]) -> None:
    """Append findings for declared grid, width, beta, shape and five relative errors; no EPED solve occurs."""
    if not _strictly_increasing_unit_grid(payload.get("rho_grid")):
        errors.append(
            {
                "path": str(path),
                "field": "rho_grid",
                "error": "rho grid must be finite and strictly increasing on [0, 1]",
            }
        )
    if not _valid_width_range(payload.get("pedestal_width_range_psi_n")):
        errors.append(
            {
                "path": str(path),
                "field": "pedestal_width_range_psi_n",
                "error": "pedestal width range must be positive, ordered, and within (0, 1)",
            }
        )
    if not _positive_ordered_pair(payload.get("beta_limit_range")):
        errors.append(
            {"path": str(path), "field": "beta_limit_range", "error": "beta limit range must be positive and ordered"}
        )
    if not _valid_shaping(payload.get("shaping")):
        errors.append(
            {
                "path": str(path),
                "field": "shaping",
                "error": "shaping must declare finite kappa, delta, R0_m, and a_m with tokamak ordering",
            }
        )
    _validate_metric_block(path, payload.get("metrics"), payload.get("tolerances"), errors)


def _validate_metric_block(path: Path, metrics: object, tolerances: object, errors: list[dict[str, object]]) -> None:
    """Compare five nonnegative declared relative errors to positive bounds, admitting equality; no metric computation."""
    if not isinstance(metrics, dict):
        errors.append({"path": str(path), "field": "metrics", "error": "metrics must be an object"})
        return
    if not isinstance(tolerances, dict):
        errors.append({"path": str(path), "field": "tolerances", "error": "tolerances must be an object"})
        return
    for field in _REQUIRED_METRICS:
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


def _strictly_increasing_unit_grid(value: object) -> bool:
    """Require at least two finite representable numbers strictly increasing in inclusive [0,1]."""
    if not isinstance(value, list | tuple) or len(value) < 2:
        return False
    last = -math.inf
    for item in value:
        if not _is_finite(item):
            return False
        current = float(item)
        if current < 0.0 or current > 1.0 or current <= last:
            return False
        last = current
    return True


def _valid_width_range(value: object) -> bool:
    """Require positive representable width endpoints with lower <= upper <1; equal endpoints remain allowed."""
    if not isinstance(value, list | tuple) or len(value) != 2:
        return False
    lo, hi = value
    if not _is_positive_finite(lo) or not _is_positive_finite(hi):
        return False
    return 0.0 < float(lo) <= float(hi) < 1.0


def _positive_ordered_pair(value: object) -> bool:
    """Require two positive representable beta endpoints with strictly greater upper endpoint."""
    if not isinstance(value, list | tuple) or len(value) != 2:
        return False
    lo, hi = value
    if not _is_positive_finite(lo) or not _is_positive_finite(hi):
        return False
    return float(hi) > float(lo)


def _valid_shaping(value: object) -> bool:
    """Require kappa,R0,a positive and delta finite with abs(delta)<1 and a<R0; allow extra fields."""
    if not isinstance(value, dict):
        return False
    if not all(field in value for field in _REQUIRED_SHAPING_FIELDS):
        return False
    kappa = value.get("kappa")
    delta = value.get("delta")
    r0 = value.get("R0_m")
    minor = value.get("a_m")
    if (
        not _is_positive_finite(kappa)
        or not _is_finite(delta)
        or not _is_positive_finite(r0)
        or not _is_positive_finite(minor)
    ):
        return False
    if abs(float(delta)) >= 1.0:
        return False
    return float(minor) < float(r0)


def _is_finite(value: object) -> TypeGuard[int | float]:
    """Recognize finite binary64-convertible numbers without raising for huge integers; exclude booleans."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _is_nonnegative_finite(value: object) -> TypeGuard[int | float]:
    """Recognize representable finite nonnegative declared errors."""
    return _is_finite(value) and value >= 0


def _is_positive_finite(value: object) -> TypeGuard[int | float]:
    """Recognize representable finite positive bounds and shape dimensions."""
    return _is_finite(value) and value > 0

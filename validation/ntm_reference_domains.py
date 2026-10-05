#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — NTM reference artifact validator

"""Check declared NTM grid, q, surface, seed, ECCD and errors without computing island dynamics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

_REQUIRED_METRICS = (
    "rational_surface_rho_error",
    "island_growth_relative_error",
    "saturated_width_relative_error",
    "suppression_time_relative_error",
    "eccd_alignment_error_m",
)


def _validate_domains(path: Path, payload: dict[str, object], errors: list[dict[str, object]]) -> None:
    """Append findings for declared grid/q/surface/seed/ECCD/error contracts, without solving or authenticating island physics."""
    if not _strictly_increasing_unit_grid(payload.get("rho_grid")):
        errors.append(
            {
                "path": str(path),
                "field": "rho_grid",
                "error": "rho grid must be finite and strictly increasing on [0, 1]",
            }
        )
    if not _valid_q_profile(payload.get("q_profile"), payload.get("rho_grid")):
        errors.append(
            {
                "path": str(path),
                "field": "q_profile",
                "error": "q profile must be finite, positive, and length-matched to rho_grid",
            }
        )
    if not _valid_rational_surface(payload.get("rational_surface")):
        errors.append(
            {
                "path": str(path),
                "field": "rational_surface",
                "error": "rational surface must declare positive m, n, q, r_s_m, finite shear, rho in (0, 1), and tokamak geometry",
            }
        )
    if not _valid_seed_island_range(payload.get("seed_island_width_range_m")):
        errors.append(
            {
                "path": str(path),
                "field": "seed_island_width_range_m",
                "error": "seed island width range must be positive and ordered",
            }
        )
    if not _valid_eccd_alignment(payload.get("eccd_alignment")):
        errors.append(
            {
                "path": str(path),
                "field": "eccd_alignment",
                "error": "ECCD alignment must declare non-negative power/current/alignment error and positive deposition width",
            }
        )
    _validate_metric_block(path, payload.get("metrics"), payload.get("tolerances"), errors)


def _validate_metric_block(path: Path, metrics: object, tolerances: object, errors: list[dict[str, object]]) -> None:
    """Compare five nonnegative declared errors with positive finite bounds, admitting equality; no metric recomputation."""
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
    """Require at least two representable nonnegative strictly increasing values in inclusive [0,1]."""
    if not isinstance(value, list | tuple) or len(value) < 2:
        return False
    last = -math.inf
    for item in value:
        if not _is_nonnegative_finite(item):
            return False
        current = float(item)
        if current > 1.0 or current <= last:
            return False
        last = current
    return True


def _valid_q_profile(value: object, rho_grid: object) -> bool:
    """Require at least two representable positive q values length-matched to declared rho; no interpolation or rational-mode equality."""
    if not isinstance(value, list | tuple) or len(value) < 2:
        return False
    if not isinstance(rho_grid, list | tuple) or len(value) != len(rho_grid):
        return False
    return all(_is_positive_finite(item) for item in value)


def _valid_rational_surface(value: object) -> bool:
    """Require positive nonboolean integer m/n, positive q/r_s/a/R0, finite shear, rho strictly(0,1), a<R0 and r_s<a; no q=m/n or rho-radius calculation."""
    if not isinstance(value, dict):
        return False
    m = value.get("m")
    n = value.get("n")
    rho = value.get("rho")
    r_s = value.get("r_s_m")
    q = value.get("q")
    shear = value.get("shear")
    a = value.get("a_m")
    r0 = value.get("R0_m")
    if isinstance(m, bool) or isinstance(n, bool) or not isinstance(m, int) or not isinstance(n, int):
        return False
    if m <= 0 or n <= 0:
        return False
    if not (
        _is_positive_finite(r_s) and _is_positive_finite(q) and _is_positive_finite(a) and _is_positive_finite(r0)
    ) or not _is_finite(shear):
        return False
    if not (_is_positive_finite(rho) and float(rho) < 1.0):
        return False
    return float(a) < float(r0) and float(r_s) < float(a)


def _valid_seed_island_range(value: object) -> bool:
    """Require two representable positive seed widths with upper >= lower, retaining equal endpoints."""
    if not isinstance(value, list | tuple) or len(value) != 2:
        return False
    lo, hi = value
    if not _is_positive_finite(lo) or not _is_positive_finite(hi):
        return False
    return float(hi) >= float(lo)


def _valid_eccd_alignment(value: object) -> bool:
    """Require representable nonnegative power/current/error and positive deposition width; no ECCD effect computation."""
    if not isinstance(value, dict):
        return False
    power = value.get("power_W")
    current = value.get("current_A")
    width = value.get("deposition_width_m")
    error = value.get("alignment_error_m")
    return (
        _is_nonnegative_finite(power)
        and _is_nonnegative_finite(current)
        and _is_positive_finite(width)
        and _is_nonnegative_finite(error)
    )


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

#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MARFE reference artifact validator

"""Check declared MARFE scans, impurity, geometry, power and errors without computing radiation physics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

_REQUIRED_METRICS = (
    "onset_temperature_relative_error",
    "density_limit_relative_error",
    "greenwald_fraction_error",
    "front_temperature_min_relative_error",
    "radiation_growth_rate_relative_error",
)


def _validate_domains(path: Path, payload: dict[str, object], errors: list[dict[str, object]]) -> None:
    """Append findings for declared scans, impurity fraction, geometry and power without model evaluation."""
    if not _ordered_positive_grid(payload.get("temperature_scan_eV")):
        errors.append(
            {
                "path": str(path),
                "field": "temperature_scan_eV",
                "error": "temperature scan must be finite, positive, and strictly increasing",
            }
        )
    if not _ordered_positive_grid(payload.get("density_scan_m3")):
        errors.append(
            {
                "path": str(path),
                "field": "density_scan_m3",
                "error": "density scan must be finite, positive, and strictly increasing",
            }
        )
    if not _valid_impurity_fraction_range(payload.get("impurity_fraction_range")):
        errors.append(
            {
                "path": str(path),
                "field": "impurity_fraction_range",
                "error": "impurity fraction range must lie inside (0, 1]",
            }
        )
    if not _valid_geometry(payload.get("geometry")):
        errors.append(
            {
                "path": str(path),
                "field": "geometry",
                "error": "geometry must declare finite R0_m, a_m, q95, and connection_length_m with tokamak ordering",
            }
        )
    if not _valid_power_balance(payload.get("power_balance")):
        errors.append(
            {
                "path": str(path),
                "field": "power_balance",
                "error": "power_balance must declare positive P_SOL_W and non-negative q_perp_W_m2",
            }
        )


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


def _ordered_positive_grid(value: object) -> bool:
    """Require at least two finite positive representable values strictly increasing, without an upper unit-grid bound."""
    if not isinstance(value, list | tuple) or len(value) < 2:
        return False
    last = -math.inf
    for item in value:
        if not _is_positive_finite(item):
            return False
        current = float(item)
        if current <= last:
            return False
        last = current
    return True


def _valid_impurity_fraction_range(value: object) -> bool:
    """Require two positive representable endpoints with lower <= upper <=1; equality and upper one remain admitted."""
    if not isinstance(value, list | tuple) or len(value) != 2:
        return False
    lo, hi = value
    if not _is_positive_finite(lo) or not _is_positive_finite(hi):
        return False
    return 0.0 < float(lo) <= float(hi) <= 1.0


def _valid_geometry(value: object) -> bool:
    """Require positive representable R0,a,q95,connection length and a<R0; no field-line geometry is computed."""
    if not isinstance(value, dict):
        return False
    r0 = value.get("R0_m")
    minor = value.get("a_m")
    q95 = value.get("q95")
    connection = value.get("connection_length_m")
    if not (
        _is_positive_finite(r0)
        and _is_positive_finite(minor)
        and _is_positive_finite(q95)
        and _is_positive_finite(connection)
    ):
        return False
    return float(minor) < float(r0)


def _valid_power_balance(value: object) -> bool:
    """Require finite positive declared P_SOL_W and nonnegative q_perp_W_m2; no radiation balance is computed."""
    if not isinstance(value, dict):
        return False
    p_sol = value.get("P_SOL_W")
    q_perp = value.get("q_perp_W_m2")
    return _is_positive_finite(p_sol) and _is_nonnegative_finite(q_perp)


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

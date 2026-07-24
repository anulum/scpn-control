# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Free-boundary tracking limit / tolerance resolvers

"""Pure config resolvers for free-boundary tracking limits and tolerances.

This leaf owns objective-tolerance and supervisor/limit resolution helpers used
by :class:`~scpn_control.control.free_boundary_tracking.FreeBoundaryTrackingController`.
Claims remain in ``free_boundary_tracking_claims`` (must not re-merge). Shot
orchestration stays on the controller (R3-S4); observation and control-law
stages live in sibling leaves.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from scpn_control._typing import FloatArray


def resolve_objective_tolerances(
    cfg_tolerances: Any,
    override_tolerances: dict[str, float] | None,
) -> dict[str, float]:
    """Validate and merge free-boundary objective tolerances for tracking."""
    allowed = {
        "shape_rms",
        "shape_max_abs",
        "x_point_position",
        "x_point_flux",
        "divertor_rms",
        "divertor_max_abs",
    }
    merged: dict[str, float] = {}
    for raw, name in (
        (cfg_tolerances, "free_boundary.objective_tolerances"),
        (override_tolerances, "objective_tolerances"),
    ):
        if raw is None:
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"{name} must be a mapping of tolerance names to non-negative floats.")
        for key, value in raw.items():
            if key not in allowed:
                allowed_keys = ", ".join(sorted(allowed))
                raise ValueError(f"Unknown {name} key {key!r}. Allowed keys: {allowed_keys}.")
            tol_value = float(value)
            if not np.isfinite(tol_value) or tol_value < 0.0:
                raise ValueError(f"{name}.{key} must be finite and >= 0.")
            merged[key] = tol_value
    return merged


def resolve_positive_float(
    cfg_value: Any,
    override_value: float | None,
    *,
    default: float,
    name: str,
) -> float:
    """Resolve a positive finite float from cfg/override/default."""
    raw_value = (
        default
        if override_value is None and cfg_value is None
        else (cfg_value if override_value is None else override_value)
    )
    value = float(raw_value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0.")
    return value


def resolve_nonnegative_int(
    cfg_value: Any,
    override_value: int | None,
    *,
    default: int,
    name: str,
) -> int:
    """Resolve a non-negative integer from cfg/override/default."""
    raw_value = (
        default
        if override_value is None and cfg_value is None
        else (cfg_value if override_value is None else override_value)
    )
    value = int(raw_value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0.")
    return value


def resolve_nonnegative_float(
    cfg_value: Any,
    *,
    default: float,
    name: str,
) -> float:
    """Resolve a non-negative float (infinity allowed) from cfg/default."""
    raw_value = default if cfg_value is None else cfg_value
    value = float(raw_value)
    if not np.isfinite(value) and not np.isinf(value):
        raise ValueError(f"{name} must be finite or infinity.")
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0.")
    return value


def resolve_fraction(
    cfg_value: Any,
    *,
    default: float,
    name: str,
) -> float:
    """Resolve a finite float in ``[0, 1]`` from cfg/default."""
    raw_value = default if cfg_value is None else cfg_value
    value = float(raw_value)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return value


def resolve_coil_slew_limits(
    n_coils: int,
    cfg_limits: Any,
    override_limits: float | list[float] | None,
) -> FloatArray:
    """Resolve per-coil positive finite slew limits (or ``+inf`` default)."""
    raw = cfg_limits if override_limits is None else override_limits
    if raw is None:
        return np.full(n_coils, np.inf, dtype=np.float64)
    if np.isscalar(raw):
        limits = np.full(n_coils, float(cast(Any, raw)), dtype=np.float64)
    else:
        limits = np.asarray(raw, dtype=np.float64).reshape(-1)
    if limits.shape != (n_coils,):
        raise ValueError("coil_slew_limits must be a scalar or match the number of coils.")
    if np.any(~np.isfinite(limits)) or np.any(limits <= 0.0):
        raise ValueError("coil_slew_limits must contain finite values > 0.")
    return cast(FloatArray, np.asarray(limits, dtype=np.float64))


def resolve_supervisor_limits(
    cfg_limits: Any,
    override_limits: dict[str, float] | None,
) -> dict[str, float]:
    """Validate and merge free-boundary tracking supervisor limits."""
    allowed = {
        "tracking_error_norm",
        "shape_rms",
        "shape_max_abs",
        "x_point_position",
        "x_point_flux",
        "divertor_rms",
        "divertor_max_abs",
        "max_abs_coil_current",
        "max_abs_actuator_lag",
    }
    merged: dict[str, float] = {}
    for raw, source_name in (
        (cfg_limits, "free_boundary_tracking.supervisor_limits"),
        (override_limits, "supervisor_limits"),
    ):
        if raw is None:
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"{source_name} must be a mapping of limit names to non-negative floats.")
        for key, value in raw.items():
            if key not in allowed:
                allowed_keys = ", ".join(sorted(allowed))
                raise ValueError(f"Unknown {source_name} key {key!r}. Allowed keys: {allowed_keys}.")
            limit_value = float(value)
            if not np.isfinite(limit_value) or limit_value < 0.0:
                raise ValueError(f"{source_name}.{key} must be finite and >= 0.")
            merged[key] = limit_value
    return merged


def resolve_fallback_currents(
    n_coils: int,
    coil_current_limits: FloatArray,
    cfg_value: Any,
) -> FloatArray | None:
    """Resolve optional fallback currents that respect per-coil current limits."""
    if cfg_value is None:
        return None
    values = np.asarray(cfg_value, dtype=np.float64).reshape(-1)
    if values.shape != (n_coils,):
        raise ValueError("free_boundary_tracking.fallback_currents must match the number of coils.")
    if np.any(~np.isfinite(values)):
        raise ValueError("free_boundary_tracking.fallback_currents must be finite.")
    if np.any(np.abs(values) - coil_current_limits > 1e-12):
        raise ValueError("free_boundary_tracking.fallback_currents must respect CoilSet.current_limits.")
    return cast(FloatArray, values.copy())

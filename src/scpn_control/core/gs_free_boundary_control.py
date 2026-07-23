# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Free-boundary coil optim and objective helpers

"""Free-boundary coil optimisation, objective tolerances, and divertor labels.

This leaf owns the CONTROL free-boundary **control** surface used by coil-current
least-squares optimisation and objective status reporting. The CONTROL
:class:`~scpn_control.core.fusion_kernel.FusionKernel` product surface remains
first-class under dual-home C and keeps thin wrappers that supply mesh/green
response operators. Full free-boundary solve orchestration stays on the owner
(R0-S7).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_control._typing import FloatArray

logger = logging.getLogger(__name__)


@dataclass
class CoilSet:
    """External coil set for the free-boundary variant.

    Attributes
    ----------
    positions : list of (R, Z) tuples
        Coil centre coordinates [m].
    currents : AnyFloatArray
        Current per coil [A].
    turns : list of int
        Number of turns per coil.
    current_limits : AnyFloatArray or None
        Per-coil maximum absolute current [A].  Shape ``(n_coils,)``.
        When set, ``optimize_coil_currents`` enforces these bounds.
    target_flux_points : AnyFloatArray or None
        Points ``(R, Z)`` on the desired separatrix for shape optimisation.
        Shape ``(n_pts, 2)``.
    target_flux_values : AnyFloatArray or None
        Explicit target flux values at ``target_flux_points``.  When set, the
        free-boundary shape objective becomes an actual target-tracking problem
        instead of reproducing the current solved boundary flux.
    x_point_target : AnyFloatArray or None
        Explicit X-point target location ``(R, Z)``. When set, the free-boundary
        objective can enforce null-field and isoflux constraints there.
    x_point_flux_target : float or None
        Optional target flux value at ``x_point_target``. When omitted, the
        free-boundary path derives a separatrix target from boundary/divertor
        objectives or falls back to the current local flux.
    x_point_weight : float
        Weight applied to the X-point isoflux objective row.
    x_point_null_weight : float
        Weight applied to the X-point null-field objective rows.
    divertor_strike_points : AnyFloatArray or None
        Explicit divertor strike-point targets ``(R, Z)``. These are enforced
        as isoflux constraints during free-boundary optimisation.
    divertor_flux_values : AnyFloatArray or None
        Optional target flux value at each divertor strike point.
    divertor_weight : float
        Weight applied to divertor strike-point isoflux constraints.
    """

    positions: list[tuple[float, float]] = field(default_factory=list)
    currents: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    turns: list[int] = field(default_factory=list)
    current_limits: NDArray[np.float64] | None = None
    target_flux_points: NDArray[np.float64] | None = None
    target_flux_values: NDArray[np.float64] | None = None
    x_point_target: NDArray[np.float64] | None = None
    x_point_flux_target: float | None = None
    x_point_weight: float = 1.0
    x_point_null_weight: float = 1.0
    divertor_strike_points: NDArray[np.float64] | None = None
    divertor_flux_values: NDArray[np.float64] | None = None
    divertor_weight: float = 1.0


def shape_error_metrics(current_flux: FloatArray, target_flux: FloatArray) -> dict[str, float]:
    """Compute RMS and max-abs error for a boundary-shape target."""
    residual = np.asarray(current_flux, dtype=np.float64) - np.asarray(target_flux, dtype=np.float64)
    if residual.size == 0:
        return {"shape_error_rms": 0.0, "shape_error_max_abs": 0.0}
    return {
        "shape_error_rms": float(np.sqrt(np.mean(residual * residual))),
        "shape_error_max_abs": float(np.max(np.abs(residual))),
    }


def estimate_point_gradient(
    sample_fn: Callable[[float, float], Any],
    R_pt: float,
    Z_pt: float,
    *,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
    dR: float,
    dZ: float,
) -> tuple[FloatArray, FloatArray]:
    """Estimate d/dR and d/dZ using central or one-sided finite differences."""
    step_r = max(0.5 * float(dR), 1e-4)
    step_z = max(0.5 * float(dZ), 1e-4)

    center = np.asarray(sample_fn(R_pt, Z_pt), dtype=np.float64)

    if (R_pt - step_r) >= r_min and (R_pt + step_r) <= r_max:
        sample_plus_r = np.asarray(sample_fn(R_pt + step_r, Z_pt), dtype=np.float64)
        sample_minus_r = np.asarray(sample_fn(R_pt - step_r, Z_pt), dtype=np.float64)
        d_dR = (sample_plus_r - sample_minus_r) / (2.0 * step_r)
    elif (R_pt + step_r) <= r_max:
        sample_plus_r = np.asarray(sample_fn(R_pt + step_r, Z_pt), dtype=np.float64)
        d_dR = (sample_plus_r - center) / step_r
    elif (R_pt - step_r) >= r_min:
        sample_minus_r = np.asarray(sample_fn(R_pt - step_r, Z_pt), dtype=np.float64)
        d_dR = (center - sample_minus_r) / step_r
    else:
        d_dR = np.zeros_like(center)

    if (Z_pt - step_z) >= z_min and (Z_pt + step_z) <= z_max:
        sample_plus_z = np.asarray(sample_fn(R_pt, Z_pt + step_z), dtype=np.float64)
        sample_minus_z = np.asarray(sample_fn(R_pt, Z_pt - step_z), dtype=np.float64)
        d_dZ = (sample_plus_z - sample_minus_z) / (2.0 * step_z)
    elif (Z_pt + step_z) <= z_max:
        sample_plus_z = np.asarray(sample_fn(R_pt, Z_pt + step_z), dtype=np.float64)
        d_dZ = (sample_plus_z - center) / step_z
    elif (Z_pt - step_z) >= z_min:
        sample_minus_z = np.asarray(sample_fn(R_pt, Z_pt - step_z), dtype=np.float64)
        d_dZ = (center - sample_minus_z) / step_z
    else:
        d_dZ = np.zeros_like(center)

    return d_dR, d_dZ


def resolve_separatrix_flux_target(
    coils: CoilSet,
    shape_target_flux: FloatArray | None,
) -> float | None:
    """Resolve a scalar separatrix-flux target from active objectives."""
    if coils.x_point_flux_target is not None:
        return float(coils.x_point_flux_target)
    if coils.divertor_flux_values is not None and coils.divertor_flux_values.size > 0:
        return float(np.mean(np.asarray(coils.divertor_flux_values, dtype=np.float64)))
    if shape_target_flux is not None and np.asarray(shape_target_flux).size > 0:
        return float(np.mean(np.asarray(shape_target_flux, dtype=np.float64)))
    return None


def resolve_x_point_flux_target(
    coils: CoilSet,
    separatrix_flux_target: float | None,
    local_flux_at_x_point: float | None,
) -> tuple[float | None, str]:
    """Resolve the X-point flux target mode and scalar target."""
    if coils.x_point_target is None:
        return None, "disabled"

    if coils.x_point_flux_target is not None:
        return float(coils.x_point_flux_target), "explicit_target"
    if separatrix_flux_target is not None:
        return float(separatrix_flux_target), "derived_separatrix"
    if local_flux_at_x_point is None:
        raise ValueError("local_flux_at_x_point is required for self_flux_tracking mode")
    return float(local_flux_at_x_point), "self_flux_tracking"


def resolve_divertor_flux_targets(
    coils: CoilSet,
    separatrix_flux_target: float | None,
    sampled_strike_flux: FloatArray | None,
) -> tuple[FloatArray | None, str]:
    """Resolve divertor strike-point flux targets and mode."""
    if coils.divertor_strike_points is None:
        return None, "disabled"
    if coils.divertor_flux_values is not None:
        return np.asarray(coils.divertor_flux_values, dtype=np.float64).reshape(-1), "explicit_target"
    if separatrix_flux_target is not None:
        n_pts = int(coils.divertor_strike_points.shape[0])
        return np.full(n_pts, float(separatrix_flux_target), dtype=np.float64), "derived_separatrix"
    if sampled_strike_flux is None:
        raise ValueError("sampled_strike_flux is required for self_flux_tracking mode")
    return np.asarray(sampled_strike_flux, dtype=np.float64), "self_flux_tracking"


def resolve_free_boundary_objective_tolerances(
    cfg_objective_tolerances: Any,
    override_objective_tolerances: dict[str, float] | None = None,
) -> dict[str, float]:
    """Validate and merge free-boundary objective tolerances."""
    allowed = {
        "shape_rms",
        "shape_max_abs",
        "x_point_position",
        "x_point_gradient",
        "x_point_flux",
        "divertor_rms",
        "divertor_max_abs",
    }

    merged: dict[str, float] = {}
    for raw, source_name in (
        (cfg_objective_tolerances, "free_boundary.objective_tolerances"),
        (override_objective_tolerances, "objective_tolerances"),
    ):
        if raw is None:
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"{source_name} must be a mapping of tolerance names to non-negative floats.")
        for key, value in raw.items():
            if key not in allowed:
                allowed_keys = ", ".join(sorted(allowed))
                raise ValueError(f"Unknown {source_name} key {key!r}. Allowed keys: {allowed_keys}.")
            tol_value = float(value)
            if not np.isfinite(tol_value) or tol_value < 0.0:
                raise ValueError(f"{source_name}.{key} must be finite and >= 0.")
            merged[key] = tol_value
    return merged


def evaluate_free_boundary_objective_status(
    tolerances: dict[str, float],
    *,
    shape_error_rms: float | None,
    shape_error_max_abs: float | None,
    x_point_detected_error: float | None,
    x_point_gradient_norm: float | None,
    x_point_flux_error: float | None,
    divertor_error_rms: float | None,
    divertor_error_max_abs: float | None,
) -> dict[str, Any]:
    """Evaluate which configured free-boundary objective tolerances are satisfied."""

    def check_metric(metric: float | None, tolerance_key: str) -> bool:
        if metric is None or not np.isfinite(metric):
            return False
        return float(abs(metric)) <= tolerances[tolerance_key]

    checks: dict[str, bool] = {}
    if "shape_rms" in tolerances and shape_error_rms is not None:
        checks["shape_rms"] = check_metric(shape_error_rms, "shape_rms")
    if "shape_max_abs" in tolerances and shape_error_max_abs is not None:
        checks["shape_max_abs"] = check_metric(shape_error_max_abs, "shape_max_abs")
    if "x_point_position" in tolerances and x_point_detected_error is not None:
        checks["x_point_position"] = check_metric(x_point_detected_error, "x_point_position")
    if "x_point_gradient" in tolerances and x_point_gradient_norm is not None:
        checks["x_point_gradient"] = check_metric(x_point_gradient_norm, "x_point_gradient")
    if "x_point_flux" in tolerances and x_point_flux_error is not None:
        checks["x_point_flux"] = check_metric(x_point_flux_error, "x_point_flux")
    if "divertor_rms" in tolerances and divertor_error_rms is not None:
        checks["divertor_rms"] = check_metric(divertor_error_rms, "divertor_rms")
    if "divertor_max_abs" in tolerances and divertor_error_max_abs is not None:
        checks["divertor_max_abs"] = check_metric(divertor_error_max_abs, "divertor_max_abs")

    return {
        "objective_tolerances": tolerances.copy(),
        "objective_checks": checks,
        "objective_convergence_active": bool(checks),
        "objective_converged": all(checks.values()) if checks else True,
    }


def divertor_configuration_label(strike_points: FloatArray | None) -> str:
    """Return a coarse divertor-target configuration label."""
    if strike_points is None:
        return "none"
    n_pts = int(np.asarray(strike_points).shape[0])
    if n_pts <= 0:
        return "none"
    if n_pts == 1:
        return "single_strike"
    if n_pts == 2:
        return "double_strike"
    return "multi_strike"


def resolve_shape_target_flux(
    coils: CoilSet,
    current_flux: FloatArray,
) -> tuple[FloatArray, str]:
    """Resolve the shape objective target for free-boundary optimisation."""
    if coils.target_flux_values is not None:
        target_flux = np.asarray(coils.target_flux_values, dtype=np.float64).reshape(-1)
        if target_flux.shape != current_flux.shape:
            raise ValueError("CoilSet.target_flux_values must match the sampled target_flux_points shape.")
        return target_flux, "explicit_target"
    return np.asarray(current_flux, dtype=np.float64).copy(), "self_flux_tracking"


def optimize_coil_currents(
    coils: CoilSet,
    target_flux: FloatArray,
    *,
    build_mutual_inductance_matrix: Callable[[CoilSet, FloatArray], FloatArray],
    coil_flux_response_at_point: Callable[[CoilSet, FloatArray], FloatArray],
    coil_flux_gradient_response: Callable[[CoilSet, FloatArray], tuple[FloatArray, FloatArray]],
    tikhonov_alpha: float = 1e-4,
    x_point_flux_target: float | None = None,
    divertor_flux_targets: FloatArray | None = None,
) -> FloatArray:
    """Find coil currents that best satisfy free-boundary target constraints.

    Solves a bounded linear least-squares problem that can include:

    - boundary-flux targets at ``target_flux_points``
    - X-point isoflux and null-field constraints
    - divertor strike-point isoflux constraints

        min_I || A I - b ||^2 + alpha * ||I||^2
        s.t.  -I_max <= I <= I_max  (per coil)

    where ``A`` stacks the active constraint blocks.

    Parameters
    ----------
    coils :
        Coil geometry and optional objective targets.
    target_flux :
        Desired poloidal flux at ``target_flux_points``. Can be empty when
        only X-point and/or divertor constraints are active.
    build_mutual_inductance_matrix :
        Operator returning per-coil flux response at observation points,
        shape ``(n_coils, n_pts)``.
    coil_flux_response_at_point :
        Operator returning per-coil flux response at a single ``(R, Z)`` point.
    coil_flux_gradient_response :
        Operator returning per-coil ``(dPsi/dR, dPsi/dZ)`` rows at a point.
    tikhonov_alpha :
        Regularisation strength to penalise large currents.
    x_point_flux_target :
        Scalar target flux at ``coils.x_point_target``.
    divertor_flux_targets :
        Flux targets at ``coils.divertor_strike_points``.

    Returns
    -------
    FloatArray
        Optimised coil currents [A], shape ``(n_coils,)``.
    """
    from scipy.optimize import lsq_linear

    n_coils = len(coils.positions)
    row_blocks: list[FloatArray] = []
    rhs_blocks: list[FloatArray] = []

    target_flux_arr = np.asarray(target_flux, dtype=np.float64).reshape(-1)
    if coils.target_flux_points is not None:
        if target_flux_arr.shape != (coils.target_flux_points.shape[0],):
            raise ValueError("target_flux must match CoilSet.target_flux_points shape.")
        mutual = build_mutual_inductance_matrix(coils, coils.target_flux_points)  # (n_coils, n_pts)
        row_blocks.append(mutual.T)
        rhs_blocks.append(target_flux_arr)
    elif target_flux_arr.size > 0:
        raise ValueError("target_flux requires CoilSet.target_flux_points to be set.")

    if coils.x_point_target is not None:
        x_target = np.asarray(coils.x_point_target, dtype=np.float64).reshape(2)
        x_flux_target = float(x_point_flux_target) if x_point_flux_target is not None else None
        if x_flux_target is not None and coils.x_point_weight > 0.0:
            x_flux_row = coil_flux_response_at_point(coils, x_target).reshape(1, -1)
            row_blocks.append(float(coils.x_point_weight) * x_flux_row)
            rhs_blocks.append(np.asarray([float(coils.x_point_weight) * x_flux_target], dtype=np.float64))

        if coils.x_point_null_weight > 0.0:
            dpsi_dR_row, dpsi_dZ_row = coil_flux_gradient_response(coils, x_target)
            gradient_block = np.vstack([dpsi_dR_row, dpsi_dZ_row])
            row_blocks.append(float(coils.x_point_null_weight) * gradient_block)
            rhs_blocks.append(np.zeros(2, dtype=np.float64))

    if coils.divertor_strike_points is not None and divertor_flux_targets is not None:
        divertor_flux_arr = np.asarray(divertor_flux_targets, dtype=np.float64).reshape(-1)
        if divertor_flux_arr.shape != (coils.divertor_strike_points.shape[0],):
            raise ValueError("divertor_flux_targets must match CoilSet.divertor_strike_points shape.")
        if coils.divertor_weight > 0.0:
            divertor_block = build_mutual_inductance_matrix(coils, coils.divertor_strike_points).T
            row_blocks.append(float(coils.divertor_weight) * divertor_block)
            rhs_blocks.append(float(coils.divertor_weight) * divertor_flux_arr)

    if not row_blocks:
        raise ValueError("At least one free-boundary optimisation target must be set.")

    a_base = np.vstack(row_blocks)
    b_base = np.concatenate(rhs_blocks)

    # Build augmented system: [A_base; sqrt(alpha)*I] I = [b_base; 0]
    a_mat = np.vstack([a_base, np.sqrt(tikhonov_alpha) * np.eye(n_coils)])
    b_vec = np.concatenate([b_base, np.zeros(n_coils, dtype=np.float64)])

    # Bounds
    if coils.current_limits is not None:
        lb = -np.abs(coils.current_limits)
        ub = np.abs(coils.current_limits)
    else:
        lb = -np.inf * np.ones(n_coils)
        ub = np.inf * np.ones(n_coils)

    result = lsq_linear(a_mat, b_vec, bounds=(lb, ub), method="trf")
    logger.info(
        "Coil optimisation: cost=%.4e, status=%d (%s)",
        result.cost,
        result.status,
        result.message,
    )
    return np.asarray(result.x, dtype=np.float64)


# Historical private names for FusionKernel wrappers.
_shape_error_metrics = shape_error_metrics
_estimate_point_gradient = estimate_point_gradient
_resolve_separatrix_flux_target = resolve_separatrix_flux_target
_resolve_x_point_flux_target = resolve_x_point_flux_target
_resolve_divertor_flux_targets = resolve_divertor_flux_targets
_resolve_free_boundary_objective_tolerances = resolve_free_boundary_objective_tolerances
_evaluate_free_boundary_objective_status = evaluate_free_boundary_objective_status
_divertor_configuration_label = divertor_configuration_label
_resolve_shape_target_flux = resolve_shape_target_flux
_optimize_coil_currents = optimize_coil_currents

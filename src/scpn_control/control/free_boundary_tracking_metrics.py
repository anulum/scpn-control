# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Free-boundary objective metrics
"""Pure objective-error and convergence metrics for free-boundary tracking."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.control import free_boundary_tracking_control_law as _control_law
from scpn_control.control import free_boundary_tracking_limits as _limits
from scpn_control.control.free_boundary_tracking_observation import ObjectiveBlock


def _finite_norm(values: FloatArray) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        norm = float(np.hypot.reduce(np.abs(values), initial=0.0))
    if not np.isfinite(norm):
        raise ValueError("objective norm is nonfinite.")
    return norm


def _stable_rms(values: FloatArray) -> float:
    if values.size < 1:
        raise ValueError("objective block must not be empty.")
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0
    normalised = values / scale
    return float(scale * np.sqrt(np.mean(normalised * normalised)))


def evaluate_objective_metrics(
    observation: AnyFloatArray,
    *,
    target_vector: FloatArray,
    objective_blocks: Sequence[ObjectiveBlock],
    objective_tolerances: Mapping[str, float],
    control_objective_weights: FloatArray,
) -> dict[str, Any]:
    """Evaluate bounded shape and control errors from an objective vector.

    Parameters
    ----------
    observation
        Current objective-space observation.
    target_vector
        Desired objective-space values.
    objective_blocks
        Named contiguous slices of the objective vector.
    objective_tolerances
        Finite non-negative thresholds named ``shape_rms``, ``shape_max_abs``,
        ``x_point_position``, ``x_point_flux``, ``divertor_rms`` or
        ``divertor_max_abs``. Each threshold uses its corresponding objective's
        units. Zero requires exact agreement. An empty mapping leaves all
        blocks active; configured checks apply only to present blocks.
    control_objective_weights
        Per-entry weights used in the control-error norm.

    Returns
    -------
    dict[str, Any]
        Tracking, block, control and convergence metrics.

    Raises
    ------
    ValueError
        If an objective or derived metric is malformed or nonfinite, or a
        tolerance has an unknown name, invalid value or negative magnitude.

    Examples
    --------
    A shape-flux error within tolerance remains visible in tracking metrics
    while its block contributes zero to the control-error norm.

    >>> metrics = evaluate_objective_metrics(
    ...     np.array([0.05]), target_vector=np.zeros(1),
    ...     objective_blocks=(ObjectiveBlock("shape_flux", 0, 1),),
    ...     objective_tolerances={"shape_rms": 0.1},
    ...     control_objective_weights=np.ones(1),
    ... )
    >>> metrics["shape_rms"], metrics["control_error_norm"]
    (0.05, 0.0)
    >>> metrics["objective_converged"]
    True
    """
    if not isinstance(objective_tolerances, Mapping):
        raise ValueError("objective_tolerances must be a mapping of tolerance names to values.")
    try:
        objective_tolerances = _limits.resolve_objective_tolerances(None, dict(objective_tolerances))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("objective_tolerances must use known names and finite values >= 0.") from exc
    obs = np.asarray(observation, dtype=np.float64).reshape(-1)
    target = np.asarray(target_vector, dtype=np.float64).reshape(-1)
    weights = np.asarray(control_objective_weights, dtype=np.float64).reshape(-1)
    if obs.shape != target.shape or weights.shape != target.shape:
        raise ValueError("observation and control weights must match the target vector shape.")
    if any(not np.isfinite(value).all() for value in (obs, target, weights)):
        raise ValueError("objective inputs contain nonfinite values.")
    with np.errstate(over="ignore", invalid="ignore"):
        error = target - obs
    if not np.isfinite(error).all():
        raise ValueError("objective error is nonfinite.")

    metrics: dict[str, Any] = {
        "tracking_error_norm": _finite_norm(error),
        "control_error_norm": 0.0,
        "shape_rms": None,
        "shape_max_abs": None,
        "x_point_position_error": None,
        "x_point_flux_error": None,
        "divertor_rms": None,
        "divertor_max_abs": None,
        "active_control_rows": 0,
    }
    for block in objective_blocks:
        block_error = error[block.start : block.stop]
        if block.name == "shape_flux":
            metrics["shape_rms"] = _stable_rms(block_error)
            metrics["shape_max_abs"] = float(np.max(np.abs(block_error)))
        elif block.name == "x_point_position":
            metrics["x_point_position_error"] = _finite_norm(block_error)
        elif block.name == "x_point_flux":
            metrics["x_point_flux_error"] = float(abs(block_error[0]))
        elif block.name == "divertor_flux":
            metrics["divertor_rms"] = _stable_rms(block_error)
            metrics["divertor_max_abs"] = float(np.max(np.abs(block_error)))

    checks: dict[str, bool] = {}
    if "shape_rms" in objective_tolerances and metrics["shape_rms"] is not None:
        checks["shape_rms"] = bool(metrics["shape_rms"] <= objective_tolerances["shape_rms"])
    if "shape_max_abs" in objective_tolerances and metrics["shape_max_abs"] is not None:
        checks["shape_max_abs"] = bool(metrics["shape_max_abs"] <= objective_tolerances["shape_max_abs"])
    if "x_point_position" in objective_tolerances and metrics["x_point_position_error"] is not None:
        checks["x_point_position"] = bool(metrics["x_point_position_error"] <= objective_tolerances["x_point_position"])
    if "x_point_flux" in objective_tolerances and metrics["x_point_flux_error"] is not None:
        checks["x_point_flux"] = bool(metrics["x_point_flux_error"] <= objective_tolerances["x_point_flux"])
    if "divertor_rms" in objective_tolerances and metrics["divertor_rms"] is not None:
        checks["divertor_rms"] = bool(metrics["divertor_rms"] <= objective_tolerances["divertor_rms"])
    if "divertor_max_abs" in objective_tolerances and metrics["divertor_max_abs"] is not None:
        checks["divertor_max_abs"] = bool(metrics["divertor_max_abs"] <= objective_tolerances["divertor_max_abs"])
    metrics["objective_checks"] = checks
    metrics["objective_convergence_active"] = bool(checks)
    metrics["objective_converged"] = all(checks.values()) if checks else True
    metrics["objective_tolerances"] = dict(objective_tolerances)
    control_mask = _control_law.build_control_activation_mask(
        int(target.size), objective_blocks, objective_tolerances, metrics
    )
    with np.errstate(over="ignore", invalid="ignore"):
        control_error: FloatArray = np.multiply(np.multiply(control_mask, weights), error)
    if not np.isfinite(control_error).all():
        raise ValueError("weighted control error is nonfinite.")
    metrics["control_error_norm"] = _finite_norm(control_error)
    metrics["active_control_rows"] = int(np.count_nonzero(np.abs(control_error) > 1.0e-12))
    return metrics

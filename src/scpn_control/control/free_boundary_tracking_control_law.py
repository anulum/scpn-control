# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Free-boundary tracking control-law helpers

"""Pure response diagnostics and coil-correction law for free-boundary tracking.

This leaf owns SVD-based response-matrix diagnostics, objective-block control
activation masks, coil headroom penalties, and the Tikhonov-regularised
least-squares coil correction. Kernel-coupled response identification and
actuator application stay on
:class:`~scpn_control.control.free_boundary_tracking.FreeBoundaryTrackingController`
(R3-S4 shot orchestration). Claims remain in ``free_boundary_tracking_claims``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast

import numpy as np

from scpn_control._typing import FloatArray
from scpn_control.control.free_boundary_tracking_observation import ObjectiveBlock


@dataclass(frozen=True)
class ResponseDiagnostics:
    """Singular-value diagnostics for a coil-to-objective response matrix."""

    rank: int
    condition_number: float
    max_singular_value: float
    degenerate: bool


def compute_response_diagnostics(response_matrix: FloatArray) -> ResponseDiagnostics:
    """Compute rank / condition / degeneracy from the response singular values."""
    matrix = np.asarray(response_matrix, dtype=np.float64)
    singular_values = np.asarray(np.linalg.svd(matrix, compute_uv=False), dtype=np.float64).reshape(-1)
    if singular_values.size < 1:  # pragma: no cover - unreachable: n_coils>=1 implies >=1 singular value
        return ResponseDiagnostics(
            rank=0,
            condition_number=float("inf"),
            max_singular_value=0.0,
            degenerate=True,
        )

    sigma_max = float(np.max(singular_values))
    eps = float(np.finfo(np.float64).eps)
    cutoff = float(max(eps * max(1.0, sigma_max), 1.0e-12))
    nonzero = singular_values[np.asarray(singular_values > cutoff, dtype=np.bool_)]
    rank = int(nonzero.size)
    condition_number = float(sigma_max / float(nonzero[-1])) if nonzero.size > 0 else float("inf")
    degenerate = bool((not np.isfinite(sigma_max)) or sigma_max <= cutoff or nonzero.size < 1)
    return ResponseDiagnostics(
        rank=rank,
        condition_number=condition_number,
        max_singular_value=sigma_max,
        degenerate=degenerate,
    )


def build_control_activation_mask(
    target_size: int,
    objective_blocks: Sequence[ObjectiveBlock],
    objective_tolerances: Mapping[str, float],
    metrics: Mapping[str, Any],
) -> FloatArray:
    """Zero objective-block slices that already satisfy configured tolerances."""
    objective_checks = cast(dict[str, bool], metrics.get("objective_checks", {}))
    mask = np.ones(int(target_size), dtype=np.float64)
    for block in objective_blocks:
        if block.name == "shape_flux":
            # Annotate as list[str] so the per-block reassignments below (each a
            # comprehension over a different literal tuple) share one type; mypy
            # 2.2.0 otherwise pins the variable to the first branch's Literal set.
            relevant: list[str] = [
                key for key in ("shape_rms", "shape_max_abs") if key in objective_tolerances
            ]
        elif block.name == "x_point_position":
            relevant = [key for key in ("x_point_position",) if key in objective_tolerances]
        elif block.name == "x_point_flux":
            relevant = [key for key in ("x_point_flux",) if key in objective_tolerances]
        elif block.name == "divertor_flux":
            relevant = [key for key in ("divertor_rms", "divertor_max_abs") if key in objective_tolerances]
        else:
            raise ValueError(f"Unknown objective block {block.name!r}.")
        if relevant and all(objective_checks.get(key, False) for key in relevant):
            mask[block.start : block.stop] = 0.0
    return cast(FloatArray, mask)


def build_coil_penalties(
    coil_currents: FloatArray,
    coil_current_limits: FloatArray,
    delta_hint: FloatArray,
) -> FloatArray:
    """Scale coil regularisation by inverse headroom along the correction direction."""
    currents = np.asarray(coil_currents, dtype=np.float64).reshape(-1)
    limits = np.asarray(coil_current_limits, dtype=np.float64).reshape(-1)
    hint = np.asarray(delta_hint, dtype=np.float64).reshape(-1)
    n_coils = int(currents.size)
    if limits.shape != (n_coils,):
        raise ValueError("coil_current_limits must match the number of coils.")
    headrooms = np.ones(n_coils, dtype=np.float64)
    penalties = np.ones(n_coils, dtype=np.float64)
    for idx in range(n_coils):
        limit = float(limits[idx])
        if not np.isfinite(limit) or limit <= 0.0:
            headrooms[idx] = np.inf
            continue
        current = float(currents[idx])
        direction = float(hint[idx]) if idx < hint.size else 0.0
        if direction > 1.0e-12:
            headroom = limit - current
        elif direction < -1.0e-12:
            headroom = limit + current
        else:
            headroom = limit - abs(current)
        headrooms[idx] = max(headroom, 1.0e-9)
    finite_headrooms = headrooms[np.isfinite(headrooms)]
    reference_headroom = float(np.max(finite_headrooms)) if finite_headrooms.size > 0 else 1.0
    for idx in range(n_coils):
        if not np.isfinite(headrooms[idx]):
            penalties[idx] = 1.0
            continue
        penalties[idx] = float(np.sqrt(max(reference_headroom / float(headrooms[idx]), 1.0)))
    return cast(FloatArray, penalties)


def compute_coil_correction(
    observation: FloatArray,
    *,
    target_vector: FloatArray,
    objective_bias_estimate: FloatArray,
    control_objective_weights: FloatArray,
    response_matrix: FloatArray,
    response_regularization: float,
    correction_limit: float,
    control_mask: FloatArray,
    coil_currents: FloatArray,
    coil_current_limits: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Compute bounded Tikhonov coil corrections and the coil penalties used.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        ``(clipped_delta_currents, coil_penalties)``.
    """
    obs = np.asarray(observation, dtype=np.float64).reshape(-1)
    target = np.asarray(target_vector, dtype=np.float64).reshape(-1)
    if obs.shape != target.shape:
        raise ValueError("observation must match the free-boundary target vector shape.")
    bias = np.asarray(objective_bias_estimate, dtype=np.float64).reshape(-1)
    if bias.shape != target.shape:
        raise ValueError("objective_bias_estimate must match the free-boundary target vector shape.")
    weights = np.asarray(control_objective_weights, dtype=np.float64).reshape(-1)
    if weights.shape != target.shape:
        raise ValueError("control_objective_weights must match the free-boundary target vector shape.")
    mask = np.asarray(control_mask, dtype=np.float64).reshape(-1)
    if mask.shape != target.shape:
        raise ValueError("control_mask must match the free-boundary target vector shape.")
    response = np.asarray(response_matrix, dtype=np.float64)
    if response.ndim != 2 or response.shape[0] != target.size:
        raise ValueError("response_matrix must be (n_objectives, n_coils).")
    n_coils = int(response.shape[1])
    if not np.isfinite(response_regularization) or response_regularization < 0.0:
        raise ValueError("response_regularization must be finite and >= 0.")
    if not np.isfinite(correction_limit) or correction_limit <= 0.0:
        raise ValueError("correction_limit must be finite and > 0.")

    error = target + bias - obs
    weight_vector = weights * mask
    weighted_response = weight_vector[:, None] * response
    weighted_error = weight_vector * error
    base_reg = np.sqrt(float(response_regularization)) * np.eye(n_coils, dtype=np.float64)
    base_aug_matrix = np.vstack([weighted_response, base_reg])
    base_aug_rhs = np.concatenate([weighted_error, np.zeros(n_coils, dtype=np.float64)])
    delta_hint, *_ = np.linalg.lstsq(base_aug_matrix, base_aug_rhs, rcond=None)
    delta_hint = np.asarray(delta_hint, dtype=np.float64)
    coil_penalties = build_coil_penalties(coil_currents, coil_current_limits, delta_hint)
    aug_matrix = np.vstack(
        [
            weighted_response,
            np.sqrt(float(response_regularization)) * np.diag(coil_penalties),
        ]
    )
    aug_rhs = np.concatenate([weighted_error, np.zeros(n_coils, dtype=np.float64)])
    delta, *_ = np.linalg.lstsq(aug_matrix, aug_rhs, rcond=None)
    clipped = np.clip(np.asarray(delta, dtype=np.float64), -float(correction_limit), float(correction_limit))
    return (
        cast(FloatArray, np.asarray(clipped, dtype=np.float64)),
        cast(FloatArray, np.asarray(coil_penalties, dtype=np.float64)),
    )

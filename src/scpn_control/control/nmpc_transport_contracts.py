# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — NMPC transport tuning contracts
"""Result contracts, source bounds, and rollout gradient audit for NMPC tuning."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.core.differentiable_transport import (
    TransportCampaignMetadata,
    TransportGradientAudit,
    transport_rollout_tracking_loss,
)


@dataclass(frozen=True)
class TransportCoefficientTuningResult:
    """Result of a gradient-based transport-coefficient tuning step."""

    loss: float
    gradient: FloatArray
    updated_chi: FloatArray
    step_norm: float
    metadata: TransportCampaignMetadata
    gradient_audit: TransportGradientAudit | None


@dataclass(frozen=True)
class TransportSourceScheduleTuningResult:
    """Result of a gradient-based transport source-schedule tuning step."""

    loss: float
    gradient: FloatArray
    updated_sources: FloatArray
    step_norm: float
    metadata: TransportCampaignMetadata
    gradient_audit: TransportGradientAudit | None


@dataclass(frozen=True)
class TransportSourceRolloutGradientAudit:
    """Finite-difference audit for multi-step source-schedule gradients."""

    loss: float
    epsilon: float
    tolerance: float
    checked_indices: tuple[tuple[int, int, int], ...]
    source_max_abs_error: float
    passed: bool


@dataclass(frozen=True)
class TransportSourceRolloutTuningResult:
    """Result of a gradient-based multi-step transport source rollout update."""

    loss: float
    gradient: FloatArray
    updated_sources: FloatArray
    final_profiles: FloatArray
    step_norm: float
    metadata: TransportCampaignMetadata
    gradient_audit: TransportSourceRolloutGradientAudit | None


def _bounded_tuning_update(
    name: str,
    baseline: FloatArray,
    gradient: FloatArray,
    learning_rate: float,
    *,
    fractional_cap: float | None = None,
    absolute_cap: float | None = None,
    lower: AnyFloatArray | float | None = None,
    upper: AnyFloatArray | float | None = None,
) -> tuple[FloatArray, float]:
    """Apply a bounded gradient update and measure a representable step norm.

    Parameters
    ----------
    name
        Diagnostic name used when an arithmetic stage is invalid.
    baseline, gradient
        Finite arrays of matching shape for the current point and derivative.
    learning_rate
        Positive finite step multiplier validated by the public entry point.
    fractional_cap, absolute_cap
        Optional per-element relative or scalar absolute step limits.
    lower, upper
        Optional post-update bounds.

    Returns
    -------
    tuple[FloatArray, float]
        Detached updated values and a finite Euclidean step norm.

    Raises
    ------
    ValueError
        Any intermediate or the final norm is not representable as finite.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        delta = np.asarray(-learning_rate * gradient, dtype=np.float64)
    if not np.all(np.isfinite(delta)):
        raise ValueError(f"{name} update must be finite.")
    if fractional_cap is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            cap = fractional_cap * np.maximum(np.abs(baseline), 1.0e-12)
        if not np.all(np.isfinite(cap)):
            raise ValueError(f"{name} update cap must be finite.")
        delta = np.clip(delta, -cap, cap)
    if absolute_cap is not None:
        delta = np.clip(delta, -absolute_cap, absolute_cap)
    with np.errstate(over="ignore", invalid="ignore"):
        proposal = np.asarray(baseline + delta, dtype=np.float64)
    if not np.all(np.isfinite(proposal)):
        raise ValueError(f"{name} updated values must be finite.")
    updated = proposal
    if lower is not None:
        updated = np.maximum(lower, updated)
    if upper is not None:
        updated = np.minimum(upper, updated)
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.asarray(updated - baseline, dtype=np.float64)
    if not np.all(np.isfinite(difference)):
        raise ValueError(f"{name} update difference must be finite.")
    scale = float(np.max(np.abs(difference))) if difference.size else 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        step_norm = 0.0 if scale == 0.0 else float(scale * np.linalg.norm(difference / scale))
    if not np.isfinite(step_norm):
        raise ValueError(f"{name} step norm must be finite.")
    return np.asarray(updated, dtype=np.float64), step_norm


def _optional_finite_array_bound(name: str, value: object, shape: tuple[int, ...]) -> FloatArray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == ():
        arr = np.full(shape, float(arr), dtype=np.float64)
    if arr.shape != shape or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite and broadcastable to source shape.")
    return arr


def _rollout_audit_indices(
    source_shape: tuple[int, ...],
    sample_indices: object | None,
) -> tuple[tuple[int, int, int], ...]:
    if len(source_shape) != 3:
        raise ValueError("source_sequence must have shape (n_steps, 4, n_rho).")
    n_steps, n_channels, n_rho = source_shape
    if n_steps < 1 or n_channels != 4 or n_rho < 3:
        raise ValueError("source_sequence must have shape (n_steps, 4, n_rho) with n_rho >= 3.")
    if sample_indices is None:
        candidates: tuple[tuple[int, int, int], ...] = (
            (0, 0, 1),
            (n_steps - 1, 1, n_rho // 2),
            (n_steps // 2, 2, n_rho - 2),
            (n_steps - 1, 3, max(1, n_rho // 3)),
        )
    else:
        if isinstance(sample_indices, (str, bytes)) or not isinstance(sample_indices, Iterable):
            raise ValueError("gradient_audit_sample_indices must be an iterable of three-part indices.")
        parsed_candidates: list[tuple[int, int, int]] = []
        for raw_index in sample_indices:
            if isinstance(raw_index, (str, bytes)) or not isinstance(raw_index, Iterable):
                raise ValueError("gradient_audit_sample_indices must contain iterable three-part indices.")
            parts = tuple(raw_index)
            if any(isinstance(part, (bool, np.bool_)) or not isinstance(part, (int, np.integer)) for part in parts):
                raise ValueError("gradient_audit_sample_indices must contain integer coordinates.")
            index_tuple = tuple(int(part) for part in parts)
            if len(index_tuple) != 3:
                raise ValueError("gradient_audit_sample_indices must contain three-part indices.")
            parsed_candidates.append((index_tuple[0], index_tuple[1], index_tuple[2]))
        candidates = tuple(parsed_candidates)
    unique: list[tuple[int, int, int]] = []
    for step, channel, radius in candidates:
        if not (0 <= step < n_steps and 0 <= channel < n_channels and 0 <= radius < n_rho):
            raise ValueError("gradient_audit_sample_indices contain an out-of-range rollout source index.")
        index = (int(step), int(channel), int(radius))
        if index not in unique:
            unique.append(index)
    if not unique:
        raise ValueError("gradient_audit_sample_indices must contain at least one index.")
    return tuple(unique)


def _audit_transport_rollout_source_gradients(
    initial_profiles: AnyFloatArray,
    chi: AnyFloatArray,
    source_sequence: AnyFloatArray,
    target_history: AnyFloatArray,
    rho: AnyFloatArray,
    dt: float,
    edge_values: AnyFloatArray,
    source_gradient: AnyFloatArray,
    *,
    weights: AnyFloatArray | None,
    epsilon: float,
    tolerance: float,
    sample_indices: object | None,
) -> TransportSourceRolloutGradientAudit:
    epsilon_float = float(epsilon)
    tolerance_float = float(tolerance)
    if not np.isfinite(epsilon_float) or epsilon_float <= 0.0:
        raise ValueError("gradient_audit_epsilon must be positive and finite.")
    if not np.isfinite(tolerance_float) or tolerance_float <= 0.0:
        raise ValueError("gradient_audit_tolerance must be positive and finite.")
    indices = _rollout_audit_indices(source_sequence.shape, sample_indices)
    base_loss = float(
        transport_rollout_tracking_loss(
            initial_profiles,
            chi,
            source_sequence,
            target_history,
            rho,
            dt,
            edge_values,
            weights=weights,
            use_jax=False,
        )
    )
    max_abs_error = 0.0
    for index in indices:
        plus_sources = source_sequence.copy()
        minus_sources = source_sequence.copy()
        plus_sources[index] += epsilon_float
        minus_sources[index] -= epsilon_float
        plus_loss = float(
            transport_rollout_tracking_loss(
                initial_profiles,
                chi,
                plus_sources,
                target_history,
                rho,
                dt,
                edge_values,
                weights=weights,
                use_jax=False,
            )
        )
        minus_loss = float(
            transport_rollout_tracking_loss(
                initial_profiles,
                chi,
                minus_sources,
                target_history,
                rho,
                dt,
                edge_values,
                weights=weights,
                use_jax=False,
            )
        )
        finite_difference = (plus_loss - minus_loss) / (2.0 * epsilon_float)
        max_abs_error = max(max_abs_error, abs(float(source_gradient[index]) - finite_difference))
    return TransportSourceRolloutGradientAudit(
        loss=base_loss,
        epsilon=epsilon_float,
        tolerance=tolerance_float,
        checked_indices=indices,
        source_max_abs_error=float(max_abs_error),
        passed=bool(max_abs_error <= tolerance_float),
    )

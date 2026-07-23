# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable transport rollout source AD

"""Multi-step rollout source gradients and finite-difference audit contracts.

This leaf owns multi-step tracking loss, JAX source-schedule gradients, and the
sampled finite-difference admission audit used by controller tuning. Core
rollout step primitives and input validators remain on the numerical facade;
equilibrium-weighted rollout AD is a later R1 stage. This module lazy-imports
the facade at call time so public re-exports and test monkeypatches of
``scpn_control.core.differentiable_transport`` remain the production path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from scpn_control._typing import FloatArray
from scpn_control.core.differentiable_transport_evidence import (
    CHANNEL_COUNT,
    TransportRolloutGradientAudit,
)


@dataclass(frozen=True)
class TransportRolloutSourceGradients:
    """JAX gradients of multi-step transport loss for source schedules."""

    loss: float
    source_gradient: FloatArray
    final_profiles: FloatArray


def _facade() -> Any:
    """Return the public numerical facade (lazy to avoid import cycles)."""
    from scpn_control.core import differentiable_transport as facade

    return facade


def transport_rollout_tracking_loss(
    initial_profiles: Any,
    chi: Any,
    source_sequence: Any,
    target_history: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    weights: Any | None = None,
    use_jax: bool = True,
    allow_numpy_fallback: bool = False,
    allow_legacy_numpy_fallback: bool = False,
) -> Any:
    """Return weighted multi-step transport tracking loss."""
    facade = _facade()
    profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array = (
        facade._validate_transport_rollout_inputs(
            initial_profiles,
            chi,
            source_sequence,
            rho,
            dt,
            edge_values,
            target_history=target_history,
            weights=weights,
        )
    )
    if target_array is None:
        raise ValueError("target_history is required")
    if weight_array is None:
        weight_array = np.ones(CHANNEL_COUNT)
    use_jax_runtime = facade._resolve_use_jax(
        use_jax,
        allow_numpy_fallback=allow_numpy_fallback,
        allow_legacy_numpy_fallback=allow_legacy_numpy_fallback,
        context="transport_rollout_tracking_loss",
    )
    if use_jax_runtime:
        if facade.jnp is None:
            raise RuntimeError("JAX rollout tracking loss requested but JAX is unavailable")
        history = facade._transport_rollout_jax(
            profile_array, chi_array, source_array, rho_array, float(dt), edge_array
        )
        residual = history - facade.jnp.asarray(target_array, dtype=facade.jnp.float64)
        return facade.jnp.mean(
            facade.jnp.asarray(weight_array, dtype=facade.jnp.float64)[None, :, None] * residual * residual
        )
    history = facade._transport_rollout_numpy(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)
    residual = history - target_array
    return float(np.mean(weight_array[None, :, None] * residual * residual))


def transport_rollout_source_gradients(
    initial_profiles: Any,
    chi: Any,
    source_sequence: Any,
    target_history: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    weights: Any | None = None,
) -> TransportRolloutSourceGradients:
    """Return JAX gradients for a multi-step transport source schedule."""
    facade = _facade()
    if not facade._HAS_JAX or facade.jax is None or facade.jnp is None:
        raise RuntimeError("transport_rollout_source_gradients requires JAX")
    profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array = (
        facade._validate_transport_rollout_inputs(
            initial_profiles,
            chi,
            source_sequence,
            rho,
            dt,
            edge_values,
            target_history=target_history,
            weights=weights,
        )
    )
    if target_array is None:
        raise ValueError("target_history is required")
    if weight_array is None:
        weight_array = np.ones(CHANNEL_COUNT)

    def loss_for_sources(source_candidate: Any) -> Any:
        history = facade._transport_rollout_jax(
            profile_array,
            chi_array,
            source_candidate,
            rho_array,
            float(dt),
            edge_array,
        )
        residual = history - facade.jnp.asarray(target_array, dtype=facade.jnp.float64)
        return facade.jnp.mean(
            facade.jnp.asarray(weight_array, dtype=facade.jnp.float64)[None, :, None] * residual * residual
        )

    loss, gradient = facade.jax.value_and_grad(loss_for_sources)(
        facade.jnp.asarray(source_array, dtype=facade.jnp.float64)
    )
    history = facade._transport_rollout_jax(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)
    return TransportRolloutSourceGradients(
        loss=float(np.asarray(loss)),
        source_gradient=np.asarray(gradient),
        final_profiles=np.asarray(history[-1]),
    )


def _rollout_gradient_audit_indices(
    source_shape: tuple[int, ...],
    sample_indices: Any | None,
) -> tuple[tuple[int, int, int], ...]:
    """Return (step, channel, radial) sample indices for a rollout source audit."""
    if len(source_shape) != 3:
        raise ValueError("source_sequence must have shape (n_steps, 4, n_rho)")
    n_steps, n_channels, n_rho = source_shape
    if n_steps < 1 or n_channels != CHANNEL_COUNT or n_rho < 3:
        raise ValueError("source_sequence must have shape (n_steps, 4, n_rho) with n_rho >= 3")
    if sample_indices is None:
        candidates: tuple[tuple[int, int, int], ...] = (
            (0, 0, 1),
            (n_steps - 1, 1, n_rho // 2),
            (n_steps // 2, 2, n_rho - 2),
            (n_steps - 1, 3, max(1, n_rho // 3)),
        )
    else:
        try:
            parsed = []
            for raw_index in sample_indices:
                index = tuple(int(part) for part in raw_index)
                if len(index) != 3:
                    raise ValueError
                parsed.append((index[0], index[1], index[2]))
            candidates = tuple(parsed)
        except (TypeError, ValueError) as exc:
            raise ValueError("sample_indices must contain three-part rollout source indices") from exc
    unique: list[tuple[int, int, int]] = []
    for step, channel, radius in candidates:
        if not (0 <= step < n_steps and 0 <= channel < n_channels and 0 <= radius < n_rho):
            raise ValueError("sample_indices contain an out-of-range rollout source index")
        index = (int(step), int(channel), int(radius))
        if index not in unique:
            unique.append(index)
    if not unique:
        raise ValueError("sample_indices must contain at least one rollout source index")
    return tuple(unique)


def audit_transport_rollout_source_gradients(
    initial_profiles: Any,
    chi: Any,
    source_sequence: Any,
    target_history: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    weights: Any | None = None,
    epsilon: float = 1.0e-5,
    tolerance: float = 5.0e-4,
    sample_indices: Any | None = None,
) -> TransportRolloutGradientAudit:
    """Compare JAX rollout source gradients with sampled finite differences.

    Public gradient and tracking-loss helpers are resolved through the facade
    so production re-exports and test monkeypatches remain authoritative.
    """
    facade = _facade()
    epsilon_float = float(epsilon)
    tolerance_float = float(tolerance)
    if not np.isfinite(epsilon_float) or epsilon_float <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    if not np.isfinite(tolerance_float) or tolerance_float <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array = (
        facade._validate_transport_rollout_inputs(
            initial_profiles,
            chi,
            source_sequence,
            rho,
            dt,
            edge_values,
            target_history=target_history,
            weights=weights,
        )
    )
    if target_array is None:
        raise ValueError("target_history is required")
    if weight_array is None:
        weight_array = np.ones(CHANNEL_COUNT)
    gradient_result = cast(
        TransportRolloutSourceGradients,
        facade.transport_rollout_source_gradients(
            profile_array,
            chi_array,
            source_array,
            target_array,
            rho_array,
            float(dt),
            edge_array,
            weights=weight_array,
        ),
    )
    indices = _rollout_gradient_audit_indices(source_array.shape, sample_indices)
    max_abs_error = 0.0
    for index in indices:
        plus_sources = source_array.copy()
        minus_sources = source_array.copy()
        plus_sources[index] += epsilon_float
        minus_sources[index] -= epsilon_float
        plus_loss = float(
            facade.transport_rollout_tracking_loss(
                profile_array,
                chi_array,
                plus_sources,
                target_array,
                rho_array,
                float(dt),
                edge_array,
                weights=weight_array,
                use_jax=False,
            )
        )
        minus_loss = float(
            facade.transport_rollout_tracking_loss(
                profile_array,
                chi_array,
                minus_sources,
                target_array,
                rho_array,
                float(dt),
                edge_array,
                weights=weight_array,
                use_jax=False,
            )
        )
        finite_difference = (plus_loss - minus_loss) / (2.0 * epsilon_float)
        max_abs_error = max(max_abs_error, abs(float(gradient_result.source_gradient[index]) - finite_difference))
    return TransportRolloutGradientAudit(
        loss=float(gradient_result.loss),
        epsilon=epsilon_float,
        tolerance=tolerance_float,
        checked_indices=indices,
        source_max_abs_error=float(max_abs_error),
        passed=bool(max_abs_error <= tolerance_float),
    )


def assert_transport_rollout_source_gradients_consistent(
    initial_profiles: Any,
    chi: Any,
    source_sequence: Any,
    target_history: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    weights: Any | None = None,
    epsilon: float = 1.0e-5,
    tolerance: float = 5.0e-4,
    sample_indices: Any | None = None,
) -> TransportRolloutGradientAudit:
    """Return rollout source-gradient audit evidence or fail closed."""
    facade = _facade()
    audit = cast(
        TransportRolloutGradientAudit,
        facade.audit_transport_rollout_source_gradients(
            initial_profiles,
            chi,
            source_sequence,
            target_history,
            rho,
            dt,
            edge_values,
            weights=weights,
            epsilon=epsilon,
            tolerance=tolerance,
            sample_indices=sample_indices,
        ),
    )
    if not audit.passed:
        raise ValueError("transport rollout source-gradient audit failed")
    return audit

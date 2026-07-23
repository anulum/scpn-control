# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable transport equilibrium weighting

"""Equilibrium-weighted transport losses, gradients, and radial weight helpers.

This leaf owns GS-flux radial weighting and the one-step / multi-step
equilibrium-weighted tracking losses and JAX gradients used for controller
tuning through equilibrium geometry. Core step and rollout primitives remain
on the numerical facade; this module lazy-imports them at call time so public
re-exports and test monkeypatches of
``scpn_control.core.differentiable_transport`` remain the production path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scpn_control._typing import FloatArray
from scpn_control.core.differentiable_transport_evidence import CHANNEL_COUNT


@dataclass(frozen=True)
class EquilibriumWeightedTransportGradient:
    """JAX gradient of equilibrium-weighted transport tracking loss."""

    loss: float
    chi_gradient: FloatArray
    equilibrium_gradient: FloatArray
    radial_weights: FloatArray


@dataclass(frozen=True)
class EquilibriumWeightedTransportRolloutGradient:
    """JAX gradient of equilibrium-weighted multi-step transport rollout loss."""

    loss: float
    source_gradient: FloatArray
    equilibrium_gradient: FloatArray
    radial_weights: FloatArray
    final_profiles: FloatArray


def _facade() -> Any:
    """Return the public numerical facade (lazy to avoid import cycles)."""
    from scpn_control.core import differentiable_transport as facade

    return facade


def _as_float_array(name: str, value: Any) -> FloatArray:
    """Coerce ``value`` to a finite floating array or raise."""
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_equilibrium_psi(equilibrium_psi: Any) -> FloatArray:
    """Validate a finite two-dimensional Grad-Shafranov flux map."""
    psi = _as_float_array("equilibrium_psi", equilibrium_psi)
    if psi.ndim != 2 or min(psi.shape) < 3:
        raise ValueError("equilibrium_psi must be a finite two-dimensional flux map with both dimensions >= 3")
    return psi


def equilibrium_radial_weights(equilibrium_psi: Any, n_rho: int) -> FloatArray:
    """Return positive mean-one radial weights from a Grad-Shafranov flux map."""
    psi = _validate_equilibrium_psi(equilibrium_psi)
    if isinstance(n_rho, bool) or int(n_rho) != n_rho or int(n_rho) < 3:
        raise ValueError("n_rho must be an integer >= 3")
    radial_profile = np.mean(np.abs(psi), axis=0)
    if radial_profile.size != int(n_rho):
        src = np.linspace(0.0, 1.0, radial_profile.size)
        dst = np.linspace(0.0, 1.0, int(n_rho))
        radial_profile = np.interp(dst, src, radial_profile)
    radial_profile = np.maximum(radial_profile, 0.0)
    mean_profile = float(np.mean(radial_profile))
    if mean_profile <= 1.0e-30:
        return np.ones(int(n_rho))
    weights = np.asarray(radial_profile / mean_profile, dtype=float)
    return weights


def _equilibrium_radial_weights_jax(equilibrium_psi: Any, n_rho: int) -> Any:
    """Return traced mean-one radial weights from a flux map inside the JAX graph."""
    facade = _facade()
    if facade.jnp is None:
        raise RuntimeError("JAX equilibrium weighting requested but JAX is unavailable")
    psi = facade.jnp.asarray(equilibrium_psi, dtype=facade.jnp.float64)
    radial_profile = facade.jnp.mean(facade.jnp.abs(psi), axis=0)
    if int(radial_profile.shape[0]) != int(n_rho):
        src = facade.jnp.linspace(0.0, 1.0, int(radial_profile.shape[0]))
        dst = facade.jnp.linspace(0.0, 1.0, int(n_rho))
        radial_profile = facade.jnp.interp(dst, src, radial_profile)
    radial_profile = facade.jnp.maximum(radial_profile, 0.0)
    mean_profile = facade.jnp.mean(radial_profile)
    return facade.jnp.where(mean_profile <= 1.0e-30, facade.jnp.ones(int(n_rho)), radial_profile / mean_profile)


def _equilibrium_weighted_tracking_loss_jax(
    profiles: Any,
    chi: Any,
    sources: Any,
    target_profiles: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    equilibrium_psi: Any,
    weights: Any,
) -> Any:
    """Return traced one-step GS-weighted MSE tracking loss."""
    facade = _facade()
    if facade.jnp is None:
        raise RuntimeError("JAX equilibrium-weighted tracking loss requested but JAX is unavailable")
    predicted = facade._transport_step_jax(profiles, chi, sources, rho, dt, edge_values)
    radial_weights = _equilibrium_radial_weights_jax(equilibrium_psi, int(predicted.shape[1]))
    residual = predicted - facade.jnp.asarray(target_profiles, dtype=facade.jnp.float64)
    channel_weights = facade.jnp.asarray(weights, dtype=facade.jnp.float64)[:, None]
    return facade.jnp.mean(channel_weights * radial_weights[None, :] * residual * residual)


def equilibrium_weighted_transport_tracking_loss(
    profiles: Any,
    chi: Any,
    sources: Any,
    target_profiles: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    equilibrium_psi: Any,
    *,
    weights: Any | None = None,
    use_jax: bool = True,
    allow_numpy_fallback: bool = False,
    allow_legacy_numpy_fallback: bool = False,
) -> Any:
    """Return transport tracking loss weighted by GS-equilibrium flux geometry."""
    facade = _facade()
    profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array = (
        facade._validate_transport_inputs(
            profiles,
            chi,
            sources,
            rho,
            dt,
            edge_values,
            target_profiles=target_profiles,
            weights=weights,
        )
    )
    if target_array is None:
        raise ValueError("target_profiles is required")
    if weight_array is None:
        weight_array = np.ones(CHANNEL_COUNT)
    psi_array = _validate_equilibrium_psi(equilibrium_psi)
    use_jax_runtime = facade._resolve_use_jax(
        use_jax,
        allow_numpy_fallback=allow_numpy_fallback,
        allow_legacy_numpy_fallback=allow_legacy_numpy_fallback,
        context="equilibrium_weighted_transport_tracking_loss",
    )
    if use_jax_runtime:
        return _equilibrium_weighted_tracking_loss_jax(
            profile_array,
            chi_array,
            source_array,
            target_array,
            rho_array,
            float(dt),
            edge_array,
            psi_array,
            weight_array,
        )
    predicted = facade._transport_step_numpy(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)
    radial_weights = equilibrium_radial_weights(psi_array, profile_array.shape[1])
    residual = predicted - target_array
    return float(np.mean(weight_array[:, None] * radial_weights[None, :] * residual * residual))


def equilibrium_weighted_transport_loss_gradient(
    profiles: Any,
    chi: Any,
    sources: Any,
    target_profiles: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    equilibrium_psi: Any,
    *,
    weights: Any | None = None,
) -> EquilibriumWeightedTransportGradient:
    """Return JAX gradients of equilibrium-weighted transport loss.

    The returned gradients are with respect to the transport coefficients and
    the supplied equilibrium flux map. If the flux map was produced inside an
    outer JAX graph by the Grad-Shafranov solver, this loss is compatible with
    further chain-rule propagation through that equilibrium solve.
    """
    facade = _facade()
    if not facade._HAS_JAX or facade.jax is None or facade.jnp is None:
        raise RuntimeError("equilibrium_weighted_transport_loss_gradient requires JAX")
    profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array = (
        facade._validate_transport_inputs(
            profiles,
            chi,
            sources,
            rho,
            dt,
            edge_values,
            target_profiles=target_profiles,
            weights=weights,
        )
    )
    if target_array is None:
        raise ValueError("target_profiles is required")
    if weight_array is None:
        weight_array = np.ones(CHANNEL_COUNT)
    psi_array = _validate_equilibrium_psi(equilibrium_psi)

    def loss_for_chi_and_equilibrium(chi_candidate: Any, psi_candidate: Any) -> Any:
        return _equilibrium_weighted_tracking_loss_jax(
            profile_array,
            chi_candidate,
            source_array,
            target_array,
            rho_array,
            float(dt),
            edge_array,
            psi_candidate,
            weight_array,
        )

    loss, gradients = facade.jax.value_and_grad(loss_for_chi_and_equilibrium, argnums=(0, 1))(
        facade.jnp.asarray(chi_array, dtype=facade.jnp.float64),
        facade.jnp.asarray(psi_array, dtype=facade.jnp.float64),
    )
    chi_gradient, equilibrium_gradient = gradients
    return EquilibriumWeightedTransportGradient(
        loss=float(np.asarray(loss)),
        chi_gradient=np.asarray(chi_gradient),
        equilibrium_gradient=np.asarray(equilibrium_gradient),
        radial_weights=equilibrium_radial_weights(psi_array, profile_array.shape[1]),
    )


def _equilibrium_weighted_rollout_tracking_loss_jax(
    initial_profiles: Any,
    chi: Any,
    source_sequence: Any,
    target_history: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    equilibrium_psi: Any,
    weights: Any,
) -> Any:
    """Return traced multi-step GS-weighted MSE tracking loss."""
    facade = _facade()
    if facade.jnp is None:
        raise RuntimeError("JAX equilibrium-weighted rollout tracking loss requested but JAX is unavailable")
    history = facade._transport_rollout_jax(initial_profiles, chi, source_sequence, rho, dt, edge_values)
    radial_weights = _equilibrium_radial_weights_jax(equilibrium_psi, int(history.shape[2]))
    residual = history - facade.jnp.asarray(target_history, dtype=facade.jnp.float64)
    channel_weights = facade.jnp.asarray(weights, dtype=facade.jnp.float64)[None, :, None]
    return facade.jnp.mean(channel_weights * radial_weights[None, None, :] * residual * residual)


def equilibrium_weighted_transport_rollout_tracking_loss(
    initial_profiles: Any,
    chi: Any,
    source_sequence: Any,
    target_history: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    equilibrium_psi: Any,
    *,
    weights: Any | None = None,
    use_jax: bool = True,
    allow_numpy_fallback: bool = False,
    allow_legacy_numpy_fallback: bool = False,
) -> Any:
    """Return multi-step transport rollout loss weighted by GS flux geometry."""
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
    psi_array = _validate_equilibrium_psi(equilibrium_psi)
    use_jax_runtime = facade._resolve_use_jax(
        use_jax,
        allow_numpy_fallback=allow_numpy_fallback,
        allow_legacy_numpy_fallback=allow_legacy_numpy_fallback,
        context="equilibrium_weighted_transport_rollout_tracking_loss",
    )
    if use_jax_runtime:
        return _equilibrium_weighted_rollout_tracking_loss_jax(
            profile_array,
            chi_array,
            source_array,
            target_array,
            rho_array,
            float(dt),
            edge_array,
            psi_array,
            weight_array,
        )
    history = facade._transport_rollout_numpy(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)
    radial_weights = equilibrium_radial_weights(psi_array, profile_array.shape[1])
    residual = history - target_array
    return float(np.mean(weight_array[None, :, None] * radial_weights[None, None, :] * residual * residual))


def equilibrium_weighted_transport_rollout_source_gradient(
    initial_profiles: Any,
    chi: Any,
    source_sequence: Any,
    target_history: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    equilibrium_psi: Any,
    *,
    weights: Any | None = None,
) -> EquilibriumWeightedTransportRolloutGradient:
    """Return JAX gradients of GS-weighted rollout loss.

    The returned gradients are with respect to the full source schedule and the
    supplied equilibrium flux map. If the flux map was produced inside an outer
    JAX graph by the Grad-Shafranov solver, this loss is compatible with
    chain-rule propagation through that equilibrium solve.
    """
    facade = _facade()
    if not facade._HAS_JAX or facade.jax is None or facade.jnp is None:
        raise RuntimeError("equilibrium_weighted_transport_rollout_source_gradient requires JAX")
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
    psi_array = _validate_equilibrium_psi(equilibrium_psi)

    def loss_for_sources_and_equilibrium(source_candidate: Any, psi_candidate: Any) -> Any:
        return _equilibrium_weighted_rollout_tracking_loss_jax(
            profile_array,
            chi_array,
            source_candidate,
            target_array,
            rho_array,
            float(dt),
            edge_array,
            psi_candidate,
            weight_array,
        )

    loss, gradients = facade.jax.value_and_grad(loss_for_sources_and_equilibrium, argnums=(0, 1))(
        facade.jnp.asarray(source_array, dtype=facade.jnp.float64),
        facade.jnp.asarray(psi_array, dtype=facade.jnp.float64),
    )
    source_gradient, equilibrium_gradient = gradients
    history = facade._transport_rollout_jax(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)
    return EquilibriumWeightedTransportRolloutGradient(
        loss=float(np.asarray(loss)),
        source_gradient=np.asarray(source_gradient),
        equilibrium_gradient=np.asarray(equilibrium_gradient),
        radial_weights=equilibrium_radial_weights(psi_array, profile_array.shape[1]),
        final_profiles=np.asarray(history[-1]),
    )

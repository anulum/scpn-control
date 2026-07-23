# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable transport parameter AD

"""One-step parameter / loss gradients and finite-difference audit contracts.

This leaf owns tracking loss, chi/source parameter gradients, and the sampled
finite-difference admission audit used by controller tuning. Rollout source
gradients and equilibrium-weighted losses remain on the numerical facade (later
R1 stages). Step primitives and input validators stay on the facade; this module
lazy-imports them at call time so facade re-exports and test monkeypatches of
``scpn_control.core.differentiable_transport`` remain the production path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.core.differentiable_transport_evidence import (
    CHANNEL_COUNT,
    TransportGradientAudit,
)


@dataclass(frozen=True)
class TransportParameterGradients:
    """JAX gradients of transport tracking loss for tunable transport inputs."""

    loss: float
    chi_gradient: FloatArray
    source_gradient: FloatArray


def _facade() -> Any:
    """Return the public numerical facade (lazy to avoid import cycles)."""
    from scpn_control.core import differentiable_transport as facade

    return facade


def _tracking_loss_jax(
    profiles: Any,
    chi: Any,
    sources: Any,
    target_profiles: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    weights: Any,
) -> Any:
    """Return traced one-step weighted MSE tracking loss inside the JAX graph."""
    facade = _facade()
    if facade.jnp is None:
        raise RuntimeError("JAX tracking loss requested but JAX is unavailable")
    predicted = facade._transport_step_jax(profiles, chi, sources, rho, dt, edge_values)
    residual = predicted - facade.jnp.asarray(target_profiles, dtype=facade.jnp.float64)
    return facade.jnp.mean(facade.jnp.asarray(weights, dtype=facade.jnp.float64)[:, None] * residual * residual)


def transport_tracking_loss(
    profiles: Any,
    chi: Any,
    sources: Any,
    target_profiles: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    weights: Any | None = None,
    use_jax: bool = True,
    allow_numpy_fallback: bool = False,
    allow_legacy_numpy_fallback: bool = False,
) -> Any:
    """Return weighted one-step transport tracking loss for controller tuning."""
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
    use_jax_runtime = facade._resolve_use_jax(
        use_jax,
        allow_numpy_fallback=allow_numpy_fallback,
        allow_legacy_numpy_fallback=allow_legacy_numpy_fallback,
        context="transport_tracking_loss",
    )
    if use_jax_runtime:
        return _tracking_loss_jax(
            profile_array,
            chi_array,
            source_array,
            target_array,
            rho_array,
            float(dt),
            edge_array,
            weight_array,
        )
    predicted = facade._transport_step_numpy(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)
    residual = predicted - target_array
    return float(np.mean(weight_array[:, None] * residual * residual))


def transport_loss_gradient(
    profiles: Any,
    chi: Any,
    sources: Any,
    target_profiles: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    weights: Any | None = None,
) -> tuple[float, FloatArray]:
    """Return the tracking loss and JAX gradient with respect to ``chi``."""
    facade = _facade()
    if not facade._HAS_JAX or facade.jax is None or facade.jnp is None:
        raise RuntimeError("transport_loss_gradient requires JAX")
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

    def loss_for_chi(chi_candidate: Any) -> Any:
        return _tracking_loss_jax(
            profile_array,
            chi_candidate,
            source_array,
            target_array,
            rho_array,
            float(dt),
            edge_array,
            weight_array,
        )

    loss, gradient = facade.jax.value_and_grad(loss_for_chi)(facade.jnp.asarray(chi_array, dtype=facade.jnp.float64))
    return float(np.asarray(loss)), np.asarray(gradient)


def transport_parameter_gradients(
    profiles: Any,
    chi: Any,
    sources: Any,
    target_profiles: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    weights: Any | None = None,
) -> TransportParameterGradients:
    """Return JAX gradients with respect to ``chi`` and source schedules.

    This is the controller-tuning primitive for differentiable auxiliary
    heating, fuelling, and impurity-source schedules.  It keeps the same
    four-channel Crank-Nicolson, source-term, core zero-gradient, and edge
    Dirichlet contracts as :func:`differentiable_transport_step`; unlike
    :func:`transport_loss_gradient`, it exposes gradients for both turbulent
    transport coefficients and additive source terms.
    """
    facade = _facade()
    if not facade._HAS_JAX or facade.jax is None or facade.jnp is None:
        raise RuntimeError("transport_parameter_gradients requires JAX")
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

    def loss_for_chi_and_sources(chi_candidate: Any, source_candidate: Any) -> Any:
        return _tracking_loss_jax(
            profile_array,
            chi_candidate,
            source_candidate,
            target_array,
            rho_array,
            float(dt),
            edge_array,
            weight_array,
        )

    loss, gradients = facade.jax.value_and_grad(loss_for_chi_and_sources, argnums=(0, 1))(
        facade.jnp.asarray(chi_array, dtype=facade.jnp.float64),
        facade.jnp.asarray(source_array, dtype=facade.jnp.float64),
    )
    chi_gradient, source_gradient = gradients
    return TransportParameterGradients(
        loss=float(np.asarray(loss)),
        chi_gradient=np.asarray(chi_gradient),
        source_gradient=np.asarray(source_gradient),
    )


def _gradient_audit_indices(
    shape: tuple[int, ...],
    sample_indices: Any | None,
) -> tuple[tuple[int, int], ...]:
    """Return (channel, radial) sample indices for a parameter-gradient audit."""
    if sample_indices is None:
        radial_indices = sorted({1, shape[1] // 2, shape[1] - 2})
        return tuple((channel, radial) for channel in range(shape[0]) for radial in radial_indices)
    indices: list[tuple[int, int]] = []
    for raw_index in sample_indices:
        try:
            channel = int(raw_index[0])
            radial = int(raw_index[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("sample_indices must contain (channel, radial) pairs") from exc
        if channel < 0 or channel >= shape[0] or radial < 0 or radial >= shape[1]:
            raise ValueError("sample_indices contains an out-of-bounds transport index")
        indices.append((channel, radial))
    if not indices:
        raise ValueError("sample_indices must contain at least one transport index")
    return tuple(indices)


def _central_difference_parameter(
    base_loss: float,
    parameter_array: AnyFloatArray,
    index: tuple[int, int],
    epsilon: float,
    loss_fn: Any,
) -> float:
    """Return a one-sided or central finite difference for one parameter index."""
    if parameter_array[index] - epsilon < 0.0:
        plus = parameter_array.copy()
        plus[index] += epsilon
        return (float(loss_fn(plus)) - base_loss) / epsilon
    plus = parameter_array.copy()
    minus = parameter_array.copy()
    plus[index] += epsilon
    minus[index] -= epsilon
    return (float(loss_fn(plus)) - float(loss_fn(minus))) / (2.0 * epsilon)


def audit_transport_parameter_gradients(
    profiles: Any,
    chi: Any,
    sources: Any,
    target_profiles: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    weights: Any | None = None,
    epsilon: float = 1.0e-5,
    tolerance: float = 5.0e-4,
    sample_indices: Any | None = None,
) -> TransportGradientAudit:
    """Audit JAX transport gradients against independent finite differences.

    The audit is intentionally sampled rather than exhaustive so it can run in
    controller-tuning admission checks. It evaluates deterministic interior
    radial points for every channel by default and compares JAX gradients for
    both turbulent transport coefficients and additive source schedules against
    independently perturbed NumPy losses.

    Calls resolve public gradient/loss helpers through the facade so production
    monkeypatches and re-export bindings remain authoritative.
    """
    facade = _facade()
    epsilon_value = float(epsilon)
    tolerance_value = float(tolerance)
    if not np.isfinite(epsilon_value) or epsilon_value <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    if not np.isfinite(tolerance_value) or tolerance_value <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    gradient_result = cast(
        TransportParameterGradients,
        facade.transport_parameter_gradients(
            profiles,
            chi,
            sources,
            target_profiles,
            rho,
            dt,
            edge_values,
            weights=weights,
        ),
    )
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
    indices = _gradient_audit_indices(chi_array.shape, sample_indices)

    def loss_for_chi(candidate: AnyFloatArray) -> float:
        return float(
            facade.transport_tracking_loss(
                profile_array,
                candidate,
                source_array,
                target_array,
                rho_array,
                float(dt),
                edge_array,
                weights=weight_array,
                use_jax=False,
            )
        )

    def loss_for_sources(candidate: AnyFloatArray) -> float:
        return float(
            facade.transport_tracking_loss(
                profile_array,
                chi_array,
                candidate,
                target_array,
                rho_array,
                float(dt),
                edge_array,
                weights=weight_array,
                use_jax=False,
            )
        )

    base_loss = loss_for_chi(chi_array)
    chi_errors: list[float] = []
    source_errors: list[float] = []
    for index in indices:
        chi_fd = _central_difference_parameter(base_loss, chi_array, index, epsilon_value, loss_for_chi)
        source_fd = _central_difference_parameter(base_loss, source_array, index, epsilon_value, loss_for_sources)
        chi_errors.append(abs(float(gradient_result.chi_gradient[index]) - chi_fd))
        source_errors.append(abs(float(gradient_result.source_gradient[index]) - source_fd))
    chi_max_error = max(chi_errors)
    source_max_error = max(source_errors)
    passed = bool(chi_max_error <= tolerance_value and source_max_error <= tolerance_value)
    return TransportGradientAudit(
        loss=gradient_result.loss,
        epsilon=epsilon_value,
        tolerance=tolerance_value,
        checked_indices=indices,
        chi_max_abs_error=float(chi_max_error),
        source_max_abs_error=float(source_max_error),
        passed=passed,
    )


def assert_transport_parameter_gradients_consistent(
    profiles: Any,
    chi: Any,
    sources: Any,
    target_profiles: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    weights: Any | None = None,
    epsilon: float = 1.0e-5,
    tolerance: float = 5.0e-4,
    sample_indices: Any | None = None,
) -> TransportGradientAudit:
    """Return the gradient audit or fail closed when consistency is violated."""
    facade = _facade()
    audit = cast(
        TransportGradientAudit,
        facade.audit_transport_parameter_gradients(
            profiles,
            chi,
            sources,
            target_profiles,
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
        raise ValueError(
            "transport parameter gradient audit failed: "
            f"chi_max_abs_error={audit.chi_max_abs_error:.6g}, "
            f"source_max_abs_error={audit.source_max_abs_error:.6g}, "
            f"tolerance={audit.tolerance:.6g}"
        )
    return audit

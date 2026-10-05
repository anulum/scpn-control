# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Nonlinear MPC transport-model tuning
"""Gradient-based transport-model tuning for NMPC reference tracking.

The nonlinear MPC controller (:mod:`scpn_control.control.nmpc_controller`) tracks
plasma references against a fixed transport model. This module holds the separate
responsibility of *fitting that transport model* to a tracking target through the
differentiable-transport primitives in
:mod:`scpn_control.core.differentiable_transport`.

Four entry points cover the supported tuning surfaces:

``tune_transport_coefficients_for_tracking``
    One gradient step on the diffusivity profile ``chi`` towards a target profile.
``tune_transport_sources_for_tracking``
    One gradient step on a single-step transport source schedule.
``tune_transport_source_rollout_for_tracking``
    One gradient step on a multi-step (rollout) transport source schedule, with an
    optional finite-difference gradient audit.
``tune_neural_transport_closure_for_tracking``
    Coefficient tuning that maps a neural transport closure to diffusivities first.

Every entry point is fail-closed: it requires the optional JAX backend
(:func:`scpn_control.core.differentiable_transport.has_jax`) and, where a gradient
audit is requested, rejects — or, in explicit ``warn`` mode, records — an audit
that does not pass. Results carry campaign-provenance metadata so a tuning step is
reproducible and auditable.

References
----------
Rawlings, Mayne & Diehl 2017, *Model Predictive Control: Theory, Computation, and
Design*, 2nd ed. — the receding-horizon tracking objective the tuned transport
model feeds. Felici et al. 2011, Nucl. Fusion 51, 083052 — real-time MPC for
plasma current-profile and kinetic-variable control.
"""

from __future__ import annotations

import warnings

import numpy as np

from scpn_control._typing import AnyFloatArray
from scpn_control.core.differentiable_transport import (
    TransportParameterGradients,
    TransportRolloutSourceGradients,
    assert_transport_parameter_gradients_consistent,
    transport_campaign_metadata,
    transport_coefficients_from_neural_closure,
    transport_loss_gradient,
    transport_parameter_gradients,
    transport_rollout_source_gradients,
)
from scpn_control.core.differentiable_transport import (
    has_jax as has_differentiable_transport_jax,
)

from .nmpc_transport_contracts import (
    TransportCoefficientTuningResult,
    TransportSourceRolloutGradientAudit,
    TransportSourceRolloutTuningResult,
    TransportSourceScheduleTuningResult,
    _audit_transport_rollout_source_gradients,
    _bounded_tuning_update,
    _optional_finite_array_bound,
    _rollout_audit_indices,
)

__all__ = [
    "TransportCoefficientTuningResult",
    "TransportSourceScheduleTuningResult",
    "TransportSourceRolloutGradientAudit",
    "TransportSourceRolloutTuningResult",
    "tune_transport_coefficients_for_tracking",
    "tune_transport_sources_for_tracking",
    "tune_transport_source_rollout_for_tracking",
    "tune_neural_transport_closure_for_tracking",
    "_audit_transport_rollout_source_gradients",
    "_optional_finite_array_bound",
    "_rollout_audit_indices",
]


def tune_transport_coefficients_for_tracking(
    profiles: AnyFloatArray,
    chi: AnyFloatArray,
    sources: AnyFloatArray,
    target_profiles: AnyFloatArray,
    rho: AnyFloatArray,
    dt: float,
    edge_values: AnyFloatArray,
    *,
    weights: AnyFloatArray | None = None,
    learning_rate: float,
    chi_min: float = 0.0,
    max_fractional_update: float | None = 0.1,
    gradient_tolerance: float | None = None,
    require_gradient_audit: bool = True,
    gradient_audit_epsilon: float = 1.0e-5,
    gradient_audit_tolerance: float = 5.0e-4,
    gradient_audit_sample_indices: object | None = None,
    equilibrium_psi: AnyFloatArray | None = None,
    _closure_for_metadata: object | None = None,
) -> TransportCoefficientTuningResult:
    """Tune transport coefficients for NMPC tracking through JAX autodiff.

    The gradient is taken with respect to the four-channel transport
    coefficient profile used by
    `scpn_control.core.differentiable_transport.transport_loss_gradient`.
    This function intentionally has no finite-difference fallback: coefficient
    tuning is exposed to NMPC only when the differentiable JAX path is present.
    """
    if not has_differentiable_transport_jax():
        raise RuntimeError("tune_transport_coefficients_for_tracking requires JAX")
    learning_rate_float = float(learning_rate)
    chi_min_float = float(chi_min)
    if not np.isfinite(learning_rate_float) or learning_rate_float <= 0.0:
        raise ValueError("learning_rate must be positive and finite.")
    if not np.isfinite(chi_min_float) or chi_min_float < 0.0:
        raise ValueError("chi_min must be non-negative and finite.")
    if max_fractional_update is not None:
        max_fractional_update_float = float(max_fractional_update)
        if not np.isfinite(max_fractional_update_float) or max_fractional_update_float <= 0.0:
            raise ValueError("max_fractional_update must be positive and finite.")
    else:
        max_fractional_update_float = None

    chi_array = np.asarray(chi, dtype=np.float64)
    if chi_array.ndim != 2 or not np.all(np.isfinite(chi_array)) or np.any(chi_array < 0.0):
        raise ValueError("chi must be a finite non-negative two-dimensional array.")

    loss, gradient = transport_loss_gradient(
        profiles,
        chi_array,
        sources,
        target_profiles,
        rho,
        dt,
        edge_values,
        weights=weights,
    )
    gradient_array = np.asarray(gradient, dtype=np.float64)
    if gradient_array.shape != chi_array.shape or not np.all(np.isfinite(gradient_array)):
        raise ValueError("transport gradient must be finite and match chi shape.")
    loss_float = float(loss)
    if not np.isfinite(loss_float):
        raise ValueError("transport loss must be finite.")
    gradient_audit = None
    if require_gradient_audit:
        gradient_audit = assert_transport_parameter_gradients_consistent(
            profiles,
            chi_array,
            sources,
            target_profiles,
            rho,
            dt,
            edge_values,
            weights=weights,
            epsilon=gradient_audit_epsilon,
            tolerance=gradient_audit_tolerance,
            sample_indices=gradient_audit_sample_indices,
        )

    updated_chi, step_norm = _bounded_tuning_update(
        "chi",
        chi_array,
        gradient_array,
        learning_rate_float,
        fractional_cap=max_fractional_update_float,
        lower=chi_min_float,
    )
    metadata = transport_campaign_metadata(
        profiles,
        chi_array,
        sources,
        rho,
        dt,
        edge_values,
        backend="jax",
        closure=_closure_for_metadata,
        gradient_tolerance=gradient_tolerance,
        equilibrium_psi=equilibrium_psi,
    )
    return TransportCoefficientTuningResult(
        loss=loss_float,
        gradient=gradient_array,
        updated_chi=updated_chi,
        step_norm=step_norm,
        metadata=metadata,
        gradient_audit=gradient_audit,
    )


def tune_transport_sources_for_tracking(
    profiles: AnyFloatArray,
    chi: AnyFloatArray,
    sources: AnyFloatArray,
    target_profiles: AnyFloatArray,
    rho: AnyFloatArray,
    dt: float,
    edge_values: AnyFloatArray,
    *,
    weights: AnyFloatArray | None = None,
    learning_rate: float,
    source_min: AnyFloatArray | float | None = None,
    source_max: AnyFloatArray | float | None = None,
    max_absolute_update: float | None = None,
    gradient_tolerance: float | None = None,
    require_gradient_audit: bool = True,
    gradient_audit_epsilon: float = 1.0e-5,
    gradient_audit_tolerance: float = 5.0e-4,
    gradient_audit_sample_indices: object | None = None,
    equilibrium_psi: AnyFloatArray | None = None,
    _closure_for_metadata: object | None = None,
) -> TransportSourceScheduleTuningResult:
    """Tune additive transport source schedules through JAX autodiff.

    This is the NMPC admission path for heating, fuelling, and impurity-source
    schedules. It uses the same differentiable transport loss as coefficient
    tuning, but applies the update to the source array and keeps source bounds
    explicit because sinks or negative feedback terms may be physically valid in
    reduced replay studies.
    """
    if not has_differentiable_transport_jax():
        raise RuntimeError("tune_transport_sources_for_tracking requires JAX")
    learning_rate_float = float(learning_rate)
    if not np.isfinite(learning_rate_float) or learning_rate_float <= 0.0:
        raise ValueError("learning_rate must be positive and finite.")
    if max_absolute_update is not None:
        max_absolute_update_float = float(max_absolute_update)
        if not np.isfinite(max_absolute_update_float) or max_absolute_update_float <= 0.0:
            raise ValueError("max_absolute_update must be positive and finite.")
    else:
        max_absolute_update_float = None

    source_array = np.asarray(sources, dtype=np.float64)
    if source_array.ndim != 2 or not np.all(np.isfinite(source_array)):
        raise ValueError("sources must be a finite two-dimensional array.")
    source_min_array = _optional_finite_array_bound("source_min", source_min, source_array.shape)
    source_max_array = _optional_finite_array_bound("source_max", source_max, source_array.shape)
    if source_min_array is not None and source_max_array is not None and np.any(source_min_array > source_max_array):
        raise ValueError("source_min entries must be less than or equal to source_max entries.")

    gradient_result = transport_parameter_gradients(
        profiles,
        chi,
        source_array,
        target_profiles,
        rho,
        dt,
        edge_values,
        weights=weights,
    )
    if not isinstance(gradient_result, TransportParameterGradients):
        raise ValueError("transport_parameter_gradients must return TransportParameterGradients.")
    loss_float = float(gradient_result.loss)
    if not np.isfinite(loss_float):
        raise ValueError("transport loss must be finite.")
    source_gradient = np.asarray(gradient_result.source_gradient, dtype=np.float64)
    if source_gradient.shape != source_array.shape or not np.all(np.isfinite(source_gradient)):
        raise ValueError("source gradient must be finite and match source shape.")
    gradient_audit = None
    if require_gradient_audit:
        gradient_audit = assert_transport_parameter_gradients_consistent(
            profiles,
            chi,
            source_array,
            target_profiles,
            rho,
            dt,
            edge_values,
            weights=weights,
            epsilon=gradient_audit_epsilon,
            tolerance=gradient_audit_tolerance,
            sample_indices=gradient_audit_sample_indices,
        )

    updated_sources, step_norm = _bounded_tuning_update(
        "sources",
        source_array,
        source_gradient,
        learning_rate_float,
        absolute_cap=max_absolute_update_float,
        lower=source_min_array,
        upper=source_max_array,
    )
    metadata = transport_campaign_metadata(
        profiles,
        chi,
        source_array,
        rho,
        dt,
        edge_values,
        backend="jax",
        closure=_closure_for_metadata,
        gradient_tolerance=gradient_tolerance,
        equilibrium_psi=equilibrium_psi,
    )
    return TransportSourceScheduleTuningResult(
        loss=loss_float,
        gradient=source_gradient,
        updated_sources=updated_sources,
        step_norm=step_norm,
        metadata=metadata,
        gradient_audit=gradient_audit,
    )


def tune_transport_source_rollout_for_tracking(
    initial_profiles: AnyFloatArray,
    chi: AnyFloatArray,
    source_sequence: AnyFloatArray,
    target_history: AnyFloatArray,
    rho: AnyFloatArray,
    dt: float,
    edge_values: AnyFloatArray,
    *,
    weights: AnyFloatArray | None = None,
    learning_rate: float,
    source_min: AnyFloatArray | float | None = None,
    source_max: AnyFloatArray | float | None = None,
    max_absolute_update: float | None = None,
    gradient_tolerance: float | None = None,
    require_gradient_audit: bool = True,
    gradient_audit_failure_mode: str = "raise",
    gradient_audit_epsilon: float = 1.0e-5,
    gradient_audit_tolerance: float = 5.0e-4,
    gradient_audit_sample_indices: object | None = None,
    equilibrium_psi: AnyFloatArray | None = None,
    _closure_for_metadata: object | None = None,
) -> TransportSourceRolloutTuningResult:
    """Tune a full NMPC transport source rollout through JAX autodiff.

    The update acts on the complete `(n_steps, 4, n_rho)` heating, fuelling,
    and impurity-source schedule. A sampled NumPy finite-difference audit is
    required by default so the controller does not admit unaudited JAX
    gradients for multi-step source optimisation.
    """
    if not has_differentiable_transport_jax():
        raise RuntimeError("tune_transport_source_rollout_for_tracking requires JAX")
    if gradient_audit_failure_mode not in {"raise", "warn"}:
        raise ValueError("gradient_audit_failure_mode must be 'raise' or 'warn'.")
    learning_rate_float = float(learning_rate)
    if not np.isfinite(learning_rate_float) or learning_rate_float <= 0.0:
        raise ValueError("learning_rate must be positive and finite.")
    if max_absolute_update is not None:
        max_absolute_update_float = float(max_absolute_update)
        if not np.isfinite(max_absolute_update_float) or max_absolute_update_float <= 0.0:
            raise ValueError("max_absolute_update must be positive and finite.")
    else:
        max_absolute_update_float = None

    source_array = np.asarray(source_sequence, dtype=np.float64)
    if source_array.ndim != 3 or source_array.shape[1] != 4 or not np.all(np.isfinite(source_array)):
        raise ValueError("source_sequence must be a finite array with shape (n_steps, 4, n_rho).")
    source_min_array = _optional_finite_array_bound("source_min", source_min, source_array.shape)
    source_max_array = _optional_finite_array_bound("source_max", source_max, source_array.shape)
    if source_min_array is not None and source_max_array is not None and np.any(source_min_array > source_max_array):
        raise ValueError("source_min entries must be less than or equal to source_max entries.")

    gradient_result = transport_rollout_source_gradients(
        initial_profiles,
        chi,
        source_array,
        target_history,
        rho,
        dt,
        edge_values,
        weights=weights,
    )
    if not isinstance(gradient_result, TransportRolloutSourceGradients):
        raise ValueError("transport_rollout_source_gradients must return TransportRolloutSourceGradients.")
    loss_float = float(gradient_result.loss)
    if not np.isfinite(loss_float):
        raise ValueError("transport loss must be finite.")
    source_gradient = np.asarray(gradient_result.source_gradient, dtype=np.float64)
    if source_gradient.shape != source_array.shape or not np.all(np.isfinite(source_gradient)):
        raise ValueError("rollout source gradient must be finite and match source_sequence shape.")
    final_profiles = np.asarray(gradient_result.final_profiles, dtype=np.float64)
    if final_profiles.shape != source_array.shape[1:] or not np.all(np.isfinite(final_profiles)):
        raise ValueError("rollout final profiles must be finite and match one transport profile shape.")
    gradient_audit = None
    if require_gradient_audit:
        gradient_audit = _audit_transport_rollout_source_gradients(
            np.asarray(initial_profiles, dtype=np.float64),
            np.asarray(chi, dtype=np.float64),
            source_array,
            np.asarray(target_history, dtype=np.float64),
            np.asarray(rho, dtype=np.float64),
            dt,
            np.asarray(edge_values, dtype=np.float64),
            source_gradient,
            weights=None if weights is None else np.asarray(weights, dtype=np.float64),
            epsilon=gradient_audit_epsilon,
            tolerance=gradient_audit_tolerance,
            sample_indices=gradient_audit_sample_indices,
        )
        if not gradient_audit.passed:
            if gradient_audit_failure_mode == "raise":
                raise ValueError("rollout source gradient audit failed.")
            warnings.warn(
                "rollout source gradient audit failed; proceeding is advisory-only and must not be used "
                "for production control admission",
                RuntimeWarning,
                stacklevel=2,
            )

    updated_sources, step_norm = _bounded_tuning_update(
        "rollout sources",
        source_array,
        source_gradient,
        learning_rate_float,
        absolute_cap=max_absolute_update_float,
        lower=source_min_array,
        upper=source_max_array,
    )
    metadata = transport_campaign_metadata(
        initial_profiles,
        chi,
        source_array[0],
        rho,
        dt,
        edge_values,
        backend="jax",
        closure=_closure_for_metadata,
        gradient_tolerance=gradient_tolerance,
        equilibrium_psi=equilibrium_psi,
    )
    return TransportSourceRolloutTuningResult(
        loss=loss_float,
        gradient=source_gradient,
        updated_sources=updated_sources,
        final_profiles=final_profiles,
        step_norm=step_norm,
        metadata=metadata,
        gradient_audit=gradient_audit,
    )


def tune_neural_transport_closure_for_tracking(
    profiles: AnyFloatArray,
    closure: object,
    sources: AnyFloatArray,
    target_profiles: AnyFloatArray,
    rho: AnyFloatArray,
    dt: float,
    edge_values: AnyFloatArray,
    *,
    weights: AnyFloatArray | None = None,
    learning_rate: float,
    impurity_diffusivity_fraction: float = 1.0,
    chi_min: float = 0.0,
    max_fractional_update: float | None = 0.1,
    gradient_tolerance: float | None = None,
    require_gradient_audit: bool = True,
    gradient_audit_epsilon: float = 1.0e-5,
    gradient_audit_tolerance: float = 5.0e-4,
    gradient_audit_sample_indices: object | None = None,
    equilibrium_psi: AnyFloatArray | None = None,
) -> TransportCoefficientTuningResult:
    """Tune NMPC transport coefficients initialised from a neural closure."""
    chi = transport_coefficients_from_neural_closure(
        closure,
        impurity_diffusivity_fraction=impurity_diffusivity_fraction,
        chi_floor=chi_min,
    )
    return tune_transport_coefficients_for_tracking(
        profiles,
        chi,
        sources,
        target_profiles,
        rho,
        dt,
        edge_values,
        weights=weights,
        learning_rate=learning_rate,
        chi_min=chi_min,
        max_fractional_update=max_fractional_update,
        gradient_tolerance=gradient_tolerance,
        require_gradient_audit=require_gradient_audit,
        gradient_audit_epsilon=gradient_audit_epsilon,
        gradient_audit_tolerance=gradient_audit_tolerance,
        gradient_audit_sample_indices=gradient_audit_sample_indices,
        equilibrium_psi=equilibrium_psi,
        _closure_for_metadata=closure,
    )

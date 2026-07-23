# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable transport facade

"""JAX-first differentiable multi-channel transport facade.

The facade advances electron temperature, ion temperature, electron density,
and impurity density through the existing cylindrical Crank-Nicolson transport
primitive. JAX mode keeps the full step and tracking loss inside the traced
graph so controller tuning can differentiate transport-coefficient schedules.
The NumPy path is deterministic and intentionally does not claim gradients.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.core import differentiable_transport_closures as _closures
from scpn_control.core import differentiable_transport_evidence as _evidence
from scpn_control.core import jax_solvers as _jax_solvers
from scpn_control.core.differentiable_transport_evidence import (
    CHANNEL_COUNT,
    CHANNELS,
)

# Stable public re-exports: campaign / claim / latency evidence leaf.
TransportCampaignMetadata = _evidence.TransportCampaignMetadata
TransportDifferentiabilityEvidence = _evidence.TransportDifferentiabilityEvidence
TransportFullFidelityReadinessEvidence = _evidence.TransportFullFidelityReadinessEvidence
TransportGradientAudit = _evidence.TransportGradientAudit
TransportGradientLatencyReport = _evidence.TransportGradientLatencyReport
TransportRolloutGradientAudit = _evidence.TransportRolloutGradientAudit
TransportRolloutGradientLatencyReport = _evidence.TransportRolloutGradientLatencyReport
TransportRuntimeMetadata = _evidence.TransportRuntimeMetadata
assert_transport_differentiability_claim_admissible = _evidence.assert_transport_differentiability_claim_admissible
assert_transport_full_fidelity_claim_ready = _evidence.assert_transport_full_fidelity_claim_ready
load_transport_campaign_metadata = _evidence.load_transport_campaign_metadata
save_transport_campaign_metadata = _evidence.save_transport_campaign_metadata
save_transport_gradient_latency_report = _evidence.save_transport_gradient_latency_report
save_transport_rollout_gradient_latency_report = _evidence.save_transport_rollout_gradient_latency_report
transport_differentiability_evidence = _evidence.transport_differentiability_evidence
transport_full_fidelity_readiness_evidence = _evidence.transport_full_fidelity_readiness_evidence
_assert_latency_report_matches_campaign = _evidence._assert_latency_report_matches_campaign
_metadata_field_matches = _evidence._metadata_field_matches
_require_int = _evidence._require_int
_require_nonnegative_finite = _evidence._require_nonnegative_finite
_transport_campaign_metadata_from_mapping = _evidence._transport_campaign_metadata_from_mapping
_validate_parameter_audit_indices = _evidence._validate_parameter_audit_indices
_validate_rollout_audit_indices = _evidence._validate_rollout_audit_indices
_validate_transport_gradient_audit = _evidence._validate_transport_gradient_audit
_validate_transport_gradient_latency_report = _evidence._validate_transport_gradient_latency_report
_validate_transport_rollout_gradient_latency_report = _evidence._validate_transport_rollout_gradient_latency_report
_validate_transport_runtime_metadata = _evidence._validate_transport_runtime_metadata

# Stable public re-exports: neural / reduced-GK closure → coefficient channels.
GyrokineticTransportClosureResult = _closures.GyrokineticTransportClosureResult
transport_coefficients_from_neural_closure = _closures.transport_coefficients_from_neural_closure
gyrokinetic_transport_closure_profiles = _closures.gyrokinetic_transport_closure_profiles
transport_coefficients_from_gyrokinetic_closure = _closures.transport_coefficients_from_gyrokinetic_closure
_closure_profile = _closures._closure_profile
_closure_channel_weights = _closures._closure_channel_weights
_three_channel_transport_coefficients_from_closure = _closures._three_channel_transport_coefficients_from_closure

# Stable public re-exports: local latency benchmarks (lazy-import AD from facade).
from scpn_control.core import differentiable_transport_latency as _latency

transport_runtime_metadata = _latency.transport_runtime_metadata
benchmark_transport_parameter_gradient_latency = _latency.benchmark_transport_parameter_gradient_latency
benchmark_transport_rollout_source_gradient_latency = _latency.benchmark_transport_rollout_source_gradient_latency
_percentile = _latency._percentile

# Stable public re-exports: one-step parameter / loss gradients + FD audit.
from scpn_control.core import differentiable_transport_parameter_ad as _parameter_ad

TransportParameterGradients = _parameter_ad.TransportParameterGradients
transport_tracking_loss = _parameter_ad.transport_tracking_loss
transport_loss_gradient = _parameter_ad.transport_loss_gradient
transport_parameter_gradients = _parameter_ad.transport_parameter_gradients
audit_transport_parameter_gradients = _parameter_ad.audit_transport_parameter_gradients
assert_transport_parameter_gradients_consistent = _parameter_ad.assert_transport_parameter_gradients_consistent
_tracking_loss_jax = _parameter_ad._tracking_loss_jax
_gradient_audit_indices = _parameter_ad._gradient_audit_indices
_central_difference_parameter = _parameter_ad._central_difference_parameter

# Stable public re-exports: multi-step rollout source gradients + FD audit.
from scpn_control.core import differentiable_transport_rollout_ad as _rollout_ad

TransportRolloutSourceGradients = _rollout_ad.TransportRolloutSourceGradients
transport_rollout_tracking_loss = _rollout_ad.transport_rollout_tracking_loss
transport_rollout_source_gradients = _rollout_ad.transport_rollout_source_gradients
audit_transport_rollout_source_gradients = _rollout_ad.audit_transport_rollout_source_gradients
assert_transport_rollout_source_gradients_consistent = _rollout_ad.assert_transport_rollout_source_gradients_consistent
_rollout_gradient_audit_indices = _rollout_ad._rollout_gradient_audit_indices


try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    _HAS_JAX = True
except Exception:
    jax = None
    jnp = cast(Any, None)  # optional-dep fallback (keeps jnp.* annotations typed)
    _HAS_JAX = False


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


def has_jax() -> bool:
    """Return whether the differentiable JAX transport path is available."""
    return _HAS_JAX


def _as_float_array(name: str, value: Any) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_transport_inputs(
    profiles: Any,
    chi: Any,
    sources: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    target_profiles: Any | None = None,
    weights: Any | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray | None, FloatArray | None]:
    profile_array = _as_float_array("profiles", profiles)
    chi_array = _as_float_array("chi", chi)
    source_array = _as_float_array("sources", sources)
    rho_array = _as_float_array("rho", rho)
    edge_array = _as_float_array("edge_values", edge_values)

    if profile_array.ndim != 2 or profile_array.shape[0] != CHANNEL_COUNT:
        raise ValueError(f"profiles must have shape ({CHANNEL_COUNT}, n_rho)")
    if chi_array.shape != profile_array.shape:
        raise ValueError("chi must match profiles shape")
    if source_array.shape != profile_array.shape:
        raise ValueError("sources must match profiles shape")
    if rho_array.ndim != 1 or rho_array.shape[0] != profile_array.shape[1] or rho_array.shape[0] < 3:
        raise ValueError("rho must be one-dimensional with the same radial length as profiles")
    if edge_array.shape != (CHANNEL_COUNT,):
        raise ValueError(f"edge_values must have shape ({CHANNEL_COUNT},)")
    if float(dt) <= 0.0 or not np.isfinite(float(dt)):
        raise ValueError("dt must be positive and finite")
    if np.any(chi_array < 0.0):
        raise ValueError("chi must be non-negative")

    rho_steps = np.diff(rho_array)
    if np.any(rho_steps <= 0.0):
        raise ValueError("rho must be strictly increasing")
    if not np.allclose(rho_steps, rho_steps[0], rtol=1.0e-9, atol=1.0e-12):
        raise ValueError("rho must use a uniform normalised radial spacing")

    target_array = None
    if target_profiles is not None:
        target_array = _as_float_array("target_profiles", target_profiles)
        if target_array.shape != profile_array.shape:
            raise ValueError("target_profiles must match profiles shape")

    weight_array = None
    if weights is not None:
        weight_array = _as_float_array("weights", weights)
        if weight_array.shape != (CHANNEL_COUNT,):
            raise ValueError(f"weights must have shape ({CHANNEL_COUNT},)")
        if np.any(weight_array < 0.0):
            raise ValueError("weights must be non-negative")

    return profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array


def transport_campaign_metadata(
    profiles: Any,
    chi: Any,
    sources: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    backend: str,
    closure: Any | None = None,
    gradient_tolerance: float | None = None,
    equilibrium_psi: Any | None = None,
) -> TransportCampaignMetadata:
    """Return serialisable provenance for differentiable transport campaigns."""
    backend_value = str(backend).strip().lower()
    if backend_value not in {"numpy", "jax"}:
        raise ValueError("backend must be either 'numpy' or 'jax'")
    tolerance_value: float | None = None
    if gradient_tolerance is not None:
        tolerance_value = float(gradient_tolerance)
        if not np.isfinite(tolerance_value) or tolerance_value <= 0.0:
            raise ValueError("gradient_tolerance must be positive and finite")

    profile_array, chi_array, source_array, rho_array, edge_array, _, _ = _validate_transport_inputs(
        profiles,
        chi,
        sources,
        rho,
        dt,
        edge_values,
    )
    dtype_name = np.result_type(
        profile_array.dtype,
        chi_array.dtype,
        source_array.dtype,
        rho_array.dtype,
        edge_array.dtype,
    ).name
    closure_source: str | None = None
    closure_weights_checksum: str | None = None
    if closure is not None:
        closure_source = str(closure.source)
        checksum = closure.weights_checksum
        closure_weights_checksum = None if checksum is None else str(checksum)
    equilibrium_grid_shape: tuple[int, int] | None = None
    if equilibrium_psi is not None:
        psi_array = _validate_equilibrium_psi(equilibrium_psi)
        equilibrium_grid_shape = (int(psi_array.shape[0]), int(psi_array.shape[1]))

    return TransportCampaignMetadata(
        backend=backend_value,
        dtype=dtype_name,
        channel_order=CHANNELS,
        n_rho=int(rho_array.size),
        rho_min=float(rho_array[0]),
        rho_max=float(rho_array[-1]),
        rho_spacing=float(rho_array[1] - rho_array[0]),
        dt=float(dt),
        core_boundary="zero_gradient",
        edge_boundary="dirichlet",
        edge_values=tuple(float(x) for x in edge_array),
        closure_source=closure_source,
        closure_weights_checksum=closure_weights_checksum,
        gradient_tolerance=tolerance_value,
        equilibrium_grid_shape=equilibrium_grid_shape,
    )


def assert_transport_campaign_metadata_replay(
    archived: TransportCampaignMetadata,
    profiles: Any,
    chi: Any,
    sources: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    backend: str,
    closure: Any | None = None,
    gradient_tolerance: float | None = None,
    equilibrium_psi: Any | None = None,
) -> TransportCampaignMetadata:
    """Validate that a candidate transport setup matches archived metadata.

    This guard is intended for replaying differentiable transport tuning
    campaigns. It fails closed on backend, dtype, grid, timestep, boundary,
    closure-provenance, gradient-tolerance, or equilibrium-shape drift before a
    controller rerun can silently compare against a different physics setup.
    """
    if not isinstance(archived, TransportCampaignMetadata):
        raise ValueError("archived transport campaign metadata must be TransportCampaignMetadata")
    current = transport_campaign_metadata(
        profiles,
        chi,
        sources,
        rho,
        dt,
        edge_values,
        backend=backend,
        closure=closure,
        gradient_tolerance=gradient_tolerance,
        equilibrium_psi=equilibrium_psi,
    )
    archived_fields = asdict(archived)
    current_fields = asdict(current)
    mismatches = [
        field_name
        for field_name, archived_value in archived_fields.items()
        if not _metadata_field_matches(archived_value, current_fields[field_name])
    ]
    if mismatches:
        raise ValueError("transport campaign metadata replay mismatch: " + ", ".join(mismatches))
    return current


def _resolve_use_jax(
    use_jax: bool,
    *,
    allow_numpy_fallback: bool,
    allow_legacy_numpy_fallback: bool,
    context: str,
) -> bool:
    if not use_jax:
        return False
    if _HAS_JAX:
        return True
    return _jax_solvers._resolve_use_jax(
        use_jax,
        allow_numpy_fallback=allow_numpy_fallback,
        allow_legacy_numpy_fallback=allow_legacy_numpy_fallback,
        context=context,
    )


def _transport_step_numpy(
    profiles: AnyFloatArray,
    chi: AnyFloatArray,
    sources: AnyFloatArray,
    rho: AnyFloatArray,
    dt: float,
    edge_values: AnyFloatArray,
) -> FloatArray:
    drho = float(rho[1] - rho[0])
    return np.stack(
        [
            _jax_solvers.crank_nicolson_step(
                profiles[channel],
                chi[channel],
                sources[channel],
                rho,
                drho,
                float(dt),
                float(edge_values[channel]),
                use_jax=False,
            )
            for channel in range(CHANNEL_COUNT)
        ]
    )


def _transport_step_jax(
    profiles: Any,
    chi: Any,
    sources: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
) -> Any:
    if jnp is None or jax is None:
        raise RuntimeError("JAX transport step requested but JAX is unavailable")
    rho_jax = jnp.asarray(rho, dtype=jnp.float64)
    drho = rho_jax[1] - rho_jax[0]
    step = jax.vmap(_jax_solvers._cn_step_jax, in_axes=(0, 0, 0, None, None, None, 0))
    return step(
        jnp.asarray(profiles, dtype=jnp.float64),
        jnp.asarray(chi, dtype=jnp.float64),
        jnp.asarray(sources, dtype=jnp.float64),
        rho_jax,
        drho,
        float(dt),
        jnp.asarray(edge_values, dtype=jnp.float64),
    )


def _validate_transport_rollout_inputs(
    initial_profiles: Any,
    chi: Any,
    source_sequence: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    target_history: Any | None = None,
    weights: Any | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray | None, FloatArray | None]:
    profile_array = _as_float_array("initial_profiles", initial_profiles)
    chi_array = _as_float_array("chi", chi)
    source_array = _as_float_array("source_sequence", source_sequence)
    rho_array = _as_float_array("rho", rho)
    edge_array = _as_float_array("edge_values", edge_values)

    if profile_array.ndim != 2 or profile_array.shape[0] != CHANNEL_COUNT:
        raise ValueError(f"initial_profiles must have shape ({CHANNEL_COUNT}, n_rho)")
    if chi_array.shape != profile_array.shape:
        raise ValueError("chi must match initial_profiles shape")
    if source_array.ndim != 3 or source_array.shape[1:] != profile_array.shape or source_array.shape[0] < 1:
        raise ValueError("source_sequence must have shape (n_steps, 4, n_rho) with n_steps >= 1")
    if rho_array.ndim != 1 or rho_array.shape[0] != profile_array.shape[1] or rho_array.shape[0] < 3:
        raise ValueError("rho must be one-dimensional with the same radial length as initial_profiles")
    if edge_array.shape != (CHANNEL_COUNT,):
        raise ValueError(f"edge_values must have shape ({CHANNEL_COUNT},)")
    if float(dt) <= 0.0 or not np.isfinite(float(dt)):
        raise ValueError("dt must be positive and finite")
    if np.any(chi_array < 0.0):
        raise ValueError("chi must be non-negative")
    rho_steps = np.diff(rho_array)
    if np.any(rho_steps <= 0.0):
        raise ValueError("rho must be strictly increasing")
    if not np.allclose(rho_steps, rho_steps[0], rtol=1.0e-9, atol=1.0e-12):
        raise ValueError("rho must use a uniform normalised radial spacing")

    target_array = None
    if target_history is not None:
        target_array = _as_float_array("target_history", target_history)
        if target_array.shape != source_array.shape:
            raise ValueError("target_history must match source_sequence shape")

    weight_array = None
    if weights is not None:
        weight_array = _as_float_array("weights", weights)
        if weight_array.shape != (CHANNEL_COUNT,):
            raise ValueError(f"weights must have shape ({CHANNEL_COUNT},)")
        if np.any(weight_array < 0.0):
            raise ValueError("weights must be non-negative")

    return profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array


def _transport_rollout_numpy(
    initial_profiles: AnyFloatArray,
    chi: AnyFloatArray,
    source_sequence: AnyFloatArray,
    rho: AnyFloatArray,
    dt: float,
    edge_values: AnyFloatArray,
) -> FloatArray:
    current = initial_profiles
    history: list[FloatArray] = []
    for source_step in source_sequence:
        current = _transport_step_numpy(current, chi, source_step, rho, dt, edge_values)
        history.append(current)
    return np.stack(history)


def _transport_rollout_jax(
    initial_profiles: Any,
    chi: Any,
    source_sequence: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
) -> Any:
    if jnp is None or jax is None:
        raise RuntimeError("JAX transport rollout requested but JAX is unavailable")
    chi_jax = jnp.asarray(chi, dtype=jnp.float64)
    rho_jax = jnp.asarray(rho, dtype=jnp.float64)
    edge_jax = jnp.asarray(edge_values, dtype=jnp.float64)

    def body(carry: Any, source_step: Any) -> tuple[Any, Any]:
        next_profiles = _transport_step_jax(carry, chi_jax, source_step, rho_jax, dt, edge_jax)
        return next_profiles, next_profiles

    _, history = jax.lax.scan(
        body,
        jnp.asarray(initial_profiles, dtype=jnp.float64),
        jnp.asarray(source_sequence, dtype=jnp.float64),
    )
    return history


def differentiable_transport_rollout(
    initial_profiles: Any,
    chi: Any,
    source_sequence: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    use_jax: bool = True,
    allow_numpy_fallback: bool = False,
    allow_legacy_numpy_fallback: bool = False,
) -> Any:
    """Advance a time-series of four-channel transport source schedules.

    The returned array has shape ``(n_steps, 4, n_rho)``. Transport
    coefficients are held fixed over the rollout, while ``source_sequence``
    supplies differentiable additive heating, fuelling, and impurity-source
    schedules at each step. This is a bounded controller-tuning primitive, not
    an externally validated integrated-transport campaign.
    """
    profile_array, chi_array, source_array, rho_array, edge_array, _, _ = _validate_transport_rollout_inputs(
        initial_profiles,
        chi,
        source_sequence,
        rho,
        dt,
        edge_values,
    )
    use_jax_runtime = _resolve_use_jax(
        use_jax,
        allow_numpy_fallback=allow_numpy_fallback,
        allow_legacy_numpy_fallback=allow_legacy_numpy_fallback,
        context="differentiable_transport_rollout",
    )
    if use_jax_runtime:
        return _transport_rollout_jax(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)
    return _transport_rollout_numpy(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)


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
    if jnp is None:
        raise RuntimeError("JAX equilibrium-weighted rollout tracking loss requested but JAX is unavailable")
    history = _transport_rollout_jax(initial_profiles, chi, source_sequence, rho, dt, edge_values)
    radial_weights = _equilibrium_radial_weights_jax(equilibrium_psi, int(history.shape[2]))
    residual = history - jnp.asarray(target_history, dtype=jnp.float64)
    channel_weights = jnp.asarray(weights, dtype=jnp.float64)[None, :, None]
    return jnp.mean(channel_weights * radial_weights[None, None, :] * residual * residual)


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
    profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array = (
        _validate_transport_rollout_inputs(
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
    use_jax_runtime = _resolve_use_jax(
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
    history = _transport_rollout_numpy(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)
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
    if not _HAS_JAX or jax is None or jnp is None:
        raise RuntimeError("equilibrium_weighted_transport_rollout_source_gradient requires JAX")
    profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array = (
        _validate_transport_rollout_inputs(
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

    loss, gradients = jax.value_and_grad(loss_for_sources_and_equilibrium, argnums=(0, 1))(
        jnp.asarray(source_array, dtype=jnp.float64),
        jnp.asarray(psi_array, dtype=jnp.float64),
    )
    source_gradient, equilibrium_gradient = gradients
    history = _transport_rollout_jax(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)
    return EquilibriumWeightedTransportRolloutGradient(
        loss=float(np.asarray(loss)),
        source_gradient=np.asarray(source_gradient),
        equilibrium_gradient=np.asarray(equilibrium_gradient),
        radial_weights=equilibrium_radial_weights(psi_array, profile_array.shape[1]),
        final_profiles=np.asarray(history[-1]),
    )


def _validate_equilibrium_psi(equilibrium_psi: Any) -> FloatArray:
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
    if jnp is None:
        raise RuntimeError("JAX equilibrium weighting requested but JAX is unavailable")
    psi = jnp.asarray(equilibrium_psi, dtype=jnp.float64)
    radial_profile = jnp.mean(jnp.abs(psi), axis=0)
    if int(radial_profile.shape[0]) != int(n_rho):
        src = jnp.linspace(0.0, 1.0, int(radial_profile.shape[0]))
        dst = jnp.linspace(0.0, 1.0, int(n_rho))
        radial_profile = jnp.interp(dst, src, radial_profile)
    radial_profile = jnp.maximum(radial_profile, 0.0)
    mean_profile = jnp.mean(radial_profile)
    return jnp.where(mean_profile <= 1.0e-30, jnp.ones(int(n_rho)), radial_profile / mean_profile)


def differentiable_transport_step(
    profiles: Any,
    chi: Any,
    sources: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    use_jax: bool = True,
    allow_numpy_fallback: bool = False,
    allow_legacy_numpy_fallback: bool = False,
) -> Any:
    """Advance four transport channels by one differentiable radial step.

    Channel order is electron temperature, ion temperature, electron density,
    and impurity density. The radial coordinate is a strictly increasing,
    uniformly spaced normalised axis from core-side interior to edge. The core
    boundary uses the inherited zero-gradient condition, and each channel uses
    its supplied Dirichlet edge value.
    """
    profile_array, chi_array, source_array, rho_array, edge_array, _, _ = _validate_transport_inputs(
        profiles,
        chi,
        sources,
        rho,
        dt,
        edge_values,
    )
    use_jax_runtime = _resolve_use_jax(
        use_jax,
        allow_numpy_fallback=allow_numpy_fallback,
        allow_legacy_numpy_fallback=allow_legacy_numpy_fallback,
        context="differentiable_transport_step",
    )
    if use_jax_runtime:
        return _transport_step_jax(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)
    return _transport_step_numpy(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)


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
    if jnp is None:
        raise RuntimeError("JAX equilibrium-weighted tracking loss requested but JAX is unavailable")
    predicted = _transport_step_jax(profiles, chi, sources, rho, dt, edge_values)
    radial_weights = _equilibrium_radial_weights_jax(equilibrium_psi, int(predicted.shape[1]))
    residual = predicted - jnp.asarray(target_profiles, dtype=jnp.float64)
    channel_weights = jnp.asarray(weights, dtype=jnp.float64)[:, None]
    return jnp.mean(channel_weights * radial_weights[None, :] * residual * residual)


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
    profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array = (
        _validate_transport_inputs(
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
    use_jax_runtime = _resolve_use_jax(
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
    predicted = _transport_step_numpy(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)
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
    if not _HAS_JAX or jax is None or jnp is None:
        raise RuntimeError("equilibrium_weighted_transport_loss_gradient requires JAX")
    profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array = (
        _validate_transport_inputs(
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

    loss, gradients = jax.value_and_grad(loss_for_chi_and_equilibrium, argnums=(0, 1))(
        jnp.asarray(chi_array, dtype=jnp.float64),
        jnp.asarray(psi_array, dtype=jnp.float64),
    )
    chi_gradient, equilibrium_gradient = gradients
    return EquilibriumWeightedTransportGradient(
        loss=float(np.asarray(loss)),
        chi_gradient=np.asarray(chi_gradient),
        equilibrium_gradient=np.asarray(equilibrium_gradient),
        radial_weights=equilibrium_radial_weights(psi_array, profile_array.shape[1]),
    )

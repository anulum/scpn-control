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

from dataclasses import asdict
from typing import Any, cast

import numpy as np

from scpn_control._typing import FloatArray
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

# Stable public re-exports: equilibrium-weighted losses / gradients / radial weights.
from scpn_control.core import differentiable_transport_equilibrium_weight as _eq_weight

EquilibriumWeightedTransportGradient = _eq_weight.EquilibriumWeightedTransportGradient
EquilibriumWeightedTransportRolloutGradient = _eq_weight.EquilibriumWeightedTransportRolloutGradient
equilibrium_radial_weights = _eq_weight.equilibrium_radial_weights
equilibrium_weighted_transport_tracking_loss = _eq_weight.equilibrium_weighted_transport_tracking_loss
equilibrium_weighted_transport_loss_gradient = _eq_weight.equilibrium_weighted_transport_loss_gradient
equilibrium_weighted_transport_rollout_tracking_loss = _eq_weight.equilibrium_weighted_transport_rollout_tracking_loss
equilibrium_weighted_transport_rollout_source_gradient = (
    _eq_weight.equilibrium_weighted_transport_rollout_source_gradient
)
_validate_equilibrium_psi = _eq_weight._validate_equilibrium_psi
_equilibrium_radial_weights_jax = _eq_weight._equilibrium_radial_weights_jax
_equilibrium_weighted_tracking_loss_jax = _eq_weight._equilibrium_weighted_tracking_loss_jax
_equilibrium_weighted_rollout_tracking_loss_jax = _eq_weight._equilibrium_weighted_rollout_tracking_loss_jax

# Stable public re-exports: core Crank-Nicolson step + multi-step rollout.
from scpn_control.core import differentiable_transport_core as _core

_transport_step_numpy = _core._transport_step_numpy
_transport_step_jax = _core._transport_step_jax
_transport_rollout_numpy = _core._transport_rollout_numpy
_transport_rollout_jax = _core._transport_rollout_jax
differentiable_transport_step = _core.differentiable_transport_step
differentiable_transport_rollout = _core.differentiable_transport_rollout


try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    _HAS_JAX = True
except Exception:
    jax = None
    jnp = cast(Any, None)  # optional-dep fallback (keeps jnp.* annotations typed)
    _HAS_JAX = False


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

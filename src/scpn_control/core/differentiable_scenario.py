# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — End-to-end differentiable control scenario facade

"""Gradient-through-equilibrium differentiable control scenario.

This facade couples a bounded analytic equilibrium parametrisation to the
differentiable transport rollout so a controller-tuning loss is differentiable
with respect to both the additive source schedule and the equilibrium shape
parameters:

    equilibrium parameters
        -> differentiable Solov'ev-form flux map
        -> Grad-Shafranov radial weighting
        -> differentiable multi-step transport rollout
        -> weighted tracking loss.

The flux parametrisation samples the analytic Solov'ev field
``psi = c1 R^4/8 + c2 Z^2`` (Solov'ev 1968) on the control grid; it is a
differentiable equilibrium *surface*, not a Grad-Shafranov PDE solve, and the
solver mathematics stay owned by the physics laboratory. The transport rollout
and equilibrium weighting are reused verbatim from
:mod:`scpn_control.core.differentiable_transport`.

The JAX path keeps the whole chain inside one traced graph. Gradient APIs fail
closed unless JAX is available, and every full-fidelity claim stays bounded
until the gradient audit, latency evidence, and traceability checks pass.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from scpn_control.core.differentiable_transport import (
    CHANNEL_COUNT,
    _equilibrium_weighted_rollout_tracking_loss_jax,
    _validate_transport_rollout_inputs,
    differentiable_transport_rollout,
    equilibrium_radial_weights,
    equilibrium_weighted_transport_rollout_tracking_loss,
)

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    _HAS_JAX = False

_EQUILIBRIUM_PARAM_COUNT = 2
_SCENARIO_METADATA_SCHEMA_VERSION = 1


def has_jax() -> bool:
    """Return whether the differentiable JAX scenario path is available."""
    return _HAS_JAX


@dataclass(frozen=True)
class DifferentiableScenarioGradient:
    """Gradients of the coupled scenario loss for controller tuning."""

    loss: float
    equilibrium_param_gradient: np.ndarray
    source_gradient: np.ndarray
    radial_weights: np.ndarray
    final_profiles: np.ndarray


@dataclass(frozen=True)
class DifferentiableScenarioGradientAudit:
    """Finite-difference audit of the coupled scenario gradients."""

    loss: float
    epsilon: float
    tolerance: float
    checked_param_indices: tuple[int, ...]
    checked_source_indices: tuple[tuple[int, int, int], ...]
    param_max_abs_error: float
    source_max_abs_error: float
    passed: bool


@dataclass(frozen=True)
class ScenarioCampaignMetadata:
    """Validated provenance for a differentiable scenario tuning campaign."""

    schema_version: int
    backend: str
    dtype: str
    n_rho: int
    n_steps: int
    equilibrium_param_count: int
    flux_grid_shape: tuple[int, int]
    dt: float
    gradient_tolerance: float | None
    jax_enable_x64: bool
    equilibrium_params: tuple[float, ...]
    inputs_sha256: str


@dataclass(frozen=True)
class ScenarioReadinessEvidence:
    """Fail-closed readiness evidence for a coupled-scenario claim."""

    schema_version: int
    backend: str
    campaign_sha256: str
    gradient_audit_sha256: str
    gradient_tolerance: float
    audit_passed: bool
    latency_p95_ms: float | None
    traceability_passed: bool
    claim_admissible: bool
    blocked_reasons: tuple[str, ...]
    claim_status: str


def _canonical_sha256(value: Any) -> str:
    payload = asdict(value) if hasattr(value, "__dataclass_fields__") else value
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _validate_equilibrium_params(equilibrium_params: Any) -> np.ndarray:
    params = np.asarray(equilibrium_params, dtype=float)
    if params.shape != (_EQUILIBRIUM_PARAM_COUNT,) or not np.all(np.isfinite(params)):
        raise ValueError(f"equilibrium_params must be a finite vector with shape ({_EQUILIBRIUM_PARAM_COUNT},)")
    return params


def _validate_flux_axis(name: str, values: Any) -> np.ndarray:
    axis = np.asarray(values, dtype=float)
    if axis.ndim != 1 or axis.size < 3 or not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must be a finite one-dimensional axis with at least three points")
    return axis


def scenario_equilibrium_flux(equilibrium_params: Any, r_grid: Any, z_grid: Any) -> np.ndarray:
    """Sample the analytic Solov'ev flux ``c1 R^4/8 + c2 Z^2`` on the control grid.

    Returns a ``(n_z, n_r)`` flux map; ``equilibrium_params`` is ``(c1, c2)``.
    """
    params = _validate_equilibrium_params(equilibrium_params)
    r_axis = _validate_flux_axis("r_grid", r_grid)
    z_axis = _validate_flux_axis("z_grid", z_grid)
    rr, zz = np.meshgrid(r_axis, z_axis)
    flux = params[0] * rr**4 / 8.0 + params[1] * zz**2
    return np.asarray(flux, dtype=float)


def _scenario_flux_jax(params: Any, rr: Any, zz: Any) -> Any:
    return params[0] * rr**4 / 8.0 + params[1] * zz**2


def differentiable_scenario_loss(
    equilibrium_params: Any,
    initial_profiles: Any,
    chi: Any,
    source_sequence: Any,
    target_history: Any,
    rho: Any,
    r_grid: Any,
    z_grid: Any,
    dt: float,
    edge_values: Any,
    *,
    weights: Any | None = None,
    use_jax: bool = True,
    allow_numpy_fallback: bool = False,
    allow_legacy_numpy_fallback: bool = False,
) -> Any:
    """Return the equilibrium-coupled multi-step transport tracking loss.

    The equilibrium parameters drive the analytic flux map that weights the
    differentiable transport rollout against ``target_history``.
    """
    flux = scenario_equilibrium_flux(equilibrium_params, r_grid, z_grid)
    return equilibrium_weighted_transport_rollout_tracking_loss(
        initial_profiles,
        chi,
        source_sequence,
        target_history,
        rho,
        dt,
        edge_values,
        flux,
        weights=weights,
        use_jax=use_jax,
        allow_numpy_fallback=allow_numpy_fallback,
        allow_legacy_numpy_fallback=allow_legacy_numpy_fallback,
    )


def differentiable_scenario_gradient(
    equilibrium_params: Any,
    initial_profiles: Any,
    chi: Any,
    source_sequence: Any,
    target_history: Any,
    rho: Any,
    r_grid: Any,
    z_grid: Any,
    dt: float,
    edge_values: Any,
    *,
    weights: Any | None = None,
) -> DifferentiableScenarioGradient:
    """Return JAX gradients of the coupled loss for equilibrium and sources.

    The returned gradients propagate through the transport rollout and the
    radial weighting back to both the source schedule and the analytic
    equilibrium parameters, so a controller can tune the equilibrium shape and
    the actuator schedule jointly. Fails closed unless JAX is available.
    """
    if not _HAS_JAX or jax is None or jnp is None:
        raise RuntimeError("differentiable_scenario_gradient requires JAX")

    params = _validate_equilibrium_params(equilibrium_params)
    r_axis = _validate_flux_axis("r_grid", r_grid)
    z_axis = _validate_flux_axis("z_grid", z_grid)
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

    rr, zz = np.meshgrid(r_axis, z_axis)
    rr_jax = jnp.asarray(rr, dtype=jnp.float64)
    zz_jax = jnp.asarray(zz, dtype=jnp.float64)

    def loss_for_params_and_sources(param_candidate: Any, source_candidate: Any) -> Any:
        flux = _scenario_flux_jax(param_candidate, rr_jax, zz_jax)
        return _equilibrium_weighted_rollout_tracking_loss_jax(
            profile_array,
            chi_array,
            source_candidate,
            target_array,
            rho_array,
            float(dt),
            edge_array,
            flux,
            weight_array,
        )

    loss, gradients = jax.value_and_grad(loss_for_params_and_sources, argnums=(0, 1))(
        jnp.asarray(params, dtype=jnp.float64),
        jnp.asarray(source_array, dtype=jnp.float64),
    )
    param_gradient, source_gradient = gradients

    flux_np = scenario_equilibrium_flux(params, r_axis, z_axis)
    history = np.asarray(
        differentiable_transport_rollout(
            profile_array, chi_array, source_array, rho_array, float(dt), edge_array, use_jax=False
        )
    )
    return DifferentiableScenarioGradient(
        loss=float(np.asarray(loss)),
        equilibrium_param_gradient=np.asarray(param_gradient),
        source_gradient=np.asarray(source_gradient),
        radial_weights=equilibrium_radial_weights(flux_np, profile_array.shape[1]),
        final_profiles=np.asarray(history[-1]),
    )


def _scenario_audit_indices(
    source_shape: tuple[int, ...],
    sample_indices: Any | None,
) -> tuple[tuple[int, int, int], ...]:
    n_steps, n_channels, n_rho = source_shape
    if sample_indices is None:
        return (
            (0, 0, 1),
            (n_steps - 1, 1, n_rho // 2),
            (n_steps // 2, 2, n_rho - 2),
        )
    parsed: list[tuple[int, int, int]] = []
    for raw in sample_indices:
        index = tuple(int(part) for part in raw)
        if len(index) != 3:
            raise ValueError("sample_indices must contain three-part rollout source indices")
        step, channel, radius = index
        if not (0 <= step < n_steps and 0 <= channel < n_channels and 0 <= radius < n_rho):
            raise ValueError("sample_indices contain an out-of-range rollout source index")
        parsed.append((step, channel, radius))
    if not parsed:
        raise ValueError("sample_indices must contain at least one rollout source index")
    return tuple(parsed)


def audit_differentiable_scenario_gradient(
    equilibrium_params: Any,
    initial_profiles: Any,
    chi: Any,
    source_sequence: Any,
    target_history: Any,
    rho: Any,
    r_grid: Any,
    z_grid: Any,
    dt: float,
    edge_values: Any,
    *,
    weights: Any | None = None,
    epsilon: float = 1.0e-5,
    tolerance: float = 5.0e-4,
    sample_indices: Any | None = None,
) -> DifferentiableScenarioGradientAudit:
    """Compare the coupled JAX gradients against sampled finite differences.

    The audit perturbs every equilibrium parameter and a deterministic subset of
    source entries, central-differencing the NumPy forward loss so the JAX
    chain-rule gradient through the equilibrium is verified independently.
    """
    eps = float(epsilon)
    tol = float(tolerance)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tolerance must be positive and finite")

    gradient = differentiable_scenario_gradient(
        equilibrium_params,
        initial_profiles,
        chi,
        source_sequence,
        target_history,
        rho,
        r_grid,
        z_grid,
        dt,
        edge_values,
        weights=weights,
    )
    params = _validate_equilibrium_params(equilibrium_params)
    source_array = np.asarray(source_sequence, dtype=float)
    indices = _scenario_audit_indices(source_array.shape, sample_indices)

    def forward(param_vec: np.ndarray, sources: np.ndarray) -> float:
        return float(
            differentiable_scenario_loss(
                param_vec,
                initial_profiles,
                chi,
                sources,
                target_history,
                rho,
                r_grid,
                z_grid,
                dt,
                edge_values,
                weights=weights,
                use_jax=False,
            )
        )

    param_max_abs_error = 0.0
    for i in range(params.size):
        plus = params.copy()
        minus = params.copy()
        plus[i] += eps
        minus[i] -= eps
        fd = (forward(plus, source_array) - forward(minus, source_array)) / (2.0 * eps)
        param_max_abs_error = max(param_max_abs_error, abs(float(gradient.equilibrium_param_gradient[i]) - fd))

    source_max_abs_error = 0.0
    for index in indices:
        plus = source_array.copy()
        minus = source_array.copy()
        plus[index] += eps
        minus[index] -= eps
        fd = (forward(params, plus) - forward(params, minus)) / (2.0 * eps)
        source_max_abs_error = max(source_max_abs_error, abs(float(gradient.source_gradient[index]) - fd))

    passed = bool(param_max_abs_error <= tol and source_max_abs_error <= tol)
    return DifferentiableScenarioGradientAudit(
        loss=float(gradient.loss),
        epsilon=eps,
        tolerance=tol,
        checked_param_indices=tuple(range(params.size)),
        checked_source_indices=indices,
        param_max_abs_error=float(param_max_abs_error),
        source_max_abs_error=float(source_max_abs_error),
        passed=passed,
    )


def assert_differentiable_scenario_gradient_consistent(
    equilibrium_params: Any,
    initial_profiles: Any,
    chi: Any,
    source_sequence: Any,
    target_history: Any,
    rho: Any,
    r_grid: Any,
    z_grid: Any,
    dt: float,
    edge_values: Any,
    *,
    weights: Any | None = None,
    epsilon: float = 1.0e-5,
    tolerance: float = 5.0e-4,
    sample_indices: Any | None = None,
) -> DifferentiableScenarioGradientAudit:
    """Return the scenario gradient audit or fail closed on a finite-difference gap."""
    audit = audit_differentiable_scenario_gradient(
        equilibrium_params,
        initial_profiles,
        chi,
        source_sequence,
        target_history,
        rho,
        r_grid,
        z_grid,
        dt,
        edge_values,
        weights=weights,
        epsilon=epsilon,
        tolerance=tolerance,
        sample_indices=sample_indices,
    )
    if not audit.passed:
        raise ValueError(
            "differentiable scenario gradient audit failed: "
            f"param_max_abs_error={audit.param_max_abs_error:.6g}, "
            f"source_max_abs_error={audit.source_max_abs_error:.6g}, tolerance={audit.tolerance:.6g}"
        )
    return audit


def scenario_campaign_metadata(
    equilibrium_params: Any,
    initial_profiles: Any,
    chi: Any,
    source_sequence: Any,
    rho: Any,
    r_grid: Any,
    z_grid: Any,
    dt: float,
    edge_values: Any,
    *,
    backend: str,
    gradient_tolerance: float | None = None,
) -> ScenarioCampaignMetadata:
    """Return serialisable provenance for a differentiable scenario campaign."""
    backend_value = str(backend).strip().lower()
    if backend_value not in {"numpy", "jax"}:
        raise ValueError("backend must be either 'numpy' or 'jax'")
    tolerance_value: float | None = None
    if gradient_tolerance is not None:
        tolerance_value = float(gradient_tolerance)
        if not np.isfinite(tolerance_value) or tolerance_value <= 0.0:
            raise ValueError("gradient_tolerance must be positive and finite")

    params = _validate_equilibrium_params(equilibrium_params)
    r_axis = _validate_flux_axis("r_grid", r_grid)
    z_axis = _validate_flux_axis("z_grid", z_grid)
    profile_array, chi_array, source_array, rho_array, edge_array, _, _ = _validate_transport_rollout_inputs(
        initial_profiles, chi, source_sequence, rho, dt, edge_values
    )
    dtype_name = np.result_type(profile_array.dtype, chi_array.dtype, source_array.dtype, rho_array.dtype).name
    inputs_digest = hashlib.sha256(
        b"|".join(
            np.ascontiguousarray(arr, dtype=np.float64).tobytes()
            for arr in (params, profile_array, chi_array, source_array, rho_array, edge_array, r_axis, z_axis)
        )
    ).hexdigest()
    return ScenarioCampaignMetadata(
        schema_version=_SCENARIO_METADATA_SCHEMA_VERSION,
        backend=backend_value,
        dtype=dtype_name,
        n_rho=int(rho_array.size),
        n_steps=int(source_array.shape[0]),
        equilibrium_param_count=int(params.size),
        flux_grid_shape=(int(z_axis.size), int(r_axis.size)),
        dt=float(dt),
        gradient_tolerance=tolerance_value,
        jax_enable_x64=bool(_HAS_JAX and jax is not None and jax.config.read("jax_enable_x64")),
        equilibrium_params=tuple(float(value) for value in params),
        inputs_sha256=inputs_digest,
    )


def differentiable_scenario_readiness_evidence(
    metadata: ScenarioCampaignMetadata,
    audit: DifferentiableScenarioGradientAudit,
    *,
    latency_p95_ms: float | None = None,
    traceability_passed: bool = False,
) -> ScenarioReadinessEvidence:
    """Build fail-closed readiness evidence for a coupled-scenario claim.

    A passing gradient audit on its own is bounded evidence. A full-fidelity
    scenario claim additionally needs a JAX campaign, measured latency, and an
    external traceability pass; missing pieces are reported as blocked reasons.
    """
    if not isinstance(metadata, ScenarioCampaignMetadata):
        raise ValueError("metadata must be ScenarioCampaignMetadata")
    if not isinstance(audit, DifferentiableScenarioGradientAudit):
        raise ValueError("audit must be DifferentiableScenarioGradientAudit")
    if metadata.gradient_tolerance is None:
        raise ValueError("metadata.gradient_tolerance is required for readiness evidence")
    if not np.isclose(audit.tolerance, metadata.gradient_tolerance, rtol=1.0e-12, atol=1.0e-15):
        raise ValueError("audit tolerance must match campaign metadata gradient_tolerance")

    blocked: list[str] = []
    if metadata.backend != "jax":
        blocked.append("jax_backend")
    if not audit.passed:
        blocked.append("gradient_audit")
    if latency_p95_ms is None:
        blocked.append("latency_evidence")
    elif not np.isfinite(latency_p95_ms) or latency_p95_ms < 0.0:
        raise ValueError("latency_p95_ms must be finite and non-negative")
    if not traceability_passed:
        blocked.append("physics_traceability")

    admissible = len(blocked) == 0
    return ScenarioReadinessEvidence(
        schema_version=_SCENARIO_METADATA_SCHEMA_VERSION,
        backend=metadata.backend,
        campaign_sha256=_canonical_sha256(metadata),
        gradient_audit_sha256=_canonical_sha256(audit),
        gradient_tolerance=float(metadata.gradient_tolerance),
        audit_passed=bool(audit.passed),
        latency_p95_ms=None if latency_p95_ms is None else float(latency_p95_ms),
        traceability_passed=bool(traceability_passed),
        claim_admissible=admissible,
        blocked_reasons=tuple(blocked),
        claim_status=(
            "bounded coupled differentiable scenario gradient evidence only"
            if not admissible
            else "coupled differentiable scenario claim admitted for declared bounds"
        ),
    )


def assert_scenario_claim_admissible(evidence: ScenarioReadinessEvidence) -> ScenarioReadinessEvidence:
    """Fail closed unless the coupled-scenario readiness evidence is admissible."""
    if not isinstance(evidence, ScenarioReadinessEvidence):
        raise ValueError("evidence must be ScenarioReadinessEvidence")
    if not evidence.claim_admissible:
        raise ValueError("coupled differentiable scenario claim is not ready: " + ", ".join(evidence.blocked_reasons))
    return evidence

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Nonlinear Model Predictive Controller
"""
Nonlinear Model Predictive Control for tokamak plasma.

NMPC formulation: minimize
    J = Σ_{k=0}^{N-1} ‖x_k − x_ref‖²_Q + ‖u_k‖²_R  + ‖x_N − x_ref‖²_P
    subject to  x_{k+1} = f(x_k, u_k),  u_min ≤ u_k ≤ u_max,  |Δu_k| ≤ Δu_max.

Rawlings, Mayne & Diehl 2017, "Model Predictive Control: Theory, Computation,
and Design", 2nd ed., Ch. 1. Terminal cost P chosen as discrete-ARE solution
to ensure recursive feasibility (Rawlings 2017, Ch. 2, Theorem 2.4).

Tokamak MPC application:
Felici et al. 2011, Nucl. Fusion 51, 083052 — real-time MPC for plasma current
profile and kinetic variable control on TCV.
"""

from __future__ import annotations

import dataclasses
import time
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.core.differentiable_transport import (
    TransportCampaignMetadata,
    TransportGradientAudit,
    TransportParameterGradients,
    TransportRolloutSourceGradients,
    assert_transport_parameter_gradients_consistent,
    transport_campaign_metadata,
    transport_coefficients_from_neural_closure,
    transport_loss_gradient,
    transport_parameter_gradients,
    transport_rollout_source_gradients,
    transport_rollout_tracking_loss,
)
from scpn_control.core.differentiable_transport import (
    has_jax as has_differentiable_transport_jax,
)

_NX = 6
_NU = 3


def _as_finite_vector(name: str, value: AnyFloatArray, size: int) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (size,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},).")
    return arr


def _percentile_ms(sorted_values: list[float], percentile: float) -> float:
    """Linear-interpolated percentile of a pre-sorted latency sample (ms)."""
    if not sorted_values:
        raise ValueError("latency sample must not be empty.")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * percentile
    lower = int(np.floor(rank))
    upper = int(np.ceil(rank))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = rank - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def _as_spd_matrix(name: str, value: AnyFloatArray, size: int) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (size, size) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite matrix with shape ({size}, {size}).")
    skew = arr - arr.T
    symmetry_scale = max(float(np.linalg.norm(arr, ord=np.inf)), 1.0)
    if float(np.max(np.abs(skew))) > 1.0e-14 * symmetry_scale:
        raise ValueError(f"{name} must be symmetric positive definite.")
    arr = 0.5 * (arr + arr.T)
    eig_min = float(np.min(np.linalg.eigvalsh(arr)))
    if eig_min <= 0.0:
        raise ValueError(f"{name} must be symmetric positive definite.")
    return arr


@dataclass
class NMPCConfig:
    """Configuration for NonlinearMPC.

    State vector: [I_p (MA), β_N, q_95, l_i, T_axis (keV), n̄ (10¹⁹ m⁻³)]
    Input vector: [P_aux (MW), I_p_ref (MA), Γ_gas (10²⁰ s⁻¹)]

    Bounds from ITER design basis (ITER Physics Basis 1999, Table 1).
    """

    horizon: int = 20
    Q: AnyFloatArray = dataclasses.field(default_factory=lambda: np.eye(6))
    R: AnyFloatArray = dataclasses.field(default_factory=lambda: np.eye(3))
    # Terminal cost P: solved from DARE; None triggers auto-solve.
    P: AnyFloatArray | None = None
    terminal_x_min: AnyFloatArray | None = None
    terminal_x_max: AnyFloatArray | None = None

    # State bounds: [I_p, β_N, q_95, l_i, T_axis, n̄]
    x_min: AnyFloatArray = dataclasses.field(default_factory=lambda: np.array([0.1, 0.0, 2.0, 0.5, 0.5, 0.1]))
    x_max: AnyFloatArray = dataclasses.field(default_factory=lambda: np.array([17.0, 3.5, 10.0, 1.5, 50.0, 12.0]))

    # Input bounds: [P_aux (MW), I_p_ref (MA), Γ_gas]
    # ITER heating: P_aux ≤ 73 MW (33 NBI + 20 ECRH + 20 ICRH)
    u_min: AnyFloatArray = dataclasses.field(default_factory=lambda: np.array([0.0, 0.1, 0.0]))
    u_max: AnyFloatArray = dataclasses.field(default_factory=lambda: np.array([73.0, 17.0, 10.0]))

    # Slew rate limits (per control step)
    du_max: AnyFloatArray = dataclasses.field(default_factory=lambda: np.array([5.0, 0.5, 2.0]))

    max_sqp_iter: int = 10
    qp_max_iter: int = 500
    qp_backend: str = "internal"  # "internal", "scipy", "osqp", "casadi", or "acados"
    linearization_backend: str = "finite_difference"  # "finite_difference" or "jax"; analytic provider still wins.
    tol: float = 1e-4
    acados_model_name: str = "scpn_control_nmpc"
    acados_qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    acados_nlp_solver_type: str = "SQP"
    acados_hessian_approximation: str = "EXACT"
    acados_integrator_type: str = "DISCRETE"
    acados_json_file: str | None = None
    acados_generate: bool = True
    acados_build: bool = True
    acados_dynamics_residual_tol: float = 1.0e-7
    # Real-Time Iteration: a single-SQP-iteration tick is admitted only when its
    # projected KKT stationarity residual stays at or below this bound.
    rti_residual_tol: float = 1.0e-3


AcadosSymbolicDynamics = Callable[[Any, Any, Any], Any]
AcadosOcpFactory = Callable[[NMPCConfig, AnyFloatArray], object]
AcadosSolverFactory = Callable[..., object]


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


@dataclass(frozen=True)
class RTIStepResult:
    """Diagnostics for a single Real-Time Iteration control tick.

    The Real-Time Iteration scheme performs exactly one SQP linearisation and one
    structured QP solve per tick, carrying the previous solution forward as the
    warm start (Diehl et al. 2005, J. Process Control 15, 593). ``admitted`` is
    a fail-closed flag: it is ``True`` only when the projected KKT stationarity
    residual is within ``rti_residual_tol`` and the rolled-out trajectory honours
    the state bounds.
    """

    u0: FloatArray
    solve_time_ms: float
    sqp_iterations: int
    stationarity_residual: float
    constraint_violation: bool
    warm_started: bool
    admitted: bool
    qp_backend: str


@dataclass(frozen=True)
class RTILatencyReport:
    """Wall-clock latency evidence for the audited Real-Time Iteration tick.

    Latency is local timing evidence on the recorded host, not a hard real-time
    guarantee; sub-millisecond claims require isolated-core measurement on the
    declared target hardware.
    """

    backend: str
    horizon: int
    nx: int
    nu: int
    warmup_ticks: int
    timed_ticks: int
    admitted_ticks: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    max_stationarity_residual: float


@dataclass(frozen=True)
class CostHessianAudit:
    """Finite-difference audit of the JAX NMPC cost Hessian."""

    epsilon: float
    tolerance: float
    max_abs_error: float
    symmetry_error: float
    min_eigenvalue: float
    is_positive_semidefinite: bool
    passed: bool


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
        if not isinstance(sample_indices, Iterable):
            raise ValueError("gradient_audit_sample_indices must be an iterable of three-part indices.")
        parsed_candidates: list[tuple[int, int, int]] = []
        for raw_index in sample_indices:
            if not isinstance(raw_index, Iterable):
                raise ValueError("gradient_audit_sample_indices must contain iterable three-part indices.")
            index_tuple = tuple(int(part) for part in raw_index)
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

    delta = -learning_rate_float * gradient_array
    if max_fractional_update_float is not None:
        cap = max_fractional_update_float * np.maximum(np.abs(chi_array), 1.0e-12)
        delta = np.clip(delta, -cap, cap)
    updated_chi = np.maximum(chi_min_float, chi_array + delta)
    step_norm = float(np.linalg.norm(updated_chi - chi_array))
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
        loss=float(loss),
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

    delta = -learning_rate_float * source_gradient
    if max_absolute_update_float is not None:
        delta = np.clip(delta, -max_absolute_update_float, max_absolute_update_float)
    updated_sources = source_array + delta
    if source_min_array is not None:
        updated_sources = np.maximum(source_min_array, updated_sources)
    if source_max_array is not None:
        updated_sources = np.minimum(source_max_array, updated_sources)
    step_norm = float(np.linalg.norm(updated_sources - source_array))
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
        loss=float(gradient_result.loss),
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

    delta = -learning_rate_float * source_gradient
    if max_absolute_update_float is not None:
        delta = np.clip(delta, -max_absolute_update_float, max_absolute_update_float)
    updated_sources = source_array + delta
    if source_min_array is not None:
        updated_sources = np.maximum(source_min_array, updated_sources)
    if source_max_array is not None:
        updated_sources = np.minimum(source_max_array, updated_sources)
    step_norm = float(np.linalg.norm(updated_sources - source_array))
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
        loss=float(gradient_result.loss),
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


class NonlinearMPC:
    """SQP-based NMPC with validated plant linearization contracts.

    Each SQP outer iteration linearizes f around the nominal trajectory with an
    optional analytic Jacobian provider. When no provider is configured, the
    controller falls back to bounded finite differences. The condensed QP is
    solved by either SciPy SLSQP or curvature-scaled projected gradient.
    """

    def __init__(
        self,
        plant_model: Callable[[AnyFloatArray, AnyFloatArray], FloatArray],
        config: NMPCConfig,
        linearization_model: Callable[[AnyFloatArray, AnyFloatArray], tuple[FloatArray, FloatArray]] | None = None,
        symbolic_dynamics_model: AcadosSymbolicDynamics | None = None,
        acados_ocp_factory: AcadosOcpFactory | None = None,
        acados_solver_factory: AcadosSolverFactory | None = None,
    ):
        self.plant_model = plant_model
        self.linearization_model = linearization_model
        self.symbolic_dynamics_model = symbolic_dynamics_model
        self.acados_ocp_factory = acados_ocp_factory
        self.acados_solver_factory = acados_solver_factory
        self.config = config

        self._validate_config(config)

        self.nx = _NX
        self.nu = _NU
        self.N = config.horizon

        self.u_traj = np.zeros((self.N, self.nu))
        self.x_traj = np.zeros((self.N + 1, self.nx))

        self.infeasibility_count = 0
        self.last_qp_iterations = 0
        self.last_qp_converged = False
        self.last_qp_step_size = 0.0
        self.last_qp_backend = "uninitialized"
        self.last_linearization_source = "uninitialized"
        self.last_acados_dynamics_residual = np.inf
        self._acados_ocp: object | None = None
        self._acados_solver: object | None = None
        self._rti_warm_started = False

    def _estimate_qp_step_size(
        self,
        A_k: list[FloatArray],
        B_k: list[FloatArray],
        P_term: AnyFloatArray,
    ) -> float:
        """Return a safe projected-gradient step from condensed QP curvature."""
        n_dec = self.N * self.nu
        state_sensitivity: AnyFloatArray = np.zeros((self.nx, n_dec))
        H: AnyFloatArray = np.zeros((n_dec, n_dec), dtype=np.float64)

        for k in range(self.N):
            H += 2.0 * state_sensitivity.T @ self.config.Q @ state_sensitivity
            block = slice(k * self.nu, (k + 1) * self.nu)
            H[block, block] += 2.0 * self.config.R

            next_sensitivity = A_k[k] @ state_sensitivity
            next_sensitivity[:, block] += B_k[k]
            state_sensitivity = next_sensitivity

        H += 2.0 * state_sensitivity.T @ P_term @ state_sensitivity
        H = 0.5 * (H + H.T)
        try:
            lipschitz = float(np.max(np.linalg.eigvalsh(H)))
        except np.linalg.LinAlgError:
            lipschitz = float(np.linalg.norm(H, ord=2))
        if not np.isfinite(lipschitz) or lipschitz <= 0.0:
            raise ValueError("condensed QP Hessian curvature must be positive finite.")
        return 1.0 / lipschitz

    @staticmethod
    def _validate_config(config: NMPCConfig) -> None:
        if isinstance(config.horizon, bool) or int(config.horizon) != config.horizon or config.horizon < 1:
            raise ValueError("horizon must be an integer >= 1.")
        if isinstance(config.max_sqp_iter, bool) or int(config.max_sqp_iter) != config.max_sqp_iter:
            raise ValueError("max_sqp_iter must be an integer >= 1.")
        if config.max_sqp_iter < 1:
            raise ValueError("max_sqp_iter must be an integer >= 1.")
        if isinstance(config.qp_max_iter, bool) or int(config.qp_max_iter) != config.qp_max_iter:
            raise ValueError("qp_max_iter must be an integer >= 1.")
        if config.qp_max_iter < 1:
            raise ValueError("qp_max_iter must be an integer >= 1.")
        if config.qp_backend not in {"internal", "scipy", "osqp", "casadi", "acados"}:
            raise ValueError("qp_backend must be 'internal', 'scipy', 'osqp', 'casadi', or 'acados'.")
        if config.linearization_backend not in {"finite_difference", "jax"}:
            raise ValueError("linearization_backend must be 'finite_difference' or 'jax'.")
        if not np.isfinite(float(config.tol)) or float(config.tol) <= 0.0:
            raise ValueError("tol must be positive finite.")
        for field in (
            "acados_model_name",
            "acados_qp_solver",
            "acados_nlp_solver_type",
            "acados_hessian_approximation",
            "acados_integrator_type",
        ):
            value = getattr(config, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string.")
            setattr(config, field, value.strip())
        if config.acados_json_file is not None and (
            not isinstance(config.acados_json_file, str) or not config.acados_json_file.strip()
        ):
            raise ValueError("acados_json_file must be None or a non-empty string.")
        if config.acados_json_file is not None:
            config.acados_json_file = config.acados_json_file.strip()
        if not isinstance(config.acados_generate, bool):
            raise ValueError("acados_generate must be boolean.")
        if not isinstance(config.acados_build, bool):
            raise ValueError("acados_build must be boolean.")
        if (
            not np.isfinite(float(config.acados_dynamics_residual_tol))
            or float(config.acados_dynamics_residual_tol) <= 0.0
        ):
            raise ValueError("acados_dynamics_residual_tol must be positive finite.")
        config.acados_dynamics_residual_tol = float(config.acados_dynamics_residual_tol)

        config.Q = _as_spd_matrix("Q", config.Q, _NX)
        config.R = _as_spd_matrix("R", config.R, _NU)
        if config.P is not None:
            config.P = _as_spd_matrix("P", config.P, _NX)

        config.x_min = _as_finite_vector("x_min", config.x_min, _NX)
        config.x_max = _as_finite_vector("x_max", config.x_max, _NX)
        config.u_min = _as_finite_vector("u_min", config.u_min, _NU)
        config.u_max = _as_finite_vector("u_max", config.u_max, _NU)
        config.du_max = _as_finite_vector("du_max", config.du_max, _NU)
        if (config.terminal_x_min is None) != (config.terminal_x_max is None):
            raise ValueError("terminal_x_min and terminal_x_max must be configured together.")
        if np.any(config.x_min >= config.x_max):
            raise ValueError("x_min entries must be strictly less than x_max entries.")
        if np.any(config.u_min >= config.u_max):
            raise ValueError("u_min entries must be strictly less than u_max entries.")
        if np.any(config.du_max <= 0.0):
            raise ValueError("du_max entries must be positive finite.")
        if config.terminal_x_min is not None and config.terminal_x_max is not None:
            if config.qp_backend not in {"scipy", "osqp", "casadi", "acados"}:
                raise ValueError("terminal_x constraints require qp_backend='scipy', 'osqp', 'casadi', or 'acados'.")
            terminal_x_min = _as_finite_vector("terminal_x_min", config.terminal_x_min, _NX)
            terminal_x_max = _as_finite_vector("terminal_x_max", config.terminal_x_max, _NX)
            config.terminal_x_min = terminal_x_min
            config.terminal_x_max = terminal_x_max
            if np.any(terminal_x_min >= terminal_x_max):
                raise ValueError("terminal_x_min entries must be strictly less than terminal_x_max entries.")
            if np.any(terminal_x_min < config.x_min) or np.any(terminal_x_max > config.x_max):
                raise ValueError("terminal_x bounds must lie inside configured state bounds.")

    def _plant_step(self, x: AnyFloatArray, u: AnyFloatArray) -> FloatArray:
        x_safe = _as_finite_vector("x", x, self.nx)
        u_safe = _as_finite_vector("u", u, self.nu)
        out = np.asarray(self.plant_model(x_safe, u_safe), dtype=np.float64)
        if out.shape != (self.nx,) or not np.all(np.isfinite(out)):
            raise ValueError(f"plant_model must return a finite vector with shape ({self.nx},).")
        return out

    @staticmethod
    def _finite_difference_column(
        f_plus: AnyFloatArray | None,
        f0: AnyFloatArray,
        f_minus: AnyFloatArray | None,
        step: float,
    ) -> FloatArray:
        if f_plus is not None and f_minus is not None:
            return np.asarray((f_plus - f_minus) / (2.0 * step), dtype=np.float64)
        if f_plus is not None:
            return np.asarray((f_plus - f0) / step, dtype=np.float64)
        if f_minus is not None:
            return np.asarray((f0 - f_minus) / step, dtype=np.float64)
        raise ValueError("finite-difference perturbation interval collapsed.")

    def _bounded_input_vector(self, name: str, value: AnyFloatArray) -> FloatArray:
        u = _as_finite_vector(name, value, self.nu)
        if np.any(u < self.config.u_min) or np.any(u > self.config.u_max):
            raise ValueError(f"{name} must satisfy configured input bounds.")
        return u

    def _linearize(self, x0: AnyFloatArray, u0: AnyFloatArray) -> tuple[FloatArray, FloatArray]:
        """Jacobians A = ∂f/∂x, B = ∂f/∂u for the local plant model."""
        x0_safe = _as_finite_vector("x0", x0, self.nx)
        u0_safe = _as_finite_vector("u0", u0, self.nu)
        if self.linearization_model is not None:
            A_raw, B_raw = self.linearization_model(x0_safe.copy(), u0_safe.copy())
            A = np.asarray(A_raw, dtype=np.float64)
            B = np.asarray(B_raw, dtype=np.float64)
            if A.shape != (self.nx, self.nx) or not np.all(np.isfinite(A)):
                raise ValueError(f"linearization_model must return finite A with shape ({self.nx}, {self.nx}).")
            if B.shape != (self.nx, self.nu) or not np.all(np.isfinite(B)):
                raise ValueError(f"linearization_model must return finite B with shape ({self.nx}, {self.nu}).")
            self.last_linearization_source = "analytic"
            return A, B

        if self.config.linearization_backend == "jax":
            return self._linearize_jax(x0_safe, u0_safe)

        A = np.zeros((self.nx, self.nx))
        B = np.zeros((self.nx, self.nu))
        eps_x = 1e-4
        eps_u = 1e-4
        f0 = self._plant_step(x0_safe, u0_safe)

        for i in range(self.nx):
            f_plus = None
            f_minus = None
            if x0_safe[i] + eps_x <= self.config.x_max[i]:
                x_plus = x0_safe.copy()
                x_plus[i] += eps_x
                f_plus = self._plant_step(x_plus, u0_safe)
            if x0_safe[i] - eps_x >= self.config.x_min[i]:
                x_minus = x0_safe.copy()
                x_minus[i] -= eps_x
                f_minus = self._plant_step(x_minus, u0_safe)
            A[:, i] = self._finite_difference_column(f_plus, f0, f_minus, eps_x)

        for i in range(self.nu):
            f_plus = None
            f_minus = None
            if u0_safe[i] + eps_u <= self.config.u_max[i]:
                u_plus = u0_safe.copy()
                u_plus[i] += eps_u
                f_plus = self._plant_step(x0_safe, u_plus)
            if u0_safe[i] - eps_u >= self.config.u_min[i]:
                u_minus = u0_safe.copy()
                u_minus[i] -= eps_u
                f_minus = self._plant_step(x0_safe, u_minus)
            B[:, i] = self._finite_difference_column(f_plus, f0, f_minus, eps_u)

        self.last_linearization_source = "finite_difference"
        return A, B

    def _linearize_jax(self, x0_safe: AnyFloatArray, u0_safe: AnyFloatArray) -> tuple[FloatArray, FloatArray]:
        """Return plant Jacobians through JAX autodiff for traceable plants."""
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as exc:
            raise RuntimeError("JAX linearization requires jax and jaxlib") from exc

        def traced_plant(x_arg: Any, u_arg: Any) -> Any:
            return jnp.asarray(self.plant_model(x_arg, u_arg), dtype=jnp.float64)

        try:
            A_raw, B_raw = jax.jacfwd(traced_plant, argnums=(0, 1))(
                jnp.asarray(x0_safe, dtype=jnp.float64),
                jnp.asarray(u0_safe, dtype=jnp.float64),
            )
        except Exception as exc:
            raise RuntimeError("plant_model must be JAX-traceable when linearization_backend='jax'") from exc

        A = np.asarray(A_raw, dtype=np.float64)
        B = np.asarray(B_raw, dtype=np.float64)
        if A.shape != (self.nx, self.nx) or not np.all(np.isfinite(A)):
            raise ValueError(f"JAX linearization must produce finite A with shape ({self.nx}, {self.nx}).")
        if B.shape != (self.nx, self.nu) or not np.all(np.isfinite(B)):
            raise ValueError(f"JAX linearization must produce finite B with shape ({self.nx}, {self.nu}).")
        self.last_linearization_source = "jax"
        return A, B

    def _compute_terminal_cost(self, A: AnyFloatArray, B: AnyFloatArray) -> FloatArray:
        """Discrete-ARE terminal cost for recursive feasibility.

        Rawlings, Mayne & Diehl 2017, Ch. 2, Theorem 2.4: choosing P as the
        LQR value function satisfies the terminal cost condition.
        """
        try:
            import scipy.linalg

            P = scipy.linalg.solve_discrete_are(A, B, self.config.Q, self.config.R)
            return _as_spd_matrix("terminal cost P", np.asarray(P), self.nx)
        except Exception:
            return np.asarray(self.config.Q * 10.0)

    def _qp_value_and_gradient(
        self,
        dU_flat: AnyFloatArray,
        A_k: list[FloatArray],
        B_k: list[FloatArray],
        P_term: AnyFloatArray,
        x_ref: AnyFloatArray,
    ) -> tuple[float, AnyFloatArray]:
        dU = np.asarray(dU_flat, dtype=np.float64).reshape(self.N, self.nu)
        dx: AnyFloatArray = np.zeros((self.N + 1, self.nx))
        for k in range(self.N):
            dx[k + 1] = A_k[k] @ dx[k] + B_k[k] @ dU[k]

        value = 0.0
        for k in range(self.N):
            x_err_k = (self.x_traj[k] + dx[k]) - x_ref
            u_k = self.u_traj[k] + dU[k]
            value += float(x_err_k @ self.config.Q @ x_err_k + u_k @ self.config.R @ u_k)
        x_err_N = (self.x_traj[self.N] + dx[self.N]) - x_ref
        value += float(x_err_N @ P_term @ x_err_N)

        adj: AnyFloatArray = np.zeros((self.N + 1, self.nx))
        adj[self.N] = 2.0 * P_term @ x_err_N
        grad_dU: AnyFloatArray = np.zeros((self.N, self.nu))
        for k in range(self.N - 1, -1, -1):
            x_err_k = (self.x_traj[k] + dx[k]) - x_ref
            adj[k] = A_k[k].T @ adj[k + 1] + 2.0 * self.config.Q @ x_err_k
            grad_dU[k] = B_k[k].T @ adj[k + 1] + 2.0 * self.config.R @ (self.u_traj[k] + dU[k])
        return value, grad_dU.reshape(-1)

    def _terminal_state_sensitivity(self, A_k: list[FloatArray], B_k: list[FloatArray]) -> AnyFloatArray:
        """Linear map from condensed control increments to terminal state."""
        sensitivity: AnyFloatArray = np.zeros((self.nx, self.N * self.nu), dtype=np.float64)
        for k in range(self.N):
            sensitivity = A_k[k] @ sensitivity
            block = slice(k * self.nu, (k + 1) * self.nu)
            sensitivity[:, block] += B_k[k]
        return sensitivity

    def _condensed_qp_terms(
        self,
        A_k: list[FloatArray],
        B_k: list[FloatArray],
        P_term: AnyFloatArray,
        x_ref: AnyFloatArray,
    ) -> tuple[FloatArray, AnyFloatArray]:
        """Return Hessian and linear term for the condensed QP objective."""
        n_dec = self.N * self.nu
        H: AnyFloatArray = np.zeros((n_dec, n_dec), dtype=np.float64)
        q: AnyFloatArray = np.zeros(n_dec, dtype=np.float64)
        sensitivity: AnyFloatArray = np.zeros((self.nx, n_dec), dtype=np.float64)

        for k in range(self.N):
            x_err = self.x_traj[k] - x_ref
            H += 2.0 * sensitivity.T @ self.config.Q @ sensitivity
            q += 2.0 * sensitivity.T @ self.config.Q @ x_err
            block = slice(k * self.nu, (k + 1) * self.nu)
            H[block, block] += 2.0 * self.config.R
            q[block] += 2.0 * self.config.R @ self.u_traj[k]

            next_sensitivity = A_k[k] @ sensitivity
            next_sensitivity[:, block] += B_k[k]
            sensitivity = next_sensitivity

        x_err_terminal = self.x_traj[self.N] - x_ref
        H += 2.0 * sensitivity.T @ P_term @ sensitivity
        q += 2.0 * sensitivity.T @ P_term @ x_err_terminal
        return 0.5 * (H + H.T), q

    def _solve_qp_scipy(
        self,
        A_k: list[FloatArray],
        B_k: list[FloatArray],
        P_term: AnyFloatArray,
        u_prev: AnyFloatArray,
        x_ref: AnyFloatArray,
    ) -> FloatArray:
        """Solve the condensed QP with SciPy SLSQP and explicit linear constraints."""
        import scipy.optimize

        n_dec = self.N * self.nu
        lower = np.zeros(n_dec)
        upper = np.zeros(n_dec)
        for k in range(self.N):
            block = slice(k * self.nu, (k + 1) * self.nu)
            lower[block] = self.config.u_min - self.u_traj[k]
            upper[block] = self.config.u_max - self.u_traj[k]

        rows = []
        lb = []
        ub = []
        for k in range(self.N):
            for j in range(self.nu):
                row = np.zeros(n_dec)
                row[k * self.nu + j] = 1.0
                if k == 0:
                    offset = self.u_traj[k, j] - u_prev[j]
                else:
                    row[(k - 1) * self.nu + j] = -1.0
                    offset = self.u_traj[k, j] - self.u_traj[k - 1, j]
                rows.append(row)
                lb.append(-self.config.du_max[j] - offset)
                ub.append(self.config.du_max[j] - offset)

        if self.config.terminal_x_min is not None and self.config.terminal_x_max is not None:
            terminal_sensitivity = self._terminal_state_sensitivity(A_k, B_k)
            terminal_offset = self.x_traj[self.N]
            for row, lower, upper, offset in zip(
                terminal_sensitivity,
                self.config.terminal_x_min,
                self.config.terminal_x_max,
                terminal_offset,
                strict=True,
            ):
                rows.append(row)
                lb.append(float(lower - offset))
                ub.append(float(upper - offset))

        bounds = scipy.optimize.Bounds(lower, upper)
        constraints = [scipy.optimize.LinearConstraint(np.vstack(rows), np.asarray(lb), np.asarray(ub))]

        def objective(z: AnyFloatArray) -> float:
            return self._qp_value_and_gradient(z, A_k, B_k, P_term, x_ref)[0]

        def gradient(z: AnyFloatArray) -> AnyFloatArray:
            return self._qp_value_and_gradient(z, A_k, B_k, P_term, x_ref)[1]

        result = scipy.optimize.minimize(
            objective,
            np.zeros(n_dec),
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": int(self.config.qp_max_iter), "ftol": float(self.config.tol), "disp": False},
        )
        self.last_qp_backend = "scipy"
        self.last_qp_iterations = int(getattr(result, "nit", 0))
        self.last_qp_converged = bool(result.success)
        if not result.success:
            raise RuntimeError(f"SciPy QP backend failed: {result.message}")
        return np.asarray(result.x, dtype=np.float64).reshape(self.N, self.nu)

    def _solve_qp_osqp(
        self,
        A_k: list[FloatArray],
        B_k: list[FloatArray],
        P_term: AnyFloatArray,
        u_prev: AnyFloatArray,
        x_ref: AnyFloatArray,
    ) -> FloatArray:
        """Solve the condensed sparse QP with OSQP and explicit constraints."""
        import warnings

        import osqp
        import scipy.sparse

        n_dec = self.N * self.nu
        H, q = self._condensed_qp_terms(A_k, B_k, P_term, x_ref)
        rows = []
        lb = []
        ub = []

        for idx in range(n_dec):
            row = np.zeros(n_dec)
            row[idx] = 1.0
            k = idx // self.nu
            j = idx % self.nu
            rows.append(row)
            lb.append(float(self.config.u_min[j] - self.u_traj[k, j]))
            ub.append(float(self.config.u_max[j] - self.u_traj[k, j]))

        for k in range(self.N):
            for j in range(self.nu):
                row = np.zeros(n_dec)
                row[k * self.nu + j] = 1.0
                if k == 0:
                    offset = self.u_traj[k, j] - u_prev[j]
                else:
                    row[(k - 1) * self.nu + j] = -1.0
                    offset = self.u_traj[k, j] - self.u_traj[k - 1, j]
                rows.append(row)
                lb.append(float(-self.config.du_max[j] - offset))
                ub.append(float(self.config.du_max[j] - offset))

        if self.config.terminal_x_min is not None and self.config.terminal_x_max is not None:
            terminal_sensitivity = self._terminal_state_sensitivity(A_k, B_k)
            terminal_offset = self.x_traj[self.N]
            for row, lower, upper, offset in zip(
                terminal_sensitivity,
                self.config.terminal_x_min,
                self.config.terminal_x_max,
                terminal_offset,
                strict=True,
            ):
                rows.append(row)
                lb.append(float(lower - offset))
                ub.append(float(upper - offset))

        solver = osqp.OSQP()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=PendingDeprecationWarning,
            )
            solver.setup(
                P=scipy.sparse.csc_matrix(H),
                q=q,
                A=scipy.sparse.csc_matrix(np.vstack(rows)),
                l=np.asarray(lb, dtype=np.float64),
                u=np.asarray(ub, dtype=np.float64),
                verbose=False,
                max_iter=int(self.config.qp_max_iter),
                eps_abs=float(self.config.tol),
                eps_rel=float(self.config.tol),
                polishing=True,
            )
            result = solver.solve()
        self.last_qp_backend = "osqp"
        self.last_qp_iterations = int(result.info.iter)
        self.last_qp_converged = int(result.info.status_val) in {1, 2}
        if not self.last_qp_converged:
            raise RuntimeError(f"OSQP backend failed: {result.info.status}")
        return np.asarray(result.x, dtype=np.float64).reshape(self.N, self.nu)

    def _solve_qp_casadi(
        self,
        A_k: list[FloatArray],
        B_k: list[FloatArray],
        P_term: AnyFloatArray,
        u_prev: AnyFloatArray,
        x_ref: AnyFloatArray,
    ) -> FloatArray:
        """Solve the condensed QP with CasADi Opti and explicit linear constraints."""
        try:
            import casadi as ca
        except ImportError as exc:
            raise ImportError("qp_backend='casadi' requires the optional casadi package.") from exc

        n_dec = self.N * self.nu
        H, q = self._condensed_qp_terms(A_k, B_k, P_term, x_ref)
        opti = ca.Opti()
        z = opti.variable(n_dec)
        opti.minimize(0.5 * ca.mtimes([z.T, ca.DM(H), z]) + ca.dot(ca.DM(q), z))

        for idx in range(n_dec):
            k = idx // self.nu
            j = idx % self.nu
            opti.subject_to(z[idx] >= float(self.config.u_min[j] - self.u_traj[k, j]))
            opti.subject_to(z[idx] <= float(self.config.u_max[j] - self.u_traj[k, j]))

        for k in range(self.N):
            for j in range(self.nu):
                if k == 0:
                    delta = z[k * self.nu + j] + float(self.u_traj[k, j] - u_prev[j])
                else:
                    delta = (
                        z[k * self.nu + j] - z[(k - 1) * self.nu + j] + float(self.u_traj[k, j] - self.u_traj[k - 1, j])
                    )
                opti.subject_to(delta >= float(-self.config.du_max[j]))
                opti.subject_to(delta <= float(self.config.du_max[j]))

        if self.config.terminal_x_min is not None and self.config.terminal_x_max is not None:
            terminal_sensitivity = self._terminal_state_sensitivity(A_k, B_k)
            terminal_state = ca.DM(terminal_sensitivity) @ z + ca.DM(self.x_traj[self.N])
            for idx in range(self.nx):
                opti.subject_to(terminal_state[idx] >= float(self.config.terminal_x_min[idx]))
                opti.subject_to(terminal_state[idx] <= float(self.config.terminal_x_max[idx]))

        opti.solver(
            "ipopt",
            {"print_time": False},
            {
                "max_iter": int(self.config.qp_max_iter),
                "tol": float(self.config.tol),
                "print_level": 0,
                "sb": "yes",
            },
        )
        solution = opti.solve()
        self.last_qp_backend = "casadi"
        self.last_qp_iterations = int(solution.stats().get("iter_count", 0))
        self.last_qp_converged = bool(solution.stats().get("success", True))
        return np.asarray(solution.value(z), dtype=np.float64).reshape(self.N, self.nu)

    def _build_acados_ocp(self, P_term: AnyFloatArray) -> object:
        """Build the acados augmented-state OCP from symbolic dynamics."""
        terminal_cost = _as_spd_matrix("terminal cost P", P_term, self.nx)
        if self.acados_ocp_factory is not None:
            return self.acados_ocp_factory(self.config, terminal_cost.copy())
        if self.symbolic_dynamics_model is None:
            raise RuntimeError(
                "qp_backend='acados' requires acados_ocp_factory or symbolic_dynamics_model for acados OCP generation."
            )
        try:
            import casadi as ca
            from acados_template import AcadosModel, AcadosOcp
        except ImportError as exc:
            raise ImportError("qp_backend='acados' requires optional casadi and acados_template packages.") from exc

        n_aug = self.nx + self.nu
        x_aug = ca.MX.sym("x", n_aug)
        u = ca.MX.sym("u", self.nu)
        x_phys = x_aug[: self.nx]
        u_last = x_aug[self.nx :]
        x_next = self.symbolic_dynamics_model(ca, x_phys, u)

        model = AcadosModel()
        model.name = self.config.acados_model_name
        model.x = x_aug
        model.u = u
        model.disc_dyn_expr = ca.vertcat(x_next, u)
        model.con_h_expr = u - u_last

        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = float(self.N)
        ocp.solver_options.integrator_type = self.config.acados_integrator_type
        ocp.solver_options.nlp_solver_type = self.config.acados_nlp_solver_type
        ocp.solver_options.qp_solver = self.config.acados_qp_solver
        ocp.solver_options.hessian_approx = self.config.acados_hessian_approximation
        ocp.solver_options.nlp_solver_max_iter = int(self.config.max_sqp_iter)
        ocp.solver_options.qp_solver_iter_max = int(self.config.qp_max_iter)
        ocp.solver_options.nlp_solver_tol_stat = float(self.config.tol)
        ocp.solver_options.nlp_solver_tol_eq = float(self.config.tol)
        ocp.solver_options.nlp_solver_tol_ineq = float(self.config.tol)
        ocp.solver_options.nlp_solver_tol_comp = float(self.config.tol)
        ocp.solver_options.print_level = 0

        ny = self.nx + self.nu
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.W = np.block(
            [
                [self.config.Q, np.zeros((self.nx, self.nu))],
                [np.zeros((self.nu, self.nx)), self.config.R],
            ]
        )
        ocp.cost.W_e = terminal_cost
        ocp.cost.Vx = np.zeros((ny, n_aug))
        ocp.cost.Vx[: self.nx, : self.nx] = np.eye(self.nx)
        ocp.cost.Vu = np.zeros((ny, self.nu))
        ocp.cost.Vu[self.nx :, :] = np.eye(self.nu)
        ocp.cost.Vx_e = np.zeros((self.nx, n_aug))
        ocp.cost.Vx_e[:, : self.nx] = np.eye(self.nx)
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(self.nx)

        ocp.constraints.idxbx = np.arange(n_aug, dtype=int)
        ocp.constraints.lbx = np.r_[self.config.x_min, self.config.u_min]
        ocp.constraints.ubx = np.r_[self.config.x_max, self.config.u_max]
        ocp.constraints.idxbu = np.arange(self.nu, dtype=int)
        ocp.constraints.lbu = self.config.u_min.copy()
        ocp.constraints.ubu = self.config.u_max.copy()
        ocp.constraints.lh = -self.config.du_max.copy()
        ocp.constraints.uh = self.config.du_max.copy()
        terminal_x_min = self.config.terminal_x_min if self.config.terminal_x_min is not None else self.config.x_min
        terminal_x_max = self.config.terminal_x_max if self.config.terminal_x_max is not None else self.config.x_max
        ocp.constraints.idxbx_e = np.arange(self.nx, dtype=int)
        ocp.constraints.lbx_e = terminal_x_min.copy()
        ocp.constraints.ubx_e = terminal_x_max.copy()
        return ocp

    def _make_acados_solver(self, ocp: object) -> object:
        kwargs = {
            "json_file": self.config.acados_json_file,
            "build": self.config.acados_build,
            "generate": self.config.acados_generate,
            "verbose": False,
        }
        if self.acados_solver_factory is not None:
            return self.acados_solver_factory(ocp, **kwargs)
        try:
            from acados_template import AcadosOcpSolver
        except ImportError as exc:
            raise ImportError("qp_backend='acados' requires the optional acados_template package.") from exc
        return AcadosOcpSolver(ocp, **kwargs)

    def close(self) -> None:
        """Release cached external solver resources held by this controller."""
        solver = self._acados_solver
        self._acados_solver = None
        self._acados_ocp = None
        if solver is None:
            return
        free_solver = getattr(solver, "free", None)
        if free_solver is None:
            return
        try:
            free_solver()
        except Exception as exc:
            raise RuntimeError("acados backend failed while releasing solver resources.") from exc

    def __enter__(self) -> NonlinearMPC:
        """Return this controller for deterministic external-solver lifetime scopes."""
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Release external solver resources without suppressing control-loop faults."""
        if exc_type is None:
            self.close()
            return
        try:
            self.close()
        except RuntimeError as cleanup_error:
            warnings.warn(
                f"acados backend cleanup failed during exception unwinding: {cleanup_error}",
                RuntimeWarning,
                stacklevel=2,
            )

    def _discard_acados_solver_after_failure(self) -> None:
        """Discard a failed acados native solver without replacing the root fault."""
        try:
            self.close()
        except RuntimeError as cleanup_error:
            warnings.warn(
                f"acados backend cleanup failed after solver fault: {cleanup_error}",
                RuntimeWarning,
                stacklevel=2,
            )

    @staticmethod
    def _acados_set(solver: object, stage: int, field: str, value: AnyFloatArray) -> None:
        solver_api: Any = solver
        try:
            solver_api.set(stage, field, np.asarray(value, dtype=np.float64))
        except Exception as exc:
            raise RuntimeError(f"acados backend failed while setting {field} at stage {stage}.") from exc

    @staticmethod
    def _acados_get(solver: object, stage: int, field: str) -> FloatArray:
        solver_api: Any = solver
        try:
            return np.asarray(solver_api.get(stage, field), dtype=np.float64)
        except Exception as exc:
            raise RuntimeError(f"acados backend failed while reading {field} at stage {stage}.") from exc

    @staticmethod
    def _acados_iterations(solver: object) -> int:
        get_stats = getattr(solver, "get_stats", None)
        if get_stats is None:
            return 0
        try:
            raw = get_stats("sqp_iter")
        except Exception:
            return 0
        arr = np.asarray(raw)
        if arr.size == 0:
            return 0
        return int(np.max(arr.astype(np.int64)))

    def _solve_qp_acados(
        self,
        P_term: AnyFloatArray,
        u_prev: AnyFloatArray,
        x_ref: AnyFloatArray,
    ) -> FloatArray:
        """Solve the full augmented-state OCP with acados."""
        try:
            if self._acados_ocp is None or self._acados_solver is None:
                self._acados_ocp = self._build_acados_ocp(P_term)
                self._acados_solver = self._make_acados_solver(self._acados_ocp)
            solver = self._acados_solver
            yref = np.r_[x_ref, np.zeros(self.nu)]
            x0_aug = np.r_[self.x_traj[0], u_prev]
            stage_lbx = np.r_[self.config.x_min, self.config.u_min]
            stage_ubx = np.r_[self.config.x_max, self.config.u_max]
            terminal_x_min = self.config.terminal_x_min if self.config.terminal_x_min is not None else self.config.x_min
            terminal_x_max = self.config.terminal_x_max if self.config.terminal_x_max is not None else self.config.x_max

            for k in range(self.N):
                u_last = u_prev if k == 0 else self.u_traj[k - 1]
                x_aug = np.r_[self.x_traj[k], u_last]
                self._acados_set(solver, k, "x", x_aug)
                self._acados_set(solver, k, "u", self.u_traj[k])
                self._acados_set(solver, k, "yref", yref)
                self._acados_set(solver, k, "lbu", self.config.u_min)
                self._acados_set(solver, k, "ubu", self.config.u_max)
                self._acados_set(solver, k, "lh", -self.config.du_max)
                self._acados_set(solver, k, "uh", self.config.du_max)
                if k == 0:
                    self._acados_set(solver, k, "lbx", x0_aug)
                    self._acados_set(solver, k, "ubx", x0_aug)
                else:
                    self._acados_set(solver, k, "lbx", stage_lbx)
                    self._acados_set(solver, k, "ubx", stage_ubx)

            terminal_aug = np.r_[self.x_traj[self.N], self.u_traj[self.N - 1]]
            self._acados_set(solver, self.N, "x", terminal_aug)
            self._acados_set(solver, self.N, "yref", x_ref)

            # Terminal set constraints are configured in OCP construction as an
            # explicit terminal-state block (idxbx_e). We intentionally avoid
            # constraining the augmented terminal control coordinates here so
            # the solver does not over-constrain the final control component.

            solver_api: Any = solver
            status = int(solver_api.solve())
            self.last_qp_backend = "acados"
            self.last_qp_iterations = self._acados_iterations(solver)
            self.last_qp_converged = status == 0
            if status != 0:
                raise RuntimeError(f"acados backend failed with status {status}.")

            u_solution = np.vstack([self._acados_get(solver, k, "u") for k in range(self.N)])
            if u_solution.shape != (self.N, self.nu) or not np.all(np.isfinite(u_solution)):
                raise RuntimeError("acados backend returned invalid control trajectory.")
            if np.any(u_solution < self.config.u_min - 1e-8) or np.any(u_solution > self.config.u_max + 1e-8):
                raise RuntimeError("acados backend returned control outside configured actuator bounds.")
            u_last = u_prev
            for u_stage in u_solution:
                if np.any(np.abs(u_stage - u_last) > self.config.du_max + 1e-8):
                    raise RuntimeError("acados backend returned control outside configured slew-rate bounds.")
                u_last = u_stage

            x_solution = np.vstack([self._acados_get(solver, k, "x")[: self.nx] for k in range(self.N + 1)])
            if x_solution.shape != (self.N + 1, self.nx) or not np.all(np.isfinite(x_solution)):
                raise RuntimeError("acados backend returned invalid state trajectory.")
            if not np.allclose(x_solution[0], self.x_traj[0], rtol=0.0, atol=self.config.acados_dynamics_residual_tol):
                raise RuntimeError("acados backend returned state trajectory with invalid initial state.")
            if np.any(x_solution < self.config.x_min - 1e-8) or np.any(x_solution > self.config.x_max + 1e-8):
                raise RuntimeError("acados backend returned state outside configured physics bounds.")
            terminal_state = x_solution[-1]
            if np.any(terminal_state < terminal_x_min - 1e-8) or np.any(terminal_state > terminal_x_max + 1e-8):
                raise RuntimeError("acados backend returned terminal state outside configured terminal state set.")
            max_residual = 0.0
            for k in range(self.N):
                plant_next = self._plant_step(x_solution[k], u_solution[k])
                residual = float(np.max(np.abs(plant_next - x_solution[k + 1])))
                max_residual = max(max_residual, residual)
            self.last_acados_dynamics_residual = max_residual
            if max_residual > self.config.acados_dynamics_residual_tol:
                raise RuntimeError(
                    "acados backend dynamics residual exceeds configured tolerance: "
                    f"{max_residual:.6e} > {self.config.acados_dynamics_residual_tol:.6e}"
                )
            return u_solution - self.u_traj
        except RuntimeError:
            self._discard_acados_solver_after_failure()
            raise

    def _solve_qp(self, x0: AnyFloatArray, u_prev: AnyFloatArray, x_ref: AnyFloatArray) -> FloatArray:
        """Projected gradient descent on condensed QP.

        Decision variables: ΔU = [δu_0, …, δu_{N−1}] where u_k = ū_k + δu_k.
        Gradient computed via backward adjoint pass; projected onto box constraints.
        """
        self.last_qp_iterations = 0
        self.last_qp_converged = False
        if self.config.qp_backend == "acados":
            self.last_qp_step_size = 0.0
            P_term_acados = self.config.P if self.config.P is not None else self.config.Q * 10.0
            return self._solve_qp_acados(P_term_acados, u_prev, x_ref)

        A_k = []
        B_k = []

        for k in range(self.N):
            Ak, Bk = self.linearize(self.x_traj[k], self.u_traj[k])
            A_k.append(Ak)
            B_k.append(Bk)

        max_iter = int(self.config.qp_max_iter)

        dU = np.zeros((self.N, self.nu))

        P_term = self.config.P if self.config.P is not None else self._compute_terminal_cost(A_k[-1], B_k[-1])
        alpha = self._estimate_qp_step_size(A_k, B_k, P_term)
        self.last_qp_step_size = alpha
        if self.config.qp_backend == "scipy":
            return self._solve_qp_scipy(A_k, B_k, P_term, u_prev, x_ref)
        if self.config.qp_backend == "osqp":
            return self._solve_qp_osqp(A_k, B_k, P_term, u_prev, x_ref)
        if self.config.qp_backend == "casadi":
            return self._solve_qp_casadi(A_k, B_k, P_term, u_prev, x_ref)
        self.last_qp_backend = "internal"

        for iter_idx in range(1, max_iter + 1):
            dx = np.zeros((self.N + 1, self.nx))
            for k in range(self.N):
                dx[k + 1] = A_k[k] @ dx[k] + B_k[k] @ dU[k]

            adj = np.zeros((self.N + 1, self.nx))

            x_err_N = (self.x_traj[self.N] + dx[self.N]) - x_ref
            adj[self.N] = 2.0 * P_term @ x_err_N

            grad_dU = np.zeros((self.N, self.nu))
            for k in range(self.N - 1, -1, -1):
                x_err_k = (self.x_traj[k] + dx[k]) - x_ref
                adj[k] = A_k[k].T @ adj[k + 1] + 2.0 * self.config.Q @ x_err_k
                grad_dU[k] = B_k[k].T @ adj[k + 1] + 2.0 * self.config.R @ (self.u_traj[k] + dU[k])

            dU_new = dU - alpha * grad_dU

            for k in range(self.N):
                u_full = self.u_traj[k] + dU_new[k]
                u_full = np.clip(u_full, self.config.u_min, self.config.u_max)

                u_last = u_prev if k == 0 else (self.u_traj[k - 1] + dU_new[k - 1])
                u_full = np.clip(u_full, u_last - self.config.du_max, u_last + self.config.du_max)
                dU_new[k] = u_full - self.u_traj[k]

            if np.max(np.abs(dU_new - dU)) < self.config.tol:
                dU[:] = dU_new
                self.last_qp_iterations = iter_idx
                self.last_qp_converged = True
                break

            dU[:] = dU_new
            self.last_qp_iterations = iter_idx

        return dU

    def linearize(self, x0: AnyFloatArray, u0: AnyFloatArray) -> tuple[FloatArray, FloatArray]:
        return self._linearize(x0, u0)

    def compute_cost(self, x_traj: AnyFloatArray, u_traj: AnyFloatArray, x_ref: AnyFloatArray) -> float:
        """Evaluate the NMPC cost J over a trajectory.

        J = Σ_{k=0}^{N-1} ‖x_k − x_ref‖²_Q + ‖u_k‖²_R
        Rawlings, Mayne & Diehl 2017, Ch. 1, Eq. (1.2).
        """
        x_arr = np.asarray(x_traj, dtype=np.float64)
        u_arr = np.asarray(u_traj, dtype=np.float64)
        x_ref_safe = _as_finite_vector("x_ref", x_ref, self.nx)
        if x_arr.ndim != 2 or x_arr.shape[1] != self.nx or not np.all(np.isfinite(x_arr)):
            raise ValueError(f"x_traj must be finite with shape (n, {self.nx}).")
        if u_arr.ndim != 2 or u_arr.shape[1] != self.nu or not np.all(np.isfinite(u_arr)):
            raise ValueError(f"u_traj must be finite with shape (n, {self.nu}).")
        if x_arr.shape[0] < u_arr.shape[0] + 1:
            raise ValueError("x_traj must contain at least one more row than u_traj.")

        J = 0.0
        for k in range(len(u_traj)):
            e = x_arr[k] - x_ref_safe
            J += float(e @ self.config.Q @ e + u_arr[k] @ self.config.R @ u_arr[k])
        e_terminal = x_arr[u_arr.shape[0]] - x_ref_safe
        P_term = self.config.P if self.config.P is not None else self.config.Q * 10.0
        J += float(e_terminal @ P_term @ e_terminal)
        return J

    def step(self, x: AnyFloatArray, x_ref: AnyFloatArray, u_prev: AnyFloatArray) -> FloatArray:
        """Compute optimal first control action via SQP.

        Warm-started from the previous solution shifted by one step.
        """
        x_safe = _as_finite_vector("x", x, self.nx)
        x_ref_safe = _as_finite_vector("x_ref", x_ref, self.nx)
        u_prev_safe = self._bounded_input_vector("u_prev", u_prev)

        if self.N > 1:
            self.u_traj[:-1] = self.u_traj[1:]
            self.u_traj[-1] = self.u_traj[-2]
        else:
            self.u_traj[0] = u_prev_safe

        for _sqp_iter in range(self.config.max_sqp_iter):
            self.x_traj[0] = x_safe
            for k in range(self.N):
                self.x_traj[k + 1] = self._plant_step(self.x_traj[k], self.u_traj[k])

            dU = self._solve_qp(x_safe, u_prev_safe, x_ref_safe)
            self.u_traj += dU

            if np.max(np.abs(dU)) < self.config.tol:
                break

        self.x_traj[0] = x_safe
        for k in range(self.N):
            self.x_traj[k + 1] = self._plant_step(self.x_traj[k], self.u_traj[k])

        viol = any(
            np.any(self.x_traj[k] < self.config.x_min - 1e-3) or np.any(self.x_traj[k] > self.config.x_max + 1e-3)
            for k in range(1, self.N + 1)
        )

        if viol:
            self.infeasibility_count += 1

        return np.asarray(self.u_traj[0])

    # ── Real-Time Iteration ───────────────────────────────────────────

    def _reduced_cost_gradient(self, x_ref_safe: AnyFloatArray) -> FloatArray:
        """Adjoint reduced gradient dJ/du over the current (x_traj, u_traj).

        Re-linearises the plant along the stored trajectory and runs one backward
        adjoint pass, giving the unconstrained cost gradient with respect to each
        control in the horizon.
        """
        A_k: list[FloatArray] = []
        B_k: list[FloatArray] = []
        for k in range(self.N):
            Ak, Bk = self.linearize(self.x_traj[k], self.u_traj[k])
            A_k.append(Ak)
            B_k.append(Bk)
        P_term = self.config.P if self.config.P is not None else self._compute_terminal_cost(A_k[-1], B_k[-1])

        adj = np.zeros((self.N + 1, self.nx))
        adj[self.N] = 2.0 * P_term @ (self.x_traj[self.N] - x_ref_safe)
        grad_u = np.zeros((self.N, self.nu))
        for k in range(self.N - 1, -1, -1):
            adj[k] = A_k[k].T @ adj[k + 1] + 2.0 * self.config.Q @ (self.x_traj[k] - x_ref_safe)
            grad_u[k] = B_k[k].T @ adj[k + 1] + 2.0 * self.config.R @ self.u_traj[k]
        return grad_u

    def _projected_stationarity_residual(self, grad_u: AnyFloatArray) -> float:
        """Infinity-norm of the box-projected KKT stationarity residual.

        Gradient components whose descent direction is blocked by an active input
        bound are projected out, so a small residual certifies that the iterate is
        first-order optimal for the active set.
        """
        proj = np.asarray(grad_u, dtype=np.float64).copy()
        at_upper = self.u_traj >= (self.config.u_max - 1.0e-9)
        at_lower = self.u_traj <= (self.config.u_min + 1.0e-9)
        proj[at_upper & (grad_u < 0.0)] = 0.0
        proj[at_lower & (grad_u > 0.0)] = 0.0
        return float(np.max(np.abs(proj))) if proj.size else 0.0

    def step_rti(self, x: AnyFloatArray, x_ref: AnyFloatArray, u_prev: AnyFloatArray) -> RTIStepResult:
        """Advance one Real-Time Iteration tick: one linearisation, one QP solve.

        The previous horizon solution is shifted forward as the warm start, then a
        single SQP iteration is taken — never an inner convergence loop. The tick
        is timed and its projected KKT stationarity residual is checked, so a
        controller can fail closed (``admitted is False``) when the linearised step
        drifts beyond ``rti_residual_tol`` or violates the state envelope.
        """
        x_safe = _as_finite_vector("x", x, self.nx)
        x_ref_safe = _as_finite_vector("x_ref", x_ref, self.nx)
        u_prev_safe = self._bounded_input_vector("u_prev", u_prev)
        warm_started = self._rti_warm_started

        if self.N > 1:
            self.u_traj[:-1] = self.u_traj[1:]
            self.u_traj[-1] = self.u_traj[-2]
        else:
            self.u_traj[0] = u_prev_safe

        start_ns = time.perf_counter_ns()
        self.x_traj[0] = x_safe
        for k in range(self.N):
            self.x_traj[k + 1] = self._plant_step(self.x_traj[k], self.u_traj[k])
        dU = self._solve_qp(x_safe, u_prev_safe, x_ref_safe)
        self.u_traj += dU
        self.x_traj[0] = x_safe
        for k in range(self.N):
            self.x_traj[k + 1] = self._plant_step(self.x_traj[k], self.u_traj[k])
        solve_time_ms = (time.perf_counter_ns() - start_ns) / 1.0e6

        grad_u = self._reduced_cost_gradient(x_ref_safe)
        stationarity = self._projected_stationarity_residual(grad_u)
        viol = any(
            np.any(self.x_traj[k] < self.config.x_min - 1e-3) or np.any(self.x_traj[k] > self.config.x_max + 1e-3)
            for k in range(1, self.N + 1)
        )
        if viol:
            self.infeasibility_count += 1
        self._rti_warm_started = True

        admitted = bool(stationarity <= self.config.rti_residual_tol and not viol)
        return RTIStepResult(
            u0=np.asarray(self.u_traj[0]),
            solve_time_ms=float(solve_time_ms),
            sqp_iterations=1,
            stationarity_residual=float(stationarity),
            constraint_violation=bool(viol),
            warm_started=bool(warm_started),
            admitted=admitted,
            qp_backend=self.last_qp_backend,
        )

    def reset_warm_start(self) -> None:
        """Clear the Real-Time Iteration warm-start memory and control horizon."""
        self.u_traj = np.zeros((self.N, self.nu))
        self.x_traj = np.zeros((self.N + 1, self.nx))
        self._rti_warm_started = False

    def benchmark_rti_latency(
        self,
        x: AnyFloatArray,
        x_ref: AnyFloatArray,
        u_prev: AnyFloatArray,
        *,
        warmup_ticks: int = 2,
        timed_ticks: int = 20,
    ) -> RTILatencyReport:
        """Measure Real-Time Iteration tick latency percentiles on this host.

        The report is local timing evidence on the recorded host, not a hard
        real-time guarantee; production sub-millisecond claims require
        isolated-core measurement on the declared target hardware.
        """
        if isinstance(warmup_ticks, bool) or not isinstance(warmup_ticks, int) or warmup_ticks < 0:
            raise ValueError("warmup_ticks must be a non-negative integer.")
        if isinstance(timed_ticks, bool) or not isinstance(timed_ticks, int) or timed_ticks < 1:
            raise ValueError("timed_ticks must be a positive integer.")

        for _ in range(warmup_ticks):
            self.step_rti(x, x_ref, u_prev)

        latencies: list[float] = []
        admitted = 0
        max_residual = 0.0
        for _ in range(timed_ticks):
            result = self.step_rti(x, x_ref, u_prev)
            latencies.append(result.solve_time_ms)
            admitted += int(result.admitted)
            max_residual = max(max_residual, result.stationarity_residual)

        ordered = sorted(latencies)
        return RTILatencyReport(
            backend=self.last_qp_backend,
            horizon=self.N,
            nx=self.nx,
            nu=self.nu,
            warmup_ticks=warmup_ticks,
            timed_ticks=timed_ticks,
            admitted_ticks=admitted,
            p50_ms=_percentile_ms(ordered, 0.50),
            p95_ms=_percentile_ms(ordered, 0.95),
            p99_ms=_percentile_ms(ordered, 0.99),
            max_ms=float(ordered[-1]),
            max_stationarity_residual=float(max_residual),
        )

    # ── JAX exact cost Hessian ────────────────────────────────────────

    def _cost_value_jax(self, x0: Any, u_flat: Any, x_ref: Any, jnp: Any) -> Any:
        """Traced rolled-out NMPC cost J(U) for JAX autodiff."""
        controls = u_flat.reshape(self.N, self.nu)
        Q = jnp.asarray(self.config.Q, dtype=jnp.float64)
        R = jnp.asarray(self.config.R, dtype=jnp.float64)
        P = jnp.asarray(self.config.P if self.config.P is not None else self.config.Q * 10.0, dtype=jnp.float64)
        x = jnp.asarray(x0, dtype=jnp.float64)
        x_ref_arr = jnp.asarray(x_ref, dtype=jnp.float64)
        cost = jnp.asarray(0.0, dtype=jnp.float64)
        for k in range(self.N):
            err = x - x_ref_arr
            cost = cost + err @ Q @ err + controls[k] @ R @ controls[k]
            x = jnp.asarray(self.plant_model(x, controls[k]), dtype=jnp.float64)
        err_terminal = x - x_ref_arr
        return cost + err_terminal @ P @ err_terminal

    def cost_hessian_jax(self, x0: AnyFloatArray, U: AnyFloatArray, x_ref: AnyFloatArray) -> FloatArray:
        """Exact Hessian d²J/dU² of the rolled-out NMPC cost through JAX autodiff.

        The Hessian is taken with respect to the flattened control sequence and
        includes the second-order dynamics curvature, unlike the Gauss-Newton
        curvature used inside the condensed QP. It fails closed when JAX is
        unavailable or the plant is not JAX-traceable; existing analytic terminal
        and linearisation providers remain authoritative for the control law.
        """
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as exc:
            raise RuntimeError("cost_hessian_jax requires jax and jaxlib") from exc

        x0_safe = _as_finite_vector("x0", x0, self.nx)
        x_ref_safe = _as_finite_vector("x_ref", x_ref, self.nx)
        u_arr = np.asarray(U, dtype=np.float64)
        if u_arr.shape != (self.N, self.nu) or not np.all(np.isfinite(u_arr)):
            raise ValueError(f"U must be finite with shape ({self.N}, {self.nu}).")

        u_flat = jnp.asarray(u_arr.reshape(-1), dtype=jnp.float64)

        def cost(candidate: Any) -> Any:
            return self._cost_value_jax(x0_safe, candidate, x_ref_safe, jnp)

        try:
            hessian_raw = jax.hessian(cost)(u_flat)
        except Exception as exc:
            raise RuntimeError("plant_model must be JAX-traceable for cost_hessian_jax") from exc

        hessian = np.asarray(hessian_raw, dtype=np.float64)
        dim = self.N * self.nu
        if hessian.shape != (dim, dim) or not np.all(np.isfinite(hessian)):
            raise ValueError(f"cost Hessian must be finite with shape ({dim}, {dim}).")
        return hessian

    def audit_cost_hessian_jax(
        self,
        x0: AnyFloatArray,
        U: AnyFloatArray,
        x_ref: AnyFloatArray,
        *,
        epsilon: float = 1.0e-4,
        tolerance: float = 1.0e-3,
        sample_indices: Iterable[tuple[int, int]] | None = None,
    ) -> CostHessianAudit:
        """Audit the JAX cost Hessian against sampled finite differences.

        A deterministic subset of Hessian entries is compared with independent
        central second differences of the NumPy cost rollout. The audit also
        reports the symmetry error and the smallest eigenvalue so callers can see
        whether the curvature is positive semidefinite at the evaluation point.
        """
        eps = float(epsilon)
        tol = float(tolerance)
        if not np.isfinite(eps) or eps <= 0.0:
            raise ValueError("epsilon must be positive and finite.")
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tolerance must be positive and finite.")

        hessian = self.cost_hessian_jax(x0, U, x_ref)
        dim = self.N * self.nu
        x0_safe = _as_finite_vector("x0", x0, self.nx)
        x_ref_safe = _as_finite_vector("x_ref", x_ref, self.nx)
        u_flat = np.asarray(U, dtype=np.float64).reshape(-1)

        def cost_np(candidate: AnyFloatArray) -> float:
            controls = candidate.reshape(self.N, self.nu)
            x = x0_safe.copy()
            terminal_p = self.config.P if self.config.P is not None else self.config.Q * 10.0
            total = 0.0
            for k in range(self.N):
                err = x - x_ref_safe
                total += float(err @ self.config.Q @ err + controls[k] @ self.config.R @ controls[k])
                x = self._plant_step(x, controls[k])
            err_terminal = x - x_ref_safe
            return total + float(err_terminal @ terminal_p @ err_terminal)

        if sample_indices is None:
            picks = sorted({0, dim // 2, dim - 1})
            indices: tuple[tuple[int, int], ...] = tuple((i, j) for i in picks for j in picks)
        else:
            indices = tuple((int(i), int(j)) for i, j in sample_indices)
            if not indices:
                raise ValueError("sample_indices must contain at least one (row, col) pair.")
            for i, j in indices:
                if not (0 <= i < dim and 0 <= j < dim):
                    raise ValueError("sample_indices contain an out-of-range Hessian entry.")

        max_abs_error = 0.0
        for i, j in indices:
            if i == j:
                plus = u_flat.copy()
                minus = u_flat.copy()
                plus[i] += eps
                minus[i] -= eps
                fd = (cost_np(plus) - 2.0 * cost_np(u_flat) + cost_np(minus)) / (eps * eps)
            else:
                pp = u_flat.copy()
                pm = u_flat.copy()
                mp = u_flat.copy()
                mm = u_flat.copy()
                pp[i] += eps
                pp[j] += eps
                pm[i] += eps
                pm[j] -= eps
                mp[i] -= eps
                mp[j] += eps
                mm[i] -= eps
                mm[j] -= eps
                fd = (cost_np(pp) - cost_np(pm) - cost_np(mp) + cost_np(mm)) / (4.0 * eps * eps)
            max_abs_error = max(max_abs_error, abs(float(hessian[i, j]) - fd))

        symmetry_error = float(np.max(np.abs(hessian - hessian.T))) if hessian.size else 0.0
        symmetric = 0.5 * (hessian + hessian.T)
        min_eigenvalue = float(np.min(np.linalg.eigvalsh(symmetric))) if hessian.size else 0.0
        return CostHessianAudit(
            epsilon=eps,
            tolerance=tol,
            max_abs_error=float(max_abs_error),
            symmetry_error=symmetry_error,
            min_eigenvalue=min_eigenvalue,
            is_positive_semidefinite=bool(min_eigenvalue >= -tol),
            passed=bool(max_abs_error <= tol),
        )

    def assert_cost_hessian_consistent(
        self,
        x0: AnyFloatArray,
        U: AnyFloatArray,
        x_ref: AnyFloatArray,
        *,
        epsilon: float = 1.0e-4,
        tolerance: float = 1.0e-3,
        sample_indices: Iterable[tuple[int, int]] | None = None,
    ) -> CostHessianAudit:
        """Return the cost-Hessian audit or fail closed on a finite-difference gap."""
        audit = self.audit_cost_hessian_jax(
            x0, U, x_ref, epsilon=epsilon, tolerance=tolerance, sample_indices=sample_indices
        )
        if not audit.passed:
            raise ValueError(
                f"NMPC cost Hessian audit failed: max_abs_error={audit.max_abs_error:.6g}, "
                f"tolerance={audit.tolerance:.6g}"
            )
        return audit

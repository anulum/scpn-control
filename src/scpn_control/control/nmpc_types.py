# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Nonlinear Model Predictive Controller
"""Configuration, solver callbacks, and result records for tokamak NMPC.

NMPC formulation: minimize
    J = Σ_{k=0}^{N-1} ‖x_k − x_ref‖²_Q + ‖u_k‖²_R  + ‖x_N − x_ref‖²_P
    subject to  x_{k+1} = f(x_k, u_k),  u_min ≤ u_k ≤ u_max,  |Δu_k| ≤ Δu_max.

The controller attempts a discrete-ARE terminal weight when ``P`` is absent.
That weight alone does not establish recursive feasibility for a constrained
nonlinear plant; the fallback ``10 Q`` is a cost heuristic.

Tokamak MPC application:
Felici et al. 2011, Nucl. Fusion 51, 083052 — real-time MPC for plasma current
profile and kinetic variable control on TCV.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

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

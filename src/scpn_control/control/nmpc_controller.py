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
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scpn_control.core.differentiable_transport import (
    has_jax as has_differentiable_transport_jax,
)
from scpn_control.core.differentiable_transport import (
    TransportCampaignMetadata,
    transport_loss_gradient,
    transport_coefficients_from_neural_closure,
    transport_campaign_metadata,
)

_NX = 6
_NU = 3


def _as_finite_vector(name: str, value: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (size,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},).")
    return arr


def _as_spd_matrix(name: str, value: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (size, size) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite matrix with shape ({size}, {size}).")
    if not np.allclose(arr, arr.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric positive definite.")
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
    Q: np.ndarray = dataclasses.field(default_factory=lambda: np.eye(6))
    R: np.ndarray = dataclasses.field(default_factory=lambda: np.eye(3))
    # Terminal cost P: solved from DARE; None triggers auto-solve.
    P: np.ndarray | None = None
    terminal_x_min: np.ndarray | None = None
    terminal_x_max: np.ndarray | None = None

    # State bounds: [I_p, β_N, q_95, l_i, T_axis, n̄]
    x_min: np.ndarray = dataclasses.field(default_factory=lambda: np.array([0.1, 0.0, 2.0, 0.5, 0.5, 0.1]))
    x_max: np.ndarray = dataclasses.field(default_factory=lambda: np.array([17.0, 3.5, 10.0, 1.5, 50.0, 12.0]))

    # Input bounds: [P_aux (MW), I_p_ref (MA), Γ_gas]
    # ITER heating: P_aux ≤ 73 MW (33 NBI + 20 ECRH + 20 ICRH)
    u_min: np.ndarray = dataclasses.field(default_factory=lambda: np.array([0.0, 0.1, 0.0]))
    u_max: np.ndarray = dataclasses.field(default_factory=lambda: np.array([73.0, 17.0, 10.0]))

    # Slew rate limits (per control step)
    du_max: np.ndarray = dataclasses.field(default_factory=lambda: np.array([5.0, 0.5, 2.0]))

    max_sqp_iter: int = 10
    qp_max_iter: int = 500
    qp_backend: str = "internal"  # "internal", "scipy", or "osqp"
    tol: float = 1e-4


@dataclass(frozen=True)
class TransportCoefficientTuningResult:
    """Result of a gradient-based transport-coefficient tuning step."""

    loss: float
    gradient: np.ndarray
    updated_chi: np.ndarray
    step_norm: float
    metadata: TransportCampaignMetadata


def tune_transport_coefficients_for_tracking(
    profiles: np.ndarray,
    chi: np.ndarray,
    sources: np.ndarray,
    target_profiles: np.ndarray,
    rho: np.ndarray,
    dt: float,
    edge_values: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    learning_rate: float,
    chi_min: float = 0.0,
    max_fractional_update: float | None = 0.1,
    gradient_tolerance: float | None = None,
    equilibrium_psi: np.ndarray | None = None,
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
    )


def tune_neural_transport_closure_for_tracking(
    profiles: np.ndarray,
    closure: object,
    sources: np.ndarray,
    target_profiles: np.ndarray,
    rho: np.ndarray,
    dt: float,
    edge_values: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    learning_rate: float,
    impurity_diffusivity_fraction: float = 1.0,
    chi_min: float = 0.0,
    max_fractional_update: float | None = 0.1,
    gradient_tolerance: float | None = None,
    equilibrium_psi: np.ndarray | None = None,
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
        plant_model: Callable[[np.ndarray, np.ndarray], np.ndarray],
        config: NMPCConfig,
        linearization_model: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
    ):
        self.plant_model = plant_model
        self.linearization_model = linearization_model
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

    def _estimate_qp_step_size(
        self,
        A_k: list[np.ndarray],
        B_k: list[np.ndarray],
        P_term: np.ndarray,
    ) -> float:
        """Return a safe projected-gradient step from condensed QP curvature."""
        n_dec = self.N * self.nu
        state_sensitivity: np.ndarray = np.zeros((self.nx, n_dec))
        H: np.ndarray = np.zeros((n_dec, n_dec), dtype=np.float64)

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
        if config.qp_backend not in {"internal", "scipy", "osqp"}:
            raise ValueError("qp_backend must be 'internal', 'scipy', or 'osqp'.")
        if not np.isfinite(float(config.tol)) or float(config.tol) <= 0.0:
            raise ValueError("tol must be positive finite.")

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
            if config.qp_backend not in {"scipy", "osqp"}:
                raise ValueError("terminal_x constraints require qp_backend='scipy' or 'osqp'.")
            terminal_x_min = _as_finite_vector("terminal_x_min", config.terminal_x_min, _NX)
            terminal_x_max = _as_finite_vector("terminal_x_max", config.terminal_x_max, _NX)
            config.terminal_x_min = terminal_x_min
            config.terminal_x_max = terminal_x_max
            if np.any(terminal_x_min >= terminal_x_max):
                raise ValueError("terminal_x_min entries must be strictly less than terminal_x_max entries.")
            if np.any(terminal_x_min < config.x_min) or np.any(terminal_x_max > config.x_max):
                raise ValueError("terminal_x bounds must lie inside configured state bounds.")

    def _plant_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x_safe = _as_finite_vector("x", x, self.nx)
        u_safe = _as_finite_vector("u", u, self.nu)
        out = np.asarray(self.plant_model(x_safe, u_safe), dtype=np.float64)
        if out.shape != (self.nx,) or not np.all(np.isfinite(out)):
            raise ValueError(f"plant_model must return a finite vector with shape ({self.nx},).")
        return out

    @staticmethod
    def _finite_difference_column(
        f_plus: np.ndarray | None,
        f0: np.ndarray,
        f_minus: np.ndarray | None,
        step: float,
    ) -> np.ndarray:
        if f_plus is not None and f_minus is not None:
            return np.asarray((f_plus - f_minus) / (2.0 * step), dtype=np.float64)
        if f_plus is not None:
            return np.asarray((f_plus - f0) / step, dtype=np.float64)
        if f_minus is not None:
            return np.asarray((f0 - f_minus) / step, dtype=np.float64)
        raise ValueError("finite-difference perturbation interval collapsed.")

    def _bounded_input_vector(self, name: str, value: np.ndarray) -> np.ndarray:
        u = _as_finite_vector(name, value, self.nu)
        if np.any(u < self.config.u_min) or np.any(u > self.config.u_max):
            raise ValueError(f"{name} must satisfy configured input bounds.")
        return u

    def _linearize(self, x0: np.ndarray, u0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    def _compute_terminal_cost(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
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
        dU_flat: np.ndarray,
        A_k: list[np.ndarray],
        B_k: list[np.ndarray],
        P_term: np.ndarray,
        x_ref: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        dU = np.asarray(dU_flat, dtype=np.float64).reshape(self.N, self.nu)
        dx: np.ndarray = np.zeros((self.N + 1, self.nx))
        for k in range(self.N):
            dx[k + 1] = A_k[k] @ dx[k] + B_k[k] @ dU[k]

        value = 0.0
        for k in range(self.N):
            x_err_k = (self.x_traj[k] + dx[k]) - x_ref
            u_k = self.u_traj[k] + dU[k]
            value += float(x_err_k @ self.config.Q @ x_err_k + u_k @ self.config.R @ u_k)
        x_err_N = (self.x_traj[self.N] + dx[self.N]) - x_ref
        value += float(x_err_N @ P_term @ x_err_N)

        adj: np.ndarray = np.zeros((self.N + 1, self.nx))
        adj[self.N] = 2.0 * P_term @ x_err_N
        grad_dU: np.ndarray = np.zeros((self.N, self.nu))
        for k in range(self.N - 1, -1, -1):
            x_err_k = (self.x_traj[k] + dx[k]) - x_ref
            adj[k] = A_k[k].T @ adj[k + 1] + 2.0 * self.config.Q @ x_err_k
            grad_dU[k] = B_k[k].T @ adj[k + 1] + 2.0 * self.config.R @ (self.u_traj[k] + dU[k])
        return value, grad_dU.reshape(-1)

    def _terminal_state_sensitivity(self, A_k: list[np.ndarray], B_k: list[np.ndarray]) -> np.ndarray:
        """Linear map from condensed control increments to terminal state."""
        sensitivity: np.ndarray = np.zeros((self.nx, self.N * self.nu), dtype=np.float64)
        for k in range(self.N):
            sensitivity = A_k[k] @ sensitivity
            block = slice(k * self.nu, (k + 1) * self.nu)
            sensitivity[:, block] += B_k[k]
        return sensitivity

    def _condensed_qp_terms(
        self,
        A_k: list[np.ndarray],
        B_k: list[np.ndarray],
        P_term: np.ndarray,
        x_ref: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return Hessian and linear term for the condensed QP objective."""
        n_dec = self.N * self.nu
        H: np.ndarray = np.zeros((n_dec, n_dec), dtype=np.float64)
        q: np.ndarray = np.zeros(n_dec, dtype=np.float64)
        sensitivity: np.ndarray = np.zeros((self.nx, n_dec), dtype=np.float64)

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
        A_k: list[np.ndarray],
        B_k: list[np.ndarray],
        P_term: np.ndarray,
        u_prev: np.ndarray,
        x_ref: np.ndarray,
    ) -> np.ndarray:
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

        def objective(z: np.ndarray) -> float:
            return self._qp_value_and_gradient(z, A_k, B_k, P_term, x_ref)[0]

        def gradient(z: np.ndarray) -> np.ndarray:
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
        A_k: list[np.ndarray],
        B_k: list[np.ndarray],
        P_term: np.ndarray,
        u_prev: np.ndarray,
        x_ref: np.ndarray,
    ) -> np.ndarray:
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

    def _solve_qp(self, x0: np.ndarray, u_prev: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """Projected gradient descent on condensed QP.

        Decision variables: ΔU = [δu_0, …, δu_{N−1}] where u_k = ū_k + δu_k.
        Gradient computed via backward adjoint pass; projected onto box constraints.
        """
        A_k = []
        B_k = []

        for k in range(self.N):
            Ak, Bk = self.linearize(self.x_traj[k], self.u_traj[k])
            A_k.append(Ak)
            B_k.append(Bk)

        max_iter = int(self.config.qp_max_iter)

        dU = np.zeros((self.N, self.nu))

        self.last_qp_iterations = 0
        self.last_qp_converged = False
        P_term = self.config.P if self.config.P is not None else self._compute_terminal_cost(A_k[-1], B_k[-1])
        alpha = self._estimate_qp_step_size(A_k, B_k, P_term)
        self.last_qp_step_size = alpha
        if self.config.qp_backend == "scipy":
            return self._solve_qp_scipy(A_k, B_k, P_term, u_prev, x_ref)
        if self.config.qp_backend == "osqp":
            return self._solve_qp_osqp(A_k, B_k, P_term, u_prev, x_ref)
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

    def linearize(self, x0: np.ndarray, u0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._linearize(x0, u0)

    def compute_cost(self, x_traj: np.ndarray, u_traj: np.ndarray, x_ref: np.ndarray) -> float:
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

    def step(self, x: np.ndarray, x_ref: np.ndarray, u_prev: np.ndarray) -> np.ndarray:
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

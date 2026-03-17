# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851  Contact: protoscience@anulum.li
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
    tol: float = 1e-4


class NonlinearMPC:
    """SQP-based NMPC with projected gradient descent inner solver.

    Each SQP outer iteration linearizes f around the nominal trajectory,
    then solves the resulting QP via projected gradient descent (PGD) on
    the condensed formulation (states eliminated).
    """

    def __init__(
        self,
        plant_model: Callable[[np.ndarray, np.ndarray], np.ndarray],
        config: NMPCConfig,
    ):
        self.plant_model = plant_model
        self.config = config

        self.nx = 6
        self.nu = 3
        self.N = config.horizon

        self.u_traj = np.zeros((self.N, self.nu))
        self.x_traj = np.zeros((self.N + 1, self.nx))

        self.infeasibility_count = 0

    def _linearize(self, x0: np.ndarray, u0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Jacobians A = ∂f/∂x, B = ∂f/∂u via forward finite differences."""
        A = np.zeros((self.nx, self.nx))
        B = np.zeros((self.nx, self.nu))
        eps_x = 1e-4
        eps_u = 1e-4

        f0 = self.plant_model(x0, u0)

        for i in range(self.nx):
            x_pert = x0.copy()
            x_pert[i] += eps_x
            A[:, i] = (self.plant_model(x_pert, u0) - f0) / eps_x

        for i in range(self.nu):
            u_pert = u0.copy()
            u_pert[i] += eps_u
            B[:, i] = (self.plant_model(x0, u_pert) - f0) / eps_u

        return A, B

    def _compute_terminal_cost(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Discrete-ARE terminal cost for recursive feasibility.

        Rawlings, Mayne & Diehl 2017, Ch. 2, Theorem 2.4: choosing P as the
        LQR value function satisfies the terminal cost condition.
        """
        try:
            import scipy.linalg

            P = scipy.linalg.solve_discrete_are(A, B, self.config.Q, self.config.R)
            return np.asarray(P)
        except Exception:
            return np.asarray(self.config.Q * 10.0)

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

        # PGD step size: α = 0.05 (empirically stable for ITER-scale dynamics)
        max_iter = 500
        alpha = 0.05

        dU = np.zeros((self.N, self.nu))

        for _iter in range(max_iter):
            dx = np.zeros((self.N + 1, self.nx))
            for k in range(self.N):
                dx[k + 1] = A_k[k] @ dx[k] + B_k[k] @ dU[k]

            adj = np.zeros((self.N + 1, self.nx))
            P_term = self.config.P if self.config.P is not None else self._compute_terminal_cost(A_k[-1], B_k[-1])

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
                break

            dU[:] = dU_new

        return dU

    def linearize(self, x0: np.ndarray, u0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._linearize(x0, u0)

    def compute_cost(self, x_traj: np.ndarray, u_traj: np.ndarray, x_ref: np.ndarray) -> float:
        """Evaluate the NMPC cost J over a trajectory.

        J = Σ_{k=0}^{N-1} ‖x_k − x_ref‖²_Q + ‖u_k‖²_R
        Rawlings, Mayne & Diehl 2017, Ch. 1, Eq. (1.2).
        """
        J = 0.0
        for k in range(len(u_traj)):
            e = x_traj[k] - x_ref
            J += float(e @ self.config.Q @ e + u_traj[k] @ self.config.R @ u_traj[k])
        return J

    def step(self, x: np.ndarray, x_ref: np.ndarray, u_prev: np.ndarray) -> np.ndarray:
        """Compute optimal first control action via SQP.

        Warm-started from the previous solution shifted by one step.
        """
        self.u_traj[:-1] = self.u_traj[1:]
        self.u_traj[-1] = self.u_traj[-2]

        for _sqp_iter in range(self.config.max_sqp_iter):
            self.x_traj[0] = x
            for k in range(self.N):
                self.x_traj[k + 1] = self.plant_model(self.x_traj[k], self.u_traj[k])

            dU = self._solve_qp(x, u_prev, x_ref)
            self.u_traj += dU

            if np.max(np.abs(dU)) < self.config.tol:
                break

        self.x_traj[0] = x
        for k in range(self.N):
            self.x_traj[k + 1] = self.plant_model(self.x_traj[k], self.u_traj[k])

        viol = any(
            np.any(self.x_traj[k] < self.config.x_min - 1e-3) or np.any(self.x_traj[k] > self.config.x_max + 1e-3)
            for k in range(1, self.N + 1)
        )

        if viol:
            self.infeasibility_count += 1

        return np.asarray(self.u_traj[0])

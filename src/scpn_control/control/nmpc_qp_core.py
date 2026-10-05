# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Nmpc Qp Core
"""QP dispatch and projected-gradient control increments."""

from __future__ import annotations

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

from .nmpc_acados import NMPCAcados


class NMPCQPCore(NMPCAcados):
    """Dispatch QP solves and perform the internal projected step."""

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

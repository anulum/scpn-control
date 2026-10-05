# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Nmpc Qp Solver
"""Condensed quadratic programs for SciPy, OSQP, and CasADi."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

from .nmpc_linearization import NMPCLinearization


class NMPCQPSolver(NMPCLinearization):
    """Constrained QP formulations and optional solver adapters."""

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

        return self._solve_qp_casadi_impl(
            ca, A_k, B_k, P_term, u_prev, x_ref
        )  # pragma: no cover - optional CasADi backend path

    def _solve_qp_casadi_impl(  # pragma: no cover - casadi optional dep, absent on CI
        self,
        ca: Any,
        A_k: list[FloatArray],
        B_k: list[FloatArray],
        P_term: AnyFloatArray,
        u_prev: AnyFloatArray,
        x_ref: AnyFloatArray,
    ) -> FloatArray:
        """Build and solve the CasADi Opti QP (requires the optional casadi package)."""
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

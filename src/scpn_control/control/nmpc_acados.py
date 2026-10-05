# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Nmpc Acados
"""Acados symbolic OCP construction, solver lifecycle, and runtime checks."""

from __future__ import annotations

import warnings
from typing import Any, Self

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

from .nmpc_qp_solver import NMPCQPSolver
from .nmpc_types import _as_spd_matrix


class NMPCAcados(NMPCQPSolver):
    """Acados OCP construction, lifecycle and solve contract."""

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
        # acados treats the initial stage separately: the slew-rate constraint at
        # stage 0 (u_0 - u_prev) must be declared via con_h_expr_0, otherwise the
        # solver has no h-constraint at stage 0 and setting lh/uh there fails.
        model.con_h_expr_0 = u - u_last

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
        # Declare the stage-0 initial-state equality so acados allocates idxbx_0
        # over all augmented states; the actual value is pinned at runtime via the
        # stage-0 lbx/ubx update. Without x0 the solver has nbx_0=0 and rejects it.
        ocp.constraints.x0 = np.r_[self.config.x_min, self.config.u_min].astype(float)
        ocp.constraints.idxbu = np.arange(self.nu, dtype=int)
        ocp.constraints.lbu = self.config.u_min.copy()
        ocp.constraints.ubu = self.config.u_max.copy()
        ocp.constraints.lh = -self.config.du_max.copy()
        ocp.constraints.uh = self.config.du_max.copy()
        ocp.constraints.lh_0 = -self.config.du_max.copy()
        ocp.constraints.uh_0 = self.config.du_max.copy()
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

    def __enter__(self) -> Self:
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
        array = np.asarray(value, dtype=np.float64)
        try:
            # Real acados routes nonlinear-constraint bounds (h) through
            # constraints_set; its set() rejects "lh"/"uh". Dispatch to
            # constraints_set when the solver exposes it, falling back to set()
            # for solvers/test doubles that accept bounds through set().
            if field in ("lh", "uh") and hasattr(solver_api, "constraints_set"):
                solver_api.constraints_set(stage, field, array)  # pragma: no cover - real acados only
            else:
                solver_api.set(stage, field, array)
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

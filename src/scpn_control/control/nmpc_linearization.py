# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Nmpc Linearization
"""Plant validation, configuration checks, and bounded Jacobian evaluation."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

from .nmpc_types import (
    _NU,
    _NX,
    AcadosOcpFactory,
    AcadosSolverFactory,
    AcadosSymbolicDynamics,
    NMPCConfig,
    _as_finite_vector,
    _as_spd_matrix,
)


class NMPCLinearization:
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
        if not np.isfinite(float(config.rti_residual_tol)) or float(config.rti_residual_tol) <= 0.0:
            raise ValueError("rti_residual_tol must be positive finite.")
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
        """Return a validated discrete-ARE weight or the ``10 Q`` fallback.

        Neither weight alone proves recursive feasibility for this nonlinear,
        constrained controller.
        """
        try:
            import scipy.linalg

            P = scipy.linalg.solve_discrete_are(A, B, self.config.Q, self.config.R)
            return _as_spd_matrix("terminal cost P", np.asarray(P), self.nx)
        except Exception:
            return np.asarray(self.config.Q * 10.0)

    def linearize(self, x0: AnyFloatArray, u0: AnyFloatArray) -> tuple[FloatArray, FloatArray]:
        """Linearise the plant about an operating point.

        Parameters
        ----------
        x0
            The state about which to linearise.
        u0
            The input about which to linearise.

        Returns
        -------
        tuple[FloatArray, FloatArray]
            The state and input Jacobians ``(A, B)``.
        """
        return self._linearize(x0, u0)

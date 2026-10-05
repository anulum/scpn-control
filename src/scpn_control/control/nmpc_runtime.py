# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Nmpc Runtime
"""NMPC control ticks, real-time iteration admission, and latency reports."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

from .nmpc_qp_core import NMPCQPCore
from .nmpc_types import RTILatencyReport, RTIStepResult, _as_finite_vector, _percentile_ms


class NMPCRuntime(NMPCQPCore):
    """Public NMPC and real-time iteration control operations."""

    @contextmanager
    def _restore_trajectory_on_failure(self) -> Iterator[None]:
        """Keep the last complete warm start if a plant or solver call fails."""
        controls = self.u_traj.copy()
        states = self.x_traj.copy()
        warm_started = self._rti_warm_started
        try:
            yield
        except Exception:
            self.u_traj[:] = controls
            self.x_traj[:] = states
            self._rti_warm_started = warm_started
            raise

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

        with self._restore_trajectory_on_failure():
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

            return np.asarray(self.u_traj[0], dtype=np.float64).copy()

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

        with self._restore_trajectory_on_failure():
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
                u0=self.u_traj[0].copy(),
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

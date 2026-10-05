# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Nmpc Hessian
"""JAX cost Hessian calculations and finite-difference audits."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

from .nmpc_runtime import NMPCRuntime
from .nmpc_types import CostHessianAudit, _as_finite_vector


class NMPCHessian(NMPCRuntime):
    """JAX cost-Hessian evaluation and independent finite-difference audit."""

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

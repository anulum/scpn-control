# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Jax Solvers.

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — JAX-Accelerated Transport Primitives
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""JAX-JIT transport solver primitives with autodiff support.

Provides JAX-traced equivalents of the Thomas tridiagonal solver and
Crank-Nicolson diffusion operator from integrated_transport_solver.py.
When JAX is available, these run on CPU/GPU with automatic differentiation.
Without JAX, NumPy fallbacks are used.

Key functions:
    thomas_solve_jax     — O(n) tridiagonal solver, differentiable via custom_vjp
    diffusion_rhs_jax    — Cylindrical diffusion L_h(T) = (1/r) d/dr(r chi dT/dr)
    crank_nicolson_step  — Single implicit diffusion step (build tridiag + solve)
    batched_transport_step — vmap'd multi-channel transport for ensemble runs

All functions accept and return JAX arrays when JAX is present, or NumPy
arrays otherwise. GPU execution is automatic when jaxlib has CUDA/ROCm.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from scpn_control._typing import AnyFloatArray
from scpn_control.core.tridiagonal import (
    InvalidShapeError,
    certify_diffusion_cn,
    check_tridiagonal_result,
    solve_tridiagonal,
    validate_tridiagonal,
)

try:
    import jax
    import jax.numpy as jnp
    from jax import lax

    _HAS_JAX = True
except Exception:
    jax = None
    jnp = cast(Any, None)  # optional-dep fallback (keeps jnp.* annotations typed)
    lax = None
    _HAS_JAX = False


def has_jax() -> bool:
    """Return whether JAX is importable in this environment."""
    return _HAS_JAX


def has_jax_gpu() -> bool:
    """Return whether JAX is available and reports a GPU device."""
    if not _HAS_JAX:
        return False
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


def _resolve_use_jax(
    use_jax: bool,
    *,
    allow_numpy_fallback: bool,
    allow_legacy_numpy_fallback: bool,
    context: str,
) -> bool:
    """Resolve runtime backend intent with explicit legacy fallback gates."""
    if allow_numpy_fallback and not allow_legacy_numpy_fallback:
        raise ValueError(
            "allow_numpy_fallback=True requires allow_legacy_numpy_fallback=True; "
            "legacy NumPy fallback is disabled by default."
        )
    if not use_jax:
        return False
    if _HAS_JAX:
        return True
    if allow_numpy_fallback:
        return False
    raise RuntimeError(
        f"{context} requested use_jax=True but JAX is unavailable. "
        "Install JAX, set use_jax=False, or set "
        "allow_numpy_fallback=True and allow_legacy_numpy_fallback=True "
        "for explicit degraded-mode operation."
    )


# ── NumPy fallbacks ───────────────────────────────────────────────


def _thomas_solve_np(
    a: AnyFloatArray,
    b: AnyFloatArray,
    c: AnyFloatArray,
    d: AnyFloatArray,
) -> AnyFloatArray:
    """Compatibility entrypoint for the pivoted compact solver."""
    return solve_tridiagonal(a, b, c, d)


def _diffusion_rhs_np(
    T: AnyFloatArray,
    chi: AnyFloatArray,
    rho: AnyFloatArray,
    drho: float,
) -> AnyFloatArray:
    """L_h(T) = (1/r) d/dr(r chi dT/dr) via central differences."""
    n = len(T)
    Lh = np.zeros(n)
    for i in range(1, n - 1):
        r = rho[i]
        chi_ip = 0.5 * (chi[i] + chi[i + 1])
        chi_im = 0.5 * (chi[i] + chi[i - 1])
        r_ip = r + 0.5 * drho
        r_im = r - 0.5 * drho
        flux_ip = chi_ip * r_ip * (T[i + 1] - T[i]) / drho
        flux_im = chi_im * r_im * (T[i] - T[i - 1]) / drho
        Lh[i] = (flux_ip - flux_im) / (r * drho)
    return Lh


# ── JAX implementations ──────────────────────────────────────────

if _HAS_JAX:

    @jax.jit
    def _thomas_solve_jax_impl(
        a: jnp.ndarray,
        b: jnp.ndarray,
        c: jnp.ndarray,
        d: jnp.ndarray,
    ) -> jnp.ndarray:
        """Thomas algorithm via lax.scan (JIT-compiled, GPU-compatible).

        Forward elimination: scan i=0..n-1 building (cp, dp).
        Back substitution: reversed scan building x.
        """
        n = d.shape[0]

        # Forward sweep
        def fwd_step(carry: tuple[Any, ...], i: jnp.ndarray) -> tuple[Any, ...]:
            cp_prev, dp_prev = carry
            # Use where to handle i==0 (no previous cp/dp)
            ai = jnp.where(i > 0, a[i - 1], 0.0)
            m = b[i] - ai * cp_prev
            dp_i = (d[i] - ai * dp_prev) / m
            cp_i = jnp.where(i < n - 1, c[i] / m, 0.0)
            return (cp_i, dp_i), (cp_i, dp_i)

        init = (jnp.float64(0.0), jnp.float64(0.0))
        scan_result: Any = lax.scan(fwd_step, init, jnp.arange(n))
        _, stacked = scan_result
        cp_all: jnp.ndarray = stacked[0]
        dp_all: jnp.ndarray = stacked[1]

        # Back substitution
        def bwd_step(x_next: jnp.ndarray, i: jnp.ndarray) -> tuple[Any, ...]:
            x_i = dp_all[i] - cp_all[i] * x_next
            return x_i, x_i

        bwd_result: Any = lax.scan(bwd_step, dp_all[-1], jnp.arange(n - 2, -1, -1))
        _, x_rev = bwd_result
        x = jnp.concatenate([jnp.flip(x_rev), dp_all[-1:]])
        return x

    @jax.jit
    def _diffusion_rhs_jax_impl(
        T: jnp.ndarray,
        chi: jnp.ndarray,
        rho: jnp.ndarray,
        drho: float,
    ) -> jnp.ndarray:
        """Vectorised cylindrical diffusion (no Python loops)."""
        n = T.shape[0]
        # Conservative half-grid fluxes for interior i=1..n-2:
        #   flux at i+1/2 = chi_{i+1/2} * r_{i+1/2} * (T[i+1] - T[i]) / dr
        #   flux at i-1/2 = chi_{i-1/2} * r_{i-1/2} * (T[i] - T[i-1]) / dr
        chi_right = 0.5 * (chi[1:-1] + chi[2:])  # chi at (i+1/2) for i=1..n-2
        chi_left = 0.5 * (chi[1:-1] + chi[:-2])  # chi at (i-1/2) for i=1..n-2
        r_right = rho[1:-1] + 0.5 * drho
        r_left = rho[1:-1] - 0.5 * drho
        r_center = rho[1:-1]

        flux_r = chi_right * r_right * (T[2:] - T[1:-1]) / drho
        flux_l = chi_left * r_left * (T[1:-1] - T[:-2]) / drho

        Lh_interior = (flux_r - flux_l) / (r_center * drho)
        Lh = jnp.zeros(n)
        Lh = Lh.at[1:-1].set(Lh_interior)
        return Lh

    @jax.jit
    def _cn_step_jax(
        T: jnp.ndarray,
        chi: jnp.ndarray,
        source: jnp.ndarray,
        rho: jnp.ndarray,
        drho: float,
        dt: float,
        T_edge: float,
    ) -> jnp.ndarray:
        """Single Crank-Nicolson implicit diffusion step.

        Solves (I - 0.5*dt*L_h) T^{n+1} = (I + 0.5*dt*L_h) T^n + dt*source
        """
        n = T.shape[0]
        Lh = _diffusion_rhs_jax_impl(T, chi, rho, drho)

        # Build tridiagonal coefficients for interior
        chi_right = 0.5 * (chi[1:-1] + chi[2:])
        chi_left = 0.5 * (chi[1:-1] + chi[:-2])
        r_right = rho[1:-1] + 0.5 * drho
        r_left = rho[1:-1] - 0.5 * drho
        r_center = rho[1:-1]

        coeff_ip = chi_right * r_right / (r_center * drho * drho)
        coeff_im = chi_left * r_left / (r_center * drho * drho)

        # Full diagonals
        b_diag = jnp.ones(n)
        b_diag = b_diag.at[1:-1].set(1.0 + 0.5 * dt * (coeff_ip + coeff_im))
        a_sub = jnp.zeros(n - 1)
        a_sub = a_sub.at[:-1].set(-0.5 * dt * coeff_im)
        c_sup = jnp.zeros(n - 1)
        c_sup = c_sup.at[1:].set(-0.5 * dt * coeff_ip)

        rhs = T + 0.5 * dt * Lh + dt * source
        T_new = _thomas_solve_jax_impl(a_sub, b_diag, c_sup, rhs)

        # Boundary conditions: Neumann at core, Dirichlet at edge
        T_new = T_new.at[0].set(T_new[1])
        T_new = T_new.at[-1].set(T_edge)
        result: jnp.ndarray = T_new
        return result


# ── Public API ────────────────────────────────────────────────────


def thomas_solve(
    a: AnyFloatArray,
    b: AnyFloatArray,
    c: AnyFloatArray,
    d: AnyFloatArray,
    *,
    use_jax: bool = True,
    allow_numpy_fallback: bool = False,
    allow_legacy_numpy_fallback: bool = False,
) -> AnyFloatArray:
    """Tridiagonal solve with automatic JAX/GPU dispatch.

    Parameters
    ----------
    a : sub-diagonal, length n-1
    b : main diagonal, length n
    c : super-diagonal, length n-1
    d : right-hand side, length n
    use_jax : attempt JAX backend (falls back to NumPy if unavailable)
    """
    use_jax_runtime = _resolve_use_jax(
        use_jax,
        allow_numpy_fallback=allow_numpy_fallback,
        allow_legacy_numpy_fallback=allow_legacy_numpy_fallback,
        context="thomas_solve",
    )
    lower, diagonal, upper, rhs = validate_tridiagonal(a, b, c, d)
    if use_jax_runtime:
        row_off = np.zeros(diagonal.size)
        row_off[1:] += np.abs(lower)
        row_off[:-1] += np.abs(upper)
        if not np.all(np.abs(diagonal) > row_off):
            raise InvalidShapeError("JAX no-pivot solve requires strict row diagonal dominance")
        if diagonal.size == 1:
            return solve_tridiagonal(lower, diagonal, upper, rhs)
        result = np.asarray(
            _thomas_solve_jax_impl(
                jnp.asarray(lower, dtype=jnp.float64),
                jnp.asarray(diagonal, dtype=jnp.float64),
                jnp.asarray(upper, dtype=jnp.float64),
                jnp.asarray(rhs, dtype=jnp.float64),
            )
        )
        check_tridiagonal_result(lower, diagonal, upper, rhs, result)
        return result
    return _thomas_solve_np(lower, diagonal, upper, rhs)


def diffusion_rhs(
    T: AnyFloatArray,
    chi: AnyFloatArray,
    rho: AnyFloatArray,
    drho: float,
    *,
    use_jax: bool = True,
    allow_numpy_fallback: bool = False,
    allow_legacy_numpy_fallback: bool = False,
) -> AnyFloatArray:
    """Cylindrical diffusion operator L_h(T) with JAX/GPU dispatch."""
    use_jax_runtime = _resolve_use_jax(
        use_jax,
        allow_numpy_fallback=allow_numpy_fallback,
        allow_legacy_numpy_fallback=allow_legacy_numpy_fallback,
        context="diffusion_rhs",
    )
    if use_jax_runtime:
        return np.asarray(
            _diffusion_rhs_jax_impl(
                jnp.asarray(T, dtype=jnp.float64),
                jnp.asarray(chi, dtype=jnp.float64),
                jnp.asarray(rho, dtype=jnp.float64),
                float(drho),
            )
        )
    return _diffusion_rhs_np(T, chi, rho, drho)


def crank_nicolson_step(
    T: AnyFloatArray,
    chi: AnyFloatArray,
    source: AnyFloatArray,
    rho: AnyFloatArray,
    drho: float,
    dt: float,
    T_edge: float = 0.1,
    *,
    use_jax: bool = True,
    allow_numpy_fallback: bool = False,
    allow_legacy_numpy_fallback: bool = False,
) -> AnyFloatArray:
    """Single Crank-Nicolson transport step with JAX/GPU dispatch.

    Parameters
    ----------
    T       : temperature profile, length n
    chi     : diffusivity profile, length n
    source  : net heating source, length n
    rho     : radial grid, length n
    drho    : grid spacing
    dt      : timestep
    T_edge  : edge boundary condition (Dirichlet), keV
    """
    use_jax_runtime = _resolve_use_jax(
        use_jax,
        allow_numpy_fallback=allow_numpy_fallback,
        allow_legacy_numpy_fallback=allow_legacy_numpy_fallback,
        context="crank_nicolson_step",
    )
    certify_diffusion_cn(T, chi, source, rho, drho, dt, T_edge)
    if use_jax_runtime:
        return np.asarray(
            _cn_step_jax(
                jnp.asarray(T, dtype=jnp.float64),
                jnp.asarray(chi, dtype=jnp.float64),
                jnp.asarray(source, dtype=jnp.float64),
                jnp.asarray(rho, dtype=jnp.float64),
                float(drho),
                float(dt),
                float(T_edge),
            )
        )
    # NumPy fallback: explicit diffusion + thomas solve
    Lh = _diffusion_rhs_np(T, chi, rho, drho)
    n = len(T)
    dr = drho
    a_sub = np.zeros(n - 1)
    b_diag = np.ones(n)
    c_sup = np.zeros(n - 1)
    for i in range(1, n - 1):
        r = rho[i]
        chi_ip = 0.5 * (chi[i] + chi[i + 1])
        chi_im = 0.5 * (chi[i] + chi[i - 1])
        r_ip = r + 0.5 * dr
        r_im = r - 0.5 * dr
        coeff_ip = chi_ip * r_ip / (r * dr * dr)
        coeff_im = chi_im * r_im / (r * dr * dr)
        b_diag[i] = 1.0 + 0.5 * dt * (coeff_ip + coeff_im)
        if i < n - 1:  # pragma: no branch - always True for i in range(1, n-1); see #129
            c_sup[i] = -0.5 * dt * coeff_ip
        a_sub[i - 1] = -0.5 * dt * coeff_im

    rhs = T + 0.5 * dt * Lh + dt * source
    T_new = _thomas_solve_np(a_sub, b_diag, c_sup, rhs)
    T_new[0] = T_new[1]
    T_new[-1] = T_edge
    return T_new


def batched_crank_nicolson(
    T_batch: AnyFloatArray,
    chi: AnyFloatArray,
    source: AnyFloatArray,
    rho: AnyFloatArray,
    drho: float,
    dt: float,
    T_edge: float = 0.1,
    *,
    allow_numpy_fallback: bool = False,
    allow_legacy_numpy_fallback: bool = False,
) -> AnyFloatArray:
    """Batched transport step via jax.vmap for ensemble/sensitivity runs.

    Parameters
    ----------
    T_batch : (batch, n) initial temperature profiles
    chi, source, rho, drho, dt, T_edge : shared across batch

    Returns
    -------
    T_new : (batch, n) updated profiles
    """
    use_jax_runtime = _resolve_use_jax(
        True,
        allow_numpy_fallback=allow_numpy_fallback,
        allow_legacy_numpy_fallback=allow_legacy_numpy_fallback,
        context="batched_crank_nicolson",
    )
    batch = np.asarray(T_batch)
    if batch.ndim != 2 or batch.shape[0] == 0:
        raise InvalidShapeError("CN batch must be a nonempty two-dimensional array")
    for row in batch:
        certify_diffusion_cn(row, chi, source, rho, drho, dt, T_edge)
    if not use_jax_runtime:
        return np.stack(
            [
                crank_nicolson_step(T_batch[i], chi, source, rho, drho, dt, T_edge, use_jax=False)
                for i in range(T_batch.shape[0])
            ]
        )

    chi_j = jnp.asarray(chi, dtype=jnp.float64)
    source_j = jnp.asarray(source, dtype=jnp.float64)
    rho_j = jnp.asarray(rho, dtype=jnp.float64)

    @jax.vmap
    def step(T_single: jnp.ndarray) -> jnp.ndarray:
        result: jnp.ndarray = _cn_step_jax(T_single, chi_j, source_j, rho_j, float(drho), float(dt), float(T_edge))
        return result

    return np.asarray(step(jnp.asarray(T_batch, dtype=jnp.float64)))

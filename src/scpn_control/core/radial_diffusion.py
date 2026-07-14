# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Radial Diffusion PDE Numerics

"""Crank-Nicolson numerics for the 1-D radial transport diffusion equation.

Stateless discretisation helpers extracted from the integrated transport
solver: the explicit cylindrical diffusion operator, the Crank-Nicolson
tridiagonal assembly, and the Thomas tridiagonal solve. The radial grid
(``rho``, ``drho``, minor radius ``a``) is passed explicitly so the numerics
are independent of any solver state.
"""

from __future__ import annotations

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

__all__ = [
    "build_cn_tridiag",
    "explicit_diffusion_rhs",
    "thomas_solve",
]


def thomas_solve(a: AnyFloatArray, b: AnyFloatArray, c: AnyFloatArray, d: AnyFloatArray) -> FloatArray:
    """Solve a tridiagonal system in O(n) with the Thomas algorithm.

    Solves ``A x = d`` where ``A`` is tridiagonal with sub-diagonal *a*, main
    diagonal *b*, and super-diagonal *c*. Near-zero or non-finite pivots are
    floored so a degenerate row cannot propagate NaNs.

    Parameters
    ----------
    a : array, length n-1
        Sub-diagonal.
    b : array, length n
        Main diagonal.
    c : array, length n-1
        Super-diagonal.
    d : array, length n
        Right-hand side.

    Returns
    -------
    x : array, length n
        Solution vector.
    """
    n = len(d)
    # Work on copies to avoid mutating input
    cp = np.empty(n - 1)
    dp = np.empty(n)

    b0 = float(b[0])
    if (not np.isfinite(b0)) or abs(b0) < 1e-30:
        b0 = 1e-30
    cp0 = float(c[0]) / b0
    dp0 = float(d[0]) / b0
    cp[0] = cp0 if np.isfinite(cp0) else 0.0
    dp[0] = dp0 if np.isfinite(dp0) else 0.0

    for i in range(1, n):
        m = b[i] - a[i - 1] * (cp[i - 1] if i - 1 < len(cp) else 0.0)
        if (not np.isfinite(m)) or abs(m) < 1e-30:
            m = 1e-30
        numer = d[i] - a[i - 1] * dp[i - 1]
        if not np.isfinite(numer):
            numer = 0.0
        dp_i = numer / m
        dp[i] = dp_i if np.isfinite(dp_i) else 0.0
        if i < n - 1:
            cp_i = c[i] / m
            cp[i] = cp_i if np.isfinite(cp_i) else 0.0

    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x_i = dp[i] - cp[i] * x[i + 1]
        x[i] = x_i if np.isfinite(x_i) else 0.0

    return x


def explicit_diffusion_rhs(
    T: AnyFloatArray, chi: AnyFloatArray, rho: AnyFloatArray, drho: float, a_minor: float
) -> FloatArray:
    """Compute the explicit cylindrical diffusion operator L_h(T).

    ``L_h(T) = (1/a^2) * (1/rho) d/drho(rho chi dT/drho)``, evaluated with
    half-grid diffusivities and central differences on the interior. Returns an
    array of the same length as *T* (boundary points left at zero).
    """
    n = len(T)
    Lh = np.zeros(n)
    dr = drho

    # Precompute 1/a^2 factor for units [1/s]
    scale = 1.0 / max(a_minor**2, 1e-6)

    for i in range(1, n - 1):
        r = rho[i]
        # half-grid chi
        chi_ip = 0.5 * (chi[i] + chi[i + 1])
        chi_im = 0.5 * (chi[i] + chi[i - 1])
        r_ip = r + 0.5 * dr
        r_im = r - 0.5 * dr

        flux_ip = chi_ip * r_ip * (T[i + 1] - T[i]) / dr
        flux_im = chi_im * r_im * (T[i] - T[i - 1]) / dr

        Lh[i] = scale * (flux_ip - flux_im) / (r * dr)

    return Lh


def build_cn_tridiag(
    chi: AnyFloatArray, dt: float, rho: AnyFloatArray, drho: float, a_minor: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Build the Crank-Nicolson LHS tridiagonal coefficients.

    The implicit system is
    ``(I - 0.5*dt*L_h) T^{n+1} = (I + 0.5*dt*L_h) T^n + dt*(S - Sink)``.

    Returns ``(a, b, c)`` sub/main/super diagonals for the interior points,
    padded to full grid size (boundary conditions applied separately).
    """
    n = len(rho)
    dr = drho
    scale = 1.0 / max(a_minor**2, 1e-6)

    a = np.zeros(n - 1)  # sub-diagonal
    b = np.ones(n)  # main diagonal
    c = np.zeros(n - 1)  # super-diagonal

    for i in range(1, n - 1):
        r = rho[i]
        chi_ip = 0.5 * (chi[i] + chi[i + 1])
        chi_im = 0.5 * (chi[i] + chi[i - 1])
        r_ip = r + 0.5 * dr
        r_im = r - 0.5 * dr

        coeff_ip = scale * chi_ip * r_ip / (r * dr * dr)
        coeff_im = scale * chi_im * r_im / (r * dr * dr)

        # Crank-Nicolson LHS: (I - 0.5·dt·L_h)
        b[i] = 1.0 + 0.5 * dt * (coeff_ip + coeff_im)
        c[i] = -0.5 * dt * coeff_ip  # T_{i+1} coefficient
        a[i - 1] = -0.5 * dt * coeff_im  # T_{i-1} coefficient

    return a, b, c

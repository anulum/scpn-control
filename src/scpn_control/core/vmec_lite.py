# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""
Simplified spectral 3D MHD equilibrium solver (VMEC-lite, fixed-boundary).

References
----------
Hirshman, S. P. & Whitson, J. C., Phys. Fluids 26 (1983) 3553.
    Variational moment method for 3D MHD equilibria; spectral representation
    R(s,θ,ζ) = Σ R_mn cos(mθ − nζ),  Z = Σ Z_mn sin(mθ − nζ)  [Eqs. 1–2].
Freidberg, J. P., "Ideal MHD" (Cambridge, 2014), Ch. 3.
    Force balance ∇p = J × B in 3D geometry.
Wesson, J., "Tokamaks" 4th ed. (Oxford, 2011), Ch. 3.
    Toroidal field B_φ ∝ 1/R; poloidal field from rotational transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VMECResult:
    R_mn: np.ndarray
    Z_mn: np.ndarray
    B_mn: np.ndarray
    force_residual: float
    iterations: int
    converged: bool


class SpectralBasis:
    """Fourier basis for VMEC spectral representation.

    Hirshman & Whitson 1983, Phys. Fluids 26, 3553, Eqs. 1–2:
        R(s,θ,ζ) = Σ_{m,n} R_mn(s) cos(mθ − n N_fp ζ)
        Z(s,θ,ζ) = Σ_{m,n} Z_mn(s) sin(mθ − n N_fp ζ)
    """

    def __init__(self, m_pol: int, n_tor: int, n_fp: int):
        self.m_pol = m_pol
        self.n_tor = n_tor
        self.n_fp = n_fp

        self.mn_modes: list[tuple[int, int]] = []
        for m in range(m_pol + 1):
            n_min = -n_tor if m > 0 else 0
            for n in range(n_min, n_tor + 1):
                self.mn_modes.append((m, n))

        self.n_modes = len(self.mn_modes)

    def evaluate(self, coeffs_mn: np.ndarray, theta: np.ndarray, zeta: np.ndarray, is_sin: bool = False) -> np.ndarray:
        """Evaluate spectral expansion at (θ, ζ) grid points.

        Hirshman & Whitson 1983, Eq. 1–2:
            Σ C_mn cos(mθ − n N_fp ζ)  or  Σ C_mn sin(mθ − n N_fp ζ)
        """
        val = np.zeros_like(theta)
        for i, (m, n) in enumerate(self.mn_modes):
            arg = m * theta - n * self.n_fp * zeta
            basis = np.sin(arg) if is_sin else np.cos(arg)
            val += coeffs_mn[i] * basis
        return val


class VMECLiteSolver:
    """Spectral 3D equilibrium solver using steepest descent on the MHD energy.

    Hirshman & Whitson 1983, Phys. Fluids 26, 3553 — variational moment method.
    Force balance ∇p = J × B (Freidberg 2014, Ch. 3) drives the Shafranov shift
    on the R₀₀ mode; radial tension (finite-difference Laplacian) regularises
    the spectral coefficients.
    """

    def __init__(self, n_s: int = 21, m_pol: int = 3, n_tor: int = 2, n_fp: int = 1):
        self.n_s = n_s
        self.basis = SpectralBasis(m_pol, n_tor, n_fp)

        self.R_mn = np.zeros((n_s, self.basis.n_modes))
        self.Z_mn = np.zeros((n_s, self.basis.n_modes))

        self.pressure = np.zeros(n_s)
        self.iota = np.zeros(n_s)

        self.s_grid = np.linspace(0.0, 1.0, n_s)

    def set_boundary(self, R_bound: dict[tuple[int, int], float], Z_bound: dict[tuple[int, int], float]) -> None:
        """Fix boundary Fourier coefficients at s = 1."""
        for i, (m, n) in enumerate(self.basis.mn_modes):
            if (m, n) in R_bound:
                self.R_mn[-1, i] = R_bound[(m, n)]
            if (m, n) in Z_bound:
                self.Z_mn[-1, i] = Z_bound[(m, n)]

    def set_profiles(self, pressure: np.ndarray, iota: np.ndarray) -> None:
        self.pressure = np.interp(self.s_grid, np.linspace(0, 1, len(pressure)), pressure)
        self.iota = np.interp(self.s_grid, np.linspace(0, 1, len(iota)), iota)

    def _initial_guess(self) -> None:
        # Linear radial interpolation from axis to boundary
        # Hirshman & Whitson 1983: s^(m/2) scaling for poloidal harmonics
        idx_00 = self.basis.mn_modes.index((0, 0)) if (0, 0) in self.basis.mn_modes else -1
        R00_bound = self.R_mn[-1, idx_00] if idx_00 >= 0 else 0.0

        for i, (m, n) in enumerate(self.basis.mn_modes):
            if m == 0 and n == 0:
                self.R_mn[:, i] = R00_bound
                self.Z_mn[:, i] = 0.0
            else:
                self.R_mn[:, i] = self.s_grid ** (m / 2.0) * self.R_mn[-1, i]
                self.Z_mn[:, i] = self.s_grid ** (m / 2.0) * self.Z_mn[-1, i]

    def solve(self, max_iter: int = 100, tol: float = 1e-4) -> VMECResult:
        """Steepest-descent equilibrium iteration.

        Hirshman & Whitson 1983, Phys. Fluids 26, 3553.

        Force balance ∇p = J × B (Freidberg 2014, Ch. 3) drives the
        pressure-gradient Shafranov shift on R₀₀ via q²(dp/ds)/R₀.
        Radial tension (d²/ds² finite difference) regularises spectral modes.
        """
        self._initial_guess()

        converged = False
        residual = float("inf")
        lr = 0.1

        idx_00 = next((i for i, mn in enumerate(self.basis.mn_modes) if mn == (0, 0)), -1)
        dp_ds = np.gradient(self.pressure, self.s_grid)
        q_profile = 1.0 / np.maximum(np.abs(self.iota), 0.01)
        R_00_bound = max(abs(self.R_mn[-1, idx_00]), 1e-3) if idx_00 >= 0 else 1.0

        for it in range(max_iter):
            F_R = np.zeros_like(self.R_mn)
            F_Z = np.zeros_like(self.Z_mn)

            for i in range(1, self.n_s - 1):
                F_R[i] = (self.R_mn[i + 1] - 2 * self.R_mn[i] + self.R_mn[i - 1]) * 2.0
                F_Z[i] = (self.Z_mn[i + 1] - 2 * self.Z_mn[i] + self.Z_mn[i - 1]) * 2.0

            # Pressure-driven Shafranov shift: F ~ −q²(dp/ds)/R₀
            # Freidberg 2014, Ideal MHD, Ch. 3 — radial force balance
            if idx_00 >= 0:
                for i in range(1, self.n_s - 1):
                    F_R[i, idx_00] -= q_profile[i] ** 2 * dp_ds[i] / R_00_bound * 1e-6

            residual = float(np.max(np.abs(F_R)) + np.max(np.abs(F_Z)))

            if residual < tol:
                converged = True
                break

            self.R_mn[1:-1] += lr * F_R[1:-1]
            self.Z_mn[1:-1] += lr * F_Z[1:-1]

        # B field from rotational transform ι and geometry
        # B_φ ∝ 1/R (Wesson 2011, Ch. 3), B_θ ∝ ι
        B_mn = np.zeros_like(self.R_mn)
        for s in range(self.n_s):
            R_00 = max(abs(self.R_mn[s, idx_00]), 1e-6) if idx_00 >= 0 else 1.0
            iota_s = self.iota[s]
            if idx_00 >= 0:
                B_mn[s, idx_00] = 1.0
            for k, (m, _n) in enumerate(self.basis.mn_modes):
                if k == idx_00:
                    continue
                B_mn[s, k] = -self.R_mn[s, k] / R_00
                if m == 1:
                    B_mn[s, k] += iota_s * abs(self.Z_mn[s, k]) / R_00

        return VMECResult(self.R_mn.copy(), self.Z_mn.copy(), B_mn, float(residual), it + 1, converged)


class AxisymmetricTokamakBoundary:
    @staticmethod
    def from_parameters(
        R0: float, a: float, kappa: float, delta: float
    ) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
        """Low-order Fourier approximation of shaped tokamak boundary.

        R = R₀ + a cos(θ + δ sin θ),  Z = a κ sin θ
        Hirshman & Whitson 1983, Phys. Fluids 26, 3553, Eqs. 1–2.
        """
        b_R: dict[tuple[int, int], float] = {
            (0, 0): R0,
            (1, 0): a,
            (2, 0): -0.5 * a * delta,
        }
        b_Z: dict[tuple[int, int], float] = {(1, 0): a * kappa}
        return b_R, b_Z


class StellaratorBoundary:
    @staticmethod
    def w7x_standard() -> tuple[dict, dict]:
        """W7-X standard configuration boundary.

        Grieger et al., Phys. Fluids B 4 (1992) 2081 — modular coil geometry.
        N_fp = 5.
        """
        b_R = {(0, 0): 5.5, (1, 0): 0.5, (1, 1): 0.1, (0, 1): 0.2}
        b_Z = {(1, 0): 0.6, (1, 1): -0.1}
        return b_R, b_Z

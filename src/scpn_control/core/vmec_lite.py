# ──────────────────────────────────────────────────────────────────────
# SCPN Control — 3D MHD Equilibrium (VMEC-lite Fixed-Boundary)
# ──────────────────────────────────────────────────────────────────────
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
    def __init__(self, m_pol: int, n_tor: int, n_fp: int):
        self.m_pol = m_pol
        self.n_tor = n_tor
        self.n_fp = n_fp

        self.mn_modes = []
        for m in range(m_pol + 1):
            n_min = -n_tor if m > 0 else 0
            for n in range(n_min, n_tor + 1):
                self.mn_modes.append((m, n))

        self.n_modes = len(self.mn_modes)

    def evaluate(self, coeffs_mn: np.ndarray, theta: np.ndarray, zeta: np.ndarray, is_sin: bool = False) -> np.ndarray:
        # Evaluate sum C_mn * cos(m*theta - n*N_fp*zeta) or sin(...)
        val = np.zeros_like(theta)
        for i, (m, n) in enumerate(self.mn_modes):
            arg = m * theta - n * self.n_fp * zeta
            basis = np.sin(arg) if is_sin else np.cos(arg)
            val += coeffs_mn[i] * basis
        return val


class VMECLiteSolver:
    """
    Highly simplified spectral 3D equilibrium solver mimicking VMEC principles.
    Uses steepest descent to minimize a mock MHD energy functional.
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
        """Set fixed boundary conditions at s=1."""
        for i, (m, n) in enumerate(self.basis.mn_modes):
            if (m, n) in R_bound:
                self.R_mn[-1, i] = R_bound[(m, n)]
            if (m, n) in Z_bound:
                self.Z_mn[-1, i] = Z_bound[(m, n)]

    def set_profiles(self, pressure: np.ndarray, iota: np.ndarray) -> None:
        # Interpolate onto s_grid
        self.pressure = np.interp(self.s_grid, np.linspace(0, 1, len(pressure)), pressure)
        self.iota = np.interp(self.s_grid, np.linspace(0, 1, len(iota)), iota)

    def _initial_guess(self) -> None:
        # Linear interpolation from magnetic axis to boundary
        R00_bound = 0.0
        # Find (0,0) index
        idx_00 = self.basis.mn_modes.index((0, 0)) if (0, 0) in self.basis.mn_modes else -1
        if idx_00 >= 0:
            R00_bound = self.R_mn[-1, idx_00]

        for i, (m, n) in enumerate(self.basis.mn_modes):
            if m == 0 and n == 0:
                self.R_mn[:, i] = R00_bound
                self.Z_mn[:, i] = 0.0
            else:
                self.R_mn[:, i] = self.s_grid ** (m / 2.0) * self.R_mn[-1, i]
                self.Z_mn[:, i] = self.s_grid ** (m / 2.0) * self.Z_mn[-1, i]

    def _mhd_energy(self) -> float:
        # Mock energy functional: W = int (B^2 / 2mu0 + p / (gamma-1)) dV
        # We simplify massively for the solver mock
        W = 0.0
        for s in range(self.n_s):
            # Penalty for R, Z deviations to simulate force balance
            dev_R = np.sum(self.R_mn[s] ** 2)
            dev_Z = np.sum(self.Z_mn[s] ** 2)
            W += dev_R + dev_Z - self.pressure[s]
        return W

    def solve(self, max_iter: int = 100, tol: float = 1e-4) -> VMECResult:
        self._initial_guess()

        converged = False
        residual = float("inf")

        # Extremely simplified steepest descent mock
        # Real VMEC evaluates spectral forces F_R, F_Z via FFTs.
        lr = 0.1

        for it in range(max_iter):
            # Mock forces pulling towards a smooth profile
            # F ~ d2/ds2 (R)
            F_R = np.zeros_like(self.R_mn)
            F_Z = np.zeros_like(self.Z_mn)

            for i in range(1, self.n_s - 1):
                F_R[i] = (self.R_mn[i + 1] - 2 * self.R_mn[i] + self.R_mn[i - 1]) * 2.0
                F_Z[i] = (self.Z_mn[i + 1] - 2 * self.Z_mn[i] + self.Z_mn[i - 1]) * 2.0

            # Add pressure drive (mock shift)
            idx_10 = self.basis.mn_modes.index((1, 0)) if (1, 0) in self.basis.mn_modes else -1
            if idx_10 >= 0:
                for i in range(1, self.n_s - 1):
                    # Shafranov shift mock: R_00 shifts due to pressure
                    dp = self.pressure[i] - self.pressure[i - 1]
                    F_R[i, 0] -= dp * 0.001

            residual = np.max(np.abs(F_R)) + np.max(np.abs(F_Z))

            if residual < tol:
                converged = True
                break

            # Update (interior only)
            self.R_mn[1:-1] += lr * F_R[1:-1]
            self.Z_mn[1:-1] += lr * F_Z[1:-1]

        # B ~ 1/R → B_mn from inverse of R Fourier expansion
        B_mn = np.zeros_like(self.R_mn)
        idx_00 = self.basis.mn_modes.index((0, 0)) if (0, 0) in self.basis.mn_modes else -1
        for s in range(self.n_s):
            R_00 = self.R_mn[s, idx_00] if idx_00 >= 0 else 1.0
            R_00 = max(abs(R_00), 1e-6)
            if idx_00 >= 0:
                B_mn[s, idx_00] = 1.0
            for k in range(self.basis.n_modes):
                if k != idx_00:
                    B_mn[s, k] = -self.R_mn[s, k] / R_00

        return VMECResult(self.R_mn.copy(), self.Z_mn.copy(), B_mn, float(residual), it + 1, converged)


class AxisymmetricTokamakBoundary:
    @staticmethod
    def from_parameters(
        R0: float, a: float, kappa: float, delta: float
    ) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
        # R = R0 + a * cos(theta + delta * sin(theta))
        # Z = a * kappa * sin(theta)
        # Approximate to low order Fourier
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
        # N_fp = 5
        b_R = {(0, 0): 5.5, (1, 0): 0.5, (1, 1): 0.1, (0, 1): 0.2}
        b_Z = {(1, 0): 0.6, (1, 1): -0.1}
        return b_R, b_Z

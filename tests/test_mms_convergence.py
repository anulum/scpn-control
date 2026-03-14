# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Method of Manufactured Solutions Convergence Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""
MMS convergence tests for the Grad-Shafranov elliptic solver.

Verifies 2nd-order spatial accuracy by solving Δ*ψ = f on grids of
increasing resolution with known exact solutions, then computing the
convergence rate from the error ratio.

Three manufactured solutions exercise different terms of the GS* operator:

  Case 1 (Solov'ev):    ψ = c₁R⁴/8 + c₂Z²
                         Δ*ψ = c₁R² + 2c₂
  Case 2 (Polynomial):  ψ = R⁴/8 + Z⁴/12
                         Δ*ψ = R² + Z²
  Case 3 (Trig+poly):   ψ = R³·sin(πZ/L_Z)
                         Δ*ψ = (3R - R³(π/L_Z)²)·sin(πZ/L_Z)

The GS* operator is  ∂²ψ/∂R² - (1/R)∂ψ/∂R + ∂²ψ/∂Z².
"""

from __future__ import annotations

import numpy as np
import pytest


# ── Solver ────────────────────────────────────────────────────────────


def _solve_gs_star(
    R_min: float,
    R_max: float,
    Z_min: float,
    Z_max: float,
    nr: int,
    nz: int,
    psi_exact_fn,
    source_fn,
    max_iter: int = 30000,
    tol: float = 1e-10,
    omega: float = 1.2,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve Δ*ψ = f with Dirichlet BCs from psi_exact_fn.

    Returns (psi_numerical, psi_exact, NRMSE).
    """
    R = np.linspace(R_min, R_max, nr)
    Z = np.linspace(Z_min, Z_max, nz)
    dR = (R_max - R_min) / (nr - 1)
    dZ = (Z_max - Z_min) / (nz - 1)
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")

    psi_exact = psi_exact_fn(RR, ZZ)
    source = source_fn(RR, ZZ)

    psi = np.zeros((nr, nz))
    psi[0, :] = psi_exact[0, :]
    psi[-1, :] = psi_exact[-1, :]
    psi[:, 0] = psi_exact[:, 0]
    psi[:, -1] = psi_exact[:, -1]

    dR2 = dR**2
    dZ2 = dZ**2

    for k in range(max_iter):
        for i in range(1, nr - 1):
            R_i = R[i]
            a_E = 1.0 / dR2 - 1.0 / (2.0 * R_i * dR)
            a_W = 1.0 / dR2 + 1.0 / (2.0 * R_i * dR)
            a_NS = 1.0 / dZ2
            a_C = 2.0 / dR2 + 2.0 / dZ2

            gs = (
                a_E * psi[i + 1, 1:-1] + a_W * psi[i - 1, 1:-1] + a_NS * (psi[i, 2:] + psi[i, :-2]) - source[i, 1:-1]
            ) / a_C
            psi[i, 1:-1] += omega * (gs - psi[i, 1:-1])

        if k % 200 == 0:
            res = (
                (1.0 / dR2 - 1.0 / (2.0 * RR[1:-1, 1:-1] * dR)) * psi[2:, 1:-1]
                + (1.0 / dR2 + 1.0 / (2.0 * RR[1:-1, 1:-1] * dR)) * psi[:-2, 1:-1]
                + (1.0 / dZ2) * (psi[1:-1, 2:] + psi[1:-1, :-2])
                - (2.0 / dR2 + 2.0 / dZ2) * psi[1:-1, 1:-1]
                - source[1:-1, 1:-1]
            )
            if np.max(np.abs(res)) < tol:
                break

    interior = psi[1:-1, 1:-1]
    exact_int = psi_exact[1:-1, 1:-1]
    abs_err = np.abs(interior - exact_int)
    psi_range = np.max(psi_exact) - np.min(psi_exact)
    nrmse = float(np.sqrt(np.mean(abs_err**2)) / max(psi_range, 1e-15))
    return psi, psi_exact, nrmse


def _convergence_rate(nrmse_coarse: float, nrmse_fine: float, h_ratio: float) -> float:
    if nrmse_fine <= 0 or nrmse_coarse <= 0:
        return 0.0
    return float(np.log(nrmse_coarse / nrmse_fine) / np.log(h_ratio))


# ── Manufactured solutions ────────────────────────────────────────────

R_MIN, R_MAX = 1.0, 3.0
Z_MIN, Z_MAX = -1.5, 1.5
L_Z = Z_MAX - Z_MIN


def _solovev_psi(R, Z):
    return R**4 / 8.0 + 0.5 * Z**2


def _solovev_source(R, Z):
    return R**2 + 1.0


def _poly_psi(R, Z):
    return R**4 / 8.0 + Z**4 / 12.0


def _poly_source(R, Z):
    return R**2 + Z**2


def _trig_psi(R, Z):
    return R**3 * np.sin(np.pi * Z / L_Z)


def _trig_source(R, Z):
    # Δ*ψ = (3R - R³(π/L_Z)²)·sin(πZ/L_Z)
    return (3.0 * R - R**3 * (np.pi / L_Z) ** 2) * np.sin(np.pi * Z / L_Z)


# ── Tests ─────────────────────────────────────────────────────────────


_RESOLUTIONS = [17, 33, 65]
# For faster CI, use 3 levels. Full study (17→129) in validation/mesh_convergence_study.py.

_MMS_CASES = [
    ("solovev", _solovev_psi, _solovev_source),
    ("polynomial", _poly_psi, _poly_source),
    ("trigonometric", _trig_psi, _trig_source),
]


class TestMMSConvergenceOrder:
    """Verify 2nd-order spatial convergence for each manufactured solution."""

    @pytest.mark.parametrize("name,psi_fn,src_fn", _MMS_CASES, ids=[c[0] for c in _MMS_CASES])
    def test_convergence_rate_ge_1_8(self, name, psi_fn, src_fn):
        """Convergence rate from coarsest-to-finest pair must be >= 1.8 (theoretical: 2.0)."""
        nrmses = []
        hs = []
        for res in _RESOLUTIONS:
            _, _, nrmse = _solve_gs_star(
                R_MIN,
                R_MAX,
                Z_MIN,
                Z_MAX,
                res,
                res,
                psi_fn,
                src_fn,
                max_iter=25000,
                tol=1e-10,
            )
            nrmses.append(nrmse)
            hs.append((R_MAX - R_MIN) / (res - 1))

        # Convergence rate between successive pairs
        rates = []
        for i in range(1, len(nrmses)):
            rate = _convergence_rate(nrmses[i - 1], nrmses[i], hs[i - 1] / hs[i])
            rates.append(rate)

        # Best rate (finest pair) must be >= 1.8
        best_rate = rates[-1]
        assert best_rate >= 1.8, (
            f"MMS '{name}': convergence rate {best_rate:.2f} < 1.8 "
            f"(errors: {[f'{e:.2e}' for e in nrmses]}, rates: {[f'{r:.2f}' for r in rates]})"
        )

    @pytest.mark.parametrize("name,psi_fn,src_fn", _MMS_CASES, ids=[c[0] for c in _MMS_CASES])
    def test_finest_grid_error_small(self, name, psi_fn, src_fn):
        """Error on the finest grid (65x65) must be < 1e-3 NRMSE."""
        res = _RESOLUTIONS[-1]
        _, _, nrmse = _solve_gs_star(
            R_MIN,
            R_MAX,
            Z_MIN,
            Z_MAX,
            res,
            res,
            psi_fn,
            src_fn,
            max_iter=25000,
            tol=1e-10,
        )
        assert nrmse < 1e-3, f"MMS '{name}': NRMSE {nrmse:.2e} >= 1e-3 on {res}x{res} grid"


class TestMMSErrorMonotonicity:
    """Error must decrease monotonically with grid refinement."""

    @pytest.mark.parametrize("name,psi_fn,src_fn", _MMS_CASES, ids=[c[0] for c in _MMS_CASES])
    def test_error_decreases(self, name, psi_fn, src_fn):
        nrmses = []
        for res in _RESOLUTIONS:
            _, _, nrmse = _solve_gs_star(
                R_MIN,
                R_MAX,
                Z_MIN,
                Z_MAX,
                res,
                res,
                psi_fn,
                src_fn,
                max_iter=25000,
                tol=1e-10,
            )
            nrmses.append(nrmse)

        for i in range(1, len(nrmses)):
            assert nrmses[i] < nrmses[i - 1], (
                f"MMS '{name}': error did not decrease: "
                f"res {_RESOLUTIONS[i - 1]} → {_RESOLUTIONS[i]}: "
                f"{nrmses[i - 1]:.2e} → {nrmses[i]:.2e}"
            )

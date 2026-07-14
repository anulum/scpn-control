# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Radial Diffusion PDE Numerics Tests
"""Tests for the Crank-Nicolson radial-diffusion numerics.

Covers the Thomas tridiagonal solve (including its singular/non-finite pivot
guards), the explicit cylindrical diffusion operator, and the Crank-Nicolson
tridiagonal assembly extracted from the integrated transport solver.
"""

from __future__ import annotations

import numpy as np

from scpn_control.core.radial_diffusion import (
    build_cn_tridiag,
    explicit_diffusion_rhs,
    thomas_solve,
)


class TestThomasSolve:
    def test_identity_system_returns_rhs(self) -> None:
        """The identity matrix returns the right-hand side unchanged."""
        n = 10
        a = np.zeros(n - 1)
        b = np.ones(n)
        c = np.zeros(n - 1)
        d = np.arange(n, dtype=float)
        x = thomas_solve(a, b, c, d)
        np.testing.assert_allclose(x, d, atol=1e-12)

    def test_known_tridiagonal_poisson_system(self) -> None:
        """A known Poisson tridiagonal system solves to a finite profile."""
        n = 50
        a = -np.ones(n - 1)
        b = 2.0 * np.ones(n)
        c = -np.ones(n - 1)
        d = np.ones(n) * (1.0 / (n - 1)) ** 2
        d[0] = 0.0
        d[-1] = 0.0
        b[0] = 1.0
        c[0] = 0.0
        a[-1] = 0.0
        b[-1] = 1.0
        x = thomas_solve(a, b, c, d)
        assert x.shape == (n,)
        assert np.all(np.isfinite(x))

    def test_floors_near_zero_initial_diagonal(self) -> None:
        """A zero first diagonal is floored without producing NaNs."""
        n = 10
        a = np.zeros(n - 1)
        b = np.ones(n)
        b[0] = 0.0
        c = np.zeros(n - 1)
        d = np.ones(n)
        x = thomas_solve(a, b, c, d)
        assert x.shape == (n,)
        assert np.all(np.isfinite(x))

    def test_repairs_nonfinite_initial_diagonal(self) -> None:
        """A non-finite first diagonal is repaired to a finite solution."""
        n = 10
        a = np.zeros(n - 1)
        b = np.ones(n)
        b[0] = float("nan")
        c = np.zeros(n - 1)
        d = np.ones(n)
        x = thomas_solve(a, b, c, d)
        assert x.shape == (n,)
        assert np.all(np.isfinite(x))

    def test_repairs_singular_and_nonfinite_inner_rows(self) -> None:
        """Singular and non-finite interior rows are floored, keeping the solve finite."""
        a = np.array([1.0, 1.0])
        b = np.array([1e-31, 1e-31, 1e-31])
        c = np.array([0.0, 0.0])
        d = np.array([1.0, np.inf, 1.0])
        x = thomas_solve(a, b, c, d)
        assert x.shape == (3,)
        assert np.all(np.isfinite(x))


class TestExplicitDiffusionRhs:
    def test_zero_gradient_gives_zero_operator(self) -> None:
        """A flat temperature profile produces a zero diffusion operator."""
        rho = np.linspace(0.0, 1.0, 21)
        T = np.full_like(rho, 3.0)
        chi = np.ones_like(rho)
        Lh = explicit_diffusion_rhs(T, chi, rho, drho=rho[1] - rho[0], a_minor=2.0)
        assert Lh.shape == rho.shape
        np.testing.assert_allclose(Lh[1:-1], 0.0, atol=1e-12)

    def test_boundaries_untouched_and_finite(self) -> None:
        """Boundary points stay zero and the interior is finite for a curved profile."""
        rho = np.linspace(0.0, 1.0, 21)
        T = 1.0 + rho**2
        chi = np.ones_like(rho)
        Lh = explicit_diffusion_rhs(T, chi, rho, drho=rho[1] - rho[0], a_minor=2.0)
        assert Lh[0] == 0.0
        assert Lh[-1] == 0.0
        assert np.all(np.isfinite(Lh))


class TestBuildCnTridiag:
    def test_shapes_and_interior_positivity(self) -> None:
        """The assembly returns correctly shaped diagonals with a diagonally dominant main."""
        rho = np.linspace(0.0, 1.0, 21)
        chi = np.ones_like(rho)
        a, b, c = build_cn_tridiag(chi, dt=0.5, rho=rho, drho=rho[1] - rho[0], a_minor=2.0)
        assert a.shape == (rho.size - 1,)
        assert b.shape == (rho.size,)
        assert c.shape == (rho.size - 1,)
        # Interior main-diagonal entries exceed unity for a stable implicit step
        assert np.all(b[1:-1] > 1.0)

    def test_solvable_against_thomas(self) -> None:
        """The assembled system is solvable by the Thomas solver to a finite profile."""
        rho = np.linspace(0.0, 1.0, 21)
        chi = np.ones_like(rho)
        a, b, c = build_cn_tridiag(chi, dt=0.1, rho=rho, drho=rho[1] - rho[0], a_minor=2.0)
        rhs = np.ones_like(rho)
        x = thomas_solve(a, b, c, rhs)
        assert x.shape == rho.shape
        assert np.all(np.isfinite(x))

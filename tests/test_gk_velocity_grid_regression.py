# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK velocity-grid regression tests
"""Regression coverage for GK quadrature and local dispersion refinement."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.gk_eigenvalue import _newton_refine_dispersion
from scpn_control.core.gk_species import VelocityGrid, _gauss_legendre_nodes_weights


def test_gauss_legendre_nodes_weights_match_four_point_reference() -> None:
    """The in-module quadrature matches the analytic four-point rule."""
    nodes, weights = _gauss_legendre_nodes_weights(4)

    np.testing.assert_allclose(
        nodes,
        np.array([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526]),
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        weights,
        np.array([0.34785484513745385, 0.6521451548625461, 0.6521451548625461, 0.34785484513745385]),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_velocity_grid_quadrature_weights_integrate_constant_domains() -> None:
    """Mapped energy and pitch-angle weights preserve interval lengths."""
    grid = VelocityGrid(n_energy=8, n_lambda=10)

    assert float(np.sum(grid.energy_weights)) == pytest.approx(grid.E_max)
    assert float(np.sum(grid.lambda_weights)) == pytest.approx(1.0)


def test_newton_refine_dispersion_converges_to_local_root() -> None:
    """Damped Newton refinement converges for a smooth scalar dispersion."""

    def dispersion(omega: complex) -> complex:
        return omega - (1.0 + 0.5j)

    def dispersion_derivative(_omega: complex) -> complex:
        return 1.0 + 0.0j

    root = _newton_refine_dispersion(
        0.0 + 0.0j,
        scale=1.0,
        dispersion=dispersion,
        dispersion_deriv=dispersion_derivative,
    )

    assert root == pytest.approx(1.0 + 0.5j)


def test_newton_refine_dispersion_damps_large_steps() -> None:
    """Large Newton steps are limited by the supplied physical scale."""

    def dispersion(_omega: complex) -> complex:
        return 10.0 + 0.0j

    def dispersion_derivative(_omega: complex) -> complex:
        return 1.0 + 0.0j

    root = _newton_refine_dispersion(
        0.0 + 0.0j,
        scale=0.25,
        dispersion=dispersion,
        dispersion_deriv=dispersion_derivative,
        max_steps=1,
    )

    assert root == pytest.approx(-0.5 + 0.0j)


def test_newton_refine_dispersion_returns_seed_for_degenerate_derivative() -> None:
    """A vanishing derivative returns the current seed instead of dividing by zero."""

    def dispersion(_omega: complex) -> complex:
        return 1.0 + 0.0j

    def dispersion_derivative(_omega: complex) -> complex:
        return 0.0 + 0.0j

    root = _newton_refine_dispersion(
        0.25 + 0.1j,
        scale=1.0,
        dispersion=dispersion,
        dispersion_deriv=dispersion_derivative,
    )

    assert root == pytest.approx(0.25 + 0.1j)

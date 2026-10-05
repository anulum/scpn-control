# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — magnetic saddle regression tests.

"""Exercise the public magnetic X-point estimate on known flux fields."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control._typing import AnyFloatArray
from scpn_control.control.realtime_efit import MagneticDiagnostics, RealtimeEFIT


def _solver() -> RealtimeEFIT:
    """Build a grid with an X-point located between sample nodes."""
    return RealtimeEFIT(
        MagneticDiagnostics([], [], 6.2),
        np.linspace(4.2, 8.2, 33),
        np.linspace(-3.0, 3.0, 33),
    )


def test_public_xpoint_finds_off_grid_saddle() -> None:
    """A quadratic saddle has an exact off-grid stationary point."""
    solver = _solver()
    rr, zz = np.meshgrid(solver.R, solver.Z, indexing="ij")
    psi = (rr - 6.17) ** 2 - 1.4 * (zz - 0.17) ** 2

    point = solver.find_xpoint(psi)

    assert point is not None
    assert point == pytest.approx((6.17, 0.17), abs=0.02)


def test_public_xpoint_refuses_flat_and_elliptic_fields() -> None:
    """Flat flux and an O-point cannot be reported as an X-point."""
    solver = _solver()
    rr, zz = np.meshgrid(solver.R, solver.Z, indexing="ij")
    assert solver.find_xpoint(np.zeros_like(rr)) is None
    assert solver.find_xpoint((rr - 6.17) ** 2 + (zz - 0.17) ** 2) is None


def test_public_xpoint_rejects_malformed_flux() -> None:
    """Grid mismatch and nonfinite flux are rejected before topology search."""
    solver = _solver()
    with pytest.raises(ValueError, match="shape"):
        solver.find_xpoint(np.zeros((3, 3)))
    psi = np.zeros((solver.nR, solver.nZ))
    psi[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        solver.find_xpoint(psi)


@pytest.mark.parametrize(
    ("r_grid", "z_grid", "message"),
    [
        (np.array([[1.0, 2.0, 3.0]]), np.array([0.0, 1.0, 2.0]), "at least three"),
        (np.array([1.0, 2.0]), np.array([0.0, 1.0, 2.0]), "at least three"),
        (np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0]), "at least three"),
        (np.array([1.0, np.nan, 3.0]), np.array([0.0, 1.0, 2.0]), "finite"),
        (np.array([1.0, 2.0, 3.0]), np.array([0.0, np.inf, 2.0]), "finite"),
        (np.array([1.0, 1.0, 3.0]), np.array([0.0, 1.0, 2.0]), "strictly increasing"),
        (np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 2.0]), "strictly increasing"),
    ],
)
def test_public_xpoint_rejects_invalid_grid(r_grid: AnyFloatArray, z_grid: AnyFloatArray, message: str) -> None:
    """The saddle estimate requires finite ordered coordinate axes."""
    solver = RealtimeEFIT(MagneticDiagnostics([], [], 2.0), r_grid, z_grid)
    with pytest.raises(ValueError, match=message):
        solver.find_xpoint(np.zeros((len(r_grid), len(z_grid))))


def test_public_xpoint_refuses_unrepresentable_derivatives_and_hessian() -> None:
    """Finite flux inputs must not publish nonfinite gradient or curvature."""
    r_grid = np.array([1.0, 1.01, 1.02])
    z_grid = np.array([0.0, 0.01, 0.02])
    solver = RealtimeEFIT(MagneticDiagnostics([], [], 1.0), r_grid, z_grid)
    steep = np.broadcast_to(np.array([0.0, 1.0e308, 0.0])[:, None], (3, 3))
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError, match="derivatives must be finite"):
        solver.find_xpoint(steep)

    r_grid = np.array([-1.0, 0.0, 1.0])
    z_grid = np.array([-1.0, 0.0, 1.0])
    solver = RealtimeEFIT(MagneticDiagnostics([], [], 1.0), r_grid, z_grid)
    rr, zz = np.meshgrid(r_grid, z_grid, indexing="ij")
    saddle = 1.0e155 * (rr**2 - zz**2)
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError, match="Hessian must be finite"):
        solver.find_xpoint(saddle)

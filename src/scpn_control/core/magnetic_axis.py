# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Checked magnetic-axis accessor for poloidal-flux maps.
"""Locate the magnetic axis of a poloidal-flux map and check that it is an interior O-point.

CONTROL's Python kernels store the poloidal flux ``Psi[iz, ir]`` with the
magnetic axis at its maximum. The global maximum is an admissible axis only when
it is an interior O-point: off the edge of the computational box and strictly
above every boundary value. A maximum on the box edge means the map has no
closed-flux O-point inside the box. A Picard-residual convergence flag can still
call such a state converged, so code that reads the axis of a solved equilibrium
must refuse it rather than control or map against an edge cell.

:func:`locate_magnetic_axis` reports the maximum and whether it is admissible,
and never refuses. Solver iterations use it: the kernel's Picard loop and the
coupled Grad-Shafranov/transport loop, which maps profiles before its first
equilibrium solve. :func:`require_magnetic_axis` refuses an inadmissible maximum
with :class:`MagneticAxisError`. Readers of a solved equilibrium use it: the MPC
shape state, the optimal controller, the flight simulator, the neuro-cybernetic
controller and the director interface. Whether a solver may report such a state
as converged is a separate contract of the solver itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scpn_control._typing import AnyFloatArray


class MagneticAxisError(ValueError):
    """The flux map has no interior magnetic axis (edge maximum, boundary tie or non-finite flux)."""


@dataclass(frozen=True)
class MagneticAxis:
    """Grid location, coordinates and flux of the global flux maximum.

    Attributes
    ----------
    iz, ir
        Row (Z) and column (R) indices of the maximum in ``Psi``.
    r, z
        Major radius and height of that grid point in metres.
    psi
        Flux at the maximum.
    interior
        True when the whole map is finite and the maximum is off the box edge
        and strictly above every boundary value, i.e. an interior O-point.
    """

    iz: int
    ir: int
    r: float
    z: float
    psi: float
    interior: bool

    def require_interior(self) -> MagneticAxis:
        """Return this axis if it is an interior O-point, else raise.

        Returns
        -------
        MagneticAxis
            ``self``.

        Raises
        ------
        MagneticAxisError
            If the maximum is not an interior O-point.
        """
        if not self.interior:
            raise MagneticAxisError(
                f"Psi maximum at (R, Z) = ({self.r:.6g}, {self.z:.6g}) m, grid ({self.iz}, {self.ir}), "
                "is not an interior O-point: it is non-finite, on the computational box edge, "
                "or not above the boundary flux"
            )
        return self


def locate_magnetic_axis(psi: AnyFloatArray, r: AnyFloatArray, z: AnyFloatArray) -> MagneticAxis:
    """Return the global flux maximum and whether it is an interior O-point.

    Parameters
    ----------
    psi
        Poloidal flux on the grid, shape ``(len(z), len(r))``, at least 3 x 3.
    r
        Major-radius grid coordinates in metres.
    z
        Height grid coordinates in metres.

    Returns
    -------
    MagneticAxis
        The maximum and its admissibility. Any non-finite value makes the map
        inadmissible (``interior=False``); NumPy then reports the first NaN cell
        as the maximum.

    Raises
    ------
    ValueError
        If the shapes disagree or the grid is smaller than 3 x 3.
    """
    flux = np.asarray(psi, dtype=np.float64)
    r_grid = np.asarray(r, dtype=np.float64).reshape(-1)
    z_grid = np.asarray(z, dtype=np.float64).reshape(-1)
    if flux.ndim != 2 or flux.shape != (z_grid.size, r_grid.size):
        raise ValueError(f"Psi shape {flux.shape} must be (len(Z), len(R)) = ({z_grid.size}, {r_grid.size})")
    if min(flux.shape) < 3:
        raise ValueError(f"Psi needs at least 3 x 3 points to have an interior, got {flux.shape}")
    iz, ir = (int(index) for index in np.unravel_index(int(np.argmax(flux)), flux.shape))
    peak = float(flux[iz, ir])
    finite = bool(np.isfinite(flux).all())
    boundary_peak = float(np.concatenate((flux[0], flux[-1], flux[:, 0], flux[:, -1])).max())
    off_edge = 0 < iz < flux.shape[0] - 1 and 0 < ir < flux.shape[1] - 1
    interior = finite and off_edge and peak > boundary_peak
    return MagneticAxis(iz=iz, ir=ir, r=float(r_grid[ir]), z=float(z_grid[iz]), psi=peak, interior=interior)


def require_magnetic_axis(psi: AnyFloatArray, r: AnyFloatArray, z: AnyFloatArray) -> MagneticAxis:
    """Return the magnetic axis of a solved flux map, refusing one that is not an interior O-point.

    Parameters
    ----------
    psi
        Poloidal flux on the grid, shape ``(len(z), len(r))``.
    r
        Major-radius grid coordinates in metres.
    z
        Height grid coordinates in metres.

    Returns
    -------
    MagneticAxis
        An axis with ``interior=True``.

    Raises
    ------
    MagneticAxisError
        If the maximum is non-finite, on the box edge or not above the boundary flux.
    ValueError
        If the shapes disagree or the grid is smaller than 3 x 3.
    """
    return locate_magnetic_axis(psi, r, z).require_interior()


__all__ = [
    "MagneticAxis",
    "MagneticAxisError",
    "locate_magnetic_axis",
    "require_magnetic_axis",
]

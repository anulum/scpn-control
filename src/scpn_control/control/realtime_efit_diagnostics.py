# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-time equilibrium reconstruction utilities.

"""Synthetic magnetic-diagnostic response for realtime EFIT."""

from __future__ import annotations

import numpy as np

from scpn_control._typing import AnyFloatArray
from scpn_control.control.realtime_efit_contracts import MU0, MagneticDiagnostics


def _trapezoid_integral(values: AnyFloatArray, grid: AnyFloatArray, *, axis: int = -1) -> AnyFloatArray:
    """Integrate profiles across NumPy 1.x and 2.x runtimes."""
    values_arr = np.asarray(values, dtype=float)
    grid_arr = np.asarray(grid, dtype=float)
    if grid_arr.ndim != 1:
        raise ValueError("trapezoidal integration grid must be one-dimensional")
    moved = np.moveaxis(values_arr, axis, -1)
    if moved.shape[-1] != grid_arr.shape[0]:
        raise ValueError("trapezoidal integration values and grid lengths must match")
    if grid_arr.size < 2:
        return np.zeros(moved.shape[:-1], dtype=float)
    widths = np.diff(grid_arr)
    return np.asarray(np.sum(0.5 * (moved[..., 1:] + moved[..., :-1]) * widths, axis=-1))


class DiagnosticResponse:
    """Synthetic magnetic-diagnostic response from a flux map.

    Parameters
    ----------
    diagnostics
        The magnetic diagnostic set (probes, loops, coils).
    R_grid
        Major-radius grid in metres.
    Z_grid
        Vertical grid in metres.
    """

    def __init__(self, diagnostics: MagneticDiagnostics, R_grid: AnyFloatArray, Z_grid: AnyFloatArray):
        self.diagnostics = diagnostics
        self.R = R_grid
        self.Z = Z_grid

    def simulate_measurements(
        self, psi: AnyFloatArray, coil_currents: AnyFloatArray
    ) -> dict[str, float | AnyFloatArray]:
        """Generate synthetic measurements from a given psi field."""
        from scipy.interpolate import RegularGridInterpolator

        psi_arr = np.asarray(psi, dtype=float)
        if psi_arr.shape != (len(self.R), len(self.Z)):
            raise ValueError("psi shape must match the diagnostic R/Z grid")
        if not np.all(np.isfinite(psi_arr)):
            raise ValueError("psi must be finite")

        interp = RegularGridInterpolator((self.R, self.Z), psi_arr)

        flux_vals = []
        for r, z in self.diagnostics.flux_loops:
            flux_vals.append(float(interp([r, z])[0]))

        b_vals = []
        # B_R = -1/(2*pi*R) * dpsi/dZ
        # B_Z =  1/(2*pi*R) * dpsi/dR
        dpsi_dR = np.gradient(psi_arr, self.R, axis=0, edge_order=2)
        dpsi_dZ = np.gradient(psi_arr, self.Z, axis=1, edge_order=2)

        interp_dR = RegularGridInterpolator((self.R, self.Z), dpsi_dR)
        interp_dZ = RegularGridInterpolator((self.R, self.Z), dpsi_dZ)

        for r, z, drct in self.diagnostics.b_probes:
            if drct == "R":
                val = -1.0 / (2.0 * np.pi * r) * interp_dZ([r, z])[0]
            else:
                val = 1.0 / (2.0 * np.pi * r) * interp_dR([r, z])[0]
            b_vals.append(float(val))

        d2psi_dR2 = np.gradient(dpsi_dR, self.R, axis=0, edge_order=2)
        d2psi_dZ2 = np.gradient(dpsi_dZ, self.Z, axis=1, edge_order=2)
        delta_star_psi = d2psi_dR2 - dpsi_dR / self.R[:, np.newaxis] + d2psi_dZ2
        j_phi = -delta_star_psi / (MU0 * self.R[:, np.newaxis])
        Ip = float(_trapezoid_integral(_trapezoid_integral(j_phi, self.Z, axis=1), self.R))

        return {
            "flux_loops": np.array(flux_vals),
            "b_probes": np.array(b_vals),
            "Ip": Ip,
            "coil_currents": coil_currents.copy(),
        }

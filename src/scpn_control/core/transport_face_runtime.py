# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Signed face-flux evolution for multi-ion profiles.

"""Signed face-flux evolution for multi-ion profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scpn_control.core.transport_flux import (
    TransportFaceFlux,
    TransportFaceGeometry,
    TransportFluxBalance,
    advance_face_flux,
)

if TYPE_CHECKING:
    from scpn_control.core.integrated_transport_solver import TransportSolver


def evolve_fluxes_impl(
    self: TransportSolver, dt: float, flux: TransportFaceFlux, *, geometry: TransportFaceGeometry | None = None
) -> TransportFluxBalance:
    """Advance multi-ion state with explicit signed physical face transport.

    Parameters
    ----------
    dt : float
        Finite nonnegative interval [s].
    flux : TransportFaceFlux
        Electron,D,T,He particle and electron/total-ion energy flux at
        nr-1 physical minor-radius faces. Electron flux must be ambipolar.
        The caller supplies species mapping and the total energy moment;
        no automatic TGLF sampling or diffusion/pinch inference is made.
    geometry : TransportFaceGeometry or None
        Explicit cell volumes and face dV/dr for this stage. None selects
        the existing circular-torus weights. Other transport operators are
        not automatically converted to the supplied geometry.

    Returns
    -------
    TransportFluxBalance
        Measured particle and thermal inventories plus reservoir exchange.

    Raises
    ------
    ValueError
        Not in multi-ion mode, invalid input or a nonphysical candidate.
        Profiles are unchanged on rejection; last_flux_balance is None.

    Notes
    -----
    Defaults to the cylindrical volumes of the thermal/species solver. The
    first and last faces exchange with core/edge reservoirs; the edge node
    is held fixed and the zero-volume axis copies adjacent ion profiles.
    This first-order frozen-flux stage supplies no diffusion, reactions,
    radiation, pumping or exchange. Compose those stages explicitly and
    avoid counting particle-associated energy or turbulent transport twice.
    """
    self._last_flux_balance = None
    if not self.multi_ion or self.n_D is None or self.n_T is None or self.n_He is None:
        raise ValueError("Signed face transport requires multi-ion species state")
    dims = self.cfg["dimensions"]
    result = advance_face_flux(
        rho=self.rho,
        major_radius_m=(dims["R_min"] + dims["R_max"]) / 2,
        minor_radius_m=self.a,
        density=np.stack((self.ne, self.n_D, self.n_T, self.n_He)),
        temperature=np.stack((self.Te, self.Ti)),
        impurity_density=self.n_impurity,
        flux=flux,
        dt=dt,
        geometry=geometry,
    )
    ne, n_D, n_T, n_He = result.density
    charge2 = n_D + n_T + 4 * n_He + 100 * self.n_impurity
    zeff = np.divide(charge2, ne, out=np.ones_like(ne), where=ne > 0)
    self.ne, self.n_D, self.n_T, self.n_He = ne, n_D, n_T, n_He
    self.Te, self.Ti = result.temperature
    self._Z_eff = float(np.clip(np.mean(zeff), 1, 10))
    self._last_flux_balance = result.balance
    return result.balance

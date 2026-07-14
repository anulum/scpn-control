# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Plasma Power Source/Sink Terms

"""Fusion-reactivity and radiation power kernels for the transport source terms.

Pure, stateless microphysics coefficients used by the integrated transport
solver's energy balance: D-T fusion reactivity (heating source) plus tungsten
line radiation and bremsstrahlung (radiated sinks). Each is a published
closed-form fit, kept here as a single responsibility so the transport solver
holds only the aggregation and evolution logic.
"""

from __future__ import annotations

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

__all__ = [
    "bosch_hale_dt_reactivity",
    "bremsstrahlung_power_density",
    "tungsten_radiation_rate",
]


def bosch_hale_dt_reactivity(T_keV: AnyFloatArray) -> FloatArray:
    """Compute the D-T fusion reactivity <sigma*v> [m^3/s].

    Uses the NRL Plasma Formulary fit (Bosch & Hale 1992), valid for
    0.2 < T < 100 keV; the temperature is floored at 0.2 keV.
    """
    T = np.maximum(T_keV, 0.2)
    return np.asarray(3.68e-18 / (T ** (2.0 / 3.0)) * np.exp(-19.94 / (T ** (1.0 / 3.0))))


def tungsten_radiation_rate(Te_keV: AnyFloatArray) -> FloatArray:
    r"""Compute the coronal-equilibrium radiation rate coefficient L_z(Te) for tungsten [W·m^3].

    Piecewise power-law fit to Pütterich et al. (2010) / ADAS data:

    - Te < 1 keV: L_z ~ 5e-31 * Te^0.5   (line radiation dominant)
    - 1 <= Te < 5: L_z ~ 5e-31             (plateau, shell opening)
    - 5 <= Te < 20: L_z ~ 2e-31 * Te^0.3  (rising continuum)
    - Te >= 20:     L_z ~ 8e-31            (fully ionised Bremsstrahlung)
    """
    Te = np.maximum(Te_keV, 0.01)
    Lz = np.where(
        Te < 1.0,
        5.0e-31 * np.sqrt(Te),
        np.where(
            Te < 5.0,
            5.0e-31 * np.ones_like(Te),
            np.where(
                Te < 20.0,
                2.0e-31 * Te**0.3,
                8.0e-31 * np.ones_like(Te),
            ),
        ),
    )
    return np.asarray(Lz)


def bremsstrahlung_power_density(ne_1e19: AnyFloatArray, Te_keV: AnyFloatArray, Z_eff: float) -> FloatArray:
    """Compute the bremsstrahlung power density [W/m^3].

    ``P_brem = 5.35e-37 * Z_eff * ne^2 * sqrt(Te)`` with ``ne`` in m^-3 and
    ``Te`` in keV (NRL Plasma Formulary 2019, p. 58).
    """
    ne_m3 = ne_1e19 * 1e19
    Te = np.maximum(Te_keV, 0.01)
    return np.asarray(5.35e-37 * Z_eff * ne_m3**2 * np.sqrt(Te))

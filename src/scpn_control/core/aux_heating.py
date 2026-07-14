# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Auxiliary Heating Source Deposition

"""Auxiliary-heating source deposition for the integrated transport solver.

Stateless helper extracted from the integrated transport solver: it turns a
requested auxiliary-heating power [MW] into per-cell ion and electron temperature
source terms [keV/s], power-normalised against the radial cell volumes so the
reconstructed injected power matches the request by construction. The radial
grid, density, and cell-volume element are passed explicitly.
"""

from __future__ import annotations

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

__all__ = ["aux_heating_source_profiles"]


def aux_heating_source_profiles(
    P_aux_MW: float,
    rho: AnyFloatArray,
    ne: AnyFloatArray,
    dV: AnyFloatArray,
    *,
    profile_width: float,
    electron_fraction: float,
) -> tuple[FloatArray, FloatArray, dict[str, float]]:
    """Deposit an auxiliary-heating power into ion/electron temperature sources.

    The deposition profile is a Gaussian ``exp(-rho^2 / profile_width)`` normalised
    over the plasma volume so the reconstructed power equals *P_aux_MW*. From the
    per-cell power density the temperature source is ``dT/dt = (2/3) P / (n e_keV)``.
    A non-positive or non-finite power, or a degenerate volume normalisation, yields
    zero sources (fail-soft) with a zeroed balance record.

    Parameters
    ----------
    P_aux_MW : float
        Requested total auxiliary heating power [MW].
    rho : array
        Normalised radius [0, 1].
    ne : array
        Electron density [10^19 m^-3].
    dV : array
        Toroidal volume element per radial cell [m^3].
    profile_width : float
        Gaussian deposition width parameter (floored at 1e-6).
    electron_fraction : float
        Fraction of the power deposited on electrons (clipped to [0, 1]); the
        remainder heats the ions.

    Returns
    -------
    s_heat_i, s_heat_e : array
        Ion and electron temperature source terms [keV/s].
    balance : dict
        Power-balance telemetry (target vs reconstructed ion/electron/total [MW]).
    """
    rho = np.asarray(rho, dtype=np.float64)
    ne = np.asarray(ne, dtype=np.float64)
    dV = np.asarray(dV, dtype=np.float64)
    nr = len(rho)

    if (not np.isfinite(P_aux_MW)) or P_aux_MW <= 0.0:
        zeros: FloatArray = np.zeros(nr, dtype=np.float64)
        balance = {
            "target_total_MW": float(max(P_aux_MW, 0.0)) if np.isfinite(P_aux_MW) else 0.0,
            "target_ion_MW": 0.0,
            "target_electron_MW": 0.0,
            "reconstructed_ion_MW": 0.0,
            "reconstructed_electron_MW": 0.0,
            "reconstructed_total_MW": 0.0,
        }
        return zeros, zeros, balance

    width = max(profile_width, 1e-6)
    shape = np.exp(-(rho**2) / width)
    norm = float(np.sum(shape * dV))
    if (not np.isfinite(norm)) or norm <= 0.0:
        shape = np.ones_like(rho)
        norm = float(np.sum(shape * dV))
    if (not np.isfinite(norm)) or norm <= 0.0:
        zeros = np.zeros(nr, dtype=np.float64)
        balance = {
            "target_total_MW": float(P_aux_MW),
            "target_ion_MW": 0.0,
            "target_electron_MW": 0.0,
            "reconstructed_ion_MW": 0.0,
            "reconstructed_electron_MW": 0.0,
            "reconstructed_total_MW": 0.0,
        }
        return zeros, zeros, balance

    e_keV_J = 1.602176634e-16
    ne_safe = np.maximum(ne, 0.1) * 1e19  # m^-3

    electron_frac = float(np.clip(electron_fraction, 0.0, 1.0))
    ion_frac = 1.0 - electron_frac
    p_aux_w = float(P_aux_MW) * 1e6

    p_i_wm3 = ion_frac * p_aux_w * shape / norm
    p_e_wm3 = electron_frac * p_aux_w * shape / norm

    # (3/2) n dT/dt = P  => dT/dt = (2/3) * P / (n e_keV)
    s_heat_i: FloatArray = (2.0 / 3.0) * p_i_wm3 / (ne_safe * e_keV_J)
    s_heat_e: FloatArray = (2.0 / 3.0) * p_e_wm3 / (ne_safe * e_keV_J)

    rec_i_w = 1.5 * np.sum(ne_safe * s_heat_i * e_keV_J * dV)
    rec_e_w = 1.5 * np.sum(ne_safe * s_heat_e * e_keV_J * dV)
    balance = {
        "target_total_MW": float(P_aux_MW),
        "target_ion_MW": ion_frac * float(P_aux_MW),
        "target_electron_MW": electron_frac * float(P_aux_MW),
        "reconstructed_ion_MW": float(rec_i_w / 1e6),
        "reconstructed_electron_MW": float(rec_e_w / 1e6),
        "reconstructed_total_MW": float((rec_i_w + rec_e_w) / 1e6),
    }

    return s_heat_i, s_heat_e, balance

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Toroidal Green's function, vacuum field, mutual inductance

"""Pure toroidal Green's function and vacuum / mutual-inductance helpers.

This leaf owns elliptic-integral Green's functions, vacuum poloidal flux from
coil sets, external flux assembly, and mutual-inductance matrix construction
used by free-boundary coil optimisation. The CONTROL
:class:`~scpn_control.core.fusion_kernel.FusionKernel` product surface remains
first-class under dual-home C and keeps thin wrappers that call these helpers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import ellipe, ellipk

from scpn_control._typing import FloatArray

_MU0_SI = 4.0e-7 * np.pi


def green_function(R_src: float, Z_src: float, R_obs: float, Z_obs: float) -> float:
    """Toroidal Green's function using elliptic integrals (scalar observers)."""
    mu0 = _MU0_SI
    denom = (R_obs + R_src) ** 2 + (Z_obs - Z_src) ** 2
    if denom < 1e-30:
        return 0.0
    k2 = 4.0 * R_obs * R_src / denom
    k2 = float(np.clip(k2, 1e-9, 0.999999))
    k_val = ellipk(k2)
    e_val = ellipe(k2)
    prefactor = mu0 / (2.0 * np.pi) * np.sqrt(R_obs * R_src)
    psi = prefactor * ((2.0 - k2) * k_val - 2.0 * e_val) / k2
    return float(psi)


def green_function_array(
    R_src: float,
    Z_src: float,
    R_obs: FloatArray,
    Z_obs: FloatArray,
) -> FloatArray:
    """Vectorised toroidal Green's function over observation grids."""
    mu0 = _MU0_SI
    r_arr = np.asarray(R_obs, dtype=np.float64)
    z_arr = np.asarray(Z_obs, dtype=np.float64)
    denom = (r_arr + float(R_src)) ** 2 + (z_arr - float(Z_src)) ** 2
    active = denom >= 1e-30
    denom_safe = np.where(active, denom, 1.0)
    k2 = 4.0 * r_arr * float(R_src) / denom_safe
    k2 = np.clip(k2, 1e-9, 0.999999)
    k_val = ellipk(k2)
    e_val = ellipe(k2)
    prefactor = mu0 / (2.0 * np.pi) * np.sqrt(np.maximum(r_arr * float(R_src), 0.0))
    psi = prefactor * ((2.0 - k2) * k_val - 2.0 * e_val) / k2
    return np.asarray(np.where(active, psi, 0.0), dtype=np.float64)


def vacuum_poloidal_flux(
    RR: FloatArray,
    ZZ: FloatArray,
    coils: Any,
    mu0: float,
) -> FloatArray:
    """Compute vacuum poloidal flux on a mesh from external coil dictionaries.

    Parameters
    ----------
    RR, ZZ :
        Meshgrid coordinate arrays with shape ``(NZ, NR)``.
    coils :
        Iterable of coil dicts with keys ``r``, ``z``, ``current``, optional ``turns``.
    mu0 :
        Vacuum permeability in the same unit system as the coil currents.
    """
    psi_vac = np.zeros_like(np.asarray(RR, dtype=np.float64), dtype=np.float64)
    rr = np.asarray(RR, dtype=np.float64)
    zz = np.asarray(ZZ, dtype=np.float64)
    mu0_value = float(mu0)
    for coil in coils:
        rc, zc = float(coil["r"]), float(coil["z"])
        i_coil = float(coil["current"]) * float(coil.get("turns", 1))
        dz = zz - zc
        r_plus_rc_sq = (rr + rc) ** 2
        k2 = (4.0 * rr * rc) / (r_plus_rc_sq + dz**2)
        k2 = np.clip(k2, 1e-9, 0.999999)
        k_val = ellipk(k2)
        e_val = ellipe(k2)
        prefactor = (mu0_value * i_coil) / (2.0 * np.pi)
        sqrt_term = np.sqrt(r_plus_rc_sq + dz**2)
        term = ((2.0 - k2) * k_val - 2.0 * e_val) / k2
        psi_vac += prefactor * sqrt_term * term
    return psi_vac


def external_flux_from_coilset(
    R: FloatArray,
    Z: FloatArray,
    coils: Any,
) -> FloatArray:
    """Sum Green's function contributions from a CoilSet-like object on a mesh."""
    r_axis = np.asarray(R, dtype=np.float64)
    z_axis = np.asarray(Z, dtype=np.float64)
    nr, nz = len(r_axis), len(z_axis)
    psi_ext = np.zeros((nz, nr), dtype=np.float64)
    r_obs, z_obs = np.meshgrid(r_axis, z_axis)
    positions = coils.positions
    currents = coils.currents
    turns_list = getattr(coils, "turns", ())
    for idx, (pos, current) in enumerate(zip(positions, currents)):
        r_c, z_c = pos
        turns = turns_list[idx] if idx < len(turns_list) else 1
        i_eff = float(current) * float(turns)
        psi_ext += i_eff * green_function_array(float(r_c), float(z_c), r_obs, z_obs)
    return psi_ext


def build_mutual_inductance_matrix(
    coils: Any,
    obs_points: FloatArray,
) -> FloatArray:
    """Build mutual-inductance matrix ``M[k, p]`` for coil optimisation.

    ``M[k, p]`` is the flux at observation point *p* due to unit current
    in coil *k*, scaled by coil turns. Uses the toroidal Green's function.
    """
    positions = coils.positions
    turns_list = getattr(coils, "turns", ())
    n_coils = len(positions)
    points = np.asarray(obs_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("obs_points must have shape (n_pts, 2)")
    n_pts = points.shape[0]
    mutual = np.zeros((n_coils, n_pts), dtype=np.float64)
    for k, (rc, zc) in enumerate(positions):
        turns = turns_list[k] if k < len(turns_list) else 1
        for p in range(n_pts):
            r_obs, z_obs = points[p]
            mutual[k, p] = float(turns) * green_function(float(rc), float(zc), float(r_obs), float(z_obs))
    return mutual


# Historical private names used by FusionKernel wrappers / re-exports.
_green_function = green_function
_green_function_array = green_function_array
_build_mutual_inductance_matrix = build_mutual_inductance_matrix

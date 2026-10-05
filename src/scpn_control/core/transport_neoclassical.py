# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neoclassical transport formulas

"""Legacy bounded neoclassical and bootstrap profiles for the control facade."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

_logger = logging.getLogger(__name__)


# ── Gyro-Bohm coefficient loader ─────────────────────────────────────

_GYRO_BOHM_COEFF_PATH = (
    Path(__file__).resolve().parents[3] / "validation" / "reference_data" / "itpa" / "gyro_bohm_coefficients.json"
)

_GYRO_BOHM_DEFAULT = 0.1  # ITPA Transport DB, Nucl. Fusion 39, 2175 (1999)


def _finite_scalar(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    """Convert to a finite float and enforce the requested positive or nonnegative domain."""
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _normalised_radius(rho: AnyFloatArray) -> FloatArray:
    """Validate a finite, strictly increasing axis-to-edge grid with endpoints zero and one."""
    arr = np.asarray(rho, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("rho must be a one-dimensional profile with at least two points")
    if not np.all(np.isfinite(arr)):
        raise ValueError("rho must contain only finite values")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("rho must stay within the normalised interval [0, 1]")
    if not np.isclose(arr[0], 0.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("rho must start at 0 for axis-to-edge transport profiles")
    if not np.isclose(arr[-1], 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("rho must end at 1 for axis-to-edge transport profiles")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError("rho must be strictly increasing")
    return arr


def _profile_array(
    name: str,
    values: AnyFloatArray,
    shape: tuple[int, ...],
    *,
    positive: bool = False,
    nonnegative: bool = False,
    allow_last_zero: bool = False,
) -> FloatArray:
    """Validate profile shape and finiteness, with optional positivity and a permitted zero edge."""
    arr = np.asarray(values, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{name} must match the rho grid shape")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and allow_last_zero and (np.any(arr[:-1] <= 0.0) or arr[-1] < 0.0):
        raise ValueError(f"{name} must be positive in the interior and non-negative at the boundary")
    if positive and not allow_last_zero and np.any(arr <= 0.0):
        raise ValueError(f"{name} must be positive everywhere")
    if nonnegative and np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative everywhere")
    return arr


def _validate_tokamak_geometry(R0: float, a: float, B0: float) -> tuple[float, float, float]:
    """Require positive radii and field, with minor radius smaller than major radius."""
    R0 = _finite_scalar("R0", R0, positive=True)
    a = _finite_scalar("a", a, positive=True)
    if a >= R0:
        raise ValueError("a must be smaller than R0 for tokamak ordering")
    B0 = _finite_scalar("B0", B0, positive=True)
    return R0, a, B0


def _load_gyro_bohm_coefficient(
    path: Path | str | None = None,
) -> float:
    """Load the calibrated gyro-Bohm coefficient c_gB from JSON.

    Parameters
    ----------
    path : Path or str, optional
        Override path.  Defaults to the file shipped in
        ``validation/reference_data/itpa/gyro_bohm_coefficients.json``.

    Returns
    -------
    float
        The calibrated c_gB value, or 0.1 if the file is not found.
    """
    p = Path(path) if path else _GYRO_BOHM_COEFF_PATH
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        c_gb_payload = data.get("c_gB")
        scaling_payload = data.get("scaling_parameters")
        if c_gb_payload is None and isinstance(scaling_payload, dict):
            c_gb_payload = scaling_payload.get("c_gB_nominal")
        if c_gb_payload is None:
            raise KeyError("c_gB")
        c_gB = _finite_scalar("c_gB", c_gb_payload, positive=True)
        _logger.debug("Loaded c_gB = %.6f from %s", c_gB, p)
        return c_gB
    except (FileNotFoundError, KeyError, json.JSONDecodeError, TypeError) as exc:
        _logger.warning(
            "Could not load c_gB from %s (%s), using default %.4f",
            p,
            exc,
            _GYRO_BOHM_DEFAULT,
        )
        return _GYRO_BOHM_DEFAULT


def chang_hinton_chi_profile(
    rho: AnyFloatArray,
    T_i: AnyFloatArray,
    n_e_19: AnyFloatArray,
    q: AnyFloatArray,
    R0: float,
    a: float,
    B0: float,
    A_ion: float = 2.0,
    Z_eff: float = 1.5,
) -> FloatArray:
    """
    Chang-Hinton (1982) neoclassical ion thermal diffusivity profile [m²/s].

    Parameters
    ----------
    rho : array  — normalised radius [0,1]
    T_i : array  — ion temperature [keV]
    n_e_19 : array  — electron density [10^19 m^-3]
    q : array  — safety factor profile
    R0 : float  — major radius [m]
    a : float  — minor radius [m]
    B0 : float  — toroidal field [T]
    A_ion : float  — ion mass number (default 2 = deuterium)
    Z_eff : float  — effective charge

    Returns
    -------
    chi_nc : array  — neoclassical chi_i [m²/s]
    """
    rho = _normalised_radius(rho)
    shape = rho.shape
    T_i = _profile_array("T_i", T_i, shape, positive=True, allow_last_zero=True)
    n_e_19 = _profile_array("n_e_19", n_e_19, shape, nonnegative=True)
    q = _profile_array("q", q, shape, positive=True)
    R0, a, B0 = _validate_tokamak_geometry(R0, a, B0)
    A_ion = _finite_scalar("A_ion", A_ion, positive=True)
    Z_eff = _finite_scalar("Z_eff", Z_eff, positive=True)

    # Fundamental constants (CODATA 2018)
    e_charge = 1.602176634e-19  # C
    eps0 = 8.8541878128e-12  # F/m
    m_p = 1.67262192369e-27  # kg
    m_i = A_ion * m_p

    chi_nc = np.zeros_like(rho)
    for i in range(len(rho)):
        r = rho[i]
        if r <= 0.0 or T_i[i] <= 0.0 or n_e_19[i] <= 0.0:
            chi_nc[i] = 0.01
            continue

        epsilon = r * a / R0
        if epsilon < 1e-6:
            chi_nc[i] = 0.01
            continue

        T_J = T_i[i] * 1.602176634e-16  # keV -> J
        v_ti = np.sqrt(2.0 * T_J / m_i)
        rho_i = m_i * v_ti / (e_charge * B0)

        # ion-ion collision frequency
        n_e = n_e_19[i] * 1e19
        ln_lambda = 17.0  # Wesson, "Tokamaks" 4th ed., Ch. 14.5
        nu_ii = n_e * Z_eff**2 * e_charge**4 * ln_lambda / (12.0 * np.pi**1.5 * eps0**2 * m_i**0.5 * T_J**1.5)

        eps32 = epsilon**1.5
        nu_star = nu_ii * q[i] * R0 / (eps32 * v_ti)

        alpha_sh = epsilon
        # Chang & Hinton, Phys. Fluids 25, 1493 (1982), Eq. 10
        chi_val = (
            0.66
            * (1.0 + 1.54 * alpha_sh)
            * q[i] ** 2
            * rho_i**2
            * nu_ii
            / (eps32 * (1.0 + 0.74 * nu_star ** (2.0 / 3.0)))
        )

        chi_nc[i] = max(chi_val, 0.01) if np.isfinite(chi_val) else 0.01

    return chi_nc


def calculate_sauter_bootstrap_current_full(
    rho: AnyFloatArray,
    Te: AnyFloatArray,
    Ti: AnyFloatArray,
    ne: AnyFloatArray,
    q: AnyFloatArray,
    R0: float,
    a: float,
    B0: float,
    Z_eff: float = 1.5,
) -> FloatArray:
    """Full Sauter bootstrap current model (Sauter et al., Phys. Plasmas 6, 1999).

    Parameters
    ----------
    rho : array — normalised radius [0,1]
    Te : array — electron temperature [keV]
    Ti : array — ion temperature [keV]
    ne : array — electron density [10^19 m^-3]
    q : array — safety factor profile
    R0 : float — major radius [m]
    a : float — minor radius [m]
    B0 : float — toroidal field [T]
    Z_eff : float — effective charge

    Returns
    -------
    j_bs : array — bootstrap current density [A/m^2]
    """
    rho = _normalised_radius(rho)
    shape = rho.shape
    Te = _profile_array("Te", Te, shape, positive=True, allow_last_zero=True)
    Ti = _profile_array("Ti", Ti, shape, positive=True, allow_last_zero=True)
    ne = _profile_array("ne", ne, shape, nonnegative=True)
    q = _profile_array("q", q, shape, positive=True)
    R0, a, B0 = _validate_tokamak_geometry(R0, a, B0)
    Z_eff = _finite_scalar("Z_eff", Z_eff, positive=True)
    n = len(rho)
    j_bs = np.zeros(n)
    # Fundamental constants (CODATA 2018)
    e_charge = 1.602176634e-19  # C
    m_e = 9.1093837015e-31  # kg
    eps0 = 8.8541878128e-12  # F/m

    for i in range(1, n - 1):
        eps = rho[i] * a / R0
        if eps < 1e-6:
            continue
        if Te[i] <= 0.0 or Ti[i] <= 0.0 or ne[i] <= 0.0:
            continue

        # Sauter et al., Phys. Plasmas 6, 2834 (1999), Eq. 13
        f_t = 1.0 - (1.0 - eps) ** 2 / (np.sqrt(1.0 - eps**2) * (1.0 + 1.46 * np.sqrt(eps)))
        f_t = max(0.0, min(f_t, 1.0))

        # Electron thermal velocity
        T_e_J = Te[i] * 1e3 * e_charge
        v_te = np.sqrt(2.0 * T_e_J / m_e)

        # Collision frequency
        n_e = ne[i] * 1e19
        ln_lambda = 17.0  # Wesson, "Tokamaks" 4th ed., Ch. 14.5
        nu_ei = n_e * Z_eff * e_charge**4 * ln_lambda / (12.0 * np.pi**1.5 * eps0**2 * m_e**0.5 * T_e_J**1.5)

        # Collisionality
        nu_star_e = nu_ei * q[i] * R0 / (eps**1.5 * v_te) if v_te > 0 else 1e6

        # Sauter et al., Phys. Plasmas 6, 2834 (1999), Eqs. 14a-14c
        alpha_31 = 1.0 / (1.0 + 0.36 / Z_eff)
        L31 = f_t * alpha_31 / (1.0 + alpha_31 * np.sqrt(nu_star_e) + 0.25 * nu_star_e * (1.0 - f_t) ** 2)

        # Sauter L32 coefficient
        L32 = f_t * (0.05 + 0.62 * Z_eff) / (Z_eff * (1.0 + 0.44 * Z_eff))
        L32 /= 1.0 + 0.22 * np.sqrt(nu_star_e) + 0.19 * nu_star_e * (1.0 - f_t)

        # Sauter L34 coefficient (ion contribution)
        L34 = L31 * Ti[i] / Te[i]

        # Gradients (central differences)
        dr = (rho[i + 1] - rho[i - 1]) * a
        if abs(dr) < 1e-12:
            continue
        dn_dr = (ne[i + 1] - ne[i - 1]) * 1e19 / dr
        dTe_dr = (Te[i + 1] - Te[i - 1]) * 1e3 * e_charge / dr
        dTi_dr = (Ti[i + 1] - Ti[i - 1]) * 1e3 * e_charge / dr

        # Poloidal field
        B_pol = B0 * eps / q[i]
        if B_pol < 1e-10:
            continue

        # Bootstrap current
        p_e = n_e * T_e_J
        j_bs[i] = -(p_e / B_pol) * (L31 * dn_dr / n_e + L32 * dTe_dr / T_e_J + L34 * dTi_dr / (Ti[i] * 1e3 * e_charge))

    return j_bs

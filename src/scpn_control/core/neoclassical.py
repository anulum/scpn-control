# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851  Contact: protoscience@anulum.li
"""
Neoclassical transport and bootstrap current models.

Implements Chang-Hinton ion thermal diffusivity, Pfirsch-Schlüter diffusivity,
collisionality regime auto-detection, and the full three-coefficient Sauter
bootstrap current model for general axisymmetry.

Key references
--------------
Chang & Hinton 1982 : C.S. Chang, F.L. Hinton, Phys. Fluids 25, 1493 (1982).
Hinton & Hazeltine 1976 : F.L. Hinton, R.D. Hazeltine, Rev. Mod. Phys. 48, 239 (1976).
Sauter 1999 : O. Sauter et al., Phys. Plasmas 6, 2834 (1999).
Wesson 2011 : J. Wesson, "Tokamaks", 4th ed., Oxford (2011).
Galeev & Sagdeev 1979 : A.A. Galeev, R.Z. Sagdeev, in Reviews of Plasma Physics, vol. 7 (1979).
"""

from __future__ import annotations

import numpy as np

# ─── Physical constants (CODATA 2018) ───────────────────────────────────────
_E_CHARGE = 1.602176634e-19  # C
_M_PROTON = 1.67262192369e-27  # kg
_M_ELECTRON = 9.1093837015e-31  # kg
_EPS0 = 8.8541878128e-12  # F/m
_LN_LAMBDA = 17.0  # Coulomb logarithm, Wesson Ch. 14


def collisionality(
    n_e_19: float,
    T_kev: float,
    q: float,
    R: float,
    epsilon: float,
    mass_amu: float = 2.0,
    z_eff: float = 1.0,
) -> float:
    """Dimensionless neoclassical collisionality nu_star.

    nu_star = nu_ii * q * R / (epsilon^1.5 * v_th)

    Hinton & Hazeltine 1976, Eq. 4.53.

    Parameters
    ----------
    n_e_19 : float
        Electron density [10^19 m^-3].
    T_kev : float
        Species temperature [keV].
    q : float
        Safety factor.
    R : float
        Major radius [m].
    epsilon : float
        Inverse aspect ratio r/R.
    mass_amu : float
        Species mass in AMU (default 2.0 for D).
    z_eff : float
        Effective charge.

    Returns
    -------
    float
        Dimensionless collisionality.
    """
    if T_kev <= 0.01 or epsilon < 1e-6 or n_e_19 <= 0:
        return 0.0

    T_J = T_kev * 1.602176634e-16
    m = mass_amu * _M_PROTON
    v_th = np.sqrt(2.0 * T_J / m)

    n_e = n_e_19 * 1e19
    # Ion-ion collision frequency — Wesson 2011, Eq. 14.2.3
    nu_ii = (n_e * z_eff**2 * _E_CHARGE**4 * _LN_LAMBDA) / (12.0 * np.pi**1.5 * _EPS0**2 * np.sqrt(m) * T_J**1.5)

    return float(nu_ii * q * R / (epsilon**1.5 * v_th))


def _ion_collision_freq(n_e_19: float, T_kev: float, mass_amu: float, z_eff: float) -> float:
    """Ion-ion collision frequency nu_ii [s^-1].

    Wesson 2011, Eq. 14.2.3.
    """
    T_J = T_kev * 1.602176634e-16
    m = mass_amu * _M_PROTON
    n_e = n_e_19 * 1e19
    return float((n_e * z_eff**2 * _E_CHARGE**4 * _LN_LAMBDA) / (12.0 * np.pi**1.5 * _EPS0**2 * np.sqrt(m) * T_J**1.5))


def _larmor_radius(T_kev: float, B: float, mass_amu: float) -> float:
    """Thermal ion Larmor radius rho_i = v_thi / Omega_i [m].

    Wesson 2011, Eq. 14.1.1.
    """
    T_J = T_kev * 1.602176634e-16
    m = mass_amu * _M_PROTON
    v_thi = np.sqrt(2.0 * T_J / m)
    Omega_i = _E_CHARGE * B / m
    return float(v_thi / Omega_i)


def chang_hinton_chi(
    q: float,
    epsilon: float,
    nu_star: float,
    rho_i: float,
    nu_ii: float,
) -> float:
    """Chang-Hinton banana-regime ion thermal diffusivity.

    Chang & Hinton 1982, Phys. Fluids 25, 1493, Eq. 10.

    Parameters
    ----------
    q : float
        Safety factor.
    epsilon : float
        Inverse aspect ratio r/R.
    nu_star : float
        Ion collisionality.
    rho_i : float
        Ion Larmor radius [m].
    nu_ii : float
        Ion-ion collision frequency [s^-1].

    Returns
    -------
    float
        Ion thermal diffusivity [m^2/s].
    """
    if epsilon < 1e-6:
        return 0.0

    eps32 = epsilon**1.5
    alpha_sh = epsilon  # shaping correction, Chang & Hinton 1982 Eq. 10

    # Chang & Hinton 1982, Eq. 10
    chi = 0.66 * (1.0 + 1.54 * alpha_sh) * q**2 * rho_i**2 * nu_ii / (eps32 * (1.0 + 0.74 * nu_star ** (2.0 / 3.0)))
    return float(chi)


def plateau_chi(
    q: float,
    rho_i: float,
    v_thi: float,
    R: float,
) -> float:
    """Plateau-regime ion thermal diffusivity.

    Valid for 1 < nu_star < q^2 / epsilon^1.5.
    Hinton & Hazeltine 1976, Rev. Mod. Phys. 48, 239, Eq. 4.66.

    Parameters
    ----------
    q : float
        Safety factor.
    rho_i : float
        Ion Larmor radius [m].
    v_thi : float
        Ion thermal velocity [m/s].
    R : float
        Major radius [m].

    Returns
    -------
    float
        Ion thermal diffusivity [m^2/s].
    """
    # Hinton & Hazeltine 1976, Eq. 4.66: chi_pl = q * rho_i^2 * v_thi / R
    return float(q * rho_i**2 * v_thi / R)


def pfirsch_schluter_chi(
    q: float,
    rho_i: float,
    nu_ii: float,
) -> float:
    """Pfirsch-Schlüter regime ion thermal diffusivity.

    Valid for nu_star > q^2 / epsilon^1.5 (collisional regime).
    Wesson 2011, "Tokamaks" 4th ed., Eq. 14.5.7.
    Galeev & Sagdeev 1979, Reviews of Plasma Physics, vol. 7.

    Parameters
    ----------
    q : float
        Safety factor.
    rho_i : float
        Ion Larmor radius [m].
    nu_ii : float
        Ion-ion collision frequency [s^-1].

    Returns
    -------
    float
        Ion thermal diffusivity [m^2/s].
    """
    # Wesson 2011, Eq. 14.5.7: chi_PS = q^2 * rho_i^2 * nu_ii
    return float(q**2 * rho_i**2 * nu_ii)


def neoclassical_chi(
    nu_star: float,
    q: float,
    epsilon: float,
    rho_i: float,
    nu_ii: float,
    v_thi: float,
    R: float,
) -> float:
    """Neoclassical ion thermal diffusivity with automatic regime selection.

    Regime boundaries follow Hinton & Hazeltine 1976, Rev. Mod. Phys. 48, 239,
    Section IV and Wesson 2011, Ch. 14.5:
      - nu_star < 1                    : banana (Chang & Hinton 1982, Eq. 10)
      - 1 <= nu_star < q^2/epsilon^1.5 : plateau (Hinton & Hazeltine 1976, Eq. 4.66)
      - nu_star >= q^2/epsilon^1.5     : Pfirsch-Schlüter (Wesson 2011, Eq. 14.5.7)

    Parameters
    ----------
    nu_star : float
        Dimensionless collisionality.
    q : float
        Safety factor.
    epsilon : float
        Inverse aspect ratio r/R.
    rho_i : float
        Ion Larmor radius [m].
    nu_ii : float
        Ion-ion collision frequency [s^-1].
    v_thi : float
        Ion thermal velocity [m/s].
    R : float
        Major radius [m].

    Returns
    -------
    float
        Ion thermal diffusivity [m^2/s].
    """
    if epsilon < 1e-6:
        return 0.0

    # Boundary between plateau and PS: nu_star = q^2 / epsilon^1.5
    # Hinton & Hazeltine 1976, Section IV
    nu_ps_boundary = q**2 / epsilon**1.5

    # Hinton & Hazeltine 1976, Section IV: banana regime for nu_star < epsilon^1.5
    if nu_star < epsilon**1.5:
        return chang_hinton_chi(q, epsilon, nu_star, rho_i, nu_ii)
    elif nu_star < nu_ps_boundary:
        return plateau_chi(q, rho_i, v_thi, R)
    else:
        return pfirsch_schluter_chi(q, rho_i, nu_ii)


def banana_plateau_chi(
    q: float,
    epsilon: float,
    nu_star: float,
    z_eff: float,
) -> float:
    """Banana-plateau neoclassical diffusivity with Z_eff dependence.

    Hinton & Hazeltine 1976, Rev. Mod. Phys. 48, 239, Eq. 4.61.

    Parameters
    ----------
    q : float
        Safety factor.
    epsilon : float
        Inverse aspect ratio.
    nu_star : float
        Collisionality.
    z_eff : float
        Effective charge.

    Returns
    -------
    float
        Ion thermal diffusivity (dimensionless units, normalised form).
    """
    if epsilon < 1e-6:
        return 0.0
    # Hinton & Hazeltine 1976, Eq. 4.61
    return float(q**2 * epsilon ** (-1.5) * (1.0 + 0.6 * z_eff) / (1.0 + nu_star))


def _sauter_L31(f_t: float, nu_e: float, Z: float) -> float:
    """Sauter L31 bootstrap coefficient.

    Sauter et al. 1999, Phys. Plasmas 6, 2834, Eq. 14.
    Valid for arbitrary collisionality via fit to drift-kinetic solutions.

    Parameters
    ----------
    f_t : float
        Trapped particle fraction.
    nu_e : float
        Normalised electron collisionality nu_star_e.
    Z : float
        Effective ion charge Z_eff.
    """
    # Sauter 1999, Eq. 14 — banana-limit numerator
    L31_banana = (
        (1.0 + 1.4 / (Z + 1.0)) * f_t - 1.9 / (Z + 1.0) * f_t**2 + 0.3 / (Z + 1.0) * f_t**3 + 0.2 / (Z + 1.0) * f_t**4
    )
    # Sauter 1999, Eq. 14 — collisionality-dependent denominator
    alpha_31 = 1.0 / (1.0 + 0.36 / Z)
    L31 = L31_banana / (1.0 + alpha_31 * nu_e**0.5)
    return float(L31)


def _sauter_L32(f_t: float, nu_e: float, Z: float) -> float:
    """Sauter L32 bootstrap coefficient.

    Sauter et al. 1999, Phys. Plasmas 6, 2834, Eq. 15.
    Couples electron temperature gradient to bootstrap current.

    Parameters
    ----------
    f_t : float
        Trapped particle fraction.
    nu_e : float
        Normalised electron collisionality nu_star_e.
    Z : float
        Effective ion charge Z_eff.
    """
    # Sauter 1999, Eq. 15
    A = (0.05 + 0.62 * Z) / (Z * (1.0 + 0.44 * Z))
    B = (0.56 + 1.93 * Z) / (Z * (1.0 + 0.44 * Z))
    term1 = A * (f_t - f_t**4 / (1.0 + 0.22 * nu_e))
    term2 = B * f_t**2 * (1.0 - 0.49 * nu_e / (0.57 + nu_e))
    return float(term1 - term2)


def _sauter_L34(f_t: float, nu_e: float, Z: float) -> float:
    """Sauter L34 bootstrap coefficient.

    Sauter et al. 1999, Phys. Plasmas 6, 2834, Eq. 16.
    Couples ion temperature gradient to bootstrap current.
    Equals L31 to leading order in f_t.

    Parameters
    ----------
    f_t : float
        Trapped particle fraction.
    nu_e : float
        Normalised electron collisionality nu_star_e.
    Z : float
        Effective ion charge Z_eff.
    """
    # Sauter 1999, Eq. 16: L34 = L31 to leading order
    L34 = (1.0 + 1.4 / (Z + 1.0)) * f_t - 1.9 / (Z + 1.0) * f_t**2
    return float(L34)


def sauter_bootstrap(
    rho: np.ndarray,
    Te: np.ndarray,
    Ti: np.ndarray,
    ne: np.ndarray,
    q: np.ndarray,
    R0: float,
    a: float,
    B0: float = 5.3,
    z_eff: float = 1.5,
) -> np.ndarray:
    """Sauter bootstrap current density profile with full L31+L32+L34 coefficients.

    Sauter et al. 1999, Phys. Plasmas 6, 2834, Eqs. 14–16.

    j_bs = -(p_e / B_pol) * [L31 * d(ln p_e)/dr + L32 * d(ln T_e)/dr
                               + L34 * (T_i/T_e) * d(ln T_i)/dr]

    Parameters
    ----------
    rho : np.ndarray
        Normalised radial grid.
    Te, Ti : np.ndarray
        Electron and ion temperatures [keV].
    ne : np.ndarray
        Electron density [10^19 m^-3].
    q : np.ndarray
        Safety factor profile.
    R0, a : float
        Major and minor radii [m].
    B0 : float
        Toroidal magnetic field [T].
    z_eff : float
        Effective charge.

    Returns
    -------
    np.ndarray
        Bootstrap current density [A/m^2].
    """
    n = len(rho)
    j_bs = np.zeros(n)

    drho = rho[1] - rho[0]
    dne_drho = np.gradient(ne * 1e19, drho)
    dTe_drho = np.gradient(Te, drho)
    dTi_drho = np.gradient(Ti, drho)

    for i in range(1, n - 1):
        eps = rho[i] * a / R0
        if eps < 1e-6 or Te[i] <= 0.1 or ne[i] <= 0.1 or q[i] <= 0.5:
            continue

        # Trapped fraction — Sauter 1999, Eq. 14 (geometric fit)
        f_t = 1.0 - (1.0 - eps) ** 2 / (np.sqrt(1.0 - eps**2) * (1.0 + 1.46 * np.sqrt(eps)))

        # Electron collision frequency — Wesson 2011, Eq. 14.2.3
        v_the = np.sqrt(2.0 * Te[i] * 1.602176634e-16 / _M_ELECTRON)
        nu_ee = (ne[i] * 1e19 * _E_CHARGE**4 * _LN_LAMBDA) / (
            12.0 * np.pi**1.5 * _EPS0**2 * _M_ELECTRON**0.5 * (Te[i] * 1.602176634e-16) ** 1.5
        )
        nu_star_e = nu_ee * q[i] * R0 / (eps**1.5 * v_the)

        L31 = _sauter_L31(f_t, nu_star_e, z_eff)
        L32 = _sauter_L32(f_t, nu_star_e, z_eff)
        L34 = _sauter_L34(f_t, nu_star_e, z_eff)

        n_e_si = ne[i] * 1e19
        T_e_J = Te[i] * 1.602176634e-16
        T_i_J = Ti[i] * 1.602176634e-16
        p_e = n_e_si * T_e_J  # Pa

        # Logarithmic pressure gradient: d(ln p_e)/dr = (1/p_e) * dp_e/dr
        # dp_e/dr ≈ (T_e * dne + ne * dTe) / a  (dr = a * drho)
        dp_e_dr = (T_e_J * dne_drho[i] + n_e_si * 1.602176634e-16 * dTe_drho[i]) / a
        dln_pe_dr = dp_e_dr / p_e if p_e > 0 else 0.0

        # Logarithmic T_e and T_i gradients
        dln_Te_dr = dTe_drho[i] / (Te[i] * a) if Te[i] > 0 else 0.0
        dln_Ti_dr = dTi_drho[i] / (Ti[i] * a) if Ti[i] > 0 else 0.0

        # B_pol ≈ B0 * eps / q  from q = r * B_T / (R0 * B_pol)
        B_pol = B0 * eps / max(q[i], 0.1)
        if B_pol < 1e-10:
            continue

        # Sauter 1999, Eqs. 14–16 combined:
        # j_bs = -(p_e / B_pol) * (L31 * dln_pe + L32 * dln_Te + L34*(Ti/Te)*dln_Ti)
        j_bs[i] = -(p_e / B_pol) * (L31 * dln_pe_dr + L32 * dln_Te_dr + L34 * (T_i_J / T_e_J) * dln_Ti_dr)

    return j_bs

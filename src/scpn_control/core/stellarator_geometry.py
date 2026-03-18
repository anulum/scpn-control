# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""
Stellarator flux-surface geometry in Boozer coordinates, neoclassical
transport, and ISS04 confinement scaling.

References
----------
Boozer, A. H., Phys. Fluids 24 (1981) 1999.
    Magnetic coordinate system with straight field lines; ∇ψ × ∇θ_B = B/B².
Grieger, G. et al., Phys. Fluids B 4 (1992) 2081.
    Wendelstein 7-X design parameters and modular coil geometry.
Yamada, H. et al., Nucl. Fusion 45 (2005) 1684.
    ISS04 empirical scaling: τ_E ∝ a^2.28 R^0.64 P^-0.61 n_e19^0.54 B^0.84 ι^0.41.
Nemov, V. V. et al., Phys. Plasmas 6 (1999) 4622.
    Effective ripple ε_eff via field-line tracing (DKES/NEO-2 basis, Eq. 30).
Beidler, C. D. et al., Nucl. Fusion 51 (2011) 076001.
    Quasi-isodynamic neoclassical transport optimization for W7-X.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from scpn_control.core._validators import require_positive_float

# ISS04 prefactor — Yamada et al. 2005, Nucl. Fusion 45, 1684, Eq. 4
ISS04_PREFACTOR: float = 0.134  # dimensionless

# ISS04 reference radius for ι evaluation — Yamada et al. 2005, Eq. 4
ISS04_S_REF: float = (2.0 / 3.0) ** 2  # normalised toroidal flux s = (r/a)²; Yamada 2005 evaluates at r/a = 2/3

# Physical constants used in neoclassical transport
_M_D: float = 3.344e-27  # deuteron mass [kg]
_E: float = 1.602e-19  # elementary charge [C]
_EV_TO_J: float = 1.602e-16  # keV → J
_EPS0: float = 8.854e-12  # vacuum permittivity [F/m]
COULOMB_LOG: float = 17.0  # Wesson, Tokamaks 4th ed., Ch. 14; dimensionless


@dataclass
class StellaratorConfig:
    """Stellarator device and magnetic configuration parameters.

    Parameters
    ----------
    N_fp : int
        Number of toroidal field periods.
        W7-X: 5 — Grieger et al. 1992, Phys. Fluids B 4, 2081.
    R0 : float
        Major radius [m]. W7-X: 5.5 m.
    a : float
        Average minor radius [m]. W7-X: 0.53 m.
    B0 : float
        On-axis magnetic field [T]. W7-X standard: 2.5 T.
    iota_0 : float
        Rotational transform at magnetic axis (s=0).
        Boozer 1981, Phys. Fluids 24, 1999 — straight field-line coordinate.
    iota_a : float
        Rotational transform at plasma edge (s=1).
    mirror_ratio : float
        Helical mirror ratio ε_h = δB / B₀. W7-X: ~0.07.
    helical_excursion : float
        Helical axis excursion amplitude [m].
    """

    N_fp: int = 5
    R0: float = 5.5
    a: float = 0.53
    B0: float = 2.5
    iota_0: float = 0.87
    iota_a: float = 1.0
    mirror_ratio: float = 0.07
    helical_excursion: float = 0.05
    name: str = "custom"


def w7x_config() -> StellaratorConfig:
    """Wendelstein 7-X standard configuration.

    Grieger et al., Phys. Fluids B 4 (1992) 2081 — coil geometry and ι profile.
    Klinger et al., Nucl. Fusion 59 (2019) 112004 — operational parameters.
    """
    return StellaratorConfig(
        N_fp=5,
        R0=5.5,
        a=0.53,
        B0=2.5,
        iota_0=0.87,
        iota_a=1.0,
        mirror_ratio=0.07,
        helical_excursion=0.05,
        name="W7-X",
    )


def iota_profile(config: StellaratorConfig, s: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    """Rotational transform ι(s) via linear interpolation.

    Boozer 1981, Phys. Fluids 24, 1999: ι = dψ_pol/dψ_tor in Boozer coordinates.

    Parameters
    ----------
    config : StellaratorConfig
    s : float or array
        Normalised toroidal flux label, s ∈ [0, 1].
    """
    return config.iota_0 + (config.iota_a - config.iota_0) * np.asarray(s)


def stellarator_flux_surface(
    config: StellaratorConfig,
    s: float,
    n_theta: int = 64,
    n_phi: int = 64,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute a single flux surface in Boozer coordinates.

    Boozer 1981, Phys. Fluids 24, 1999: (θ_B, ζ_B) are 2π-periodic straight
    field-line angles such that B · ∇θ_B / B · ∇ζ_B = ι.

    |B| is represented as B₀(1 − ε_t cos θ − ε_h cos(N_fp ζ − ι θ)).
    This is the standard two-harmonic Boozer spectrum used for stellarator
    neoclassical calculations; Grieger et al. 1992, Phys. Fluids B 4, 2081.

    Parameters
    ----------
    config : StellaratorConfig
    s : float
        Normalised toroidal flux, s ∈ (0, 1].
    n_theta, n_phi : int
        Poloidal and toroidal grid resolution.

    Returns
    -------
    R : ndarray, shape (n_theta, n_phi)
        Major radius [m].
    Z : ndarray, shape (n_theta, n_phi)
        Vertical position [m].
    B : ndarray, shape (n_theta, n_phi)
        Magnetic field magnitude [T].
    """
    s = require_positive_float("s", s)
    r = config.a * np.sqrt(s)
    iota = float(iota_profile(config, s))

    # Boozer angles — 2π-periodic by construction (Boozer 1981)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")

    # Helical axis excursion modulates major-radius position
    # Grieger et al. 1992, Phys. Fluids B 4, 2081 — W7-X geometry
    delta_R = config.helical_excursion * np.cos(config.N_fp * PH)
    R = config.R0 + r * np.cos(TH) + delta_R
    Z = r * np.sin(TH) + config.helical_excursion * np.sin(config.N_fp * PH)

    # Two-harmonic |B| in Boozer coordinates
    # ε_t = r/R₀ (toroidal), ε_h grows as √s from axis to edge
    epsilon_t = r / config.R0
    epsilon_h = config.mirror_ratio * np.sqrt(s)
    B = config.B0 * (1.0 - epsilon_t * np.cos(TH) - epsilon_h * np.cos(config.N_fp * PH - iota * TH))

    return R, Z, B


def effective_ripple(config: StellaratorConfig, s: float) -> float:
    """Effective helical ripple ε_eff for neoclassical transport.

    Nemov et al., Phys. Plasmas 6 (1999) 4622, Eq. 30.
    Analytic proxy: ε_eff ~ ε_h^(3/2) / √N_fp.
    Full evaluation requires field-line tracing (DKES/NEO-2).

    Parameters
    ----------
    config : StellaratorConfig
    s : float
        Normalised toroidal flux, s ∈ (0, 1].

    Returns
    -------
    float
        Effective ripple ε_eff ∈ (0, 1).
    """
    s = require_positive_float("s", s)
    epsilon_h = config.mirror_ratio * np.sqrt(s)
    eps_eff = epsilon_h**1.5 / np.sqrt(config.N_fp)
    return float(np.clip(eps_eff, 0.0, 1.0))


def iss04_scaling(
    config: StellaratorConfig,
    n_e: float,
    P_heat: float,
) -> float:
    """ISS04 energy confinement scaling law for stellarators.

    Yamada et al., Nucl. Fusion 45 (2005) 1684, Eq. 4:
        τ_E = 0.134 · a^2.28 · R^0.64 · P^-0.61 · n_e19^0.54 · B^0.84 · ι_{2/3}^0.41

    Parameters
    ----------
    config : StellaratorConfig
    n_e : float
        Line-averaged electron density [10^19 m^-3].
    P_heat : float
        Absorbed heating power [MW].

    Returns
    -------
    float
        Predicted energy confinement time [s].
    """
    n_e = require_positive_float("n_e", n_e)
    P_heat = require_positive_float("P_heat", P_heat)

    iota_ref = float(iota_profile(config, ISS04_S_REF))

    tau = (
        ISS04_PREFACTOR
        * config.a**2.28
        * config.R0**0.64
        * P_heat**-0.61
        * n_e**0.54
        * config.B0**0.84
        * iota_ref**0.41
    )
    return float(tau)


def stellarator_neoclassical_chi(
    config: StellaratorConfig,
    s: float,
    T_keV: float,
    n_e19: float,
) -> float:
    """Neoclassical thermal diffusivity in the 1/ν regime.

    Beidler et al., Nucl. Fusion 51 (2011) 076001 — quasi-isodynamic transport.
    1/ν regime: χ_neo ~ ε_eff^(3/2) · v_th² / (ν · R · N_fp)

    Collision frequency from Braginskii:
        ν_ii ~ n_e e⁴ ln Λ / (4π ε₀² m_i² v_th³)

    Parameters
    ----------
    config : StellaratorConfig
    s : float
        Normalised toroidal flux, s ∈ (0, 1].
    T_keV : float
        Ion temperature [keV].
    n_e19 : float
        Electron density [10^19 m^-3].

    Returns
    -------
    float
        Neoclassical χ [m²/s].
    """
    s = require_positive_float("s", s)
    T_keV = require_positive_float("T_keV", T_keV)
    n_e19 = require_positive_float("n_e19", n_e19)

    eps_eff = effective_ripple(config, s)
    v_th = np.sqrt(T_keV * _EV_TO_J / _M_D)
    n_e_m3 = n_e19 * 1e19

    nu_ii = n_e_m3 * _E**4 * COULOMB_LOG / (4.0 * np.pi * _EPS0**2 * _M_D**2 * v_th**3)
    chi = eps_eff**1.5 * v_th**2 / (nu_ii * config.R0 * config.N_fp)

    return float(chi)

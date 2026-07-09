# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Stellarator Geometry and ISS04 Scaling
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
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class StellaratorConfigSchema(BaseModel):
    """Pydantic v2 schema for stellarator device and magnetic parameters."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True)

    N_fp: int = Field(default=5, gt=0, description="Positive number of toroidal field periods.")
    R0: float = Field(default=5.5, gt=0.0, description="Major radius in metres.")
    a: float = Field(default=0.53, gt=0.0, description="Average minor radius in metres.")
    B0: float = Field(default=2.5, gt=0.0, description="On-axis magnetic field in tesla.")
    iota_0: float = Field(default=0.87, gt=0.0, description="Rotational transform at magnetic axis.")
    iota_a: float = Field(default=1.0, gt=0.0, description="Rotational transform at plasma edge.")
    mirror_ratio: float = Field(default=0.07, ge=0.0, description="Helical mirror ratio.")
    helical_excursion: float = Field(default=0.05, ge=0.0, description="Helical axis excursion in metres.")
    name: str = Field(default="custom", min_length=1, description="Configuration label.")

    @model_validator(mode="after")
    def validate_positive_major_radius_margin(self) -> StellaratorConfigSchema:
        """Ensure the helical flux surface cannot cross non-positive major radius."""

        if self.a + self.helical_excursion >= self.R0:
            raise ValueError("R0 must exceed a + helical_excursion to keep flux surfaces at positive major radius")
        return self


def _normalised_flux(
    name: str,
    value: float | NDArray[np.float64],
    *,
    include_axis: bool,
) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    lower_bad = arr < 0.0 if include_axis else arr <= 0.0
    if np.any(lower_bad) or np.any(arr > 1.0):
        interval = "[0, 1]" if include_axis else "(0, 1]"
        raise ValueError(f"{name} must stay within the normalised flux interval {interval}")
    return arr


def _positive_grid_size(name: str, value: int) -> int:
    if not isinstance(value, int) or value < 2:
        raise ValueError(f"{name} must be an integer >= 2")
    return value


@dataclass(frozen=True)
class StellaratorConfig:
    """Stellarator device and magnetic configuration parameters.

    Parameters
    ----------
    N_fp : int
        Positive number of toroidal field periods.
        W7-X: 5 — Grieger et al. 1992, Phys. Fluids B 4, 2081.
    R0 : float
        Positive major radius [m]. W7-X: 5.5 m.
    a : float
        Positive average minor radius [m]. W7-X: 0.53 m.
    B0 : float
        Positive on-axis magnetic field [T]. W7-X standard: 2.5 T.
    iota_0 : float
        Positive rotational transform at magnetic axis (s=0).
        Boozer 1981, Phys. Fluids 24, 1999 — straight field-line coordinate.
    iota_a : float
        Positive rotational transform at plasma edge (s=1).
    mirror_ratio : float
        Non-negative helical mirror ratio ε_h = δB / B₀. W7-X: ~0.07.
    helical_excursion : float
        Non-negative helical axis excursion amplitude [m]; R0 must exceed a plus this excursion.
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

    def __post_init__(self) -> None:
        """Validate and normalise configuration values through the Pydantic schema."""

        schema = StellaratorConfigSchema.model_validate(
            {
                "N_fp": self.N_fp,
                "R0": self.R0,
                "a": self.a,
                "B0": self.B0,
                "iota_0": self.iota_0,
                "iota_a": self.iota_a,
                "mirror_ratio": self.mirror_ratio,
                "helical_excursion": self.helical_excursion,
                "name": self.name,
            }
        )
        object.__setattr__(self, "N_fp", schema.N_fp)
        object.__setattr__(self, "R0", schema.R0)
        object.__setattr__(self, "a", schema.a)
        object.__setattr__(self, "B0", schema.B0)
        object.__setattr__(self, "iota_0", schema.iota_0)
        object.__setattr__(self, "iota_a", schema.iota_a)
        object.__setattr__(self, "mirror_ratio", schema.mirror_ratio)
        object.__setattr__(self, "helical_excursion", schema.helical_excursion)
        object.__setattr__(self, "name", schema.name)

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]:
        """Return the Pydantic JSON Schema for stellarator configuration."""

        return StellaratorConfigSchema.model_json_schema()


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

    Raises
    ------
    ValueError
        If the configuration is non-physical or s leaves [0, 1].
    """
    s_arr = _normalised_flux("s", s, include_axis=True)
    iota = config.iota_0 + (config.iota_a - config.iota_0) * s_arr
    if np.ndim(s) == 0:
        return float(iota)
    return iota


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

    Raises
    ------
    ValueError
        If the configuration is non-physical, s leaves (0, 1], or either grid
        resolution is below two points.
    """
    s = float(_normalised_flux("s", s, include_axis=False))
    n_theta = _positive_grid_size("n_theta", n_theta)
    n_phi = _positive_grid_size("n_phi", n_phi)
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

    Raises
    ------
    ValueError
        If the configuration is non-physical or s leaves (0, 1].
    """
    s = float(_normalised_flux("s", s, include_axis=False))
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

    Raises
    ------
    ValueError
        If the configuration, density, or heating power is outside the positive
        finite ISS04 domain.
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

    Raises
    ------
    ValueError
        If the configuration is non-physical, s leaves (0, 1], or plasma inputs
        are not positive finite values.
    """
    s = float(_normalised_flux("s", s, include_axis=False))
    T_keV = require_positive_float("T_keV", T_keV)
    n_e19 = require_positive_float("n_e19", n_e19)

    eps_eff = effective_ripple(config, s)
    v_th = np.sqrt(T_keV * _EV_TO_J / _M_D)
    n_e_m3 = n_e19 * 1e19

    nu_ii = n_e_m3 * _E**4 * COULOMB_LOG / (4.0 * np.pi * _EPS0**2 * _M_D**2 * v_th**3)
    chi = eps_eff**1.5 * v_th**2 / (nu_ii * config.R0 * config.N_fp)

    return float(chi)

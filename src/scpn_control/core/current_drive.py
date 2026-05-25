# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851  Contact: protoscience@anulum.li
"""ECCD, NBI, and mixed current-drive source models for transport coupling."""

from __future__ import annotations

import numpy as np

# SI constants
E_CHARGE = 1.602176634e-19  # C
M_E = 9.1093837015e-31  # kg
M_P = 1.67262192369e-27  # kg
EPS_0 = 8.8541878128e-12  # F/m

# Deuterium beam defaults
_A_BEAM_D = 2.0  # dimensionless atomic mass
_Z_BEAM_D = 1.0  # charge state

# Coulomb logarithm for thermal-beam collisions; weakly sensitive to exact value
_LN_LAMBDA = 17.0  # dimensionless — Wesson, Tokamaks, 4th ed., §14.3

# LHCD efficiency range — Fisch 1978, PRL 41, 873; ITER design value ≈ 0.1–0.2
_ETA_LHCD_DEFAULT = 0.15  # A/W / (1e19 m^-3 · keV)^-1 normalised form

# ECCD baseline efficiency — Fisch & Boozer 1980, PRL 45, 720; typical range 0.02–0.05
_ETA_ECCD_DEFAULT = 0.03  # same normalisation as LHCD

# Stix 1972, Plasma Physics 14, 367, Eq. 8 — pre-factor for critical energy
_E_CRIT_PREFACTOR = 14.8  # keV · (A_b/A_i)^(2/3) per keV of T_e

# Stix 1972, Eq. 6 — pre-factor in the ion slowing-down time expression
_TAU_S_PREFACTOR = 3.0 * np.sqrt(2.0 * np.pi)  # = 3√(2π)


def _require_nonnegative_scalar(name: str, value: float) -> float:
    """Return a finite non-negative scalar or fail closed."""
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return scalar


def _require_positive_scalar(name: str, value: float) -> float:
    """Return a finite positive scalar or fail closed."""
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return scalar


def _require_unit_interval(name: str, value: float) -> float:
    """Return a finite scalar in [0, 1] or fail closed."""
    scalar = float(value)
    if not np.isfinite(scalar) or not (0.0 <= scalar <= 1.0):
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return scalar


def _require_positive_profile(
    name: str, values: float | np.ndarray, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    """Return finite positive profile values with optional exact shape."""
    arr = np.asarray(values, dtype=float)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    return arr


def _require_current_drive_profiles(
    rho: np.ndarray,
    ne_19: np.ndarray,
    Te_keV: np.ndarray,
    Ti_keV: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Validate grid and kinetic profiles used by current-drive sources."""
    rho_arr = np.asarray(rho, dtype=float)
    if rho_arr.ndim != 1:
        raise ValueError("rho grid must be one-dimensional")
    if not np.all(np.isfinite(rho_arr)):
        raise ValueError("rho grid must be finite")
    if rho_arr.size > 1 and not np.all(np.diff(rho_arr) > 0.0):
        raise ValueError("rho grid must be strictly increasing")
    ne_arr = _require_positive_profile("ne_19", ne_19, rho_arr.shape)
    te_arr = _require_positive_profile("Te_keV", Te_keV, rho_arr.shape)
    ti_arr = None if Ti_keV is None else _require_positive_profile("Ti_keV", Ti_keV, rho_arr.shape)
    return rho_arr, ne_arr, te_arr, ti_arr


def _require_current_drive_grid(rho: np.ndarray) -> np.ndarray:
    """Validate a one-dimensional finite strictly increasing rho grid."""
    rho_arr = np.asarray(rho, dtype=float)
    if rho_arr.ndim != 1:
        raise ValueError("rho grid must be one-dimensional")
    if not np.all(np.isfinite(rho_arr)):
        raise ValueError("rho grid must be finite")
    if rho_arr.size > 1 and not np.all(np.diff(rho_arr) > 0.0):
        raise ValueError("rho grid must be strictly increasing")
    return rho_arr


def _normalised_radial_deposition(
    rho: np.ndarray,
    total_power_w: float,
    rho_centre: float,
    width_rho: float,
) -> np.ndarray:
    """Return a finite-width radial deposition kernel conserving total power on the supplied grid."""
    rho_arr = _require_current_drive_grid(rho)
    if rho_arr.size == 0:
        return np.asarray([], dtype=float)
    _require_unit_interval("rho_centre", rho_centre)
    if width_rho <= 0.0 or total_power_w <= 0.0:
        return np.zeros_like(rho_arr, dtype=float)
    _require_positive_scalar("width_rho", width_rho)
    _require_positive_scalar("total_power_w", total_power_w)

    kernel = np.exp(-((rho_arr - rho_centre) ** 2) / (2.0 * width_rho**2))
    norm = float(np.trapezoid(kernel, rho_arr)) if rho_arr.size > 1 else float(kernel[0])
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("deposition kernel cannot be normalised on the supplied rho grid")
    return np.asarray(total_power_w * kernel / norm)


def eccd_efficiency(
    Te_keV: float | np.ndarray,
    Z_eff: float,
    N_parallel: float,
    eta_0: float = _ETA_ECCD_DEFAULT,
) -> float | np.ndarray:
    """
    ECCD figure of merit including T_e scaling and launch-angle factor.

    η_ECCD = η_0 · T_e / (5 + Z_eff) · ξ / (1 + ξ²)
    Prater 2004, Phys. Plasmas 11, 2349, Eq. 5.

    Parameters
    ----------
    Te_keV     : electron temperature [keV]
    Z_eff      : effective charge [dimensionless]
    N_parallel : parallel refractive index ξ [dimensionless]
    eta_0      : baseline coefficient [A/W per (1e19 m^-3 · keV) normalisation]
                 Fisch & Boozer 1980, PRL 45, 720

    Returns
    -------
    η_ECCD in same units as eta_0
    """
    Te_arr = _require_positive_profile("Te_keV", Te_keV)
    Z_eff = _require_positive_scalar("Z_eff", Z_eff)
    eta_0 = _require_nonnegative_scalar("eta_0", eta_0)
    if not np.isfinite(N_parallel):
        raise ValueError("N_parallel must be finite")
    xi = N_parallel
    angle_factor = xi / (1.0 + xi**2)
    result = eta_0 * Te_arr / (5.0 + Z_eff) * angle_factor
    return float(result) if np.ndim(result) == 0 else np.asarray(result)


def nbi_slowing_down_time(
    Te_keV: float | np.ndarray,
    ne_19: float | np.ndarray,
    A_beam: float = _A_BEAM_D,
    Z_eff: float = 1.5,
) -> float | np.ndarray:
    """
    Beam-ion classical slowing-down time on electrons.

    τ_s = 3√(2π) m_i T_e^(3/2) / (4 √(m_e) n_e e^4 ln_Λ Z_eff)
    Stix 1972, Plasma Physics 14, 367, Eq. 6.

    Parameters
    ----------
    Te_keV  : electron temperature [keV]
    ne_19   : electron density [10^19 m^-3]
    A_beam  : beam ion mass number [dimensionless]
    Z_eff   : effective plasma charge [dimensionless]

    Returns
    -------
    τ_s [s]
    """
    Te_arr = _require_positive_profile("Te_keV", Te_keV)
    ne_arr = _require_positive_profile("ne_19", ne_19, Te_arr.shape)
    A_beam = _require_positive_scalar("A_beam", A_beam)
    Z_eff = _require_positive_scalar("Z_eff", Z_eff)
    m_beam = A_beam * M_P
    Te_J = Te_arr * 1e3 * E_CHARGE
    ne = ne_arr * 1e19

    numerator = _TAU_S_PREFACTOR * m_beam * Te_J**1.5
    denominator = 4.0 * np.sqrt(M_E) * ne * E_CHARGE**4 * _LN_LAMBDA * Z_eff
    result = numerator / denominator
    return float(result) if np.ndim(result) == 0 else np.asarray(result)


def nbi_critical_energy(
    Te_keV: float | np.ndarray,
    A_beam: float = _A_BEAM_D,
    A_ion: float = 2.0,
) -> float | np.ndarray:
    """
    Critical energy separating electron- from ion-dominated beam heating.

    E_crit = 14.8 · T_e · (A_b / A_i)^(2/3)
    Stix 1972, Plasma Physics 14, 367, Eq. 8.

    Parameters
    ----------
    Te_keV : electron temperature [keV]
    A_beam : beam mass number [dimensionless]
    A_ion  : majority-ion mass number [dimensionless]

    Returns
    -------
    E_crit [keV]
    """
    Te_arr = _require_positive_profile("Te_keV", Te_keV)
    A_beam = _require_positive_scalar("A_beam", A_beam)
    A_ion = _require_positive_scalar("A_ion", A_ion)
    result = _E_CRIT_PREFACTOR * Te_arr * (A_beam / A_ion) ** (2.0 / 3.0)
    return float(result) if np.ndim(result) == 0 else np.asarray(result)


class ECCDSource:
    """
    Electron Cyclotron Current Drive.

    Default η_cd: Fisch & Boozer 1980, PRL 45, 720 (theory);
    range 0.02–0.05 A/W depending on launch angle and T_e.
    For angle-resolved efficiency, use eccd_efficiency().
    """

    def __init__(
        self,
        P_ec_MW: float,
        rho_dep: float,
        sigma_rho: float,
        eta_cd: float = _ETA_ECCD_DEFAULT,
    ):
        self.P_ec_MW = _require_nonnegative_scalar("P_ec_MW", P_ec_MW)
        self.rho_dep = _require_unit_interval("rho_dep", rho_dep)
        self.sigma_rho = _require_nonnegative_scalar("sigma_rho", sigma_rho)
        self.eta_cd = _require_nonnegative_scalar("eta_cd", eta_cd)

    def P_absorbed(self, rho: np.ndarray) -> np.ndarray:
        """Absorbed power per unit rho [W] using a grid-normalised finite-width deposition kernel."""
        return _normalised_radial_deposition(rho, self.P_ec_MW * 1e6, self.rho_dep, self.sigma_rho)

    def j_cd(self, rho: np.ndarray, ne_19: np.ndarray, Te_keV: np.ndarray) -> np.ndarray:
        """
        Driven current density [A/m^2].

        j_cd = η_cd · P_abs / (n_e · T_e)
        Fisch & Boozer 1980, PRL 45, 720, normalised form.
        """
        rho_arr, ne_arr, te_arr, _ = _require_current_drive_profiles(rho, ne_19, Te_keV)
        p_abs = self.P_absorbed(rho_arr)
        denom = ne_arr * te_arr
        return np.asarray(self.eta_cd * p_abs / denom)


class NBISource:
    """
    Neutral Beam Injection.

    Current-drive efficiency formula: Ehst & Karney 1991, Nucl. Fusion 31, 1933.
    Slowing-down time: Stix 1972, Plasma Physics 14, 367, Eq. 6.
    """

    def __init__(
        self,
        P_nbi_MW: float,
        E_beam_keV: float,
        rho_tangency: float,
        sigma_rho: float = 0.15,
        A_beam: float = _A_BEAM_D,
        Z_beam: float = _Z_BEAM_D,
    ):
        self.P_nbi_MW = _require_nonnegative_scalar("P_nbi_MW", P_nbi_MW)
        self.E_beam_keV = _require_positive_scalar("E_beam_keV", E_beam_keV)
        self.rho_tangency = _require_unit_interval("rho_tangency", rho_tangency)
        self.sigma_rho = _require_nonnegative_scalar("sigma_rho", sigma_rho)
        self.A_beam = _require_positive_scalar("A_beam", A_beam)
        self.Z_beam = _require_positive_scalar("Z_beam", Z_beam)

    def P_heating(self, rho: np.ndarray) -> np.ndarray:
        """Beam heating power per unit rho [W] using a grid-normalised finite-width deposition kernel."""
        return _normalised_radial_deposition(rho, self.P_nbi_MW * 1e6, self.rho_tangency, self.sigma_rho)

    def j_cd(
        self,
        rho: np.ndarray,
        ne_19: np.ndarray,
        Te_keV: np.ndarray,
        Ti_keV: np.ndarray,
        Z_eff: float = 1.5,
    ) -> np.ndarray:
        """
        Beam-driven current density [A/m^2].

        Fast-ion density from steady-state balance: n_fast = P_heat · τ_s / E_beam
        τ_s from Stix 1972, Eq. 6 (nbi_slowing_down_time).
        j_cd = e · n_fast · v_∥ / Z_beam
        Ehst & Karney 1991, Nucl. Fusion 31, 1933.

        Note: Ti_keV enters via Z_eff-dependent collisionality; here held constant.
        """
        rho_arr, ne_arr, te_arr, _ = _require_current_drive_profiles(rho, ne_19, Te_keV, Ti_keV)
        Z_eff = _require_positive_scalar("Z_eff", Z_eff)
        p_heat = self.P_heating(rho_arr)
        m_beam = self.A_beam * M_P
        E_beam_J = self.E_beam_keV * 1e3 * E_CHARGE
        v_parallel = np.sqrt(2.0 * E_beam_J / m_beam)

        j_prof = np.zeros_like(rho_arr, dtype=float)
        for i in range(len(rho_arr)):
            if p_heat[i] <= 0.0:
                continue
            tau_s = nbi_slowing_down_time(
                Te_keV=float(te_arr[i]),
                ne_19=float(ne_arr[i]),
                A_beam=self.A_beam,
                Z_eff=Z_eff,
            )
            n_fast = p_heat[i] * tau_s / E_beam_J
            j_prof[i] = E_CHARGE * n_fast * v_parallel / self.Z_beam

        return j_prof


class LHCDSource:
    """
    Lower Hybrid Current Drive.

    Default η_cd = 0.15 A/W (normalised):
    Fisch 1978, PRL 41, 873 (theory); ITER design range 0.1–0.2.
    """

    def __init__(
        self,
        P_lh_MW: float,
        rho_dep: float,
        sigma_rho: float,
        eta_cd: float = _ETA_LHCD_DEFAULT,
    ):
        self.P_lh_MW = _require_nonnegative_scalar("P_lh_MW", P_lh_MW)
        self.rho_dep = _require_unit_interval("rho_dep", rho_dep)
        self.sigma_rho = _require_nonnegative_scalar("sigma_rho", sigma_rho)
        self.eta_cd = _require_nonnegative_scalar("eta_cd", eta_cd)

    def P_absorbed(self, rho: np.ndarray) -> np.ndarray:
        return _normalised_radial_deposition(rho, self.P_lh_MW * 1e6, self.rho_dep, self.sigma_rho)

    def j_cd(self, rho: np.ndarray, ne_19: np.ndarray, Te_keV: np.ndarray) -> np.ndarray:
        """
        j_cd = η_cd · P_abs / (n_e · T_e)
        Fisch 1978, PRL 41, 873, normalised form.
        """
        rho_arr, ne_arr, te_arr, _ = _require_current_drive_profiles(rho, ne_19, Te_keV)
        p_abs = self.P_absorbed(rho_arr)
        denom = ne_arr * te_arr
        return np.asarray(self.eta_cd * p_abs / denom)


class CurrentDriveMix:
    """Superposition of multiple current-drive sources."""

    def __init__(self, a: float = 1.0):
        self.sources: list[ECCDSource | NBISource | LHCDSource] = []
        self.a = _require_positive_scalar("a", a)  # minor radius [m]

    def add_source(self, source: ECCDSource | NBISource | LHCDSource) -> None:
        self.sources.append(source)

    def total_j_cd(
        self,
        rho: np.ndarray,
        ne: np.ndarray,
        Te: np.ndarray,
        Ti: np.ndarray,
    ) -> np.ndarray:
        rho_arr, ne_arr, te_arr, ti_arr = _require_current_drive_profiles(rho, ne, Te, Ti)
        assert ti_arr is not None
        j_tot = np.zeros_like(rho_arr, dtype=float)
        for src in self.sources:
            if isinstance(src, NBISource):
                j_tot += src.j_cd(rho_arr, ne_arr, te_arr, ti_arr)
            else:
                j_tot += src.j_cd(rho_arr, ne_arr, te_arr)
        return j_tot

    def total_heating_power(self, rho: np.ndarray) -> np.ndarray:
        rho_arr = _require_current_drive_grid(rho)
        p_tot = np.zeros_like(rho_arr, dtype=float)
        for src in self.sources:
            if isinstance(src, NBISource):
                p_tot += src.P_heating(rho_arr)
            else:
                p_tot += src.P_absorbed(rho_arr)
        return p_tot

    def total_driven_current(
        self,
        rho: np.ndarray,
        ne: np.ndarray,
        Te: np.ndarray,
        Ti: np.ndarray,
    ) -> float:
        """
        I_cd = ∫ j_cd · 2π r dr  [A]

        r = ρ · a,  dr = dρ · a  →  dA = 2π ρ a² dρ
        """
        rho_arr, ne_arr, te_arr, ti_arr = _require_current_drive_profiles(rho, ne, Te, Ti)
        assert ti_arr is not None
        j_tot = self.total_j_cd(rho_arr, ne_arr, te_arr, ti_arr)
        current_density_integrand = j_tot * 2.0 * np.pi * rho_arr * self.a**2
        return float(np.trapezoid(current_density_integrand, rho_arr)) if rho_arr.size > 1 else 0.0

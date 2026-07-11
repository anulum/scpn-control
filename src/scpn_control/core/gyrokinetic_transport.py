# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Quasilinear Gyrokinetic Transport Model
"""Quasilinear gyrokinetic transport model with reduced instability-branch screening."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

# Physical constants
# m_p = 1.67262192369e-27  kg


def _finite_float(name: str, value: Any) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _positive_float(name: str, value: Any) -> float:
    scalar = _finite_float(name, value)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _nonnegative_float(name: str, value: Any) -> float:
    scalar = _finite_float(name, value)
    if scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _unit_interval(name: str, value: Any) -> float:
    scalar = _finite_float(name, value)
    if scalar < 0.0 or scalar > 1.0:
        raise ValueError(f"{name} must stay within [0, 1]")
    return scalar


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_params(params: GyrokineticsParams) -> GyrokineticsParams:
    params.R_L_Ti = _nonnegative_float("R_L_Ti", params.R_L_Ti)
    params.R_L_Te = _nonnegative_float("R_L_Te", params.R_L_Te)
    params.R_L_ne = _nonnegative_float("R_L_ne", params.R_L_ne)
    params.q = _positive_float("q", params.q)
    params.s_hat = _finite_float("s_hat", params.s_hat)
    params.alpha_MHD = _nonnegative_float("alpha_MHD", params.alpha_MHD)
    params.Te_Ti = _positive_float("Te_Ti", params.Te_Ti)
    params.Z_eff = _positive_float("Z_eff", params.Z_eff)
    params.nu_star = _nonnegative_float("nu_star", params.nu_star)
    params.beta_e = _nonnegative_float("beta_e", params.beta_e)
    params.epsilon = _positive_float("epsilon", params.epsilon)
    if params.epsilon > 1.0:
        raise ValueError("epsilon must be <= 1")
    return params


def _validate_spectrum(spectrum: SpectrumResult) -> SpectrumResult:
    lengths = {
        len(spectrum.k_y),
        len(spectrum.gamma_linear),
        len(spectrum.omega_r),
        len(spectrum.mode_type),
    }
    if len(lengths) != 1:
        raise ValueError("spectrum arrays must have matching lengths")
    if not np.all(np.isfinite(spectrum.k_y)) or np.any(spectrum.k_y <= 0.0):
        raise ValueError("spectrum k_y values must be finite and positive")
    if not np.all(np.isfinite(spectrum.gamma_linear)):
        raise ValueError("spectrum gamma_linear values must be finite")
    if not np.all(np.isfinite(spectrum.omega_r)):
        raise ValueError("spectrum omega_r values must be finite")
    if not np.all(np.isin(spectrum.mode_type, [0, 1, 2, 3])):
        raise ValueError("spectrum mode_type values must be 0, 1, 2, or 3")
    return spectrum


@dataclass
class GyrokineticsParams:
    """TGLF-10 style input vector for the quasilinear model.

    Gradient drives are non-negative normalised ``R/L`` values. ``q``,
    ``Te_Ti``, ``Z_eff``, and ``epsilon`` are positive; ``nu_star``,
    ``beta_e``, and ``alpha_MHD`` are non-negative.
    """

    R_L_Ti: float
    R_L_Te: float
    R_L_ne: float
    q: float
    s_hat: float
    alpha_MHD: float
    Te_Ti: float
    Z_eff: float
    nu_star: float
    beta_e: float
    epsilon: float = 0.1  # r / R


@dataclass
class SpectrumResult:
    """Computed instability spectrum."""

    k_y: AnyFloatArray
    gamma_linear: AnyFloatArray
    omega_r: AnyFloatArray
    mode_type: AnyFloatArray  # 0: stable, 1: ITG, 2: TEM, 3: ETG


@dataclass
class TransportFluxes:
    """Quasilinear fluxes and effective diffusivities."""

    chi_i: float
    chi_e: float
    D_e: float


def saturated_growth_rate(gamma_linear: Any, q: Any) -> float:
    """Return bounded mixing-length growth rate for quasilinear saturation.

    The reduced closure uses ``gamma_sat = gamma / (1 + gamma / gamma_max)``
    with ``gamma_max = 1/q`` in ``c_s/R`` units. The map is monotone,
    non-negative, and strictly bounded by ``gamma_max`` for finite positive
    ``q``. Stable or marginal modes return zero.
    """
    gamma = _nonnegative_float("gamma_linear", gamma_linear)
    q_value = _positive_float("q", q)
    if gamma == 0.0:
        return 0.0
    gamma_max = 1.0 / q_value
    return float(gamma / (1.0 + gamma / gamma_max))


def solve_dispersion(
    params: GyrokineticsParams, k_theta_rho_s: float, etg_scale: bool = False
) -> tuple[float, float, int]:
    """
    Solve the local electrostatic dispersion relation for growth rate gamma and real frequency omega_r.

    Parameters
    ----------
    params : GyrokineticsParams
        Local plasma parameters.
    k_theta_rho_s : float
        Positive normalised perpendicular wavenumber.
    etg_scale : bool
        If True, evaluate the ETG mode dispersion.

    Returns
    -------
    gamma : float
        Growth rate [c_s / R]
    omega_r : float
        Real frequency [c_s / R]
    mode_type : int
        1 for ITG, 2 for TEM, 3 for ETG, 0 for stable

    Raises
    ------
    ValueError
        If the local gyrokinetic parameters or wavenumber leave their finite
        physical domains.
    """
    params = _validate_params(params)
    k_y = _positive_float("k_theta_rho_s", k_theta_rho_s)

    if etg_scale:
        # ETG mode (Jenko et al. 2000)
        # R/L_Te_crit = (1 + Z_eff) * max(1.33 + 1.91 s_hat/q, 0)
        R_L_Te_crit = (1.0 + params.Z_eff) * max(1.33 + 1.91 * params.s_hat / params.q, 0.0)
        drive = params.R_L_Te - R_L_Te_crit
        if drive > 0.0:
            gamma = k_y * params.R_L_Te * np.sqrt(drive) / (1.0 + k_y**2)
            omega_r = k_y * params.R_L_Te  # omega_*Te
            return gamma, omega_r, 3
        return 0.0, 0.0, 0

    # ITG mode
    gamma_ITG = 0.0
    omega_ITG = 0.0
    # Dimits shift included
    R_L_Ti_crit = max(
        (4.0 / 3.0) * (1.0 + 1.0 / params.Te_Ti) * (1.0 + 2.0 * params.s_hat / params.q),
        0.0,
    )
    drive_ITG = params.R_L_Ti - R_L_Ti_crit
    if drive_ITG > 0.0:
        gamma_ITG = k_y * params.R_L_Ti * np.sqrt(drive_ITG) / (1.0 + k_y**2)
        # Ion diamagnetic direction
        omega_ITG = -k_y * params.R_L_Ti / params.Te_Ti

    # TEM mode
    gamma_TEM = 0.0
    omega_TEM = 0.0
    f_t = np.sqrt(2.0 * params.epsilon / (1.0 + params.epsilon))
    # Approximate omega_be_norm from nu_star. nu_star = nu_eff / omega_be
    # Thus nu_eff / omega_be = nu_star
    nu_eff_over_omega_be = params.nu_star
    omega_star_e = k_y * params.R_L_ne

    # Romanelli & Zonca 1993 TEM model
    drive_TEM = omega_star_e
    if drive_TEM > 0.0:
        gamma_TEM = f_t * omega_star_e / (1.0 + k_y**2 * (1.0 + nu_eff_over_omega_be))
        omega_TEM = omega_star_e

    # Identify dominant mode
    if gamma_ITG > gamma_TEM and gamma_ITG > 0.0:
        return gamma_ITG, omega_ITG, 1
    elif gamma_TEM > gamma_ITG and gamma_TEM > 0.0:
        return gamma_TEM, omega_TEM, 2

    return 0.0, 0.0, 0


def compute_spectrum(params: GyrokineticsParams, n_modes: int = 16, include_etg: bool = False) -> SpectrumResult:
    """
    Scan k_theta rho_s and compute growth rate spectrum over a positive mode count.
    """
    params = _validate_params(params)
    n_modes = _positive_int("n_modes", n_modes)
    k_y_list = []
    gamma_list = []
    omega_list = []
    type_list = []

    # Ion scale (ITG/TEM)
    k_y_ion = np.linspace(0.1, 2.0, n_modes)
    for ky in k_y_ion:
        g, w, mt = solve_dispersion(params, ky, etg_scale=False)
        k_y_list.append(ky)
        gamma_list.append(g)
        omega_list.append(w)
        type_list.append(mt)

    # Electron scale (ETG)
    if include_etg:
        k_y_elec = np.linspace(2.0, 30.0, n_modes)
        for ky in k_y_elec:
            g, w, mt = solve_dispersion(params, ky, etg_scale=True)
            # Normalization factor for ETG (c_e / R vs c_s / R)
            # Actually, standard TGLF treats them together but with sqrt(m_i/m_e)
            # Let's scale ETG gamma by sqrt(m_i/m_e) ~ 60 to put it in c_s/R units
            # For deuterium, sqrt(m_i/m_e) = 60.6
            mass_ratio_sqrt = 60.6
            k_y_list.append(ky)
            gamma_list.append(g * mass_ratio_sqrt)
            omega_list.append(w * mass_ratio_sqrt)
            type_list.append(mt)

    return SpectrumResult(
        np.array(k_y_list),
        np.array(gamma_list),
        np.array(omega_list),
        np.array(type_list, dtype=int),
    )


def quasilinear_fluxes(params: GyrokineticsParams, spectrum: SpectrumResult) -> TransportFluxes:
    """
    Apply saturation rule and return effective diffusivities for a valid spectrum.
    """
    params = _validate_params(params)
    spectrum = _validate_spectrum(spectrum)
    # The mixing-length growth-rate cap gamma_max = c_s/(q R) (= 1/q in c_s/R
    # units) is applied per mode inside saturated_growth_rate below, so it is
    # not recomputed here (CONTROL-F841-REVIEW: verified redundant, not a gap).

    chi_i = 0.0
    chi_e = 0.0
    D_e = 0.0

    for i in range(len(spectrum.k_y)):
        ky = spectrum.k_y[i]
        gamma_lin = spectrum.gamma_linear[i]
        omega_r = spectrum.omega_r[i]
        mt = spectrum.mode_type[i]

        if gamma_lin <= 0.0 or mt == 0:
            continue

        # Saturation rule
        gamma_sat = saturated_growth_rate(gamma_lin, params.q)

        # Mixing length amplitude
        phi_sq = 1.0 / ky**2

        # Quasilinear weights
        # Q_s = sum gamma_sat * phi_sq * (omega_*Ts / omega_r) * n_s T_s
        # chi_s = sum gamma_sat * phi_sq * (omega_*Ts / omega_r) / (R/L_Ts) * R
        # In normalized units (chi_s / chi_gB):
        # chi_s_norm = sum gamma_sat * phi_sq * (omega_*Ts / omega_r) / R_L_Ts

        if mt == 1:  # ITG
            # omega_*Ti = - ky * R_L_Ti / Te_Ti
            omega_star_Ti = -ky * params.R_L_Ti / params.Te_Ti
            weight_i = omega_star_Ti / omega_r if omega_r != 0.0 else 0.0
            if params.R_L_Ti > 0:
                chi_i += gamma_sat * phi_sq * weight_i

        elif mt == 2:  # TEM
            omega_star_Te = ky * params.R_L_Te
            omega_star_n = ky * params.R_L_ne
            weight_e = omega_star_Te / omega_r if omega_r != 0.0 else 0.0
            weight_n = omega_star_n / omega_r if omega_r != 0.0 else 0.0

            if params.R_L_Te > 0:
                chi_e += gamma_sat * phi_sq * weight_e
            if params.R_L_ne > 0:
                D_e += gamma_sat * phi_sq * weight_n

        elif mt == 3:  # ETG
            omega_star_Te = ky * params.R_L_Te
            weight_e = omega_star_Te / omega_r if omega_r != 0.0 else 0.0
            # For ETG, scale by (rho_e/rho_s)^2 ~ 1/3600
            # Actually, standard TGLF has separate scaling.
            # We already scaled gamma by 60. 1/ky^2 is in 1/(k_y_e)^2 = (rho_e/rho_s)^2 / (k_theta rho_s)^2
            # So multiply by 1 / 60.6**2
            mass_ratio_sq = 60.6**2
            if params.R_L_Te > 0:
                chi_e += (gamma_sat * phi_sq * weight_e) / mass_ratio_sq

    # Ensure positivity
    chi_i = max(chi_i, 0.0)
    chi_e = max(chi_e, 0.0)
    D_e = max(D_e, 0.0)

    return TransportFluxes(chi_i, chi_e, D_e)


class GyrokineticTransportModel:
    """
    Drop-in replacement for Gyro-Bohm transport scaling.

    Public evaluation points use ``rho`` in ``[0, 1]`` and positive finite local
    geometry, temperature, density, safety-factor, and charge inputs.
    """

    def __init__(self, n_modes: int = 16, include_etg: bool = False):
        self.n_modes = _positive_int("n_modes", n_modes)
        self.include_etg = include_etg
        # Typical tuning constant for macroscopic match
        self.c_tune = 0.5

    def evaluate(self, rho: float, profiles: dict[str, Any]) -> tuple[float, float, float]:
        """
        Evaluate transport coefficients at a single radial point.
        """
        rho = _unit_interval("rho", rho)
        if rho <= 0.05:
            # Axis boundary
            return 0.01, 0.01, 0.01

        # Extract local gradients and parameters
        R0 = _positive_float("R0", profiles.get("R0", 2.0))
        a = _positive_float("a", profiles.get("a", 0.5))
        B0 = _positive_float("B0", profiles.get("B0", 1.0))
        if a >= R0:
            raise ValueError("a must be smaller than R0 for tokamak ordering")
        q = _positive_float("q", profiles.get("q", 1.0))
        s_hat = _finite_float("s_hat", profiles.get("s_hat", 1.0))
        Te = _positive_float("Te", profiles.get("Te", 1.0))
        Ti = _positive_float("Ti", profiles.get("Ti", 1.0))
        ne = _positive_float("ne", profiles.get("ne", 1.0))
        Z_eff = _positive_float("Z_eff", profiles.get("Z_eff", 1.5))
        dTe_dr = _finite_float("dTe_dr", profiles.get("dTe_dr", 0.0))
        dTi_dr = _finite_float("dTi_dr", profiles.get("dTi_dr", 0.0))
        dne_dr = _finite_float("dne_dr", profiles.get("dne_dr", 0.0))

        # Gradients R/L
        # L_x = - x / (dx/dr) => R/L_x = - R/x * dx/dr
        R_L_Te = -R0 / max(Te, 1e-3) * dTe_dr
        R_L_Ti = -R0 / max(Ti, 1e-3) * dTi_dr
        R_L_ne = -R0 / max(ne, 1e-3) * dne_dr

        # Clamp gradients to reasonable bounds for stability
        R_L_Te = max(0.0, R_L_Te)
        R_L_Ti = max(0.0, R_L_Ti)
        R_L_ne = max(0.0, R_L_ne)

        # Secondary parameters
        Te_Ti = max(Te / max(Ti, 1e-3), 0.1)
        epsilon = max(rho * a / R0, 1e-3)

        # Collisionality estimate
        # nu_star ~ R * q / (v_te * tau_e * eps^1.5)
        # We can just use a proxy or 0.1 if not fully provided
        nu_star = _nonnegative_float("nu_star", profiles.get("nu_star", 0.1))
        beta_e = _nonnegative_float("beta_e", profiles.get("beta_e", 0.01))
        alpha_MHD = _nonnegative_float("alpha_MHD", profiles.get("alpha_MHD", 0.0))

        params = GyrokineticsParams(
            R_L_Ti=R_L_Ti,
            R_L_Te=R_L_Te,
            R_L_ne=R_L_ne,
            q=max(q, 0.5),
            s_hat=s_hat,
            alpha_MHD=alpha_MHD,
            Te_Ti=Te_Ti,
            Z_eff=Z_eff,
            nu_star=nu_star,
            beta_e=beta_e,
            epsilon=epsilon,
        )

        spec = compute_spectrum(params, self.n_modes, self.include_etg)
        fluxes = quasilinear_fluxes(params, spec)

        # Convert normalized chi to physical units (Gyro-Bohm scaling)
        # chi_gB = rho_s^2 * c_s / a (or R)
        # c_s = sqrt(Te / m_i)
        m_i = 2.0 * 1.6726219e-27
        e_charge = 1.602176634e-19
        Te_J = Te * 1e3 * e_charge
        c_s = np.sqrt(Te_J / m_i)
        rho_s = m_i * c_s / (e_charge * B0)
        chi_gB = rho_s**2 * c_s / R0

        chi_i_phys = fluxes.chi_i * chi_gB * self.c_tune
        chi_e_phys = fluxes.chi_e * chi_gB * self.c_tune
        D_e_phys = fluxes.D_e * chi_gB * self.c_tune

        return chi_i_phys, chi_e_phys, D_e_phys

    def evaluate_profile(
        self, rho: AnyFloatArray, profiles: dict[str, AnyFloatArray]
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """
        Evaluate full radial profile over a finite one-dimensional rho grid in [0, 1].
        """
        rho = np.asarray(rho, dtype=float)
        if rho.ndim != 1 or not np.all(np.isfinite(rho)) or np.any(rho < 0.0) or np.any(rho > 1.0):
            raise ValueError("rho must be a finite one-dimensional profile within [0, 1]")
        if np.any(np.diff(rho) <= 0.0):
            raise ValueError("rho must be strictly increasing")
        nr = len(rho)
        profile_keys = (
            "q",
            "s_hat",
            "Te",
            "Ti",
            "ne",
            "dTe_dr",
            "dTi_dr",
            "dne_dr",
            "nu_star",
            "beta_e",
            "alpha_MHD",
            "Z_eff",
        )
        for key in profile_keys:
            value = profiles.get(key)
            if value is None or isinstance(value, (int, float)):
                continue
            arr = np.asarray(value, dtype=float)
            if arr.shape != rho.shape:
                raise ValueError(f"profile {key} must match rho shape")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"profile {key} must contain only finite values")

        chi_i = np.zeros(nr)
        chi_e = np.zeros(nr)
        D_e = np.zeros(nr)

        R0 = profiles.get("R0", 2.0)
        a = profiles.get("a", 0.5)
        B0 = profiles.get("B0", 1.0)

        for i in range(nr):
            if rho[i] <= 0.05:
                chi_i[i] = 0.01
                chi_e[i] = 0.01
                D_e[i] = 0.01
                continue

            def _at(key: str, default: float) -> float:
                v = profiles.get(key, default)
                return float(v[i]) if hasattr(v, "__getitem__") and not isinstance(v, (int, float)) else float(v)

            local_profs = {
                "R0": R0,
                "a": a,
                "B0": B0,
                "q": _at("q", 1.0),
                "s_hat": _at("s_hat", 1.0),
                "Te": _at("Te", 1.0),
                "Ti": _at("Ti", 1.0),
                "ne": _at("ne", 1.0),
                "dTe_dr": _at("dTe_dr", 0.0),
                "dTi_dr": _at("dTi_dr", 0.0),
                "dne_dr": _at("dne_dr", 0.0),
                "nu_star": _at("nu_star", 0.1),
                "beta_e": _at("beta_e", 0.01),
                "alpha_MHD": _at("alpha_MHD", 0.0),
                "Z_eff": _at("Z_eff", 1.5),
            }
            ci, ce, de = self.evaluate(rho[i], local_profs)
            chi_i[i] = ci
            chi_e[i] = ce
            D_e[i] = de

        return chi_i, chi_e, D_e

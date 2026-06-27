# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — SOL Two-Point Physics
"""Two-point scrape-off-layer model and divertor heat-flux utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Parallel electron thermal conductivity coefficient κ_0 [W m^{-1} eV^{-7/2}].
# Stangeby 2000, "The Plasma Boundary of Magnetic Fusion Devices", Eq. 5.69.
KAPPA_0_ELECTRON = 2000.0

# Sheath heat transmission coefficient γ_sh (deuterium, classical sheath).
# Stangeby 2000, Ch. 2.
GAMMA_SHEATH = 7.0

# Divertor detachment onset criterion T_t <= 5 eV.
# Stangeby 2000, Ch. 16; Lipschultz et al. 1999, PPCF 41, A585.
DETACHMENT_ONSET_EV = 5.0


def _finite_scalar(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


@dataclass
class SOLSolution:
    """Scrape-off-layer two-point-model solution.

    Attributes
    ----------
    T_upstream_eV
        Upstream electron temperature in eV.
    T_target_eV
        Divertor-target electron temperature in eV.
    n_target_19
        Target electron density in 10¹⁹ m⁻³.
    q_parallel_MW_m2
        Parallel heat flux in MW/m².
    lambda_q_mm
        Heat-flux decay width in mm.
    """

    T_upstream_eV: float
    T_target_eV: float
    n_target_19: float
    q_parallel_MW_m2: float
    lambda_q_mm: float


def eich_heat_flux_width(P_SOL_MW: float, R0: float, B_pol: float, epsilon: float) -> float:
    """
    SOL heat-flux width λ_q [mm] from multi-machine regression.

    λ_q [mm] = 1.35 · P_SOL^{-0.02} · R^{0.04} · B_pol^{-0.92} · ε^{0.42}

    Eich et al. 2013, Nucl. Fusion 53, 093031, Eq. 6.
    """
    P_SOL_MW = _finite_scalar("P_SOL_MW", P_SOL_MW, positive=True)
    R0 = _finite_scalar("R0", R0, positive=True)
    B_pol = _finite_scalar("B_pol", B_pol, positive=True)
    epsilon = _finite_scalar("epsilon", epsilon, positive=True)
    if epsilon >= 1.0:
        raise ValueError("epsilon must be less than 1 for tokamak ordering")
    return float(1.35 * (P_SOL_MW**-0.02) * (R0**0.04) * (B_pol**-0.92) * (epsilon**0.42))


def peak_target_heat_flux(
    P_SOL_MW: float, R0: float, lambda_q_m: float, f_expansion: float = 5.0, alpha_deg: float = 3.0
) -> float:
    """
    Peak divertor target heat flux q_target [MW m^{-2}].

    q_target = P_SOL / (4π R λ_q f_exp) · sin α

    Stangeby 2000, Ch. 5.
    """
    P_SOL_MW = _finite_scalar("P_SOL_MW", P_SOL_MW, nonnegative=True)
    R0 = _finite_scalar("R0", R0, positive=True)
    lambda_q_m = _finite_scalar("lambda_q_m", lambda_q_m, positive=True)
    f_expansion = _finite_scalar("f_expansion", f_expansion, positive=True)
    alpha_deg = _finite_scalar("alpha_deg", alpha_deg, nonnegative=True)
    if alpha_deg > 90.0:
        raise ValueError("alpha_deg must be within [0, 90]")
    alpha_rad = np.radians(alpha_deg)
    q_peak = P_SOL_MW / (4.0 * np.pi * R0 * lambda_q_m * f_expansion) * np.sin(alpha_rad)
    return float(q_peak)


def detachment_threshold(n_u_19: float, P_SOL_MW: float, L_par: float) -> bool:
    """Return whether the two-point sheath target is below detachment onset.

    `P_SOL_MW` is interpreted as parallel heat-flux density in MW m^-2 for
    this local threshold helper.  The upstream temperature follows the Spitzer
    conduction integral, then the sheath heat-transmission relation gives the
    target temperature.  Detachment onset is taken as T_target <= 5 eV.
    """
    n_u_19 = _finite_scalar("n_u_19", n_u_19, positive=True)
    P_SOL_MW = _finite_scalar("P_SOL_MW", P_SOL_MW, positive=True)
    L_par = _finite_scalar("L_par", L_par, positive=True)

    q_parallel_W_m2 = P_SOL_MW * 1.0e6
    T_upstream_eV = ((3.5 * L_par * q_parallel_W_m2) / KAPPA_0_ELECTRON) ** (2.0 / 7.0)

    e_charge = 1.602e-19
    m_i = 2.0 * 1.6726e-27
    n_u = n_u_19 * 1.0e19
    sheath_denom = GAMMA_SHEATH * n_u * T_upstream_eV * e_charge
    T_target_eV = (2.0 * q_parallel_W_m2 / sheath_denom) ** 2 * m_i / e_charge
    return bool(T_target_eV <= DETACHMENT_ONSET_EV)


class TwoPointSOL:
    """
    Two-point model for the Scrape-Off Layer.

    Parallel heat conduction: q_∥ = κ_0 T_u^{7/2} / (7 L_∥).
    Stangeby 2000, Eq. 5.69.
    """

    def __init__(self, R0: float, a: float, q95: float, B_pol: float, kappa: float = 1.0):
        self.R0 = _finite_scalar("R0", R0, positive=True)
        self.a = _finite_scalar("a", a, positive=True)
        if self.a >= self.R0:
            raise ValueError("a must be smaller than R0 for tokamak ordering")
        self.q95 = _finite_scalar("q95", q95, positive=True)
        self.B_pol = _finite_scalar("B_pol", B_pol, positive=True)
        self.kappa = _finite_scalar("kappa", kappa, positive=True)
        self.epsilon = a / R0
        self.L_par = np.pi * q95 * R0

    def solve(self, P_SOL_MW: float, n_u_19: float, f_rad: float = 0.0) -> SOLSolution:
        """
        Solve two-point model for upstream and target conditions.

        Upstream temperature from conduction integral (Stangeby 2000, Eq. 5.69):
            T_u = (3.5 · L_∥ · q_∥ / κ_0)^{2/7}

        Target heat flux: q_target = P_SOL / (4π R λ_q f_exp).
        Stangeby 2000, Ch. 5.
        """
        P_SOL_MW = _finite_scalar("P_SOL_MW", P_SOL_MW, positive=True)
        n_u_19 = _finite_scalar("n_u_19", n_u_19, positive=True)
        f_rad = _finite_scalar("f_rad", f_rad, nonnegative=True)
        if f_rad >= 1.0:
            raise ValueError("f_rad must be less than 1")
        lambda_q_mm = eich_heat_flux_width(P_SOL_MW, self.R0, self.B_pol, self.epsilon)
        lambda_q_m = lambda_q_mm * 1e-3

        # B / B_p = q95 / ε; maps poloidal to total flux.
        B_ratio = self.q95 / self.epsilon
        q_par_u_W_m2 = (P_SOL_MW * 1e6) / (4.0 * np.pi * self.R0 * lambda_q_m) * B_ratio

        # T_u from conduction integral: Stangeby 2000, Eq. 5.69.
        T_u = ((3.5 * self.L_par * q_par_u_W_m2) / KAPPA_0_ELECTRON) ** (2.0 / 7.0)

        q_par_t_W_m2 = max(q_par_u_W_m2 * (1.0 - f_rad), 1e3)

        e_charge = 1.602e-19  # J eV^{-1}
        m_i = 2.0 * 1.6726e-27  # kg, deuterium
        n_u = n_u_19 * 1e19

        # Two-point model: conduction integral gives T_t from T_u.
        # T_t^{7/2} = T_u^{7/2} - (7/2) q_∥ L_∥ / κ_0
        # Stangeby 2000, Eq. 5.69 (integrated form).
        T_t_72 = T_u**3.5 - 3.5 * q_par_t_W_m2 * self.L_par / KAPPA_0_ELECTRON
        T_t_cond = T_t_72 ** (2.0 / 7.0) if T_t_72 > 0.0 else 0.0

        # Sheath-limited: q_∥ = γ n_t T_t √(eT_t/m_i), with n_u T_u = 2 n_t T_t.
        # Stangeby 2000, Ch. 2, Eqs. 2.84-2.86.
        # Solving for T_t: T_t = [2 q_∥ / (γ n_u T_u e)]² × m_i / e
        sheath_denom = GAMMA_SHEATH * n_u * max(T_u, 0.1) * e_charge
        T_t_sheath = (2.0 * q_par_t_W_m2 / sheath_denom) ** 2 * m_i / e_charge

        T_t = max(T_t_cond, T_t_sheath, 0.1)
        T_t = min(T_t, T_u)

        # Pressure balance: n_u T_u = 2 n_t T_t  (Stangeby 2000, Ch. 2)
        n_t = n_u * T_u / (2.0 * max(T_t, 0.1))

        return SOLSolution(
            T_upstream_eV=float(T_u),
            T_target_eV=float(T_t),
            n_target_19=float(n_t / 1e19),
            q_parallel_MW_m2=float(q_par_u_W_m2 / 1e6),
            lambda_q_mm=float(lambda_q_mm),
        )

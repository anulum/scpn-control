# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: protoscience@anulum.li
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Parallel electron thermal conductivity coefficient κ_0 [W m^{-1} eV^{-7/2}].
# Stangeby 2000, "The Plasma Boundary of Magnetic Fusion Devices", Eq. 5.69.
KAPPA_0_ELECTRON = 2000.0

# Sheath heat transmission coefficient γ_sh (deuterium, strong sheath).
# Stangeby 2000, Ch. 2.
GAMMA_SHEATH = 7.0


@dataclass
class SOLSolution:
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
    if P_SOL_MW <= 0.0 or B_pol <= 0.0 or R0 <= 0.0 or epsilon <= 0.0:
        return 1.0
    return float(1.35 * (P_SOL_MW**-0.02) * (R0**0.04) * (B_pol**-0.92) * (epsilon**0.42))


def peak_target_heat_flux(
    P_SOL_MW: float, R0: float, lambda_q_m: float, f_expansion: float = 5.0, alpha_deg: float = 3.0
) -> float:
    """
    Peak divertor target heat flux q_target [MW m^{-2}].

    q_target = P_SOL / (4π R λ_q f_exp) · sin α

    Stangeby 2000, Ch. 5.
    """
    if lambda_q_m <= 0.0:
        return 0.0
    alpha_rad = np.radians(alpha_deg)
    q_peak = P_SOL_MW / (4.0 * np.pi * R0 * lambda_q_m * f_expansion) * np.sin(alpha_rad)
    return float(q_peak)


def detachment_threshold(n_u_19: float, P_SOL_MW: float, L_par: float) -> bool:
    """Placeholder — detachment evaluated inside TwoPointSOL.solve."""
    return False


class TwoPointSOL:
    """
    Two-point model for the Scrape-Off Layer.

    Parallel heat conduction: q_∥ = κ_0 T_u^{7/2} / (7 L_∥).
    Stangeby 2000, Eq. 5.69.
    """

    def __init__(self, R0: float, a: float, q95: float, B_pol: float, kappa: float = 1.0):
        self.R0 = R0
        self.a = a
        self.q95 = q95
        self.B_pol = B_pol
        self.kappa = kappa
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
        if sheath_denom > 0.0:
            T_t_sheath = (2.0 * q_par_t_W_m2 / sheath_denom) ** 2 * m_i / e_charge
        else:
            T_t_sheath = 0.1

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

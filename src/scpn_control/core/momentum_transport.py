# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851  Contact: protoscience@anulum.li
from __future__ import annotations

import numpy as np

# Momentum Prandtl number: χ_φ / χ_i ≈ 0.7 (deuterium, ρ* ~ 0.004–0.007)
# Peeters et al. 2011, Nucl. Fusion 51, 083015, Fig. 5.
PRANDTL_MOMENTUM: float = 0.7


def nbi_torque(
    P_nbi_profile: np.ndarray,
    R0: float,
    v_beam: float,
    theta_inj_deg: float,
) -> np.ndarray:
    """NBI torque density [N·m/m³].

    T_NBI = P_NBI · R_tang / v_beam  where R_tang = R₀ sin(θ).
    Stacey & Sigmar 1985, Phys. Fluids 28, 2800.
    """
    if v_beam <= 0.0:
        return np.zeros_like(P_nbi_profile)

    theta_rad = np.radians(theta_inj_deg)
    R_tang = R0 * np.sin(theta_rad)  # tangential injection radius [m]
    return np.asarray(P_nbi_profile * R_tang / v_beam)


def intrinsic_rotation_torque(
    grad_Ti: np.ndarray,
    grad_ne: np.ndarray,
    R0: float,
    a: float,
) -> np.ndarray:
    """Residual-stress torque driving intrinsic counter-current rotation [N·m/m³].

    Residual stress Π_res ∝ −∇T_i → counter-current rotation.
    Empirical Rice scaling: v_φ,intr ∝ W_p / I_p.
    Rice et al. 2007, Nucl. Fusion 47, 1618, Eq. 3.
    """
    return np.asarray(-1e-3 * grad_Ti)


def exb_shearing_rate(
    omega_phi: np.ndarray,
    B_theta: np.ndarray,
    B0: float,
    R0: float,
    rho: np.ndarray,
    a: float,
) -> np.ndarray:
    """E×B shearing rate [rad/s].

    ω_E×B = (R B_θ / B) · d/dr (E_r / R B_θ)
    Burrell 1997, Phys. Plasmas 4, 1499, Eq. 1.

    In the rotation-dominated limit E_r ≈ v_φ B_θ = R₀ ω_φ B_θ, so
    E_r / (R₀ B_θ) ≈ ω_φ and ω_E×B ≈ (R₀ B_θ / B) · dω_φ/dr.
    """
    dr = (rho[1] - rho[0] if len(rho) > 1 else 0.1) * a
    domega_dr = np.gradient(omega_phi, dr)
    B_tot = np.sqrt(B0**2 + B_theta**2)
    rate = (R0 * B_theta / np.maximum(B_tot, 1e-6)) * domega_dr
    return np.asarray(np.abs(rate))


def turbulence_suppression_factor(
    omega_ExB: np.ndarray,
    gamma_max: np.ndarray,
) -> np.ndarray:
    """Suppression factor on anomalous transport.

    F = 1 / (1 + (ω_E×B / γ_max)²)
    Biglari, Diamond & Terry 1990, Phys. Fluids B 2, 1.
    """
    gamma_safe = np.maximum(gamma_max, 1e-6)
    return np.asarray(1.0 / (1.0 + (omega_ExB / gamma_safe) ** 2))


def radial_electric_field(
    ne: np.ndarray,
    Ti_keV: np.ndarray,
    omega_phi: np.ndarray,
    B_theta: np.ndarray,
    B0: float,
    R0: float,
    rho: np.ndarray,
    a: float,
) -> np.ndarray:
    """E_r [V/m] from radial force balance (neoclassical), Z_i = 1.

    E_r = (1 / Z_i e n_i) dp_i/dr + v_φ B_θ   (v_θ ≈ 0)
    Hinton & Hazeltine 1976, Rev. Mod. Phys. 48, 239, Eq. 2.3.
    """
    e_charge = 1.602e-19  # C
    p_i = ne * 1e19 * Ti_keV * 1e3 * e_charge

    drho = rho[1] - rho[0] if len(rho) > 1 else 0.1
    dp_dr = np.gradient(p_i, drho * a)

    term1 = dp_dr / np.maximum(e_charge * ne * 1e19, 1e-6)
    term2 = R0 * omega_phi * B_theta  # v_φ B_θ
    return np.asarray(term1 + term2)


def rice_intrinsic_velocity(W_p_MJ: float, I_p_MA: float) -> float:
    """Intrinsic toroidal velocity [km/s] from Rice scaling.

    v_φ,intr ≈ c_Rice · (W_p / I_p)
    Rice et al. 2007, Nucl. Fusion 47, 1618, Eq. 3.

    c_Rice = 3.5 km s⁻¹ MA MJ⁻¹ (empirical, Ohmic + NBI plasmas, Fig. 4).
    """
    C_RICE: float = 3.5  # km/s per (MJ/MA), Rice et al. 2007 Fig. 4
    return C_RICE * W_p_MJ / max(I_p_MA, 1e-9)


class RotationDiagnostics:
    @staticmethod
    def mach_number(
        omega_phi: np.ndarray,
        Ti_keV: np.ndarray,
        R0: float,
    ) -> np.ndarray:
        e_charge = 1.602e-19  # C
        m_i = 2.0 * 1.67e-27  # kg, deuterium
        v_phi = omega_phi * R0
        c_s = np.sqrt(Ti_keV * 1e3 * e_charge / m_i)
        return np.asarray(np.abs(v_phi) / np.maximum(c_s, 1e-3))

    @staticmethod
    def rwm_stabilization_criterion(
        omega_phi: np.ndarray,
        tau_wall: float,
    ) -> bool:
        # ω τ_wall > O(1) criterion; Bondeson & Ward 1994, Phys. Rev. Lett. 72, 2709.
        return bool(np.abs(omega_phi[0]) * tau_wall > 0.01)


class MomentumTransportSolver:
    def __init__(
        self,
        rho: np.ndarray,
        R0: float,
        a: float,
        B0: float,
        prandtl: float = PRANDTL_MOMENTUM,
    ) -> None:
        # prandtl = χ_φ / χ_i; Peeters et al. 2011, Nucl. Fusion 51, 083015, Fig. 5.
        self.rho = rho
        self.R0 = R0
        self.a = a
        self.B0 = B0
        self.prandtl = prandtl
        self.nr = len(rho)
        self.drho = rho[1] - rho[0]
        self.omega_phi = np.zeros(self.nr)

    def step(
        self,
        dt: float,
        chi_i: np.ndarray,
        ne: np.ndarray,
        Ti_keV: np.ndarray,
        T_nbi: np.ndarray,
        T_intrinsic: np.ndarray,
    ) -> np.ndarray:
        """Advance rotation profile one step [rad/s].

        Momentum equation (cylindrical approximation):
          ∂(ρ_m R² ω)/∂t + (1/r) ∂/∂r(r Π_φ) = T_tot
          Π_φ = −χ_φ ∂(ρ_m R² ω)/∂r   (pinch neglected)
        Wesson 2011, "Tokamaks", §4.10.
        """
        import scipy.linalg

        chi_phi = self.prandtl * chi_i  # Peeters et al. 2011

        m_i = 2.0 * 1.67e-27  # kg, deuterium
        rho_m = ne * 1e19 * m_i
        L = rho_m * self.R0**2 * self.omega_phi  # angular momentum density

        T_tot = T_nbi + T_intrinsic
        dr = self.drho * self.a

        diag = np.zeros(self.nr)
        upper = np.zeros(self.nr)
        lower = np.zeros(self.nr)
        rhs = np.zeros(self.nr)

        # Axis: zero-gradient BC
        diag[0] = 1.0
        upper[0] = -1.0
        rhs[0] = 0.0

        # Edge: no-slip BC
        diag[-1] = 1.0
        rhs[-1] = 0.0

        for i in range(1, self.nr - 1):
            r_val = self.rho[i] * self.a
            c_plus = chi_phi[i] / dr**2 + chi_phi[i] / (2.0 * r_val * dr)
            c_minus = chi_phi[i] / dr**2 - chi_phi[i] / (2.0 * r_val * dr)
            c_0 = -2.0 * chi_phi[i] / dr**2

            lower[i] = -dt * c_minus
            diag[i] = 1.0 - dt * c_0
            upper[i] = -dt * c_plus
            rhs[i] = L[i] + dt * T_tot[i]

        ab = np.zeros((3, self.nr))
        ab[0, 1:] = upper[:-1]
        ab[1, :] = diag
        ab[2, :-1] = lower[1:]

        L_new = scipy.linalg.solve_banded((1, 1), ab, rhs)
        self.omega_phi = L_new / np.maximum(rho_m * self.R0**2, 1e-12)
        return self.omega_phi

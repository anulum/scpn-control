# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: protoscience@anulum.li
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

# Physical constants
MU_0 = 4.0 * np.pi * 1e-7  # H m⁻¹


def coulomb_log(Te_keV: float, ne_19: float) -> float:
    """Temperature-dependent Coulomb logarithm ln_Λ.

    For T_e > 10 eV (Te_keV > 0.01):
        ln_Λ = 14.9 − 0.5·ln(n_e/10²⁰) + ln(T_e/10³)
    Reference: Wesson (2011), "Tokamaks" 4th ed., Eq. 2.12.4.
    n_e in m⁻³; T_e in eV.  n_e/10²⁰ and T_e/10³ are dimensionless ratios.
    """
    Te_eV = Te_keV * 1e3
    ne_m3 = ne_19 * 1e19
    # Wesson 2011 Eq. 2.12.4 — valid for T_e > 10 eV
    ln_lam = 14.9 - 0.5 * np.log(ne_m3 / 1e20) + np.log(Te_eV / 1e3)
    return float(max(ln_lam, 2.0))  # guard against unphysically low values


def neoclassical_resistivity(
    Te_keV: float, ne_19: float, Z_eff: float, epsilon: float, q: float = 1.0, R0: float = 2.0
) -> float:
    """Neoclassical parallel resistivity [Ω·m].

    Spitzer resistivity — Wesson (2011), "Tokamaks" 4th ed., Eq. 2.5.4:
        η_S = 1.65×10⁻⁹ · Z_eff · ln_Λ / T_e[keV]^{3/2}

    Trapped fraction — Sauter, Angioni & Lin-Liu (1999), Phys. Plasmas 6, 2834,
    Appendix A, Eq. A.1 (large-aspect-ratio limit retained here for efficiency):
        f_t = 1 − (1−ε)² / (√(1−ε²) · (1 + 1.46√ε))

    Neoclassical correction C_R — Sauter (2002), Phys. Plasmas 9, 5140,
    Eq. 8 (analytical limit, collisionless banana regime):
        C_R = 1 − (1 + 0.36/Z_eff) f_t + (0.59/Z_eff) f_t²

    Full neoclassical formula — Hirshman (1981), Phys. Fluids 24, 1274,
    resistive diffusion coefficient η_∥/μ₀:
        η_neo = η_S · C_R / (1 − f_t)

    Regime validity note:
        banana regime:    ν*_e < 1       → C_R formula valid as implemented
        plateau regime:   1 < ν*_e < ε⁻³/²
        Pfirsch-Schlüter: ν*_e > ε⁻³/²  → C_R → 1 (Spitzer limit)
    The correction saturates to η_S in the PS limit via the max() guard below.
    """
    Te_keV = max(Te_keV, 1e-3)
    ne_19 = max(ne_19, 1e-3)
    epsilon = max(epsilon, 1e-6)

    ln_lam = coulomb_log(Te_keV, ne_19)
    # Wesson 2011 Eq. 2.5.4
    eta_Spitzer = 1.65e-9 * Z_eff * ln_lam / (Te_keV**1.5)

    # Sauter 1999 Eq. A.1 trapped fraction
    f_t = 1.0 - (1.0 - epsilon) ** 2 / (np.sqrt(1.0 - epsilon**2) * (1.0 + 1.46 * np.sqrt(epsilon)))
    f_t = float(np.clip(f_t, 0.0, 1.0))

    # Electron collisionality ν*_e — Sauter 1999 Eq. A.5
    e_charge = 1.602e-19
    m_e = 9.109e-31
    v_te = np.sqrt(2.0 * Te_keV * 1e3 * e_charge / m_e)
    nu_ei = (
        (ne_19 * 1e19)
        * Z_eff
        * e_charge**4
        * ln_lam
        / (12.0 * np.pi**1.5 * (8.854e-12) ** 2 * np.sqrt(m_e) * (Te_keV * 1e3 * e_charge) ** 1.5)
    )
    nu_star_e = nu_ei * max(q, 0.5) * R0 / (epsilon**1.5 * v_te)  # noqa: F841 — retained for future PS-regime branch

    # Sauter 2002 Eq. 8 — banana regime analytical limit
    C_R = 1.0 - (1.0 + 0.36 / Z_eff) * f_t + (0.59 / Z_eff) * f_t**2

    # Hirshman 1981 neoclassical formula; capped at Spitzer (PS limit)
    eta_neo = eta_Spitzer / (1.0 - f_t) * C_R
    return float(max(eta_neo, eta_Spitzer))


def q_from_psi(rho: np.ndarray, psi: np.ndarray, R0: float, a: float, B0: float) -> np.ndarray:
    """Safety factor q(ρ) from poloidal flux ψ(ρ).

    q(ρ) = −ρ a² B₀ / (R₀ · ∂ψ/∂ρ)
    L'Hôpital's rule at ρ=0: q(0) = −a² B₀ / (R₀ · ∂²ψ/∂ρ²|₀)
    Reference: Jardin (2010), "Computational Methods in Plasma Physics",
    CRC Press, Ch. 8, Eq. 8.2.
    """
    nr = len(rho)
    q = np.zeros(nr)
    drho = rho[1] - rho[0]
    dpsi_drho = np.gradient(psi, drho)

    for i in range(1, nr):
        denom = R0 * dpsi_drho[i]
        if abs(denom) < 1e-12:
            q[i] = q[i - 1] if i > 1 else 1.0
        else:
            q[i] = -rho[i] * a**2 * B0 / denom

    d2psi = (psi[2] - 2 * psi[1] + psi[0]) / (drho**2)
    q[0] = -(a**2) * B0 / (R0 * d2psi) if abs(d2psi) > 1e-12 else q[1]

    np.abs(q, out=q)
    return q


def resistive_diffusion_time(a: float, eta: float) -> float:
    """Resistive timescale τ_R = μ₀ a² / η.

    Reference: Jardin (2010), Ch. 8, Eq. 8.5.
    """
    return MU_0 * a**2 / max(eta, 1e-12)


class CurrentDiffusionSolver:
    """Implicit Crank-Nicolson solver for poloidal flux diffusion.

    Governing equation (1D in ρ, cylindrical limit):
        ∂ψ/∂t = (η_∥/μ₀) · [∂²ψ/∂ρ² + (1/ρ)·∂ψ/∂ρ] + R₀·η_∥·j_source
    Reference: Jardin (2010), Ch. 8, Eq. 8.8.

    η_∥ from neoclassical_resistivity() — Sauter 1999/2002 + Hirshman 1981.
    Crank-Nicolson discretisation — θ=½ implicit-explicit split,
    second-order accurate in time and space.
    """

    def __init__(self, rho: np.ndarray, R0: float, a: float, B0: float):
        self.rho = rho
        self.R0 = R0
        self.a = a
        self.B0 = B0
        self.nr = len(rho)
        self.drho = rho[1] - rho[0]

        # Initialise ψ for q(0)=1, q(1)=3 (parabolic current profile)
        # ψ(ρ) = −∫₀^ρ ρ′ a² B₀ / (R₀ q(ρ′)) dρ′,  q(ρ) = 1 + 2ρ²
        self.psi = np.zeros(self.nr)
        for i in range(1, self.nr):
            r = self.rho[i]
            q_r = 1.0 + 2.0 * r**2
            self.psi[i] = self.psi[i - 1] - r * a**2 * B0 / (R0 * q_r) * self.drho

        self.psi -= self.psi[-1]  # ψ(edge) = 0 convention

    def step(
        self,
        dt: float,
        Te: np.ndarray,
        ne: np.ndarray,
        Z_eff: float,
        j_bs: np.ndarray,
        j_cd: np.ndarray,
        j_ext: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advance ψ by one timestep dt [s].

        Source term: j_source = j_bs + j_cd + j_ext
        Boundary conditions: axis (ρ=0) — regularity (L'Hôpital);
                             edge (ρ=1) — Dirichlet ψ=const.
        """
        if j_ext is None:
            j_ext = np.zeros(self.nr)

        j_tot_source = j_bs + j_cd + j_ext

        q_prof = q_from_psi(self.rho, self.psi, self.R0, self.a, self.B0)
        eta_prof = np.array(
            [
                neoclassical_resistivity(
                    Te[i],
                    ne[i],
                    Z_eff,
                    self.rho[i] * self.a / self.R0,
                    q_prof[i],
                    self.R0,
                )
                for i in range(self.nr)
            ]
        )

        # D(ρ) = η_∥ / (μ₀ a²) — diffusion coefficient in ρ-space
        D = eta_prof / (MU_0 * self.a**2)

        alpha = dt / 2.0
        drho2 = self.drho**2

        sub = np.zeros(self.nr)
        diag = np.zeros(self.nr)
        sup = np.zeros(self.nr)
        rhs = np.zeros(self.nr)

        # Axis (ρ=0): regularity condition — L'Hôpital gives d²ψ/dρ² coefficient ×4
        diag[0] = 1.0 + alpha * 4.0 * D[0] / drho2
        sup[0] = -alpha * 4.0 * D[0] / drho2
        rhs[0] = (
            self.psi[0]
            + alpha * 4.0 * D[0] * (self.psi[1] - self.psi[0]) / drho2
            + dt * self.R0 * eta_prof[0] * j_tot_source[0]
        )

        for i in range(1, self.nr - 1):
            r = self.rho[i]
            c_prev = D[i] * (1.0 / drho2 - 1.0 / (2.0 * r * self.drho))
            c_curr = D[i] * (-2.0 / drho2)
            c_next = D[i] * (1.0 / drho2 + 1.0 / (2.0 * r * self.drho))

            sub[i] = -alpha * c_prev
            diag[i] = 1.0 - alpha * c_curr
            sup[i] = -alpha * c_next

            L_psi_n = c_prev * self.psi[i - 1] + c_curr * self.psi[i] + c_next * self.psi[i + 1]
            rhs[i] = self.psi[i] + alpha * L_psi_n + dt * self.R0 * eta_prof[i] * j_tot_source[i]

        # Edge (ρ=1): Dirichlet ψ(edge) = const — flux boundary
        diag[-1] = 1.0
        sub[-1] = 0.0
        rhs[-1] = self.psi[-1]

        ab = np.zeros((3, self.nr))
        ab[0, 1:] = sup[:-1]
        ab[1, :] = diag
        ab[2, :-1] = sub[1:]

        self.psi = solve_banded((1, 1), ab, rhs)
        return self.psi

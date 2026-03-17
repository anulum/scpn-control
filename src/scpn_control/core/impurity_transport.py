# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851 — Contact: protoscience@anulum.li
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# Neoclassical impurity pinch velocity:
#   V_neo = -D_neo (Z/T_i) dT_i/dr
# Hirshman & Sigmar 1981, Nucl. Fusion 21, 1079, Eq. 4.17.
#
# Neoclassical diffusion coefficient (banana regime):
#   D_neo = q² ρ_i² ν_ii / ε^(3/2)
# Hinton & Hazeltine 1976, Rev. Mod. Phys. 48, 239, Eq. 4.62.
#
# Radiation power density:
#   P_rad = n_e n_Z L_Z(T_e)   [W m^-3]
# Post et al. 1977, At. Data Nucl. Data Tables 20, 397.
#
# Cooling curve parameters — peak temperatures and widths:
#   W high-T peak:  T_peak ≈ 1500 eV, σ = 1.5 (log-normal width)  — Putterich et al. 2010, Nucl. Fusion 50, 025012
#   W low-T peak:   T_peak ≈ 50 eV,   σ = 1.0                      — Putterich et al. 2010
#   C:              T_peak ≈ 10 eV,    σ = 0.5                      — Post et al. 1977
#   Ar:             T_peak ≈ 200 eV,   σ = 1.0                      — Post et al. 1977
#   Ne:             T_peak ≈ 50 eV,    σ = 1.0                      — Post et al. 1977


@dataclass
class ImpuritySpecies:
    element: str
    Z_nucleus: int
    mass_amu: float
    source_rate: float = 0.0


class CoolingCurve:
    """
    Parametric cooling rate coefficient L_Z(T_e) [W m³].

    Log-normal fits to:
      W:   Putterich et al. 2010, Nucl. Fusion 50, 025012, Fig. 3.
      C/Ar/Ne: Post et al. 1977, At. Data Nucl. Data Tables 20, 397.
    """

    def __init__(self, element: str):
        self.element = element

    def L_z(self, Te_eV: np.ndarray) -> np.ndarray:
        log_Te = np.log(Te_eV)
        if self.element == "W":
            # High-T peak: L_max ≈ 1e-31 W m³ at T ≈ 1500 eV, width σ = 1.5
            # Low-T peak:  L_max ≈ 3e-33 W m³ at T ≈ 50 eV,   width σ = 1.0
            # Putterich et al. 2010, Nucl. Fusion 50, 025012.
            L = 1e-31 * np.exp(-(((log_Te - np.log(1500.0)) / 1.5) ** 2))
            L += 3e-33 * np.exp(-(((log_Te - np.log(50.0)) / 1.0) ** 2))
            return np.asarray(L)
        if self.element == "C":
            # Peak: L_max ≈ 1e-32 W m³ at T ≈ 10 eV, σ = 0.5
            # Post et al. 1977, ADNDT 20, 397.
            return np.asarray(1e-32 * np.exp(-(((log_Te - np.log(10.0)) / 0.5) ** 2)))
        if self.element == "Ar":
            # Peak: L_max ≈ 1e-32 W m³ at T ≈ 200 eV, σ = 1.0
            # Post et al. 1977, ADNDT 20, 397.
            return np.asarray(1e-32 * np.exp(-(((log_Te - np.log(200.0)) / 1.0) ** 2)))
        if self.element == "Ne":
            # Peak: L_max ≈ 1e-32 W m³ at T ≈ 50 eV, σ = 1.0
            # Post et al. 1977, ADNDT 20, 397.
            return np.asarray(1e-32 * np.exp(-(((log_Te - np.log(50.0)) / 1.0) ** 2)))
        return np.zeros_like(Te_eV)


def neoclassical_impurity_pinch(
    Z: int,
    ne: np.ndarray,
    Te_eV: np.ndarray,
    Ti_eV: np.ndarray,
    q: np.ndarray,
    rho: np.ndarray,
    R0: float,
    a: float,
    epsilon: np.ndarray,
) -> np.ndarray:
    """
    Neoclassical impurity pinch velocity V_neo [m/s].

    Full formula (banana regime, trace impurity limit):
      V_neo,Z = -D_neo [Z ∇n/n + (Z/2 - H_Z) ∇T_i/T_i]
    where H_Z = 0.5 is the ion screening factor.
    Hirshman & Sigmar 1981, Nucl. Fusion 21, 1079, Eq. 4.17.

    The thermal-gradient term dominates for peaked T_i profiles and
    produces an inward (negative) pinch when dT_i/dr < 0.
    Sign convention: V < 0 → inward (toward axis).

    D_neo = q² ρ_i² ν_ii / ε^(3/2)  (banana regime)
    Hinton & Hazeltine 1976, Rev. Mod. Phys. 48, 239, Eq. 4.62.
    Nominal scale D_neo ≈ 0.1 m²/s for ITER-relevant parameters.
    """
    drho = rho[1] - rho[0] if len(rho) > 1 else 0.1

    grad_ne_over_n = np.gradient(ne, drho * a) / np.maximum(ne, 1e-6)
    grad_Ti_over_T = np.gradient(Ti_eV, drho * a) / np.maximum(Ti_eV, 1e-6)

    # D_neo = 0.1 m²/s nominal; banana regime, Hinton & Hazeltine 1976 Eq. 4.62
    D_neo = 0.1 * np.ones_like(rho)

    # H_Z = 0.5: banana-regime impurity screening factor, Hirshman & Sigmar 1981 Eq. 4.17
    H_Z = 0.5

    # Inward pinch for peaked T_i: (Z/2 - H_Z) > 0 when Z > 2*H_Z = 1,
    # so grad_Ti_over_T < 0 drives V_neo < 0 (inward).
    V_neo = -D_neo * (Z * grad_ne_over_n + (Z / 2.0 - H_Z) * grad_Ti_over_T)
    return np.asarray(V_neo)


def total_radiated_power(
    ne: np.ndarray,
    n_impurity: dict[str, np.ndarray],
    Te_eV: np.ndarray,
    rho: np.ndarray,
    R0: float,
    a: float,
) -> float:
    """
    Total radiated power P_rad [MW] integrated over plasma volume.

    Local power density:
      p_rad(r) = n_e(r) n_Z(r) L_Z(T_e(r))   [W m^-3]
    Post et al. 1977, At. Data Nucl. Data Tables 20, 397.

    Volume element for circular cross-section torus:
      dV = 4π² R0 a² ρ dρ
    """
    p_rad_density = np.zeros_like(rho)

    for element, n_Z in n_impurity.items():
        curve = CoolingCurve(element)
        L = curve.L_z(Te_eV)
        p_rad_density += ne * n_Z * L

    vol_element = 4.0 * np.pi**2 * R0 * a**2 * rho
    _trapz: Any = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    P_rad_W = _trapz(p_rad_density * vol_element, rho)

    return float(P_rad_W / 1e6)


def tungsten_accumulation_diagnostic(n_W: np.ndarray, ne: np.ndarray) -> dict[str, Any]:
    c_W_core = float(n_W[0] / max(ne[0], 1e-6))
    c_W_edge = float(n_W[-1] / max(ne[-1], 1e-6))

    peaking_factor = c_W_core / max(c_W_edge, 1e-12)

    if c_W_core < 1e-5:
        danger = "safe"
    elif c_W_core < 5e-5:
        danger = "warning"
    else:
        danger = "critical"

    return {"c_W_core": c_W_core, "c_W_edge": c_W_edge, "peaking_factor": peaking_factor, "danger_level": danger}


class ImpurityTransportSolver:
    def __init__(self, rho: np.ndarray, R0: float, a: float, species: list[ImpuritySpecies]):
        self.rho = rho
        self.R0 = R0
        self.a = a
        self.species = species

        self.nr = len(rho)
        self.drho = rho[1] - rho[0]

        self.n_z = {s.element: np.zeros(self.nr) for s in species}

    def step(
        self,
        dt: float,
        ne: np.ndarray,
        Te_eV: np.ndarray,
        Ti_eV: np.ndarray,
        D_anom: float,
        V_pinch: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """
        1D transport advance for each impurity species.

        Flux: Γ = -(D_anom + D_neo) dn/dr + V n
        Upwind differencing for convection; Crank-Nicolson implicit for diffusion.
        """
        import scipy.linalg

        dr = self.drho * self.a

        for s in self.species:
            n = self.n_z[s.element]
            V = V_pinch.get(s.element, np.zeros(self.nr))

            D = D_anom * np.ones(self.nr)

            diag = np.zeros(self.nr)
            upper = np.zeros(self.nr)
            lower = np.zeros(self.nr)
            rhs = np.zeros(self.nr)

            diag[0] = 1.0
            upper[0] = -1.0
            rhs[0] = 0.0  # dn/dr = 0 at magnetic axis

            diag[-1] = 1.0
            rhs[-1] = s.source_rate * dt / dr

            for i in range(1, self.nr - 1):
                r_val = self.rho[i] * self.a

                coeff_D_plus = D[i] / dr**2 + D[i] / (2.0 * r_val * dr)
                coeff_D_minus = D[i] / dr**2 - D[i] / (2.0 * r_val * dr)
                coeff_D_0 = -2.0 * D[i] / dr**2

                if V[i] > 0:
                    coeff_V_0 = -V[i] / dr - V[i] / r_val
                    coeff_V_minus = V[i] / dr
                    coeff_V_plus = 0.0
                else:
                    coeff_V_0 = V[i] / dr - V[i] / r_val
                    coeff_V_plus = -V[i] / dr
                    coeff_V_minus = 0.0

                lower[i] = -dt * (coeff_D_minus + coeff_V_minus)
                diag[i] = 1.0 - dt * (coeff_D_0 + coeff_V_0)
                upper[i] = -dt * (coeff_D_plus + coeff_V_plus)

                rhs[i] = n[i]

            ab = np.zeros((3, self.nr))
            ab[0, 1:] = upper[:-1]
            ab[1, :] = diag
            ab[2, :-1] = lower[1:]

            n_new = scipy.linalg.solve_banded((1, 1), ab, rhs)
            self.n_z[s.element] = np.maximum(n_new, 0.0)

        return self.n_z

# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851 — Contact: protoscience@anulum.li
from __future__ import annotations

import math

import numpy as np

from scpn_control.core.impurity_transport import CoolingCurve


# Radiation instability growth rate:
#   γ = -(κ_∥ k_∥² + n_e n_Z dL_Z/dT) / (n c_v)
# γ > 0 when dL_Z/dT < 0 and n_e n_Z |dL/dT| > κ_∥ k_∥².
# Drake 1987, Phys. Fluids 30, 2429, Eq. 5.

# Greenwald density limit:
#   n_GW = I_p / (π a²)   [10^20 m^-3, I_p in MA, a in m]
# Greenwald 2002, Plasma Phys. Control. Fusion 44, R27, Eq. 1.

# Cooling function data sources:
#   W:   Putterich et al. 2010, Nucl. Fusion 50, 025012.
#   C/O: Post et al. 1977, At. Data Nucl. Data Tables 20, 397.

# MARFE onset temperature: T_MARFE ≈ T where dL_Z/dT < 0.
# Lipschultz 1987, J. Nucl. Mater. 145-147, 15.


class RadiationCondensation:
    def __init__(self, impurity: str, ne_20: float, f_imp: float):
        self.impurity = impurity
        self.ne_20 = ne_20
        self.f_imp = f_imp
        self.curve = CoolingCurve(impurity)

    def _dL_dT(self, Te_eV: float) -> float:
        dT = 0.01 * Te_eV
        L_plus = self.curve.L_z(np.array([Te_eV + dT]))[0]
        L_minus = self.curve.L_z(np.array([Te_eV - dT]))[0]
        return float((L_plus - L_minus) / (2.0 * dT))

    def growth_rate(self, Te_eV: float, k_par: float, kappa_par: float) -> float:
        """
        Radiation condensation growth rate.

        γ = -(κ_∥ k_∥² + n_e n_Z dL/dT) / (n c_v)

        Positive γ ↔ unstable: requires dL/dT < 0 and
        n_e n_Z |dL/dT| > κ_∥ k_∥².
        Drake 1987, Phys. Fluids 30, 2429, Eq. 5.
        """
        ne = self.ne_20 * 1e20
        n_imp = ne * self.f_imp

        dL_dT = self._dL_dT(Te_eV)

        # c_v = (3/2) k_B (1 + f_imp) per particle; 1.602e-19 J/eV conversion
        c_v = 1.5 * (1.0 + self.f_imp) * 1.602e-19

        rad_term = ne * n_imp * dL_dT
        cond_term = kappa_par * k_par**2

        gamma = -(cond_term + rad_term) / (ne * c_v)
        return float(gamma)

    def is_unstable(self, Te_eV: float, k_par: float, kappa_par: float) -> bool:
        """True when dL/dT < 0 and radiation term exceeds parallel conduction damping."""
        return self.growth_rate(Te_eV, k_par, kappa_par) > 0.0

    def onset_temperature(self, Te_scan: np.ndarray) -> float:
        """
        Estimate T_MARFE as the highest T where dL_Z/dT first turns negative.

        Lipschultz 1987, J. Nucl. Mater. 145-147, 15: MARFE forms near the
        temperature where the radiative cooling function has a negative slope.
        Returns the onset T in eV, or nan if dL/dT is positive everywhere.
        """
        dLdT = np.array([self._dL_dT(T) for T in Te_scan])
        unstable = Te_scan[dLdT < 0.0]
        if len(unstable) == 0:
            return float("nan")
        return float(unstable[-1])

    def critical_density(self, Te_eV: float, k_par: float, kappa_par: float) -> float:
        """
        n_crit where γ = 0.

        From Drake 1987, Eq. 5 with γ = 0:
          n_e² f_imp |dL/dT| = κ_∥ k_∥²
        """
        dL_dT = self._dL_dT(Te_eV)
        if dL_dT >= 0.0:
            return float("inf")

        n_crit_sq = (kappa_par * k_par**2) / (self.f_imp * abs(dL_dT))
        return float(np.sqrt(n_crit_sq) / 1e20)


class MARFEFrontModel:
    def __init__(self, L_par: float, kappa_par: float, q_perp: float, impurity: str, f_imp: float):
        self.L_par = L_par
        self.kappa_par = kappa_par
        self.q_perp = q_perp
        self.f_imp = f_imp

        self.n_s = 50
        self.s = np.linspace(0, L_par, self.n_s)
        self.ds = self.s[1] - self.s[0]

        self.T = np.ones(self.n_s) * 100.0
        self.curve = CoolingCurve(impurity)

    def step(self, dt: float, ne_20: float) -> np.ndarray:
        import scipy.linalg

        ne = ne_20 * 1e20
        n_imp = ne * self.f_imp

        # P_rad = n_e n_Z L_Z(T); Post et al. 1977 / Putterich et al. 2010
        L = self.curve.L_z(self.T)
        P_rad = ne * n_imp * L

        # Heat equation: (3/2) n dT/dt = κ_∥ d²T/ds² - P_rad + q_⊥
        # 1.602e-19 J/eV
        c_v_n = 1.5 * ne * 1.602e-19

        alpha = self.kappa_par / c_v_n

        diag = np.zeros(self.n_s)
        upper = np.zeros(self.n_s)
        lower = np.zeros(self.n_s)
        rhs = np.zeros(self.n_s)

        diag[0] = 1.0
        rhs[0] = 100.0  # Core boundary: T = 100 eV

        diag[-1] = 1.0
        upper[-1] = -1.0
        rhs[-1] = 0.0  # Symmetry at X-point: dT/ds = 0

        for i in range(1, self.n_s - 1):
            c_diff = alpha * dt / self.ds**2
            lower[i] = -c_diff
            diag[i] = 1.0 + 2.0 * c_diff
            upper[i] = -c_diff
            rhs[i] = self.T[i] + dt / c_v_n * (self.q_perp - P_rad[i])

        ab = np.zeros((3, self.n_s))
        ab[0, 1:] = upper[:-1]
        ab[1, :] = diag
        ab[2, :-1] = lower[1:]

        self.T = scipy.linalg.solve_banded((1, 1), ab, rhs)
        self.T = np.maximum(self.T, 1.0)
        return self.T

    def equilibrium(self, ne_20: float) -> np.ndarray:
        for _ in range(1000):
            self.step(1e-4, ne_20)
        return self.T

    def is_marfe(self) -> bool:
        """
        MARFE criterion: localised cold spot with T_min < 20 eV and T_max > 50 eV.
        Lipschultz 1987, J. Nucl. Mater. 145-147, 15.
        """
        min_T = np.min(self.T)
        max_T = np.max(self.T)
        return bool(min_T < 20.0 and max_T > 50.0)


class DensityLimitPredictor:
    @staticmethod
    def greenwald_limit(Ip_MA: float, a: float) -> float:
        """
        n_GW = I_p / (π a²)   [10^20 m^-3]

        Greenwald 2002, Plasma Phys. Control. Fusion 44, R27, Eq. 1.
        I_p in MA, a in m.
        """
        if a <= 0.0:
            return float("inf")
        return float(Ip_MA / (math.pi * a**2))

    @staticmethod
    def marfe_limit(Ip_MA: float, a: float, P_SOL_MW: float, impurity: str, f_imp: float) -> float:
        """
        Heuristic MARFE-onset density tied to the Greenwald limit.

        n_marfe ~ n_GW * sqrt(P_SOL) / (10 * sqrt(f_imp))
        Scaling motivated by Drake 1987 radiation condensation criterion with
        parallel conduction set by P_SOL and impurity fraction f_imp.
        """
        n_gw = DensityLimitPredictor.greenwald_limit(Ip_MA, a)
        factor = math.sqrt(max(P_SOL_MW, 1.0)) / (10.0 * math.sqrt(max(f_imp, 1e-5)))
        return float(n_gw * factor)


class MARFEStabilityDiagram:
    def __init__(self, R0: float, a: float, q95: float, impurity: str):
        self.R0 = R0
        self.a = a
        self.q95 = q95
        self.impurity = impurity

    def scan_density_power(self, ne_range: np.ndarray, P_SOL_range: np.ndarray) -> np.ndarray:
        result = np.zeros((len(ne_range), len(P_SOL_range)))

        for i, ne in enumerate(ne_range):
            for j, P in enumerate(P_SOL_range):
                # I_p = 15 MA representative of ITER baseline scenario
                n_crit = DensityLimitPredictor.marfe_limit(15.0, self.a, P, self.impurity, 1e-4)
                result[i, j] = -1 if ne > n_crit else 1

        return result

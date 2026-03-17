# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# Contact: protoscience@anulum.li
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ─── Kim-Diamond 2003 predator-prey coefficients ────────────────────────────
# Kim & Diamond, Phys. Rev. Lett. 90, 185006 (2003), Eqs. (1–3).
# Turbulence (epsilon) is the prey; zonal flow (V_ZF) is the predator.
#
#   dε/dt = γ_L(p/p₀) ε − α₁ ε² − α₂ ε V²  + S_bg
#   dV/dt = α₃ ε V    − γ_damp V
#   dp/dt = Q          − p / τ_E(ε)
#
# Default coefficients representative of a medium-sized tokamak (cf. Table I).
_KD_GAMMA_L: float = 5e4  # s⁻¹  turbulence linear growth rate
_KD_ALPHA1: float = 1e-4  # turbulence self-saturation (ε²)
_KD_ALPHA2: float = 2e-8  # ZF suppression coupling (ε V²)
_KD_ALPHA3: float = 1e-8  # ZF excitation by turbulence (ε V)
_KD_GAMMA_DAMP: float = 1e3  # s⁻¹  ZF viscous damping rate

# Background noise drive ensures no trivial ε→0 fixed point (Kim & Diamond 2003, Sec. II).
_S_BG: float = 1e2

# Pressure normalisation used in the dimensionless growth rate factor (p/p₀).
_P0: float = 10.0

# ─── Martin 2008 scaling constants ──────────────────────────────────────────
# Martin et al., J. Phys.: Conf. Ser. 123, 012033 (2008), Eq. (1).
# P_LH [MW] = C_M * n₂₀^α_n * B_T^α_B * S^α_S
# where n₂₀ = n_e / 10²⁰ m⁻³, B_T in T, S in m².
_MARTIN_C: float = 0.0488  # prefactor [MW · (10²⁰ m⁻³)^{−α_n} · T^{−α_B} · m^{−2α_S}]
_MARTIN_ALPHA_N: float = 0.717  # density exponent
_MARTIN_ALPHA_B: float = 0.803  # field exponent
_MARTIN_ALPHA_S: float = 0.941  # surface area exponent

# ─── Ryter 2014 low-density branch ──────────────────────────────────────────
# Ryter et al., Nucl. Fusion 54, 083003 (2014), Fig. 3.
# Below n_min ≈ 0.4 n_GW the threshold turns up again (U-shaped curve).
# The branch is approximated as P_LH_low = P_LH_Martin * (n_min / n_e)².
_RYTER_LOW_DENS_FRAC: float = 0.4  # n_min / n_GW


@dataclass
class PredatorPreyResult:
    epsilon_trace: np.ndarray
    V_ZF_trace: np.ndarray
    p_trace: np.ndarray
    time: np.ndarray
    regime: str


class PredatorPreyModel:
    """Zonal-flow turbulence model — Kim & Diamond (2003)."""

    def __init__(
        self,
        gamma_L: float = _KD_GAMMA_L,
        alpha1: float = _KD_ALPHA1,
        alpha2: float = _KD_ALPHA2,
        alpha3: float = _KD_ALPHA3,
        gamma_damp: float = _KD_GAMMA_DAMP,
    ):
        self.gamma_L = gamma_L
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.gamma_damp = gamma_damp

    def confinement_time(self, epsilon: float) -> float:
        # τ_E = τ_E0 / (1 + C·ε); degradation with turbulence level.
        tau_E0 = 1.0  # s, normalisation
        C = 1e-4
        return tau_E0 / (1.0 + C * max(0.0, epsilon))

    def zonal_flow_growth_rate(self) -> float:
        """Characteristic ZF growth rate γ_ZF ≡ α₃ / (α₁ γ_damp) * γ_L² — Kim & Diamond (2003), Eq. 6.

        Used by IPhaseFrequency to estimate the I-phase limit-cycle frequency.
        """
        # Saturation turbulence level at ZF onset: ε* = γ_damp / α₃
        eps_star = self.gamma_damp / self.alpha3
        # γ_ZF ≈ α₃ ε* − γ_damp evaluated at ε = 2 ε* (ZF-growing regime).
        return float(self.alpha3 * eps_star)

    def step(self, state: np.ndarray, dt: float, Q_heating: float) -> np.ndarray:
        eps, V, p = state
        eps = max(0.0, eps)
        V = max(0.0, V)
        p = max(0.0, p)

        d_eps = self.gamma_L * (p / _P0) * eps - self.alpha1 * eps**2 - self.alpha2 * eps * V**2 + _S_BG
        d_V = self.alpha3 * eps * V - self.gamma_damp * V
        tau_e = self.confinement_time(eps)
        d_p = Q_heating - p / tau_e

        return np.maximum(state + np.array([d_eps, d_V, d_p]) * dt, 0.0)

    def evolve(self, Q_heating: float, t_span: tuple[float, float], dt: float) -> PredatorPreyResult:
        n_steps = int((t_span[1] - t_span[0]) / dt)
        t_arr = np.linspace(t_span[0], t_span[1], n_steps)

        state = np.array([1e4, 1.0, 1.0])

        eps_trace = np.zeros(n_steps)
        V_trace = np.zeros(n_steps)
        p_trace = np.zeros(n_steps)

        for i in range(n_steps):
            state = self.step(state, dt, Q_heating)
            eps_trace[i] = state[0]
            V_trace[i] = state[1]
            p_trace[i] = state[2]

        eps_final, V_final = state[0], state[1]
        regime = "H_MODE" if (V_final > 100.0 and eps_final < 5e4) else "L_MODE"

        return PredatorPreyResult(eps_trace, V_trace, p_trace, t_arr, regime)


class LHTrigger:
    def __init__(self, model: PredatorPreyModel):
        self.model = model

    def find_threshold(self, Q_range: np.ndarray) -> float:
        """Bisect Q_range to find onset of L→H bifurcation."""
        for Q in Q_range:
            res = self.model.evolve(Q, (0.0, 1.0), 0.001)
            if res.regime == "H_MODE":
                return float(Q)
        return float(Q_range[-1])


class MartinThreshold:
    """L-H power threshold scaling — Martin et al. (2008)."""

    @staticmethod
    def power_threshold_MW(ne_19: float, B_T: float, S_m2: float) -> float:
        """P_LH [MW] via Martin scaling.

        Martin et al., J. Phys.: Conf. Ser. 123, 012033 (2008), Eq. (1).

        Parameters
        ----------
        ne_19:
            Line-averaged electron density in units of 10¹⁹ m⁻³.
        B_T:
            Toroidal field in T.
        S_m2:
            Plasma surface area in m².

        Notes
        -----
        The published fit uses n₂₀ = n_e / 10²⁰ m⁻³.  Since ne_19 is in
        10¹⁹ m⁻³, n₂₀ = ne_19 / 10.
        """
        if ne_19 <= 0 or B_T <= 0 or S_m2 <= 0:
            return 0.0
        n20 = ne_19 / 10.0
        return float(_MARTIN_C * (n20**_MARTIN_ALPHA_N) * (B_T**_MARTIN_ALPHA_B) * (S_m2**_MARTIN_ALPHA_S))

    @staticmethod
    def power_threshold_with_low_density_branch_MW(
        ne_19: float,
        B_T: float,
        S_m2: float,
        I_p_MA: float,
        a_m: float,
    ) -> float:
        """P_LH including the low-density upswing branch.

        Below n_min ≈ 0.4 n_GW the threshold increases as (n_min/n_e)²
        (Ryter et al., Nucl. Fusion 54, 083003 (2014), Fig. 3).

        Parameters
        ----------
        ne_19:
            Line-averaged electron density in units of 10¹⁹ m⁻³.
        B_T:
            Toroidal field in T.
        S_m2:
            Plasma surface area in m².
        I_p_MA:
            Plasma current in MA (for Greenwald density).
        a_m:
            Minor radius in m (for Greenwald density).

        Notes
        -----
        Greenwald density: n_GW [10²⁰ m⁻³] = I_p [MA] / (π a²) [MA m⁻²].
        Converted to 10¹⁹ m⁻³: n_GW_19 = 10 * I_p / (π a²).
        """
        p_martin = MartinThreshold.power_threshold_MW(ne_19, B_T, S_m2)
        if I_p_MA <= 0 or a_m <= 0:
            return p_martin

        n_gw_19 = 10.0 * I_p_MA / (np.pi * a_m**2)
        n_min_19 = _RYTER_LOW_DENS_FRAC * n_gw_19

        if ne_19 < n_min_19:
            # U-shaped upswing: P_LH_low = P_LH_Martin * (n_min / n_e)²
            return p_martin * (n_min_19 / ne_19) ** 2

        return p_martin


class IPhaseDetector:
    """Detects I-phase limit-cycle oscillations in turbulence traces."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

    def detect(self, epsilon_trace: np.ndarray) -> bool:
        """True if recent trace shows relative std > 10 %."""
        if len(epsilon_trace) < self.window_size:
            return False
        recent = epsilon_trace[-self.window_size :]
        mean_val = np.mean(recent)
        std_val = np.std(recent)
        return bool(mean_val > 0 and std_val / mean_val > 0.1)


class IPhaseFrequency:
    """I-phase limit-cycle frequency estimate.

    Tynan et al., Nucl. Fusion 53, 073053 (2013), Sec. 4.
    f_I ≈ γ_ZF / (2π), where γ_ZF is the zonal-flow linear growth rate
    evaluated at the turbulent saturation level.
    """

    @staticmethod
    def estimate_hz(model: PredatorPreyModel) -> float:
        """Return f_I in Hz."""
        gamma_zf = model.zonal_flow_growth_rate()
        return float(gamma_zf / (2.0 * np.pi))


class LHTransitionController:
    def __init__(self, model: PredatorPreyModel, Q_target: float):
        self.model = model
        self.Q_target = Q_target
        self.detector = IPhaseDetector()

    def step(self, epsilon_measured: float, Q_current: float, dt: float) -> float:
        """Ramp Q toward Q_target; hold when H-mode is confirmed."""
        if epsilon_measured < 5e4:
            return self.Q_target
        return Q_current + 10.0 * dt

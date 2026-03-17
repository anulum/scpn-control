# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: protoscience@anulum.li
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from scpn_control.core.impurity_transport import CoolingCurve

# ── Townsend ionisation coefficients for D₂ ───────────────────────────────
# Lieberman & Lichtenberg (2005), "Principles of Plasma Discharges and
# Materials Processing", 2nd ed., Ch. 14.3, Table 14.1 (deuterium gas).
# A = first Townsend coefficient  [cm⁻¹ Torr⁻¹] → converted to [Pa⁻¹ m⁻¹]
# B = reduced electric-field threshold [V cm⁻¹ Torr⁻¹] → [V Pa⁻¹ m⁻¹]
# γ_SE = secondary-electron emission yield [dimensionless]
# SI conversion: 1 Torr = 133.322 Pa, 1 cm = 0.01 m
#   A_SI = 15 / (133.322 × 0.01) = 11.25 Pa⁻¹ m⁻¹
#   B_SI = 365 / (133.322 × 0.01) = 273.7 V Pa⁻¹ m⁻¹
_A_TOWNSEND = 11.25  # Pa⁻¹ m⁻¹ — Lieberman 2005 Table 14.1
_B_TOWNSEND = 273.7  # V Pa⁻¹ m⁻¹ — Lieberman 2005 Table 14.1
_GAMMA_SE = 0.01  # secondary-electron yield, D₂ — Lieberman 2005 Table 14.1

# Pre-computed log term that appears in Paschen formulas
# Lieberman 2005 Eq. 14.3.2: ln(1/γ_SE + 1)
_LN_GAMMA = math.log(1.0 / _GAMMA_SE + 1.0)  # ≈ 4.615

# ── D₂ ionisation energy (Janev et al. 1987) ──────────────────────────────
# Janev et al. (1987), "Elementary Processes in Hydrogen-Helium Plasmas",
# Springer, Ch. 2.  E_iz = 15.4 eV for molecular D₂ effective ionisation.
# σ_0 = 1e-20 m² — Janev 1987 cross-section scale for rate-coefficient fit.
_E_IZ_EV = 15.4  # eV — D₂ ionisation energy, Janev 1987 Ch. 2
_SIGMA_0 = 1e-20  # m² — Janev 1987 cross-section scale

# Physical constants
_KB = 1.38e-23  # J K⁻¹
_ME = 9.109e-31  # kg
_ECHARGE = 1.6e-19  # C


class PaschenBreakdown:
    """Paschen curve breakdown model for D₂ gas.

    Paschen (1889), Wied. Ann. 37, 69.
    Quantitative coefficients from Lieberman & Lichtenberg (2005), Ch. 14.3.
    Breakdown voltage: V_bd(pd) = B·pd / (ln(A·pd) - ln(ln(1/γ+1)))
    Minimum voltage:  V_min = e·B/A · ln(1/γ+1)  at  (pd)_min = e/A · ln(1/γ+1)
    Reference: Lieberman 2005, Eq. 14.3.2.
    """

    def __init__(self, gas: str = "D2", R0: float = 6.2, a: float = 2.0):
        self.gas = gas
        self.R0 = R0
        self.a = a
        # Townsend coefficients — Lieberman 2005 Table 14.1, SI units
        self.A = _A_TOWNSEND
        self.B_V = _B_TOWNSEND
        self.gamma_se = _GAMMA_SE

    # V_min and (pd)_min — Lieberman 2005 Eq. 14.3.2
    @property
    def v_paschen_min(self) -> float:
        """Paschen-curve minimum breakdown voltage [V]."""
        return math.e * self.B_V / self.A * _LN_GAMMA

    @property
    def pd_at_minimum(self) -> float:
        """Pressure·distance at Paschen minimum [Pa·m]."""
        return math.e / self.A * _LN_GAMMA

    def breakdown_voltage(self, p_Pa: float, connection_length_m: float) -> float:
        """Paschen breakdown voltage [V].

        Lieberman 2005 Eq. 14.3.1:
            V_bd = B·(pd) / (ln(A·pd) - C₂)
        where C₂ = ln(ln(1/γ+1)).
        """
        pd = p_Pa * connection_length_m
        if pd <= 0.0:
            return float("inf")
        log_arg = self.A * pd
        if log_arg <= 1.0:
            return float("inf")
        denom = math.log(log_arg) - math.log(_LN_GAMMA)
        if denom <= 0.0:
            return float("inf")
        return float(self.B_V * pd / denom)

    def is_breakdown(self, V_loop: float, p_Pa: float, connection_length_m: float = 100.0) -> bool:
        return V_loop > self.breakdown_voltage(p_Pa, connection_length_m)

    def paschen_curve(self, p_range: np.ndarray, connection_length_m: float = 100.0) -> np.ndarray:
        return np.array([self.breakdown_voltage(p, connection_length_m) for p in p_range])

    def optimal_prefill_pressure(self, V_loop_max: float, connection_length_m: float = 100.0) -> float:
        # (pd)_min from Paschen minimum — Lieberman 2005 Eq. 14.3.2
        pd_opt = self.pd_at_minimum
        return float(pd_opt / connection_length_m)


@dataclass
class AvalancheResult:
    ne_trace: np.ndarray
    Te_trace: np.ndarray
    time_to_full_ionization_ms: float


class TownsendAvalanche:
    """Townsend avalanche and burn-through pre-ionisation model.

    Energy balance follows Lieberman & Lichtenberg (2005), Ch. 14.4:
        d(n_e)/dt = n_e · k_iz · n_0
    Ionisation rate coefficient (Maxwellian electrons):
        k_iz = σ_0 · √(8kT_e/(πm_e)) · exp(−E_iz/T_e)
    Reference: Janev et al. (1987), Ch. 2, rate coefficient parameterisation.
    """

    def __init__(self, V_loop: float, p_Pa: float, R0: float, a: float):
        self.V_loop = V_loop
        self.p_Pa = p_Pa
        self.R0 = R0
        self.a = a
        self.E_par = V_loop / (2.0 * math.pi * R0)
        # Ideal-gas neutral density at 300 K — n₀ = p/(k_B T)
        self.n_neutral = p_Pa / (_KB * 300.0)

    def ionization_rate(self, Te_eV: float) -> float:
        """Rate coefficient k_iz [m³ s⁻¹] × n_neutral [m⁻³] → [s⁻¹].

        k_iz = σ_0 · √(8kT_e/(πm_e)) · exp(−E_iz/T_e)
        Janev et al. (1987) Ch. 2; σ_0 = 1e-20 m², E_iz = 15.4 eV for D₂.
        """
        if Te_eV < 0.1:
            return 0.0
        Te_J = Te_eV * _ECHARGE
        v_th = math.sqrt(8.0 * Te_J / (math.pi * _ME))  # mean thermal speed
        k_iz = _SIGMA_0 * v_th * math.exp(-_E_IZ_EV / max(Te_eV, 0.1))
        return float(self.n_neutral * k_iz)

    def evolve(self, dt: float, n_steps: int) -> AvalancheResult:
        ne = 1e13  # seed density [m⁻³]
        Te = 1.0  # electron temperature [eV]

        ne_trace = np.zeros(n_steps)
        Te_trace = np.zeros(n_steps)
        full_ion_time = -1.0

        for i in range(n_steps):
            t = i * dt
            nu_ion = self.ionization_rate(Te)
            ne = min(ne + ne * nu_ion * dt, self.n_neutral)

            # Ohmic heating: P = E²/η, Spitzer η ~ T_e^{-3/2}
            # Lieberman 2005 §1.4: η ≈ 1e-4 · T_e[eV]^{-3/2} Ω·m
            eta = 1e-4 / max(Te, 0.1) ** 1.5
            P_ohmic = self.E_par**2 / eta

            # Inelastic energy loss: _E_IZ_EV eV per ionisation event
            # Lieberman 2005 §14.4, energy balance Eq. 14.4.3
            P_loss = nu_ion * _E_IZ_EV * _ECHARGE * ne

            dTe_J = (P_ohmic - P_loss) * dt / max(ne, 1e-6)
            Te = min(max(Te + dTe_J / _ECHARGE, 0.5), 10.0)

            ne_trace[i] = ne
            Te_trace[i] = Te

            if ne >= 0.99 * self.n_neutral and full_ion_time < 0.0:
                full_ion_time = t * 1000.0

        return AvalancheResult(ne_trace, Te_trace, full_ion_time)


@dataclass
class BurnThroughResult:
    Te_trace: np.ndarray
    success: bool
    time_to_burn_through_ms: float


class BurnThrough:
    """Burn-through energy balance model.

    Criterion: P_ohmic > P_radiation_barrier.
    Energy balance: 3/2 · n_e · dT_e/dt = P_ohmic − P_rad
    Reference: Wesson (2011), "Tokamaks", 4th ed., §6.4, Eq. 6.4.1–6.4.3.
    Spitzer resistivity: η = 1.65×10⁻⁹ · Z_eff · ln_Λ / T_e[keV]^{3/2}
    Reference: Wesson 2011 Eq. 2.5.4.
    """

    # Spitzer prefactor — Wesson 2011 Eq. 2.5.4
    _ETA_SPITZER_PREFACTOR = 1.65e-9  # Ω·m keV^{3/2}
    _Z_EFF_DEFAULT = 1.5
    _LN_LAMBDA_DEFAULT = 17.0  # Wesson 2011 §2.12, mid-range tokamak value

    def __init__(self, R0: float, a: float, B0: float, V_loop: float):
        self.R0 = R0
        self.a = a
        self.B0 = B0
        self.V_loop = V_loop
        self.E_par = V_loop / (2.0 * math.pi * R0)

    def ohmic_power(self, Te_eV: float, ne_19: float, Ip_kA: float) -> float:
        # Spitzer resistivity — Wesson 2011 Eq. 2.5.4
        T_keV = max(Te_eV / 1000.0, 1e-6)
        eta = self._ETA_SPITZER_PREFACTOR * self._Z_EFF_DEFAULT * self._LN_LAMBDA_DEFAULT / T_keV**1.5
        R_p = eta * 2.0 * math.pi * self.R0 / (math.pi * self.a**2)
        return float((Ip_kA * 1e3) ** 2 * R_p)

    def radiation_barrier(self, Te_eV: float, ne_19: float, f_imp: float, impurity: str = "C") -> float:
        curve = CoolingCurve(impurity)
        L_z = curve.L_z(np.array([Te_eV]))[0]
        ne = ne_19 * 1e19
        V = 2.0 * math.pi**2 * self.R0 * self.a**2
        return float(ne * ne * f_imp * L_z * V)

    def burn_through_condition(
        self, Te_eV: float, ne_19: float, Ip_kA: float, f_imp: float, impurity: str = "C"
    ) -> bool:
        return self.ohmic_power(Te_eV, ne_19, Ip_kA) > self.radiation_barrier(Te_eV, ne_19, f_imp, impurity)

    def critical_impurity_fraction(self, Te_eV: float, ne_19: float, Ip_kA: float, impurity: str) -> float:
        curve = CoolingCurve(impurity)
        L_z = curve.L_z(np.array([Te_eV]))[0]
        if L_z <= 0.0:
            return 1.0
        P_oh = self.ohmic_power(Te_eV, ne_19, Ip_kA)
        ne = ne_19 * 1e19
        V = 2.0 * math.pi**2 * self.R0 * self.a**2
        n_imp_crit = P_oh / (ne * L_z * V)
        return float(n_imp_crit / ne)

    def evolve(self, ne_19: float, f_imp: float, dt: float, n_steps: int, impurity: str = "C") -> BurnThroughResult:
        # Energy equation: 3/2 n_e V dT_e = (P_ohmic − P_rad) dt
        # Wesson 2011 §6.4, Eq. 6.4.3
        Te = 5.0  # eV, pre-ionised seed value
        Ip = 100.0  # kA
        ne = ne_19 * 1e19
        V = 2.0 * math.pi**2 * self.R0 * self.a**2

        Te_trace = np.zeros(n_steps)
        success = False
        time_to_bt = -1.0

        for i in range(n_steps):
            t = i * dt
            P_oh = self.ohmic_power(Te, ne_19, Ip)
            P_rad = self.radiation_barrier(Te, ne_19, f_imp, impurity)
            dTe_J = (P_oh - P_rad) * dt / (1.5 * ne * V)
            Te += dTe_J / _ECHARGE

            if Te > 20.0:
                Ip += 1000.0 * dt  # 1 MA s⁻¹ ramp

            Te_trace[i] = Te

            if Te > 100.0 and not success:
                success = True
                time_to_bt = t * 1000.0

        return BurnThroughResult(Te_trace, success, time_to_bt)


@dataclass
class StartupResult:
    breakdown_time_ms: float
    burn_through_time_ms: float
    Ip_at_100ms_kA: float
    Te_at_100ms_eV: float
    success: bool


class StartupSequence:
    def __init__(self, R0: float, a: float, B0: float, V_loop: float, p_prefill_Pa: float, f_imp: float = 0.01):
        self.R0 = R0
        self.a = a
        self.B0 = B0
        self.V_loop = V_loop
        self.p_prefill_Pa = p_prefill_Pa
        self.f_imp = f_imp

    def run(self) -> StartupResult:
        conn = 100.0
        paschen = PaschenBreakdown("D2", self.R0, self.a)

        if not paschen.is_breakdown(self.V_loop, self.p_prefill_Pa, conn):
            return StartupResult(-1.0, -1.0, 0.0, 0.0, False)

        ava = TownsendAvalanche(self.V_loop, self.p_prefill_Pa, self.R0, self.a)
        ava_res = ava.evolve(1e-4, 50)

        bt = BurnThrough(self.R0, self.a, self.B0, self.V_loop)
        ne_19 = 0.2  # ~2×10¹⁸ m⁻³ from 1 mPa prefill at 300 K
        bt_res = bt.evolve(ne_19, self.f_imp, 1e-3, 200, impurity="C")

        return StartupResult(
            breakdown_time_ms=ava_res.time_to_full_ionization_ms,
            burn_through_time_ms=bt_res.time_to_burn_through_ms,
            Ip_at_100ms_kA=100.0 + 1000.0 * 0.1 if bt_res.success else 0.0,
            Te_at_100ms_eV=bt_res.Te_trace[-1],
            success=bt_res.success,
        )


class StartupPhase(Enum):
    GAS_PUFF = auto()
    BREAKDOWN = auto()
    BURN_THROUGH = auto()
    RAMP = auto()


@dataclass
class StartupCommand:
    V_loop: float
    gas_puff_rate: float
    phase: StartupPhase


class StartupController:
    def __init__(self, V_loop_max: float, gas_puff_max: float):
        self.V_loop_max = V_loop_max
        self.gas_puff_max = gas_puff_max
        self.phase = StartupPhase.GAS_PUFF

    def step(self, ne: float, Te: float, Ip: float, t: float, dt: float) -> StartupCommand:
        if self.phase == StartupPhase.GAS_PUFF:
            if t > 0.1:
                self.phase = StartupPhase.BREAKDOWN
        elif self.phase == StartupPhase.BREAKDOWN:
            if ne > 1e18:
                self.phase = StartupPhase.BURN_THROUGH
        elif self.phase == StartupPhase.BURN_THROUGH:
            if Te > 50.0:
                self.phase = StartupPhase.RAMP

        if self.phase == StartupPhase.GAS_PUFF:
            return StartupCommand(0.0, self.gas_puff_max, self.phase)
        elif self.phase in (StartupPhase.BREAKDOWN, StartupPhase.BURN_THROUGH):
            return StartupCommand(self.V_loop_max, 0.0, self.phase)
        else:
            return StartupCommand(self.V_loop_max * 0.5, self.gas_puff_max * 0.1, self.phase)

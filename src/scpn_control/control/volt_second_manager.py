# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
from __future__ import annotations

# Volt-second balance: ∫ V_loop dt = L_p dI_p + R_p I_p dt
# (inductive + resistive consumption).
# Reference: Wesson 2011, Tokamaks 4th ed., Eq. 3.7.4.
#
# Ejima coefficient: ΔΨ_startup = C_Ejima · μ₀ · R₀ · I_p.
# C_Ejima ≈ 0.4 for ITER.
# Reference: Ejima et al. 1982, Nucl. Fusion 22, 1313.
#
# Flat-top duration: τ_flat = (Ψ_avail − Ψ_startup) / (R_p I_p).
# Reference: ITER Physics Basis 1999, Nucl. Fusion 39, 2137, §3.

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

# Ejima et al. 1982, Nucl. Fusion 22, 1313 — startup flux coefficient.
# C_Ejima ≈ 0.4 is the ITER design value.
C_EJIMA: float = 0.4  # dimensionless

# Vacuum permeability, SI.
MU_0: float = 4.0 * math.pi * 1e-7  # H/m


@dataclass
class FluxStatus:
    flux_consumed_Vs: float
    flux_remaining_Vs: float
    estimated_remaining_time_s: float
    fraction_consumed: float


@dataclass
class FluxReport:
    ramp_flux: float
    flat_top_flux: float
    ramp_down_flux: float
    total_flux: float
    within_budget: bool
    margin_Vs: float


class FluxBudget:
    """Volt-second budget tracker.

    Implements the balance:
        ∫ V_loop dt = L_p dI_p + R_p I_p dt
    Reference: Wesson 2011, Tokamaks 4th ed., Eq. 3.7.4.
    """

    def __init__(self, Phi_CS_Vs: float, L_plasma_uH: float, R_plasma_uOhm: float):
        self.Phi_CS_Vs = Phi_CS_Vs
        self.L_plasma_H = L_plasma_uH * 1e-6
        self.R_plasma_Ohm = R_plasma_uOhm * 1e-6

    def inductive_flux(self, Ip_MA: float) -> float:
        """L_p · I_p — inductive volt-second consumption.

        Wesson 2011, Tokamaks 4th ed., Eq. 3.7.4 (first term).
        """
        return self.L_plasma_H * (Ip_MA * 1e6)

    def resistive_flux_ramp(self, Ip_trace: np.ndarray, dt: float) -> float:
        """∫ R_p I_p dt — resistive volt-second consumption during ramp.

        Wesson 2011, Tokamaks 4th ed., Eq. 3.7.4 (second term).
        """
        return float(np.sum(self.R_plasma_Ohm * (Ip_trace * 1e6) * dt))

    def ejima_startup_flux(self, R0_m: float, Ip_MA: float) -> float:
        """Startup flux via Ejima coefficient: ΔΨ = C_Ejima · μ₀ · R₀ · I_p.

        Ejima et al. 1982, Nucl. Fusion 22, 1313, Eq. 2.
        C_EJIMA = 0.4 is the ITER design value.
        """
        return C_EJIMA * MU_0 * R0_m * (Ip_MA * 1e6)

    def remaining_flux(self, Ip_MA: float, ramp_flux: float) -> float:
        ind = self.inductive_flux(Ip_MA)
        consumed = ind + ramp_flux
        return max(0.0, self.Phi_CS_Vs - consumed)

    def max_flattop_duration(self, Ip_MA: float, I_bs_MA: float, ramp_flux: float) -> float:
        """τ_flat = (Ψ_avail − Ψ_startup) / (R_p I_p).

        ITER Physics Basis 1999, Nucl. Fusion 39, 2137, §3.
        """
        rem = self.remaining_flux(Ip_MA, ramp_flux)
        I_driven = max((Ip_MA - I_bs_MA) * 1e6, 1e-6)
        return float(rem / (self.R_plasma_Ohm * I_driven))


class VoltSecondOptimizer:
    def __init__(self, flux_budget: FluxBudget, transport_model: Callable | None = None):
        self.budget = flux_budget
        self.transport_model = transport_model

    def optimize_ramp(self, Ip_target_MA: float, t_ramp_max: float, n_segments: int = 10) -> np.ndarray:
        t_arr = np.linspace(0, t_ramp_max, n_segments)
        Ip_trace = Ip_target_MA * (t_arr / t_ramp_max)
        return Ip_trace


class BootstrapCurrentEstimate:
    @staticmethod
    def from_profiles(
        ne: np.ndarray, Te: np.ndarray, Ti: np.ndarray, q: np.ndarray, rho: np.ndarray, R0: float, a: float
    ) -> float:
        """Simplified bootstrap current proxy: I_bs ~ ε^{1/2} · ∫ dp/dr dr.

        Full neoclassical expression in Wesson 2011, Ch. 4.9.
        ε = a/R₀ is the inverse aspect ratio.
        """
        epsilon = a / R0
        p = 2.0 * ne * 1e19 * Te * 1e3 * 1.6e-19
        grad_p = np.gradient(p, rho[1] - rho[0] if len(rho) > 1 else 0.1)

        # J_bs ~ −ε^{1/2} / B_pol · dp/dr  (Wesson 2011, Eq. 4.9.4, rough scaling)
        J_bs_integral = np.sum(-grad_p * math.sqrt(epsilon)) * 1e-5

        I_bs_MA = max(0.0, J_bs_integral * 0.1)
        return float(I_bs_MA)


class FluxConsumptionMonitor:
    def __init__(self, flux_budget: FluxBudget):
        self.budget = flux_budget
        self.consumed = 0.0

    def step(self, Ip: float, V_loop: float, dt: float) -> FluxStatus:
        """Integrate V_loop dt to track consumed volt-seconds.

        Wesson 2011, Tokamaks 4th ed., Eq. 3.7.4 — V_loop drives both
        inductive and resistive flux consumption.
        """
        self.consumed += V_loop * dt
        rem = self.budget.Phi_CS_Vs - self.consumed

        est_time = rem / max(V_loop, 1e-3) if rem > 0 else 0.0
        frac = self.consumed / self.budget.Phi_CS_Vs

        return FluxStatus(
            flux_consumed_Vs=self.consumed,
            flux_remaining_Vs=max(0.0, rem),
            estimated_remaining_time_s=est_time,
            fraction_consumed=frac,
        )


class ScenarioFluxAnalysis:
    def __init__(self, flux_budget: FluxBudget):
        self.budget = flux_budget

    def analyze(self, ramp_dur: float, flat_dur: float, down_dur: float, Ip_MA: float, I_bs_MA: float) -> FluxReport:
        """Decompose total flux consumption into ramp / flat-top / ramp-down.

        Ramp: L_p I_p (inductive) + R_p · 0.5 I_p · t_ramp (resistive at mean current).
        Flat-top: R_p · (I_p − I_bs) · t_flat.
        Ramp-down: resistive loss minus partial inductive recovery.
        Reference: ITER Physics Basis 1999, Nucl. Fusion 39, 2137, §3.
        """
        L_term = self.budget.inductive_flux(Ip_MA)

        R_term_ramp = self.budget.R_plasma_Ohm * (Ip_MA * 1e6 * 0.5) * ramp_dur
        ramp_flux = L_term + R_term_ramp

        flat_flux = self.budget.R_plasma_Ohm * max((Ip_MA - I_bs_MA) * 1e6, 0.0) * flat_dur

        down_flux = self.budget.R_plasma_Ohm * (Ip_MA * 1e6 * 0.5) * down_dur - L_term * 0.5

        tot = ramp_flux + flat_flux + down_flux

        return FluxReport(
            ramp_flux=ramp_flux,
            flat_top_flux=flat_flux,
            ramp_down_flux=down_flux,
            total_flux=tot,
            within_budget=tot <= self.budget.Phi_CS_Vs,
            margin_Vs=self.budget.Phi_CS_Vs - tot,
        )

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Volt-second monitoring and scenario analysis.

"""Online flux monitoring and per-phase scenario accounting."""

from __future__ import annotations

from scpn_control.control.volt_second_core import FluxBudget, FluxReport, FluxStatus, _finite_scalar


class FluxConsumptionMonitor:
    """Online volt-second consumption tracker integrating the loop voltage.

    Parameters
    ----------
    flux_budget
        The available flux budget against which consumption is tracked.
    """

    def __init__(self, flux_budget: FluxBudget):
        self.budget = flux_budget
        self.consumed = 0.0

    def step(self, Ip: float, V_loop: float, dt: float) -> FluxStatus:
        """Integrate V_loop dt and publish only a finite complete status.

        Wesson 2011, Tokamaks 4th ed., Eq. 3.7.4 — V_loop drives both
        inductive and resistive flux consumption.
        """
        _finite_scalar("Ip", Ip, nonnegative=True)
        V_loop = _finite_scalar("V_loop", V_loop, nonnegative=True)
        dt = _finite_scalar("dt", dt, positive=True)
        budget = _finite_scalar("Phi_CS_Vs", self.budget.Phi_CS_Vs, positive=True)
        previous = _finite_scalar("consumed", self.consumed, nonnegative=True)
        increment = _finite_scalar("volt-second increment", V_loop * dt, nonnegative=True)
        consumed = _finite_scalar("consumed", previous + increment, nonnegative=True)
        rem = _finite_scalar("remaining flux", budget - consumed)

        est_time = _finite_scalar("estimated remaining time", rem / max(V_loop, 1e-3) if rem > 0 else 0.0)
        frac = _finite_scalar("fraction consumed", consumed / budget, nonnegative=True)
        self.consumed = consumed

        return FluxStatus(
            flux_consumed_Vs=consumed,
            flux_remaining_Vs=max(0.0, rem),
            estimated_remaining_time_s=est_time,
            fraction_consumed=frac,
        )


class ScenarioFluxAnalysis:
    """Per-phase volt-second budget analysis of a full scenario.

    Parameters
    ----------
    flux_budget
        The available flux budget.
    """

    def __init__(self, flux_budget: FluxBudget):
        self.budget = flux_budget

    def analyze(self, ramp_dur: float, flat_dur: float, down_dur: float, Ip_MA: float, I_bs_MA: float) -> FluxReport:
        """Decompose total flux consumption into ramp / flat-top / ramp-down.

        Ramp: L_p I_p (inductive) + R_p · 0.5 I_p · t_ramp (resistive at mean current).
        Flat-top: R_p · (I_p − I_bs) · t_flat.
        Ramp-down: resistive loss minus partial inductive recovery.
        Reference: ITER Physics Basis 1999, Nucl. Fusion 39, 2137, §3.
        """
        ramp_dur = _finite_scalar("ramp_dur", ramp_dur, nonnegative=True)
        flat_dur = _finite_scalar("flat_dur", flat_dur, nonnegative=True)
        down_dur = _finite_scalar("down_dur", down_dur, nonnegative=True)
        Ip_MA = _finite_scalar("Ip_MA", Ip_MA, positive=True)
        I_bs_MA = _finite_scalar("I_bs_MA", I_bs_MA, nonnegative=True)
        L_term = self.budget.inductive_flux(Ip_MA)

        R_term_ramp = _finite_scalar("ramp resistive flux", self.budget.R_plasma_Ohm * (Ip_MA * 1e6 * 0.5) * ramp_dur)
        ramp_flux = _finite_scalar("ramp flux", L_term + R_term_ramp)

        flat_flux = _finite_scalar(
            "flat-top flux", self.budget.R_plasma_Ohm * max((Ip_MA - I_bs_MA) * 1e6, 0.0) * flat_dur
        )

        down_flux = _finite_scalar(
            "ramp-down flux", self.budget.R_plasma_Ohm * (Ip_MA * 1e6 * 0.5) * down_dur - L_term * 0.5
        )

        tot = _finite_scalar("total flux", ramp_flux + flat_flux + down_flux)
        margin = _finite_scalar("budget margin", self.budget.Phi_CS_Vs - tot)

        return FluxReport(
            ramp_flux=ramp_flux,
            flat_top_flux=flat_flux,
            ramp_down_flux=down_flux,
            total_flux=tot,
            within_budget=tot <= self.budget.Phi_CS_Vs,
            margin_Vs=margin,
        )

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Volt-second management
"""Volt-second flux budgeting, consumption monitoring, and scenario feasibility utilities."""

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
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

# Ejima et al. 1982, Nucl. Fusion 22, 1313 — startup flux coefficient.
# C_Ejima ≈ 0.4 is the ITER design value.
C_EJIMA: float = 0.4  # dimensionless

# Vacuum permeability, SI.
MU_0: float = 4.0 * math.pi * 1e-7  # H/m


def _finite_scalar(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    """Return a finite scalar in the requested numeric domain."""
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and scalar < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return scalar


def _positive_int(name: str, value: object, *, minimum: int = 1) -> int:
    """Require an exact integer at or above the declared minimum."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _finite_profile(
    name: str, values: NDArray[np.float64], *, positive: bool = False, nonnegative: bool = False
) -> NDArray[np.float64]:
    """Require a finite, nonempty one-dimensional profile."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a one-dimensional non-empty profile")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    if positive and np.any(arr <= 0.0):
        raise ValueError(f"{name} must be positive")
    if nonnegative and np.any(arr < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return arr


def _strict_rho(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Require a strictly increasing normalized radial grid."""
    rho = _finite_profile("rho", values, nonnegative=True)
    if rho.size < 2:
        raise ValueError("rho must contain at least two points")
    if rho[0] < 0.0 or rho[-1] > 1.0:
        raise ValueError("rho must stay within the normalised interval [0, 1]")
    if np.any(np.diff(rho) <= 0.0):
        raise ValueError("rho must be strictly increasing")
    return rho


@dataclass
class FluxStatus:
    """Instantaneous volt-second consumption status.

    Attributes
    ----------
    flux_consumed_Vs
        Volt-seconds consumed so far.
    flux_remaining_Vs
        Volt-seconds remaining in the budget.
    estimated_remaining_time_s
        Estimated time to budget exhaustion at the current loop voltage, in
        seconds.
    fraction_consumed
        Fraction of the total budget consumed.
    """

    flux_consumed_Vs: float
    flux_remaining_Vs: float
    estimated_remaining_time_s: float
    fraction_consumed: float


@dataclass
class FluxReport:
    """Per-phase decomposition of scenario volt-second consumption.

    Attributes
    ----------
    ramp_flux
        Volt-seconds consumed during the current ramp-up.
    flat_top_flux
        Volt-seconds consumed during flat-top.
    ramp_down_flux
        Volt-seconds consumed (net) during ramp-down.
    total_flux
        Total volt-seconds consumed across the scenario.
    within_budget
        ``True`` if the total stays within the available budget.
    margin_Vs
        Volt-second margin (budget minus total); negative if over budget.
    """

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
        self.Phi_CS_Vs = _finite_scalar("Phi_CS_Vs", Phi_CS_Vs, positive=True)
        self.L_plasma_H = _finite_scalar(
            "L_plasma_H", _finite_scalar("L_plasma_uH", L_plasma_uH, positive=True) * 1e-6, positive=True
        )
        self.R_plasma_Ohm = _finite_scalar(
            "R_plasma_Ohm", _finite_scalar("R_plasma_uOhm", R_plasma_uOhm, positive=True) * 1e-6, positive=True
        )

    def inductive_flux(self, Ip_MA: float) -> float:
        """L_p · I_p — inductive volt-second consumption.

        Wesson 2011, Tokamaks 4th ed., Eq. 3.7.4 (first term).
        """
        Ip_MA = _finite_scalar("Ip_MA", Ip_MA, nonnegative=True)
        return _finite_scalar("inductive flux", self.L_plasma_H * (Ip_MA * 1e6), nonnegative=True)

    def resistive_flux_ramp(self, Ip_trace: NDArray[np.float64], dt: float) -> float:
        """∫ R_p I_p dt — resistive volt-second consumption during ramp.

        Wesson 2011, Tokamaks 4th ed., Eq. 3.7.4 (second term).
        """
        Ip_trace = _finite_profile("Ip_trace", Ip_trace, nonnegative=True)
        dt = _finite_scalar("dt", dt, positive=True)
        with np.errstate(over="ignore", invalid="ignore"):
            flux = float(np.sum(self.R_plasma_Ohm * (Ip_trace * 1e6) * dt))
        return _finite_scalar("resistive ramp flux", flux, nonnegative=True)

    def ejima_startup_flux(self, R0_m: float, Ip_MA: float) -> float:
        """Startup flux via Ejima coefficient: ΔΨ = C_Ejima · μ₀ · R₀ · I_p.

        Ejima et al. 1982, Nucl. Fusion 22, 1313, Eq. 2.
        C_EJIMA = 0.4 is the ITER design value.
        """
        R0_m = _finite_scalar("R0_m", R0_m, positive=True)
        Ip_MA = _finite_scalar("Ip_MA", Ip_MA, nonnegative=True)
        return _finite_scalar("Ejima startup flux", C_EJIMA * MU_0 * R0_m * (Ip_MA * 1e6), nonnegative=True)

    def remaining_flux(self, Ip_MA: float, ramp_flux: float) -> float:
        """Volt-seconds left after inductive and ramp consumption.

        Parameters
        ----------
        Ip_MA
            Plasma current in MA.
        ramp_flux
            Volt-seconds already consumed during ramp-up; must be non-negative.

        Returns
        -------
        float
            Remaining volt-seconds, floored at zero.
        """
        ramp_flux = _finite_scalar("ramp_flux", ramp_flux, nonnegative=True)
        ind = self.inductive_flux(Ip_MA)
        consumed = _finite_scalar("consumed flux", ind + ramp_flux, nonnegative=True)
        return max(0.0, self.Phi_CS_Vs - consumed)

    def max_flattop_duration(self, Ip_MA: float, I_bs_MA: float, ramp_flux: float) -> float:
        """τ_flat = (Ψ_avail − Ψ_startup) / (R_p I_p).

        ITER Physics Basis 1999, Nucl. Fusion 39, 2137, §3.
        """
        Ip_MA = _finite_scalar("Ip_MA", Ip_MA, positive=True)
        I_bs_MA = _finite_scalar("I_bs_MA", I_bs_MA, nonnegative=True)
        ramp_flux = _finite_scalar("ramp_flux", ramp_flux, nonnegative=True)
        rem = self.remaining_flux(Ip_MA, ramp_flux)
        I_driven = _finite_scalar("driven current", max((Ip_MA - I_bs_MA) * 1e6, 1e-6), positive=True)
        denominator = _finite_scalar("resistive denominator", self.R_plasma_Ohm * I_driven, positive=True)
        return _finite_scalar("maximum flat-top duration", rem / denominator, nonnegative=True)


class VoltSecondOptimizer:
    """Current-ramp planner that respects the volt-second budget.

    Parameters
    ----------
    flux_budget
        The available flux budget.
    transport_model
        Optional transport model used to refine the ramp; the linear default
        ramp does not use it.
    """

    def __init__(self, flux_budget: FluxBudget, transport_model: Callable[..., Any] | None = None):
        self.budget = flux_budget
        self.transport_model = transport_model

    def optimize_ramp(self, Ip_target_MA: float, t_ramp_max: float, n_segments: int = 10) -> NDArray[np.float64]:
        """Plan a current ramp to the target over the allowed ramp time.

        Parameters
        ----------
        Ip_target_MA
            Target plasma current in MA; must be non-negative.
        t_ramp_max
            Maximum ramp duration in seconds; must be positive.
        n_segments
            Number of ramp samples; must be at least 2.

        Returns
        -------
        NDArray[np.float64]
            The plasma-current trace in MA, shape ``(n_segments,)``.
        """
        Ip_target_MA = _finite_scalar("Ip_target_MA", Ip_target_MA, nonnegative=True)
        t_ramp_max = _finite_scalar("t_ramp_max", t_ramp_max, positive=True)
        n_segments = _positive_int("n_segments", n_segments, minimum=2)
        t_arr = np.linspace(0, t_ramp_max, n_segments)
        Ip_trace = Ip_target_MA * (t_arr / t_ramp_max)
        return np.asarray(Ip_trace, dtype=np.float64)

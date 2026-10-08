# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Volt-second analytic configuration and result records
"""Store the bounded volt-second validation configuration and outcomes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypedDict

from scpn_control.control.volt_second_manager import FluxBudget


@dataclass(frozen=True)
class VoltSecondConfig:
    """Pulse and circuit parameters for the volt-second budget."""

    flux_budget_vs: float
    plasma_inductance_uh: float
    plasma_resistance_uohm: float
    major_radius_m: float
    plasma_current_ma: float
    bootstrap_current_ma: float
    ramp_duration_s: float
    flat_duration_s: float
    ramp_down_duration_s: float
    standalone_ramp_flux_vs: float

    def __post_init__(self) -> None:
        """Validate finite circuit, current and duration domains at construction."""
        _positive_float("flux_budget_vs", self.flux_budget_vs)
        _positive_float("plasma_inductance_uh", self.plasma_inductance_uh)
        _positive_float("plasma_resistance_uohm", self.plasma_resistance_uohm)
        _positive_float("major_radius_m", self.major_radius_m)
        _positive_float("plasma_current_ma", self.plasma_current_ma)
        _nonnegative_float("bootstrap_current_ma", self.bootstrap_current_ma)
        _nonnegative_float("ramp_duration_s", self.ramp_duration_s)
        _nonnegative_float("flat_duration_s", self.flat_duration_s)
        _nonnegative_float("ramp_down_duration_s", self.ramp_down_duration_s)
        _nonnegative_float("standalone_ramp_flux_vs", self.standalone_ramp_flux_vs)
        if self.bootstrap_current_ma >= self.plasma_current_ma:
            raise ValueError("bootstrap current must be smaller than the plasma current")

    def budget(self) -> FluxBudget:
        """Build the flux-budget model for this flat-top configuration."""
        return FluxBudget(
            Phi_CS_Vs=self.flux_budget_vs,
            L_plasma_uH=self.plasma_inductance_uh,
            R_plasma_uOhm=self.plasma_resistance_uohm,
        )


def default_config() -> VoltSecondConfig:
    """Build the default parameters for the bounded analytic validation."""
    return VoltSecondConfig(
        flux_budget_vs=300.0,
        plasma_inductance_uh=10.0,
        plasma_resistance_uohm=5.0,
        major_radius_m=6.2,
        plasma_current_ma=15.0,
        bootstrap_current_ma=2.0,
        ramp_duration_s=10.0,
        flat_duration_s=100.0,
        ramp_down_duration_s=10.0,
        standalone_ramp_flux_vs=5.0,
    )


@dataclass(frozen=True)
class DecompositionCheck:
    """Scenario flux-decomposition closed-form agreement."""

    ramp_rel_error: float
    flat_top_rel_error: float
    ramp_down_rel_error: float
    sum_rel_error: float
    margin_abs_error: float
    max_rel_error: float


@dataclass(frozen=True)
class MonitorCheck:
    """Consumption-integrator closed-form agreement."""

    consumed_rel_error: float
    remaining_rel_error: float
    fraction_rel_error: float
    max_rel_error: float


@dataclass(frozen=True)
class RampOptimizerCheck:
    """Linear ramp optimiser agreement."""

    start_abs_error: float
    end_rel_error: float
    spacing_max_rel_error: float
    is_linear: bool


@dataclass(frozen=True)
class ScalingCheck:
    """One flux scaling-law observation."""

    name: str
    measured_ratio: float
    expected_ratio: float
    rel_error: float


@dataclass(frozen=True)
class VoltSecondValidationResult:
    """Outcome of the volt-second flux-budget validation."""

    config: VoltSecondConfig
    inductive_rel_error: float
    ejima_rel_error: float
    resistive_ramp_rel_error: float
    flat_top_closure_rel_error: float
    decomposition: DecompositionCheck
    monitor: MonitorCheck
    ramp_optimizer: RampOptimizerCheck
    scaling: tuple[ScalingCheck, ...]
    max_scaling_rel_error: float
    exact_tol: float
    margin_abs_tol: float
    fluxes_passed: bool
    decomposition_passed: bool
    monitor_passed: bool
    optimizer_passed: bool
    scaling_passed: bool
    passed: bool


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(name: str, value: object) -> float:
    result = _finite_float(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_float(name: str, value: object) -> float:
    result = _finite_float(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


VoltSecondConfig.__module__ = "validation.validate_volt_second"
DecompositionCheck.__module__ = "validation.validate_volt_second"
MonitorCheck.__module__ = "validation.validate_volt_second"
RampOptimizerCheck.__module__ = "validation.validate_volt_second"
ScalingCheck.__module__ = "validation.validate_volt_second"
VoltSecondValidationResult.__module__ = "validation.validate_volt_second"


class RampEvidence(TypedDict):
    """Store the ramp endpoint, spacing errors and linearity declaration."""

    start_abs_error: float
    end_rel_error: float
    spacing_max_rel_error: float
    is_linear: bool


class ScalingEvidence(TypedDict):
    """Store one named linear scaling comparison."""

    name: str
    measured_ratio: float
    expected_ratio: float
    rel_error: float


class VoltSecondEvidence(TypedDict):
    """Describe the complete existing volt-second v1 report."""

    schema_version: str
    generated_utc: str
    target_id: str
    config: dict[str, float]
    exact_tol: float
    margin_abs_tol: float
    inductive_rel_error: float
    ejima_rel_error: float
    resistive_ramp_rel_error: float
    flat_top_closure_rel_error: float
    decomposition: dict[str, float]
    monitor: dict[str, float]
    ramp_optimizer: RampEvidence
    scaling: list[ScalingEvidence]
    max_scaling_rel_error: float
    fluxes_passed: bool
    decomposition_passed: bool
    monitor_passed: bool
    optimizer_passed: bool
    scaling_passed: bool
    passed: bool
    payload_sha256: str

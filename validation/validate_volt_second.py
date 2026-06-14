#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Volt-second flux-budget analytic validation
"""Validate the volt-second flux budget against exact closed forms.

The volt-second manager (``src/scpn_control/control/volt_second_manager.py``)
budgets the central-solenoid flux against the inductive and resistive plasma
consumption that drives a tokamak pulse. Every quantity it exposes — the
inductive flux ``L_p I_p``, the Ejima startup flux ``C_E mu_0 R_0 I_p``, the
resistive ramp integral ``sum R_p I_p dt``, the scenario flux decomposition, the
flat-top duration, and the consumption integrator — is an exact algebraic
closed form, so the validation needs no measured loop-voltage trace and is fully
self-contained.

Exact references checked against the production classes:

1. **Inductive flux.** ``FluxBudget.inductive_flux(I_p) = L_p I_p`` with its
   linear scaling in ``I_p`` and ``L_p``.
2. **Ejima startup flux.** ``ejima_startup_flux(R_0, I_p) = C_E mu_0 R_0 I_p``
   with its linear scaling in ``R_0`` and ``I_p``.
3. **Resistive ramp integral.** ``resistive_flux_ramp = sum R_p I_p dt`` against
   the exact Riemann sum for a constant current trace.
4. **Flat-top budget closure.** At ``tau_flat = (Phi_avail - Phi_startup)/(R_p
   I_drive)`` the flat-top resistive consumption exactly equals the remaining
   flux, ``R_p I_drive tau_flat = Phi_remaining``.
5. **Scenario decomposition.** ``ScenarioFluxAnalysis.analyze`` ramp, flat-top,
   and ramp-down terms, their sum, and the budget margin against their closed
   forms.
6. **Consumption integrator.** ``FluxConsumptionMonitor.step`` accumulates
   ``V_loop dt`` exactly, with the matching remaining flux and consumed
   fraction.
7. **Ramp optimiser.** ``VoltSecondOptimizer.optimize_ramp`` returns a uniform
   linear ramp from zero to the target current.

References:
  Wesson J. (2011) *Tokamaks*, 4th ed., Oxford University Press, Eq. 3.7.4.
  Ejima S. et al. (1982) *Nucl. Fusion* 22, 1313 (startup flux coefficient).
  ITER Physics Basis (1999) *Nucl. Fusion* 39, 2137, §3 (flat-top flux budget).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scpn_control.control.volt_second_manager import (
    C_EJIMA,
    MU_0,
    FluxBudget,
    FluxConsumptionMonitor,
    ScenarioFluxAnalysis,
    VoltSecondOptimizer,
)

VOLT_SECOND_SCHEMA_VERSION = "scpn-control.volt-second-validation.v1"


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
        return FluxBudget(
            Phi_CS_Vs=self.flux_budget_vs,
            L_plasma_uH=self.plasma_inductance_uh,
            R_plasma_uOhm=self.plasma_resistance_uohm,
        )


def default_config() -> VoltSecondConfig:
    """An ITER-like flat-top flux budget with positive flat-top margin."""
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


def inductive_flux_rel_error(config: VoltSecondConfig) -> float:
    """Relative error of ``inductive_flux`` against ``L_p I_p``."""
    budget = config.budget()
    analytic = budget.L_plasma_H * (config.plasma_current_ma * 1e6)
    return float(abs(budget.inductive_flux(config.plasma_current_ma) - analytic) / analytic)


def ejima_flux_rel_error(config: VoltSecondConfig) -> float:
    """Relative error of ``ejima_startup_flux`` against ``C_E mu_0 R_0 I_p``."""
    budget = config.budget()
    analytic = C_EJIMA * MU_0 * config.major_radius_m * (config.plasma_current_ma * 1e6)
    measured = budget.ejima_startup_flux(config.major_radius_m, config.plasma_current_ma)
    return float(abs(measured - analytic) / analytic)


def resistive_ramp_rel_error(config: VoltSecondConfig, *, n_steps: int = 100, dt: float = 0.01) -> float:
    """Relative error of ``resistive_flux_ramp`` against the exact Riemann sum."""
    budget = config.budget()
    trace = np.full(n_steps, config.plasma_current_ma)
    analytic = budget.R_plasma_Ohm * (config.plasma_current_ma * 1e6) * n_steps * dt
    return float(abs(budget.resistive_flux_ramp(trace, dt) - analytic) / analytic)


def flat_top_closure_rel_error(config: VoltSecondConfig) -> float:
    """Relative error of the flat-top budget closure ``R_p I_drive tau_flat = Phi_remaining``."""
    budget = config.budget()
    remaining = budget.remaining_flux(config.plasma_current_ma, config.standalone_ramp_flux_vs)
    tau_flat = budget.max_flattop_duration(
        config.plasma_current_ma, config.bootstrap_current_ma, config.standalone_ramp_flux_vs
    )
    i_drive = (config.plasma_current_ma - config.bootstrap_current_ma) * 1e6
    flat_flux = budget.R_plasma_Ohm * i_drive * tau_flat
    return float(abs(flat_flux - remaining) / remaining)


@dataclass(frozen=True)
class DecompositionCheck:
    """Scenario flux-decomposition closed-form agreement."""

    ramp_rel_error: float
    flat_top_rel_error: float
    ramp_down_rel_error: float
    sum_rel_error: float
    margin_abs_error: float
    max_rel_error: float


def scenario_decomposition_check(config: VoltSecondConfig) -> DecompositionCheck:
    """Verify the ramp, flat-top, and ramp-down flux terms against their closed forms."""
    budget = config.budget()
    analysis = ScenarioFluxAnalysis(budget)
    report = analysis.analyze(
        config.ramp_duration_s,
        config.flat_duration_s,
        config.ramp_down_duration_s,
        config.plasma_current_ma,
        config.bootstrap_current_ma,
    )
    l_term = budget.inductive_flux(config.plasma_current_ma)
    ip_a = config.plasma_current_ma * 1e6
    i_drive = max((config.plasma_current_ma - config.bootstrap_current_ma) * 1e6, 0.0)
    exp_ramp = l_term + budget.R_plasma_Ohm * (ip_a * 0.5) * config.ramp_duration_s
    exp_flat = budget.R_plasma_Ohm * i_drive * config.flat_duration_s
    exp_down = budget.R_plasma_Ohm * (ip_a * 0.5) * config.ramp_down_duration_s - l_term * 0.5
    exp_total = exp_ramp + exp_flat + exp_down

    ramp_err = abs(report.ramp_flux - exp_ramp) / abs(exp_ramp)
    flat_err = abs(report.flat_top_flux - exp_flat) / abs(exp_flat)
    down_err = abs(report.ramp_down_flux - exp_down) / abs(exp_down)
    sum_err = abs(report.total_flux - exp_total) / abs(exp_total)
    margin_err = abs(report.margin_Vs - (budget.Phi_CS_Vs - report.total_flux))
    return DecompositionCheck(
        ramp_rel_error=float(ramp_err),
        flat_top_rel_error=float(flat_err),
        ramp_down_rel_error=float(down_err),
        sum_rel_error=float(sum_err),
        margin_abs_error=float(margin_err),
        max_rel_error=float(max(ramp_err, flat_err, down_err, sum_err)),
    )


@dataclass(frozen=True)
class MonitorCheck:
    """Consumption-integrator closed-form agreement."""

    consumed_rel_error: float
    remaining_rel_error: float
    fraction_rel_error: float
    max_rel_error: float


def monitor_integration_check(
    config: VoltSecondConfig, *, n_steps: int = 50, v_loop: float = 1.5, dt: float = 0.02
) -> MonitorCheck:
    """Verify ``FluxConsumptionMonitor`` integrates ``V_loop dt`` exactly."""
    budget = config.budget()
    monitor = FluxConsumptionMonitor(budget)
    status = None
    for _ in range(n_steps):
        status = monitor.step(config.plasma_current_ma, v_loop, dt)
    assert status is not None
    consumed = v_loop * dt * n_steps
    remaining = budget.Phi_CS_Vs - consumed
    fraction = consumed / budget.Phi_CS_Vs
    consumed_err = abs(status.flux_consumed_Vs - consumed) / consumed
    remaining_err = abs(status.flux_remaining_Vs - remaining) / remaining
    fraction_err = abs(status.fraction_consumed - fraction) / fraction
    return MonitorCheck(
        consumed_rel_error=float(consumed_err),
        remaining_rel_error=float(remaining_err),
        fraction_rel_error=float(fraction_err),
        max_rel_error=float(max(consumed_err, remaining_err, fraction_err)),
    )


@dataclass(frozen=True)
class RampOptimizerCheck:
    """Linear ramp optimiser agreement."""

    start_abs_error: float
    end_rel_error: float
    spacing_max_rel_error: float
    is_linear: bool


def ramp_optimizer_check(config: VoltSecondConfig, *, n_segments: int = 11, t_ramp: float = 5.0) -> RampOptimizerCheck:
    """Verify the ramp optimiser returns a uniform linear ramp to the target current."""
    budget = config.budget()
    optimiser = VoltSecondOptimizer(budget)
    trace = optimiser.optimize_ramp(config.plasma_current_ma, t_ramp, n_segments)
    start_err = abs(float(trace[0]))
    end_err = abs(float(trace[-1]) - config.plasma_current_ma) / config.plasma_current_ma
    spacing = np.diff(trace)
    expected_step = config.plasma_current_ma / (n_segments - 1)
    spacing_err = float(np.max(np.abs(spacing - expected_step)) / expected_step)
    return RampOptimizerCheck(
        start_abs_error=float(start_err),
        end_rel_error=float(end_err),
        spacing_max_rel_error=spacing_err,
        is_linear=bool(spacing_err < 1e-12),
    )


@dataclass(frozen=True)
class ScalingCheck:
    """One flux scaling-law observation."""

    name: str
    measured_ratio: float
    expected_ratio: float
    rel_error: float


def flux_scaling_checks(config: VoltSecondConfig) -> tuple[ScalingCheck, ...]:
    """Verify the inductive and Ejima flux scale linearly in their drivers."""
    import dataclasses

    budget = config.budget()
    base_ind = budget.inductive_flux(config.plasma_current_ma)
    base_ejima = budget.ejima_startup_flux(config.major_radius_m, config.plasma_current_ma)
    specs = (
        (
            "inductive_current_linear",
            budget.inductive_flux(2.0 * config.plasma_current_ma) / base_ind,
            2.0,
        ),
        (
            "inductive_inductance_linear",
            dataclasses.replace(config, plasma_inductance_uh=2.0 * config.plasma_inductance_uh)
            .budget()
            .inductive_flux(config.plasma_current_ma)
            / base_ind,
            2.0,
        ),
        (
            "ejima_major_radius_linear",
            budget.ejima_startup_flux(2.0 * config.major_radius_m, config.plasma_current_ma) / base_ejima,
            2.0,
        ),
        (
            "ejima_current_linear",
            budget.ejima_startup_flux(config.major_radius_m, 2.0 * config.plasma_current_ma) / base_ejima,
            2.0,
        ),
    )
    return tuple(
        ScalingCheck(
            name=name, measured_ratio=ratio, expected_ratio=expected, rel_error=abs(ratio - expected) / expected
        )
        for name, ratio, expected in specs
    )


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


def validate_volt_second(
    *, config: VoltSecondConfig | None = None, exact_tol: float = 1e-9, margin_abs_tol: float = 1e-6
) -> VoltSecondValidationResult:
    """Validate the production volt-second budget against its exact relations.

    Every flux relation, the scenario decomposition, the consumption integrator,
    the ramp optimiser, and the scaling laws must hold to ``exact_tol`` (the
    budget margin to ``margin_abs_tol`` volt-seconds).
    """
    config = config or default_config()

    ind_err = inductive_flux_rel_error(config)
    ejima_err = ejima_flux_rel_error(config)
    res_err = resistive_ramp_rel_error(config)
    closure_err = flat_top_closure_rel_error(config)
    decomposition = scenario_decomposition_check(config)
    monitor = monitor_integration_check(config)
    optimiser = ramp_optimizer_check(config)
    scaling = flux_scaling_checks(config)
    max_scaling = max(check.rel_error for check in scaling)

    fluxes_passed = bool(
        ind_err < exact_tol and ejima_err < exact_tol and res_err < exact_tol and closure_err < exact_tol
    )
    decomposition_passed = bool(
        decomposition.max_rel_error < exact_tol and decomposition.margin_abs_error < margin_abs_tol
    )
    monitor_passed = bool(monitor.max_rel_error < exact_tol)
    optimizer_passed = bool(
        optimiser.is_linear and optimiser.start_abs_error < exact_tol and optimiser.end_rel_error < exact_tol
    )
    scaling_passed = bool(max_scaling < exact_tol)

    passed = bool(fluxes_passed and decomposition_passed and monitor_passed and optimizer_passed and scaling_passed)
    return VoltSecondValidationResult(
        config=config,
        inductive_rel_error=ind_err,
        ejima_rel_error=ejima_err,
        resistive_ramp_rel_error=res_err,
        flat_top_closure_rel_error=closure_err,
        decomposition=decomposition,
        monitor=monitor,
        ramp_optimizer=optimiser,
        scaling=scaling,
        max_scaling_rel_error=max_scaling,
        exact_tol=exact_tol,
        margin_abs_tol=margin_abs_tol,
        fluxes_passed=fluxes_passed,
        decomposition_passed=decomposition_passed,
        monitor_passed=monitor_passed,
        optimizer_passed=optimizer_passed,
        scaling_passed=scaling_passed,
        passed=passed,
    )


def build_evidence(result: VoltSecondValidationResult, *, target_id: str) -> dict[str, Any]:
    """Build a tamper-evident, schema-versioned validation evidence payload."""
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    payload: dict[str, Any] = {
        "schema_version": VOLT_SECOND_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "target_id": target_id,
        "config": {
            "flux_budget_vs": result.config.flux_budget_vs,
            "plasma_inductance_uh": result.config.plasma_inductance_uh,
            "plasma_resistance_uohm": result.config.plasma_resistance_uohm,
            "major_radius_m": result.config.major_radius_m,
            "plasma_current_ma": result.config.plasma_current_ma,
            "bootstrap_current_ma": result.config.bootstrap_current_ma,
            "ramp_duration_s": result.config.ramp_duration_s,
            "flat_duration_s": result.config.flat_duration_s,
            "ramp_down_duration_s": result.config.ramp_down_duration_s,
            "standalone_ramp_flux_vs": result.config.standalone_ramp_flux_vs,
        },
        "exact_tol": result.exact_tol,
        "margin_abs_tol": result.margin_abs_tol,
        "inductive_rel_error": result.inductive_rel_error,
        "ejima_rel_error": result.ejima_rel_error,
        "resistive_ramp_rel_error": result.resistive_ramp_rel_error,
        "flat_top_closure_rel_error": result.flat_top_closure_rel_error,
        "decomposition": {
            "ramp_rel_error": result.decomposition.ramp_rel_error,
            "flat_top_rel_error": result.decomposition.flat_top_rel_error,
            "ramp_down_rel_error": result.decomposition.ramp_down_rel_error,
            "sum_rel_error": result.decomposition.sum_rel_error,
            "margin_abs_error": result.decomposition.margin_abs_error,
            "max_rel_error": result.decomposition.max_rel_error,
        },
        "monitor": {
            "consumed_rel_error": result.monitor.consumed_rel_error,
            "remaining_rel_error": result.monitor.remaining_rel_error,
            "fraction_rel_error": result.monitor.fraction_rel_error,
            "max_rel_error": result.monitor.max_rel_error,
        },
        "ramp_optimizer": {
            "start_abs_error": result.ramp_optimizer.start_abs_error,
            "end_rel_error": result.ramp_optimizer.end_rel_error,
            "spacing_max_rel_error": result.ramp_optimizer.spacing_max_rel_error,
            "is_linear": result.ramp_optimizer.is_linear,
        },
        "scaling": [
            {
                "name": check.name,
                "measured_ratio": check.measured_ratio,
                "expected_ratio": check.expected_ratio,
                "rel_error": check.rel_error,
            }
            for check in result.scaling
        ],
        "max_scaling_rel_error": result.max_scaling_rel_error,
        "fluxes_passed": result.fluxes_passed,
        "decomposition_passed": result.decomposition_passed,
        "monitor_passed": result.monitor_passed,
        "optimizer_passed": result.optimizer_passed,
        "scaling_passed": result.scaling_passed,
        "passed": result.passed,
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Return ``True`` when a payload is well-formed, sealed, and passing."""
    if payload.get("schema_version") != VOLT_SECOND_SCHEMA_VERSION:
        raise ValueError("unsupported volt-second evidence schema_version")
    declared = payload.get("payload_sha256")
    if not _is_sha256(declared):
        raise ValueError("payload_sha256 must be a SHA-256 hex digest")
    if declared != _payload_sha256(payload):
        raise ValueError("payload_sha256 does not match payload")
    return bool(payload.get("passed"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned["payload_sha256"] = ""
    return hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
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


def _write_report(evidence: Mapping[str, Any], json_path: Path) -> None:
    """Persist the sealed JSON evidence and a human-readable Markdown summary."""
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = json_path.with_suffix(".md")
    decomposition = evidence["decomposition"]
    monitor = evidence["monitor"]
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "",
        "# Volt-Second Flux-Budget Validation",
        "",
        f"- Schema: `{evidence['schema_version']}`",
        f"- Generated (UTC): {evidence['generated_utc']}",
        f"- Target: `{evidence['target_id']}`",
        f"- Status: **{'pass' if evidence['passed'] else 'fail'}**",
        "",
        f"## Exact flux relations (relative error, gate < {evidence['exact_tol']:.1e})",
        "",
        "| relation | value |",
        "| --- | --- |",
        f"| inductive flux L_p I_p | {evidence['inductive_rel_error']:.3e} |",
        f"| Ejima startup flux C_E mu0 R0 I_p | {evidence['ejima_rel_error']:.3e} |",
        f"| resistive ramp integral | {evidence['resistive_ramp_rel_error']:.3e} |",
        f"| flat-top budget closure | {evidence['flat_top_closure_rel_error']:.3e} |",
        f"| scenario decomposition (max) | {decomposition['max_rel_error']:.3e} |",
        f"| consumption integrator (max) | {monitor['max_rel_error']:.3e} |",
        f"| flux scaling laws (max) | {evidence['max_scaling_rel_error']:.3e} |",
        "",
        "## Ramp optimiser",
        "",
        f"- linear ramp: {evidence['ramp_optimizer']['is_linear']}; "
        f"endpoint relative error: {evidence['ramp_optimizer']['end_rel_error']:.3e}",
        "",
        "## Budget margin",
        "",
        f"- margin closed-form absolute error: {decomposition['margin_abs_error']:.3e} V s "
        f"(gate < {evidence['margin_abs_tol']:.1e})",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point producing schema-versioned validation evidence."""
    parser = argparse.ArgumentParser(description="Validate the volt-second flux budget against exact closed forms")
    parser.add_argument("--target-id", type=str, default="local-volt-second")
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    parser.add_argument("--report", type=str, default=None, help="write sealed JSON evidence and a Markdown summary")
    args = parser.parse_args(argv)

    result = validate_volt_second()
    evidence = build_evidence(result, target_id=args.target_id)

    if args.report:
        _write_report(evidence, Path(args.report))

    if args.json_out:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print("Volt-second flux-budget validation")
        print(
            f"  fluxes:     inductive={result.inductive_rel_error:.3e} ejima={result.ejima_rel_error:.3e} "
            f"resistive={result.resistive_ramp_rel_error:.3e} closure={result.flat_top_closure_rel_error:.3e} "
            f"{'ok' if result.fluxes_passed else 'FAIL'}"
        )
        print(
            f"  scenario:   decomposition={result.decomposition.max_rel_error:.3e} "
            f"monitor={result.monitor.max_rel_error:.3e} "
            f"{'ok' if result.decomposition_passed and result.monitor_passed else 'FAIL'}"
        )
        print(
            f"  optimiser:  linear={result.ramp_optimizer.is_linear} "
            f"scaling={result.max_scaling_rel_error:.3e} "
            f"{'ok' if result.optimizer_passed and result.scaling_passed else 'FAIL'}"
        )
        print(f"Status: {'pass' if result.passed else 'fail'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

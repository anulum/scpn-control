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

References
----------
  Wesson J. (2011) *Tokamaks*, 4th ed., Oxford University Press, Eq. 3.7.4.
  Ejima S. et al. (1982) *Nucl. Fusion* 22, 1313 (startup flux coefficient).
  ITER Physics Basis (1999) *Nucl. Fusion* 39, 2137, §3 (flat-top flux budget).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from scpn_control.control.volt_second_manager import (
    C_EJIMA,
    MU_0,
    FluxConsumptionMonitor,
    ScenarioFluxAnalysis,
    VoltSecondOptimizer,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation.volt_second_evidence import SCHEMA_VERSION
from validation.volt_second_evidence import build_evidence as build_evidence
from validation.volt_second_evidence import validate_evidence_payload as validate_evidence_payload
from validation.volt_second_models import (
    DecompositionCheck as DecompositionCheck,
)
from validation.volt_second_models import (
    MonitorCheck as MonitorCheck,
)
from validation.volt_second_models import (
    RampOptimizerCheck as RampOptimizerCheck,
)
from validation.volt_second_models import (
    ScalingCheck as ScalingCheck,
)
from validation.volt_second_models import (
    VoltSecondConfig as VoltSecondConfig,
)
from validation.volt_second_models import (
    VoltSecondValidationResult as VoltSecondValidationResult,
)
from validation.volt_second_models import (
    _positive_float,
)
from validation.volt_second_models import (
    default_config as default_config,
)
from validation.volt_second_report import write_report as _write_report

VOLT_SECOND_SCHEMA_VERSION = SCHEMA_VERSION


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
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    dt = _positive_float("dt", dt)
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
    return float(abs(flat_flux - remaining) / (abs(remaining) or budget.Phi_CS_Vs))


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

    ramp_err = abs(report.ramp_flux - exp_ramp) / (abs(exp_ramp) or budget.Phi_CS_Vs)
    flat_err = abs(report.flat_top_flux - exp_flat) / (abs(exp_flat) or budget.Phi_CS_Vs)
    down_err = abs(report.ramp_down_flux - exp_down) / (abs(exp_down) or budget.Phi_CS_Vs)
    sum_err = abs(report.total_flux - exp_total) / (abs(exp_total) or budget.Phi_CS_Vs)
    margin_err = abs(report.margin_Vs - (budget.Phi_CS_Vs - report.total_flux))
    return DecompositionCheck(
        ramp_rel_error=float(ramp_err),
        flat_top_rel_error=float(flat_err),
        ramp_down_rel_error=float(down_err),
        sum_rel_error=float(sum_err),
        margin_abs_error=float(margin_err),
        max_rel_error=float(max(ramp_err, flat_err, down_err, sum_err)),
    )


def monitor_integration_check(
    config: VoltSecondConfig, *, n_steps: int = 50, v_loop: float = 1.5, dt: float = 0.02
) -> MonitorCheck:
    """Verify ``FluxConsumptionMonitor`` integrates ``V_loop dt`` exactly."""
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    dt = _positive_float("dt", dt)
    budget = config.budget()
    monitor = FluxConsumptionMonitor(budget)
    status = monitor.step(config.plasma_current_ma, v_loop, dt)
    for _ in range(1, n_steps):
        status = monitor.step(config.plasma_current_ma, v_loop, dt)
    consumed = v_loop * dt * n_steps
    remaining = budget.Phi_CS_Vs - consumed
    fraction = consumed / budget.Phi_CS_Vs
    consumed_err = abs(status.flux_consumed_Vs - consumed) / (abs(consumed) or budget.Phi_CS_Vs)
    remaining_err = abs(status.flux_remaining_Vs - remaining) / (abs(remaining) or budget.Phi_CS_Vs)
    fraction_err = abs(status.fraction_consumed - fraction) / (abs(fraction) or 1.0)
    return MonitorCheck(
        consumed_rel_error=float(consumed_err),
        remaining_rel_error=float(remaining_err),
        fraction_rel_error=float(fraction_err),
        max_rel_error=float(max(consumed_err, remaining_err, fraction_err)),
    )


def ramp_optimizer_check(config: VoltSecondConfig, *, n_segments: int = 11, t_ramp: float = 5.0) -> RampOptimizerCheck:
    """Verify the ramp optimiser returns a uniform linear ramp to the target current."""
    if isinstance(n_segments, bool) or not isinstance(n_segments, int) or n_segments < 2:
        raise ValueError("n_segments must be an integer of at least two")
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


def validate_volt_second(
    *, config: VoltSecondConfig | None = None, exact_tol: float = 1e-9, margin_abs_tol: float = 1e-6
) -> VoltSecondValidationResult:
    """Validate the production volt-second budget against its exact relations.

    Every flux relation, the scenario decomposition, the consumption integrator,
    the ramp optimiser, and the scaling laws must hold to ``exact_tol`` (the
    budget margin to ``margin_abs_tol`` volt-seconds).
    """
    exact_tol = _positive_float("exact_tol", exact_tol)
    margin_abs_tol = _positive_float("margin_abs_tol", margin_abs_tol)
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


def _threshold(text: str) -> float:
    """Read a finite positive CLI tolerance with an authored input refusal."""
    try:
        return _positive_float("tolerance", float(text))
    except ValueError:
        raise argparse.ArgumentTypeError("Tolerance must be a finite positive number") from None


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point producing schema-versioned validation evidence."""
    parser = argparse.ArgumentParser(description="Validate the volt-second flux budget against exact closed forms")
    parser.add_argument("--target-id", type=str, default="local-volt-second")
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    parser.add_argument("--report", type=str, default=None, help="write sealed JSON evidence and a Markdown summary")
    parser.add_argument("--exact-tol", type=_threshold, default=1e-9, help="positive relative-error tolerance")
    parser.add_argument(
        "--margin-abs-tol", type=_threshold, default=1e-6, help="positive margin tolerance in volt-seconds"
    )
    args = parser.parse_args(argv)

    result = validate_volt_second(exact_tol=args.exact_tol, margin_abs_tol=args.margin_abs_tol)
    evidence = build_evidence(result, target_id=args.target_id)

    if args.report:
        try:
            _write_report(evidence, Path(args.report))
        except (OSError, ValueError, TypeError):
            print("Volt-second report could not be published", file=sys.stderr)
            return 2

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

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Volt-second manager tests
"""Module-specific tests for volt-second accounting and claim admission."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_control.control.volt_second_manager import (
    BootstrapCurrentEstimate,
    FluxBudget,
    FluxConsumptionMonitor,
    ScenarioFluxAnalysis,
    VoltSecondOptimizer,
    assert_volt_second_facility_claim_admissible,
    save_volt_second_claim_evidence,
    volt_second_claim_evidence,
)


def test_flux_budget_accounts_inductive_resistive_and_ejima_terms() -> None:
    budget = FluxBudget(Phi_CS_Vs=120.0, L_plasma_uH=1.2, R_plasma_uOhm=0.08)
    ramp = np.linspace(0.0, 15.0, 6)
    ramp_flux = budget.resistive_flux_ramp(ramp, dt=2.0)

    assert budget.inductive_flux(15.0) == pytest.approx(18.0)
    assert ramp_flux > 0.0
    assert budget.ejima_startup_flux(R0_m=6.2, Ip_MA=15.0) > 0.0
    assert budget.remaining_flux(15.0, ramp_flux) < budget.Phi_CS_Vs
    assert budget.max_flattop_duration(Ip_MA=15.0, I_bs_MA=4.0, ramp_flux=ramp_flux) > 0.0


def test_scenario_analysis_and_monitor_preserve_flux_budget_contracts() -> None:
    budget = FluxBudget(Phi_CS_Vs=120.0, L_plasma_uH=1.2, R_plasma_uOhm=0.08)
    report = ScenarioFluxAnalysis(budget).analyze(
        ramp_dur=80.0,
        flat_dur=400.0,
        down_dur=60.0,
        Ip_MA=15.0,
        I_bs_MA=4.0,
    )
    assert report.total_flux == pytest.approx(report.ramp_flux + report.flat_top_flux + report.ramp_down_flux)
    assert report.margin_Vs == pytest.approx(budget.Phi_CS_Vs - report.total_flux)

    monitor = FluxConsumptionMonitor(budget)
    first = monitor.step(Ip=10.0, V_loop=0.5, dt=2.0)
    second = monitor.step(Ip=10.0, V_loop=0.5, dt=2.0)
    assert second.flux_consumed_Vs > first.flux_consumed_Vs
    assert second.fraction_consumed == pytest.approx(second.flux_consumed_Vs / budget.Phi_CS_Vs)


def test_bootstrap_proxy_rejects_nonphysical_profiles() -> None:
    rho = np.linspace(0.0, 1.0, 5)
    ne = np.linspace(8.0, 4.0, 5)
    Te = np.linspace(12.0, 2.0, 5)
    Ti = np.linspace(10.0, 2.0, 5)
    q = np.linspace(1.0, 4.0, 5)
    assert BootstrapCurrentEstimate.from_profiles(ne, Te, Ti, q, rho, R0=6.2, a=2.0) >= 0.0

    with pytest.raises(ValueError, match="same length"):
        BootstrapCurrentEstimate.from_profiles(ne[:-1], Te, Ti, q, rho, R0=6.2, a=2.0)
    with pytest.raises(ValueError, match="rho must be strictly increasing"):
        BootstrapCurrentEstimate.from_profiles(ne, Te, Ti, q, np.array([0.0, 0.3, 0.3, 0.8, 1.0]), R0=6.2, a=2.0)
    with pytest.raises(ValueError, match="a must be smaller than R0"):
        BootstrapCurrentEstimate.from_profiles(ne, Te, Ti, q, rho, R0=2.0, a=2.0)


def test_volt_second_optimizer_rejects_invalid_domains() -> None:
    budget = FluxBudget(Phi_CS_Vs=120.0, L_plasma_uH=1.2, R_plasma_uOhm=0.08)
    optimizer = VoltSecondOptimizer(budget)
    ramp = optimizer.optimize_ramp(Ip_target_MA=15.0, t_ramp_max=80.0, n_segments=5)
    assert np.all(np.diff(ramp) >= 0.0)
    with pytest.raises(ValueError, match="n_segments must be at least 2"):
        optimizer.optimize_ramp(Ip_target_MA=15.0, t_ramp_max=80.0, n_segments=1)
    with pytest.raises(ValueError, match="t_ramp_max must be positive"):
        optimizer.optimize_ramp(Ip_target_MA=15.0, t_ramp_max=0.0, n_segments=5)


def test_volt_second_claim_evidence_records_bounded_boundary(tmp_path) -> None:
    budget = FluxBudget(Phi_CS_Vs=120.0, L_plasma_uH=1.2, R_plasma_uOhm=0.08)
    report = ScenarioFluxAnalysis(budget).analyze(80.0, 400.0, 60.0, Ip_MA=15.0, I_bs_MA=4.0)
    evidence = volt_second_claim_evidence(
        budget,
        report,
        Ip_MA=15.0,
        I_bs_MA=4.0,
        ramp_duration_s=80.0,
        flat_duration_s=400.0,
        ramp_down_duration_s=60.0,
        R0_m=6.2,
        ramp_flux_for_flattop_Vs=report.ramp_flux,
        source="repository_volt_second_regression",
        source_id="volt-second-regression-v1",
    )
    assert evidence.claim_status == "bounded_volt_second_evidence"
    assert evidence.facility_claim_allowed is False
    assert evidence.total_flux_Vs == pytest.approx(report.total_flux)
    with pytest.raises(ValueError, match="facility volt-second claim requires matched"):
        assert_volt_second_facility_claim_admissible(evidence)

    output = tmp_path / "volt_second_claim.json"
    save_volt_second_claim_evidence(evidence, output)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == 1
    assert persisted["claim_status"] == "bounded_volt_second_evidence"


def test_volt_second_facility_claim_requires_reference_artifact() -> None:
    budget = FluxBudget(Phi_CS_Vs=120.0, L_plasma_uH=1.2, R_plasma_uOhm=0.08)
    report = ScenarioFluxAnalysis(budget).analyze(80.0, 200.0, 60.0, Ip_MA=15.0, I_bs_MA=4.0)
    artifact = {
        "source": "external_scenario_benchmark",
        "reference_dataset_id": "volt-second-scenario-fixture-v1",
        "reference_artifact_sha256": "c" * 64,
        "reference_case_count": 2,
        "units": {
            "flux": "V s",
            "voltage": "V",
            "current": "A",
            "current_MA": "MA",
            "time": "s",
            "resistance": "ohm",
            "inductance": "H",
            "radius": "m",
            "dimensionless": "1",
        },
        "metrics": {
            "total_flux_relative_error": 0.01,
            "flat_top_duration_relative_error": 0.02,
            "ejima_flux_relative_error": 0.01,
            "bootstrap_current_abs_error_MA": 0.05,
            "margin_abs_error_Vs": 0.1,
        },
        "tolerances": {
            "total_flux_relative_error": 0.03,
            "flat_top_duration_relative_error": 0.05,
            "ejima_flux_relative_error": 0.05,
            "bootstrap_current_abs_error_MA": 0.25,
            "margin_abs_error_Vs": 0.5,
        },
    }
    evidence = volt_second_claim_evidence(
        budget,
        report,
        Ip_MA=15.0,
        I_bs_MA=4.0,
        ramp_duration_s=80.0,
        flat_duration_s=200.0,
        ramp_down_duration_s=60.0,
        R0_m=6.2,
        ramp_flux_for_flattop_Vs=report.ramp_flux,
        source="external_scenario_benchmark",
        source_id="volt-second-scenario-fixture-v1",
        reference_artifact=artifact,
    )
    assert evidence.facility_claim_allowed is True
    assert assert_volt_second_facility_claim_admissible(evidence) is evidence

    bad_artifact = dict(artifact)
    bad_artifact["metrics"] = dict(artifact["metrics"])
    bad_artifact["metrics"]["total_flux_relative_error"] = 0.5
    with pytest.raises(ValueError, match="total_flux_relative_error exceeds declared tolerance"):
        volt_second_claim_evidence(
            budget,
            report,
            Ip_MA=15.0,
            I_bs_MA=4.0,
            ramp_duration_s=80.0,
            flat_duration_s=200.0,
            ramp_down_duration_s=60.0,
            R0_m=6.2,
            ramp_flux_for_flattop_Vs=report.ramp_flux,
            source="external_scenario_benchmark",
            source_id="volt-second-scenario-fixture-v1",
            reference_artifact=bad_artifact,
        )

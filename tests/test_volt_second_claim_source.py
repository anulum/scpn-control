# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Volt-second claim-source admission tests.

"""Exercise the public facility-claim boundary with caller-declared metadata."""

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from scpn_control.control.volt_second_manager import (
    FluxBudget,
    ScenarioFluxAnalysis,
    assert_volt_second_facility_claim_admissible,
    save_volt_second_claim_evidence,
    volt_second_claim_evidence,
)


def test_caller_declared_reference_cannot_admit_facility_claim(tmp_path: Path) -> None:
    """A digest-shaped string and self-reported errors are not source evidence."""
    budget = FluxBudget(120.0, 1.2, 0.08)
    report = ScenarioFluxAnalysis(budget).analyze(80.0, 200.0, 60.0, 15.0, 4.0)
    metric_names = (
        "total_flux_relative_error",
        "flat_top_duration_relative_error",
        "ejima_flux_relative_error",
        "bootstrap_current_abs_error_MA",
        "margin_abs_error_Vs",
    )
    artifact = {
        "source": "external_scenario_benchmark",
        "reference_dataset_id": "caller-fixture",
        "reference_artifact_sha256": "c" * 64,
        "reference_case_count": 1,
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
        "metrics": dict.fromkeys(metric_names, 0.0),
        "tolerances": dict.fromkeys(metric_names, 1.0),
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
        source_id="caller-fixture",
        reference_artifact=artifact,
    )
    assert evidence.facility_claim_allowed is False
    assert evidence.claim_status == "reference_metadata_unverified"
    with pytest.raises(ValueError, match="independent reference"):
        assert_volt_second_facility_claim_admissible(evidence)
    with pytest.raises(ValueError, match="independent reference"):
        assert_volt_second_facility_claim_admissible(replace(evidence, facility_claim_allowed=True))
    destination = tmp_path / "forged-facility.json"
    with pytest.raises(ValueError, match="unverified facility"):
        save_volt_second_claim_evidence(replace(evidence, facility_claim_allowed=True), destination)
    assert not destination.exists()
    with pytest.raises(ValueError, match="unverified facility"):
        save_volt_second_claim_evidence(
            replace(evidence, claim_status="facility_volt_second_reference_matched"), destination
        )
    assert not destination.exists()
    with pytest.raises(ValueError, match="JSON"):
        save_volt_second_claim_evidence(replace(evidence, total_flux_Vs=float("nan")), destination)
    assert not destination.exists()

    with pytest.raises(ValueError, match="total_flux"):
        volt_second_claim_evidence(
            budget,
            replace(report, total_flux=float("nan")),
            Ip_MA=15.0,
            I_bs_MA=4.0,
            ramp_duration_s=80.0,
            flat_duration_s=200.0,
            ramp_down_duration_s=60.0,
            R0_m=6.2,
            ramp_flux_for_flattop_Vs=report.ramp_flux,
            source="repository_volt_second_regression",
            source_id="caller-fixture",
        )
    with pytest.raises(ValueError, match="phase flux"):
        volt_second_claim_evidence(
            budget,
            replace(report, total_flux=report.total_flux + 1.0),
            Ip_MA=15.0,
            I_bs_MA=4.0,
            ramp_duration_s=80.0,
            flat_duration_s=200.0,
            ramp_down_duration_s=60.0,
            R0_m=6.2,
            ramp_flux_for_flattop_Vs=report.ramp_flux,
            source="repository_volt_second_regression",
            source_id="caller-fixture",
        )

    missing_dataset = dict(artifact)
    missing_dataset.pop("reference_dataset_id")
    with pytest.raises(ValueError, match="reference_dataset_id"):
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
            source_id="caller-fixture",
            reference_artifact=missing_dataset,
        )


def test_claim_builder_rejects_inconsistent_report_fields() -> None:
    """Caller-constructed phase and budget summaries cannot contradict inputs."""
    budget = FluxBudget(120.0, 1.2, 0.08)
    report = ScenarioFluxAnalysis(budget).analyze(80.0, 200.0, 60.0, 15.0, 4.0)
    variants = (
        (replace(report, margin_Vs=report.margin_Vs + 1.0), "margin_Vs"),
        (replace(report, within_budget=not report.within_budget), "within_budget"),
        (replace(report, within_budget=cast(bool, "yes")), "within_budget"),
    )
    for invalid_report, message in variants:
        with pytest.raises(ValueError, match=message):
            volt_second_claim_evidence(
                budget,
                invalid_report,
                Ip_MA=15.0,
                I_bs_MA=4.0,
                ramp_duration_s=80.0,
                flat_duration_s=200.0,
                ramp_down_duration_s=60.0,
                R0_m=6.2,
                ramp_flux_for_flattop_Vs=report.ramp_flux,
                source="repository_volt_second_regression",
                source_id="caller-fixture",
            )

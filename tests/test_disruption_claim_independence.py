# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption mitigation claim independence tests
"""Exercise public admission paths against forged disruption claims."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from scpn_control.control.halo_re_physics import (
    DisruptionMitigationClaimEvidence,
    DisruptionMitigationReport,
    assert_disruption_mitigation_claim_admissible,
    disruption_mitigation_claim_evidence,
    run_disruption_ensemble,
    save_disruption_mitigation_claim_evidence,
)


def test_forged_ensemble_claim_cannot_be_asserted_or_saved(tmp_path: Path) -> None:
    """A caller cannot promote bounded ensemble data by editing two fields."""
    report = run_disruption_ensemble(ensemble_runs=1, seed=19)
    bounded = disruption_mitigation_claim_evidence(
        report,
        source="synthetic_regression_reference",
        source_id="independence-test",
        ensemble_seed=19,
    )
    forged = replace(
        bounded,
        mitigation_claim_allowed=True,
        claim_status="matched disruption reference admission passed",
    )
    with pytest.raises(ValueError, match="independently verified"):
        assert_disruption_mitigation_claim_admissible(forged)
    with pytest.raises(ValueError, match="independently verified"):
        save_disruption_mitigation_claim_evidence(forged, tmp_path / "forged.json")


def test_bounded_evidence_cannot_be_saved_with_validated_status(tmp_path: Path) -> None:
    """Reject a validated status string when the admission flag is false."""
    report = run_disruption_ensemble(ensemble_runs=1, seed=23)
    evidence = disruption_mitigation_claim_evidence(
        report, source="synthetic_regression_reference", source_id="status-test", ensemble_seed=23
    )
    forged = replace(evidence, claim_status="matched disruption reference admission passed")
    with pytest.raises(ValueError, match="status is inconsistent"):
        save_disruption_mitigation_claim_evidence(forged, tmp_path / "forged_status.json")


def test_bounded_evidence_cannot_carry_reference_metrics(tmp_path: Path) -> None:
    """Reject each kind of unverified reference comparison in bounded evidence."""
    report = run_disruption_ensemble(ensemble_runs=1, seed=29)
    evidence = disruption_mitigation_claim_evidence(
        report, source="synthetic_regression_reference", source_id="reference-test", ensemble_seed=29
    )
    forged_records = [
        replace(evidence, reference_source="measured_disruption_campaign"),
        replace(evidence, reference_dataset_id="unverified-shot"),
        replace(evidence, reference_artifact_sha256="a" * 64),
        replace(evidence, reference_case_count=1),
        replace(evidence, halo_current_relative_error=0.01),
    ]
    for forged in forged_records:
        with pytest.raises(ValueError, match="cannot carry unverified reference comparisons"):
            save_disruption_mitigation_claim_evidence(forged, tmp_path / "forged_reference.json")


def _bounded_evidence() -> tuple[DisruptionMitigationReport, DisruptionMitigationClaimEvidence]:
    """Bounded evidence."""
    report = run_disruption_ensemble(ensemble_runs=4, seed=8)
    evidence = disruption_mitigation_claim_evidence(
        report,
        source="documented_public_reference",
        source_id="tests/test_halo_re_physics.py::bounded",
        ensemble_seed=8,
    )
    return report, evidence


def test_public_claim_builder_rejects_blank_and_non_string_sources() -> None:
    """Reject unusable source labels through the public claim builder."""
    report, _ = _bounded_evidence()
    for source in ("   ", cast(str, 5)):
        with pytest.raises(ValueError, match="source must be a non-empty string"):
            disruption_mitigation_claim_evidence(report, source=source, source_id="case", ensemble_seed=8)


def test_public_claim_builder_rejects_invalid_tpf_products() -> None:
    """Reject invalid ensemble TPF values through the public claim builder."""
    report, _ = _bounded_evidence()
    for value in (True, -1.0, float("nan")):
        with pytest.raises(ValueError, match="mean_tpf_product must be finite and non-negative"):
            disruption_mitigation_claim_evidence(
                replace(report, mean_tpf_product=cast(float, value)),
                source="documented_public_reference",
                source_id="case",
                ensemble_seed=8,
            )


def test_public_claim_builder_rejects_invalid_prevention_rates() -> None:
    """Reject invalid prevention rates through the public claim builder."""
    report, _ = _bounded_evidence()
    for value in (1.5, float("nan")):
        with pytest.raises(ValueError, match=r"prevention_rate must be finite in \[0, 1\]"):
            disruption_mitigation_claim_evidence(
                replace(report, prevention_rate=value),
                source="documented_public_reference",
                source_id="case",
                ensemble_seed=8,
            )


def test_claim_evidence_rejects_non_report_object() -> None:
    """Claim evidence rejects non report object."""
    with pytest.raises(ValueError, match="must be DisruptionMitigationReport"):
        disruption_mitigation_claim_evidence(
            cast(DisruptionMitigationReport, {"not": "report"}),
            source="documented_public_reference",
            source_id="case",
            ensemble_seed=1,
        )


def test_claim_evidence_rejects_non_positive_ensemble_runs() -> None:
    """Claim evidence rejects non positive ensemble runs."""
    report, _ = _bounded_evidence()
    with pytest.raises(ValueError, match="ensemble_runs must be positive"):
        disruption_mitigation_claim_evidence(
            replace(report, ensemble_runs=0),
            source="documented_public_reference",
            source_id="case",
            ensemble_seed=1,
        )


def test_claim_evidence_requires_present_mean_tpf_product() -> None:
    """Claim evidence requires present mean tpf product."""
    report, _ = _bounded_evidence()
    with pytest.raises(ValueError, match="mean_tpf_product must be present"):
        disruption_mitigation_claim_evidence(
            replace(report, mean_tpf_product=cast(float, None)),
            source="documented_public_reference",
            source_id="case",
            ensemble_seed=1,
        )


def test_assert_admissible_rejects_non_evidence_and_bad_schema() -> None:
    """Assert admissible rejects non evidence and bad schema."""
    with pytest.raises(ValueError, match="must be DisruptionMitigationClaimEvidence"):
        assert_disruption_mitigation_claim_admissible(cast(DisruptionMitigationClaimEvidence, {"not": "evidence"}))
    _, evidence = _bounded_evidence()
    with pytest.raises(ValueError, match="schema_version is unsupported"):
        assert_disruption_mitigation_claim_admissible(replace(evidence, schema_version=999))


def test_save_claim_evidence_rejects_non_evidence(tmp_path: Path) -> None:
    """Save claim evidence rejects non evidence."""
    with pytest.raises(ValueError, match="must be DisruptionMitigationClaimEvidence"):
        save_disruption_mitigation_claim_evidence(
            cast(DisruptionMitigationClaimEvidence, {"not": "evidence"}), tmp_path / "x.json"
        )

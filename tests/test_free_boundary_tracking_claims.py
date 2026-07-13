# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Free-boundary tracking claim-evidence tests

"""Fail-closed validation, extraction and admission branches of the free-boundary
tracking claim-evidence surface.

Exercises the summary/reference scalar validators, the reference-artifact
extraction matrix (unit, digest, case-count and per-metric tolerance rejections),
and the bounded/facility claim-evidence build, admission gate, and JSON persistence
round-trip. The claim surface consumes a plain run-summary ``dict``, so these tests
build synthetic summaries directly and never construct a controller.
"""

from __future__ import annotations

import json

import pytest

from scpn_control.control.free_boundary_tracking_claims import (
    _extract_free_boundary_reference_artifact,
    _finite_summary_scalar,
    _finite_summary_scalar_or_none,
    _non_empty_text,
    _nonnegative_reference_scalar,
    _positive_reference_scalar,
    _summary_int,
    assert_free_boundary_tracking_facility_claim_admissible,
    free_boundary_tracking_claim_evidence,
    save_free_boundary_tracking_claim_evidence,
)


def _valid_free_boundary_summary() -> dict:
    return {
        "config_path": "dummy.json",
        "steps": 3,
        "boundary_variant": "limiter",
        "final_tracking_error_norm": 0.01,
        "final_true_tracking_error_norm": 0.012,
        "true_shape_rms": 0.004,
        "true_x_point_position_error": 0.006,
        "true_x_point_flux_error": 0.003,
        "true_divertor_rms": 0.004,
        "max_abs_coil_current": 1.5,
        "max_abs_delta_i": 0.2,
        "min_response_rank": 4,
        "max_response_condition_number": 12.0,
        "response_degenerate_count": 0,
        "supervisor_intervention_count": 0,
        "fallback_active_steps": 0,
        "objective_converged": True,
        "measurement_distortion_enabled": False,
        "measurement_compensation_enabled": False,
        "measurement_latency_enabled": False,
        "latency_compensation_enabled": False,
    }


def _valid_free_boundary_reference_artifact() -> dict:
    return {
        "source": "external_equilibrium_benchmark",
        "reference_dataset_id": "efit-free-boundary-fixture-v1",
        "reference_artifact_sha256": "a" * 64,
        "reference_case_count": 2,
        "units": {
            "position": "m",
            "flux": "Wb/rad",
            "current": "MA",
            "time": "s",
            "tracking_error": "1",
        },
        "metrics": {
            "shape_rms_abs_error": 0.004,
            "x_point_position_abs_error_m": 0.006,
            "x_point_flux_abs_error": 0.003,
            "divertor_rms_abs_error": 0.004,
            "coil_current_relative_error": 0.01,
        },
        "tolerances": {
            "shape_rms_abs_error": 0.01,
            "x_point_position_abs_error_m": 0.02,
            "x_point_flux_abs_error": 0.01,
            "divertor_rms_abs_error": 0.01,
            "coil_current_relative_error": 0.03,
        },
    }


def test_non_empty_text_rejects_blank_and_non_string():
    with pytest.raises(ValueError, match="must be a non-empty string"):
        _non_empty_text("field", "   ")
    with pytest.raises(ValueError, match="must be a non-empty string"):
        _non_empty_text("field", 9)


def test_finite_summary_scalar_rejects_non_numeric():
    with pytest.raises(ValueError, match="must be a finite numeric value"):
        _finite_summary_scalar({}, "missing")
    with pytest.raises(ValueError, match="must be a finite numeric value"):
        _finite_summary_scalar({"k": float("inf")}, "k")


def test_finite_summary_scalar_or_none_handles_none_and_rejects_non_numeric():
    assert _finite_summary_scalar_or_none({"k": None}, "k") is None
    assert _finite_summary_scalar_or_none({"k": float("nan")}, "k") is None
    with pytest.raises(ValueError, match="must be numeric when supplied"):
        _finite_summary_scalar_or_none({"k": "x"}, "k")


def test_summary_int_rejects_non_integers():
    with pytest.raises(ValueError, match="must be an integer"):
        _summary_int({"k": 1.5}, "k")
    with pytest.raises(ValueError, match="must be an integer"):
        _summary_int({"k": True}, "k")


@pytest.mark.parametrize("value", [True, float("inf"), "x"])
def test_positive_reference_scalar_rejects_non_numeric_or_non_finite(value):
    with pytest.raises(ValueError, match="finite and positive"):
        _positive_reference_scalar("metric", value)


def test_positive_reference_scalar_rejects_non_positive():
    with pytest.raises(ValueError, match="finite and positive"):
        _positive_reference_scalar("metric", 0.0)


@pytest.mark.parametrize("value", [True, float("nan"), "x"])
def test_nonnegative_reference_scalar_rejects_non_numeric_or_non_finite(value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        _nonnegative_reference_scalar("metric", value)


def test_nonnegative_reference_scalar_rejects_negative():
    with pytest.raises(ValueError, match="finite and non-negative"):
        _nonnegative_reference_scalar("metric", -1.0)


def test_extract_reference_artifact_none_returns_inactive():
    assert _extract_free_boundary_reference_artifact(None) == (None, False)


def test_extract_reference_artifact_rejects_non_dict():
    with pytest.raises(ValueError, match="must be a dictionary"):
        _extract_free_boundary_reference_artifact(["not", "a", "dict"])


def test_extract_reference_artifact_rejects_inadmissible_source():
    artifact = _valid_free_boundary_reference_artifact()
    artifact["source"] = "repository_free_boundary_regression"  # bounded but not facility
    with pytest.raises(ValueError, match="source must be one of"):
        _extract_free_boundary_reference_artifact(artifact)


def test_extract_reference_artifact_rejects_bad_units():
    artifact = _valid_free_boundary_reference_artifact()
    artifact["units"] = dict(artifact["units"])
    artifact["units"]["flux"] = "Wb"
    with pytest.raises(ValueError, match="units must declare free-boundary SI units"):
        _extract_free_boundary_reference_artifact(artifact)


def test_extract_reference_artifact_rejects_non_digest_sha():
    artifact = _valid_free_boundary_reference_artifact()
    artifact["reference_artifact_sha256"] = "abc"
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        _extract_free_boundary_reference_artifact(artifact)


@pytest.mark.parametrize("count", [0, -1, True])
def test_extract_reference_artifact_rejects_bad_case_count(count):
    artifact = _valid_free_boundary_reference_artifact()
    artifact["reference_case_count"] = count
    with pytest.raises(ValueError, match="reference_case_count must be a positive integer"):
        _extract_free_boundary_reference_artifact(artifact)


def test_extract_reference_artifact_rejects_non_dict_metric_blocks():
    artifact = _valid_free_boundary_reference_artifact()
    artifact["metrics"] = "not a dict"
    with pytest.raises(ValueError, match="metrics and tolerances must be dictionaries"):
        _extract_free_boundary_reference_artifact(artifact)


def test_extract_reference_artifact_rejects_metric_exceeding_tolerance():
    artifact = _valid_free_boundary_reference_artifact()
    artifact["metrics"] = dict(artifact["metrics"])
    artifact["metrics"]["shape_rms_abs_error"] = 0.5
    with pytest.raises(ValueError, match="shape_rms_abs_error exceeds declared tolerance"):
        _extract_free_boundary_reference_artifact(artifact)


def test_claim_evidence_rejects_inadmissible_source():
    with pytest.raises(ValueError, match="source must be one of"):
        free_boundary_tracking_claim_evidence(
            _valid_free_boundary_summary(), source="not_a_declared_source", source_id="case"
        )


def test_claim_evidence_bounded_path_and_save_round_trip(tmp_path) -> None:
    evidence = free_boundary_tracking_claim_evidence(
        _valid_free_boundary_summary(),
        source="repository_free_boundary_regression",
        source_id="free-boundary-tracking-regression-v1",
    )
    assert evidence.claim_status == "bounded_free_boundary_tracking_evidence"
    assert evidence.facility_claim_allowed is False
    assert evidence.reference_artifact_sha256 is None
    assert evidence.max_response_condition_number == 12.0
    with pytest.raises(ValueError, match="facility free-boundary tracking claim requires matched reference"):
        assert_free_boundary_tracking_facility_claim_admissible(evidence)

    output = tmp_path / "free_boundary_claim.json"
    save_free_boundary_tracking_claim_evidence(evidence, output)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == 1
    assert persisted["claim_status"] == "bounded_free_boundary_tracking_evidence"


def test_claim_evidence_facility_path_admits_matched_reference() -> None:
    evidence = free_boundary_tracking_claim_evidence(
        _valid_free_boundary_summary(),
        source="external_equilibrium_benchmark",
        source_id="free-boundary-external-benchmark-v1",
        reference_artifact=_valid_free_boundary_reference_artifact(),
    )
    assert evidence.facility_claim_allowed is True
    assert evidence.claim_status == "facility_free_boundary_reference_matched"
    assert evidence.reference_dataset_id == "efit-free-boundary-fixture-v1"
    assert evidence.shape_rms_abs_error == 0.004
    assert_free_boundary_tracking_facility_claim_admissible(evidence)

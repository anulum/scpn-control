# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Static structured-mu claim validation tests
"""Module-specific tests for bounded static mu-analysis contracts."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pytest

import scpn_control.control.static_mu_analysis as mu
from scpn_control._typing import FloatArray
from scpn_control.control.static_mu_analysis import (
    RiccatiStateFeedbackController,
    StaticMuAnalysisClaimEvidence,
    StructuredUncertainty,
    UncertaintyBlock,
    assert_static_mu_analysis_validated_claim_admissible,
    load_static_mu_analysis_claim_evidence,
    save_static_mu_analysis_claim_evidence,
    static_mu_analysis_claim_evidence,
)


def _plant() -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    A = np.array([[-1.4, 0.2], [-0.1, -0.9]], dtype=float)
    B = np.eye(2)
    C = np.eye(2)
    D = np.zeros((2, 2), dtype=float)
    return A, B, C, D


def _uncertainty() -> StructuredUncertainty:
    return StructuredUncertainty(
        [
            UncertaintyBlock("plasma_position", 1, 0.02, "real_scalar"),
            UncertaintyBlock("plasma_current", 1, 0.03, "real_scalar"),
        ]
    )


def test_static_mu_claim_evidence_records_bounded_boundary(tmp_path: Path) -> None:
    """Exercise static mu claim evidence records bounded boundary."""
    controller = RiccatiStateFeedbackController(_plant(), _uncertainty())
    controller.design()
    evidence = static_mu_analysis_claim_evidence(
        controller,
        source="repository_static_mu_regression",
        source_id="mu-static-regression-v1",
    )
    assert evidence.claim_status == "bounded_static_mu_evidence"
    assert evidence.validated_claim_allowed is False
    assert evidence.static_dc_analysis_only is True
    assert evidence.closed_loop_spectral_abscissa < 0.0
    with pytest.raises(ValueError, match="validated static mu-analysis claim requires matched"):
        assert_static_mu_analysis_validated_claim_admissible(evidence)

    output = tmp_path / "mu_claim.json"
    save_static_mu_analysis_claim_evidence(evidence, output)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == 1
    assert persisted["claim_status"] == "bounded_static_mu_evidence"
    assert persisted["payload_sha256"]
    assert load_static_mu_analysis_claim_evidence(output) == evidence
    with pytest.raises(ValueError, match="validated static mu-analysis claim requires matched"):
        load_static_mu_analysis_claim_evidence(output, require_validated_claim=True)

    with pytest.raises(ValueError, match="evidence must"):
        save_static_mu_analysis_claim_evidence(cast(StaticMuAnalysisClaimEvidence, object()), tmp_path / "bad.json")


def test_static_mu_claim_evidence_loader_rejects_tampering_and_duplicate_keys(tmp_path: Path) -> None:
    """Exercise static mu claim evidence loader rejects tampering and duplicate keys."""
    controller = RiccatiStateFeedbackController(_plant(), _uncertainty())
    controller.design()
    evidence = static_mu_analysis_claim_evidence(
        controller,
        source="repository_static_mu_regression",
        source_id="mu-static-regression-v1",
    )
    output = tmp_path / "mu_claim.json"
    save_static_mu_analysis_claim_evidence(evidence, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["mu_peak_upper_bound"] = payload["mu_peak_upper_bound"] * 2.0
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="payload_sha256"):
        load_static_mu_analysis_claim_evidence(output)

    duplicate = tmp_path / "duplicate_mu_claim.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_static_mu_analysis_claim_evidence(duplicate)


def test_static_mu_claim_evidence_rejects_undesignated_controller_and_bad_claim_domains() -> None:
    """Exercise static mu claim evidence rejects undesignated controller and bad claim domains."""
    controller = RiccatiStateFeedbackController(_plant(), _uncertainty())
    with pytest.raises(ValueError, match="designed"):
        static_mu_analysis_claim_evidence(
            controller,
            source="repository_static_mu_regression",
            source_id="mu-static-regression-v1",
        )

    controller.design()
    with pytest.raises(ValueError, match="source must be one"):
        static_mu_analysis_claim_evidence(controller, source="internal_claim", source_id="mu-static-regression-v1")
    with pytest.raises(ValueError, match="source_id"):
        static_mu_analysis_claim_evidence(controller, source="repository_static_mu_regression", source_id=" ")
    with pytest.raises(ValueError, match="model_id"):
        static_mu_analysis_claim_evidence(
            controller,
            source="repository_static_mu_regression",
            source_id="mu-static-regression-v1",
            model_id=" ",
        )
    with pytest.raises(ValueError, match="mu_upper_bound_relative_tolerance"):
        static_mu_analysis_claim_evidence(
            controller,
            source="repository_static_mu_regression",
            source_id="mu-static-regression-v1",
            mu_upper_bound_relative_tolerance=0.0,
        )


def test_static_mu_rejects_unverified_reference_artifact() -> None:
    """A caller-supplied reference dictionary cannot establish validation."""
    controller = RiccatiStateFeedbackController(_plant(), _uncertainty())
    controller.design()
    artifact = {
        "source": "external_mu_toolbox_benchmark",
        "reference_dataset_id": "mu-toolbox-static-fixture-v1",
        "reference_artifact_sha256": "e" * 64,
        "reference_case_count": 2,
        "units": {
            "mu": "1",
            "robustness_margin": "1",
            "controller_gain": "1",
            "d_scaling": "1",
            "spectral_abscissa": "s^-1",
        },
        "metrics": {
            "mu_upper_bound_relative_error": 0.01,
            "robustness_margin_abs_error": 0.02,
            "controller_gain_relative_error": 0.03,
            "d_scaling_relative_error": 0.04,
            "closed_loop_spectral_abscissa_abs_error": 0.01,
        },
        "tolerances": {
            "mu_upper_bound_relative_error": 0.05,
            "robustness_margin_abs_error": 0.05,
            "controller_gain_relative_error": 0.10,
            "d_scaling_relative_error": 0.10,
            "closed_loop_spectral_abscissa_abs_error": 0.05,
        },
    }
    with pytest.raises(ValueError, match="independently verified reference evidence"):
        static_mu_analysis_claim_evidence(
            controller,
            source="external_mu_toolbox_benchmark",
            source_id="mu-toolbox-static-fixture-v1",
            reference_artifact=artifact,
        )

    bad_artifact = dict(artifact)
    bad_metrics = dict(cast(dict[str, Any], artifact["metrics"]))
    bad_metrics["mu_upper_bound_relative_error"] = 0.5
    bad_artifact["metrics"] = bad_metrics
    with pytest.raises(ValueError, match="mu_upper_bound_relative_error exceeds declared tolerance"):
        static_mu_analysis_claim_evidence(
            controller,
            source="external_mu_toolbox_benchmark",
            source_id="mu-toolbox-static-fixture-v1",
            reference_artifact=bad_artifact,
        )


def _valid_reference_artifact() -> dict[str, Any]:
    return {
        "source": "external_mu_toolbox_benchmark",
        "reference_dataset_id": "mu-toolbox-static-fixture-v1",
        "reference_artifact_sha256": "b" * 64,
        "reference_case_count": 3,
        "units": {
            "mu": "1",
            "robustness_margin": "1",
            "controller_gain": "1",
            "d_scaling": "1",
            "spectral_abscissa": "s^-1",
        },
        "metrics": {
            "mu_upper_bound_relative_error": 0.01,
            "robustness_margin_abs_error": 0.01,
            "controller_gain_relative_error": 0.02,
            "d_scaling_relative_error": 0.02,
            "closed_loop_spectral_abscissa_abs_error": 0.01,
        },
        "tolerances": {
            "mu_upper_bound_relative_error": 0.05,
            "robustness_margin_abs_error": 0.05,
            "controller_gain_relative_error": 0.10,
            "d_scaling_relative_error": 0.10,
            "closed_loop_spectral_abscissa_abs_error": 0.05,
        },
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda artifact: artifact.update(source="repository_static_mu_regression"), "source must be one"),
        (lambda artifact: artifact.update(reference_artifact_sha256="not-a-digest"), "SHA-256"),
        (lambda artifact: artifact.update(reference_case_count=0), "positive integer"),
        (lambda artifact: artifact.update(metrics=[]), "metrics and tolerances"),
        (lambda artifact: artifact["metrics"].update(mu_upper_bound_relative_error=-0.1), "non-negative"),
        (lambda artifact: artifact["tolerances"].update(mu_upper_bound_relative_error=0.0), "positive"),
    ],
)
def test_static_mu_reference_artifact_rejects_invalid_validation_payloads(
    mutation: Callable[[dict[str, Any]], None], message: str
) -> None:
    """Exercise static mu reference artifact rejects invalid validation payloads."""
    controller = RiccatiStateFeedbackController(_plant(), _uncertainty())
    controller.design()
    artifact = _valid_reference_artifact()
    mutation(artifact)

    with pytest.raises(ValueError, match=message):
        static_mu_analysis_claim_evidence(
            controller,
            source="external_mu_toolbox_benchmark",
            source_id="static-two-state-reference",
            reference_artifact=artifact,
        )


def test_static_mu_reference_artifact_rejects_non_mapping_reference_payload() -> None:
    """Exercise static mu reference artifact rejects non mapping reference payload."""
    controller = RiccatiStateFeedbackController(_plant(), _uncertainty())
    controller.design()

    with pytest.raises(ValueError, match="reference_artifact must be a dictionary"):
        static_mu_analysis_claim_evidence(
            controller,
            source="external_mu_toolbox_benchmark",
            source_id="static-two-state-reference",
            reference_artifact=cast(dict[str, Any], ["not", "a", "mapping"]),
        )


def test_static_mu_reference_artifact_rejects_unit_mismatches() -> None:
    """Exercise static mu reference artifact rejects unit mismatches."""
    controller = RiccatiStateFeedbackController(_plant(), _uncertainty())
    controller.design()
    artifact = {
        "source": "external_mu_toolbox_benchmark",
        "reference_dataset_id": "mu-toolbox-static-fixture-v1",
        "reference_artifact_sha256": "b" * 64,
        "reference_case_count": 3,
        "units": {"mu": "wrong"},
        "metrics": {
            "mu_upper_bound_relative_error": 0.01,
            "robustness_margin_abs_error": 0.01,
            "controller_gain_relative_error": 0.02,
            "d_scaling_relative_error": 0.02,
            "closed_loop_spectral_abscissa_abs_error": 0.01,
        },
        "tolerances": {
            "mu_upper_bound_relative_error": 0.05,
            "robustness_margin_abs_error": 0.05,
            "controller_gain_relative_error": 0.10,
            "d_scaling_relative_error": 0.10,
            "closed_loop_spectral_abscissa_abs_error": 0.05,
        },
    }

    with pytest.raises(ValueError, match="unit contracts"):
        static_mu_analysis_claim_evidence(
            controller,
            source="external_mu_toolbox_benchmark",
            source_id="static-two-state-reference",
            reference_artifact=artifact,
        )


# ── Validator-helper, claim-payload and numeric-guard branch contracts ────────


def _bounded_evidence() -> StaticMuAnalysisClaimEvidence:
    evidence = mu.StaticMuAnalysisClaimEvidence(
        schema_version=1,
        source="repository_static_mu_regression",
        source_id="sid",
        model_id="mid",
        state_dimension=2,
        control_dimension=1,
        output_dimension=1,
        uncertainty_block_count=1,
        uncertainty_total_size=1,
        max_uncertainty_bound=0.5,
        block_structure=[(1, "full")],
        mu_peak_upper_bound=0.8,
        robustness_margin=1.25,
        controller_gain_frobenius_norm=2.0,
        d_scalings=[1.0],
        closed_loop_spectral_abscissa=-0.5,
        static_dc_analysis_only=True,
        reference_source=None,
        reference_dataset_id=None,
        reference_artifact_sha256=None,
        reference_case_count=None,
        mu_upper_bound_relative_error=None,
        robustness_margin_abs_error=None,
        controller_gain_relative_error=None,
        d_scaling_relative_error=None,
        closed_loop_spectral_abscissa_abs_error=None,
        mu_upper_bound_relative_tolerance=0.05,
        robustness_margin_abs_tolerance=0.05,
        controller_gain_relative_tolerance=0.10,
        d_scaling_relative_tolerance=0.10,
        closed_loop_spectral_abscissa_abs_tolerance=0.05,
        validated_claim_allowed=False,
        claim_status="bounded_static_mu_evidence",
    )
    return mu._with_payload_digest(evidence)


def _validated_evidence() -> StaticMuAnalysisClaimEvidence:
    evidence = replace(
        _bounded_evidence(),
        source="documented_public_reference",
        reference_source="public-ref",
        reference_dataset_id="dataset-1",
        reference_artifact_sha256="a" * 64,
        reference_case_count=3,
        mu_upper_bound_relative_error=0.01,
        robustness_margin_abs_error=0.01,
        controller_gain_relative_error=0.01,
        d_scaling_relative_error=0.01,
        closed_loop_spectral_abscissa_abs_error=0.01,
        validated_claim_allowed=True,
        claim_status="validated_static_mu_reference_matched",
    )
    return mu._with_payload_digest(evidence)


def _reseal(payload: dict[str, Any]) -> dict[str, Any]:
    payload["payload_sha256"] = mu._claim_payload_sha256(payload)
    return payload


def test_finite_scalar_rejects_non_finite() -> None:
    """Exercise finite scalar rejects non finite."""
    with pytest.raises(ValueError, match="must be finite"):
        mu._finite_scalar("x", float("nan"))


@pytest.mark.parametrize(
    ("fn_name", "value", "match"),
    [
        ("_positive_reference_scalar", float("inf"), "finite and positive"),
        ("_positive_reference_scalar", True, "finite and positive"),
        ("_nonnegative_reference_scalar", float("nan"), "finite and non-negative"),
        ("_require_positive_claim_int", 0, "positive integer"),
    ],
)
def test_reference_scalar_validators_reject(fn_name: str, value: object, match: str) -> None:
    """Exercise reference scalar validators reject."""
    with pytest.raises(ValueError, match=match):
        getattr(mu, fn_name)("field", value)


def test_require_bool_rejects_non_bool() -> None:
    """Exercise require bool rejects non bool."""
    with pytest.raises(ValueError, match="must be boolean"):
        mu._require_bool("field", 1)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"block_structure": "notlist"}, "block_structure must be a list"),
        ({"block_structure": []}, "length must match uncertainty_block_count"),
        ({"block_structure": [(1, "full", 9)]}, "entries must be \\[size, block_type\\]"),
        ({"block_structure": [(1, "bogus_type")]}, "block_type must be one of"),
        ({"block_structure": [(2, "full")]}, "must sum to uncertainty_total_size"),
    ],
)
def test_validate_claim_structure_rejects(overrides: dict[str, Any], match: str) -> None:
    """Exercise validate claim structure rejects."""
    evidence = replace(_bounded_evidence(), **overrides)
    with pytest.raises(ValueError, match=match):
        mu._validate_claim_structure(evidence)


def test_validate_payload_rejects_missing_field() -> None:
    """Exercise validate payload rejects missing field."""
    payload = asdict(_bounded_evidence())
    del payload["source"]
    with pytest.raises(ValueError, match="missing fields"):
        mu._validate_static_mu_analysis_claim_payload(payload, require_validated_claim=False)


def test_validate_payload_rejects_unsupported_field() -> None:
    """Exercise validate payload rejects unsupported field."""
    payload = asdict(_bounded_evidence())
    payload["bogus_field"] = 1
    with pytest.raises(ValueError, match="unsupported fields"):
        mu._validate_static_mu_analysis_claim_payload(payload, require_validated_claim=False)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"schema_version": 2}, "schema_version is unsupported"),
        ({"source": "unlisted_source"}, "source must be one of"),
        ({"static_dc_analysis_only": False}, "must declare static_dc_analysis_only"),
        ({"closed_loop_spectral_abscissa": float("inf")}, "closed_loop_spectral_abscissa must be finite"),
        ({"closed_loop_spectral_abscissa": 0.1}, "must be negative"),
        ({"d_scalings": [1.0, 2.0]}, "one positive value per uncertainty block"),
        ({"claim_status": "wrong"}, "claim_status does not match"),
        ({"reference_case_count": 5}, "cannot carry partial reference fields"),
    ],
)
def test_validate_bounded_payload_rejects(overrides: dict[str, Any], match: str) -> None:
    """Exercise validate bounded payload rejects."""
    payload = asdict(replace(_bounded_evidence(), **overrides))
    _reseal(payload)
    with pytest.raises(ValueError, match=match):
        mu._validate_static_mu_analysis_claim_payload(payload, require_validated_claim=False)


def test_validate_validated_payload_rejects_non_validated_source() -> None:
    """Exercise validate validated payload rejects non validated source."""
    payload = asdict(replace(_validated_evidence(), source="repository_static_mu_regression"))
    _reseal(payload)
    with pytest.raises(ValueError, match="require a validated source"):
        mu._validate_static_mu_analysis_claim_payload(payload, require_validated_claim=True)


def test_validate_validated_payload_rejects_metric_over_tolerance() -> None:
    """Exercise validate validated payload rejects metric over tolerance."""
    payload = asdict(replace(_validated_evidence(), mu_upper_bound_relative_error=0.5))
    _reseal(payload)
    with pytest.raises(ValueError, match="exceeds declared tolerance"):
        mu._validate_static_mu_analysis_claim_payload(payload, require_validated_claim=True)


def test_resealed_self_reported_reference_cannot_admit_validated_claim(tmp_path: Path) -> None:
    """A forged payload digest and claimed reference hash cannot grant admission."""
    payload = _reseal(asdict(_validated_evidence()))
    with pytest.raises(ValueError, match="independently verified reference evidence"):
        mu._validate_static_mu_analysis_claim_payload(payload, require_validated_claim=True)
    with pytest.raises(ValueError, match="independently verified reference evidence"):
        assert_static_mu_analysis_validated_claim_admissible(_validated_evidence())
    path = tmp_path / "forged_static_mu_claim.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="independently verified reference evidence"):
        load_static_mu_analysis_claim_evidence(path, require_validated_claim=True)


def test_assert_validated_claim_rejects_non_evidence() -> None:
    """Exercise assert validated claim rejects non evidence."""
    with pytest.raises(ValueError, match="must be StaticMuAnalysisClaimEvidence"):
        assert_static_mu_analysis_validated_claim_admissible(cast(StaticMuAnalysisClaimEvidence, object()))


def test_load_claim_evidence_rejects_non_object(tmp_path: Path) -> None:
    """Exercise load claim evidence rejects non object."""
    path = tmp_path / "claim.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_static_mu_analysis_claim_evidence(path)

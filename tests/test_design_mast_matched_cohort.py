# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tests for the fail-closed MAST cohort-design gate
"""Adversarial tests for L2F-23a statistical design and claim boundaries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from validation.audit_mast_label_proxy_sensitivity import AUDIT_SCHEMA
from validation.design_mast_matched_cohort import (
    CANONICAL_CHANNELS,
    COHORT_DESIGN_REPORT_SCHEMA,
    COHORT_DESIGN_SCHEMA,
    CohortDesignError,
    _auc_design_metrics,
    _auc_standard_error,
    _load_json,
    design_matched_cohort,
    main,
    minimum_auc_group_sizes,
    minimum_binomial_sample_size,
    wilson_half_width,
)
from validation.mast_source_object_manifest import canonical_json_sha256, file_sha256

_FIXED_TS = "2026-07-22T17:00:00Z"
_CLAIMS = {
    "scientific_validation": False,
    "independent_label_validation": False,
    "cohort_admission": False,
    "training_admission": False,
    "facility_prediction": False,
    "control_admission": False,
}


def _seal(payload: dict[str, Any]) -> dict[str, Any]:
    payload["payload_sha256"] = None
    payload["payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _design_payload() -> dict[str, Any]:
    return _seal(
        {
            "schema_version": COHORT_DESIGN_SCHEMA,
            "machine": "MAST",
            "design_status": "draft_owner_ratification_required",
            "confidence_level": 0.95,
            "target_power": 0.90,
            "expected_sensitivity": 0.80,
            "expected_specificity": 0.80,
            "max_sensitivity_ci_half_width": 0.05,
            "max_specificity_ci_half_width": 0.05,
            "target_auc": 0.75,
            "null_auc": 0.50,
            "max_auc_ci_half_width": 0.05,
            "control_to_case_ratio": 1,
            "admissible_label_authorities": ["defuse", "facility", "independent_manual_review"],
            "excluded_outcomes": ["aborted_no_plasma", "ambiguous"],
            "required_channels": list(CANONICAL_CHANNELS),
            "exact_match_fields": ["campaign", "configuration", "diagnostic_availability", "programme_class"],
            "caliper_policy_status": "required_not_supplied",
            "payload_sha256": None,
        }
    )


def _proxy_payload(*, verified: int = 93, stable: int = 42, unstable: int = 51) -> dict[str, Any]:
    return _seal(
        {
            "schema_version": AUDIT_SCHEMA,
            "status": "blocked",
            "dataset_id": "mast-disruption-campaign01",
            "source_dataset_report_file_sha256": "a" * 64,
            "source_dataset_report_payload_sha256": "b" * 64,
            "source_label_algorithm_sha256": "c" * 64,
            "verified_shot_artifact_count": verified,
            "stable_shot_count": stable,
            "unstable_shot_count": unstable,
            "unstable_shot_ids": list(range(1, unstable + 1)),
            "programme_comparison": {
                "status": "not_computable",
                "independent_validation_claim": False,
            },
            "claim_boundary": dict(_CLAIMS),
            "payload_sha256": None,
        }
    )


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _inputs(root: Path) -> tuple[Path, Path]:
    return (
        _write_json(root / "design.json", _design_payload()),
        _write_json(root / "proxy.json", _proxy_payload()),
    )


def _rewrite(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    mutate(payload)
    _seal(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_design_is_statistically_explicit_and_remains_blocked(tmp_path: Path) -> None:
    """Compute sample minima without converting proxy outcomes into a cohort."""
    design_path, proxy_path = _inputs(tmp_path)
    report = design_matched_cohort(
        design_spec_path=design_path,
        proxy_sensitivity_path=proxy_path,
        generated_at=_FIXED_TS,
    )

    assert report["schema_version"] == COHORT_DESIGN_REPORT_SCHEMA
    assert report["status"] == "blocked"
    assert report["admitted_shot_ids"] == []
    assert set(cast(dict[str, bool], report["claim_boundary"]).values()) == {False}
    blockers = cast(list[str], report["blockers"])
    assert "authoritative_label_manifest_not_supplied" in blockers
    assert "proxy_outcomes_are_parameter_sensitive" in blockers
    assert "verified_proxy_population_is_smaller_than_design_minimum" in blockers

    statistical = cast(dict[str, Any], report["statistical_design"])
    positive = cast(int, statistical["required_positive_count"])
    negative = cast(int, statistical["required_negative_count"])
    assert positive >= 2 and negative >= 2
    assert statistical["required_total_count"] == positive + negative
    assert cast(dict[str, Any], report["population_comparison"])["nominal_count_shortfall"] == (
        positive + negative - 93
    )
    matching = cast(dict[str, Any], report["matching_contract"])
    assert matching["forbidden_label_authorities"] == ["ip_proxy", "programme_metadata"]
    assert matching["matching_with_replacement"] is False
    assert matching["post_outcome_matching_variables_permitted"] is False
    assert cast(dict[str, Any], report["source_proxy_audit"])["label_authority_admission"] is False
    assert cast(dict[str, Any], report["design_spec"])["file_sha256"] == file_sha256(design_path)
    assert report["payload_sha256"] == canonical_json_sha256({**report, "payload_sha256": None})
    assert report["admitted_shot_ids"] == []
    encoded = json.dumps(report)
    assert '"shot_stability"' not in encoded
    assert '"outcomes_by_parameter"' not in encoded
    assert '"sensitivity_points"' not in encoded


def test_sample_size_searches_return_the_first_admissible_counts() -> None:
    """Prove the reported minima, rather than accepting a rounded guess."""
    n = minimum_binomial_sample_size(
        expected_rate=0.8,
        confidence_level=0.95,
        max_half_width=0.05,
    )
    assert wilson_half_width(expected_rate=0.8, sample_size=n, confidence_level=0.95) <= 0.05
    assert wilson_half_width(expected_rate=0.8, sample_size=n - 1, confidence_level=0.95) > 0.05

    positive, negative, standard_error, half_width, power = minimum_auc_group_sizes(
        target_auc=0.75,
        null_auc=0.5,
        confidence_level=0.95,
        target_power=0.9,
        max_ci_half_width=0.05,
        control_to_case_ratio=2,
    )
    assert negative == positive * 2
    assert standard_error > 0.0
    assert half_width <= 0.05
    assert power >= 0.9
    _, previous_half_width, previous_power = _auc_design_metrics(
        target_auc=0.75,
        null_auc=0.5,
        positive_count=positive - 1,
        negative_count=(positive - 1) * 2,
        confidence_level=0.95,
    )
    assert previous_half_width > 0.05 or previous_power < 0.9


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("confidence_level", 0.5, "exceed 0.5"),
        ("confidence_level", float("nan"), "finite"),
        ("target_power", 0.5, "exceed 0.5"),
        ("expected_sensitivity", 0.0, "within"),
        ("expected_specificity", 1.0, "within"),
        ("max_sensitivity_ci_half_width", 0.0, "half_width"),
        ("max_specificity_ci_half_width", 0.5, "half_width"),
        ("null_auc", 0.0, "within"),
        ("null_auc", 0.4, "must equal 0.5"),
        ("target_auc", 1.0, "within"),
        ("target_auc", 0.5, "exceed null_auc"),
        ("max_auc_ci_half_width", float("inf"), "half_width"),
        ("control_to_case_ratio", 0, "positive integer"),
        ("control_to_case_ratio", True, "must be an integer"),
    ],
)
def test_design_rejects_invalid_statistical_parameters(tmp_path: Path, field: str, value: object, message: str) -> None:
    """Reject non-finite, unbounded, or statistically incoherent assumptions."""
    design_path, proxy_path = _inputs(tmp_path)
    _rewrite(design_path, lambda payload: payload.update({field: value}))
    with pytest.raises(CohortDesignError, match=message):
        design_matched_cohort(
            design_spec_path=design_path,
            proxy_sensitivity_path=proxy_path,
            generated_at=_FIXED_TS,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.pop("machine"), "fields do not match"),
        (lambda payload: payload.update({"schema_version": "v0"}), "unsupported cohort-design"),
        (lambda payload: payload.update({"machine": "DIII-D"}), "machine must be MAST"),
        (lambda payload: payload.update({"design_status": "approved"}), "must remain draft"),
        (lambda payload: payload.update({"excluded_outcomes": ["ambiguous"]}), "excluded_outcomes"),
        (lambda payload: payload.update({"caliper_policy_status": "complete"}), "calipers"),
        (lambda payload: payload.update({"admissible_label_authorities": []}), "must not be empty"),
        (
            lambda payload: payload.update({"admissible_label_authorities": ["facility", "ip_proxy"]}),
            "non-independent",
        ),
        (
            lambda payload: payload.update({"admissible_label_authorities": ["facility", "facility"]}),
            "unique, and sorted",
        ),
        (lambda payload: payload.update({"required_channels": ["Ip_MA"]}), "canonical 11-channel"),
        (
            lambda payload: payload.update({"exact_match_fields": ["campaign", "configuration"]}),
            "lacks required fields",
        ),
        (lambda payload: payload.update({"exact_match_fields": [""]}), "non-empty strings"),
        (lambda payload: payload.update({"required_channels": "Ip_MA"}), "array of strings"),
        (lambda payload: payload.update({"confidence_level": "high"}), "must be numeric"),
        (lambda payload: payload.update({"control_to_case_ratio": 1.0}), "must be an integer"),
    ],
)
def test_design_schema_and_evidence_policy_fail_closed(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None], message: str
) -> None:
    """Forbid weak authorities and incomplete matching/diagnostic contracts."""
    design_path, proxy_path = _inputs(tmp_path)
    _rewrite(design_path, mutate)
    with pytest.raises(CohortDesignError, match=message):
        design_matched_cohort(
            design_spec_path=design_path,
            proxy_sensitivity_path=proxy_path,
            generated_at=_FIXED_TS,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update({"schema_version": "v0"}), "unsupported proxy-sensitivity"),
        (lambda payload: payload.update({"status": "ready"}), "must remain status:blocked"),
        (
            lambda payload: payload.update({"source_dataset_report_file_sha256": "BAD"}),
            "lowercase SHA-256",
        ),
        (lambda payload: payload.update({"verified_shot_artifact_count": 0}), "counts are inconsistent"),
        (lambda payload: payload.update({"stable_shot_count": -1}), "counts are inconsistent"),
        (lambda payload: payload.update({"unstable_shot_count": -1}), "counts are inconsistent"),
        (lambda payload: payload.update({"stable_shot_count": 41}), "counts are inconsistent"),
        (lambda payload: payload.update({"unstable_shot_ids": []}), "does not match"),
        (
            lambda payload: payload["claim_boundary"].update({"cohort_admission": True}),
            "every admission claim false",
        ),
        (
            lambda payload: payload["claim_boundary"].update({"cohort_admission": 0}),
            "every admission claim false",
        ),
        (
            lambda payload: payload["claim_boundary"].update({"cohort_admission": 0.0}),
            "every admission claim false",
        ),
        (lambda payload: payload.update({"claim_boundary": {}}), "every admission claim false"),
        (
            lambda payload: payload["programme_comparison"].update({"independent_validation_claim": True}),
            "must not assert independent",
        ),
        (lambda payload: payload.update({"programme_comparison": []}), "must not assert independent"),
        (lambda payload: payload.update({"dataset_id": ""}), "dataset_id"),
        (lambda payload: payload.update({"verified_shot_artifact_count": True}), "must be an integer"),
    ],
)
def test_proxy_audit_contract_is_reverified(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None], message: str
) -> None:
    """Reject promotion, count drift, digest drift, and false independence."""
    design_path, proxy_path = _inputs(tmp_path)
    _rewrite(proxy_path, mutate)
    with pytest.raises(CohortDesignError, match=message):
        design_matched_cohort(
            design_spec_path=design_path,
            proxy_sensitivity_path=proxy_path,
            generated_at=_FIXED_TS,
        )


def test_payload_tamper_and_ambiguous_json_are_rejected(tmp_path: Path) -> None:
    """Reject substituted bytes, duplicate keys, non-object roots, and bad text."""
    design_path, proxy_path = _inputs(tmp_path)
    assert _load_json(design_path)["machine"] == "MAST"
    payload = cast(dict[str, Any], json.loads(design_path.read_text(encoding="utf-8")))
    payload["target_power"] = 0.8
    design_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CohortDesignError, match="does not match"):
        design_matched_cohort(
            design_spec_path=design_path,
            proxy_sensitivity_path=proxy_path,
            generated_at=_FIXED_TS,
        )

    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(CohortDesignError, match="duplicate JSON key"):
        _load_json(invalid)
    invalid.write_text("[]", encoding="utf-8")
    with pytest.raises(CohortDesignError, match="root must be an object"):
        _load_json(invalid)
    invalid.write_bytes(b"\xff")
    with pytest.raises(CohortDesignError, match="cannot read verified JSON"):
        _load_json(invalid)
    with pytest.raises(CohortDesignError, match="cannot read verified JSON"):
        _load_json(tmp_path / "missing.json")


def test_input_provenance_hashes_the_same_bytes_that_are_parsed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent a path substitution from mixing parsed and hashed generations."""
    design_path, proxy_path = _inputs(tmp_path)
    original = design_path.read_bytes()
    replacement_payload = _design_payload()
    replacement_payload["target_power"] = 0.85
    replacement = json.dumps(_seal(replacement_payload), sort_keys=True).encode()
    original_read_bytes = Path.read_bytes
    substituted = False

    def substitute_after_read(path: Path) -> bytes:
        nonlocal substituted
        raw = original_read_bytes(path)
        if path == design_path and not substituted:
            substituted = True
            path.write_bytes(replacement)
        return raw

    monkeypatch.setattr(Path, "read_bytes", substitute_after_read)
    report = design_matched_cohort(
        design_spec_path=design_path,
        proxy_sensitivity_path=proxy_path,
        generated_at=_FIXED_TS,
    )

    reported = cast(dict[str, Any], report["design_spec"])["file_sha256"]
    assert reported == hashlib.sha256(original).hexdigest()
    assert reported != hashlib.sha256(replacement).hexdigest()


def test_empty_timestamp_and_non_shortfall_path(tmp_path: Path) -> None:
    """Require run provenance and keep capacity arithmetic bounded at zero."""
    design_path, proxy_path = _inputs(tmp_path)
    with pytest.raises(CohortDesignError, match="generated_at"):
        design_matched_cohort(
            design_spec_path=design_path,
            proxy_sensitivity_path=proxy_path,
            generated_at="",
        )
    _write_json(proxy_path, _proxy_payload(verified=10_000, stable=10_000, unstable=0))
    report = design_matched_cohort(
        design_spec_path=design_path,
        proxy_sensitivity_path=proxy_path,
        generated_at=_FIXED_TS,
    )
    assert cast(dict[str, Any], report["population_comparison"])["nominal_count_shortfall"] == 0
    assert "proxy_outcomes_are_parameter_sensitive" not in cast(list[str], report["blockers"])
    assert "verified_proxy_population_is_smaller_than_design_minimum" not in cast(list[str], report["blockers"])
    assert report["status"] == "blocked"


def test_planning_math_rejects_invalid_direct_calls_and_caps() -> None:
    """Keep public helpers bounded even when called outside the JSON gate."""
    with pytest.raises(CohortDesignError, match="sample_size"):
        wilson_half_width(expected_rate=0.8, sample_size=0, confidence_level=0.95)
    with pytest.raises(CohortDesignError, match="exceed 0.5"):
        wilson_half_width(expected_rate=0.8, sample_size=10, confidence_level=0.5)
    with pytest.raises(CohortDesignError, match="max_n"):
        minimum_binomial_sample_size(
            expected_rate=0.8,
            confidence_level=0.95,
            max_half_width=0.01,
            max_n=1,
        )
    with pytest.raises(CohortDesignError, match="positive"):
        minimum_binomial_sample_size(
            expected_rate=0.8,
            confidence_level=0.95,
            max_half_width=0.05,
            max_n=0,
        )
    with pytest.raises(CohortDesignError, match="at least two"):
        _auc_standard_error(auc=0.75, positive_count=1, negative_count=2)
    with pytest.raises(CohortDesignError, match="exceed null_auc"):
        minimum_auc_group_sizes(
            target_auc=0.5,
            null_auc=0.5,
            confidence_level=0.95,
            target_power=0.9,
            max_ci_half_width=0.05,
            control_to_case_ratio=1,
        )
    with pytest.raises(CohortDesignError, match="must equal 0.5"):
        minimum_auc_group_sizes(
            target_auc=0.75,
            null_auc=0.4,
            confidence_level=0.95,
            target_power=0.9,
            max_ci_half_width=0.05,
            control_to_case_ratio=1,
        )
    with pytest.raises(CohortDesignError, match="positive integer"):
        minimum_auc_group_sizes(
            target_auc=0.75,
            null_auc=0.5,
            confidence_level=0.95,
            target_power=0.9,
            max_ci_half_width=0.05,
            control_to_case_ratio=0,
        )
    with pytest.raises(CohortDesignError, match="at least two"):
        minimum_auc_group_sizes(
            target_auc=0.75,
            null_auc=0.5,
            confidence_level=0.95,
            target_power=0.9,
            max_ci_half_width=0.05,
            control_to_case_ratio=1,
            max_positive_count=1,
        )
    with pytest.raises(CohortDesignError, match="exceeds"):
        minimum_auc_group_sizes(
            target_auc=0.51,
            null_auc=0.5,
            confidence_level=0.999,
            target_power=0.999,
            max_ci_half_width=0.001,
            control_to_case_ratio=1,
            max_positive_count=2,
        )


def test_cli_writes_the_same_self_digested_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Exercise the documented command-line evidence path."""
    design_path, proxy_path = _inputs(tmp_path)
    output = tmp_path / "reports" / "cohort.json"
    assert (
        main(
            [
                "--design-spec",
                str(design_path),
                "--proxy-sensitivity",
                str(proxy_path),
                "--generated-at",
                _FIXED_TS,
                "--json-out",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["payload_sha256"] == canonical_json_sha256({**payload, "payload_sha256": None})
    assert "matched-cohort design blocked" in capsys.readouterr().out

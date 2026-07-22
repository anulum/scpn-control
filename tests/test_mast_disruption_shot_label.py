# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tests for the MAST disruption shot-label contract
"""Contract tests for :mod:`validation.mast_disruption_shot_label`."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from validation.mast_disruption_shot_label import (
    IP_QUENCH_PROXY_ALGORITHM_SCHEMA,
    SHOT_LABEL_RECORD_SCHEMA,
    LabelConfidence,
    ShotLabelRecord,
    ShotLabelRecordError,
    derive_ip_quench_proxy,
    ip_quench_proxy_algorithm,
    shot_label_record_from_dict,
    shot_label_record_from_json,
)
from validation.mast_source_object_manifest import canonical_json_sha256


def _time(n: int) -> NDArray[np.float64]:
    return np.arange(n, dtype=np.float64) * 1.0e-3


def _disruptive_ip() -> NDArray[np.float64]:
    ip = np.ones(300, dtype=np.float64)
    ip[:100] = np.linspace(0.0, 1.0, 100)
    ip[250] = 0.5
    ip[251:] = 0.05
    return ip


def _manual_record(**updates: Any) -> ShotLabelRecord:
    fields: dict[str, Any] = {
        "schema_version": SHOT_LABEL_RECORD_SCHEMA,
        "machine": "MAST",
        "shot_id": 30421,
        "outcome": "disruption",
        "programme_class": "spontaneous",
        "thermal_quench_time_s": 0.249,
        "current_quench_time_s": 0.250,
        "termination_time_s": 0.251,
        "authority": "independent_manual_review",
        "label_detail": "independent double review of calibrated diagnostics",
        "confidence": LabelConfidence(kind="bounded_score", value=0.9, method="two-reviewer consensus score"),
        "reviewers": ("reviewer-a", "reviewer-b"),
        "source_channels": ("equilibrium.q95", "summary.ip"),
        "source_digest": "a" * 64,
        "algorithm_digest": "b" * 64,
    }
    fields.update(updates)
    return ShotLabelRecord(**fields)


def test_ip_quench_proxy_record_round_trips_with_exact_authority() -> None:
    """Round-trip a disruptive Ip proxy without upgrading its authority."""
    result = derive_ip_quench_proxy(
        _disruptive_ip(),
        _time(300),
        shot_id=30421,
        programme_class="forced_vde",
    )
    assert result.is_disruption is True
    assert result.onset_index == 249
    assert result.termination_index == 251
    assert result.classification == "current_quench"
    assert result.record.outcome == "disruption"
    assert result.record.authority == "ip_proxy"
    assert result.record.programme_class == "forced_vde"
    assert result.record.thermal_quench_time_s is None
    assert result.record.current_quench_time_s == pytest.approx(0.249)
    assert result.record.termination_time_s == pytest.approx(0.251)
    assert result.record.confidence.kind == "unquantified"
    assert result.record.independent_of_ip_features is False
    payload = result.record.to_dict()
    assert shot_label_record_from_dict(payload) == result.record
    assert shot_label_record_from_json(json.dumps(payload)) == result.record


@pytest.mark.parametrize(
    ("ip", "outcome", "classification", "termination_index"),
    [
        (np.zeros(50), "ambiguous", "no_current", -1),
        (np.ones(50), "non_disruption", "no_quench", -1),
        (
            np.concatenate([np.linspace(0.0, 1.0, 100), np.ones(100), np.linspace(1.0, 0.0, 100)]),
            "non_disruption",
            "slow_rampdown",
            280,
        ),
    ],
)
def test_ip_proxy_outcomes_do_not_collapse_into_one_negative_class(
    ip: NDArray[np.float64],
    outcome: str,
    classification: str,
    termination_index: int,
) -> None:
    """Keep ambiguous, quench-free, and slow-ramp outcomes distinct."""
    result = derive_ip_quench_proxy(ip, _time(ip.shape[0]), shot_id=11766)
    assert result.is_disruption is False
    assert result.onset_index == -1
    assert result.termination_index == termination_index
    assert result.classification == classification
    assert result.record.outcome == outcome
    assert result.record.current_quench_time_s is None


def test_proxy_digests_bind_signal_values_and_algorithm_parameters() -> None:
    """Bind source and algorithm digests to their independent inputs."""
    ip = _disruptive_ip()
    baseline = derive_ip_quench_proxy(ip, _time(ip.shape[0]), shot_id=30421)
    changed_ip = ip.copy()
    changed_ip[150] = 0.9
    signal_variant = derive_ip_quench_proxy(changed_ip, _time(ip.shape[0]), shot_id=30421)
    algorithm_variant = derive_ip_quench_proxy(
        ip,
        _time(ip.shape[0]),
        shot_id=30421,
        quench_window_ms=4.0,
    )
    assert signal_variant.record.source_digest != baseline.record.source_digest
    assert algorithm_variant.record.algorithm_digest != baseline.record.algorithm_digest


def test_proxy_algorithm_is_self_digested_and_explicitly_non_ground_truth() -> None:
    """Describe the proxy algorithm without a ground-truth implication."""
    algorithm = ip_quench_proxy_algorithm(drop_fraction=0.8, quench_window_ms=5.0)
    assert algorithm["schema_version"] == IP_QUENCH_PROXY_ALGORITHM_SCHEMA
    assert algorithm["authority"] == "ip_proxy"
    assert algorithm["inputs"] == ["Ip_MA", "time_s"]
    assert "does not infer" in str(algorithm["missing_ground_truth"])
    digest = algorithm["payload_sha256"]
    assert isinstance(digest, str) and len(digest) == 64


@pytest.mark.parametrize(
    ("ip", "time_s", "message"),
    [
        (np.array([]), np.array([]), "non-empty 1-D"),
        (np.ones((2, 2)), np.arange(4.0), "non-empty 1-D"),
        (np.ones(2), np.ones(3), "identical shape"),
        (np.array([1.0, np.nan]), np.array([0.0, 1.0]), "must be finite"),
        (np.ones(2), np.array([0.0, np.inf]), "must be finite"),
        (np.ones(3), np.array([0.0, 1.0, 1.0]), "strictly increasing"),
    ],
)
def test_proxy_rejects_invalid_physical_arrays(
    ip: NDArray[np.float64],
    time_s: NDArray[np.float64],
    message: str,
) -> None:
    """Reject arrays that cannot represent one ordered shot trace."""
    with pytest.raises(ShotLabelRecordError, match=message):
        derive_ip_quench_proxy(ip, time_s, shot_id=1)


@pytest.mark.parametrize(
    ("drop_fraction", "quench_window_ms", "message"),
    [
        (0.0, 5.0, "drop_fraction"),
        (1.0, 5.0, "drop_fraction"),
        (float("nan"), 5.0, "drop_fraction"),
        (0.8, 0.0, "quench_window_ms"),
        (0.8, float("inf"), "quench_window_ms"),
    ],
)
def test_proxy_rejects_invalid_algorithm_parameters(
    drop_fraction: float,
    quench_window_ms: float,
    message: str,
) -> None:
    """Reject thresholds that do not define a finite bounded detector."""
    with pytest.raises(ShotLabelRecordError, match=message):
        ip_quench_proxy_algorithm(drop_fraction=drop_fraction, quench_window_ms=quench_window_ms)


def test_independent_manual_record_requires_two_reviewers_and_is_independent() -> None:
    """Require two identities before a manual record is independent."""
    record = _manual_record()
    assert record.independent_of_ip_features is True
    assert shot_label_record_from_dict(record.to_dict()) == record
    with pytest.raises(ShotLabelRecordError, match="at least two reviewers"):
        _manual_record(reviewers=("reviewer-a",))


@pytest.mark.parametrize("authority", ["facility", "defuse"])
def test_facility_and_defuse_authorities_are_independent(authority: str) -> None:
    """Classify facility and DEFUSE labels as independent of Ip features."""
    record = _manual_record(authority=authority, reviewers=())
    assert record.independent_of_ip_features is True


def test_programme_metadata_cannot_assert_quench_event_times() -> None:
    """Prevent programme metadata from inventing diagnostic event times."""
    with pytest.raises(ShotLabelRecordError, match="cannot assert TQ or CQ"):
        _manual_record(authority="programme_metadata", reviewers=())


@pytest.mark.parametrize(
    ("confidence", "message"),
    [
        (LabelConfidence(kind="bounded_score", value=0.5, method="bounded review rubric"), "ip_proxy confidence"),
        (LabelConfidence(kind="unquantified", value=None, method="not calibrated"), "Ip_MA"),
    ],
)
def test_ip_proxy_record_rejects_false_authority_metadata(confidence: LabelConfidence, message: str) -> None:
    """Reject calibrated-looking confidence and unrelated proxy sources."""
    updates: dict[str, Any] = {
        "authority": "ip_proxy",
        "reviewers": (),
        "thermal_quench_time_s": None,
        "current_quench_time_s": None,
        "termination_time_s": None,
        "outcome": "ambiguous",
        "confidence": confidence,
    }
    if message == "Ip_MA":
        updates["source_channels"] = ("summary.ip",)
    else:
        updates["source_channels"] = ("Ip_MA", "time_s")
    with pytest.raises(ShotLabelRecordError, match=message):
        _manual_record(**updates)


@pytest.mark.parametrize(
    ("kind", "value", "method", "message"),
    [
        ("unquantified", 0.5, "rule", "cannot carry"),
        ("bounded_score", None, "rule", "within"),
        ("bounded_score", -0.1, "rule", "within"),
        ("calibrated_probability", float("nan"), "calibration", "within"),
        ("unknown", None, "rule", "unsupported"),
        ("unquantified", None, "", "non-empty"),
    ],
)
def test_confidence_contract_rejects_invalid_semantics(
    kind: Any,
    value: float | None,
    method: str,
    message: str,
) -> None:
    """Require each confidence value to carry valid bounded semantics."""
    with pytest.raises(ShotLabelRecordError, match=message):
        LabelConfidence(kind=kind, value=value, method=method)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"schema_version": "v0"}, "schema_version"),
        ({"machine": ""}, "machine"),
        ({"shot_id": 0}, "shot_id"),
        ({"outcome": "unknown"}, "unsupported outcome"),
        ({"programme_class": "aborted"}, "unsupported programme_class"),
        ({"authority": "owner_guess"}, "unsupported authority"),
        ({"label_detail": ""}, "label_detail"),
        ({"thermal_quench_time_s": 0.252}, "cannot follow current"),
        ({"termination_time_s": 0.248}, "cannot follow termination"),
        ({"current_quench_time_s": float("inf")}, "finite"),
        ({"outcome": "ambiguous"}, "only disruption"),
        ({"reviewers": ("reviewer-b", "reviewer-a")}, "sorted"),
        ({"reviewers": ("", "reviewer-b")}, "non-empty"),
        ({"source_channels": ("summary.ip", "summary.ip")}, "source_channels"),
        ({"source_digest": "A" * 64}, "source_digest"),
        ({"algorithm_digest": "b" * 63}, "algorithm_digest"),
    ],
)
def test_record_rejects_inconsistent_taxonomy_and_provenance(updates: dict[str, Any], message: str) -> None:
    """Reject contradictory outcome, timing, review, and digest fields."""
    with pytest.raises(ShotLabelRecordError, match=message):
        _manual_record(**updates)


def test_non_manual_authority_rejects_reviewers() -> None:
    """Reserve reviewer identities for independent manual authority."""
    with pytest.raises(ShotLabelRecordError, match="only for independent"):
        _manual_record(authority="facility")


def test_parser_rejects_tamper_duplicate_keys_and_non_object_json() -> None:
    """Fail closed on tampered, ambiguous, or malformed JSON input."""
    payload = _manual_record().to_dict()
    payload["shot_id"] = 30422
    with pytest.raises(ShotLabelRecordError, match="payload_sha256"):
        shot_label_record_from_dict(payload)
    with pytest.raises(ShotLabelRecordError, match="duplicate JSON key"):
        shot_label_record_from_json('{"schema_version":"x","schema_version":"y"}')
    with pytest.raises(ShotLabelRecordError, match="root must be an object"):
        shot_label_record_from_json("[]")
    with pytest.raises(ShotLabelRecordError, match="invalid shot-label JSON"):
        shot_label_record_from_json("{")


def test_parser_rejects_schema_shape_and_derived_flag_mismatch() -> None:
    """Reject missing fields and a forged independence projection."""
    payload = _manual_record().to_dict()
    del payload["label_detail"]
    with pytest.raises(ShotLabelRecordError, match="fields do not match"):
        shot_label_record_from_dict(payload)

    payload = _manual_record().to_dict()
    payload["independent_of_ip_features"] = False
    payload["payload_sha256"] = None
    payload["payload_sha256"] = canonical_json_sha256(payload)
    with pytest.raises(ShotLabelRecordError, match="contradicts"):
        shot_label_record_from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("machine", "", "non-empty string"),
        ("shot_id", True, "integer"),
        ("reviewers", "reviewer-a", "string array"),
        ("source_channels", [1], "string array"),
        ("thermal_quench_time_s", "0.2", "numeric or null"),
        ("confidence", [], "confidence must contain"),
    ],
)
def test_parser_rejects_wrong_json_types(field: str, value: object, message: str) -> None:
    """Reject JSON values whose runtime types violate the schema."""
    payload = _manual_record().to_dict()
    payload[field] = value
    payload["payload_sha256"] = None
    payload["payload_sha256"] = canonical_json_sha256(payload)
    with pytest.raises(ShotLabelRecordError, match=message):
        shot_label_record_from_dict(payload)

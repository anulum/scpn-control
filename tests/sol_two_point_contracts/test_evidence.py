# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — SOL self-sealed evidence semantic contracts.
"""Reject resealed contradictions through the public v1 evidence reader."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import pytest

from validation.sol_two_point_contracts.evidence import (
    SOL_TWO_POINT_SCHEMA_VERSION,
    build_evidence,
    validate_evidence_payload,
)
from validation.sol_two_point_contracts.models import SOLConfig, validate_sol_two_point


def sealed(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply the documented sorted compact ASCII self-digest without private code."""
    payload["payload_sha256"] = ""
    data = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    payload["payload_sha256"] = hashlib.sha256(data.encode()).hexdigest()
    return payload


@pytest.mark.parametrize("epsilon", [0.3, 0.5, 0.9])
def test_complete_reports_replay_in_both_scaling_probe_domains(epsilon: float) -> None:
    """The public builder/reader preserve passing content and all declared arrays."""
    result = validate_sol_two_point(config=SOLConfig(2.0, 2.0 * epsilon, 3.5, 0.4))
    evidence = build_evidence(result, target_id="readable λ\nsecond line")
    restored = json.loads(json.dumps(evidence, allow_nan=False))
    assert validate_evidence_payload(restored) is True
    assert restored["target_id"] == "readable λ\nsecond line"


def test_complete_failed_report_returns_false() -> None:
    """A real stricter diagnostic failure remains valid sealed evidence of failure."""
    evidence = build_evidence(validate_sol_two_point(exact_tol=1e-30), target_id="strict")
    assert validate_evidence_payload(evidence) is False


def test_minimal_truthy_string_cannot_be_admitted() -> None:
    """A valid digest does not make a minimal truthy string into complete evidence."""
    payload = sealed({"schema_version": SOL_TWO_POINT_SCHEMA_VERSION, "passed": "false"})
    with pytest.raises(ValueError, match="declared fields"):
        validate_evidence_payload(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_id", " "),
        ("target_id", 1),
        ("generated_utc", 0),
        ("generated_utc", "2026-02-30T00:00:00Z"),
        ("generated_utc", "2026-1-1T00:00:00Z"),
        ("config", []),
        ("config", {"r0": 1.7}),
        ("operating_points", []),
        ("operating_points", {}),
        ("operating_points", [[1.0]]),
        ("operating_points", [None]),
        ("operating_points", [[0.0, 1.0]]),
        ("exact_tol", float("inf")),
        ("connection_length_rel_error", -1.0),
        ("max_flux_mapping_rel_error", "0"),
        ("max_conduction_rel_error", 1.0),
        ("scaling", []),
        ("scaling", [None] * 4),
        ("connection_passed", "false"),
        ("detachment", {}),
        ("detachment_passed", False),
        ("passed", "false"),
        ("passed", False),
    ],
)
def test_resealed_invalid_content_is_refused(field: str, value: object) -> None:
    """Recomputing the digest cannot bypass shapes, finite domains or gate consistency."""
    payload = build_evidence(validate_sol_two_point(), target_id="negative")
    payload[field] = value
    with pytest.raises(ValueError):
        validate_evidence_payload(sealed(payload))


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "wrong"),
        ("expected_ratio", 1.0),
        ("measured_ratio", 0.0),
        ("rel_error", 0.2),
    ],
)
def test_resealed_scaling_contradiction_is_refused(field: str, value: object) -> None:
    """Each ordered probe has its declared exponent and arithmetic relative error."""
    payload = build_evidence(validate_sol_two_point(), target_id="scaling")
    payload["scaling"][0][field] = value
    with pytest.raises(ValueError):
        validate_evidence_payload(sealed(payload))


def test_resealed_scaling_maximum_is_checked_independently() -> None:
    """A consistent individual observation cannot disagree with its declared maximum."""
    payload = build_evidence(validate_sol_two_point(), target_id="maximum")
    check = payload["scaling"][0]
    check["measured_ratio"] = check["expected_ratio"] * 1.1
    check["rel_error"] = abs(check["measured_ratio"] - check["expected_ratio"]) / check["expected_ratio"]
    with pytest.raises(ValueError, match="max_scaling_rel_error"):
        validate_evidence_payload(sealed(payload))


@pytest.mark.parametrize(
    "field,value",
    [
        ("critical_density_19", 0.0),
        ("detached_below_critical", 0),
        ("detached_above_critical", "true"),
    ],
)
def test_resealed_detachment_domains_are_checked(field: str, value: object) -> None:
    """Critical density and both boundary flags have strict numeric/boolean domains."""
    payload = build_evidence(validate_sol_two_point(), target_id="detachment")
    payload["detachment"][field] = value
    with pytest.raises(ValueError):
        validate_evidence_payload(sealed(payload))


def test_internally_consistent_detachment_failure_is_not_authenticated() -> None:
    """The reader reports a coherent false outcome without authenticating the producer."""
    result = validate_sol_two_point()
    evidence = build_evidence(
        replace(
            result,
            detachment=replace(result.detachment, detached_below_critical=True),
            detachment_passed=False,
            passed=False,
        ),
        target_id="declared-failure",
    )
    assert validate_evidence_payload(evidence) is False


@pytest.mark.parametrize("digest", [None, 1, "", "A" * 64, "g" * 64, "0" * 63, "0" * 65])
def test_reader_refuses_invalid_digest_syntax(digest: object) -> None:
    """Only exactly 64 lowercase hexadecimal characters constitute a digest."""
    payload = build_evidence(validate_sol_two_point(), target_id="digest")
    payload["payload_sha256"] = digest
    with pytest.raises(ValueError, match="SHA-256 hex digest"):
        validate_evidence_payload(payload)


def test_reader_refuses_unserializable_content() -> None:
    """A typed mapping still has to be representable as finite JSON."""
    payload = build_evidence(validate_sol_two_point(), target_id="json")
    payload["unknown"] = {1, 2}
    with pytest.raises(ValueError, match="serializable finite JSON"):
        validate_evidence_payload(payload)


def test_reader_refuses_non_mapping_payload() -> None:
    """An unrelated decoded JSON value refuses with the authored schema error."""
    with pytest.raises(ValueError, match="unsupported"):
        validate_evidence_payload(cast(Mapping[str, Any], []))


def test_builder_refuses_overflowing_report_number() -> None:
    """A manually constructed result cannot seal an unrepresentable numeric error."""
    result = replace(validate_sol_two_point(), max_conduction_rel_error=cast(float, 10**400))
    with pytest.raises(ValueError, match="finite"):
        build_evidence(result, target_id="large")

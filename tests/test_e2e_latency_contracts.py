# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — E2E Latency Evidence Validation

"""Public reader boundaries using real observations and labeled authored derivatives."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import pytest
from e2e_latency_observation import derived_report
from e2e_latency_observation import measured_report as measured_report

from validation.validate_e2e_latency_evidence import (
    build_e2e_latency_evidence_payload,
    validate_e2e_latency_evidence,
)


def test_observed_report_keeps_local_boundary_and_inclusive_budget(measured_report: Path) -> None:
    """The real report passes local checks, equal p95 budget, and fails a zero budget."""
    result = validate_e2e_latency_evidence(measured_report, require_target_hardware=False)
    assert result.status == "pass" and result.errors == () and result.p95_us is not None
    equal = validate_e2e_latency_evidence(measured_report, require_target_hardware=False, max_e2e_p95_us=result.p95_us)
    assert equal.status == "pass"
    zero = validate_e2e_latency_evidence(measured_report, require_target_hardware=False, max_e2e_p95_us=0.0)
    assert zero.status == "fail" and any("exceeds admission threshold 0.0" in e for e in zero.errors)
    qualified = validate_e2e_latency_evidence(measured_report)
    assert qualified.status == "fail" and len(qualified.errors) == 3
    assert qualified.target_hardware_id is None and qualified.target_hardware_class is None
    assert qualified.rt_kernel is None


@pytest.mark.parametrize("budget", [math.nan, math.inf, -math.inf, -1.0, True, "1000", 10**400])
def test_invalid_budget_refused_before_report_io(tmp_path: Path, budget: object) -> None:
    """Invalid runtime budget objects fail before an absent report is opened."""
    with pytest.raises(ValueError, match="max_e2e_p95_us must be a finite non-negative number"):
        validate_e2e_latency_evidence(tmp_path / "absent.json", max_e2e_p95_us=cast(float, budget))


@pytest.mark.parametrize(
    "timestamp", [None, "", "nonsense", "2026-10-02", "2026-10-02T10:00:00", "2026-10-02T10:00:00+02:00"]
)
def test_non_utc_authored_timestamp_is_refused(measured_report: Path, tmp_path: Path, timestamp: object) -> None:
    """Malformed, naive and nonzero-offset derivatives cannot declare UTC."""
    report = derived_report(measured_report, tmp_path / "declared.json", "generated_utc", timestamp)
    result = validate_e2e_latency_evidence(report, require_target_hardware=False)
    assert result.status == "fail"
    assert "generated_utc must record an ISO-8601 UTC timestamp" in result.errors


def test_explicit_zero_offset_derivative_is_accepted(measured_report: Path, tmp_path: Path) -> None:
    """An explicit UTC offset is accepted without claiming timestamp freshness."""
    report = derived_report(measured_report, tmp_path / "declared.json", "generated_utc", "2020-01-01T00:00:00+00:00")
    assert validate_e2e_latency_evidence(report, require_target_hardware=False).status == "pass"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("iterations", True, "iterations must"),
        ("warmup", -1, "warmup must"),
        ("target_hardware", None, "target_hardware must"),
        ("target_hardware.id", 7, "target_hardware.id must"),
        ("target_hardware.class", " UNKNOWN ", "target_hardware.class must"),
        ("kernel_only_us", None, "kernel_only_us must"),
        ("e2e_us.p95", None, "e2e_us.p95 must"),
        ("e2e_us.p95", True, "e2e_us.p95 must"),
        ("e2e_us.p95", -1.0, "e2e_us.p95 must"),
        ("e2e_us.p95", math.nan, "e2e_us.p95 must"),
        ("e2e_us.p99", 0.01, "percentiles must satisfy"),
        ("e2e_overhead_factor", None, "e2e_overhead_factor must"),
        ("e2e_overhead_factor", 999999.0, "must match p50"),
        ("command", None, "command must"),
        ("command", "unrelated.py", "command must"),
        ("evidence_class", "hardware", "evidence_class must"),
        ("production_claim_allowed", True, "production_claim_allowed must"),
        ("context", None, "context must"),
        ("context.cpu_affinity", None, "cpu_affinity must"),
        ("context.cpu_affinity", [], "cpu_affinity must"),
        ("context.cpu_affinity", [True], "cpu_affinity must"),
        ("context.cpu_affinity", [-1], "cpu_affinity must"),
        ("context.isolation_method", "", "isolation_method must"),
        ("context.isolation_method", None, "isolation_method must"),
        ("context.loadavg_start", [0], "loadavg_start must"),
        ("context.loadavg_start", [True, 0, 0], "loadavg_start must"),
        ("context.loadavg_start", ["0", 0, 0], "loadavg_start must"),
        ("context.loadavg_end", [math.inf, 0, 0], "loadavg_end must"),
        ("context.heavy_jobs_running", "", "heavy_jobs_running must"),
        ("context.heavy_jobs_running", None, "heavy_jobs_running must"),
    ],
)
def test_authored_field_derivative_is_refused(
    measured_report: Path,
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """Each altered declaration is refused through the public persisted reader."""
    report = derived_report(measured_report, tmp_path / "declared.json", field, value)
    result = validate_e2e_latency_evidence(report)
    assert result.status == "fail" and any(message in e for e in result.errors)


@pytest.mark.parametrize("field", ["claim_status", "claim_boundary", "schema_version", "context.governor"])
def test_unsealed_declaration_change_is_refused(measured_report: Path, tmp_path: Path, field: str) -> None:
    """Changes to fixed builder fields or required context cannot hide behind the checksum."""
    payload = json.loads(measured_report.read_text(encoding="utf-8"))
    if field == "context.governor":
        payload["context"].pop("governor")
    else:
        payload[field] = "altered"
    report = tmp_path / "tampered.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    result = validate_e2e_latency_evidence(report, require_target_hardware=False)
    assert result.status == "fail" and any("payload_sha256 does not match" in e for e in result.errors)
    assert len(result.errors) >= 2


@pytest.mark.parametrize("digest", [None, "short", "g" * 64])
def test_invalid_declared_digest_is_refused(measured_report: Path, tmp_path: Path, digest: object) -> None:
    """Malformed or mismatching digests do not validate the observed payload."""
    payload = json.loads(measured_report.read_text(encoding="utf-8"))
    payload["payload_sha256"] = digest
    report = tmp_path / "invalid-digest.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    result = validate_e2e_latency_evidence(report, require_target_hardware=False)
    assert result.status == "fail" and result.errors and "payload_sha256" in result.errors[0]


@pytest.mark.parametrize("text", ["[]", "{", '{"e2e_us":'])
def test_actual_bad_json_or_root_raises(tmp_path: Path, text: str) -> None:
    """Native JSON/object-root errors retain the reader's exception contract."""
    report = tmp_path / "invalid.json"
    report.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        validate_e2e_latency_evidence(report)


def test_builder_defaults_and_shared_nested_metadata(measured_report: Path) -> None:
    """Public builder defaults local fields while preserving nested identity and explicit flags."""
    payload = json.loads(measured_report.read_text(encoding="utf-8"))
    payload.pop("evidence_class")
    payload.pop("production_claim_allowed")
    built = build_e2e_latency_evidence_payload(payload)
    assert "evidence_class" not in payload and "production_claim_allowed" not in payload
    assert built["evidence_class"] == "local_regression" and built["production_claim_allowed"] is False
    assert built is not payload and built["context"] is payload["context"]
    payload["production_claim_allowed"] = True
    assert build_e2e_latency_evidence_payload(payload)["production_claim_allowed"] is True


def test_declared_target_labels_remain_metadata_not_hardware_approval(measured_report: Path, tmp_path: Path) -> None:
    """Nonplaceholder labels pass presence checks while production permission remains false."""
    payload = json.loads(measured_report.read_text(encoding="utf-8"))
    payload["target_hardware"] = {
        "id": " reader-only-label ",
        "class": " reader-only-class ",
        "rt_kernel": " reader-only-scheduler ",
    }
    payload["payload_sha256"] = build_e2e_latency_evidence_payload(payload)["payload_sha256"].upper()
    path = tmp_path / "authored-labels.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = validate_e2e_latency_evidence(path)
    assert result.status == "pass" and result.target_hardware_id == "reader-only-label"
    assert result.target_hardware_class == "reader-only-class" and result.rt_kernel == "reader-only-scheduler"
    assert payload["production_claim_allowed"] is False

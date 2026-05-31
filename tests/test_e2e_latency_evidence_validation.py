# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — E2E Latency Evidence Validation Tests
"""Behavioral tests for E2E latency evidence admission."""

from __future__ import annotations

import json

import pytest

from validation.validate_e2e_latency_evidence import (
    E2E_LATENCY_SCHEMA_VERSION,
    build_e2e_latency_evidence_payload,
    validate_e2e_latency_evidence,
)


def _latency_payload() -> dict:
    return {
        "iterations": 1000,
        "warmup": 50,
        "target_hardware": {
            "id": "jetson-orin-nx-lab-unit-03",
            "class": "jetson",
            "machine": "aarch64",
            "processor": "arm",
            "platform": "Linux",
            "python": "3.12.0",
            "numpy": "2.0.0",
            "rt_kernel": "PREEMPT_RT-6.8-lab",
        },
        "kernel_only_us": {"p50": 45.0, "p95": 60.0, "p99": 80.0},
        "e2e_us": {"p50": 450.0, "p95": 700.0, "p99": 850.0},
        "e2e_overhead_factor": 10.0,
    }


def test_e2e_latency_evidence_accepts_qualified_hardware_report(tmp_path):
    report = tmp_path / "e2e_latency.json"
    report.write_text(json.dumps(build_e2e_latency_evidence_payload(_latency_payload())), encoding="utf-8")

    result = validate_e2e_latency_evidence(report, max_e2e_p95_us=1000.0)

    assert result.status == "pass"
    assert result.errors == ()
    assert result.p95_us == 700.0
    assert result.target_hardware_class == "jetson"
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["schema_version"] == E2E_LATENCY_SCHEMA_VERSION
    assert len(payload["payload_sha256"]) == 64


def test_e2e_latency_evidence_rejects_unqualified_local_report(tmp_path):
    payload = _latency_payload()
    payload["target_hardware"]["id"] = "local-host-unqualified"
    payload["target_hardware"]["class"] = "unspecified-local"
    payload["target_hardware"]["rt_kernel"] = "unknown"
    report = tmp_path / "e2e_latency.json"
    report.write_text(json.dumps(build_e2e_latency_evidence_payload(payload)), encoding="utf-8")

    result = validate_e2e_latency_evidence(report)

    assert result.status == "fail"
    assert "target_hardware.id must identify the measured hardware" in result.errors
    assert "target_hardware.class must identify the hardware class" in result.errors
    assert "target_hardware.rt_kernel must identify scheduler or RT-kernel evidence" in result.errors


def test_e2e_latency_evidence_rejects_threshold_regression(tmp_path):
    payload = _latency_payload()
    payload["e2e_us"]["p95"] = 1500.0
    report = tmp_path / "e2e_latency.json"
    report.write_text(json.dumps(build_e2e_latency_evidence_payload(payload)), encoding="utf-8")

    result = validate_e2e_latency_evidence(report, max_e2e_p95_us=1000.0)

    assert result.status == "fail"
    assert "e2e_us.p95 exceeds admission threshold 1000.0" in result.errors


def test_e2e_latency_evidence_can_allow_unqualified_local_development_report(tmp_path):
    payload = _latency_payload()
    payload["target_hardware"]["id"] = "local-host-unqualified"
    payload["target_hardware"]["class"] = "unspecified-local"
    payload["target_hardware"]["rt_kernel"] = "unknown"
    report = tmp_path / "e2e_latency.json"
    report.write_text(json.dumps(build_e2e_latency_evidence_payload(payload)), encoding="utf-8")

    result = validate_e2e_latency_evidence(report, require_target_hardware=False)

    assert result.status == "pass"


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda payload: payload.__setitem__("iterations", 0), "iterations must be a positive integer"),
        (
            lambda payload: payload["e2e_us"].__setitem__("p50", payload["e2e_us"]["p99"] + 1.0),
            "e2e_us percentiles must satisfy p50 <= p95 <= p99",
        ),
        (
            lambda payload: payload.__setitem__("e2e_overhead_factor", 99.0),
            "e2e_overhead_factor must match p50 e2e/kernel ratio",
        ),
    ],
)
def test_e2e_latency_evidence_rejects_runtime_metadata_regressions(tmp_path, mutator, message):
    payload = build_e2e_latency_evidence_payload(_latency_payload())
    mutator(payload)
    payload = build_e2e_latency_evidence_payload(payload)
    report = tmp_path / "e2e_latency.json"
    report.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_e2e_latency_evidence(report)

    assert result.status == "fail"
    assert message in result.errors


def test_e2e_latency_evidence_rejects_tampered_payload_digest(tmp_path):
    payload = build_e2e_latency_evidence_payload(_latency_payload())
    payload["e2e_us"]["p95"] = 999.0
    report = tmp_path / "e2e_latency.json"
    report.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_e2e_latency_evidence(report, max_e2e_p95_us=1000.0)

    assert result.status == "fail"
    assert "payload_sha256 does not match latency evidence payload" in result.errors

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable Transport Latency Validation Tests

"""Behavioral tests for differentiable transport gradient-latency evidence."""

from __future__ import annotations

import json
from pathlib import Path

from validation.validate_differentiable_transport_latency import validate_differentiable_transport_latency


def _one_step_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "backend": "jax",
        "dtype": "float64",
        "n_rho": 21,
        "channel_count": 4,
        "warmup_runs": 1,
        "timed_runs": 5,
        "p50_ms": 10.0,
        "p95_ms": 12.0,
        "max_ms": 13.0,
        "claim_status": "local audited gradient-admission latency only; not a real-time control-loop guarantee",
        "audit": {
            "loss": 0.1,
            "epsilon": 5.0e-5,
            "tolerance": 2.0e-3,
            "checked_indices": [[0, 5], [1, 10], [2, 7], [3, 12]],
            "chi_max_abs_error": 1.0e-6,
            "source_max_abs_error": 1.0e-6,
            "passed": True,
        },
    }


def _rollout_report() -> dict[str, object]:
    payload = _one_step_report()
    payload["claim_status"] = (
        "local audited rollout source-gradient latency only; not a real-time control-loop guarantee"
    )
    payload["n_steps"] = 4
    audit = payload["audit"]
    assert isinstance(audit, dict)
    audit.pop("chi_max_abs_error")
    audit["checked_indices"] = [[0, 0, 5], [1, 1, 8], [2, 2, 10], [3, 3, 12]]
    return payload


def _readiness_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "backend": "jax",
        "campaign_sha256": "a" * 64,
        "gradient_latency_report_sha256": "b" * 64,
        "gradient_audit_sha256": "c" * 64,
        "rollout_latency_report_sha256": "d" * 64,
        "rollout_audit_sha256": "e" * 64,
        "external_reference_artifact_sha256": None,
        "external_reference_admitted": False,
        "controller_formal_artifact_sha256": None,
        "n_rho": 21,
        "rollout_steps": 4,
        "channel_order": ["electron_temperature", "ion_temperature", "electron_density", "impurity_density"],
        "equilibrium_coupled": True,
        "full_fidelity_claim_admissible": False,
        "blocked_reasons": ["controller_formal_artifact_sha256", "external_reference_artifact_sha256"],
        "claim_status": "bounded differentiable transport readiness only; full-fidelity claim remains blocked",
    }


def test_differentiable_transport_latency_admits_complete_reports(tmp_path: Path) -> None:
    one_step = tmp_path / "one_step.json"
    rollout = tmp_path / "rollout.json"
    one_step.write_text(json.dumps(_one_step_report()), encoding="utf-8")
    rollout.write_text(json.dumps(_rollout_report()), encoding="utf-8")

    report = validate_differentiable_transport_latency(one_step, rollout, require_admitted=True)

    assert report["status"] == "pass"
    assert report["admitted_reports"] == 2
    assert report["blocked_reports"] == 0
    assert report["errors"] == []


def test_differentiable_transport_latency_validates_full_fidelity_readiness(tmp_path: Path) -> None:
    one_step = tmp_path / "one_step.json"
    rollout = tmp_path / "rollout.json"
    readiness = tmp_path / "readiness.json"
    one_step.write_text(json.dumps(_one_step_report()), encoding="utf-8")
    rollout.write_text(json.dumps(_rollout_report()), encoding="utf-8")
    readiness.write_text(json.dumps(_readiness_report()), encoding="utf-8")

    report = validate_differentiable_transport_latency(
        one_step,
        rollout,
        readiness_report=readiness,
        require_admitted=True,
    )

    assert report["status"] == "pass"
    assert report["full_fidelity_ready"] is False
    assert report["readiness_entry"]["status"] == "blocked"
    assert report["readiness_entry"]["blocked_reasons"] == [
        "controller_formal_artifact_sha256",
        "external_reference_artifact_sha256",
    ]


def test_differentiable_transport_latency_rejects_inconsistent_readiness(tmp_path: Path) -> None:
    one_step = tmp_path / "one_step.json"
    rollout = tmp_path / "rollout.json"
    readiness = tmp_path / "readiness.json"
    readiness_payload = _readiness_report()
    readiness_payload["full_fidelity_claim_admissible"] = True
    one_step.write_text(json.dumps(_one_step_report()), encoding="utf-8")
    rollout.write_text(json.dumps(_rollout_report()), encoding="utf-8")
    readiness.write_text(json.dumps(readiness_payload), encoding="utf-8")

    report = validate_differentiable_transport_latency(one_step, rollout, readiness_report=readiness)

    assert report["status"] == "fail"
    assert any(error["field"] == "readiness.blocked_reasons" for error in report["errors"])


def test_differentiable_transport_latency_rejects_unordered_latency(tmp_path: Path) -> None:
    one_step_payload = _one_step_report()
    one_step_payload["p95_ms"] = 9.0
    one_step = tmp_path / "one_step.json"
    rollout = tmp_path / "rollout.json"
    one_step.write_text(json.dumps(one_step_payload), encoding="utf-8")
    rollout.write_text(json.dumps(_rollout_report()), encoding="utf-8")

    report = validate_differentiable_transport_latency(one_step, rollout)

    assert report["status"] == "fail"
    assert any(error["field"] == "latency" for error in report["errors"])


def test_differentiable_transport_latency_rejects_failed_audit(tmp_path: Path) -> None:
    one_step_payload = _one_step_report()
    audit = one_step_payload["audit"]
    assert isinstance(audit, dict)
    audit["passed"] = False
    one_step = tmp_path / "one_step.json"
    rollout = tmp_path / "rollout.json"
    one_step.write_text(json.dumps(one_step_payload), encoding="utf-8")
    rollout.write_text(json.dumps(_rollout_report()), encoding="utf-8")

    report = validate_differentiable_transport_latency(one_step, rollout)

    assert report["status"] == "fail"
    assert any(error["field"] == "audit.passed" for error in report["errors"])


def test_differentiable_transport_latency_rejects_duplicate_keys(tmp_path: Path) -> None:
    one_step = tmp_path / "one_step.json"
    rollout = tmp_path / "rollout.json"
    one_step.write_text('{"schema_version": 1, "schema_version": 2}', encoding="utf-8")
    rollout.write_text(json.dumps(_rollout_report()), encoding="utf-8")

    report = validate_differentiable_transport_latency(one_step, rollout)

    assert report["status"] == "fail"
    assert any(error["error"] == "duplicate JSON key: schema_version" for error in report["errors"])


def test_repository_differentiable_transport_latency_evidence_is_admitted() -> None:
    root = Path(__file__).resolve().parents[1]
    report = validate_differentiable_transport_latency(
        root / "validation" / "reports" / "differentiable_transport_latency.json",
        root / "validation" / "reports" / "differentiable_transport_rollout_latency.json",
        readiness_report=root / "validation" / "reports" / "differentiable_transport_full_fidelity_readiness.json",
        require_admitted=True,
    )

    assert report["status"] == "pass"
    assert report["admitted_reports"] == 2
    assert report["full_fidelity_ready"] is False

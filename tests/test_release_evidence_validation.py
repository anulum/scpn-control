# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Release Evidence Validation Tests

"""Behavioral tests for release evidence report admission."""

from __future__ import annotations

import json

from validation.validate_release_evidence import validate_release_evidence


def _valid_report() -> dict[str, object]:
    cases = ("cyclone_base_case", "tem_kinetic_electron", "stable_mode")
    backends = ("cpu", "gpu")
    return {
        "transport_solver_available": True,
        "import_clean": True,
        "status": "pass",
        "data_manifests": {
            "status": "pass",
            "total": 5,
            "real": 4,
            "synthetic": 1,
            "artifact_coverage": {"expected": 21, "covered": 21, "missing": []},
        },
        "jax_gk_parity": {
            "status": "pass",
            "parity_artifacts": 6,
            "required_cases": list(cases),
            "required_backends": list(backends),
            "entries": [{"case": case, "backend": backend} for case in cases for backend in backends],
        },
        "physics_traceability": {
            "status": "pass",
            "total": 54,
            "open_fidelity_gaps": 53,
            "public_claim_blocked": 53,
        },
        "native_formal_certificate": {
            "status": "pass",
            "admitted_cases": ["std:spin:aot_certificate:stride_1"],
            "certificate_assumption_sha256": "a" * 64,
            "errors": [],
            "report_sha256": "b" * 64,
        },
    }


def test_release_evidence_admits_complete_passing_report(tmp_path):
    """A complete passing top-level validation report is admitted."""
    path = tmp_path / "release_evidence_report.json"
    path.write_text(json.dumps(_valid_report()), encoding="utf-8")

    result = validate_release_evidence(path)

    assert result.status == "pass"
    assert result.errors == ()
    assert result.report_sha256 is not None
    assert result.admitted_gates == (
        "data_manifests",
        "jax_gk_parity",
        "physics_traceability",
        "native_formal_certificate",
    )


def test_release_evidence_rejects_skipped_required_gate(tmp_path):
    """Release evidence cannot skip mandatory provenance, parity, traceability, or formal gates."""
    report = _valid_report()
    report["jax_gk_parity"] = {"status": "skipped"}
    path = tmp_path / "release_evidence_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    result = validate_release_evidence(path)

    assert result.status == "fail"
    assert "jax_gk_parity.status must be 'pass', got 'skipped'" in result.errors


def test_release_evidence_rejects_invalid_native_certificate_digest(tmp_path):
    """Native formal certificate evidence must bind a SHA-256 assumption digest."""
    report = _valid_report()
    native_formal = report["native_formal_certificate"]
    assert isinstance(native_formal, dict)
    native_formal["certificate_assumption_sha256"] = "not-a-digest"
    path = tmp_path / "release_evidence_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    result = validate_release_evidence(path)

    assert result.status == "fail"
    assert (
        "native_formal_certificate.certificate_assumption_sha256 must be a SHA-256 hex digest"
        in result.errors
    )


def test_release_evidence_rejects_incomplete_jax_case_backend_pairs(tmp_path):
    """Every required JAX GK case/backend pair must be present in the report."""
    report = _valid_report()
    parity = report["jax_gk_parity"]
    assert isinstance(parity, dict)
    entries = parity["entries"]
    assert isinstance(entries, list)
    parity["entries"] = entries[:-1]
    path = tmp_path / "release_evidence_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    result = validate_release_evidence(path)

    assert result.status == "fail"
    assert "jax_gk_parity.entries must include every required case/backend pair" in result.errors


def test_release_evidence_rejects_duplicate_json_keys(tmp_path):
    """Duplicate JSON keys are rejected so an attacker cannot shadow status fields."""
    path = tmp_path / "release_evidence_report.json"
    path.write_text('{"status": "pass", "status": "fail"}', encoding="utf-8")

    result = validate_release_evidence(path)

    assert result.status == "fail"
    assert result.errors == ("duplicate JSON key: status",)

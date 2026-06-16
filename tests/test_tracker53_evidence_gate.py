# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tracker #53 evidence-gate tests
"""Tests for the tracker #53 hardware/runtime evidence manifest gate."""

from __future__ import annotations

import json
from pathlib import Path

from validation.validate_tracker53_evidence import (
    TRACKER53_SCHEMA_VERSION,
    build_tracker53_manifest,
    validate_tracker53_evidence,
)


def test_repository_tracker53_manifest_is_bounded_and_blocked() -> None:
    result = validate_tracker53_evidence()

    assert result.status == "pass"
    assert result.production_claim_allowed is False
    assert result.errors == ()
    assert result.tracker_issue == 53
    assert {entry["module_path"] for entry in result.entries} == {
        "src/scpn_control/core/checkpoint.py",
        "src/scpn_control/phase/kuramoto.py",
        "src/scpn_control/scpn/formal_verification.py",
        "src/scpn_control/scpn/fpga_export.py",
        "src/scpn_control",
        "src/scpn_control/core/runtime_admission.py",
    }
    assert result.evidence_classes["src/scpn_control/core/runtime_admission.py"] == "runtime_local_regression"
    assert result.evidence_classes["src/scpn_control/scpn/fpga_export.py"] == "generated_hdl"


def test_tracker53_production_claim_requirement_fails_closed() -> None:
    result = validate_tracker53_evidence(require_production_claim=True)

    assert result.status == "fail"
    assert result.production_claim_allowed is False
    assert any("production tracker #53 claim requires qualified hardware evidence" in error for error in result.errors)


def test_tracker53_manifest_is_digest_bound() -> None:
    result = validate_tracker53_evidence()
    manifest = build_tracker53_manifest(result)

    assert manifest["schema_version"] == TRACKER53_SCHEMA_VERSION
    assert manifest["status"] == "pass"
    assert len(manifest["manifest_sha256"]) == 64
    assert manifest["production_claim_allowed"] is False


def test_tracker53_manifest_json_output(tmp_path: Path) -> None:
    output = tmp_path / "tracker53.json"

    result = validate_tracker53_evidence(output_json=output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.status == "pass"
    assert payload["schema_version"] == TRACKER53_SCHEMA_VERSION
    assert payload["manifest_sha256"] == build_tracker53_manifest(result)["manifest_sha256"]
    assert payload["tracker_issue"] == 53


def test_tracker53_rejects_missing_runtime_report(tmp_path: Path) -> None:
    missing = tmp_path / "missing_runtime_report.json"

    result = validate_tracker53_evidence(runtime_report=missing)

    assert result.status == "fail"
    assert any("runtime_admission.report" in error for error in result.errors)


def test_tracker53_cli_reports_fail_closed_production_requirement(capsys) -> None:
    import validation.validate_tracker53_evidence as mod

    assert mod.main(["--require-production-claim", "--json-out"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fail"
    assert payload["production_claim_allowed"] is False
    assert any(
        "production tracker #53 claim requires qualified hardware evidence" in error for error in payload["errors"]
    )

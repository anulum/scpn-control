# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Project: SCPN Control
# Description: Lean formal-verification report tests.
"""Tests for Lean 4 formal-verification report admission contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from scpn_control.scpn.lean_verification import (
    LeanFormalVerificationError,
    LeanFormalVerificationReport,
    build_lean_formal_report_payload,
    load_lean_formal_report,
    validate_lean_formal_report_payload,
    write_lean_formal_report,
)


def _lean_report() -> LeanFormalVerificationReport:
    return LeanFormalVerificationReport(
        status="pass",
        solver="Lean 4.13.0",
        lean_version="4.13.0",
        checked_specs=["pid.actuator_saturation", "snn.marking_bounds"],
        artifact_sha256="a" * 64,
        proof_source_sha256="b" * 64,
        lakefile_sha256="c" * 64,
        theorem_names=[
            "ScpnControl.PID.actuatorSaturationPreserved",
            "ScpnControl.SNN.markingBoundsPreserved",
        ],
        theorem_modules=["ScpnControl.PID", "ScpnControl.SNN"],
        proved_contracts=["pid.actuator_saturation", "snn.marking_bounds"],
        module_paths=[
            "src/scpn_control/control/pid_controller.py",
            "src/scpn_control/scpn/controller.py",
        ],
        safety_case_ids=["SC-PID-ACTUATOR-SATURATION", "SC-SNN-MARKING-BOUNDS"],
        claim_boundary="bounded Lean proof over exported controller envelope",
    )


def test_lean_formal_report_payload_carries_canonical_digest() -> None:
    payload = build_lean_formal_report_payload(_lean_report())

    assert payload["schema_version"] == "scpn-control.lean4-formal-report.v1"
    assert payload["backend"] == "lean4"
    assert payload["payload_sha256"]

    validate_lean_formal_report_payload(payload)


def test_lean_formal_report_rejects_payload_digest_mismatch() -> None:
    payload = build_lean_formal_report_payload(_lean_report())
    payload["theorem_names"] = [
        "ScpnControl.PID.changedAfterDigest",
        "ScpnControl.SNN.markingBoundsPreserved",
    ]

    with pytest.raises(LeanFormalVerificationError, match="payload_sha256"):
        validate_lean_formal_report_payload(payload)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("claim_boundary", "unbounded theorem over all plants", "bounded proof boundary"),
        ("module_paths", ["../src/scpn_control/scpn/controller.py"], "safe relative paths"),
        ("proved_contracts", ["pid.actuator_saturation"], "missing required contracts"),
        ("checked_specs", ["pid.actuator_saturation"], "checked_specs missing proved contracts"),
        ("theorem_names", ["bad theorem"], "invalid identifier"),
        ("safety_case_ids", ["bad id"], "invalid identifier"),
    ],
)
def test_lean_formal_report_rejects_malformed_contract_payload(field: str, value: object, match: str) -> None:
    payload = build_lean_formal_report_payload(_lean_report())
    payload.pop("payload_sha256")
    payload[field] = value

    with pytest.raises(LeanFormalVerificationError, match=match):
        validate_lean_formal_report_payload(payload)


def test_write_and_load_lean_formal_report_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "lean-report.json"

    written = write_lean_formal_report(_lean_report(), path)
    loaded = load_lean_formal_report(path)

    assert loaded == written
    assert loaded["theorem_modules"] == ["ScpnControl.PID", "ScpnControl.SNN"]

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Lean formal-verification report tests.
"""Tests for Lean 4 formal-verification report admission contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from scpn_control.scpn.lean_verification import (
    LeanFormalVerificationError,
    LeanFormalVerificationReport,
    build_lean_formal_report_payload,
    compute_assumption_sha256,
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
        proof_assumptions=[
            "bounded actuator command interval from exported artifact readout limits",
            "bounded SNN marking interval [0, 1] from compiled artifact topology",
        ],
    )


def test_lean_formal_report_payload_carries_canonical_digest() -> None:
    payload = build_lean_formal_report_payload(_lean_report())

    assert payload["schema_version"] == "scpn-control.lean4-formal-report.v1"
    assert payload["backend"] == "lean4"
    assert payload["payload_sha256"]

    validate_lean_formal_report_payload(payload)


def test_lean_formal_report_rejects_missing_payload_digest() -> None:
    payload = build_lean_formal_report_payload(_lean_report())
    payload.pop("payload_sha256")

    with pytest.raises(LeanFormalVerificationError, match="payload_sha256 is required"):
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
        ("proof_assumptions", ["plant is linear"], "bounded assumptions"),
        ("assumption_sha256", "0" * 64, "assumption_sha256"),
        ("solver", "Coq 8.19.0", "solver must identify Lean"),
        ("solver", "Lean 4.12.0", "solver must include lean_version"),
        (
            "theorem_names",
            ["ScpnControl.Transport.unrelated", "ScpnControl.SNN.markingBoundsPreserved"],
            "pid.actuator_saturation requires theorem_names",
        ),
        (
            "theorem_modules",
            ["ScpnControl.Transport", "ScpnControl.SNN"],
            "pid.actuator_saturation requires theorem_modules",
        ),
        (
            "theorem_names",
            [
                "ScpnControl.PID.actuatorSaturationPreserved",
                "ScpnControl.SNN.markingBoundsPreserved",
                "ScpnControl.Transport.unrelatedProof",
            ],
            "theorem_names contains unsupported namespaces",
        ),
        (
            "theorem_modules",
            ["ScpnControl.PID", "ScpnControl.SNN", "ScpnControl.Transport"],
            "theorem_modules cannot exceed theorem_names",
        ),
        (
            "module_paths",
            [
                "src/scpn_control/control/pid_controller.py",
                "src/scpn_control/scpn/geometry_neutral_replay.py",
            ],
            "module_paths missing required paths",
        ),
        (
            "safety_case_ids",
            ["SC-PID-ACTUATOR-SATURATION", "SC-UNRELATED-FORMAL-EVIDENCE"],
            "safety_case_ids missing required IDs",
        ),
    ],
)
def test_lean_formal_report_rejects_malformed_contract_payload(field: str, value: object, match: str) -> None:
    payload = build_lean_formal_report_payload(_lean_report())
    payload.pop("payload_sha256")
    payload[field] = value
    if field == "proof_assumptions":
        payload["assumption_sha256"] = compute_assumption_sha256(value)

    with pytest.raises(LeanFormalVerificationError, match=match):
        validate_lean_formal_report_payload(payload)


def test_lean_formal_report_rejects_unsupported_proved_contract() -> None:
    payload = build_lean_formal_report_payload(_lean_report())
    payload.pop("payload_sha256")
    payload["proved_contracts"] = [
        "pid.actuator_saturation",
        "snn.marking_bounds",
        "facility.full_certification",
    ]
    payload["checked_specs"] = list(payload["proved_contracts"])

    with pytest.raises(LeanFormalVerificationError, match="unsupported contracts"):
        validate_lean_formal_report_payload(payload)


def test_lean_formal_report_rejects_unsupported_theorem_module_padding() -> None:
    payload = build_lean_formal_report_payload(_lean_report())
    payload.pop("payload_sha256")
    payload["theorem_names"] = [
        "ScpnControl.PID.actuatorSaturationPreserved",
        "ScpnControl.SNN.markingBoundsPreserved",
        "ScpnControl.PID.helperInvariant",
    ]
    payload["theorem_modules"] = ["ScpnControl.PID", "ScpnControl.SNN", "ScpnControl.Transport"]

    with pytest.raises(LeanFormalVerificationError, match="theorem_modules contains unsupported namespaces"):
        validate_lean_formal_report_payload(payload)


def test_lean_formal_report_rejects_unknown_fields() -> None:
    payload = build_lean_formal_report_payload(_lean_report())
    payload.pop("payload_sha256")
    payload["external_certification_status"] = "certified"

    with pytest.raises(LeanFormalVerificationError, match="unsupported fields"):
        validate_lean_formal_report_payload(payload)


def test_write_and_load_lean_formal_report_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "lean-report.json"

    written = write_lean_formal_report(_lean_report(), path)
    loaded = load_lean_formal_report(path)

    assert loaded == written
    assert loaded["theorem_modules"] == ["ScpnControl.PID", "ScpnControl.SNN"]


def test_load_lean_formal_report_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-report.json"
    path.write_text('{"status": "pass", "status": "fail"}', encoding="utf-8")

    with pytest.raises(LeanFormalVerificationError, match="duplicate JSON key: status"):
        load_lean_formal_report(path)

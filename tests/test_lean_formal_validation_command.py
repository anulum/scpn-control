# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Formal validator declaration and public command evidence

"""Exercise the actual Lean reader/CLI with authored declaration-only report bytes."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from formal_validator_declaration_fixtures import declared_lean_case

from validation.validate_scpn_lean_formal import validate_lean_formal_evidence

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entry", ["api", "script"])
@pytest.mark.parametrize("matching", [True, False])
def test_actual_named_report_and_artifact_must_agree(tmp_path: Path, entry: str, matching: bool) -> None:
    """Two valid distinct report declarations cannot stand in for each other."""
    named, other, artifact = declared_lean_case(tmp_path)
    path = named if matching else other
    if entry == "api":
        result = validate_lean_formal_evidence(path, artifact_path=artifact, formal_report_root=tmp_path)
        status, admitted, errors = result.status, result.artifact_admitted, result.errors
        assert result.report_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        result_cli = subprocess.run(
            [
                sys.executable,
                str(ROOT / "validation/validate_scpn_lean_formal.py"),
                str(path),
                "--artifact",
                str(artifact),
                "--formal-report-root",
                str(tmp_path),
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        payload = json.loads(result_cli.stdout)
        status, admitted, errors = payload["status"], payload["artifact_admitted"], payload["errors"]
        assert result_cli.returncode == (0 if matching else 1)
        assert payload["report_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert status == ("pass" if matching else "fail")
    assert admitted is matching
    if matching:
        assert not errors
    else:
        assert any("supplied Lean report" in error for error in errors)


@pytest.mark.parametrize("entry", ["api", "script"])
@pytest.mark.parametrize("case", ["missing", "duplicate"])
def test_actual_invalid_or_missing_report_refuses(tmp_path: Path, entry: str, case: str) -> None:
    """The real reader reports absent/duplicate reports without proof execution."""
    path = tmp_path / "invalid.json"
    if case == "duplicate":
        path.write_text('{"status":"pass","status":"fail"}', encoding="utf-8")
    if entry == "api":
        result = validate_lean_formal_evidence(path)
        assert result.status == "fail" and result.errors and result.artifact_admitted is False
        assert (result.report_sha256 is None) is (case == "missing")
    else:
        process = subprocess.run(
            [sys.executable, str(ROOT / "validation/validate_scpn_lean_formal.py"), str(path)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert process.returncode == 1
        payload = json.loads(process.stdout)
        assert payload["status"] == "fail" and payload["errors"] and payload["artifact_admitted"] is False
        assert (payload["report_sha256"] is None) is (case == "missing")


def test_actual_report_without_artifact_is_only_report_validation(tmp_path: Path) -> None:
    """An authored report can pass schema/digest checks without artifact admission."""
    named, _other, _artifact = declared_lean_case(tmp_path)
    result = validate_lean_formal_evidence(named)
    assert result.status == "pass" and result.errors == () and result.backend == "lean4"
    assert result.artifact_admitted is False


@pytest.mark.parametrize("status", ["fail", "blocked"])
def test_actual_nonpassing_named_report_cannot_admit_metadata_only_artifact(tmp_path: Path, status: str) -> None:
    """A matched declared digest alone cannot admit a nonpassing named report."""
    named, _other, artifact = declared_lean_case(tmp_path)
    payload = json.loads(named.read_text())
    payload["status"] = status
    from scpn_control.scpn.lean_verification import validate_lean_formal_report_payload

    # Rebuild the canonical self-digest after this explicit authored declaration change.
    canonical = dict(payload)
    canonical.pop("payload_sha256")
    payload["payload_sha256"] = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    validate_lean_formal_report_payload(payload)
    named.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    declared_artifact = json.loads(artifact.read_text())
    declared_artifact["formal_verification"]["report_sha256"] = hashlib.sha256(named.read_bytes()).hexdigest()
    artifact.write_text(json.dumps(declared_artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = validate_lean_formal_evidence(named, artifact_path=artifact)
    assert result.status == "fail" and result.artifact_admitted is False
    assert any("supplied Lean report" in error for error in result.errors)


def test_actual_non_lean_manifest_cannot_reuse_named_lean_digest(tmp_path: Path) -> None:
    """An authored non-Lean manifest cannot admit an otherwise valid Lean declaration."""
    named, _other, artifact = declared_lean_case(tmp_path)
    declared_artifact = json.loads(artifact.read_text())
    previous = declared_artifact["formal_verification"]
    declared_artifact["formal_verification"] = {
        "required": True,
        "status": "pass",
        "backend": "z3",
        "solver": "z3-solver metadata-declaration-only",
        "max_depth": 2,
        "checked_specs": ["marking_bounds"],
        "artifact_sha256": previous["artifact_sha256"],
        "report_sha256": previous["report_sha256"],
        "claim_boundary": "bounded metadata declaration only",
        "report_uri": previous["report_uri"],
    }
    artifact.write_text(json.dumps(declared_artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = validate_lean_formal_evidence(named, artifact_path=artifact)
    assert result.status == "fail" and result.artifact_admitted is False
    assert any("supplied Lean report" in error for error in result.errors)

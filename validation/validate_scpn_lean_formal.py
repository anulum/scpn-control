#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Lean formal-verification evidence validator.
"""Validate Lean 4 formal-verification evidence for SCPN controller artefacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scpn_control.scpn.artifact import ArtifactValidationError, load_artifact
from scpn_control.scpn.lean_verification import LeanFormalVerificationError, validate_lean_formal_report_payload


@dataclass(frozen=True)
class LeanFormalValidationResult:
    """Result of a Lean formal-verification evidence validation pass."""

    status: str
    report_sha256: str | None
    errors: tuple[str, ...]
    backend: str | None = None
    lean_version: str | None = None
    theorem_names: tuple[str, ...] = ()
    proved_contracts: tuple[str, ...] = ()
    artifact_admitted: bool = False


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_no_duplicates(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise LeanFormalVerificationError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise LeanFormalVerificationError("Lean 4 report must be a JSON object")
    return payload


def validate_lean_formal_evidence(
    report_path: str | Path,
    *,
    artifact_path: str | Path | None = None,
    formal_report_root: str | Path | None = None,
) -> LeanFormalValidationResult:
    """Validate a Lean report and optionally its safety-critical artifact admission."""
    path = Path(report_path)
    try:
        report_bytes = path.read_bytes()
    except OSError as exc:
        return LeanFormalValidationResult(status="fail", report_sha256=None, errors=(str(exc),))
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    errors: list[str] = []
    payload: dict[str, Any] | None = None
    try:
        payload = _load_json_no_duplicates(path)
        validate_lean_formal_report_payload(payload)
    except LeanFormalVerificationError as exc:
        errors.append(str(exc))
    artifact_admitted = False
    if artifact_path is not None and not errors:
        try:
            load_artifact(
                artifact_path,
                require_formal_verification=True,
                formal_report_root=formal_report_root,
            )
            artifact_admitted = True
        except ArtifactValidationError as exc:
            errors.append(str(exc))
    if errors:
        return LeanFormalValidationResult(
            status="fail",
            report_sha256=report_sha256,
            errors=tuple(errors),
            backend=str(payload.get("backend")) if payload is not None and "backend" in payload else None,
            lean_version=str(payload.get("lean_version"))
            if payload is not None and "lean_version" in payload
            else None,
        )
    assert payload is not None
    report_status = str(payload["status"])
    return LeanFormalValidationResult(
        status=report_status,
        report_sha256=report_sha256,
        errors=(),
        backend="lean4",
        lean_version=str(payload["lean_version"]),
        theorem_names=tuple(str(item) for item in payload["theorem_names"]),
        proved_contracts=tuple(str(item) for item in payload["proved_contracts"]),
        artifact_admitted=artifact_admitted,
    )


def _result_payload(result: LeanFormalValidationResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "backend": result.backend,
        "lean_version": result.lean_version,
        "report_sha256": result.report_sha256,
        "errors": list(result.errors),
        "theorem_names": list(result.theorem_names),
        "proved_contracts": list(result.proved_contracts),
        "artifact_admitted": result.artifact_admitted,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Lean formal-verification report JSON")
    parser.add_argument("--artifact", type=Path, help="Optional .scpnctl.json artifact to admit against the report")
    parser.add_argument(
        "--formal-report-root",
        type=Path,
        help="Root used to resolve artifact formal_verification.report_uri",
    )
    args = parser.parse_args(argv)
    result = validate_lean_formal_evidence(
        args.report,
        artifact_path=args.artifact,
        formal_report_root=args.formal_report_root,
    )
    print(json.dumps(_result_payload(result), sort_keys=True))
    return 0 if result.status == "pass" and not result.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

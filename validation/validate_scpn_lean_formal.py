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
from scpn_control.scpn.lean_verification import LeanFormalVerificationError, load_lean_formal_report


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


def validate_lean_formal_evidence(
    report_path: str | Path,
    *,
    artifact_path: str | Path | None = None,
    formal_report_root: str | Path | None = None,
) -> LeanFormalValidationResult:
    """Validate a named Lean declaration and optionally its matching artifact.

    Parameters
    ----------
    report_path : str or pathlib.Path
        Report path resolved from the caller's working directory.
    artifact_path : str or pathlib.Path or None
        Optional artifact requiring passing formal evidence. Its backend must
        be lean4 and its declared report digest must match the named report.
    formal_report_root : str or pathlib.Path or None
        Forwarded to the artifact loader for resolving its safe relative
        report_uri. None performs manifest validation without opening that URI.

    Returns
    -------
    LeanFormalValidationResult
        Report status pass/fail/blocked, first-observed raw-byte SHA-256,
        errors and validated declared theorem/contract names. artifact_admitted
        is True only after the artifact loader and named-report checks pass.

    Raises
    ------
    OSError
        Report reread or artifact I/O fails after the initial read. Initial
        report read errors become a fail result with no digest.
    ValueError, KeyError, TypeError
        Native artifact parsing failures outside ArtifactValidationError
        propagate. LeanFormalVerificationError and ArtifactValidationError
        become fail results.

    Notes
    -----
    No Lean/Lake process, proof-source read or physical controller is executed.
    Passing proves this reader's declaration/schema/digest checks, not theorem
    compilation or source authenticity. With artifact_path, named status must
    be pass and its raw hash must match the artifact's formal report digest.
    The initial hash read, defining report-loader reread and optional artifact/
    reference reads are sequential observations without locks or snapshots.
    No output file is written and calls do not share mutable validator state.
    """
    path = Path(report_path)
    try:
        report_bytes = path.read_bytes()
    except OSError as exc:
        return LeanFormalValidationResult(status="fail", report_sha256=None, errors=(str(exc),))
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    errors: list[str] = []
    payload: dict[str, Any] | None = None
    try:
        payload = load_lean_formal_report(path)
    except LeanFormalVerificationError as exc:
        errors.append(str(exc))
    artifact_admitted = False
    if artifact_path is not None and not errors:
        assert payload is not None
        try:
            artifact = load_artifact(
                artifact_path,
                require_formal_verification=True,
                formal_report_root=formal_report_root,
            )
            evidence = artifact.formal_verification
            assert evidence is not None
            if (
                payload["status"] != "pass"
                or evidence.backend != "lean4"
                or evidence.report_sha256.lower() != report_sha256
            ):
                raise ArtifactValidationError("artifact evidence does not reference the supplied Lean report")
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
    """Return a fresh JSON-ready mapping of the observed validation result.

    Parameters
    ----------
    result : LeanFormalValidationResult
        Reader result with tuple-valued errors, names and contract declarations.

    Returns
    -------
    dict[str, Any]
        New mapping with fresh list containers; scalars/digests are unchanged.
        This converts representation without rerunning validation.
    """
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
    """Validate explicit CLI paths and print one sorted JSON result to stdout.

    Parameters
    ----------
    argv : list[str] or None
        Parser arguments or process sys.argv when None. report is required;
        --artifact and --formal-report-root are optional caller-relative paths.

    Returns
    -------
    int
        Zero only for pass with no errors; one for fail or blocked. Direct
        callers receive the code, while standalone execution exits with it.

    Raises
    ------
    SystemExit
        Argument parsing prints help/code zero or refuses syntax/code two.
    OSError, ValueError, KeyError, TypeError
        Unconverted defining-reader/parser failures propagate.

    Notes
    -----
    No report/artifact output is created. Schema-valid authored report
    declarations are not independently generated Lean proof witnesses.
    """
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

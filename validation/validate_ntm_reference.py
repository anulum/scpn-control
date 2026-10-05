#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — NTM reference artifact validator

"""Validate persisted NTM island-dynamics reference artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.ntm_reference_contracts import (
    _validate_artifact,
)
from validation.ntm_reference_contracts import (
    canonical_artifact_sha256 as canonical_artifact_sha256,
)


def validate_ntm_reference(
    artifact_root: str | Path,
    *,
    require_reference_artifacts: bool = False,
) -> dict[str, Any]:
    """Read selected JSON declarations and report schema, source, unit and declared-tolerance findings.

    Directories select immediate sorted *.json files; a regular file selects
    itself regardless of suffix. Missing/nonfile roots select nothing: optional
    mode passes with zero entries, required mode fails. Each file is read once.
    Unsupported JSON/UTF8, duplicate keys and IO failures become authored json
    findings; valid declarations return only identity/count metadata. Reads are
    sequential observations, not a coherent directory snapshot.

    Referenced SHA256 is format-only; payload SHA256 checks the canonical body.
    Referenced bytes are not hashed, downloaded,
    resolved or executed. Model/version/dataset/time/DOI and declared metric
    computations are not authenticated. A passing report establishes declaration
    consistency and does not qualify island-dynamics physics or facility evidence.
    No island-dynamics solve, integration, training or file mutation occurs during inspection.

    Examples
    --------
    >>> from tempfile import TemporaryDirectory
    >>> with TemporaryDirectory() as directory:
    ...     report = validate_ntm_reference(directory, require_reference_artifacts=True)
    >>> report["status"], report["reference_artifacts"], report["errors"][0]["field"]
    ('fail', 0, 'artifact_root')
    """
    root = Path(artifact_root)
    paths = sorted(root.glob("*.json")) if root.is_dir() else ([root] if root.is_file() else [])
    report: dict[str, Any] = {
        "status": "pass",
        "root": str(root),
        "reference_artifacts": 0,
        "require_reference_artifacts": bool(require_reference_artifacts),
        "entries": [],
        "errors": [],
    }
    entries: list[dict[str, object]] = report["entries"]
    errors: list[dict[str, object]] = report["errors"]

    if require_reference_artifacts and not paths:
        errors.append({"path": str(root), "field": "artifact_root", "error": "no NTM reference artifacts found"})

    for path in paths:
        try:
            payload = json.loads(
                path.read_bytes().decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_float=_decode_json_float,
            )
            entry = _validate_artifact(path, payload, errors)
        except (OSError, ValueError):
            errors.append(
                {"path": str(path), "field": "json", "error": "artifact must be readable UTF-8 JSON with unique keys"}
            )
            continue
        if entry is not None:
            entries.append(entry)
            report["reference_artifacts"] += 1

    if errors:
        report["status"] = "fail"
    return report


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Decode JSON objects while refusing duplicate keys, including in nested objects."""
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("artifact JSON contains duplicate keys")
        out[key] = value
    return out


def _decode_json_float(token: str) -> float:
    """Decode a JSON float while refusing nonzero decimal tokens that collapse to binary64 zero."""
    number = float(token)
    if number == 0 and any(char in "123456789" for char in token.lower().partition("e")[0]):
        raise ValueError("artifact JSON number is not representable")
    return number


def write_ntm_reference_report(report: dict[str, Any], output_path: str | Path, *, artifact_root: str | Path) -> None:
    """Write sorted UTF8 JSON, creating parents while protecting selected input aliases.

    Direct, resolved, symbolic and existing hard-link aliases of the supplied
    root or its immediate JSON inputs raise ValueError before any write. Other
    existing output is replaced. IO/path-resolution/encoding/serialization errors propagate.
    Discovery is a fresh sequential observation, without locking or transactional
    coupling to prior validation. Concurrent pathname changes remain outside it.
    """
    output = Path(output_path)
    root = Path(artifact_root)
    inputs = [root, *(sorted(root.glob("*.json")) if root.is_dir() else [])]
    for source in inputs:
        if output.resolve() == source.resolve() or (output.exists() and source.exists() and output.samefile(source)):
            raise ValueError("NTM reference report output must not overwrite selected input")
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Return zero for declaration pass, one for findings or supported inspection/write failure.

    Paths are caller-relative; absent root uses the canonical reference folder.
    JSON mode prints the report, text mode prints summary/stdout and findings/
    stderr. Operational refusal uses fixed authored stderr with no exception
    details. Parser help/usage retain exits0/2; this CLI authenticates no NTM equilibrium.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        default=str(ROOT / "validation" / "reports" / "ntm_reference"),
        help="Directory or JSON artifact containing persisted NTM island-dynamics reference evidence",
    )
    parser.add_argument(
        "--require-reference-artifacts", action="store_true", help="Fail if no NTM reference artifacts are present"
    )
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)

    try:
        report = validate_ntm_reference(
            args.artifact_root, require_reference_artifacts=args.require_reference_artifacts
        )
        if args.output_json:
            write_ntm_reference_report(report, args.output_json, artifact_root=args.artifact_root)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        print("NTM reference FAILED: could not inspect artifacts or write report", file=sys.stderr)
        return 1
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"NTM reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            print(f"ERROR {error['path']}: {error['error']}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

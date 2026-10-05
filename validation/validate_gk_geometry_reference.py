#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK geometry reference validator


"""Validate stored local Miller cases with exact input-byte custody and protected report persistence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from validation.gk_geometry_reference_cases import _REQUIRED_CASES, _validate_case
from validation.gk_geometry_reference_contracts import _finalise_report, _new_report, _validate_header


def validate_gk_geometry_reference(reference_path: str | Path) -> dict[str, Any]:
    """Read one reference file once, hash exact bytes and compare original local Miller cases.

    Invalid UTF8/JSON/duplicate keys, unreadable files and nonzero decimal tokens
    collapsed to binary64zero yield authored findings. Header schema1.0, seven
    identities, three required case names, eight required numeric parameters and
    float-coercible optional fields remain. Four-angle n_theta4/n_period1 Miller
    evaluation, nearest-grid sample selection and nine field comparisons retain
    atol1e-11/rtol1e-10. Output v2 uses dimensionally correct metric units and canonical
    report/case digests. Passing local cases keep full-equilibrium admission false;
    no independent reference is generated, input changed or tolerance relaxed.
    Observations are sequential without snapshot or producer authentication.

    Examples
    --------
    >>> from tempfile import TemporaryDirectory
    >>> with TemporaryDirectory() as directory:
    ...     report = validate_gk_geometry_reference(Path(directory) / "absent.json")
    >>> report["status"], report["cases"], report["public_claims"]["full_equilibrium_reconstruction"]
    ('fail', 0, False)
    """
    path = Path(reference_path)
    report = _new_report(path)
    entries: list[dict[str, object]] = report["entries"]
    errors: list[dict[str, object]] = report["errors"]
    try:
        raw_payload = path.read_bytes()
        report["reference_file_sha256"] = hashlib.sha256(raw_payload).hexdigest()
        payload = json.loads(
            raw_payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys, parse_float=_decode_json_float
        )
    except (OSError, ValueError):
        report["status"] = "fail"
        errors.append(
            {"path": str(path), "field": "json", "error": "reference must be readable UTF-8 JSON with unique keys"}
        )
        return _finalise_report(report)

    if not isinstance(payload, dict):
        errors.append({"path": str(path), "field": "root", "error": "reference root must be an object"})
        report["status"] = "fail"
        return _finalise_report(report)
    _validate_header(path, payload, errors)
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append({"path": str(path), "field": "cases", "error": "cases must be a non-empty array"})
        report["status"] = "fail"
        return _finalise_report(report)

    seen_cases: set[str] = set()
    for index, case_payload in enumerate(cases):
        entry = _validate_case(path, index, case_payload, errors)
        if entry is not None:
            case_name = str(entry["case"])
            if case_name in seen_cases:
                errors.append(
                    {"path": str(path), "index": index, "field": "case", "error": f"duplicate case: {case_name}"}
                )
                continue
            seen_cases.add(case_name)
            entries.append(entry)
    missing_cases = sorted(_REQUIRED_CASES - seen_cases)
    for case_name in missing_cases:
        errors.append({"path": str(path), "field": "case", "error": f"missing required reference case: {case_name}"})
    report["cases"] = len(entries)
    if errors:
        report["status"] = "fail"
    return _finalise_report(report)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Refuse duplicate object keys at every nesting level without echoing untrusted names."""
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("reference JSON contains duplicate keys")
        out[key] = value
    return out


def _decode_json_float(token: str) -> float:
    """Decode a JSON float while refusing nonzero decimal tokens that collapse to binary64 zero."""
    number = float(token)
    if number == 0 and any(char in "123456789" for char in token.lower().partition("e")[0]):
        raise ValueError("artifact JSON number is not representable")
    return number


def write_gk_geometry_reference_report(
    report: dict[str, Any], output_path: str | Path, *, reference_path: str | Path
) -> None:
    """Write sorted UTF8 JSON after refusing direct/resolved/symlink/existing-hardlink aliases of the reference.

    Output parents are created and unrelated output replaced. API path/IO/
    serialization failures propagate; no locks or coherent pathname snapshot
    are promised. The reference is a single file, not a scanned input directory.
    """
    output = Path(output_path)
    root = Path(reference_path)
    inputs = [root]
    for source in inputs:
        if output.resolve() == source.resolve() or (output.exists() and source.exists() and output.samefile(source)):
            raise ValueError("GK geometry reference report output must not overwrite selected input")
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Return zero for bounded comparison pass, one for findings or authored operational refusal; parser help/usage retain exits0/2."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-path",
        default=str(ROOT / "validation" / "reference_data" / "gk_geometry" / "miller_reference_cases.json"),
        help="Immutable Miller geometry reference case JSON",
    )
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)

    try:
        report = validate_gk_geometry_reference(args.reference_path)
        if args.output_json:
            write_gk_geometry_reference_report(report, args.output_json, reference_path=args.reference_path)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        print("GK geometry reference FAILED: could not inspect reference or write report", file=sys.stderr)
        return 1
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"GK geometry reference: {report['status']} cases={report['cases']}")
        for error in report["errors"]:
            print(f"ERROR {error['path']}: {error['error']}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

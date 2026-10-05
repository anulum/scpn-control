# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK species exact-byte bounded reference reader and protected persistence

"""GK species exact-byte bounded reference reader and protected persistence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def ensure_repo_src_on_path() -> None:
    """Allow direct script execution from a source checkout without installation."""
    repo_src = str(Path(__file__).resolve().parents[1] / "src")
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)


ensure_repo_src_on_path()
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.gk_species_reference_cases import _validate_case
from validation.gk_species_reference_contracts import (
    _REQUIRED_CASES,
    _finalise_report,
    _portable_reference_path,
    _validate_header,
)
from validation.gk_species_reference_contracts import EXPECTED_UNITS as EXPECTED_UNITS
from validation.gk_species_reference_contracts import FULL_FIDELITY_BLOCKERS as FULL_FIDELITY_BLOCKERS
from validation.gk_species_reference_contracts import REPORT_SCHEMA_VERSION as REPORT_SCHEMA_VERSION
from validation.gk_species_reference_contracts import verify_payload_digest as verify_payload_digest
from validation.gk_species_reference_operators import _validate_operator_checks


def validate_gk_species_reference(reference_path: str | Path) -> dict[str, Any]:
    """Validate repository species and collision outputs against reference cases."""
    path = Path(reference_path)
    report: dict[str, Any] = {
        "status": "pass",
        "reference_path": _portable_reference_path(path),
        "cases": 0,
        "entries": [],
        "operator_checks": {},
        "errors": [],
    }
    entries: list[dict[str, object]] = report["entries"]
    errors: list[dict[str, object]] = report["errors"]
    reference_sha256: str | None = None
    try:
        raw = path.read_bytes()
        reference_sha256 = hashlib.sha256(raw).hexdigest()
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys, parse_float=_decode_json_float
        )
    except (OSError, ValueError):
        report.update(
            {
                "status": "fail",
                "errors": [
                    {
                        "path": str(path),
                        "field": "json",
                        "error": "reference must be readable UTF-8 JSON with unique keys",
                    }
                ],
            }
        )
        return _finalise_report(report, reference_sha256=reference_sha256)

    if not isinstance(payload, dict):
        errors.append(
            {
                "path": str(path),
                "field": "root",
                "error": "reference root must be an object",
            }
        )
        report["status"] = "fail"
        return _finalise_report(report, reference_sha256=reference_sha256)
    _validate_header(path, payload, errors)
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append(
            {
                "path": str(path),
                "field": "cases",
                "error": "cases must be a non-empty array",
            }
        )
        report["status"] = "fail"
        return _finalise_report(report, reference_sha256=reference_sha256)

    seen_cases: set[str] = set()
    for index, case_payload in enumerate(cases):
        entry = _validate_case(path, index, case_payload, errors)
        if entry is not None:
            case_name = str(entry["case"])
            if case_name in seen_cases:
                errors.append(
                    {
                        "path": str(path),
                        "index": index,
                        "field": "case",
                        "error": f"duplicate case: {case_name}",
                    }
                )
            else:
                seen_cases.add(case_name)
                entries.append(entry)
    for case_name in sorted(_REQUIRED_CASES - seen_cases):
        errors.append(
            {
                "path": str(path),
                "field": "case",
                "error": f"missing required reference case: {case_name}",
            }
        )
    operator_checks = _validate_operator_checks(path, payload.get("operator_checks"), errors)
    if operator_checks is not None:
        report["operator_checks"] = operator_checks
    report["cases"] = len(entries)
    if errors:
        report["status"] = "fail"
    return _finalise_report(report, reference_sha256=reference_sha256)


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


def write_gk_species_reference_report(
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
            raise ValueError("GK species reference report output must not overwrite selected input")
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Return0 for bounded pass,1 for findings/authored operational refusal; parser retains help/usage exits0/2."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-path",
        default=str(ROOT / "validation" / "reference_data" / "gk_species" / "species_collision_reference_cases.json"),
        help="Immutable GK species and collision reference case JSON",
    )
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)

    try:
        report = validate_gk_species_reference(args.reference_path)
        if args.output_json:
            write_gk_species_reference_report(report, args.output_json, reference_path=args.reference_path)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        print("GK species reference FAILED: could not inspect reference or write report", file=sys.stderr)
        return 1
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"GK species reference: {report['status']} cases={report['cases']}")
        for error in report["errors"]:
            print(f"ERROR {error['path']}: {error['error']}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

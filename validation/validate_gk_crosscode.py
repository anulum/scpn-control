#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Linear GK cross-code evidence validator

"""Inspect local self-declared GK comparison JSON; launch no binary or external scientific computation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.gk_crosscode_reference_contracts import _validate_evidence_payload


class _GKDeclarationRefusal(ValueError):
    """Carry only an authored duplicate/nonfinite/nonzero-underflow decoder finding."""


def validate_gk_crosscode_evidence(evidence_root: str | Path, *, require_external_runs: bool = False) -> dict[str, Any]:
    """Read selected declarations and compare their own external/native scalar claims.

    A directory selects immediate sorted JSON entries; a regular file selects
    itself regardless of suffix. Optional missing/nonfile roots pass with zero
    declarations; required mode refuses. Each entry reads one captured byte
    sequence. Fixed IO/UTF8/JSON/duplicate/nonfinite/underflow findings omit raw
    exceptions and input keys. Reads are sequential, not a concurrent snapshot.

    Preserve schema/source/identity/hash/unit and original declared-error bounds.
    external_runs counts accepted declarations, including repeated IDs. No binary,
    input deck, external output, source bytes, code version, timestamp or actual
    run is authenticated. Body SHA verifies author consistency only. A pass is
    not independent cross-code, measured, facility or control/action admission.

    Examples
    --------
    >>> from tempfile import TemporaryDirectory
    >>> with TemporaryDirectory() as directory:
    ...     optional = validate_gk_crosscode_evidence(directory)
    ...     required = validate_gk_crosscode_evidence(directory, require_external_runs=True)
    >>> optional["status"], optional["external_runs"], required["status"]
    ('pass', 0, 'fail')
    """
    root = Path(evidence_root)
    paths = sorted(root.glob("*.json")) if root.is_dir() else ([root] if root.is_file() else [])
    report: dict[str, Any] = {
        "status": "pass",
        "root": str(root),
        "external_runs": 0,
        "require_external_runs": bool(require_external_runs),
        "entries": [],
        "errors": [],
    }
    entries: list[dict[str, object]] = report["entries"]
    errors: list[dict[str, object]] = report["errors"]
    if require_external_runs and not paths:
        errors.append(
            {"path": str(root), "field": "evidence_root", "error": "no real external GK evidence reports found"}
        )
    for path in paths:
        try:
            raw = path.read_bytes()
            payload = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json_constant,
                parse_float=_parse_finite_json_float,
            )
            entry = _validate_evidence_payload(path, payload, errors)
        except _GKDeclarationRefusal as exc:
            errors.append({"path": str(path), "field": "json", "error": str(exc)})
            continue
        except OSError:
            errors.append({"path": str(path), "field": "json", "error": "could not read GK evidence declaration"})
            continue
        except UnicodeError:
            errors.append({"path": str(path), "field": "json", "error": "GK evidence declaration is not UTF-8"})
            continue
        except (ValueError, RecursionError):
            errors.append({"path": str(path), "field": "json", "error": "GK evidence declaration is not valid JSON"})
            continue
        if entry is not None:
            entries.append(entry)
            report["external_runs"] += 1
    if errors:
        report["status"] = "fail"
    return report


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Preserve key order and refuse duplicate keys without disclosing private member names."""
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise _GKDeclarationRefusal("GK evidence declaration contains duplicate JSON keys")
        out[key] = value
    return out


def _reject_nonfinite_json_constant(token: str) -> None:
    """Refuse nonstandard decoder NaN and signed infinity extensions at every depth."""
    raise _GKDeclarationRefusal("GK evidence declaration contains non-finite JSON numbers")


def _parse_finite_json_float(token: str) -> float:
    """Refuse decimal overflow and nonzero underflow before silently altering declared values."""
    value = float(token)
    if not math.isfinite(value):
        raise _GKDeclarationRefusal("GK evidence declaration contains non-finite JSON numbers")
    if value == 0.0 and any(char in "123456789" for char in token.lower().split("e", 1)[0]):
        raise _GKDeclarationRefusal("GK evidence declaration contains underflowed JSON numbers")
    return value


def write_gk_crosscode_report(report: dict[str, Any], output_path: str | Path, *, evidence_root: str | Path) -> None:
    """Persist sorted UTF8 JSON while protecting root/immediate selected input aliases.

    Resolved/symlink/existing hardlink aliases raise ValueError before writing.
    Unrelated destinations may replace; parents are created. IO/path/encoding/
    serialization failures propagate. Sequential alias checks provide no lock
    against concurrent pathname replacement. No scientific source is resealed.
    """
    output = Path(output_path)
    root = Path(evidence_root)
    inputs = [root, *(sorted(root.glob("*.json")) if root.is_dir() else [])]
    for source in inputs:
        if output.resolve() == source.resolve() or (output.exists() and source.exists() and output.samefile(source)):
            raise ValueError("GK cross-code report output must not overwrite selected input")
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Inspect declarations with caller-relative options; fixed operational refusal/findings return one.

    Default root is canonical reports/gk_crosscode. JSON stdout or text summary/
    stderr findings retain original shape. Protected output is persisted before
    stdout. Parser help0/usage2 and optional absence0 are unchanged. No binary,
    download, training or independent scientific admission occurs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-root",
        default=str(ROOT / "validation" / "reports" / "gk_crosscode"),
        help="Directory or JSON report containing real external-code comparison evidence",
    )
    parser.add_argument(
        "--require-external-runs",
        action="store_true",
        help="Fail if no real external-code evidence reports are present",
    )
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)
    try:
        report = validate_gk_crosscode_evidence(args.evidence_root, require_external_runs=args.require_external_runs)
        if args.output_json:
            write_gk_crosscode_report(report, args.output_json, evidence_root=args.evidence_root)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        print("GK cross-code evidence FAILED: could not inspect declarations or write report", file=sys.stderr)
        return 1
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"GK cross-code evidence: {report['status']} external_runs={report['external_runs']}")
        for error in report["errors"]:
            print(f"ERROR {error['path']}: {error['error']}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

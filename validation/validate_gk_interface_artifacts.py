#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — External GK interface artifact validator

"""Inspect declared external GK parser metadata and bind bytes without external execution or scientific admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.gk_interface_reference_contracts import (
    _BLOCKED_REASON,
    _REPORT_SCHEMA,
    _portable_path,
    _validate_artifact,
)
from validation.gk_interface_reference_contracts import (
    canonical_artifact_sha256 as canonical_artifact_sha256,
)


def validate_gk_interface_artifacts(
    artifact_root: str | Path,
    *,
    require_interface_artifacts: bool = False,
) -> dict[str, Any]:
    """Inspect captured interface declarations and protect original metadata-only acceptance semantics.

    Select sorted immediate JSON files or any regular single file. Optional absent/
    nonfile roots pass zero; required absence refuses. Duplicate code/run pairs
    refuse after the first accepted entry. Reads/checks are sequential rather than
    a pathname lock or snapshot. Fixed decode/IO findings expose no raw exception
    or duplicate member text. Actual captured bytes bind artifact_file_sha256.

    Original generated_at_utc changes on every real call and contributes to the
    report hash. external_interface_artifacts_admitted means accepted author
    metadata, not executable/run/reference bytes authentication, external execution
    or physical/facility/control adoption. Full cross-code claim remains false.

    Examples
    --------
    >>> from tempfile import TemporaryDirectory
    >>> with TemporaryDirectory() as directory:
    ...     optional = validate_gk_interface_artifacts(directory)
    ...     required = validate_gk_interface_artifacts(directory, require_interface_artifacts=True)
    >>> optional["status"], optional["interface_artifacts"], required["status"]
    ('pass', 0, 'fail')
    """
    root = Path(artifact_root)
    paths = sorted(root.glob("*.json")) if root.is_dir() else ([root] if root.is_file() else [])
    report = _new_report(root, require_interface_artifacts=require_interface_artifacts)
    entries: list[dict[str, object]] = report["entries"]
    errors: list[dict[str, object]] = report["errors"]

    if require_interface_artifacts and not paths:
        errors.append(
            {
                "path": _portable_path(root),
                "field": "artifact_root",
                "error": "no external GK interface artefacts found",
            }
        )

    seen_code_runs: set[tuple[str, str]] = set()
    for path in paths:
        try:
            raw_payload = path.read_bytes()
            payload = json.loads(
                raw_payload.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json_constant,
                parse_float=_parse_finite_json_float,
            )
            entry = _validate_artifact(path, raw_payload, payload, errors)
        except _InterfaceDeclarationRefusal as exc:
            errors.append({"path": _portable_path(path), "field": "json", "error": str(exc)})
            continue
        except OSError:
            errors.append(
                {"path": _portable_path(path), "field": "json", "error": "could not read GK interface declaration"}
            )
            continue
        except UnicodeError:
            errors.append(
                {"path": _portable_path(path), "field": "json", "error": "GK interface declaration is not UTF-8"}
            )
            continue
        except (ValueError, RecursionError):
            errors.append(
                {"path": _portable_path(path), "field": "json", "error": "GK interface declaration is not valid JSON"}
            )
            continue
        if entry is not None:
            code_run = (str(entry["interface_code"]), str(entry["run_id"]))
            if code_run in seen_code_runs:
                errors.append(
                    {
                        "path": _portable_path(path),
                        "field": "run_id",
                        "error": f"duplicate interface_code/run_id: {code_run[0]} {code_run[1]}",
                    }
                )
                continue
            seen_code_runs.add(code_run)
            entries.append(entry)
            report["interface_artifacts"] += 1

    if errors:
        report["status"] = "fail"
    return _finalise_report(report)


class _InterfaceDeclarationRefusal(ValueError):
    """Carry only an authored duplicate/nonfinite/nonzero-underflow decoder finding."""


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Preserve decoded key order and refuse duplicates without exposing member names."""
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise _InterfaceDeclarationRefusal("GK interface declaration contains duplicate JSON keys")
        out[key] = value
    return out


def _reject_nonfinite_json_constant(token: str) -> None:
    """Refuse nonstandard NaN and signed infinity decoder extensions at every depth."""
    raise _InterfaceDeclarationRefusal("GK interface declaration contains non-finite JSON numbers")


def _parse_finite_json_float(token: str) -> float:
    """Refuse decimal overflow and nonzero binary64 underflow before altering declared values."""
    value = float(token)
    if not math.isfinite(value):
        raise _InterfaceDeclarationRefusal("GK interface declaration contains non-finite JSON numbers")
    if value == 0.0 and any(char in "123456789" for char in token.lower().split("e", 1)[0]):
        raise _InterfaceDeclarationRefusal("GK interface declaration contains underflowed JSON numbers")
    return value


def write_gk_interface_artifacts_report(
    report: dict[str, Any], output_path: str | Path, *, artifact_root: str | Path
) -> None:
    """Persist sorted UTF8 JSON+LF while refusing root/immediate selected direct/resolved/hardlink aliases.

    Aliases raise ValueError before writing. Other destinations may replace;
    parent creation and path/IO/encoding/serialisation failures propagate.
    Sequential checks provide no pathname lock against concurrent replacement.
    No source/parser/run evidence is resealed or independently admitted.
    """
    output = Path(output_path)
    root = Path(artifact_root)
    inputs = [root, *(sorted(root.glob("*.json")) if root.is_dir() else [])]
    for source in inputs:
        if output.resolve() == source.resolve() or (output.exists() and source.exists() and output.samefile(source)):
            raise ValueError("GK interface artifacts report output must not overwrite selected input")
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def _new_report(root: Path, *, require_interface_artifacts: bool) -> dict[str, Any]:
    """Construct the original v2 report with current real UTC timestamp and initial false claims."""
    return {
        "schema_version": _REPORT_SCHEMA,
        "status": "pass",
        "root": _portable_path(root),
        "generated_at_utc": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        "payload_sha256": None,
        "interface_artifacts": 0,
        "require_interface_artifacts": bool(require_interface_artifacts),
        "public_claims": {
            "external_interface_artifacts_admitted": False,
            "full_gk_cross_code_claim_admitted": False,
            "blocked_reason": _BLOCKED_REASON,
        },
        "entries": [],
        "errors": [],
    }


def _json_sha256(payload: object) -> str:
    """Hash original sorted compact ASCII JSON; timestamp remains part of the actual report body."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _finalise_report(report: dict[str, Any]) -> dict[str, Any]:
    """Preserve original accepted-metadata flag, full cross-code false and payload_sha256=None report hash."""
    admitted = report["status"] == "pass" and report["interface_artifacts"] > 0
    report["public_claims"]["external_interface_artifacts_admitted"] = admitted
    payload = dict(report)
    payload["payload_sha256"] = None
    report["payload_sha256"] = _json_sha256(payload)
    return report


def main(argv: list[str] | None = None) -> int:
    """Inspect declared interfaces with protected output; report failure or fixed operational refusal returns one."""
    parser = argparse.ArgumentParser(description="Validate persisted external gyrokinetic interface parser artifacts.")
    parser.add_argument(
        "--artifact-root",
        default=str(ROOT / "validation" / "reports" / "gk_interfaces"),
        help="Directory or JSON artifact containing persisted external GK interface evidence",
    )
    parser.add_argument(
        "--require-interface-artifacts", action="store_true", help="Fail if no external interface artifacts are present"
    )
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)

    try:
        report = validate_gk_interface_artifacts(
            args.artifact_root, require_interface_artifacts=args.require_interface_artifacts
        )
        if args.output_json:
            write_gk_interface_artifacts_report(report, args.output_json, artifact_root=args.artifact_root)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        print("GK interface artifacts FAILED: could not inspect declarations or write report", file=sys.stderr)
        return 1
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"GK interface artifacts: {report['status']} interface_artifacts={report['interface_artifacts']}")
        for error in report["errors"]:
            print(f"ERROR {error['path']}: {error['error']}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

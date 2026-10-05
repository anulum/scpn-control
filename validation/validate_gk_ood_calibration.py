#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK OOD calibration artifact validator

"""Inspect local declared OOD campaigns, bind bytes and protect inputs; run no fitting or external scientific operation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.gk_ood_reference_contracts import _finalise_report, _new_report, _validate_artifact
from validation.gk_ood_reference_domains import _portable_path


class _OODDeclarationRefusal(ValueError):
    """Carry only an authored duplicate/nonfinite/nonzero-underflow decoder finding."""


def validate_gk_ood_calibration(
    artifact_root: str | Path, *, require_campaign_artifacts: bool = False
) -> dict[str, Any]:
    """Inspect captured declaration bytes and original inclusive author acceptance comparisons.

    Select sorted immediate JSON files for a directory, any regular file itself,
    or zero for missing/nonfile roots. Optional absence passes; required refuses.
    Duplicate campaign IDs refuse after the first accepted declaration. Fixed
    IO/UTF8/JSON/duplicate/nonfinite/nonzero-underflow findings expose no decoder
    exception or duplicate member names. Reads are sequential, not a snapshot.

    Keep original v2 shape/hash algorithm. Accepted campaign metadata sets the
    original deployment_calibration_admitted flag; it does not install thresholds,
    authenticate source/covariance/run provenance or recompute held-out metrics.
    Full GK envelope remains false; no physical/facility/control action is admitted.

    Examples
    --------
    >>> from tempfile import TemporaryDirectory
    >>> with TemporaryDirectory() as directory:
    ...     optional = validate_gk_ood_calibration(directory)
    ...     required = validate_gk_ood_calibration(directory, require_campaign_artifacts=True)
    >>> optional["status"], optional["campaign_artifacts"], required["status"]
    ('pass', 0, 'fail')
    """
    root = Path(artifact_root)
    paths = sorted(root.glob("*.json")) if root.is_dir() else ([root] if root.is_file() else [])
    report = _new_report(root, require_campaign_artifacts=require_campaign_artifacts)
    entries: list[dict[str, object]] = report["entries"]
    errors: list[dict[str, object]] = report["errors"]
    if require_campaign_artifacts and not paths:
        errors.append(
            {"path": _portable_path(root), "field": "artifact_root", "error": "no GK OOD calibration artifacts found"}
        )
    seen_campaigns: set[str] = set()
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
        except _OODDeclarationRefusal as exc:
            errors.append({"path": _portable_path(path), "field": "json", "error": str(exc)})
            continue
        except OSError:
            errors.append(
                {"path": _portable_path(path), "field": "json", "error": "could not read OOD campaign declaration"}
            )
            continue
        except UnicodeError:
            errors.append(
                {"path": _portable_path(path), "field": "json", "error": "OOD campaign declaration is not UTF-8"}
            )
            continue
        except (ValueError, RecursionError):
            errors.append(
                {"path": _portable_path(path), "field": "json", "error": "OOD campaign declaration is not valid JSON"}
            )
            continue
        if entry is not None:
            campaign_id = str(entry["campaign_id"])
            if campaign_id in seen_campaigns:
                errors.append(
                    {
                        "path": _portable_path(path),
                        "field": "campaign_id",
                        "error": f"duplicate campaign_id: {campaign_id}",
                    }
                )
                continue
            seen_campaigns.add(campaign_id)
            entries.append(entry)
            report["campaign_artifacts"] += 1
    if errors:
        report["status"] = "fail"
    return _finalise_report(report)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Preserve decoded key order and refuse duplicates without exposing member names."""
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise _OODDeclarationRefusal("OOD campaign declaration contains duplicate JSON keys")
        out[key] = value
    return out


def _reject_nonfinite_json_constant(token: str) -> None:
    """Refuse nonstandard NaN and signed infinity decoder extensions at every depth."""
    raise _OODDeclarationRefusal("OOD campaign declaration contains non-finite JSON numbers")


def _parse_finite_json_float(token: str) -> float:
    """Refuse decimal overflow and nonzero binary64 underflow before altering declared values."""
    value = float(token)
    if not math.isfinite(value):
        raise _OODDeclarationRefusal("OOD campaign declaration contains non-finite JSON numbers")
    if value == 0.0 and any(char in "123456789" for char in token.lower().split("e", 1)[0]):
        raise _OODDeclarationRefusal("OOD campaign declaration contains underflowed JSON numbers")
    return value


def write_gk_ood_calibration_report(
    report: dict[str, Any], output_path: str | Path, *, artifact_root: str | Path
) -> None:
    """Persist sorted UTF8 JSON+LF while refusing root/immediate selected direct/resolved/hardlink aliases.

    Aliases raise ValueError before writing. Other destinations may replace;
    parent creation and path/IO/encoding/serialization failures propagate.
    Sequential checks provide no pathname lock against concurrent replacement.
    No calibration/source evidence is resealed or installed.
    """
    output = Path(output_path)
    root = Path(artifact_root)
    inputs = [root, *(sorted(root.glob("*.json")) if root.is_dir() else [])]
    for source in inputs:
        if output.resolve() == source.resolve() or (output.exists() and source.exists() and output.samefile(source)):
            raise ValueError("GK OOD calibration report output must not overwrite selected input")
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Inspect local declarations with protected output; findings/fixed operational refusal return one.

    Original caller-relative flags/default root and summary/JSON output remain.
    Parser help0/usage2 and optional absence0 persist. No fit, download, covariance
    recomputation, external run or physical/control deployment occurs.
    """
    parser = argparse.ArgumentParser(description="Validate persisted gyrokinetic OOD calibration campaign artifacts.")
    parser.add_argument(
        "--artifact-root",
        default=str(ROOT / "validation" / "reports" / "gk_ood_calibration"),
        help="Directory or JSON artifact containing persisted GK OOD calibration evidence",
    )
    parser.add_argument(
        "--require-campaign-artifacts", action="store_true", help="Fail if no calibration artifacts are present"
    )
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)
    try:
        report = validate_gk_ood_calibration(
            args.artifact_root, require_campaign_artifacts=args.require_campaign_artifacts
        )
        if args.output_json:
            write_gk_ood_calibration_report(report, args.output_json, artifact_root=args.artifact_root)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        print("GK OOD calibration FAILED: could not inspect campaigns or write report", file=sys.stderr)
        return 1
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"GK OOD calibration: {report['status']} campaign_artifacts={report['campaign_artifacts']}")
        for error in report["errors"]:
            print(f"ERROR {error['path']}: {error['error']}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

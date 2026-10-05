#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural equilibrium reference artifact validator

"""Validate persisted neural-equilibrium declarations; inspect no referenced array or P-EFIT execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.neural_equilibrium_reference_contracts import (
    _portable_path,
    _validate_artifact,
)
from validation.neural_equilibrium_reference_contracts import (
    canonical_artifact_sha256 as canonical_artifact_sha256,
)
from validation.neural_equilibrium_reference_metrics import _validate_metric_block

_REPORT_SCHEMA = "scpn-control.neural-equilibrium-reference-report.v2"
_BLOCKED_REASON = "Requires persisted real P-EFIT or documented public-reference neural equilibrium artefacts."


class NeuralReferenceReportRefusal(ValueError):
    """Refuse a report destination aliasing selected input; no system exception text is included."""


class _NeuralArtifactRefusal(ValueError):
    """Carry an authored duplicate or nonrepresentable JSON-number finding."""


def validate_neural_equilibrium_reference(
    artifact_root: str | Path,
    *,
    require_reference_artifacts: bool = False,
) -> dict[str, Any]:
    """Read selected JSON declarations, validating schema, identity, checksums and declared tolerances.

    A directory selects its immediate sorted JSON files; a regular file selects
    itself regardless of suffix. Missing/nonfile roots select nothing: optional
    mode passes with zero declarations, required mode fails. Per-file read,
    UTF-8, duplicate-key/JSON and supported value errors become fixed findings.
    Nonfinite numbers and nonzero decimal underflow are refused at every depth.
    Each JSON file is read once; its reported digest covers exact captured bytes,
    including newline spelling. Identical model/weight/dataset triples are refused.
    Results are mutable report dictionaries with ordered entries/errors and a
    canonical report checksum. Reads are sequential, not a concurrent snapshot.

    All five declared nonnegative finite errors must be within positive finite
    tolerances; units/grid/count/source are declaration contracts, not recomputed
    from arrays. Referenced files, weights and binary execution are not inspected
    or authenticated. reference_artifacts_admitted denotes accepted declarations;
    predictive_equilibrium_claim_admitted remains false. Consumers remain
    responsible for model matching, actual data/provenance and scientific review.
    No fitting, P-EFIT, download or file mutation occurs here.

    Examples
    --------
    >>> from tempfile import TemporaryDirectory
    >>> with TemporaryDirectory() as directory:
    ...     report = validate_neural_equilibrium_reference(directory, require_reference_artifacts=True)
    >>> report["status"], report["reference_artifacts"], report["public_claims"]["predictive_equilibrium_claim_admitted"]
    ('fail', 0, False)
    """
    root = Path(artifact_root)
    paths = sorted(root.glob("*.json")) if root.is_dir() else ([root] if root.is_file() else [])
    report = _new_report(root, require_reference_artifacts=require_reference_artifacts)
    entries: list[dict[str, object]] = report["entries"]
    errors: list[dict[str, object]] = report["errors"]

    seen_reference_sets: set[tuple[str, str, str]] = set()
    for path in paths:
        try:
            raw_payload = path.read_bytes()
            payload = json.loads(
                raw_payload.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_float=_parse_finite_json_float,
            )
            if _has_nonfinite_json_number(payload):
                errors.append(
                    {
                        "path": _portable_path(path),
                        "field": "json",
                        "error": "reference artifact contains non-finite JSON numbers",
                    }
                )
                if isinstance(payload, dict):
                    _validate_metric_block(
                        _portable_path(path), payload.get("metrics"), payload.get("tolerances"), errors
                    )
                continue
            entry = _validate_artifact(path, raw_payload, payload, errors)
        except _NeuralArtifactRefusal as exc:
            errors.append({"path": _portable_path(path), "field": "json", "error": str(exc)})
            continue
        except OSError:
            errors.append({"path": _portable_path(path), "field": "json", "error": "could not read reference artifact"})
            continue
        except UnicodeError:
            errors.append({"path": _portable_path(path), "field": "json", "error": "reference artifact is not UTF-8"})
            continue
        except (ValueError, RecursionError):
            errors.append(
                {"path": _portable_path(path), "field": "json", "error": "reference artifact is not valid JSON"}
            )
            continue
        if entry is not None:
            reference_set = (
                str(entry["model_id"]),
                str(entry["trained_weights_sha256"]),
                str(entry["reference_dataset_id"]),
            )
            if reference_set in seen_reference_sets:
                errors.append(
                    {
                        "path": _portable_path(path),
                        "field": "reference_dataset_id",
                        "error": (
                            "duplicate model_id/trained_weights_sha256/reference_dataset_id: "
                            f"{reference_set[0]} {reference_set[2]}"
                        ),
                    }
                )
                continue
            seen_reference_sets.add(reference_set)
            entries.append(entry)
            report["reference_artifacts"] += 1

    if require_reference_artifacts and report["reference_artifacts"] == 0 and not errors:
        errors.append(
            {
                "path": _portable_path(root),
                "field": "artifact_root",
                "error": "no neural equilibrium reference artefacts found",
            }
        )
    if errors:
        report["status"] = "fail"
    return _finalise_report(report)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Preserve JSON member order while refusing duplicate keys in any decoded object."""
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise _NeuralArtifactRefusal("reference artifact contains duplicate JSON keys")
        out[key] = value
    return out


def _has_nonfinite_json_number(payload: object) -> bool:
    """Inspect decoded values at every depth; retain original metric-field diagnostics on refusal."""
    values = [payload]
    while values:
        value = values.pop()
        if isinstance(value, float) and not math.isfinite(value):
            return True
        if isinstance(value, dict):
            values.extend(value.values())
        elif isinstance(value, list):
            values.extend(value)
    return False


def _parse_finite_json_float(token: str) -> float:
    """Refuse nonzero decimal underflow; decoded overflow retains metric diagnostics before refusal."""
    value = float(token)
    if value == 0.0 and any(char in "123456789" for char in token.lower().split("e", 1)[0]):
        raise _NeuralArtifactRefusal("reference artifact contains underflowed JSON numbers")
    return value


def _new_report(root: Path, *, require_reference_artifacts: bool) -> dict[str, Any]:
    """Create a mutable v2 report with zero declarations and both public claims initially false."""
    return {
        "schema_version": _REPORT_SCHEMA,
        "status": "pass",
        "root": _portable_path(root),
        "payload_sha256": None,
        "reference_artifacts": 0,
        "require_reference_artifacts": bool(require_reference_artifacts),
        "public_claims": {
            "reference_artifacts_admitted": False,
            "predictive_equilibrium_claim_admitted": False,
            "blocked_reason": _BLOCKED_REASON,
        },
        "entries": [],
        "errors": [],
    }


def _json_sha256(payload: object) -> str:
    """Hash sorted compact ASCII JSON report spelling; no keyed authentication is provided."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _finalise_report(report: dict[str, Any]) -> dict[str, Any]:
    """Bind report contents with its own digest nulled and admit only passing declaration metadata.

    This mutates the supplied report, retains predictive admission false and
    keeps the historical blocked-reason text even when declarations pass.
    """
    admitted = report["status"] == "pass" and report["reference_artifacts"] > 0
    report["public_claims"]["reference_artifacts_admitted"] = admitted
    payload = dict(report)
    payload["payload_sha256"] = None
    report["payload_sha256"] = _json_sha256(payload)
    return report


def write_neural_equilibrium_reference_report(
    report: dict[str, Any], output_path: str | Path, *, artifact_root: str | Path
) -> None:
    """Persist sorted indented v2 JSON without overwriting selected input file aliases.

    Caller-relative output parents are created; an ordinary non-alias output may
    be replaced. Resolved paths, symlinks and existing hard links to the root or
    its current immediate JSON inputs raise NeuralReferenceReportRefusal (a ValueError). IO/encoding/serialization
    errors propagate. Input discovery is a fresh observation, not a transaction
    with prior validation or protection against concurrent pathname replacement.
    """
    output = Path(output_path)
    root = Path(artifact_root)
    inputs = [root, *(sorted(root.glob("*.json")) if root.is_dir() else [])]
    for source in inputs:
        if output.resolve() == source.resolve() or (output.exists() and source.exists() and output.samefile(source)):
            raise NeuralReferenceReportRefusal("neural reference report output must not overwrite selected input")
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Return zero for a passing declaration report, one for findings or supported inspection/persistence errors.

    Root and output options are caller-relative; absent root uses the canonical
    reference directory. --json-out prints the report; otherwise stdout carries
    a summary and stderr findings. The shared writer protects selected input
    aliases. Expected operational errors receive authored stderr; parser help
    and usage retain exits zero/two. Cyclic report paths receive fixed operational
    refusal; public writer path errors still propagate. No array/model qualification follows a pass.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        default=str(ROOT / "validation" / "reports" / "neural_equilibrium_reference"),
        help="Directory or JSON artifact containing persisted neural equilibrium reference evidence",
    )
    parser.add_argument(
        "--require-reference-artifacts",
        action="store_true",
        help="Fail if no neural equilibrium reference artifacts are present",
    )
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)

    try:
        report = validate_neural_equilibrium_reference(
            args.artifact_root,
            require_reference_artifacts=args.require_reference_artifacts,
        )
        if args.output_json:
            write_neural_equilibrium_reference_report(report, args.output_json, artifact_root=args.artifact_root)
    except NeuralReferenceReportRefusal:
        print(
            "Neural equilibrium reference FAILED: neural reference report output must not overwrite selected input",
            file=sys.stderr,
        )
        return 1
    except (OSError, UnicodeError, ValueError, RuntimeError):
        print("Neural equilibrium reference FAILED: could not inspect artifacts or write report", file=sys.stderr)
        return 1
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Neural equilibrium reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            print(f"ERROR {error['path']}: {error['error']}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

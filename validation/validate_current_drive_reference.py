#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Current-drive reference artifact validator

"""Inspect local current-drive reference declarations without running a solver.

The direct script and registered ``scpn-control validate-current-drive-reference``
command call :func:`validate_current_drive_reference`. A directory contributes
sorted, immediate ``*.json`` paths; a single file is inspected regardless of its
suffix. Relative paths use the caller's working directory. Symlinks are followed
and no reference-root containment, freshness or snapshot guarantee is provided.

The default directory has no reference artefacts. Optional inspection therefore
passes with zero entries; ``--require-reference-artifacts`` makes absence fail.
The separate ``current_drive_claims.json`` corpus describes bounded analytic
plumbing and does not satisfy this reference schema.

UTF-8 JSON must have unique keys and finite floating-point values at every depth;
nonzero decimal tokens rounded to binary64 zero are refused.
Each artefact declares version ``1.0``, identity/provenance strings, exact unit
labels, positive source metadata and a positive integer case count. Five declared
non-negative errors must be at most their positive declared tolerances. No unit
conversion, reference fetch, digest recomputation or metric recomputation occurs.
A passing declaration is not external-code, experimental or facility admission.

The reader API only reads local files and returns fresh report containers. Concurrent
file edits can produce different observations; no shared cache, lock, FFI or
actuation is involved. Public report persistence protects selected input aliases, creates parents and
replaces unrelated output with sorted UTF8 JSON+LF; no atomic-write or durability guarantee.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from validation.current_drive_reference_contracts import _validate_artifact


class _CurrentDriveArtifactRefusal(ValueError):
    """Carry only authored refusals for ambiguous or non-finite JSON."""


def validate_current_drive_reference(
    artifact_root: str | Path,
    *,
    require_reference_artifacts: bool = False,
) -> dict[str, Any]:
    """Inspect schema and declared tolerances in local reference artefacts.

    Parameters
    ----------
    artifact_root
        File or directory to inspect, relative to the caller when not absolute.
        Directories use sorted, non-recursive ``*.json`` discovery. Missing roots
        and directories without matching files produce no candidates.
    require_reference_artifacts
        Refuse zero candidates. Otherwise an empty inspection returns ``pass``;
        malformed candidates always fail, including in optional mode.

    Returns
    -------
    dict[str, Any]
        ``status`` is ``pass`` only when ``errors`` is empty. ``root`` preserves
        the Path spelling; ``entries`` lists accepted declarations in discovery
        order and ``reference_artifacts`` counts them even if another file fails.
        Errors carry ``path``, ``field`` and an authored ``error`` string. Partial
        entries in a failed report must not be treated as overall admission.

    Notes
    -----
    The SHA-256 field is checked for 64 hex characters only. Date, URL, DOI and
    diagnostic/reference URI fields are nonblank strings, not parsed or fetched.
    Units are exact: W, A, A/m^2, 10^19 m^-3, keV, dimensionless rho, s and keV.
    Source metadata requires positive total_power_W and rho_points, with
    ``0 < rho_min < rho_max <= 1``; rho_points need not be integral. No profiles
    or actual case records are inspected. All five metric/tolerance pairs use
    inclusive ``metric <= tolerance``, without an inferred tolerance policy.
    Expected file/decoding failures become report errors. Root discovery is a
    pathlib observation and does not certify existence or readable contents.

    Examples
    --------
    Inspect the actual persisted bounded corpus, which is not reference evidence:

    >>> result = validate_current_drive_reference(ROOT / "validation/reports/current_drive_claims.json")
    >>> result["status"], result["reference_artifacts"]
    ('fail', 0)
    >>> any(error["field"] == "schema_version" for error in result["errors"])
    True
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
        errors.append(
            {"path": str(root), "field": "artifact_root", "error": "no current-drive reference artifacts found"}
        )

    for path in paths:
        try:
            payload = json.loads(
                path.read_bytes().decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_json_constant,
                parse_float=_finite_json_float,
            )
            entry = _validate_artifact(path, payload, errors)
        except _CurrentDriveArtifactRefusal as exc:
            errors.append({"path": str(path), "field": "json", "error": str(exc)})
            continue
        except OSError:
            errors.append({"path": str(path), "field": "json", "error": "could not read reference artifact"})
            continue
        except (ValueError, RecursionError):
            errors.append(
                {"path": str(path), "field": "json", "error": "reference artifact must contain valid UTF-8 JSON"}
            )
            continue
        if entry is not None:
            entries.append(entry)
            report["reference_artifacts"] += 1

    if errors:
        report["status"] = "fail"
    return report


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject repeated keys at every decoded object depth without echoing key text."""
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise _CurrentDriveArtifactRefusal("reference artifact contains duplicate JSON keys")
        out[key] = value
    return out


def _reject_json_constant(value: str) -> object:
    """Refuse JSON decoder extensions NaN and signed infinity at any depth."""
    raise _CurrentDriveArtifactRefusal("reference artifact contains non-finite JSON numbers")


def _finite_json_float(value: str) -> float:
    """Refuse nonfinite and nonzero underflowed floating-point tokens before field validation."""
    number = float(value)
    if not math.isfinite(number):
        raise _CurrentDriveArtifactRefusal("reference artifact contains non-finite JSON numbers")
    if number == 0 and any(char in "123456789" for char in value.lower().partition("e")[0]):
        raise _CurrentDriveArtifactRefusal("reference artifact contains underflowed JSON numbers")
    return number


def write_current_drive_reference_report(
    report: dict[str, Any], output_path: str | Path, *, artifact_root: str | Path
) -> None:
    """Write sorted UTF8 JSON+LF while protecting root/immediate selected input aliases.

    Direct, resolved, symlink and existing hardlink aliases raise ValueError
    before writing. Other output may replace. IO/path/encoding/serialisation
    errors propagate. Checks are sequential without locks or a concurrent snapshot.
    """
    output = Path(output_path)
    root = Path(artifact_root)
    inputs = [root, *(sorted(root.glob("*.json")) if root.is_dir() else [])]
    for source in inputs:
        if output.resolve() == source.resolve() or (output.exists() and source.exists() and output.samefile(source)):
            raise ValueError("Current-drive reference report output must not overwrite selected input")
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Inspect references through the stdlib CLI and optionally persist the report.

    Parameters
    ----------
    argv
        Options without the executable name; None reads process arguments.
        --artifact-root overrides the script-root default reference directory;
        --require-reference-artifacts refuses absence, --json-out prints the
        report, and --output-json writes the same sorted, indented JSON plus LF.

    Returns
    -------
    int
        0 for a passing inspection, 1 for a failed inspection, or 2 when a report
        destination cannot be written. Write refusal uses fixed stderr text and
        precedes stdout emission. Text mode prints field refusals to stderr.

    Raises
    ------
    SystemExit
        ArgumentParser uses 0 for help and 2 for invalid arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        default=str(ROOT / "validation" / "reports" / "current_drive_reference"),
        help="Directory or JSON artifact containing persisted current-drive reference evidence",
    )
    parser.add_argument(
        "--require-reference-artifacts",
        action="store_true",
        help="Fail if no current-drive reference artifacts are present",
    )
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)

    report = validate_current_drive_reference(
        args.artifact_root, require_reference_artifacts=args.require_reference_artifacts
    )
    if args.output_json:
        try:
            write_current_drive_reference_report(report, args.output_json, artifact_root=args.artifact_root)
        except (OSError, UnicodeError, ValueError, RuntimeError):
            print("could not write current-drive reference report", file=sys.stderr)
            return 2
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Current-drive reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            print(f"ERROR {error['path']}: {error['error']}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

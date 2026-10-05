#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Density reference artifact validator

"""Inspect local density-reference declarations and their caller-supplied errors.

The API, direct script and registered ``validate-density-reference`` command
check JSON metadata; they do not execute a density model. A directory contributes
sorted immediate ``*.json`` paths, including matching directories that produce
read errors. A single file is inspected regardless of suffix. Relative paths use
the caller's working directory; symlinks are followed without root containment.

Each accepted declaration has schema version ``1.0``, required identity strings,
a 64-character hexadecimal digest, exact unit labels, positive geometry and case
count, finite nonnegative actuator settings, and four declared errors within
positive declared tolerances. Duplicate JSON keys at any depth are refused.
Required numeric values must be representable as finite floats. Nonfinite and nonzero underflowed JSON
floating-point tokens are refused at every depth, including unused metadata;
other unused fields are not schema-validated. Geometry checks do not establish a valid radial mesh.

Source labels, reference DOI/URL, shot, URI, digest and metrics are declarations,
not independently authenticated evidence. No network fetch, reference-file hash,
model execution, metric recomputation or facility/action admission occurs. A
passing report means only that the inspected declarations meet these checks.
Empty/missing roots pass with zero entries unless references are required.

The reader only reads files and returns fresh containers. The public writer protects selected
input aliases before replacing unrelated output with sorted UTF8 JSON+LF. File reads/parsing errors
become report findings; root enumeration failures may propagate as OSError.
There is no cache, lock, concurrent snapshot or atomic CLI report-write guarantee.
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

from validation.density_reference_contracts import _validate_artifact


class _DensityDeclarationRefusal(ValueError):
    """Carry only an authored decoder diagnostic into the public declaration report."""


def validate_density_reference(
    artifact_root: str | Path,
    *,
    require_reference_artifacts: bool = False,
) -> dict[str, Any]:
    """Return a local metadata/declared-metric report without authenticating references.

    Parameters
    ----------
    artifact_root
        Directory of immediate JSON candidates or one file of any suffix.
        Relative paths use the current directory; symlink targets are inspected.
    require_reference_artifacts
        Make an empty or missing selection fail. Defaults to optional inspection.

    Returns
    -------
    dict[str, Any]
        Fresh status/root/count/policy/entries/errors containers. Status is pass
        only when there are no findings; count includes accepted declarations,
        not measured cases. Entries retain declared identity/source/case count.
        Passing syntax and declared tolerances supplies no external provenance.

    Raises
    ------
    OSError
        Root selection or enumeration fails. Candidate read/JSON/UTF-8 errors
        are instead reported per path; no passing placeholder replaces them.

    Examples
    --------
    The actual Python owner is not a JSON reference artifact. Inspect its bytes
    through the public API and retain failure without inventing reference data.

    >>> report = validate_density_reference(Path(__file__), require_reference_artifacts=True)
    >>> report["status"], report["reference_artifacts"]
    ('fail', 0)
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

    for path in paths:
        try:
            payload = json.loads(
                path.read_bytes().decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_float=_parse_json_float,
                parse_constant=_parse_json_float,
            )
            entry = _validate_artifact(path, payload, errors)
        except _DensityDeclarationRefusal as exc:
            errors.append({"path": str(path), "field": "json", "error": str(exc)})
            continue
        except UnicodeError:
            errors.append({"path": str(path), "field": "json", "error": "density reference declaration is not UTF-8"})
            continue
        except OSError:
            errors.append({"path": str(path), "field": "json", "error": "could not read density reference declaration"})
            continue
        except (ValueError, RecursionError):
            errors.append(
                {"path": str(path), "field": "json", "error": "density reference declaration is not valid JSON"}
            )
            continue
        if entry is not None:
            entries.append(entry)
            report["reference_artifacts"] += 1

    if require_reference_artifacts and report["reference_artifacts"] == 0 and not errors:
        errors.append({"path": str(root), "field": "artifact_root", "error": "no density reference artifacts found"})
    if errors:
        report["status"] = "fail"
    return report


def _parse_json_float(token: str) -> float:
    """Refuse nonfinite and nonzero underflowed floating tokens at every JSON depth, including unused metadata."""
    value = float(token)
    if not math.isfinite(value):
        raise _DensityDeclarationRefusal("density reference JSON numbers must be finite")
    if value == 0 and any(char in "123456789" for char in token.lower().partition("e")[0]):
        raise _DensityDeclarationRefusal("density reference declaration contains underflowed JSON numbers")
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate object keys at every JSON depth instead of selecting the last value."""
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise _DensityDeclarationRefusal("density reference declaration contains duplicate JSON keys")
        out[key] = value
    return out


def write_density_reference_report(
    report: dict[str, Any], output_path: str | Path, *, artifact_root: str | Path
) -> None:
    """Write sorted UTF8 JSON+LF while protecting the selected root and immediate JSON inputs.

    Direct, resolved, symlink and existing hardlink aliases raise ValueError
    before writing. Other output may replace. IO/path/encoding/serialization
    errors propagate. Checks are sequential, without locks or a concurrent snapshot.
    """
    output = Path(output_path)
    root = Path(artifact_root)
    inputs = [root, *(sorted(root.glob("*.json")) if root.is_dir() else [])]
    for source in inputs:
        if output.resolve() == source.resolve() or (output.exists() and source.exists() and output.samefile(source)):
            raise ValueError("Density reference report output must not overwrite selected input")
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Inspect declarations and protect report output; return zero for pass, one for findings or supported operational refusal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        default=str(ROOT / "validation" / "reports" / "density_reference"),
        help="Directory or JSON artifact containing persisted density reference evidence",
    )
    parser.add_argument(
        "--require-reference-artifacts", action="store_true", help="Fail if no density reference artifacts are present"
    )
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)

    try:
        report = validate_density_reference(
            args.artifact_root, require_reference_artifacts=args.require_reference_artifacts
        )
        if args.output_json:
            write_density_reference_report(report, args.output_json, artifact_root=args.artifact_root)
    except (OSError, UnicodeError, ValueError, RuntimeError):
        print("Density reference FAILED: could not inspect artifacts or write report", file=sys.stderr)
        return 1
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Density reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            print(f"ERROR {error['path']}: {error['error']}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

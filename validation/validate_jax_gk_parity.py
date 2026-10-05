#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — JAX GK parity artifact validator

"""Inspect persisted native/JAX local-dispersion parity declarations.

No JAX import, device execution or external gyrokinetic reference is required.
Directory inputs select sorted immediate ``*.json`` children; file inputs
select that file regardless of suffix. Nonexistent paths have no artifacts.
Accepted entries retain backend-parity-only and external-validation-required
boundaries; PASS grants no control authority or source authentication.

Examples
--------
An empty directory is allowed without an explicit evidence requirement:

>>> from tempfile import TemporaryDirectory
>>> with TemporaryDirectory() as directory:
...     empty = validate_jax_gk_parity(directory)
>>> empty["status"], empty["parity_artifacts"]
('pass', 0)
>>> empty["complete_required_case_backend_coverage"] is None
True
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.jax_gk_parity_contracts import _ALLOWED_BACKENDS, _ALLOWED_CASES, _validate_artifact
from validation.jax_gk_parity_domains import (
    _display_path,
    _finite_json_float,
    _ParityDeclarationRefusal,
    _reject_duplicate_json_keys,
    _reject_json_constant,
)
from validation.jax_gk_parity_summary import (
    _attach_summary_fields,
    _normalise_required_values,
    _validate_required_coverage,
)


def validate_jax_gk_parity(
    artifact_root: str | Path,
    *,
    require_parity_artifacts: bool = False,
    require_cases: tuple[str, ...] | list[str] | set[str] | None = None,
    require_backends: tuple[str, ...] | list[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Check file declarations, digests, drift, spectra and requested coverage.

    Parameters
    ----------
    artifact_root : str or Path
        Directory of immediate JSON children, a single file, or a missing path.
        Relative paths follow cwd; symlinks are followed without containment.
    require_parity_artifacts : bool, optional
        Require at least one admitted artifact. Nonboolean values add a finding.
    require_cases, require_backends : tuple, list, set of str or None, optional
        Strip names, ignore blanks and deduplicate supported cases/backends.
        Both nonempty sets require every Cartesian case/backend pair. Each set
        alone requires its named values. Unsupported names raise ValueError.

    Returns
    -------
    dict[str, Any]
        ``status`` pass/fail, root display path, admitted ``parity_artifacts``,
        normalized requirements, admitted ``entries`` and path/field/error
        findings, sorted counts/pair lists, coverage bool (None without pairs),
        maximum admitted gamma/frequency drift (None when empty), entry-digest
        multiset digest and canonical report digest. Admitted entries remain
        visible when another file or coverage requirement fails the report.
        Duplicate case/backend files count separately; coverage uses a set.

    Raises
    ------
    ValueError
        Unsupported requested case/backend name. This is configuration error,
        unlike supported artifact read/decode/domain failures returned as FAIL.

    Notes
    -----
    Finite JSON floats and unique keys are required at every depth. Numeric
    fields exclude booleans and integers overflowing float conversion. Six
    declared scalar values and case growth bounds are checked; other nested
    solver/species metadata is digest-bound but not physically validated.
    Growth drift uses ``abs(jax-native)/max(abs(native),1e-12)``; frequency
    drift is absolute. Declared tolerances must be positive. Ordered stripped
    spectra must match, dominant modes must agree and belong to both spectra,
    and all declared required modes/bounds must hold.

    Artifact and report self digests omit top-level ``payload_sha256`` and
    ``report_payload_sha256``. Nested solver/case digests include every key.
    Hex spelling is accepted in either case, but computed digest comparison
    is case-sensitive. Fixed IO/UTF8/JSON/duplicate/nonfinite/nonzero-underflow findings disclose no
    raw exception or duplicate member name. No raw-byte artifact identity,
    authenticated provenance, source
    execution, timestamp/device verification, solver replay or external-code
    validation is established. Results are ordinary mutable dictionaries.
    """
    root = Path(artifact_root)
    paths = sorted(root.glob("*.json")) if root.is_dir() else ([root] if root.is_file() else [])
    required_cases = _normalise_required_values(require_cases, _ALLOWED_CASES, "case")
    required_backends = _normalise_required_values(require_backends, _ALLOWED_BACKENDS, "backend")
    report: dict[str, Any] = {
        "status": "pass",
        "root": _display_path(root),
        "parity_artifacts": 0,
        "require_parity_artifacts": require_parity_artifacts if isinstance(require_parity_artifacts, bool) else False,
        "required_cases": sorted(required_cases),
        "required_backends": sorted(required_backends),
        "entries": [],
        "errors": [],
    }
    entries: list[dict[str, object]] = report["entries"]
    errors: list[dict[str, object]] = report["errors"]

    if not isinstance(require_parity_artifacts, bool):
        errors.append(
            {
                "path": _display_path(root),
                "field": "require_parity_artifacts",
                "error": "require_parity_artifacts must be boolean",
            }
        )
    if report["require_parity_artifacts"] and not paths:
        errors.append(
            {"path": _display_path(root), "field": "artifact_root", "error": "no JAX GK parity artifacts found"}
        )

    for path in paths:
        try:
            raw = path.read_bytes()
            payload = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_json_constant,
                parse_float=_finite_json_float,
            )
            entry = _validate_artifact(path, payload, errors)
        except _ParityDeclarationRefusal as exc:
            errors.append({"path": _display_path(path), "field": "json", "error": str(exc)})
            continue
        except OSError:
            errors.append(
                {"path": _display_path(path), "field": "json", "error": "could not read JAX GK parity declaration"}
            )
            continue
        except UnicodeError:
            errors.append(
                {"path": _display_path(path), "field": "json", "error": "JAX GK parity declaration is not UTF-8"}
            )
            continue
        except (ValueError, RecursionError):
            errors.append(
                {"path": _display_path(path), "field": "json", "error": "JAX GK parity declaration is not valid JSON"}
            )
            continue
        if entry is not None:
            entries.append(entry)
            report["parity_artifacts"] += 1

    _validate_required_coverage(
        root,
        entries,
        errors,
        required_cases=required_cases,
        required_backends=required_backends,
    )
    if errors:
        report["status"] = "fail"
    _attach_summary_fields(report)
    return report


def write_jax_gk_parity_report(report: dict[str, Any], output_path: str | Path, *, artifact_root: str | Path) -> None:
    """Persist sorted UTF8 JSON+LF while refusing root/immediate selected direct/resolved/hardlink aliases.

    Aliases raise ValueError before writing. Other destinations may replace;
    parent creation and path/IO/encoding/serialization failures propagate.
    Sequential checks provide no pathname lock against concurrent replacement.
    No source/parser/run evidence is resealed or independently admitted.
    """
    output = Path(output_path)
    root = Path(artifact_root)
    inputs = [root, *(sorted(root.glob("*.json")) if root.is_dir() else [])]
    for source in inputs:
        if output.resolve() == source.resolve() or (output.exists() and source.exists() and output.samefile(source)):
            raise ValueError("JAX GK parity report output must not overwrite selected input")
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def _split_csv(value: str | None) -> tuple[str, ...]:
    """Strip CLI CSV requirement tokens, omitting blank items.

    None returns an empty tuple; validation later checks supported names.
    """
    if value is None:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    """Run the standalone persisted-evidence CLI without JAX dependencies.

    argv follows argparse semantics, including SystemExit for parse errors.
    Defaults inspect the canonical artifact directory. JSON output is UTF-8,
    sorted and newline-terminated; supported write/path errors append a FAIL
    finding and update the report digest. --json-out prints to stdout; text
    mode prints the status/count and findings to stderr. Return0 on PASS or1
    on FAIL. Unsupported required names retain the API ValueError contract.
    Output aliases now refuse before writing; output failures still append the
    original output_json finding and rehash the retained report. Other destinations
    may replace. Output files are not scientific or control admission authority.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        default=str(ROOT / "validation" / "reports" / "jax_gk_parity"),
        help="Directory or JSON artifact containing persisted JAX/native GK parity evidence",
    )
    parser.add_argument(
        "--require-parity-artifacts", action="store_true", help="Fail if no persisted parity artifacts are present"
    )
    parser.add_argument("--require-cases", help="Comma-separated required parity cases")
    parser.add_argument("--require-backends", help="Comma-separated required JAX backends")
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)

    report = validate_jax_gk_parity(
        args.artifact_root,
        require_parity_artifacts=args.require_parity_artifacts,
        require_cases=_split_csv(args.require_cases),
        require_backends=_split_csv(args.require_backends),
    )
    if args.output_json:
        output_path = Path(args.output_json)
        try:
            write_jax_gk_parity_report(report, output_path, artifact_root=args.artifact_root)
        except (OSError, UnicodeError, ValueError, RuntimeError):
            report["errors"].append(
                {
                    "path": str(output_path),
                    "field": "output_json",
                    "error": "could not write JAX GK parity report without overwriting selected input",
                }
            )
            report["status"] = "fail"
            _attach_summary_fields(report)
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"JAX GK parity: {report['status']} parity_artifacts={report['parity_artifacts']}")
        for error in report["errors"]:
            print(f"ERROR {error['path']}: {error['error']}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

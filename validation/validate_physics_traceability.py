#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Physics Traceability Validation Runner
"""Inspect physics traceability declarations and local source/path coverage.

Metadata and path existence do not authenticate references, facility evidence
or scientific truth. Invalid JSON/domain/IO values yield a report with FAIL;
entries and counts describe inspected declarations, not admitted evidence.

Examples
--------
>>> from tempfile import TemporaryDirectory
>>> with TemporaryDirectory() as directory:
...     refused = validate_physics_traceability(Path(directory) / "missing.json")
>>> refused["status"], refused["entries"]
('fail', [])
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.report_output_paths import checked_report_destination

_ALLOWED_STATUSES = {
    "facility_validated",
    "reference_validated",
    "bounded_model",
    "validation_gap",
    "external_dependency_blocked",
}
_OPEN_GAP_STATUSES = {"bounded_model", "validation_gap", "external_dependency_blocked"}
_REQUIRED_HEADER_FIELDS = (
    "spdx_license_id",
    "commercial_license",
    "concepts_copyright",
    "code_copyright",
    "orcid",
    "contact",
    "file",
)
_REQUIRED_STR_FIELDS = ("component", "module_path", "equation_contract", "unit_contract", "validity_domain")
_REQUIRED_LIST_FIELDS = ("model_references", "validation_evidence", "claim_admission_requirements")
_SOURCE_MARKER_RE = re.compile(
    r"\b(simplification|simplified|approximation|approximate|heuristic|bounded model|reduced-order)\b",
    re.IGNORECASE,
)
_GITHUB_ISSUE_URL_RE = re.compile(r"^https://github\.com/anulum/scpn-control/issues/[1-9][0-9]*$")


def validate_physics_traceability(registry_path: str | Path) -> dict[str, Any]:
    """Inspect one UTF-8 JSON registry against its local declaration contract.

    Parameters
    ----------
    registry_path : str or Path
        File to decode. Relative input follows cwd. A resolved file directly
        under a directory named validation uses that directory's parent as
        repository root; every other location uses current cwd.

    Returns
    -------
    dict[str, Any]
        pass/fail status, registry spelling, declared entry/open-gap/literal
        false-claim counts, resolved module/evidence counts, accepted tracker
        metadata/count, dictionary-entry summaries, findings with path/field/
        error and optional index, plus total/covered/missing marker coverage.
        Dictionary entries are summarized even when invalid. Nondictionary
        entries count in total but have no summary. Counters are observations,
        not independent scientific admission; raw claim flags remain visible.

    Notes
    -----
    Require schema1.1, seven nonblank header fields, exact AGPL SPDX value and
    nonempty entry array. List/string contracts, fidelity vocabulary, literal
    claim booleans, open-gap tracker links, unique positive nonboolean tracker
    integers and matching issue URLs are checked. Optional marker enforcement
    must be boolean. Invalid trackers are refused from the returned tracker list.
    All module/evidence/covered paths must resolve inside the inferred root,
    including absolute paths and symlinks. Existence permits files/directories;
    evidence bytes, hashes, freshness and provenance are not verified. Covered
    paths must lie within resolved module scope. Approximation-marker scanning
    reads local Python source under src/scpn_control; unreadable, invalid UTF-8
    or escaping symlink targets cause findings instead of silently ignored data.

    JSON duplicate keys and nonfinite float tokens are refused at every depth.
    Supported read/decode/path failures yield FAIL, with no size/depth budget
    or coherent concurrent filesystem snapshot. Results contain mutable nested
    lists from decoded declarations. PASS does not validate equations, source
    execution, tracker state, external measurements or full-fidelity promotion.
    """
    path = Path(registry_path)
    report: dict[str, Any] = {
        "status": "pass",
        "registry": str(path),
        "total": 0,
        "open_fidelity_gaps": 0,
        "public_claim_blocked": 0,
        "resolved_module_paths": 0,
        "resolved_evidence_paths": 0,
        "external_validation_trackers": [],
        "external_validation_tracker_count": 0,
        "entries": [],
        "errors": [],
        "source_marker_coverage": {"total": 0, "covered": 0, "missing": []},
    }
    errors: list[dict[str, object]] = report["errors"]
    entries_report: list[dict[str, object]] = report["entries"]

    try:
        registry_root = _registry_root(path)
        with path.open(encoding="utf-8") as handle:
            payload = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_constant,
                parse_float=_finite_json_float,
            )
    except (OSError, ValueError, RuntimeError) as exc:
        errors.append({"path": str(path), "field": "json", "error": str(exc)})
        report["status"] = "fail"
        return report

    if not isinstance(payload, dict):
        errors.append({"path": str(path), "field": "root", "error": "registry root must be a JSON object"})
        report["status"] = "fail"
        return report
    for field in _REQUIRED_HEADER_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(
                {"path": str(path), "field": field, "error": "JSON registry requires canonical header metadata"}
            )
    if payload.get("spdx_license_id") is not None and payload["spdx_license_id"] != "AGPL-3.0-or-later":
        errors.append(
            {"path": str(path), "field": "spdx_license_id", "error": "spdx_license_id must be AGPL-3.0-or-later"}
        )
    if payload.get("schema_version") != "1.1":
        errors.append({"path": str(path), "field": "schema_version", "error": "schema_version must be '1.1'"})
    if "enforce_source_marker_coverage" in payload and not isinstance(payload["enforce_source_marker_coverage"], bool):
        errors.append(
            {
                "path": str(path),
                "field": "enforce_source_marker_coverage",
                "error": "field must be boolean when present",
            }
        )
    report["external_validation_trackers"] = _validate_external_validation_trackers(path, payload, errors)
    report["external_validation_tracker_count"] = len(report["external_validation_trackers"])
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append({"path": str(path), "field": "entries", "error": "entries must be a non-empty array"})
        report["status"] = "fail"
        return report

    source_marker_paths = _iter_source_marker_paths(registry_root, errors)
    covered_source_paths: set[str] = set()
    report["total"] = len(entries)
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append({"path": str(path), "index": index, "field": "entry", "error": "entry must be an object"})
            continue
        module_resolved, evidence_resolved, covered_paths = _validate_entry(
            registry_root,
            path,
            index,
            entry,
            errors,
            {int(tracker["issue"]) for tracker in report["external_validation_trackers"]},
        )
        covered_source_paths.update(covered_paths)
        if module_resolved:
            report["resolved_module_paths"] += 1
        report["resolved_evidence_paths"] += evidence_resolved
        status = entry.get("fidelity_status")
        if isinstance(status, str) and status in _OPEN_GAP_STATUSES:
            report["open_fidelity_gaps"] += 1
        if entry.get("public_claim_allowed") is False:
            report["public_claim_blocked"] += 1
        entries_report.append(
            {
                "component": entry.get("component", ""),
                "module_path": entry.get("module_path", ""),
                "equation_contract": entry.get("equation_contract", ""),
                "fidelity_status": status,
                "model_references": entry.get("model_references", []),
                "public_claim_allowed": entry.get("public_claim_allowed"),
                "unit_contract": entry.get("unit_contract", ""),
                "validation_evidence": entry.get("validation_evidence", []),
                "claim_admission_requirements": entry.get("claim_admission_requirements", []),
                "external_validation_tracker_issue": entry.get("external_validation_tracker_issue"),
                "covered_source_paths": sorted(covered_paths),
                "covered_source_count": len(covered_paths),
            }
        )

    marker_relative_paths = [_repo_relative(registry_root, marker_path) for marker_path in source_marker_paths]
    missing_source_paths = sorted(path for path in marker_relative_paths if path not in covered_source_paths)
    report["source_marker_coverage"] = {
        "total": len(marker_relative_paths),
        "covered": len(marker_relative_paths) - len(missing_source_paths),
        "missing": missing_source_paths,
    }
    if payload.get("enforce_source_marker_coverage") is True and missing_source_paths:
        errors.append(
            {
                "path": str(path),
                "field": "covered_source_paths",
                "error": "source files with approximation markers must be covered by traceability entries",
            }
        )
    if report["open_fidelity_gaps"] > 0 and report["external_validation_tracker_count"] == 0:
        errors.append(
            {
                "path": str(path),
                "field": "external_validation_trackers",
                "error": "registries with open fidelity gaps must link external validation collaboration trackers",
            }
        )
    if errors:
        report["status"] = "fail"
    return report


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build every decoded object while refusing repeated key spellings.

    The JSON hook applies at all depths and raises ValueError before inspection.
    """
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _reject_constant(token: str) -> None:
    """Raise ValueError for nonfinite JSON constant tokens at every depth.

    The public reader records this decoder refusal in its json finding.
    """
    raise ValueError(f"non-finite JSON constant: {token}")


def _finite_json_float(token: str) -> float:
    """Decode a floating token and refuse exponent overflow to infinity.

    Exact JSON integers are retained; metadata domains decide their validity.
    """
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number: {token}")
    return value


def _validate_external_validation_trackers(
    path: Path,
    payload: dict[str, Any],
    errors: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Append tracker findings and return only structurally valid unique trackers.

    Absent/empty lists are allowed. Positive issue integers exclude booleans;
    title/scope and exact matching repository issue URL are required. Seen
    positive issue numbers reserve uniqueness even if another field fails.
    No HTTP request, remote issue state or scientific evidence is inspected.
    """
    trackers = payload.get("external_validation_trackers", [])
    if trackers == []:
        return []
    if not isinstance(trackers, list):
        errors.append(
            {
                "path": str(path),
                "field": "external_validation_trackers",
                "error": "field must be an array when present",
            }
        )
        return []

    validated: list[dict[str, object]] = []
    seen_issues: set[int] = set()
    for index, tracker in enumerate(trackers):
        prior_errors = len(errors)
        if not isinstance(tracker, dict):
            errors.append(
                {
                    "path": str(path),
                    "index": index,
                    "field": "external_validation_trackers",
                    "error": "tracker must be an object",
                }
            )
            continue
        title = tracker.get("title")
        issue = tracker.get("issue")
        url = tracker.get("url")
        scope = tracker.get("scope")
        if not isinstance(title, str) or not title.strip():
            errors.append(
                {"path": str(path), "index": index, "field": "title", "error": "field must be a non-empty string"}
            )
        if not isinstance(scope, str) or not scope.strip():
            errors.append(
                {"path": str(path), "index": index, "field": "scope", "error": "field must be a non-empty string"}
            )
        if isinstance(issue, bool) or not isinstance(issue, int) or issue <= 0:
            errors.append(
                {"path": str(path), "index": index, "field": "issue", "error": "field must be a positive integer"}
            )
        elif issue in seen_issues:
            errors.append(
                {"path": str(path), "index": index, "field": "issue", "error": "issue numbers must be unique"}
            )
        else:
            seen_issues.add(issue)
        expected_url = f"https://github.com/anulum/scpn-control/issues/{issue}" if isinstance(issue, int) else None
        if not isinstance(url, str) or not _GITHUB_ISSUE_URL_RE.fullmatch(url):
            errors.append(
                {
                    "path": str(path),
                    "index": index,
                    "field": "url",
                    "error": "field must be an anulum/scpn-control GitHub issue URL",
                }
            )
        elif expected_url is not None and url != expected_url:
            errors.append({"path": str(path), "index": index, "field": "url", "error": "URL must match issue number"})
        if len(errors) == prior_errors:
            validated.append({"title": title, "issue": issue, "url": url, "scope": scope})
    return validated


def _validate_entry(
    registry_root: Path,
    path: Path,
    index: int,
    entry: dict[str, Any],
    errors: list[dict[str, object]],
    tracker_issues: set[int],
) -> tuple[bool, int, set[str]]:
    """Append indexed contract findings and return local path observations.

    Paths are existence/containment checks, not evidence-content validation.
    Returned covered paths are canonical relative spellings of scoped resolved
    paths. Fidelity and public flags remain declarations even on overall FAIL.
    """
    module_resolved = False
    evidence_resolved = 0
    covered_source_paths: set[str] = set()
    for field in _REQUIRED_STR_FIELDS:
        value = entry.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(
                {"path": str(path), "index": index, "field": field, "error": "field must be a non-empty string"}
            )
    for field in _REQUIRED_LIST_FIELDS:
        value = entry.get(field)
        if (
            not isinstance(value, list)
            or not value
            or not all(isinstance(item, str) and item.strip() for item in value)
        ):
            errors.append(
                {"path": str(path), "index": index, "field": field, "error": "field must be a non-empty string array"}
            )
    module_path = entry.get("module_path")
    module_scope_path: Path | None = None
    if isinstance(module_path, str) and module_path.strip():
        module_scope_path = _resolve_repo_path(registry_root, module_path)
        module_resolved = module_scope_path is not None
        if module_scope_path is None:
            errors.append(
                {
                    "path": str(path),
                    "index": index,
                    "field": "module_path",
                    "error": "module_path must resolve in repository",
                }
            )
    evidence_paths = entry.get("evidence_paths")
    if (
        not isinstance(evidence_paths, list)
        or not evidence_paths
        or not all(isinstance(item, str) and item.strip() for item in evidence_paths)
    ):
        errors.append(
            {
                "path": str(path),
                "index": index,
                "field": "evidence_paths",
                "error": "field must be a non-empty string array",
            }
        )
    else:
        for evidence_path in evidence_paths:
            if _resolve_repo_path(registry_root, evidence_path) is None:
                errors.append(
                    {
                        "path": str(path),
                        "index": index,
                        "field": "evidence_paths",
                        "error": f"evidence path does not resolve: {evidence_path}",
                    }
                )
            else:
                evidence_resolved += 1
    declared_source_paths = entry.get("covered_source_paths")
    if declared_source_paths is not None:
        if (
            not isinstance(declared_source_paths, list)
            or not declared_source_paths
            or not all(isinstance(item, str) and item.strip() for item in declared_source_paths)
        ):
            errors.append(
                {
                    "path": str(path),
                    "index": index,
                    "field": "covered_source_paths",
                    "error": "field must be a non-empty string array when present",
                }
            )
        else:
            for source_path in declared_source_paths:
                resolved_source_path = _resolve_repo_path(registry_root, source_path)
                if resolved_source_path is None:
                    errors.append(
                        {
                            "path": str(path),
                            "index": index,
                            "field": "covered_source_paths",
                            "error": f"source path does not resolve: {source_path}",
                        }
                    )
                elif module_scope_path is not None and not _path_within_scope(resolved_source_path, module_scope_path):
                    errors.append(
                        {
                            "path": str(path),
                            "index": index,
                            "field": "covered_source_paths",
                            "error": f"source path is outside module_path scope: {source_path}",
                        }
                    )
                else:
                    covered_source_paths.add(_repo_relative(registry_root, resolved_source_path))
    status = entry.get("fidelity_status")
    if not isinstance(status, str) or status not in _ALLOWED_STATUSES:
        allowed = ", ".join(sorted(_ALLOWED_STATUSES))
        errors.append(
            {"path": str(path), "index": index, "field": "fidelity_status", "error": f"must be one of: {allowed}"}
        )
    if status == "synthetic_only":
        errors.append(
            {
                "path": str(path),
                "index": index,
                "field": "fidelity_status",
                "error": "synthetic_only is forbidden; use real/reference artefacts or mark the claim as a bounded non-facility domain",
            }
        )
    public_claim_allowed = entry.get("public_claim_allowed")
    if not isinstance(public_claim_allowed, bool):
        errors.append(
            {"path": str(path), "index": index, "field": "public_claim_allowed", "error": "field must be boolean"}
        )
    if isinstance(status, str) and status in _OPEN_GAP_STATUSES and public_claim_allowed:
        errors.append(
            {
                "path": str(path),
                "index": index,
                "field": "public_claim_allowed",
                "error": "open or bounded fidelity entries cannot allow public full-fidelity claims",
            }
        )
    tracker_issue = entry.get("external_validation_tracker_issue")
    if isinstance(status, str) and status in _OPEN_GAP_STATUSES:
        if isinstance(tracker_issue, bool) or not isinstance(tracker_issue, int) or tracker_issue <= 0:
            errors.append(
                {
                    "path": str(path),
                    "index": index,
                    "field": "external_validation_tracker_issue",
                    "error": "open fidelity entries must reference an external-validation tracker issue",
                }
            )
        elif tracker_issue not in tracker_issues:
            errors.append(
                {
                    "path": str(path),
                    "index": index,
                    "field": "external_validation_tracker_issue",
                    "error": "tracker issue must exist in external_validation_trackers",
                }
            )
    elif tracker_issue is not None:
        errors.append(
            {
                "path": str(path),
                "index": index,
                "field": "external_validation_tracker_issue",
                "error": "reference-validated entries must not reference unresolved external-validation trackers",
            }
        )
    return module_resolved, evidence_resolved, covered_source_paths


def _registry_root(path: Path) -> Path:
    """Infer the root from the resolved registry parent or current cwd.

    A direct validation parent selects its parent. Other locations use cwd;
    supported resolution failures are mapped by the public reader to findings.
    """
    resolved = path.resolve()
    return resolved.parents[1] if resolved.parent.name == "validation" else Path.cwd()


def _iter_source_marker_paths(root: Path, errors: list[dict[str, object]]) -> list[Path]:
    """Scan sorted local Python source for literal approximation-marker vocabulary.

    Append a finding for read/decode/escape failures rather than ignoring bytes.
    No import, AST semantic interpretation or numerical execution is performed.
    """
    source_root = root / "src" / "scpn_control"
    if not source_root.exists():
        return []
    marker_paths: list[Path] = []
    for source_path in sorted(source_root.rglob("*.py")):
        try:
            resolved = source_path.resolve()
            resolved.relative_to(root.resolve())
            source_text = source_path.read_text(encoding="utf-8")
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append({"path": str(source_path), "field": "source_marker_coverage", "error": str(exc)})
            continue
        if _SOURCE_MARKER_RE.search(source_text):
            marker_paths.append(resolved)
    return marker_paths


def _repo_relative(root: Path, path: Path) -> str:
    """Render an already resolved/contained internal path relative to the root.

    The defining resolver and marker scanner establish containment before this
    display step. No external-path fallback grants an alternate admission.
    """
    return path.resolve().relative_to(root.resolve()).as_posix()


def _path_within_scope(path: Path, scope: Path) -> bool:
    """Check canonical file equality or directory containment for resolved paths.

    A file module scope admits only itself; a directory scope admits descendants.
    """
    resolved_path = path.resolve()
    resolved_scope = scope.resolve()
    if resolved_scope.is_file():
        return resolved_path == resolved_scope
    try:
        resolved_path.relative_to(resolved_scope)
    except ValueError:
        return False
    return True


def _resolve_repo_path(root: Path, value: str) -> Path | None:
    """Resolve an existing path inside root or return None for refusal.

    Relative paths, absolute paths and symlinks share the same canonical
    containment rule. Invalid/null/loop/missing/outside paths are not admitted.
    """
    try:
        candidate = Path(value)
        resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
        resolved.relative_to(root.resolve())
        return resolved if resolved.exists() else None
    except (OSError, ValueError, RuntimeError):
        return None


def main(argv: list[str] | None = None) -> int:
    """Run the standalone standard-library registry inspection CLI.

    Parameters
    ----------
    argv : list of str or None
        argparse tokens; None reads process arguments. The default input is
        the canonical source-tree registry, independent of the working directory.

    Returns
    -------
    int
        0 for PASS or 1 for validation and supported output failures.
        ``--json-out`` prints diagnostic JSON; text mode prints counts and
        stderr findings. Optional file output uses sorted UTF-8 JSON and a
        trailing newline, including reports that failed validation.

    Raises
    ------
    SystemExit
        argparse help or argument refusal.

    Notes
    -----
    The selected registry and its resolved or existing hard-link aliases
    cannot be output destinations. Supported output errors append an
    ``output_json`` finding and set FAIL. Other registry-evidence paths are
    outside this destination check. Discovery and writing are sequential;
    partial writes remain possible. PASS grants no scientific admission or
    authenticated custody.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default=str(ROOT / "validation" / "physics_traceability.json"),
        help="Physics traceability registry JSON path",
    )
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    parser.add_argument("--output-json", help="Write JSON report to this path")
    args = parser.parse_args(argv)

    report = validate_physics_traceability(args.registry)
    if args.output_json:
        output_path = Path(args.output_json)
        try:
            output_path = checked_report_destination(output_path, inputs=[args.registry])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except (OSError, ValueError, RuntimeError) as exc:
            report["status"] = "fail"
            report["errors"].append({"path": str(output_path), "field": "output_json", "error": str(exc)})
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "Physics traceability: "
            f"{report['status']} "
            f"total={report['total']} "
            f"open_fidelity_gaps={report['open_fidelity_gaps']} "
            f"public_claim_blocked={report['public_claim_blocked']} "
            f"external_validation_trackers={report['external_validation_tracker_count']}"
        )
        for error in report["errors"]:
            print(
                f"ERROR {error['path']}[{error.get('index', '-')}].{error['field']}: {error['error']}", file=sys.stderr
            )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Physics Traceability Validation Runner
"""Validate physics fidelity traceability records."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

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
_REQUIRED_LIST_FIELDS = ("model_references", "validation_evidence", "required_actions")
_SOURCE_MARKER_RE = re.compile(
    r"\b(simplification|simplified|approximation|approximate|heuristic|bounded model|reduced-order)\b",
    re.IGNORECASE,
)
_GITHUB_ISSUE_URL_RE = re.compile(r"^https://github\.com/anulum/scpn-control/issues/[1-9][0-9]*$")


def validate_physics_traceability(registry_path: str | Path) -> dict[str, Any]:
    """Validate a physics traceability registry and return a report."""
    path = Path(registry_path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
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
    }
    errors: list[dict[str, object]] = report["errors"]
    entries_report: list[dict[str, object]] = report["entries"]

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
    if payload.get("spdx_license_id") not in {None, "AGPL-3.0-or-later"}:
        errors.append(
            {"path": str(path), "field": "spdx_license_id", "error": "spdx_license_id must be AGPL-3.0-or-later"}
        )
    if payload.get("schema_version") != "1.0":
        errors.append({"path": str(path), "field": "schema_version", "error": "schema_version must be '1.0'"})
    report["external_validation_trackers"] = _validate_external_validation_trackers(path, payload, errors)
    report["external_validation_tracker_count"] = len(report["external_validation_trackers"])
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append({"path": str(path), "field": "entries", "error": "entries must be a non-empty array"})
        report["status"] = "fail"
        return report

    registry_root = _registry_root(path)
    source_marker_paths = _iter_source_marker_paths(registry_root)
    covered_source_paths: set[str] = set()
    report["total"] = len(entries)
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append({"path": str(path), "index": index, "field": "entry", "error": "entry must be an object"})
            continue
        module_resolved, evidence_resolved, covered_paths = _validate_entry(registry_root, path, index, entry, errors)
        covered_source_paths.update(covered_paths)
        if module_resolved:
            report["resolved_module_paths"] += 1
        report["resolved_evidence_paths"] += evidence_resolved
        status = entry.get("fidelity_status")
        if status in _OPEN_GAP_STATUSES:
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
                "required_actions": entry.get("required_actions", []),
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


def _validate_external_validation_trackers(
    path: Path,
    payload: dict[str, Any],
    errors: list[dict[str, object]],
) -> list[dict[str, object]]:
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
        if not isinstance(issue, int) or issue <= 0:
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
            errors.append(
                {"path": str(path), "index": index, "field": "url", "error": "URL must match issue number"}
            )
        if (
            isinstance(title, str)
            and title.strip()
            and isinstance(issue, int)
            and issue > 0
            and isinstance(url, str)
            and _GITHUB_ISSUE_URL_RE.fullmatch(url)
            and isinstance(scope, str)
            and scope.strip()
        ):
            validated.append({"title": title, "issue": issue, "url": url, "scope": scope})
    return validated


def _validate_entry(
    registry_root: Path,
    path: Path,
    index: int,
    entry: dict[str, Any],
    errors: list[dict[str, object]],
) -> tuple[bool, int, set[str]]:
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
    if status not in _ALLOWED_STATUSES:
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
    if status in _OPEN_GAP_STATUSES and public_claim_allowed:
        errors.append(
            {
                "path": str(path),
                "index": index,
                "field": "public_claim_allowed",
                "error": "open or bounded fidelity entries cannot allow public full-fidelity claims",
            }
        )
    return module_resolved, evidence_resolved, covered_source_paths


def _registry_root(path: Path) -> Path:
    resolved = path.resolve()
    return resolved.parents[1] if resolved.parent.name == "validation" else Path.cwd()


def _iter_source_marker_paths(root: Path) -> list[Path]:
    source_root = root / "src" / "scpn_control"
    if not source_root.exists():
        return []
    marker_paths: list[Path] = []
    for source_path in sorted(source_root.rglob("*.py")):
        try:
            source_text = source_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            source_text = source_path.read_text(encoding="utf-8", errors="ignore")
        if _SOURCE_MARKER_RE.search(source_text):
            marker_paths.append(source_path.resolve())
    return marker_paths


def _repo_relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _path_within_scope(path: Path, scope: Path) -> bool:
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
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    resolved = root / candidate
    return resolved if resolved.exists() else None


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for local and CI traceability validation."""
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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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

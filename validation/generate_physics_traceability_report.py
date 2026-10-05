#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Physics Traceability Report Generator
"""Render local physics traceability metadata with a bounded claim display.

The diagnostic API can render FAIL reports. The CLI and strict API mode refuse
invalid registries before writing. Neither a valid registry nor byte-fresh
Markdown authenticates scientific references, host/facility state or controls.

Examples
--------
>>> from tempfile import TemporaryDirectory
>>> with TemporaryDirectory() as directory:
...     diagnostic = generate_physics_traceability_markdown(Path(directory) / "missing.json")
>>> "- Status: fail" in diagnostic
True
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.report_output_paths import checked_report_destination
from validation.validate_physics_traceability import validate_physics_traceability


def generate_physics_traceability_markdown(registry_path: str | Path, *, require_valid_registry: bool = False) -> str:
    """Render one validator result as deterministic Markdown.

    Parameters
    ----------
    registry_path : str or Path
        Passed to the defining validator; its root/path rules apply unchanged.
    require_valid_registry : bool, optional
        Default false permits diagnostic rendering of FAIL. True raises
        ValueError on failed validation; nonboolean values also raise ValueError.

    Returns
    -------
    str
        Newline-terminated summary, tracker counts, metadata table, component
        actions and findings. Tables escape pipes and collapse whitespace.
        Invalid/nonlist actions have no action lines. Full-fidelity flag display
        is allowed only on PASS with a literal true declaration; all FAIL
        components display blocked, though raw report counts remain visible.

    Notes
    -----
    Diagnostic entries are observed declarations, not admitted scientific
    evidence. No source/reference execution, provenance or remote issue check.
    The canonical valid registry yields the same bytes as before this remedy.
    """
    if not isinstance(require_valid_registry, bool):
        raise ValueError("require_valid_registry must be boolean")
    report = validate_physics_traceability(registry_path)
    if require_valid_registry and report["status"] != "pass":
        raise ValueError("physics traceability registry failed validation")
    lines = [
        "# Physics Traceability and Bounded Claims",
        "",
        "This report is generated from `validation/physics_traceability.json`.",
        "It blocks full-fidelity public claims for entries whose evidence status is still open or bounded.",
        "",
        "## Summary",
        "",
        f"- Status: {report['status']}",
        f"- Registry entries: {report['total']}",
        f"- Open fidelity gaps: {report['open_fidelity_gaps']}",
        f"- Full-fidelity public claims blocked: {report['public_claim_blocked']}",
        f"- Resolved module paths: {report['resolved_module_paths']}",
        f"- Resolved evidence paths: {report['resolved_evidence_paths']}",
        f"- External validation trackers: {report['external_validation_tracker_count']}",
        f"- Source marker coverage: {_source_marker_coverage(report)}",
        "",
    ]
    trackers = _external_validation_trackers(report)
    if trackers:
        lines.extend(["## External Validation Collaboration Trackers", ""])
        ownership = _tracker_ownership_counts(report)
        status_counts = _tracker_status_counts(report)
        for tracker in trackers:
            issue = tracker["issue"]
            owned_count = ownership.get(issue, 0)
            status_summary = _format_status_counts(status_counts.get(issue, {}))
            lines.append(
                f"- {tracker['title']}: [#{issue}]({tracker['url']}) — "
                f"{owned_count} open claim(s)"
                f"{status_summary} — {tracker['scope']}"
            )
        lines.append("")
    lines.extend(
        [
            "## Module Traceability Table",
            "",
            "| Module | Equation or contract | References | Unit contract | Validation evidence | Status | Tracker |",
            "|--------|----------------------|------------|---------------|---------------------|--------|---------|",
        ]
    )
    for entry in sorted(_entries(report), key=lambda item: str(item["component"])):
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown_cell(f"`{entry['module_path']}`"),
                    _markdown_cell(str(entry.get("equation_contract", ""))),
                    _markdown_cell(_join_list(entry.get("model_references"))),
                    _markdown_cell(str(entry.get("unit_contract", ""))),
                    _markdown_cell(_join_list(entry.get("validation_evidence"))),
                    _markdown_cell(str(entry.get("fidelity_status", ""))),
                    _markdown_cell(_tracker_link(entry, report)),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Components",
            "",
        ]
    )
    for entry in sorted(_entries(report), key=lambda item: str(item["component"])):
        claim_status = "allowed" if report["status"] == "pass" and entry["public_claim_allowed"] is True else "blocked"
        lines.extend(
            [
                f"### {entry['component']}",
                "",
                f"- Fidelity status: `{entry['fidelity_status']}`",
                f"- Module path: `{entry['module_path']}`",
                f"- Full-fidelity public claim: {claim_status}",
                f"- External validation tracker: {_tracker_line(entry, report)}",
                f"- Covered source paths: {entry.get('covered_source_count', 0)}",
                "- Claim admission requirements:",
            ]
        )
        actions = entry.get("claim_admission_requirements")
        if isinstance(actions, list):
            for action in actions:
                lines.append(f"  - {action}")
        lines.append("")
    if report["errors"]:
        lines.extend(["## Validation Errors", ""])
        for error in report["errors"]:
            lines.append(f"- `{error['field']}`: {error['error']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _entries(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return dictionary-entry summaries from the validator report.

    A nonlist legacy diagnostic shape returns empty. Normal validator entries
    retain invalid field values for diagnostic display, not claim admission.
    """
    entries = report.get("entries")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _external_validation_trackers(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return validator-accepted tracker dictionaries or an empty legacy diagnostic list.

    This formatter does not independently contact or authenticate remote issues.
    """
    trackers = report.get("external_validation_trackers")
    if not isinstance(trackers, list):
        return []
    return [tracker for tracker in trackers if isinstance(tracker, dict)]


def _tracker_by_issue(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Index inspected tracker declarations by integer issue number.

    Normal validator output has unique nonboolean positive integers.
    """
    return {cast(int, tracker["issue"]): tracker for tracker in _external_validation_trackers(report)}


def _tracker_ownership_counts(report: dict[str, Any]) -> dict[int, int]:
    """Count declared integer entry links for the inspected tracker table.

    Counts include diagnostic entries; they do not establish issue ownership.
    """
    counts: dict[int, int] = {issue: 0 for issue in _tracker_by_issue(report)}
    for entry in _entries(report):
        issue = entry.get("external_validation_tracker_issue")
        if isinstance(issue, int):
            counts[issue] = counts.get(issue, 0) + 1
    return counts


def _tracker_status_counts(report: dict[str, Any]) -> dict[int, dict[str, int]]:
    """Count string fidelity declarations grouped by integer issue links.

    Nonstring diagnostic statuses are skipped; no physical classification is inferred.
    """
    counts: dict[int, dict[str, int]] = {issue: {} for issue in _tracker_by_issue(report)}
    for entry in _entries(report):
        issue = entry.get("external_validation_tracker_issue")
        status = entry.get("fidelity_status")
        if not isinstance(issue, int) or not isinstance(status, str):
            continue
        tracker_counts = counts.setdefault(issue, {})
        tracker_counts[status] = tracker_counts.get(status, 0) + 1
    return counts


def _format_status_counts(counts: dict[str, int]) -> str:
    """Render known nonzero fidelity counts in fixed severity/vocabulary order.

    Empty/unknown-only counts return empty, retaining the existing tracker layout.
    """
    ordered_statuses = (
        "external_dependency_blocked",
        "validation_gap",
        "bounded_model",
        "reference_validated",
        "facility_validated",
    )
    parts = [f"{status}={counts[status]}" for status in ordered_statuses if counts.get(status, 0) > 0]
    if not parts:
        return ""
    return f" ({', '.join(parts)})"


def _tracker_link(entry: dict[str, Any], report: dict[str, Any]) -> str:
    """Render an inspected issue link or an empty table cell when no tracker matches.

    Existence in this local metadata list is not a remote issue-state check.
    """
    issue = entry.get("external_validation_tracker_issue")
    tracker = _tracker_by_issue(report).get(issue) if isinstance(issue, int) else None
    if tracker is None:
        return ""
    return f"[#{issue}]({tracker['url']})"


def _tracker_line(entry: dict[str, Any], report: dict[str, Any]) -> str:
    """Render local tracker title/link or the literal none for unmatched metadata.

    No live ownership, completion or authority is inferred.
    """
    issue = entry.get("external_validation_tracker_issue")
    tracker = _tracker_by_issue(report).get(issue) if isinstance(issue, int) else None
    if tracker is None:
        return "none"
    return f"[#{issue}]({tracker['url']}) — {tracker['title']}"


def _source_marker_coverage(report: dict[str, Any]) -> str:
    """Render integer marker counters as covered/total or zero/zero for legacy malformed shapes.

    The defining validator supplies actual scan findings separately.
    """
    coverage = report.get("source_marker_coverage")
    if not isinstance(coverage, dict):
        return "0/0"
    covered = coverage.get("covered")
    total = coverage.get("total")
    if not isinstance(covered, int) or not isinstance(total, int):
        return "0/0"
    return f"{covered}/{total}"


def _join_list(value: object) -> str:
    """Join list elements as diagnostic text or return empty for nonlist fields.

    This display helper does not validate physical content or provenance.
    """
    if not isinstance(value, list):
        return ""
    return "; ".join(str(item) for item in value if isinstance(item, str) and item.strip())


def _markdown_cell(value: str) -> str:
    """Escape table pipes and collapse whitespace in one display cell.

    Formatting changes text presentation rather than evidence admission.
    """
    return " ".join(value.replace("|", "\\|").split())


def main(argv: list[str] | None = None) -> int:
    """Run the standalone generator with strict source validation before any output write.

    Parameters
    ----------
    argv : list of str or None
        argparse tokens; None reads process arguments. Defaults select the
        canonical source-tree registry and Markdown report, independent of cwd.

    Returns
    -------
    int
        0 after writing valid Markdown, or 1 after reporting an invalid registry
        or supported IO/path failure to stderr. Successful output is UTF-8 with
        a trailing newline; missing parent directories are created.

    Raises
    ------
    SystemExit
        argparse help or argument refusal.

    Notes
    -----
    The selected registry and its resolved or existing hard-link aliases
    cannot be output destinations. Other registry-evidence paths are outside
    this destination check. Validation and writing are sequential; partial
    writes remain possible. Generating the report establishes no authenticated
    custody or publication.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default=str(ROOT / "validation" / "physics_traceability.json"),
        help="Physics traceability registry JSON path",
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "docs" / "physics_traceability.md"),
        help="Markdown report output path",
    )
    args = parser.parse_args(argv)

    try:
        markdown = generate_physics_traceability_markdown(args.registry, require_valid_registry=True)
        output_path = checked_report_destination(args.output_md, inputs=[args.registry])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Physics traceability report refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

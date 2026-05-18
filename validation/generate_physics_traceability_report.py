#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Physics Traceability Report Generator
"""Generate public bounded-claim physics traceability reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.validate_physics_traceability import validate_physics_traceability

_MARKDOWN_HEADER = """<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Physics Traceability and Bounded Claims -->
"""


def generate_physics_traceability_markdown(registry_path: str | Path) -> str:
    """Generate a public Markdown report from the checked traceability registry."""
    report = validate_physics_traceability(registry_path)
    lines = [
        _MARKDOWN_HEADER,
        "",
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
        f"- Source marker coverage: {_source_marker_coverage(report)}",
        "",
        "## Components",
        "",
    ]
    for entry in sorted(_entries(report), key=lambda item: str(item["component"])):
        claim_status = "allowed" if entry["public_claim_allowed"] else "blocked"
        lines.extend(
            [
                f"### {entry['component']}",
                "",
                f"- Fidelity status: `{entry['fidelity_status']}`",
                f"- Module path: `{entry['module_path']}`",
                f"- Full-fidelity public claim: {claim_status}",
                f"- Covered source paths: {entry.get('covered_source_count', 0)}",
                "- Required actions:",
            ]
        )
        for action in entry["required_actions"]:
            lines.append(f"  - {action}")
        lines.append("")
    if report["errors"]:
        lines.extend(["## Validation Errors", ""])
        for error in report["errors"]:
            lines.append(f"- `{error['field']}`: {error['error']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _entries(report: dict[str, Any]) -> list[dict[str, Any]]:
    entries = report.get("entries")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _source_marker_coverage(report: dict[str, Any]) -> str:
    coverage = report.get("source_marker_coverage")
    if not isinstance(coverage, dict):
        return "0/0"
    covered = coverage.get("covered")
    total = coverage.get("total")
    if not isinstance(covered, int) or not isinstance(total, int):
        return "0/0"
    return f"{covered}/{total}"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for Markdown traceability report generation."""
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

    markdown = generate_physics_traceability_markdown(args.registry)
    output_path = Path(args.output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

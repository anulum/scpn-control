#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Generated traceability freshness check
"""Fail if generated traceability documentation is stale."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.generate_physics_traceability_report import generate_physics_traceability_markdown


def generated_traceability_is_current(registry: Path, report_path: Path) -> bool:
    """Return True when the checked-in Markdown matches generated output."""
    expected = generate_physics_traceability_markdown(registry)
    actual = report_path.read_text(encoding="utf-8")
    return actual == expected


def main(argv: list[str] | None = None) -> int:
    """Check generated physics traceability Markdown against the registry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default=str(ROOT / "validation" / "physics_traceability.json"),
        help="Physics traceability registry JSON path",
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / "docs" / "physics_traceability.md"),
        help="Generated physics traceability Markdown path",
    )
    args = parser.parse_args(argv)

    registry = Path(args.registry)
    report_path = Path(args.report)
    if not registry.exists():
        print(f"{registry} is missing", file=sys.stderr)
        return 1
    if not report_path.exists():
        print(f"{report_path} is missing", file=sys.stderr)
        return 1
    if not generated_traceability_is_current(registry, report_path):
        print(
            f"{report_path} is stale; run "
            "python validation/generate_physics_traceability_report.py",
            file=sys.stderr,
        )
        return 1
    print("Generated traceability documentation is current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

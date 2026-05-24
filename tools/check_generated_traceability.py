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

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.generate_physics_traceability_report import generate_physics_traceability_markdown


def main() -> int:
    """Check generated physics traceability Markdown against the registry."""
    registry = ROOT / "validation" / "physics_traceability.json"
    report_path = ROOT / "docs" / "physics_traceability.md"
    expected = generate_physics_traceability_markdown(registry)
    actual = report_path.read_text(encoding="utf-8")
    if actual != expected:
        print(
            "docs/physics_traceability.md is stale; run "
            "python validation/generate_physics_traceability_report.py",
            file=sys.stderr,
        )
        return 1
    print("Generated traceability documentation is current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

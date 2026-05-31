#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Z3 Formal Verification Evidence Publisher

"""Publish bounded Z3 formal-verification evidence for a deterministic SCPN."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scpn_control.scpn.formal_verification import EventuallyFires, FireLeadsToMarking, NeverCoMarked
from scpn_control.scpn.structure import StochasticPetriNet
from scpn_control.scpn.z3_model_checking import verify_z3_formal_contracts, write_z3_formal_report

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "validation" / "reports" / "scpn_z3_formal.json"
DEFAULT_MARKDOWN = ROOT / "validation" / "reports" / "scpn_z3_formal.md"


def _reference_net() -> StochasticPetriNet:
    net = StochasticPetriNet()
    net.add_place("source", initial_tokens=1.0)
    net.add_place("sink", initial_tokens=0.0)
    net.add_transition("move", threshold=1.0)
    net.add_arc("source", "move", weight=1.0)
    net.add_arc("move", "sink", weight=1.0)
    net.compile()
    return net


def _blocked_report(error: str) -> dict[str, Any]:
    return {
        "status": "blocked",
        "backend": "z3",
        "holds": False,
        "reason": error,
        "scope": "bounded SMT evidence for compiled Petri-net control logic",
        "claim_boundary": "not hardware timing evidence, PCS certification, or unbounded liveness proof",
    }


def _write_blocked(report: dict[str, Any], *, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(
        "\n".join(
            [
                "# SCPN Z3 Formal Verification Report",
                "",
                "- Status: `blocked`",
                "- Backend: `z3`",
                f"- Reason: {report['reason']}",
                "- Scope: bounded SMT evidence for compiled Petri-net control logic.",
                "- Claim boundary: not hardware timing evidence, PCS certification, or unbounded liveness proof.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def publish_report(*, json_path: Path, markdown_path: Path, require_z3: bool) -> dict[str, Any]:
    """Publish deterministic Z3 evidence or an explicit blocked report."""
    try:
        report = verify_z3_formal_contracts(
            _reference_net(),
            max_depth=2,
            marking_bounds={"source": (0.0, 1.0), "sink": (0.0, 1.0)},
            temporal_specs=[
                EventuallyFires("move_eventually_fires", "move"),
                FireLeadsToMarking("move_marks_sink", "move", "sink", threshold=0.5, within=0),
                NeverCoMarked("exclusive_source_sink", "source", "sink", threshold=0.5),
            ],
        )
    except RuntimeError as exc:
        blocked = _blocked_report(str(exc))
        _write_blocked(blocked, json_path=json_path, markdown_path=markdown_path)
        if require_z3:
            raise
        return blocked
    write_z3_formal_report(report, json_path=json_path, markdown_path=markdown_path)
    return {"status": "pass" if report.holds else "fail", "backend": report.backend, "holds": report.holds}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--require-z3", action="store_true", help="Fail if z3-solver is unavailable")
    args = parser.parse_args(argv)
    try:
        result = publish_report(json_path=args.json_out, markdown_path=args.markdown_out, require_z3=args.require_z3)
    except RuntimeError as exc:
        print(f"SCPN Z3 formal verification: blocked: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in {"pass", "blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

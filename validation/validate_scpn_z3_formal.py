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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scpn_control.scpn.formal_verification import EventuallyFires, FireLeadsToMarking, NeverCoMarked
from scpn_control.scpn.structure import StochasticPetriNet
from scpn_control.scpn.z3_formal_report import (
    build_blocked_z3_formal_report_payload,
    verify_z3_formal_contracts,
    write_z3_formal_report,
)

DEFAULT_JSON = ROOT / "validation" / "reports" / "scpn_z3_formal.json"
DEFAULT_MARKDOWN = ROOT / "validation" / "reports" / "scpn_z3_formal.md"


def _reference_net() -> StochasticPetriNet:
    """Build a fresh compiled two-place, one-transition deterministic model.

    Returns
    -------
    StochasticPetriNet
        Source marking 1, sink marking 0 and move threshold/unit arc weights 1.
        Token densities/weights are dimensionless; no input file is read.
    """
    net = StochasticPetriNet()
    net.add_place("source", initial_tokens=1.0)
    net.add_place("sink", initial_tokens=0.0)
    net.add_transition("move", threshold=1.0)
    net.add_arc("source", "move", weight=1.0)
    net.add_arc("move", "sink", weight=1.0)
    net.compile()
    return net


def _blocked_report(error: str) -> dict[str, Any]:
    """Build the defining unavailable-Z3 declaration from a failure message.

    Parameters
    ----------
    error : str
        Reason forwarded to the schema/payload-digest builder.

    Returns
    -------
    dict[str, Any]
        Fresh blocked schema payload; no proof obligation is certified.
    """
    return build_blocked_z3_formal_report_payload(error)


def _write_blocked(report: dict[str, Any], *, json_path: Path, markdown_path: Path) -> None:
    """Persist a blocked declaration sequentially, without a transaction.

    Parameters
    ----------
    report : dict[str, Any]
        Defining blocked payload; this writer does not revalidate its fields.
    json_path, markdown_path : pathlib.Path
        Explicit targets resolved from caller cwd, with parent creation.

    Returns
    -------
    None
        Write UTF-8 JSON with final newline, then Markdown with final newline.

    Raises
    ------
    OSError
        Directory creation or write fails; earlier output may remain.
    KeyError, TypeError
        Missing formatting fields or unserialisable payload propagate.

    Notes
    -----
    Existing files are replaced. Output aliases are not checked; producer
    locks, atomic replacement and campaign/source authentication are absent.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(
        "\n".join(
            [
                "# SCPN Z3 Formal Verification Report",
                "",
                f"- Schema: `{report['schema_version']}`",
                "- Status: `blocked`",
                "- Backend: `z3`",
                f"- Solver: `{report['solver']}`",
                f"- Payload SHA-256: `{report['payload_sha256']}`",
                f"- Reason: {report['reason']}",
                f"- Scope: {report['scope']}.",
                f"- Claim boundary: {report['claim_boundary']}.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def publish_report(*, json_path: Path, markdown_path: Path, require_z3: bool) -> dict[str, Any]:
    """Prove fixed bounded obligations with installed Z3 and persist the result.

    Parameters
    ----------
    json_path, markdown_path : pathlib.Path
        Destinations with parent creation; relative paths use caller cwd.
    require_z3 : bool
        Re-raise a RuntimeError after writing its blocked declaration when True.

    Returns
    -------
    dict[str, Any]
        Passing/failing proof summary, or full blocked payload on a caught
        RuntimeError when require_z3 is False.

    Raises
    ------
    RuntimeError
        Model verification failed and require_z3 is True, after blocked output.
    OSError
        Sequential directory/writes fail; earlier output can remain.
    ValueError, TypeError, KeyError
        Unconverted defining model/schema/serialisation failures propagate.

    Notes
    -----
    Fresh source/sink/move net has dimensionless token/weight values, marking
    bounds [0,1] and at most two firings. Obligations are move eventual firing,
    same-step sink marking >=0.5 after firing and exclusion of co-marking >=0.5.
    All caught RuntimeError cases are labelled blocked, not only unavailable
    dependency errors. The publisher does no independent backend/proof-source
    authentication. JSON is written before Markdown; shared outputs/aliases
    have no lock or transaction. No hardware timing, PCS certification,
    unbounded liveness or physical claim follows from bounded holds=True.
    """
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
    return {
        "status": "pass" if report.holds else "fail",
        "backend": report.backend,
        "holds": report.holds,
        "max_depth": report.max_depth,
    }


def main(argv: list[str] | None = None) -> int:
    """Publish the fixed bounded Z3 model report and print a result summary.

    Parameters
    ----------
    argv : list[str] or None
        Parser arguments or process sys.argv. --json-out/--markdown-out
        override canonical checkout defaults; --require-z3 makes blocked fatal.

    Returns
    -------
    int
        Zero for pass or permitted blocked; one for fail or required-Z3
        RuntimeError. Standalone execution exits with this returned code.

    Raises
    ------
    SystemExit
        Parser help/code zero or invalid command syntax/code two.
    OSError, ValueError, TypeError, KeyError
        Unconverted model/persistence errors propagate.

    Notes
    -----
    Default outputs are checkout validation/reports/scpn_z3_formal.json/.md.
    Explicit relative destinations use caller cwd. This actually invokes the
    installed bounded Z3 checker; a successful process with status blocked
    is an availability declaration rather than a proof.
    """
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

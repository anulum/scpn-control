# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Free-boundary tracking acceptance.

"""Run and report the full normalized same-model tracking diagnostic.

Legacy run_campaign/generate_report/render_markdown and fixed threshold names
remain importable here. Persistent outputs require recorded campaign custody;
a successful diagnostic does not admit independent physical evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _source_root in (ROOT, ROOT / "src"):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

from scpn_control.benchmark_records import require_recorded_campaign
from validation.free_boundary_acceptance_campaign import run_campaign
from validation.free_boundary_acceptance_presets import (
    ACTUATOR_SLEW_LIMIT_SWEEP,
    COIL_KICK_SCALE_SWEEP,
    CORRECTED_THRESHOLDS,
    KICK_THRESHOLDS,
    LATENCY_CORRECTED_THRESHOLDS,
    LATENCY_STEP_SWEEP,
    LATENCY_THRESHOLDS,
    MEASUREMENT_SWEEP_SCALES,
    MEASUREMENT_THRESHOLDS,
    NOMINAL_THRESHOLDS,
    SUPERVISOR_FALLBACK_THRESHOLDS,
    TOPOLOGY_CORRECTED_THRESHOLDS,
    TOPOLOGY_DIVERTOR_STRIKE_POINTS,
    TOPOLOGY_KICK_SCALE_SWEEP,
    TOPOLOGY_LATENCY_MEASUREMENT_THRESHOLDS,
    TOPOLOGY_MEASUREMENT_THRESHOLDS,
    TOPOLOGY_SUPERVISOR_LATENCY_CORRECTED_THRESHOLDS,
    TOPOLOGY_THRESHOLDS,
)
from validation.free_boundary_acceptance_reports import generate_report, render_markdown
from validation.report_output_paths import checked_report_destination

__all__ = [
    "NOMINAL_THRESHOLDS",
    "ROOT",
    "KICK_THRESHOLDS",
    "MEASUREMENT_THRESHOLDS",
    "CORRECTED_THRESHOLDS",
    "LATENCY_THRESHOLDS",
    "LATENCY_CORRECTED_THRESHOLDS",
    "TOPOLOGY_THRESHOLDS",
    "TOPOLOGY_MEASUREMENT_THRESHOLDS",
    "TOPOLOGY_LATENCY_MEASUREMENT_THRESHOLDS",
    "TOPOLOGY_CORRECTED_THRESHOLDS",
    "TOPOLOGY_SUPERVISOR_LATENCY_CORRECTED_THRESHOLDS",
    "SUPERVISOR_FALLBACK_THRESHOLDS",
    "MEASUREMENT_SWEEP_SCALES",
    "LATENCY_STEP_SWEEP",
    "ACTUATOR_SLEW_LIMIT_SWEEP",
    "COIL_KICK_SCALE_SWEEP",
    "TOPOLOGY_KICK_SCALE_SWEEP",
    "TOPOLOGY_DIVERTOR_STRIKE_POINTS",
    "run_campaign",
    "generate_report",
    "render_markdown",
    "main",
]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the real full cohort after validating both destinations and custody.

    Parameters
    ----------
    argv : sequence of str, optional
        CLI arguments; None uses process arguments. --json-out/--md-out
        select caller-relative outputs. --require-thresholds requests exit
        one after both reports when any local diagnostic check fails.

    Returns
    -------
    int
        Zero after reporting, one for requested diagnostic failure, or two
        for supported input/custody/runtime/IO refusals. Parser help/usage
        retain argparse exits zero/two.

    Notes
    -----
        Selected implementation and core source aliases, including symlinks
        and hard links, and output/output aliases refuse before computing.
        Only listed sources are protected; no filesystem snapshot or lock
        exists. Temporary destinations need no campaign. Persistent evidence
        roots require the recorded runner's campaign identifier.
        JSON refuses nonfinite values. JSON then Markdown replace unrelated
        outputs sequentially; there is no atomic pair or automatic history.
        Independent physical admission remains false regardless of thresholds.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out", type=Path, default=ROOT / "validation/reports/free_boundary_tracking_acceptance.json"
    )
    parser.add_argument("--md-out", type=Path, default=ROOT / "validation/reports/free_boundary_tracking_acceptance.md")
    parser.add_argument("--require-thresholds", action="store_true")
    args = parser.parse_args(argv)
    try:
        sources = [
            Path(__file__),
            *sorted((ROOT / "validation").glob("free_boundary_acceptance_*.py")),
            ROOT / "src/scpn_control/core/fusion_kernel.py",
            ROOT / "src/scpn_control/control/free_boundary_tracking.py",
        ]
        json_out = checked_report_destination(args.json_out, inputs=[*sources, args.md_out])
        md_out = checked_report_destination(args.md_out, inputs=[*sources, args.json_out])
        require_recorded_campaign(json_out, md_out, repository_root=ROOT)
        report = generate_report()
        json_text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        markdown = render_markdown(report)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json_text, encoding="utf-8")
        md_out.write_text(markdown, encoding="utf-8")
    except (ValueError, RuntimeError, OSError):
        print("Free-boundary acceptance input, custody or execution refused.", file=sys.stderr)
        return 2
    print(f"Wrote {json_out}")
    print(f"Wrote {md_out}")
    return 1 if args.require_thresholds and not report["free_boundary_tracking_acceptance"]["passes_thresholds"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

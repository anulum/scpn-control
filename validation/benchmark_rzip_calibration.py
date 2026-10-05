# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Bounded RZIP calibration benchmark and report CLI.
"""Generate an actual symmetric-wall RZIP observation with explicit output custody."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from scpn_control.benchmark_records import require_recorded_campaign
from scpn_control.control import rzip_model
from scpn_control.control.rzip_model import (
    RZIPCalibrationEvidence,
    RZIPModel,
    rzip_calibration_evidence,
    save_rzip_calibration_evidence,
)
from scpn_control.core.vessel_model import VesselElement, VesselModel
from validation.report_output_paths import checked_report_destination

REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "rzip_calibration.json"
MD_REPORT = REPORT_DIR / "rzip_calibration.md"


def _benchmark_model() -> RZIPModel:
    """Build the unchanged local 2m, 1MA, n=-1, 1kg two-loop reference plant."""
    vessel = VesselModel(
        [
            VesselElement(R=2.0, Z=0.5, resistance=1.0e-3, cross_section=0.1, inductance=1.0e-5),
            VesselElement(R=2.0, Z=-0.5, resistance=1.0e-3, cross_section=0.1, inductance=1.0e-5),
        ]
    )
    return RZIPModel(
        R0=2.0,
        a=0.5,
        kappa=1.7,
        Ip_MA=1.0,
        B0=1.0,
        n_index=-1.0,
        vessel=vessel,
        vertical_inertia_kg=1.0,
    )


def _markdown_report(evidence: RZIPCalibrationEvidence) -> str:
    """Render the historical bounded-source Markdown from a checked observation."""
    return "\n".join(
        [
            "",
            "# RZIP Calibration Benchmark",
            "",
            "This report is bounded local regression evidence for the RZIP plant.",
            "It is not external CREATE-L/CREATE-NL/TSC or measured-discharge validation.",
            "",
            f"- Source: `{evidence.source}`",
            f"- Source ID: `{evidence.source_id}`",
            f"- Vertical inertia [kg]: `{evidence.vertical_inertia_kg:.6e}`",
            f"- Wall time constant [s]: `{evidence.wall_time_constant_s:.6e}`",
            f"- Growth rate [s^-1]: `{evidence.growth_rate_s_inv:.6e}`",
            f"- Growth time [ms]: `{evidence.growth_time_ms:.6e}`",
            f"- Evidence payload SHA-256: `{evidence.evidence_payload_sha256}`",
            f"- Facility claim allowed: `{evidence.facility_claim_allowed}`",
            f"- Claim boundary: `{evidence.claim_status}`",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the real fixed plant and print its checked, non-facility JSON observation.

    --no-write prints without IO. Otherwise JSON/Markdown paths retain their
    historical defaults; persistent evidence destinations still require the
    recorded-campaign wrapper. Explicit temporary paths require no campaign.
    Outputs must be distinct from each other and the three runtime sources.
    Parents are created and files replaced directly; a later Markdown failure
    may leave only JSON. No pair atomicity, calibration, source authentication
    or physical-control admission is implied. Returns 0 on success or 1 with
    a diagnostic for inspection/model/IO/custody refusal; argparse uses 0/2.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=JSON_REPORT)
    parser.add_argument("--markdown-out", type=Path, default=MD_REPORT)
    parser.add_argument("--no-write", action="store_true", help="print checked JSON without writing reports")
    args = parser.parse_args(argv)
    try:
        if not args.no_write:
            protected = [
                Path(__file__),
                Path(rzip_model.__file__),
                Path(rzip_model.__file__).with_name("_rzip_calibration.py"),
            ]
            json_path = checked_report_destination(args.json_out, inputs=[args.markdown_out, *protected])
            markdown_path = checked_report_destination(args.markdown_out, inputs=[args.json_out, *protected])
            require_recorded_campaign(json_path, markdown_path, repository_root=REPORT_DIR.parents[1])
        evidence = rzip_calibration_evidence(
            _benchmark_model(),
            source="local_regression_reference",
            source_id="validation/benchmark_rzip_calibration.py::symmetric_wall_case",
            wall_time_constant_s=0.01,
        )
        if not args.no_write:
            save_rzip_calibration_evidence(evidence, json_path)
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(_markdown_report(evidence), encoding="utf-8")
        print(json.dumps(asdict(evidence), indent=2, sort_keys=True))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"RZIP calibration benchmark refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

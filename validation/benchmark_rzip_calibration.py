# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — RZIP calibration benchmark
"""Publish bounded RZIP calibration/admission evidence."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from scpn_control.control.rzip_model import RZIPModel, rzip_calibration_evidence, save_rzip_calibration_evidence
from scpn_control.core.vessel_model import VesselElement, VesselModel

REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "rzip_calibration.json"
MD_REPORT = REPORT_DIR / "rzip_calibration.md"


def _benchmark_model() -> RZIPModel:
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


def main() -> None:
    """Run deterministic bounded RZIP calibration benchmark and write reports."""

    model = _benchmark_model()
    evidence = rzip_calibration_evidence(
        model,
        source="local_regression_reference",
        source_id="validation/benchmark_rzip_calibration.py::symmetric_wall_case",
        wall_time_constant_s=0.01,
    )
    payload = asdict(evidence)
    save_rzip_calibration_evidence(evidence, JSON_REPORT)
    MD_REPORT.write_text(
        "\n".join(
            [
                "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
                "<!-- Commercial license available -->",
                "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
                "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
                "<!-- ORCID: 0009-0009-3560-0851 -->",
                "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
                "<!-- SCPN Control — RZIP calibration benchmark report -->",
                "",
                "# RZIP Calibration Benchmark",
                "",
                "This report is bounded local regression evidence for the RZIP plant.",
                "It is not external CREATE-L/CREATE-NL/TSC or measured-discharge validation.",
                "",
                f"- Source: `{payload['source']}`",
                f"- Source ID: `{payload['source_id']}`",
                f"- Vertical inertia [kg]: `{payload['vertical_inertia_kg']:.6e}`",
                f"- Wall time constant [s]: `{payload['wall_time_constant_s']:.6e}`",
                f"- Growth rate [s^-1]: `{payload['growth_rate_s_inv']:.6e}`",
                f"- Growth time [ms]: `{payload['growth_time_ms']:.6e}`",
                f"- Facility claim allowed: `{payload['facility_claim_allowed']}`",
                f"- Claim boundary: `{payload['claim_status']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

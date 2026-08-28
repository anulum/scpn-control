# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Static structured-mu claim-admission benchmark
"""Publish bounded static structured-mu claim-admission evidence."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scpn_control.benchmark_records import require_recorded_campaign
from scpn_control.control.static_mu_analysis import (
    RiccatiStateFeedbackController,
    StructuredUncertainty,
    UncertaintyBlock,
    save_static_mu_analysis_claim_evidence,
    static_mu_analysis_claim_evidence,
)

REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "static_mu_analysis_claims.json"
MARKDOWN_REPORT = REPORT_DIR / "static_mu_analysis_claims.md"


def main() -> None:
    """Design the reference controller and write the declared evidence pair."""
    require_recorded_campaign(JSON_REPORT, MARKDOWN_REPORT, repository_root=REPORT_DIR.parents[1])
    plant = (
        np.array([[-1.4, 0.2], [-0.1, -0.9]], dtype=float),
        np.eye(2),
        np.eye(2),
        np.zeros((2, 2), dtype=float),
    )
    uncertainty = StructuredUncertainty(
        [
            UncertaintyBlock("plasma_position", 1, 0.02, "real_scalar"),
            UncertaintyBlock("plasma_current", 1, 0.03, "real_scalar"),
        ]
    )
    controller = RiccatiStateFeedbackController(plant, uncertainty)
    controller.design()
    evidence = static_mu_analysis_claim_evidence(
        controller,
        source="repository_static_mu_regression",
        source_id="static-mu-analysis-claim-benchmark-v1",
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    save_static_mu_analysis_claim_evidence(evidence, JSON_REPORT)
    payload = asdict(evidence)
    MARKDOWN_REPORT.write_text(
        "\n".join(
            [
                "# Static Structured-Mu Claim-Admission Benchmark",
                "",
                "This report records bounded repository-regression evidence for",
                "the static D-scaled mu-analysis claim boundary. It captures plant",
                "dimensions, uncertainty blocks, static mu upper bound, its",
                "reciprocal, controller gain norm, D-scalings, closed-loop spectral",
                "abscissa, and the explicit validated-claim boundary. The JSON",
                "retains legacy schema-v1 field names for wire compatibility.",
                "",
                f"- Claim status: `{payload['claim_status']}`",
                f"- Validated claim allowed: `{payload['validated_claim_allowed']}`",
                f"- Static mu upper bound at 0 rad/s: `{payload['mu_peak_upper_bound']:.12g}`",
                f"- Reciprocal static upper bound: `{payload['robustness_margin']:.12g}`",
                f"- Controller gain norm: `{payload['controller_gain_frobenius_norm']:.12g}`",
                f"- Closed-loop spectral abscissa: `{payload['closed_loop_spectral_abscissa']:.12g}` s^-1",
                "",
                "Bounded repository regression evidence is not full frequency-dependent D-K synthesis evidence.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    JSON_REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

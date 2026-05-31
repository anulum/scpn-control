# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Mu-synthesis claim-admission benchmark

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scpn_control.control.mu_synthesis import (
    MuSynthesisController,
    StructuredUncertainty,
    UncertaintyBlock,
    mu_synthesis_claim_evidence,
    save_mu_synthesis_claim_evidence,
)

REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "mu_synthesis_claims.json"
MARKDOWN_REPORT = REPORT_DIR / "mu_synthesis_claims.md"


def main() -> None:
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
    controller = MuSynthesisController(plant, uncertainty)
    controller.synthesize(n_dk_iter=3)
    evidence = mu_synthesis_claim_evidence(
        controller,
        source="repository_static_mu_regression",
        source_id="mu-synthesis-claim-benchmark-v1",
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    save_mu_synthesis_claim_evidence(evidence, JSON_REPORT)
    payload = asdict(evidence)
    MARKDOWN_REPORT.write_text(
        "\n".join(
            [
                "# Mu-synthesis Claim-Admission Benchmark",
                "",
                "This report records bounded repository-regression evidence for",
                "the static D-scaled mu-analysis claim boundary. It captures plant",
                "dimensions, uncertainty blocks, mu upper bound, robustness margin,",
                "controller gain norm, D-scalings, closed-loop spectral abscissa,",
                "and the explicit validated-claim boundary.",
                "",
                f"- Claim status: `{payload['claim_status']}`",
                f"- Validated claim allowed: `{payload['validated_claim_allowed']}`",
                f"- Mu upper bound: `{payload['mu_peak_upper_bound']:.12g}`",
                f"- Robustness margin: `{payload['robustness_margin']:.12g}`",
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

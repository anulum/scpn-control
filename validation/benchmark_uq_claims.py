# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Uncertainty quantification claim-admission benchmark

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from scpn_control.core.uncertainty import PlasmaScenario, quantify_full_chain, uq_claim_evidence


REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "uq_claims.json"
MARKDOWN_REPORT = REPORT_DIR / "uq_claims.md"


def build_reference_scenario() -> PlasmaScenario:
    return PlasmaScenario(I_p=15.0, B_t=5.3, P_heat=50.0, n_e=10.1, R=6.2, A=3.1, kappa=1.7, M=2.5)


def main() -> None:
    scenario = build_reference_scenario()
    seed = 31
    result = quantify_full_chain(scenario, n_samples=256, seed=seed)
    evidence = uq_claim_evidence(
        scenario,
        result,
        source="synthetic_regression_reference",
        source_id="uq-bounded-regression-v1",
        scenario_source="repository ITER-like scenario fixture",
        prior_source="repository IPB98 covariance registry",
        propagation_chain="IPB98 -> Bosch-Hale fusion power -> Q",
        sensitivity_source="finite-difference density and temperature sensitivities",
        seed=seed,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MARKDOWN_REPORT.write_text(
        "\n".join(
            [
                "# UQ Claim-Admission Benchmark",
                "",
                "This report records bounded synthetic-regression evidence for",
                "full-chain uncertainty quantification claim admission. It captures",
                "scenario provenance, prior provenance, propagation chain, seed,",
                "sample count, ordered percentile checks, finite outputs, D-T fuel",
                "dilution, and density/temperature sensitivity provenance.",
                "",
                f"- Claim status: `{evidence.claim_status}`",
                f"- Calibrated UQ claim allowed: `{evidence.calibrated_uq_claim_allowed}`",
                f"- Seed: `{evidence.seed}`",
                f"- Samples: `{evidence.n_samples}`",
                f"- tau_E: `{evidence.tau_E_s:.12g}` s",
                f"- P_fusion: `{evidence.P_fusion_MW:.12g}` MW",
                f"- Q: `{evidence.Q:.12g}`",
                f"- Finite outputs: `{evidence.finite_outputs}`",
                "",
                "Synthetic regression evidence is not calibrated facility predictive uncertainty.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

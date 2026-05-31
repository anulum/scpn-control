# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Burn-control claim-admission benchmark

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scpn_control.control.burn_controller import (
    AlphaHeating,
    BurnController,
    burn_control_claim_evidence,
    save_burn_control_claim_evidence,
)

REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "burn_control_claims.json"
MARKDOWN_REPORT = REPORT_DIR / "burn_control_claims.md"


def main() -> None:
    rho = np.linspace(0.0, 1.0, 48)
    ne = 0.85 + 0.25 * (1.0 - rho**2)
    temperature = 14.0 + 8.0 * (1.0 - rho**1.7)
    alpha = AlphaHeating(R0=6.2, a=2.0, kappa=1.7)
    controller = BurnController(Q_target=10.0, T_target_keV=20.0, P_aux_max_MW=73.0)
    evidence = burn_control_claim_evidence(
        alpha,
        controller,
        rho=rho,
        ne_20=ne,
        Te_keV=temperature,
        Ti_keV=temperature,
        tau_E_s=3.7,
        P_aux_MW=50.0,
        source="repository_burn_regression",
        source_id="burn-control-claim-benchmark-v1",
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    save_burn_control_claim_evidence(evidence, JSON_REPORT)
    payload = asdict(evidence)
    MARKDOWN_REPORT.write_text(
        "\n".join(
            [
                "# Burn-control Claim-Admission Benchmark",
                "",
                "This report records bounded repository-regression evidence for",
                "the DT burn-control and alpha-heating claim boundary. It captures",
                "alpha power, auxiliary power, Q, Lawson margin, burn fraction,",
                "reactivity exponent, thermal stability, controller limits, and the",
                "explicit reactor-claim boundary.",
                "",
                f"- Claim status: `{payload['claim_status']}`",
                f"- Reactor claim allowed: `{payload['reactor_claim_allowed']}`",
                f"- P_alpha: `{payload['P_alpha_MW']:.12g}` MW",
                f"- P_aux: `{payload['P_aux_MW']:.12g}` MW",
                f"- Q: `{payload['Q']:.12g}`",
                f"- Lawson margin: `{payload['lawson_margin']:.12g}`",
                f"- Burn fraction: `{payload['burn_fraction']:.12g}`",
                f"- Reactivity exponent: `{payload['reactivity_exponent']:.12g}`",
                f"- Thermally stable: `{payload['thermally_stable']}`",
                f"- Controller command: `{payload['controller_command_MW']:.12g}` MW",
                "",
                "Bounded repository regression evidence is not validated reactor burn-control evidence.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    JSON_REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

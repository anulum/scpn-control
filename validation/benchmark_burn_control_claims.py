# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Burn-control claim-admission benchmark

"""Publish a fixed bounded model declaration through recorded output custody."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scpn_control.benchmark_records import require_recorded_campaign
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
    """Write a fixed bounded DT burn declaration using fresh model/controller state.

    Returns
    -------
    None
        Write burn_control_claims.json, burn_control_claims.md, then the same
        JSON payload again in this module's reports directory. Outputs use
        UTF-8 with final newlines and the defining evidence dataclass schema.

    Raises
    ------
    RuntimeError
        Persistent paths lack a recorded-campaign identifier.
    ValueError
        The identifier or defining scientific inputs/evidence are refused.
    OSError
        Creating the directory or a sequential write fails. Earlier writes may
        remain; there is no atomic replacement or multi-file transaction.

    Notes
    -----
    There are no CLI parameters. rho has 48 points from 0 to 1; ne_20 is
    0.85+0.25*(1-rho**2) in 1e20 m^-3 and both temperatures are
    14+8*(1-rho**1.7) keV. Alpha geometry fixes R0=6.2 m, a=2 m, kappa=1.7.
    Confinement time is 3.7 s and auxiliary power 50 MW. Controller targets are
    Q=10 and T=20 keV with maximum auxiliary power 73 MW. The defining builder
    computes weighted profile metrics and one controller step with dt=0.1 s;
    it does not evolve a burn trajectory or apply a closed-loop actuator replay.
    No reference artifact is supplied and reactor_claim_allowed remains False.
    Shared fixed filenames have no producer locks. The destination guard checks
    campaign ID presence/syntax, not authentic source or reactor evidence; the
    actual recorded wrapper separately reserves and preserves output custody.
    Numerical alpha/reactivity/Lawson/command semantics belong to defining APIs.
    """
    require_recorded_campaign(JSON_REPORT, MARKDOWN_REPORT, repository_root=REPORT_DIR.parents[1])
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

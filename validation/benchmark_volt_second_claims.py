# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Volt-second claim-admission benchmark

"""Publish a fixed bounded model declaration through recorded output custody."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from scpn_control.benchmark_records import require_recorded_campaign
from scpn_control.control.volt_second_manager import (
    FluxBudget,
    ScenarioFluxAnalysis,
    save_volt_second_claim_evidence,
    volt_second_claim_evidence,
)

REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "volt_second_claims.json"
MARKDOWN_REPORT = REPORT_DIR / "volt_second_claims.md"


def main() -> None:
    """Write one fixed bounded volt-second accounting declaration and summary.

    Returns
    -------
    None
        Write volt_second_claims.json, volt_second_claims.md, then the same
        JSON payload again in this module's reports directory. Outputs are
        UTF-8 with final newlines, following the defining dataclass schema.

    Raises
    ------
    RuntimeError
        Persistent paths lack a recorded-campaign identifier.
    ValueError
        The identifier or defining budget/scenario/evidence inputs are refused.
    OSError
        Directory creation or a sequential write fails; earlier writes can
        remain. Replacement is not atomic or a multi-file transaction.

    Notes
    -----
    No CLI parameters are parsed. Fresh scalar budget state sets Phi_CS=120 V s,
    plasma inductance 1.2 microhenry and resistance 0.08 microohm. The scenario
    fixes ramp/flat/down durations 80/400/60 s, plasma current 15 MA and supplied
    bootstrap current 4 MA. R0=6.2 m is passed to the defining evidence builder,
    along with the actual computed ramp flux for its flat-top estimate.
    This is scalar scenario accounting, not measured voltage integration,
    current-profile evolution or central-solenoid commissioning. No external
    reference is supplied and facility_claim_allowed remains False.
    Reports expose defining-model flux in V s and timing in seconds. Calls
    share fixed filenames without locks. The guard checks campaign ID syntax/
    presence; the actual wrapper separately records custody. Neither supplies
    authentic independent physical references or source authentication.
    """
    require_recorded_campaign(JSON_REPORT, MARKDOWN_REPORT, repository_root=REPORT_DIR.parents[1])
    budget = FluxBudget(Phi_CS_Vs=120.0, L_plasma_uH=1.2, R_plasma_uOhm=0.08)
    report = ScenarioFluxAnalysis(budget).analyze(
        ramp_dur=80.0,
        flat_dur=400.0,
        down_dur=60.0,
        Ip_MA=15.0,
        I_bs_MA=4.0,
    )
    evidence = volt_second_claim_evidence(
        budget,
        report,
        Ip_MA=15.0,
        I_bs_MA=4.0,
        ramp_duration_s=80.0,
        flat_duration_s=400.0,
        ramp_down_duration_s=60.0,
        R0_m=6.2,
        ramp_flux_for_flattop_Vs=report.ramp_flux,
        source="repository_volt_second_regression",
        source_id="volt-second-claim-benchmark-v1",
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    save_volt_second_claim_evidence(evidence, JSON_REPORT)
    payload = asdict(evidence)
    MARKDOWN_REPORT.write_text(
        "\n".join(
            [
                "# Volt-second Claim-Admission Benchmark",
                "",
                "This report records bounded repository-regression evidence for",
                "the scenario volt-second accounting claim boundary. It captures",
                "ramp, flat-top, and ramp-down flux consumption, Ejima startup",
                "flux, bootstrap-current correction, remaining flat-top time,",
                "budget margin, and the explicit facility-claim boundary.",
                "",
                f"- Claim status: `{payload['claim_status']}`",
                f"- Facility claim allowed: `{payload['facility_claim_allowed']}`",
                f"- Total flux: `{payload['total_flux_Vs']:.12g}` V s",
                f"- Margin: `{payload['margin_Vs']:.12g}` V s",
                f"- Within budget: `{payload['within_budget']}`",
                f"- Ejima startup flux: `{payload['ejima_startup_flux_Vs']:.12g}` V s",
                f"- Max flat-top duration: `{payload['max_flattop_duration_s']:.12g}` s",
                "",
                "Bounded repository regression evidence is not validated pulse-design or solenoid commissioning evidence.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    JSON_REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Current-drive claim-admission benchmark

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scpn_control.core.current_drive import (
    CurrentDriveMix,
    ECCDSource,
    LHCDSource,
    NBISource,
    current_drive_claim_evidence,
    save_current_drive_claim_evidence,
)

REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "current_drive_claims.json"
MARKDOWN_REPORT = REPORT_DIR / "current_drive_claims.md"


def main() -> None:
    rho = np.linspace(0.0, 1.0, 80)
    ne = 7.0 - 2.0 * rho**2
    te = 12.0 - 6.0 * rho**1.5
    ti = 10.0 - 5.0 * rho**1.4
    mix = CurrentDriveMix(a=2.0)
    mix.add_source(ECCDSource(P_ec_MW=8.0, rho_dep=0.35, sigma_rho=0.08))
    mix.add_source(LHCDSource(P_lh_MW=3.0, rho_dep=0.65, sigma_rho=0.12))
    mix.add_source(NBISource(P_nbi_MW=14.0, E_beam_keV=1000.0, rho_tangency=0.45, sigma_rho=0.15))
    evidence = current_drive_claim_evidence(
        mix,
        rho=rho,
        ne_19=ne,
        Te_keV=te,
        Ti_keV=ti,
        source="repository_current_drive_regression",
        source_id="current-drive-claim-benchmark-v1",
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    save_current_drive_claim_evidence(evidence, JSON_REPORT)
    payload = asdict(evidence)
    MARKDOWN_REPORT.write_text(
        "\n".join(
            [
                "# Current-drive Claim-Admission Benchmark",
                "",
                "This report records bounded repository-regression evidence for",
                "the ECCD, LHCD, and NBI current-drive claim boundary. It captures",
                "grid-normalised absorbed power, total driven current, peak current",
                "density, source powers, efficiency coefficients, NBI slowing-down",
                "metadata, and the explicit external-claim boundary.",
                "",
                f"- Claim status: `{payload['claim_status']}`",
                f"- External claim allowed: `{payload['external_claim_allowed']}`",
                f"- Total absorbed power: `{payload['total_absorbed_power_W']:.12g}` W",
                f"- Total driven current: `{payload['total_driven_current_A']:.12g}` A",
                f"- Peak current density: `{payload['peak_current_density_A_m2']:.12g}` A/m^2",
                f"- Current-drive efficiency: `{payload['current_drive_efficiency_A_W']:.12g}` A/W",
                f"- Grid-normalised power: `{payload['grid_normalised_power']}`",
                "",
                "Bounded repository regression evidence is not ray-traced, Fokker-Planck, or facility deposition evidence.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    JSON_REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

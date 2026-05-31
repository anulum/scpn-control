# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Orbit-following claim-admission benchmark

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scpn_control.core.orbit_following import EnsembleResult, first_orbit_loss, orbit_following_claim_evidence


REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "orbit_following_claims.json"
MARKDOWN_REPORT = REPORT_DIR / "orbit_following_claims.md"


def main() -> None:
    loss = first_orbit_loss(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0)
    ensemble = EnsembleResult(
        loss_fraction=0.2,
        heating_profile=np.zeros(50),
        current_drive=0.0,
        n_passing=6,
        n_trapped=2,
        n_lost=2,
    )
    evidence = orbit_following_claim_evidence(
        source="synthetic_regression_reference",
        source_id="orbit-following-bounded-regression-v1",
        geometry_source="repository large-aspect-ratio tokamak fixture",
        particle_source="repository alpha-particle birth fixture",
        collision_model="Stix slowing-down fixture",
        loss_boundary_source="repository first-orbit wall boundary fixture",
        q=2.0,
        rho_L_m=0.05,
        epsilon=0.25,
        first_orbit_loss_fraction=loss,
        ensemble_result=ensemble,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MARKDOWN_REPORT.write_text(
        "\n".join(
            [
                "# Orbit-following Claim-Admission Benchmark",
                "",
                "This report records bounded synthetic-regression evidence for",
                "guiding-centre orbit-following claim admission. It captures",
                "geometry provenance, particle provenance, collision model, loss",
                "boundary provenance, banana width, first-orbit loss, and ensemble",
                "classification counts.",
                "",
                f"- Claim status: `{evidence.claim_status}`",
                f"- External orbit claim allowed: `{evidence.external_orbit_claim_allowed}`",
                f"- Banana width: `{evidence.banana_width_m:.12g}` m",
                f"- First-orbit loss: `{evidence.first_orbit_loss_fraction:.12g}`",
                f"- Ensemble particles: `{evidence.ensemble_particles}`",
                f"- Ensemble lost: `{evidence.ensemble_lost}`",
                "",
                "Synthetic regression evidence is not external orbit-code validation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Density-control claim-admission benchmark

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scpn_control.control.density_controller import (
    DensityController,
    ParticleTransportModel,
    density_control_claim_evidence,
)


REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "density_control_claims.json"
MARKDOWN_REPORT = REPORT_DIR / "density_control_claims.md"


def main() -> None:
    model = ParticleTransportModel(n_rho=12, R0=6.2, a=2.0)
    controller = DensityController(model, dt_control=0.01)
    controller.set_constraints(n_GW=1.0e20, gas_max=1.0e22, pellet_freq_max=10.0, pump_max=10.0)
    controller.set_target(np.ones(model.n_rho) * 5.0e19)
    ne_before = np.ones(model.n_rho) * 2.0e19
    sources = model.gas_puff_source(rate=1.0e20, penetration_depth=0.08)
    ne_after = model.step(ne_before, sources, dt=1.0)
    command = controller.step(ne_before)
    evidence = density_control_claim_evidence(
        model,
        controller,
        source="synthetic_regression_reference",
        source_id="density-control-bounded-regression-v1",
        geometry_source="repository circular ITER-like geometry fixture",
        transport_source="repository finite-volume diffusion-pinch fixture",
        actuator_source="repository gas-puff and cryopump actuator limits",
        diagnostic_source="repository density profile fixture",
        ne_before=ne_before,
        ne_after=ne_after,
        sources=sources,
        command=command,
        dt_requested_s=1.0,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MARKDOWN_REPORT.write_text(
        "\n".join(
            [
                "# Density-control Claim-Admission Benchmark",
                "",
                "This report records bounded synthetic-regression evidence for",
                "density-control claim admission. It captures geometry provenance,",
                "transport provenance, actuator provenance, diagnostic provenance,",
                "CFL limiting, Greenwald fraction, source integral, particle",
                "inventory change, and actuator command bounds.",
                "",
                f"- Claim status: `{evidence.claim_status}`",
                f"- Facility density claim allowed: `{evidence.facility_density_claim_allowed}`",
                f"- Greenwald fraction: `{evidence.greenwald_fraction:.12g}`",
                f"- Below ITER margin: `{evidence.below_iter_greenwald_margin}`",
                f"- CFL limited: `{evidence.cfl_limited}`",
                f"- Total source: `{evidence.total_source_particles_per_s:.12g}` particles/s",
                f"- Particle inventory delta: `{evidence.particle_inventory_delta:.12g}`",
                "",
                "Synthetic regression evidence is not facility-calibrated density-control validation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

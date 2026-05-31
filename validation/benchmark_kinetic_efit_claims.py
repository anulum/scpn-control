# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Kinetic EFIT claim-admission benchmark

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scpn_control.control.realtime_efit import MagneticDiagnostics
from scpn_control.core.kinetic_efit import (
    FastIonPressure,
    KineticConstraints,
    KineticEFIT,
    kinetic_efit_claim_evidence,
)


REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "kinetic_efit_claims.json"
MARKDOWN_REPORT = REPORT_DIR / "kinetic_efit_claims.md"


def build_reference_case() -> tuple[KineticEFIT, KineticConstraints, FastIonPressure]:
    diagnostics = MagneticDiagnostics([(2.0, 1.0)], [(2.0, 1.0, "R")], rogowski_radius=3.0)
    kinetic = KineticConstraints(
        Te_points=[(6.0, 0.0, 10.0), (7.9, 0.0, 1.0)],
        ne_points=[(6.0, 0.0, 5.0), (7.9, 0.0, 0.5)],
        Ti_points=[(6.0, 0.0, 8.0), (7.9, 0.0, 0.8)],
        mse_points=[(6.5, 0.0, 5.0)],
    )
    fast_ions = FastIonPressure(E_fast_keV=100.0, n_fast_frac=0.1, anisotropy_sigma=0.2)
    r_grid = np.linspace(4.0, 8.0, 33)
    z_grid = np.linspace(-3.0, 3.0, 33)
    return KineticEFIT(diagnostics, kinetic, fast_ions, r_grid, z_grid), kinetic, fast_ions


def main() -> None:
    kefit, kinetic, fast_ions = build_reference_case()
    result = kefit.reconstruct({})
    evidence = kinetic_efit_claim_evidence(
        result,
        kinetic,
        fast_ions,
        source="synthetic_regression_reference",
        source_id="kinetic-efit-bounded-regression-v1",
        diagnostic_source="repository magnetic diagnostics fixture",
        profile_source="repository Thomson and ion-temperature fixture",
        fast_ion_source="repository anisotropic fast-ion fixture",
        mse_calibration_source="repository MSE pitch-angle fixture",
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = asdict(evidence)
    JSON_REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MARKDOWN_REPORT.write_text(
        "\n".join(
            [
                "# Kinetic EFIT Claim-Admission Benchmark",
                "",
                "This report records bounded synthetic-regression evidence for kinetic pressure,",
                "q-profile, anisotropy, diagnostic provenance, profile provenance, fast-ion",
                "provenance, MSE calibration, and interpolation geometry.",
                "",
                f"- Claim status: `{evidence.claim_status}`",
                f"- Facility claim allowed: `{evidence.facility_claim_allowed}`",
                f"- Interpolation geometry: `{evidence.interpolation_geometry}`",
                f"- Pressure consistency: `{evidence.pressure_consistency:.12g}`",
                f"- Fast-ion beta: `{evidence.beta_fast:.12g}`",
                f"- q-axis: `{evidence.q_axis:.12g}`",
                f"- q-edge: `{evidence.q_edge:.12g}`",
                "",
                "Synthetic regression evidence is not a facility EFIT or P-EFIT validation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

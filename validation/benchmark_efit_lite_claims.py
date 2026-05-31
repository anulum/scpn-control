# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — EFIT-lite claim-admission benchmark
"""Publish bounded EFIT-lite claim-admission evidence."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from scpn_control.control.realtime_efit import (
    MagneticDiagnostics,
    RealtimeEFIT,
    efit_lite_claim_evidence,
    save_efit_lite_claim_evidence,
)

REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "efit_lite_claims.json"
MD_REPORT = REPORT_DIR / "efit_lite_claims.md"


def main() -> None:
    """Run deterministic bounded EFIT-lite claim benchmark and write reports."""

    diagnostics = MagneticDiagnostics(
        flux_loops=[(4.8, -1.5), (6.2, 0.0), (7.6, 1.5)],
        b_probes=[(4.8, 0.0, "R"), (6.2, 0.0, "Z"), (7.6, 0.0, "Z")],
        rogowski_radius=6.2,
    )
    r_grid = np.linspace(4.2, 8.2, 33)
    z_grid = np.linspace(-3.0, 3.0, 33)
    efit = RealtimeEFIT(diagnostics, r_grid, z_grid)
    result = efit.reconstruct(
        {
            "flux_loops": np.zeros(len(diagnostics.flux_loops)),
            "b_probes": np.zeros(len(diagnostics.b_probes)),
            "Ip": 15.0e6,
            "coil_currents": np.zeros(5),
        }
    )
    evidence = efit_lite_claim_evidence(
        result,
        diagnostics,
        source="synthetic_regression_reference",
        source_id="validation/benchmark_efit_lite_claims.py::synthetic_fixed_boundary_case",
        diagnostic_source="synthetic diagnostic response",
    )
    payload = asdict(evidence)
    save_efit_lite_claim_evidence(evidence, JSON_REPORT)
    MD_REPORT.write_text(
        "\n".join(
            [
                "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
                "<!-- Commercial license available -->",
                "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
                "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
                "<!-- ORCID: 0009-0009-3560-0851 -->",
                "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
                "<!-- SCPN Control — EFIT-lite claim-admission benchmark report -->",
                "",
                "# EFIT-lite Claim-Admission Benchmark",
                "",
                "This report is bounded synthetic regression evidence for EFIT-lite.",
                "It is not matched EFIT/P-EFIT or measured-discharge validation.",
                "",
                f"- Source: `{payload['source']}`",
                f"- Diagnostic source: `{payload['diagnostic_source']}`",
                f"- Grid shape: `{payload['grid_shape']}`",
                f"- Flux loops: `{payload['n_flux_loops']}`",
                f"- B probes: `{payload['n_b_probes']}`",
                f"- Reconstructed Ip [A]: `{payload['ip_reconstructed_A']:.6e}`",
                f"- q95: `{payload['q95']:.6e}`",
                f"- beta_pol: `{payload['beta_pol']:.6e}`",
                f"- li: `{payload['li']:.6e}`",
                f"- Facility claim allowed: `{payload['facility_claim_allowed']}`",
                f"- Claim boundary: `{payload['claim_status']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

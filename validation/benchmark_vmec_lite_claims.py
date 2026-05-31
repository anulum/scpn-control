# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — VMEC-lite claim-admission benchmark

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scpn_control.core.vmec_lite import StellaratorBoundary, VMECLiteSolver, vmec_lite_claim_evidence


REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "vmec_lite_claims.json"
MARKDOWN_REPORT = REPORT_DIR / "vmec_lite_claims.md"


def build_reference_case() -> VMECLiteSolver:
    solver = VMECLiteSolver(n_s=11, m_pol=2, n_tor=1, n_fp=5)
    b_R, b_Z = StellaratorBoundary.w7x_standard()
    solver.set_boundary(b_R, b_Z)
    solver.set_profiles(np.linspace(5.0e4, 0.0, 11), np.ones(11) * 0.9)
    return solver


def main() -> None:
    solver = build_reference_case()
    result = solver.solve(max_iter=100, tol=1e-3)
    evidence = vmec_lite_claim_evidence(
        solver,
        result,
        source="synthetic_regression_reference",
        source_id="vmec-lite-bounded-regression-v1",
        geometry_source="repository W7-X-like Fourier boundary fixture",
        profile_source="repository pressure and rotational-transform fixture",
        current_assumption="fixed-boundary reduced MHD with no external current-profile claim",
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MARKDOWN_REPORT.write_text(
        "\n".join(
            [
                "# VMEC-lite Claim-Admission Benchmark",
                "",
                "This report records bounded synthetic-regression evidence for the",
                "VMEC-lite fixed-boundary spectral facade. It captures Fourier",
                "truncation, field periods, pressure and rotational-transform",
                "profile provenance, current-assumption provenance, positive sampled",
                "major radius, force residual, and q-domain.",
                "",
                f"- Claim status: `{evidence.claim_status}`",
                f"- Full VMEC claim allowed: `{evidence.full_vmec_claim_allowed}`",
                f"- Fourier modes: `{evidence.n_modes}`",
                f"- Field periods: `{evidence.n_fp}`",
                f"- Force residual: `{evidence.force_residual:.12g}`",
                f"- Minimum sampled major radius: `{evidence.min_major_radius:.12g}`",
                f"- q range: `{evidence.q_min:.12g}` to `{evidence.q_max:.12g}`",
                "",
                "Synthetic regression evidence is not a full VMEC validation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

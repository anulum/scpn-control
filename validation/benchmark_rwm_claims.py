# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — RWM claim-admission benchmark
"""Publish bounded RWM feedback claim-admission evidence."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from scpn_control.control.rwm_feedback import (
    RWMFeedbackController,
    RWMPhysics,
    rwm_claim_evidence,
    save_rwm_claim_evidence,
)

REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "rwm_claims.json"
MD_REPORT = REPORT_DIR / "rwm_claims.md"


def main() -> None:
    """Run deterministic bounded RWM claim-admission benchmark and write reports."""

    rwm = RWMPhysics(
        beta_n=3.0,
        beta_n_nowall=2.8,
        beta_n_wall=3.5,
        tau_wall=0.01,
        omega_phi=50.0,
        wall_radius=0.6,
        plasma_radius=0.5,
    )
    controller = RWMFeedbackController(
        n_sensors=3,
        n_coils=2,
        G_p=1.1,
        G_d=0.02,
        tau_controller=2.0e-4,
        M_coil=0.8,
    )
    evidence = rwm_claim_evidence(
        rwm,
        controller,
        source="local_regression_reference",
        source_id="validation/benchmark_rwm_claims.py::wall_rotation_feedback_case",
        closed_loop_growth_rate_abs_tolerance=0.5,
    )
    payload = asdict(evidence)
    save_rwm_claim_evidence(evidence, JSON_REPORT)
    MD_REPORT.write_text(
        "\n".join(
            [
                "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
                "<!-- Commercial license available -->",
                "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
                "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
                "<!-- ORCID: 0009-0009-3560-0851 -->",
                "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
                "<!-- SCPN Control — RWM claim-admission benchmark report -->",
                "",
                "# RWM Claim-Admission Benchmark",
                "",
                "This report is bounded local regression evidence for the RWM feedback model.",
                "It is not external MHD stability or measured-shot validation.",
                "",
                f"- Source: `{payload['source']}`",
                f"- Source ID: `{payload['source_id']}`",
                f"- Beta_N: `{payload['beta_n']:.6e}`",
                f"- No-wall beta_N limit: `{payload['beta_n_nowall']:.6e}`",
                f"- Ideal-wall beta_N limit: `{payload['beta_n_wall']:.6e}`",
                f"- Effective wall time [s]: `{payload['tau_eff_s']:.6e}`",
                f"- Rotation [rad/s]: `{payload['omega_phi_rad_s']:.6e}`",
                f"- Open-loop growth rate [s^-1]: `{payload['open_loop_growth_rate_s_inv']:.6e}`",
                f"- Closed-loop growth rate [s^-1]: `{payload['closed_loop_growth_rate_s_inv']:.6e}`",
                f"- Required proportional gain: `{payload['required_proportional_gain']:.6e}`",
                f"- Facility claim allowed: `{payload['facility_claim_allowed']}`",
                f"- Claim boundary: `{payload['claim_status']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

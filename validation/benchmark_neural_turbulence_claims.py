# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural turbulence claim-admission benchmark
"""Publish bounded neural-turbulence claim-admission evidence."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from scpn_control.core.neural_turbulence import (
    QLKNNSurrogate,
    cross_validate_neural_turbulence,
    neural_turbulence_claim_evidence,
)

REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "neural_turbulence_claims.json"
MD_REPORT = REPORT_DIR / "neural_turbulence_claims.md"


def main() -> None:
    """Run deterministic bounded neural-turbulence claim benchmark."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    surrogate = QLKNNSurrogate(pretrained=True)
    validation_result = cross_validate_neural_turbulence(surrogate, n_samples=256, seed=20240531)
    evidence = neural_turbulence_claim_evidence(
        validation_result,
        source="synthetic_regression_reference",
        source_id="validation/benchmark_neural_turbulence_claims.py::analytic_target_reference",
    )
    payload = {
        "validation_result": validation_result,
        "claim_evidence": asdict(evidence),
        "claim_boundary": (
            "Local analytic-target regression evidence only. Quantitative gyrokinetic, "
            "QuaLiKiz, or documented-reference neural-turbulence claims require matched reference artifacts."
        ),
    }
    JSON_REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MD_REPORT.write_text(
        "\n".join(
            [
                "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
                "<!-- Commercial license available -->",
                "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
                "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
                "<!-- ORCID: 0009-0009-3560-0851 -->",
                "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
                "<!-- SCPN Control — Neural turbulence claim-admission benchmark report -->",
                "",
                "# Neural turbulence claim-admission benchmark",
                "",
                f"- Local samples: {evidence.local_sample_count}",
                f"- Q_i RMSE: {evidence.local_q_i_rmse_gB:.8e} gyro-Bohm",
                f"- Q_e RMSE: {evidence.local_q_e_rmse_gB:.8e} gyro-Bohm",
                f"- Gamma_e RMSE: {evidence.local_gamma_e_rmse_gB:.8e} gyro-Bohm",
                f"- Critical-gradient accuracy: {evidence.local_critical_gradient_accuracy:.8f}",
                f"- Claim admission: {evidence.claim_status}",
                f"- Quantitative claim allowed: {evidence.quantitative_claim_allowed}",
                "",
                "Claim boundary: this report checks the deterministic local",
                "analytic-target regression path and records fail-closed admission",
                "evidence. It is not gyrokinetic campaign or measured turbulence validation.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

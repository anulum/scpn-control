# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural transport claim-admission benchmark
"""Publish bounded neural-transport claim-admission evidence."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from scpn_control.core.neural_transport import (
    NeuralTransportModel,
    cross_validate_neural_transport,
    neural_transport_claim_evidence,
)

REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "neural_transport_claims.json"
MD_REPORT = REPORT_DIR / "neural_transport_claims.md"


def main() -> None:
    """Run deterministic bounded neural-transport claim benchmark."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    model = NeuralTransportModel(auto_discover=False)
    validation_result = cross_validate_neural_transport(model)
    evidence = neural_transport_claim_evidence(
        validation_result,
        source="synthetic_regression_reference",
        source_id="validation/benchmark_neural_transport_claims.py::analytic_fallback_reference",
    )
    payload = {
        "validation_result": validation_result,
        "claim_evidence": asdict(evidence),
        "claim_boundary": (
            "Local analytic-fallback regression evidence only. Quantitative QuaLiKiz or "
            "documented-reference neural-transport claims require matched reference artifacts."
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
                "<!-- SCPN Control — Neural transport claim-admission benchmark report -->",
                "",
                "# Neural transport claim-admission benchmark",
                "",
                f"- Surrogate mode: {evidence.surrogate_mode}",
                f"- Local benchmark cases: {evidence.local_case_count}",
                f"- Local channel agreement: {evidence.local_channel_agreement:.8f}",
                f"- Local max absolute error: {evidence.local_max_abs_error:.8e}",
                f"- Claim admission: {evidence.claim_status}",
                f"- Quantitative claim allowed: {evidence.quantitative_claim_allowed}",
                "",
                "Claim boundary: this report checks the deterministic local",
                "critical-gradient fallback and records fail-closed admission evidence.",
                "It is not QuaLiKiz, QLKNN, or measured transport validation.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

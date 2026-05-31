# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption mitigation claim-admission benchmark
"""Publish bounded halo/runaway disruption-mitigation claim evidence."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scpn_control.control.halo_re_physics import (
    disruption_mitigation_claim_evidence,
    run_disruption_ensemble,
)

REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "disruption_mitigation_claims.json"
MD_REPORT = REPORT_DIR / "disruption_mitigation_claims.md"


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def main() -> None:
    """Run deterministic bounded disruption-mitigation claim benchmark."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    seed = 20240531
    report = run_disruption_ensemble(ensemble_runs=24, seed=seed)
    evidence = disruption_mitigation_claim_evidence(
        report,
        source="synthetic_regression_reference",
        source_id="validation/benchmark_disruption_mitigation_claims.py::bounded_halo_runaway_ensemble",
        ensemble_seed=seed,
    )
    payload = {
        "ensemble_report": asdict(report),
        "claim_evidence": asdict(evidence),
        "claim_boundary": (
            "Bounded halo/runaway ensemble evidence only. Mitigation claims require measured, "
            "external-benchmark, or documented public disruption reference artifacts."
        ),
    }
    JSON_REPORT.write_text(
        json.dumps(payload, default=_json_default, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    MD_REPORT.write_text(
        "\n".join(
            [
                "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
                "<!-- Commercial license available -->",
                "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
                "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
                "<!-- ORCID: 0009-0009-3560-0851 -->",
                "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
                "<!-- SCPN Control — Disruption mitigation claim-admission benchmark report -->",
                "",
                "# Disruption mitigation claim-admission benchmark",
                "",
                f"- Ensemble runs: {evidence.ensemble_runs}",
                f"- Ensemble seed: {evidence.ensemble_seed}",
                f"- Prevention rate: {evidence.prevention_rate:.8f}",
                f"- P95 halo peak: {evidence.p95_halo_peak_ma:.8f} MA",
                f"- P95 runaway peak: {evidence.p95_re_peak_ma:.8f} MA",
                f"- Mean TPF product: {evidence.mean_tpf_product:.8f}",
                f"- Claim admission: {evidence.claim_status}",
                f"- Mitigation claim allowed: {evidence.mitigation_claim_allowed}",
                "",
                "Claim boundary: this report records deterministic bounded ensemble",
                "evidence for the halo-current and runaway-electron mitigation model.",
                "It is not measured disruption-campaign or external-MHD validation.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Federated disruption benchmark
"""Deterministic synthetic multi-facility federated disruption benchmark."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from scpn_control.control.federated_disruption import (
    DifferentialPrivacyConfig,
    run_synthetic_multifacility_benchmark,
)


REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "federated_disruption_benchmark.json"
MD_REPORT = REPORT_DIR / "federated_disruption_benchmark.md"


def main() -> None:
    """Run the benchmark and write machine-readable plus human reports."""
    dp_config = DifferentialPrivacyConfig(
        max_update_norm=0.15,
        noise_multiplier=2.5,
        delta=1e-5,
        seed=20240531,
    )
    summary = run_synthetic_multifacility_benchmark(
        machines=("DIII-D", "JET", "KSTAR", "EAST"),
        n_rounds=4,
        local_epochs=2,
        aggregation="fedprox",
        dp_config=dp_config,
        seed=20240531,
    )
    payload = asdict(summary)
    payload["claim_boundary"] = (
        "Synthetic multi-facility benchmark only; no measured cross-facility "
        "disruption-validation claim is made by this artefact."
    )
    payload["feature_contract"] = [
        "Ip",
        "beta_N",
        "q95",
        "n_nGW",
        "li",
        "dBp_dt",
        "locked_mode_amp",
        "n1_rms",
    ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
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
                "<!-- SCPN Control — Federated disruption benchmark report -->",
                "",
                "# Federated disruption benchmark",
                "",
                "- Evidence kind: synthetic multi-facility",
                f"- Aggregation: `{summary.aggregation}`",
                f"- Facilities: {', '.join(summary.machines)}",
                f"- Rounds: {summary.n_rounds}",
                f"- Mean accuracy: {summary.mean_accuracy:.6f}",
                f"- Mean loss: {summary.mean_loss:.6f}",
                f"- Differential privacy epsilon: {summary.privacy_epsilon:.6f}",
                f"- Differential privacy delta: {summary.privacy_delta:.1e}",
                "",
                "Per-facility final accuracy:",
                "",
                *[
                    f"- `{machine}`: {accuracy:.6f}"
                    for machine, accuracy in sorted(summary.per_machine_accuracy.items())
                ],
                "",
                "Claim boundary: this report exercises the production federation,",
                "heterogeneity, and facility-update differential privacy contracts on",
                "deterministic synthetic facility distributions. It does not claim",
                "measured cross-facility validation against DIII-D, JET, KSTAR, or EAST",
                "shot databases.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

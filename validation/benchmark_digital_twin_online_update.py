# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Digital twin online update benchmark
"""Publish deterministic bounded online-update evidence for the digital twin."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scpn_control.control.digital_twin_online_update import (
    TwinObservation,
    TwinParameterPrior,
    digital_twin_update_evidence,
    synthetic_online_update_benchmark,
    validate_external_simulator_artifact,
)
from scpn_control.control.tokamak_digital_twin import run_digital_twin

REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "digital_twin_online_update.json"
MD_REPORT = REPORT_DIR / "digital_twin_online_update.md"


def main() -> None:
    """Run the synthetic online-update benchmark and write reports."""
    result = synthetic_online_update_benchmark(seed=20240531)
    target_summary = run_digital_twin(
        time_steps=24,
        seed=101,
        save_plot=False,
        verbose=False,
        n_e=1.18e20,
        Z_eff=2.2,
        actuator_tau_steps=2.0,
        actuator_rate_limit=0.08,
    )
    observation = TwinObservation(
        targets={
            "final_avg_temp": float(target_summary["final_avg_temp"]),
            "final_reward": float(target_summary["final_reward"]),
            "mean_abs_actuator_lag": float(target_summary["mean_abs_actuator_lag"]),
        },
        tolerances={
            "final_avg_temp": 0.05,
            "final_reward": 0.05,
            "mean_abs_actuator_lag": 0.02,
        },
        source="synthetic_digital_twin_reference",
    )
    priors = (
        TwinParameterPrior("n_e", 0.8e20, 1.5e20, 1.0e20),
        TwinParameterPrior("Z_eff", 1.0, 3.0, 1.5),
        TwinParameterPrior("actuator_tau_steps", 0.0, 5.0, 0.0),
        TwinParameterPrior("actuator_rate_limit", 0.03, 0.3, 0.12),
    )
    signal_units = {
        "final_avg_temp": "keV",
        "final_reward": "1",
        "mean_abs_actuator_lag": "1",
    }
    artifacts = tuple(
        validate_external_simulator_artifact(
            {
                "schema_version": "1.0",
                "simulator_code": code,
                "artifact_uri": f"synthetic://digital-twin-online-update/{code.lower()}",
                "artifact_sha256": digest,
                "case_id": f"{code}-synthetic-online-update",
                "time_base_s": [0.0, 0.012, 0.024],
                "signal_units": signal_units,
            }
        )
        for code, digest in (("TRANSP", "a" * 64), ("TSC", "b" * 64))
    )
    evidence = digital_twin_update_evidence(observation, priors, result, artifacts)
    payload = asdict(result)
    payload["evidence"] = asdict(evidence)
    payload["claim_boundary"] = (
        "Bounded synthetic online-update benchmark only. TRANSP/TSC coupling "
        "requires validated external simulator artifacts before measured replay claims."
    )

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
                "<!-- SCPN Control — Digital twin online update report -->",
                "",
                "# Digital twin online update benchmark",
                "",
                "- Evidence kind: bounded synthetic online update",
                f"- Source: {result.source}",
                f"- Evaluated points: {result.evaluated_points}",
                f"- Baseline loss: {result.baseline_loss:.8e}",
                f"- Best loss: {result.best_loss:.8e}",
                f"- Evidence result digest: `{evidence.result_sha256}`",
                f"- Evidence observation digest: `{evidence.observation_sha256}`",
                f"- Evidence prior digest: `{evidence.priors_sha256}`",
                f"- Evidence simulator digest: `{evidence.external_artifacts_sha256}`",
                "",
                "Best parameters:",
                "",
                *[f"- `{name}`: {value:.8e}" for name, value in sorted(result.best_parameters.items())],
                "",
                "Claim boundary: this benchmark exercises deterministic Bayesian",
                "model updating against a synthetic digital-twin reference. External",
                "TRANSP/TSC coupling is fail-closed behind validated simulator",
                "artifact metadata and the digital-twin reference gate.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

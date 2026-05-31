# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK online learner benchmark
"""Publish deterministic bounded GK online-learner update evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scpn_control.core.gk_online_learner import LearnerConfig, OnlineLearner

REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "gk_online_learner.json"
MD_REPORT = REPORT_DIR / "gk_online_learner.md"
DECISION_REPORT = REPORT_DIR / "gk_online_learner_decisions.json"


def _sample_target(x: np.ndarray) -> np.ndarray:
    chi_e = 0.15 + 0.04 * x[0] + 0.02 * x[3]
    chi_i = 0.12 + 0.05 * x[1] + 0.01 * x[4]
    d_e = 0.03 + 0.02 * x[2] + 0.01 * x[5]
    return np.array([chi_e, chi_i, d_e], dtype=np.float64)


def main() -> None:
    """Run deterministic bounded online-retraining campaign."""
    rng = np.random.default_rng(20240531)
    learner = OnlineLearner(
        LearnerConfig(
            buffer_size=48,
            validation_fraction=0.25,
            n_epochs=8,
            learning_rate=5.0e-4,
            max_ood_score=1.5,
        )
    )
    for _ in range(48):
        x = rng.random(10)
        learner.add_sample(x, _sample_target(x), ood_score=float(rng.uniform(0.0, 1.0)), source="synthetic_gk_spot_check")
    weights = learner.try_retrain()
    learner.save_retrain_report(DECISION_REPORT)
    decision = learner.latest_decision()
    if decision is None:
        raise RuntimeError("benchmark did not produce a retrain decision")

    payload = {
        "schema_version": "1.0",
        "evidence_kind": "bounded_synthetic_online_retraining",
        "accepted": decision.accepted,
        "generation": decision.generation,
        "val_loss": decision.val_loss,
        "train_samples": decision.train_samples,
        "validation_samples": decision.validation_samples,
        "weights_returned": weights is not None,
        "claim_boundary": (
            "Synthetic online learner evidence only. Immutable external GK campaign "
            "artefacts are required before quantitative gyrokinetic retraining claims."
        ),
    }
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
                "<!-- SCPN Control — GK online learner report -->",
                "",
                "# GK online learner benchmark",
                "",
                "- Evidence kind: bounded synthetic online retraining",
                f"- Accepted: {decision.accepted}",
                f"- Generation: {decision.generation}",
                f"- Validation loss: {decision.val_loss:.8e}",
                f"- Train samples: {decision.train_samples}",
                f"- Validation samples: {decision.validation_samples}",
                "",
                "Claim boundary: this benchmark checks deterministic sample admission,",
                "validation holdout, rollback decision persistence, and update reporting.",
                "It does not claim quantitative retraining on immutable external GK",
                "campaign artefacts.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

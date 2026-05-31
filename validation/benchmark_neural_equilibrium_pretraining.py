# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural equilibrium pretraining benchmark
"""Publish bounded synthetic neural-equilibrium pretraining evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from scpn_control.core.neural_equilibrium import pretrain_neural_equilibrium_synthetic

REPORT_DIR = Path(__file__).resolve().parent / "reports"
WEIGHTS_PATH = REPORT_DIR / "neural_equilibrium_synthetic_pretrain.npz"
JSON_REPORT = REPORT_DIR / "neural_equilibrium_pretraining.json"
MD_REPORT = REPORT_DIR / "neural_equilibrium_pretraining.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Run deterministic synthetic pretraining and write benchmark reports."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result = pretrain_neural_equilibrium_synthetic(
        n_samples=2048,
        save_path=WEIGHTS_PATH,
        grid_shape=(65, 65),
        n_components=20,
        seed=20240531,
    )
    payload = asdict(result)
    payload["weights_sha256"] = _sha256(WEIGHTS_PATH)
    payload["claim_boundary"] = (
        "Synthetic pretraining evidence only. Real EFIT/P-EFIT fine-tuning and "
        "cross-validation require passing validation/reports/neural_equilibrium_reference artifacts."
    )

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
                "<!-- SCPN Control — Neural equilibrium pretraining report -->",
                "",
                "# Neural equilibrium synthetic pretraining benchmark",
                "",
                "- Evidence kind: synthetic pretraining",
                f"- Samples: {result.n_samples}",
                f"- Grid: {result.campaign.grid_shape[0]} x {result.campaign.grid_shape[1]}",
                f"- PCA components: {result.n_components}",
                f"- Explained variance: {result.explained_variance:.8f}",
                f"- Train MSE: {result.train_mse:.8e}",
                f"- Validation MSE: {result.val_mse:.8e}",
                f"- Test MSE: {result.test_mse:.8e}",
                f"- Test max error: {result.test_max_error:.8e}",
                f"- GS residual: {result.gs_residual:.8e}",
                f"- Weights SHA-256: `{payload['weights_sha256']}`",
                "",
                "Claim boundary: this benchmark trains JAX-compatible PCA plus MLP",
                "weights on bounded synthetic Solovev-like equilibria. It is useful",
                "for pretraining and performance plumbing, but it is not real EFIT",
                "or P-EFIT validation. Real fine-tuning remains gated by persisted",
                "reference artefacts in `validation/reports/neural_equilibrium_reference/`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

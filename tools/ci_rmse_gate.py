#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Ci Rmse Gate.

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — CI RMSE Regression Gate
# Parses rmse_dashboard_ci.json and fails if key metrics regress.
# ──────────────────────────────────────────────────────────────────────
"""CI gate: fail the build if physics RMSE metrics exceed thresholds.

Thresholds are set as **regression guards** — slightly above current
best values so that future changes cannot silently degrade physics
fidelity.  They are NOT publication-quality targets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ── Thresholds ────────────────────────────────────────────────────────
# Set ~30% above current best so they catch regressions, not noise.
THRESHOLDS: dict[str, float] = {
    # tau_E absolute RMSE (s) across 20 ITPA H-mode points
    # Current best: ~0.129 s
    "confinement_itpa_tau_rmse_s": 0.20,
    # Magnetic axis RMSE (m) across SPARC GEQDSKs
    # Current best: ~1.60 m (dominated by synthetic lmode files;
    # real EFIT files achieve <0.01 m)
    "sparc_axis_rmse_m": 2.50,
    # beta_N absolute RMSE across ITER/SPARC design points
    # Current best: ~0.042 (DynamicBurnModel + profile peaking factor 1.446)
    "beta_iter_sparc_beta_n_rmse": 0.10,
    # Disruption false-positive rate — hard fail (promoted from soft warn in v3.1)
    "disruption_fpr": 0.15,
}


def main() -> int:
    artifact = Path("artifacts/rmse_dashboard_ci.json")
    if not artifact.exists():
        print(f"ERROR: {artifact} not found — run rmse_dashboard.py first.")
        return 1

    data = json.loads(artifact.read_text(encoding="utf-8"))
    failures: list[str] = []

    # ── confinement_itpa ──────────────────────────────────────────────
    itpa = data.get("confinement_itpa", {})
    if itpa:
        tau_rmse = itpa.get("tau_rmse_s", 0.0)
        thresh = THRESHOLDS["confinement_itpa_tau_rmse_s"]
        if tau_rmse > thresh:
            failures.append(f"confinement_itpa: tau_rmse {tau_rmse:.4f} s > {thresh:.4f} s")
        else:
            print(f"PASS  confinement_itpa: tau_rmse {tau_rmse:.4f} s <= {thresh:.4f} s")

    # ── sparc_axis ────────────────────────────────────────────────────
    sparc = data.get("sparc_axis", {})
    if sparc:
        axis_rmse = sparc.get("axis_rmse_m", 0.0)
        thresh = THRESHOLDS["sparc_axis_rmse_m"]
        if axis_rmse > thresh:
            failures.append(f"sparc_axis: RMSE {axis_rmse:.4f} m > {thresh:.4f} m")
        else:
            print(f"PASS  sparc_axis: RMSE {axis_rmse:.4f} m <= {thresh:.4f} m")

    # ── beta_iter_sparc ───────────────────────────────────────────────
    beta = data.get("beta_iter_sparc", {})
    if beta:
        beta_rmse = beta.get("beta_n_rmse", 0.0)
        thresh = THRESHOLDS["beta_iter_sparc_beta_n_rmse"]
        if beta_rmse > thresh:
            failures.append(f"beta_iter_sparc: beta_N RMSE {beta_rmse:.4f} > {thresh:.4f}")
        else:
            print(f"PASS  beta_iter_sparc: beta_N RMSE {beta_rmse:.4f} <= {thresh:.4f}")

    # ── disruption FPR (hard gate since v3.1) ──────────────────────────
    reference_artifact = Path("artifacts/reference_evidence_validation.json")
    if reference_artifact.exists():
        reference_data = json.loads(reference_artifact.read_text(encoding="utf-8"))
        dis = reference_data.get("lanes", {}).get("disruption_synthetic", {})
        fpr = dis.get("false_positive_rate", 0.0)
        fpr_thresh = THRESHOLDS["disruption_fpr"]
        if fpr > fpr_thresh:
            failures.append(f"disruption FPR: {fpr:.2f} > {fpr_thresh:.2f} (hard gate since v3.1 — FPR must be <= 15%)")
        else:
            print(f"PASS  disruption FPR: {fpr:.2f} <= {fpr_thresh:.2f}")

    if failures:
        print("\nFAILED RMSE regression gate:")
        for f in failures:
            print(f"  FAIL  {f}")
        return 1

    print("\nAll RMSE regression gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

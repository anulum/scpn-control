# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Uncertainty quantification claim-admission benchmark

"""Publish a fixed Monte Carlo model declaration through recorded output custody."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from scpn_control.benchmark_records import require_recorded_campaign
from scpn_control.core.uncertainty import PlasmaScenario, quantify_full_chain, uq_claim_evidence

REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "uq_claims.json"
MARKDOWN_REPORT = REPORT_DIR / "uq_claims.md"


def build_reference_scenario() -> PlasmaScenario:
    """Return a fresh fixed ITER-like scaling-law input, without running UQ.

    Returns
    -------
    PlasmaScenario
        Mutable caller-owned scenario: I_p=15 MA, B_t=5.3 T, P_heat=50 MW,
        n_e=10.1 in 1e19 m^-3, R=6.2 m, A=3.1, kappa=1.7 and M=2.5 AMU.
        Default D/T fuel fractions are each 0.5 and dilution fraction is 1.

    Notes
    -----
    This is a repository preset, with no measured shot, input-file loading,
    random sampling, cache, shared scenario state or reference verification.
    """
    return PlasmaScenario(I_p=15.0, B_t=5.3, P_heat=50.0, n_e=10.1, R=6.2, A=3.1, kappa=1.7, M=2.5)


def main() -> None:
    """Write bounded UQ evidence from 256 fixed-seed model samples.

    Returns
    -------
    None
        Write uq_claims.json then uq_claims.md beside this module under reports.
        Both use UTF-8 with final newlines; JSON uses the defining dataclass.

    Raises
    ------
    RuntimeError
        Persistent destinations lack a recorded-campaign identifier.
    ValueError
        Campaign syntax or defining scenario, sampling or evidence validation
        fails before persistence.
    OSError
        Directory creation or a sequential write fails. Earlier output may
        remain; writes do not form an atomic transaction.

    Notes
    -----
    No CLI parameters are parsed. A fresh reference scenario and local NumPy
    Generator seed 31 feed quantify_full_chain; global RNG state is untouched.
    IPB98 coefficients and transport/pedestal/boundary proxy uncertainties
    propagate to tau_E in seconds, fusion power in MW and dimensionless Q.
    This path runs no equilibrium or transport PDE solver. Its provenance
    strings describe repository models, not a calibrated uncertainty witness.
    No reference values or sigma reference are supplied, so
    calibrated_uq_claim_allowed remains False even when finite/order checks pass.
    Filenames are shared, without locks. The campaign guard checks identifier
    presence/syntax; the recorded wrapper separately preserves output custody.
    Numerical propagation and sensitivity semantics belong to defining APIs.
    """
    require_recorded_campaign(JSON_REPORT, MARKDOWN_REPORT, repository_root=REPORT_DIR.parents[1])
    scenario = build_reference_scenario()
    seed = 31
    result = quantify_full_chain(scenario, n_samples=256, seed=seed)
    evidence = uq_claim_evidence(
        scenario,
        result,
        source="synthetic_regression_reference",
        source_id="uq-bounded-regression-v1",
        scenario_source="repository ITER-like scenario fixture",
        prior_source="repository IPB98 covariance registry",
        propagation_chain="IPB98 -> Bosch-Hale fusion power -> Q",
        sensitivity_source="finite-difference density and temperature sensitivities",
        seed=seed,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MARKDOWN_REPORT.write_text(
        "\n".join(
            [
                "# UQ Claim-Admission Benchmark",
                "",
                "This report records bounded synthetic-regression evidence for",
                "full-chain uncertainty quantification claim admission. It captures",
                "scenario provenance, prior provenance, propagation chain, seed,",
                "sample count, ordered percentile checks, finite outputs, D-T fuel",
                "dilution, and density/temperature sensitivity provenance.",
                "",
                "This fixture samples scaling-law and transport proxies; no equilibrium or transport PDE solver is run.",
                "",
                f"- Claim status: `{evidence.claim_status}`",
                f"- Calibrated UQ claim allowed: `{evidence.calibrated_uq_claim_allowed}`",
                f"- Seed: `{evidence.seed}`",
                f"- Samples: `{evidence.n_samples}`",
                f"- tau_E: `{evidence.tau_E_s:.12g}` s",
                f"- P_fusion: `{evidence.P_fusion_MW:.12g}` MW",
                f"- Q: `{evidence.Q:.12g}`",
                f"- Finite outputs: `{evidence.finite_outputs}`",
                "",
                "Synthetic regression evidence is not calibrated facility predictive uncertainty.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

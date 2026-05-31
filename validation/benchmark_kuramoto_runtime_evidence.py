# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Kuramoto Runtime Evidence Producer
"""Produce bounded Kuramoto runtime parity and timestep-refinement evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scpn_control.phase.kuramoto import kuramoto_runtime_evidence, save_kuramoto_runtime_evidence


def _deterministic_case(oscillators: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if oscillators <= 0:
        raise ValueError("oscillators must be positive")
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, oscillators).astype(np.float64)
    omega = rng.normal(0.0, 0.3, oscillators).astype(np.float64)
    return theta, omega


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True, help="Destination JSON evidence path")
    parser.add_argument("--target-id", default="local-python-runtime")
    parser.add_argument("--oscillators", type=int, default=4096)
    parser.add_argument("--deployment-target-oscillators", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--K", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--zeta", type=float, default=0.5)
    parser.add_argument("--psi-driver", type=float, default=0.0)
    parser.add_argument("--psi-mode", choices=("external", "mean_field"), default="external")
    parser.add_argument("--parity-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--timestep-refinement-tolerance", type=float, default=5.0e-3)
    parser.add_argument("--deployment-claim", action="store_true")
    args = parser.parse_args()

    theta, omega = _deterministic_case(args.oscillators, args.seed)
    evidence = kuramoto_runtime_evidence(
        theta,
        omega,
        dt=args.dt,
        K=args.K,
        alpha=args.alpha,
        zeta=args.zeta,
        psi_driver=args.psi_driver,
        psi_mode=args.psi_mode,
        target_id=args.target_id,
        deployment_target_oscillators=args.deployment_target_oscillators,
        parity_tolerance=args.parity_tolerance,
        timestep_refinement_tolerance=args.timestep_refinement_tolerance,
        deployment_claim_allowed=args.deployment_claim,
    )
    save_kuramoto_runtime_evidence(evidence, Path(args.output_json))


if __name__ == "__main__":
    main()

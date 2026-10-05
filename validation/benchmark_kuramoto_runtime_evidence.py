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
import numpy.typing as npt

from scpn_control.benchmark_records import require_recorded_campaign
from scpn_control.phase.kuramoto import kuramoto_runtime_evidence, save_kuramoto_runtime_evidence


def _deterministic_case(oscillators: int, seed: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Draw reproducible synthetic phase and frequency vectors.

    Parameters
    ----------
    oscillators : int
        Positive vector length. No physical oscillator states are ingested.
    seed : int
        Entropy passed to a fresh NumPy default_rng; global RNG state is unchanged.

    Returns
    -------
    theta, omega : tuple of numpy.ndarray
        Independent float64 arrays of shape (oscillators,). Phases are uniform
        in [-pi, pi) radians; frequencies are normal with mean zero and
        standard deviation 0.3 radians per second.

    Raises
    ------
    ValueError
        If the count is non-positive, or NumPy rejects seed/shape values.
    TypeError
        If NumPy cannot interpret the supplied seed or count.
    """
    if oscillators <= 0:
        raise ValueError("oscillators must be positive")
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, oscillators).astype(np.float64)
    omega = rng.normal(0.0, 0.3, oscillators).astype(np.float64)
    return theta, omega


def main() -> None:
    """Produce one wrapped-step parity and timestep-refinement JSON report.

    Arguments are read from sys.argv. --output-json is required; default
    count and deployment target are 4096, with seed 20260531.
    --dt is seconds; --K, --zeta and sampled omega are rates; --alpha and
    --psi-driver are radians. --psi-mode selects external/mean_field.
    Parity tolerances bound the core's phase/order-parameter error tests;
    --timestep-refinement-tolerance bounds the phase error in radians.

    Returns
    -------
    None
        The public core producer compares one Python Euler step with two half
        steps, and checks the optional actually imported Rust backend.
        Sorted UTF-8 JSON with a final newline is written, creating parents.

    Raises
    ------
    SystemExit
        Argparse help exits zero; malformed CLI arguments exit two.
    RuntimeError
        A persistent evidence destination requires recorded-runner custody.
    ValueError
        Invalid counts/seed/core inputs or a requested deployment claim fail
        before writing. Deployment requires passing Rust parity, refinement
        and target count coverage. No backend is installed by this command.
    OSError
        Parent creation or writing fails; an existing output can be overwritten.

    Notes
    -----
    The default report is bounded even if native parity is available.
    --deployment-claim requests the core's model-local admission, not hardware,
    facility or safety authority. Outside-repository outputs need no campaign;
    persistent evidence roots include validation/reports, artifacts, benchmarks
    and gpu_results. Numerical kernels, devices and time are not authenticated.
    """
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
    output_path = Path(args.output_json)
    require_recorded_campaign(output_path, repository_root=Path(__file__).resolve().parents[1])

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
    save_kuramoto_runtime_evidence(evidence, output_path)


if __name__ == "__main__":
    main()

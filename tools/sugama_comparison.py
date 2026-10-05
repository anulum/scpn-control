#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Sugama vs Krook Collision Comparison.

"""Report three fixed CBC cases using Krook and Sugama-like collisions.

Adiabatic cases request JAX; the implicit kinetic-electron case uses NumPy.
These switch both physics and backend, so they are not a paired performance
benchmark. Importing this module runs no campaign.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TypedDict

import numpy as np


class CollisionComparisonResult(TypedDict):
    """Carry requested switches and the producer's saved case diagnostics.

    collision_model records the supplied name; only the exact string sugama
    selects that model in the current backends, while other names take Krook.
    chi_i_gB is normalised mean ion flux or None, not a calibrated physical
    diffusivity. Histories and converged are copied without admission checks.
    """

    label: str
    collision_model: str
    kinetic_electrons: bool
    implicit: bool
    chi_i_gB: float | None
    elapsed_s: float
    converged: bool
    phi_rms: list[float]
    Q_i: list[float]
    time: list[float]


def run(label: str, coll: str, ke: bool, implicit: bool, mr: float) -> CollisionComparisonResult:
    """Run one original 5,000-step fixed-grid collision comparison case.

    Parameters
    ----------
    label
        Display and returned name.
    coll
        Collision-model string passed unchanged to the solver configuration.
        sugama selects the current simplified model; other names take Krook.
    ke, implicit
        Kinetic-electron switch and backend choice. implicit=True uses NumPy;
        otherwise JAX is required without a fallback.
    mr
        Dimensionless electron/ion mass ratio.

    Returns
    -------
    CollisionComparisonResult
        Requested switches, scalar flux normalisation, run() wall time,
        producer flag and saved phi/ion-flux/normalised-time histories. Grid is
        128 x 16 x 32 x 16 x 8; dt 0.05 is CFL-adaptive and samples save every 100 steps.

    Raises
    ------
    RuntimeError
        The requested JAX backend is unavailable.
    Exception
        Import, allocation and solver errors propagate. No process timeout or
        scientific input/reference admission is supplied by this reporter.

    Notes
    -----
    Timing excludes solver construction and includes work first triggered in
    run(). Saved flux finiteness and converged do not prove finite final state
    or nonlinear saturation. Nonfinite scalar chi becomes None; histories do
    not receive the same filtering.
    """
    from scpn_control.core.gk_nonlinear import NonlinearGKConfig

    print(f"\n=== {label} ===", flush=True)
    cfg = NonlinearGKConfig(
        n_kx=128,
        n_ky=16,
        n_theta=32,
        n_vpar=16,
        n_mu=8,
        dt=0.05,
        n_steps=5000,
        save_interval=100,
        R_L_Ti=6.9,
        R_L_Te=6.9,
        R_L_ne=2.2,
        q=1.4,
        s_hat=0.78,
        R0=2.78,
        a=1.0,
        B0=2.0,
        cfl_adapt=True,
        cfl_factor=0.5,
        nonlinear=True,
        collisions=True,
        nu_collision=0.01,
        collision_model=coll,
        hyper_coeff=0.2,
        kinetic_electrons=ke,
        implicit_electrons=implicit,
        mass_ratio_me_mi=mr,
    )
    solver: NonlinearGKSolver | JaxNonlinearGKSolver
    if implicit:
        from scpn_control.core.gk_nonlinear import NonlinearGKSolver

        solver = NonlinearGKSolver(cfg)
    else:
        from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver

        solver = JaxNonlinearGKSolver(cfg)

    t0 = time.perf_counter()
    r = solver.run()
    elapsed = time.perf_counter() - t0
    chi_gB = r.chi_i / max(cfg.R_L_Ti, 0.01)
    print(f"  elapsed={elapsed:.0f}s, chi_gB={chi_gB:.4f}", flush=True)
    if len(r.phi_rms_t) > 0:
        print(f"  phi: [{r.phi_rms_t[0]:.4e}...{r.phi_rms_t[-1]:.4e}]", flush=True)
        print(f"  time: [{r.time[0]:.2f}, {r.time[-1]:.2f}]", flush=True)
    return {
        "label": label,
        "collision_model": coll,
        "kinetic_electrons": ke,
        "implicit": implicit,
        "chi_i_gB": float(chi_gB) if np.isfinite(chi_gB) else None,
        "elapsed_s": elapsed,
        "converged": bool(r.converged),
        "phi_rms": r.phi_rms_t.tolist(),
        "Q_i": r.Q_i_t.tolist(),
        "time": r.time.tolist(),
    }


def main() -> None:
    """Run the three original unequal-physics cases and overwrite cwd JSON.

    Krook/adiabatic and Sugama/adiabatic request JAX; Sugama/kinetic uses
    implicit NumPy. All use mass 1/400 and 5,000 steps. No CLI options are parsed.
    After all calls return, gpu_results/sugama_comparison.json is overwritten
    with default JSON nonfinite-number behaviour. The subsequent scalar console
    summary can raise on None after writing the report. Native and filesystem
    errors propagate. This entrypoint may run a long NumPy campaign.
    """
    try:
        import jax

        print(f"JAX {jax.__version__}, {jax.devices()}", flush=True)
    except ImportError:
        pass

    results = {}
    results["krook_adiabatic"] = run("Krook + adiabatic", "krook", ke=False, implicit=False, mr=1 / 400)
    results["sugama_adiabatic"] = run("Sugama + adiabatic", "sugama", ke=False, implicit=False, mr=1 / 400)
    results["sugama_kinetic_e"] = run("Sugama + kinetic_e (implicit)", "sugama", ke=True, implicit=True, mr=1 / 400)

    out = Path("gpu_results")
    out.mkdir(exist_ok=True)
    outpath = out / "sugama_comparison.json"
    outpath.write_text(json.dumps({"results": results}, indent=2, default=str))
    print(f"\nSaved to {outpath}", flush=True)

    print("\n=== Summary ===", flush=True)
    for k, v in results.items():
        print(f"  {k}: chi_gB={v['chi_i_gB']:.3f}", flush=True)


if __name__ == "__main__":
    main()

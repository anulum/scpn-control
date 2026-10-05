#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Kinetic Electron Comparison.

"""Compare two fixed JAX CBC runs with adiabatic and kinetic electrons.

The requested backend uses its available JAX device, without certifying GPU
execution or silently selecting NumPy. Reports carry the solver's saved
diagnostics and flags; they supply no physical or production admission.
Importing this module runs no campaign.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TypedDict

import numpy as np


class KineticElectronResult(TypedDict):
    """Describe one saved JAX case without authenticating solver convergence.

    Scalar chi values become None when nonfinite; histories and late_growth
    are not finite-filtered. chi_i_raw is the mean saved ion-flux quantity,
    chi_i_gB divides it by R_L_Ti. Time lists use normalized solver time and
    elapsed_s measures run() wall time, excluding construction.
    """

    label: str
    kinetic_electrons: bool
    chi_i_raw: float | None
    chi_i_gB: float | None
    late_growth: float
    elapsed_s: float
    converged: bool
    phi_rms: list[float]
    zonal_rms: list[float]
    Q_i: list[float]
    time: list[float]


def run_case(label: str, kinetic_e: bool, n_steps: int = 5000) -> KineticElectronResult:
    """Run a fresh fixed-grid CBC case with a selected electron response.

    Parameters
    ----------
    label
        Display and returned name; no configuration is inferred from it.
    kinetic_e
        Kinetic-electron switch; mass ratio remains1/400 in both cases.
    n_steps
        Requested iteration count, default 5,000. Histories save every 100 steps;
        nominal dt 0.05 can decrease through CFL adaptation.

    Returns
    -------
    KineticElectronResult
        Scalar summaries and phi/zonal/ion-flux/time histories from the
        128 x 16 x 32 x 16 x 8 JAX case. Late growth is an endpoint fractional change
        over the last quarter of saved samples, not an exponential fit.

    Raises
    ------
    RuntimeError
        JAX is unavailable under the backend's default no-fallback contract.
    IndexError
        No samples were saved; this reporter indexes the first and last sample.
    Exception
        Import, allocation and backend errors propagate. No process timeout,
        argument admission or physical calibration is supplied here.

    Notes
    -----
    run() wall time includes JAX work triggered there. The producer's converged
    flag concerns saved fluxes and is not proof of saturation or finite final
    state. The ion-flux normalization is not an independent m^2/s measurement.
    """
    from scpn_control.core.gk_nonlinear import NonlinearGKConfig
    from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver

    print(f"\n=== {label}: kinetic_e={kinetic_e}, {n_steps} steps ===", flush=True)
    cfg = NonlinearGKConfig(
        n_kx=128,
        n_ky=16,
        n_theta=32,
        n_vpar=16,
        n_mu=8,
        dt=0.05,
        n_steps=n_steps,
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
        hyper_coeff=0.2,
        kinetic_electrons=kinetic_e,
        mass_ratio_me_mi=1.0 / 400.0,
    )
    solver = JaxNonlinearGKSolver(cfg)
    t0 = time.perf_counter()
    r = solver.run()
    elapsed = time.perf_counter() - t0
    chi_gB = r.chi_i / max(cfg.R_L_Ti, 0.01)
    print(f"  elapsed={elapsed:.0f}s", flush=True)
    print(f"  chi_i_gB={chi_gB:.4f}", flush=True)
    print(f"  phi: [{r.phi_rms_t[0]:.4e}...{r.phi_rms_t[-1]:.4e}]", flush=True)
    print(f"  time: [{r.time[0]:.2f}, {r.time[-1]:.2f}]", flush=True)

    n4 = 3 * len(r.phi_rms_t) // 4
    phi_lq = r.phi_rms_t[n4:]
    late_growth = 0.0
    if len(phi_lq) > 2 and phi_lq[0] > 0:
        late_growth = (phi_lq[-1] - phi_lq[0]) / (phi_lq[0] * max(r.time[-1] - r.time[n4], 0.01))
    print(f"  late_growth={late_growth:.4f}", flush=True)

    for i in range(0, len(r.phi_rms_t), max(1, len(r.phi_rms_t) // 10)):
        pr = r.phi_rms_t[i]
        zr = r.zonal_rms_t[i]
        print(
            f"    t={r.time[i]:.1f} phi={pr:.3e} z/p={zr / max(pr, 1e-30):.2f} Q={r.Q_i_t[i]:.3e}",
            flush=True,
        )

    return {
        "label": label,
        "kinetic_electrons": kinetic_e,
        "chi_i_raw": float(r.chi_i) if np.isfinite(r.chi_i) else None,
        "chi_i_gB": float(chi_gB) if np.isfinite(chi_gB) else None,
        "late_growth": float(late_growth),
        "elapsed_s": elapsed,
        "converged": bool(r.converged),
        "phi_rms": r.phi_rms_t.tolist(),
        "zonal_rms": r.zonal_rms_t.tolist(),
        "Q_i": r.Q_i_t.tolist(),
        "time": r.time.tolist(),
    }


def main() -> None:
    """Run the original two 5,000-step cases and overwrite their cwd JSON report.

    No CLI options are parsed. JAX device printing is best effort; both cases
    still require JAX. Results overwrite gpu_results/kinetic_electron_comparison.json
    after both calls return, using json.dumps with its default NaN/Infinity
    behavior. The final console summary requires finite scalar chi and a saved
    phi sample; None or empty histories can raise after the file was written.
    Native and filesystem errors propagate; this is a long campaign entrypoint.
    """
    try:
        import jax

        print(f"JAX {jax.__version__}, {jax.devices()}", flush=True)
    except ImportError:
        pass

    results = {}
    results["adiabatic"] = run_case("CBC adiabatic", kinetic_e=False, n_steps=5000)
    results["kinetic"] = run_case("CBC kinetic_e", kinetic_e=True, n_steps=5000)

    out = Path("gpu_results")
    out.mkdir(exist_ok=True)
    outpath = out / "kinetic_electron_comparison.json"
    outpath.write_text(json.dumps({"results": results}, indent=2, default=str))
    print(f"\nSaved to {outpath}", flush=True)

    print("\n=== Comparison ===", flush=True)
    for k, v in results.items():
        print(
            f"  {k}: chi_gB={v['chi_i_gB']:.3f}, late_growth={v['late_growth']:.4f}, phi_final={v['phi_rms'][-1]:.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()

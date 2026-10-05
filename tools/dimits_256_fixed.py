#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Dimits Shift at n_kx=256 with CFL Fix.

"""Manual 256-kx drive comparison with hyper coefficient 0.02; not convergence admission."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def run(label: str, rlt: float) -> dict[str, Any]:
    """Run one fixed 256-kx JAX experiment and return saved raw diagnostics.

    Parameters
    ----------
    label : str
        Copied console/report label, without physical-case authentication.
    rlt : float
        R_L_Ti and R_L_Te drive values; this runner adds no domain validation.
        The display ratio is result.chi_i / max(rlt, 0.01).

    Returns
    -------
    dict
        Label, R_L_Ti, chi_i_gB (None if nonfinite), wall-clock elapsed_s,
        the solver converged flag and phi_rms/Q_i/time lists. Empty histories
        are returned unchanged. Histories retain native nonfinite values.

    Raises
    ------
    RuntimeError
        JAX is unavailable under the backend's unchanged no-fallback policy.
    TypeError, ValueError
        The defining config, grids or backend refuse the supplied drive.

    Notes
    -----
    The fixed grid is (256, 16, 32, 16, 8), with 10000 steps, save interval
    200, dt 0.05, CFL adaptation and hyper coefficient 0.02. Each call creates
    a fresh solver with default initial-state seed 42. JAX chooses its configured
    backend; device/runtime resources are shared. Timing brackets run(), while
    saved time is solver code time. Printed late_growth is a fractional endpoint
    difference per code time, not logarithmic growth; fewer than three
    last-quarter samples, or a nonpositive first sample, suppress that
    diagnostic. The converged flag only
    tests for more than one finite flux sample. The kx print accesses the
    current solver's private NumPy grid and is coupled to that implementation.
    This function writes no file and admits no physical Dimits shift,
    convergence, external-reference validation or facility-control evidence.
    """
    from scpn_control.core.gk_nonlinear import NonlinearGKConfig
    from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver

    print(f"\n=== {label} ===", flush=True)
    cfg = NonlinearGKConfig(
        n_kx=256,
        n_ky=16,
        n_theta=32,
        n_vpar=16,
        n_mu=8,
        dt=0.05,
        n_steps=10000,
        save_interval=200,
        R_L_Ti=rlt,
        R_L_Te=rlt,
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
        hyper_coeff=0.02,  # reduced for faster dt at n_kx=256
    )
    solver = JaxNonlinearGKSolver(cfg)
    kx = solver._np_solver.kx
    print(f"  kx_max={np.max(np.abs(kx)):.1f}, hyper_max={0.02 * np.max(np.abs(kx)) ** 4:.0f}", flush=True)
    t0 = time.perf_counter()
    r = solver.run()
    elapsed = time.perf_counter() - t0
    chi_gB = r.chi_i / max(rlt, 0.01)
    print(f"  {elapsed:.0f}s, chi_gB={chi_gB:.4f}", flush=True)
    if len(r.phi_rms_t) > 0:
        print(f"  phi: [{r.phi_rms_t[0]:.4e}...{r.phi_rms_t[-1]:.4e}]", flush=True)
        print(f"  time: [{r.time[0]:.2f}, {r.time[-1]:.2f}]", flush=True)
        n4 = 3 * len(r.phi_rms_t) // 4
        phi_lq = r.phi_rms_t[n4:]
        if len(phi_lq) > 2 and phi_lq[0] > 0:
            lg = (phi_lq[-1] - phi_lq[0]) / (phi_lq[0] * max(r.time[-1] - r.time[n4], 0.01))
            print(f"  late_growth={lg:.4f}", flush=True)
        for i in range(0, len(r.phi_rms_t), max(1, len(r.phi_rms_t) // 8)):
            print(f"    t={r.time[i]:.2f} phi={r.phi_rms_t[i]:.3e} Q={r.Q_i_t[i]:.3e}", flush=True)
    return {
        "label": label,
        "R_L_Ti": rlt,
        "chi_i_gB": float(chi_gB) if np.isfinite(chi_gB) else None,
        "elapsed_s": elapsed,
        "converged": bool(r.converged),
        "phi_rms": r.phi_rms_t.tolist(),
        "Q_i": r.Q_i_t.tolist(),
        "time": r.time.tolist(),
    }


def main() -> None:
    """Run both fixed drive cases and overwrite a caller-relative raw report.

    Returns
    -------
    None
        Print backend, case diagnostics and summaries; run() is called with
        drives 3.0 and 6.9 before gpu_results/dimits_256_fixed.json is written.

    Raises
    ------
    RuntimeError, TypeError, ValueError
        The defining solver/config/backend refuses a case before report output.
    OSError
        Creating the output directory or writing its report fails.
    IndexError
        A case returns an empty history and the final comparison indexes it;
        the raw report has already been written when this summary fails.

    Notes
    -----
    There are no CLI parameters. Both experiments use the fixed 10000-step
    grid and hyper coefficient declared by run(). Output uses the platform
    text codec and legacy json.dumps(indent=2, default=str), retaining its
    nonfinite-float convention. Existing output may be replaced without atomic
    replacement, locking, campaign custody or an authenticated source digest.
    Successful completion is not physical or asymptotic convergence admission.
    """
    try:
        import jax

        print(f"JAX {jax.__version__}, {jax.devices()}", flush=True)
    except ImportError:
        pass

    results = {}
    results["rlt_3"] = run("R/L_Ti=3.0 n_kx=256", 3.0)
    results["rlt_69"] = run("R/L_Ti=6.9 n_kx=256", 6.9)

    out = Path("gpu_results")
    out.mkdir(exist_ok=True)
    outpath = out / "dimits_256_fixed.json"
    outpath.write_text(json.dumps({"results": results}, indent=2, default=str))
    print(f"\nSaved to {outpath}", flush=True)

    print("\n=== Comparison ===", flush=True)
    for k, v in results.items():
        phi = v.get("phi_rms", [0])
        print(f"  {k}: chi_gB={v['chi_i_gB']}, phi_final={phi[-1]:.3e}", flush=True)


if __name__ == "__main__":
    main()

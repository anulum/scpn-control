#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Dimits Shift Long Run.

"""Manual 20K-step drive comparison; diagnostics do not establish a physical Dimits shift."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def main() -> None:
    """Run two fixed 20K-step JAX experiments and overwrite a raw JSON report.

    Returns
    -------
    None
        Print backend and saved diagnostics, then write both drive cases to
        caller-relative gpu_results/dimits_long_3_vs_69.json. Each case uses
        (n_kx, n_ky, n_theta, n_vpar, n_mu) = (128, 16, 32, 16, 8), 20000
        steps, save interval 200, dt 0.05, CFL adaptation and hyper coefficient
        0.2. R_L_Ti/R_L_Te are 3.0 or 6.9; other declared CBC values are fixed.

    Raises
    ------
    RuntimeError, TypeError, ValueError
        The defining solver/config/backend refuses an experiment before output.
    IndexError
        A result has no saved history; the endpoint print indexes that history
        before completing remaining cases or writing a final report.
    OSError
        Creating gpu_results or writing dimits_long_3_vs_69.json fails.

    Notes
    -----
    No CLI parameters are parsed. Each case creates a fresh solver with default
    initial-state seed 42; JAX chooses its configured CPU/GPU backend and no
    NumPy fallback is enabled. Wall-clock elapsed_s brackets run(); saved time
    is solver code time. chi_i_gB is raw chi_i / max(R_L_Ti, 0.01), with None
    for a nonfinite scalar. Histories retain native nonfinite values. Printed
    late_growth is a fractional endpoint difference per code time rather than
    logarithmic growth. The converged flag means more than one finite flux
    sample, not physical or asymptotic convergence. The legacy serialiser uses
    indent=2/default=str and its nonfinite-float convention, via the platform
    text codec. Existing output may be replaced without atomic replacement,
    locks, campaign custody or authenticated source digests. The raw report
    provides no validated Dimits shift, external reference or facility evidence.
    """
    from scpn_control.core.gk_nonlinear import NonlinearGKConfig
    from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver

    try:
        import jax

        print(f"JAX {jax.__version__}, {jax.devices()}", flush=True)
    except ImportError:
        pass

    results = {}
    for rlt in [3.0, 6.9]:
        print(f"\n=== R/L_Ti = {rlt}, 20000 steps ===", flush=True)
        cfg = NonlinearGKConfig(
            n_kx=128,
            n_ky=16,
            n_theta=32,
            n_vpar=16,
            n_mu=8,
            dt=0.05,
            n_steps=20000,
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
            hyper_coeff=0.2,
        )
        solver = JaxNonlinearGKSolver(cfg)
        t0 = time.perf_counter()
        r = solver.run()
        elapsed = time.perf_counter() - t0
        chi_gB = r.chi_i / max(rlt, 0.01)
        print(f"  elapsed={elapsed:.0f}s, chi_i_gB={chi_gB:.4f}", flush=True)
        print(f"  phi: [{r.phi_rms_t[0]:.4e}...{r.phi_rms_t[-1]:.4e}]", flush=True)
        print(f"  time: [{r.time[0]:.2f}, {r.time[-1]:.2f}]", flush=True)

        n4 = 3 * len(r.phi_rms_t) // 4
        phi_lq = r.phi_rms_t[n4:]
        if len(phi_lq) > 2 and phi_lq[0] > 0:
            g = (phi_lq[-1] - phi_lq[0]) / (phi_lq[0] * max(r.time[-1] - r.time[n4], 0.01))
            print(f"  late_growth={g:.4f}", flush=True)

        for i in range(0, len(r.phi_rms_t), max(1, len(r.phi_rms_t) // 12)):
            print(
                f"    t={r.time[i]:.1f} phi={r.phi_rms_t[i]:.3e} Q={r.Q_i_t[i]:.3e}",
                flush=True,
            )

        results[str(rlt)] = {
            "R_L_Ti": rlt,
            "chi_i_raw": float(r.chi_i) if np.isfinite(r.chi_i) else None,
            "chi_i_gB": float(chi_gB) if np.isfinite(chi_gB) else None,
            "elapsed_s": elapsed,
            "converged": bool(r.converged),
            "phi_rms": r.phi_rms_t.tolist(),
            "Q_i": r.Q_i_t.tolist(),
            "time": r.time.tolist(),
        }

    out = Path("gpu_results")
    out.mkdir(exist_ok=True)
    outpath = out / "dimits_long_3_vs_69.json"
    outpath.write_text(json.dumps({"results": results}, indent=2, default=str))
    print(f"\nSaved to {outpath}", flush=True)


if __name__ == "__main__":
    main()

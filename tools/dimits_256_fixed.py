#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Dimits Shift at n_kx=256 with CFL Fix
"""n_kx=256 Dimits: R/L_Ti=3 vs 6.9, hyper=0.02 for faster time advancement."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def run(label: str, rlt: float) -> dict:
    from scpn_control.core.gk_nonlinear import NonlinearGKConfig
    from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver

    print(f"\n=== {label} ===", flush=True)
    cfg = NonlinearGKConfig(
        n_kx=256, n_ky=16, n_theta=32, n_vpar=16, n_mu=8,
        dt=0.05, n_steps=10000, save_interval=200,
        R_L_Ti=rlt, R_L_Te=rlt, R_L_ne=2.2,
        q=1.4, s_hat=0.78, R0=2.78, a=1.0, B0=2.0,
        cfl_adapt=True, cfl_factor=0.5,
        nonlinear=True, collisions=True, nu_collision=0.01,
        hyper_coeff=0.02,  # reduced for faster dt at n_kx=256
    )
    solver = JaxNonlinearGKSolver(cfg)
    kx = solver._np_solver.kx
    print(f"  kx_max={np.max(np.abs(kx)):.1f}, hyper_max={0.02*np.max(np.abs(kx))**4:.0f}", flush=True)
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
        "label": label, "R_L_Ti": rlt,
        "chi_i_gB": float(chi_gB) if np.isfinite(chi_gB) else None,
        "elapsed_s": elapsed, "converged": bool(r.converged),
        "phi_rms": r.phi_rms_t.tolist(), "Q_i": r.Q_i_t.tolist(), "time": r.time.tolist(),
    }


def main() -> None:
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

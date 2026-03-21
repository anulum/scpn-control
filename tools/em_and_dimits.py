#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — EM GPU Test + Long Dimits Shift
"""Two tests: (1) EM vs ES at CBC, (2) Dimits R/L_Ti=3 vs 6.9 at 20K steps with n_kx=256."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def run_jax(label: str, **overrides: object) -> dict:
    from scpn_control.core.gk_nonlinear import NonlinearGKConfig
    from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver

    print(f"\n=== {label} ===", flush=True)
    defaults = dict(
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
        hyper_coeff=0.2,
    )
    defaults.update(overrides)
    cfg = NonlinearGKConfig(**defaults)
    solver = JaxNonlinearGKSolver(cfg)
    t0 = time.perf_counter()
    r = solver.run()
    elapsed = time.perf_counter() - t0
    rlt = float(defaults.get("R_L_Ti", 6.9))
    chi_gB = r.chi_i / max(rlt, 0.01)
    print(f"  {elapsed:.0f}s, chi_gB={chi_gB:.4f}, phi=[{r.phi_rms_t[-1]:.3e}], t={r.time[-1]:.1f}", flush=True)
    n4 = 3 * len(r.phi_rms_t) // 4
    phi_lq = r.phi_rms_t[n4:]
    lg = 0.0
    if len(phi_lq) > 2 and phi_lq[0] > 0:
        lg = (phi_lq[-1] - phi_lq[0]) / (phi_lq[0] * max(r.time[-1] - r.time[n4], 0.01))
    print(f"  late_growth={lg:.4f}", flush=True)
    return {
        "label": label,
        "chi_i_gB": float(chi_gB) if np.isfinite(chi_gB) else None,
        "late_growth": float(lg),
        "elapsed_s": elapsed,
        "phi_final": float(r.phi_rms_t[-1]),
        "time_final": float(r.time[-1]),
        "converged": bool(r.converged),
    }


def main() -> None:
    try:
        import jax

        print(f"JAX {jax.__version__}, {jax.devices()}", flush=True)
    except ImportError:
        pass

    results = {}

    # 1. EM test: ES vs EM at beta=0.02 and beta=0.1
    results["es_beta0"] = run_jax("ES (beta=0)", electromagnetic=False)
    results["em_beta002"] = run_jax("EM beta=0.02", electromagnetic=True, beta_e=0.02)
    results["em_beta01"] = run_jax("EM beta=0.1", electromagnetic=True, beta_e=0.1)

    # 2. Dimits: R/L_Ti=3 vs 6.9, n_kx=256, 10K steps for longer physical time
    results["dimits_3_256"] = run_jax(
        "Dimits R/L_Ti=3.0 n_kx=256",
        R_L_Ti=3.0,
        R_L_Te=3.0,
        n_kx=256,
        n_steps=10000,
        save_interval=200,
    )
    results["dimits_69_256"] = run_jax(
        "Dimits R/L_Ti=6.9 n_kx=256",
        R_L_Ti=6.9,
        R_L_Te=6.9,
        n_kx=256,
        n_steps=10000,
        save_interval=200,
    )

    out = Path("gpu_results")
    out.mkdir(exist_ok=True)
    outpath = out / "em_and_dimits.json"
    outpath.write_text(json.dumps({"results": results}, indent=2, default=str))
    print(f"\nSaved to {outpath}", flush=True)

    print("\n=== Summary ===", flush=True)
    for k, v in results.items():
        print(
            f"  {k}: chi_gB={v['chi_i_gB']}, late_growth={v['late_growth']:.4f}, phi={v['phi_final']:.3e}", flush=True
        )


if __name__ == "__main__":
    main()

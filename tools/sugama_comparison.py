#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Sugama vs Krook Collision Comparison
"""3-way CBC comparison: Krook, Sugama (adiabatic), Sugama+kinetic electrons."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def run(label: str, coll: str, ke: bool, implicit: bool, mr: float) -> dict:
    from scpn_control.core.gk_nonlinear import NonlinearGKConfig

    print(f"\n=== {label} ===", flush=True)
    cfg = NonlinearGKConfig(
        n_kx=128, n_ky=16, n_theta=32, n_vpar=16, n_mu=8,
        dt=0.05, n_steps=5000, save_interval=100,
        R_L_Ti=6.9, R_L_Te=6.9, R_L_ne=2.2,
        q=1.4, s_hat=0.78, R0=2.78, a=1.0, B0=2.0,
        cfl_adapt=True, cfl_factor=0.5,
        nonlinear=True, collisions=True, nu_collision=0.01,
        collision_model=coll, hyper_coeff=0.2,
        kinetic_electrons=ke, implicit_electrons=implicit,
        mass_ratio_me_mi=mr,
    )
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
        "label": label, "collision_model": coll,
        "kinetic_electrons": ke, "implicit": implicit,
        "chi_i_gB": float(chi_gB) if np.isfinite(chi_gB) else None,
        "elapsed_s": elapsed, "converged": bool(r.converged),
        "phi_rms": r.phi_rms_t.tolist(), "Q_i": r.Q_i_t.tolist(),
        "time": r.time.tolist(),
    }


def main() -> None:
    try:
        import jax
        print(f"JAX {jax.__version__}, {jax.devices()}", flush=True)
    except ImportError:
        pass

    results = {}
    results["krook_adiabatic"] = run(
        "Krook + adiabatic", "krook", ke=False, implicit=False, mr=1/400)
    results["sugama_adiabatic"] = run(
        "Sugama + adiabatic", "sugama", ke=False, implicit=False, mr=1/400)
    results["sugama_kinetic_e"] = run(
        "Sugama + kinetic_e (implicit)", "sugama", ke=True, implicit=True, mr=1/400)

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

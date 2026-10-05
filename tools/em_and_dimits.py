#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — EM GPU Test + Long Dimits Shift.

"""Manual JAX experiments: three ES/EM presets and two 10K-step Dimits presets.

These runs report solver diagnostics, not independently validated Dimits shifts,
physical convergence, experimental transport or facility control evidence.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def run_jax(label: str, **overrides: object) -> dict[str, Any]:
    """Run one fresh JAX nonlinear GK configuration and report saved diagnostics.

    Parameters
    ----------
    label : str
        Copied report/console label; it does not authenticate a physical case.
    **overrides : object
        NonlinearGKConfig fields, overriding the local defaults. Defaults use
        (n_kx, n_ky, n_theta, n_vpar, n_mu) = (128, 16, 32, 16, 8), 5000 steps,
        save interval 100, dt 0.05 with CFL adaptation and hyper coefficient 0.2.
        Other CBC defaults are supplied explicitly below. Unsupported names or
        types follow the defining config/backend errors, without substitution.

    Returns
    -------
    dict
        Label, chi_i_gB, fractional endpoint late_growth, wall-clock elapsed_s,
        final saved phi RMS/code time, and the original solver converged flag.
        chi_i_gB is result.chi_i / max(R_L_Ti, 0.01), or None when nonfinite;
        this display label does not establish a physical gyro-Bohm conversion.
        With fewer than three last-quarter samples, late_growth remains zero.

    Raises
    ------
    RuntimeError
        JAX is unavailable under the unchanged no-fallback policy, or the actual
        result has no saved history. No final sample is invented for that case.
    TypeError, ValueError
        The defining config, grid or backend refuses supplied overrides.

    Notes
    -----
    Each call creates a new solver; its default initial-state seed is 42.
    JAX selects its configured CPU/GPU backend; this runner enables no NumPy
    fallback. Elapsed seconds bracket run(), while result time is solver code
    time. late_growth uses last-minus-first over first times elapsed code time,
    not logarithmic growth. The converged flag means more than one finite flux
    sample, not empirical or asymptotic convergence. Diagnostics go to stdout;
    this function writes no report. Calls share JAX runtime/device resources;
    no concurrent-state isolation or timing comparability is guaranteed.
    """
    from scpn_control.core.gk_nonlinear import NonlinearGKConfig
    from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver

    print(f"\n=== {label} ===", flush=True)
    defaults: dict[str, Any] = dict(
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
    if len(r.phi_rms_t) == 0:
        raise RuntimeError("manual GK experiment produced no saved history")
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
    """Run the five fixed long presets and overwrite a caller-relative raw report.

    Returns
    -------
    None
        Print backend, per-case diagnostics and final summaries. Three base
        presets run 5000 steps (ES, EM beta_e 0.02/0.1); two 256-kx presets run
        10000 steps with R_L_Ti/R_L_Te 3.0 or 6.9 and save interval 200.

    Raises
    ------
    RuntimeError, TypeError, ValueError
        A defining experiment refuses its backend/config/history; remaining
        cases and the final report are not produced.
    OSError
        Creating gpu_results or writing em_and_dimits.json fails.

    Notes
    -----
    No CLI arguments are parsed. JAX chooses its configured backend. Output
    uses legacy json.dumps(indent=2, default=str), including its nonfinite-float
    convention, without atomic replacement, locking, campaign custody or an
    authenticated source digest. All cases must finish before the report write.
    Existing output may be replaced; this raw experiment grants no physical,
    convergence, independently verified reference or facility admission.
    """
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

#!/usr/bin/env python3
"""Long Dimits shift: R/L_Ti=3.0 vs 6.9 at 20K steps."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def main() -> None:
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
        print(
            f"  phi: [{r.phi_rms_t[0]:.4e}...{r.phi_rms_t[-1]:.4e}]", flush=True
        )
        print(f"  time: [{r.time[0]:.2f}, {r.time[-1]:.2f}]", flush=True)

        n4 = 3 * len(r.phi_rms_t) // 4
        phi_lq = r.phi_rms_t[n4:]
        if len(phi_lq) > 2 and phi_lq[0] > 0:
            g = (phi_lq[-1] - phi_lq[0]) / (
                phi_lq[0] * max(r.time[-1] - r.time[n4], 0.01)
            )
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

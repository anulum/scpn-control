#!/usr/bin/env python3
"""Nonlinear GK CBC convergence study — runs on GPU via JAX.

Usage:
    pip install "jax[cuda12]"
    pip install -e .
    python tools/gk_convergence_benchmark.py

Writes results to /tmp/gk_convergence.json after each run completes.
"""

import json
import time

# Verify JAX backend
import jax

devices = jax.devices()
backend = jax.default_backend()
print(f"JAX backend: {backend}, devices: {devices}", flush=True)
if backend == "cpu":
    print("WARNING: JAX is CPU-only. GPU not detected.", flush=True)
    print("Install: pip install 'jax[cuda12]'", flush=True)

from scpn_control.core.gk_nonlinear import NonlinearGKConfig
from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver

RESULTS_FILE = "/tmp/gk_convergence.json"


def save(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  -> saved to {RESULTS_FILE}", flush=True)


def run_benchmark(name, config):
    print(f"\n{'=' * 60}", flush=True)
    print(f"[{name}] n_kx={config.n_kx} n_ky={config.n_ky} steps={config.n_steps} dt={config.dt}", flush=True)
    print(f"{'=' * 60}", flush=True)
    t0 = time.time()
    solver = JaxNonlinearGKSolver(config)
    result = solver.run()
    wall = time.time() - t0
    print(f"  chi_i_gB  = {result.chi_i_gB:.6f}", flush=True)
    print(f"  converged = {result.converged}", flush=True)
    print(f"  wall_time = {wall:.1f}s", flush=True)
    print(f"  Q_i_t     = {[float(x) for x in result.Q_i_t]}", flush=True)
    return {
        "chi_i_gB": float(result.chi_i_gB) if not __import__("math").isnan(result.chi_i_gB) else None,
        "converged": result.converged,
        "wall_s": round(wall, 1),
        "Q_i": [float(x) for x in result.Q_i_t],
    }


def main():
    results = {}

    # 1. Timing calibration: 10 steps at full resolution
    print("\n[CALIBRATION] 10 steps at n_kx=128 to estimate total time...", flush=True)
    t0 = time.time()
    c_cal = NonlinearGKConfig(
        n_kx=128,
        n_ky=32,
        n_vpar=16,
        n_mu=8,
        n_steps=10,
        dt=0.02,
        save_interval=10,
        kinetic_electrons=False,
        beta_e=0.0,
    )
    solver_cal = JaxNonlinearGKSolver(c_cal)
    _ = solver_cal.run()
    cal_time = time.time() - t0
    per_step = cal_time / 10
    est_2000 = per_step * 2000 / 60
    print(f"  {cal_time:.1f}s for 10 steps = {per_step:.3f}s/step", flush=True)
    print(f"  Estimated 2000 steps: {est_2000:.1f} min", flush=True)
    results["calibration"] = {"per_step_s": round(per_step, 4), "est_2000_min": round(est_2000, 1)}
    save(results)

    if est_2000 > 120:
        print(f"\n  WARNING: 2000 steps would take {est_2000:.0f} min. Reducing to 500 steps.", flush=True)
        n_steps_main = 500
    else:
        n_steps_main = 2000

    # 2. Adiabatic CBC
    c_adi = NonlinearGKConfig(
        n_kx=128,
        n_ky=32,
        n_vpar=16,
        n_mu=8,
        n_steps=n_steps_main,
        dt=0.02,
        save_interval=max(n_steps_main // 10, 1),
        kinetic_electrons=False,
        beta_e=0.0,
    )
    results["adiabatic"] = run_benchmark("ADIABATIC", c_adi)
    save(results)

    # 3. Kinetic electrons
    c_kin = NonlinearGKConfig(
        n_kx=128,
        n_ky=32,
        n_vpar=16,
        n_mu=8,
        n_steps=n_steps_main,
        dt=0.02,
        save_interval=max(n_steps_main // 10, 1),
        kinetic_electrons=True,
        beta_e=0.01,
    )
    results["kinetic"] = run_benchmark("KINETIC", c_kin)
    save(results)

    # 4. Grid convergence scan (adiabatic, 500 steps each)
    for nkx in [64, 256]:
        c_grid = NonlinearGKConfig(
            n_kx=nkx,
            n_ky=32,
            n_vpar=16,
            n_mu=8,
            n_steps=500,
            dt=0.02,
            save_interval=50,
            kinetic_electrons=False,
            beta_e=0.0,
        )
        results[f"grid_nkx{nkx}"] = run_benchmark(f"GRID nkx={nkx}", c_grid)
        save(results)

    print("\n" + "=" * 60, flush=True)
    print("ALL DONE", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()

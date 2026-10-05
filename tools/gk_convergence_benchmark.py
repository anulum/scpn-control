#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Gk Convergence Benchmark.

"""Manual JAX nonlinear CBC grid study with disposable raw JSON output.

Import initialises the actual configured JAX backend and prints its devices;
CPU is permitted and missing JAX refuses import. run_benchmark accepts a caller
config, while main keeps the original large calibration and four-case campaign.
The provider's converged flag is finite-sample bookkeeping, not saturation or
grid-convergence admission. RESULTS_FILE is mutable caller-owned scratch state.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from typing import TypedDict

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

RESULTS_FILE: str = "/tmp/gk_convergence.json"


class BenchmarkResult(TypedDict):
    """Raw chi_i_gB, finite-sample flag, rounded wall seconds and saved code-unit ion flux."""

    chi_i_gB: float | None
    converged: bool
    wall_s: float
    Q_i: list[float]


class CalibrationResult(TypedDict):
    """Cold ten-step wall-time estimate, including construction; not steady-state throughput."""

    per_step_s: float
    est_2000_min: float


def save(results: Mapping[str, object]) -> None:
    """Replace RESULTS_FILE with a caller-supplied raw JSON mapping and print its path.

    RESULTS_FILE defaults to absolute /tmp/gk_convergence.json and is mutable;
    a caller-assigned relative value follows cwd. The parent must exist. UTF-8
    is not forced: open uses the platform text codec. JSON retains the original
    indent=2 and allow_nan=True behaviour, including nonstandard NaN/Infinity.
    None chi values serialise as null. Input mappings are borrowed, not verified
    or authenticated. This disposable scratch is not canonical benchmark evidence.

    OSError propagates on creation/write; TypeError/ValueError propagate for
    unsupported/circular JSON inputs. Existing output is truncated before
    serialisation and can remain partial after failure. No alias guard, atomic
    replacement, locking or ownership check exists. Callers must own the target
    and serialise module-global changes and writes themselves.
    """
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  -> saved to {RESULTS_FILE}", flush=True)


def run_benchmark(name: str, config: NonlinearGKConfig) -> BenchmarkResult:
    """Run the actual fresh JAX provider for the borrowed config and return diagnostics.

    Parameters
    ----------
    name : str
        Console label without physical-case authentication.
    config : NonlinearGKConfig
        Mutable caller-supplied solver controls. This wrapper adds no grid,
        count, domain or physical validation and requests no NumPy fallback.

    Returns
    -------
    BenchmarkResult
        chi_i_gB is the provider's second-half saved Q_i mean divided by
        max(R_L_Ti,0.01), or None only when NaN. Empty histories give zero;
        one sample yields the provider's empty-mean NaN. Infinity and nonfinite
        Q_i samples are preserved. Q_i contains saved post-step code-unit flux,
        without times/config or reference admission. converged is copied from
        the provider: more than one finite saved flux, not saturation, complete
        integration, final-state validity or a grid-convergence decision.

    Raises
    ------
    RuntimeError, ValueError, IndexError, ZeroDivisionError
        Provider/config/grid refusal propagates unchanged. Allocation/backend
        errors also propagate; no finite-report or scientific admission follows.

    Notes
    -----
    A fresh solver initialises its unchanged seed-42 state and configured JAX
    device/precision; its default grid axes are species,kx,ky,theta,vpar,mu.
    The system wall clock brackets construction and run(), including first-use
    compilation/allocation and synchronised diagnostic conversions. Import and
    console result formatting/printing/file writing are excluded. wall_s is
    rounded to one decimal and is not a monotonic or controlled timing claim.
    This call prints diagnostics, writes no file and retains no solver state.
    """
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
        "chi_i_gB": float(result.chi_i_gB) if not math.isnan(result.chi_i_gB) else None,
        "converged": result.converged,
        "wall_s": round(wall, 1),
        "Q_i": [float(x) for x in result.Q_i_t],
    }


def main() -> None:
    """Run the original large calibration and four-case study, saving after each stage.

    There are no parsed CLI arguments, help, reduced preset or return status.
    Successful script completion exits zero; unhandled provider/I/O/JSON errors
    produce a Python failure. main() returns None and prints the accumulated map.
    Import already printed actual JAX backend/devices; CPU is not refused.

    Calibration uses 128kx,32ky,16vpar,8mu,10steps,dt0.02,save10,adiabatic
    electrons/beta0, with other config defaults including theta64/species2.
    Construction/run wall time divided by10 estimates steps2000 in minutes.
    If that estimate exceeds120, the next two cases use500 steps; otherwise2000.
    This does not distinguish compilation time or early stopping from throughput.
    Adiabatic beta0 and kinetic-electron beta0.01 cases use that count and
    save=max(count//10,1); electromagnetic remains its False default. Both grid
    cases use kx64/256,adiabatic beta0,500steps,save50. All use dt0.02 and
    unchanged CFL adaptation, initialisation and remaining provider defaults.

    save() overwrites the mutable caller-owned RESULTS_FILE after calibration
    and each actual case, retaining partial campaign progress on later failure.
    Large memory/runtime requirements and finite-sample flags do not establish
    successful saturation, grid convergence or admitted CBC/reference evidence.
    """
    results: dict[str, BenchmarkResult | CalibrationResult] = {}

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

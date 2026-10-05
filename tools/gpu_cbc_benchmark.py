#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GPU CBC Benchmark.

"""Manual fixed-grid CBC model diagnostics and NumPy/JAX run-clock observations.

The public functions return raw in-memory model reports. CPU JAX is permitted;
no backend parity, speedup or physical-reference test is performed. main retains
the original large campaign and guarded cwd-relative gpu_results output. The
standalone entry can install CUDA JAX when import fails, before the campaign
guard; import of this module itself neither installs nor runs a solver.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal, TypedDict

from scpn_control.benchmark_records import require_recorded_campaign

REPO_ROOT = Path(__file__).resolve().parents[1]


class LinearBenchmark(TypedDict):
    """Six raw linear spectrum fields; growth is normalised and elapsed time is run-local."""

    gamma_max: float
    k_y_max: float
    elapsed_s: float
    gamma: list[float]
    k_y: list[float]
    mode_type: list[str]


class NonlinearBenchmark(TypedDict):
    """Requested count, raw code-unit means/trace and provider flag, without final solver state."""

    label: str
    chi_i: float
    chi_e: float
    converged: bool
    elapsed_s: float
    n_steps: int
    phi_rms: list[float]
    zonal_rms: list[float]
    Q_i: list[float]
    time: list[float]


class UnavailableJaxBenchmark(TypedDict):
    """Explicit unavailable-provider report; no NumPy fallback or measured timing."""

    label: str
    skipped: Literal[True]
    reason: str


class TGLFBenchmark(TypedDict):
    """SAT1 model diffusivities in square metres per second and raw normalised spectrum."""

    chi_i: float
    chi_e: float
    D_e: float
    dominant_mode: str
    elapsed_s: float
    gamma: list[float]
    k_y: list[float]


def install_jax() -> bool:
    """Provision CUDA JAX through this interpreter's pip and report actual devices.

    Executes [sys.executable,-m,pip,install,-q,jax[cuda12]], inheriting cwd/env
    and console. This changes the caller's environment and may use the network;
    it is not a solver, GPU availability check or pinned provisioning contract.
    Returns True only after pip succeeds and actual JAX import/devices succeed.
    CalledProcessError/ImportError/backend errors propagate. There is no rollback,
    timeout or campaign guard in this explicit operator function.
    """
    print("=== Installing JAX[cuda] ===")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "jax[cuda12]",
        ]
    )
    import jax

    print(f"JAX {jax.__version__}, devices: {jax.devices()}")
    return True


def run_linear_benchmark() -> LinearBenchmark:
    """Return the actual fixed native linear CBC spectrum without a reference comparison.

    Deuterium and adiabatic electrons use T2keV,n5e19m^-3,R/L_T6.9,R/L_n2.2;
    R0=2.78m,a=1m,B0=2T,q1.4,shear0.78,16 ion-scale ky,theta64,periods2.
    gamma is provider-normalised growth, ky is k_y*rho_s, mode_type has one
    string per bin; peak fields identify the largest growth, not acceptance.
    Monotonic elapsed_s brackets solve_linear_gk only, excluding imports/species
    construction, console and serialisation. No files/state persist; provider
    errors propagate. This native linear path does not require JAX or a GPU.
    """
    from scpn_control.core.gk_eigenvalue import solve_linear_gk
    from scpn_control.core.gk_species import deuterium_ion, electron

    print("\n=== Linear GK — CBC spectrum ===")
    species = [
        deuterium_ion(T_keV=2.0, n_19=5.0, R_L_T=6.9, R_L_n=2.2),
        electron(T_keV=2.0, n_19=5.0, R_L_T=6.9, R_L_n=2.2, adiabatic=True),
    ]
    t0 = time.perf_counter()
    result = solve_linear_gk(
        species_list=species,
        R0=2.78,
        a=1.0,
        B0=2.0,
        q=1.4,
        s_hat=0.78,
        n_ky_ion=16,
        n_theta=64,
        n_period=2,
    )
    elapsed = time.perf_counter() - t0
    print(f"  gamma_max={result.gamma_max:.4f} at k_y={result.k_y_max:.3f}")
    print(f"  Elapsed: {elapsed:.2f}s")
    return {
        "gamma_max": float(result.gamma_max),
        "k_y_max": float(result.k_y_max),
        "elapsed_s": elapsed,
        "gamma": result.gamma.tolist(),
        "k_y": result.k_y.tolist(),
        "mode_type": result.mode_type,
    }


def run_nonlinear_numpy(n_steps: int = 500, label: str = "numpy") -> NonlinearBenchmark:
    """Return raw diagnostics from the original fresh fixed-grid NumPy provider.

    n_steps is the requested integer count and label is unauthenticated console
    text. The wrapper adds no count validation. Grid16kx/16ky/64theta/16vpar/8mu,
    species2,dt0.02,save50,seed42; CBC drives6.9/6.9/2.2, geometry2.78/1/2,
    q1.4/shear0.78, nonlinear/collisions on,nu0.01,hyper0.1,CFL adaptation off.
    Remaining config defaults are unchanged, including adiabatic electrons.
    chi_i/e are second-half saved raw code-unit flux means, not SI diffusivity
    or chi_i_gB. Zero steps yield empty traces/zero means; one saved sample gives
    NaN means. time is saved post-step normalised model time, not wall time;
    n_steps does not prove completed integration. Finite saved ion flux plus
    no detected divergence supplies converged, not saturation/reference admission.
    Run-only monotonic elapsed_s excludes imports/config/solver construction,
    console and writes, but includes initialisation and diagnostics. No file or
    final state persists. Provider/allocation errors propagate unchanged.
    """
    from scpn_control.core.gk_nonlinear import NonlinearGKConfig, NonlinearGKSolver

    print(f"\n=== Nonlinear GK — {label} (16x16x64x16x8 x {n_steps}) ===")
    cfg = NonlinearGKConfig(
        n_kx=16,
        n_ky=16,
        n_theta=64,
        n_vpar=16,
        n_mu=8,
        dt=0.02,
        n_steps=n_steps,
        save_interval=50,
        R_L_Ti=6.9,
        R_L_Te=6.9,
        R_L_ne=2.2,
        q=1.4,
        s_hat=0.78,
        R0=2.78,
        a=1.0,
        B0=2.0,
        cfl_adapt=False,
        nonlinear=True,
        collisions=True,
        nu_collision=0.01,
        hyper_coeff=0.1,
    )
    solver = NonlinearGKSolver(cfg)
    t0 = time.perf_counter()
    result = solver.run()
    elapsed = time.perf_counter() - t0
    print(f"  chi_i={result.chi_i:.6f}, converged={result.converged}")
    print(f"  phi_rms final={result.phi_rms_t[-1]:.6e}" if len(result.phi_rms_t) else "  (empty)")
    print(f"  zonal_rms final={result.zonal_rms_t[-1]:.6e}" if len(result.zonal_rms_t) else "  (empty)")
    print(f"  Elapsed: {elapsed:.2f}s")
    return {
        "label": label,
        "chi_i": float(result.chi_i),
        "chi_e": float(result.chi_e),
        "converged": result.converged,
        "elapsed_s": elapsed,
        "n_steps": n_steps,
        "phi_rms": result.phi_rms_t.tolist(),
        "zonal_rms": result.zonal_rms_t.tolist(),
        "Q_i": result.Q_i_t.tolist(),
        "time": result.time.tolist(),
    }


def run_nonlinear_jax(n_steps: int = 500, label: str = "jax") -> NonlinearBenchmark | UnavailableJaxBenchmark:
    """Use actual JAX on its configured device for the unchanged fixed nonlinear grid.

    Parameters/count/trace/units follow run_nonlinear_numpy. CPU is permitted;
    no device selection, installation or NumPy fallback is requested. If the
    provider import raises ImportError or jax_available is False, return only
    label/skipped=True/reason, respectively import failed or JAX not available.
    Other errors propagate. Available reports retain raw NaN/Infinity values.
    Provider converged tests multiple finite saved ion fluxes; unlike NumPy it
    has no separate divergence flag and proves no final-state validity.
    Monotonic elapsed_s covers run initialisation, first-use JIT, synchronisation
    and diagnostic conversion, excluding imports/config/solver construction,
    printing/writes. Separate clocks are not isolated matched-throughput or
    backend-parity evidence. The actual configured precision/cache affects results.
    """
    try:
        from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver, jax_available

        if not jax_available():
            print(f"\n=== Nonlinear GK — {label}: JAX not available, skipping ===")
            return {"label": label, "skipped": True, "reason": "JAX not available"}
    except ImportError:
        print(f"\n=== Nonlinear GK — {label}: import failed, skipping ===")
        return {"label": label, "skipped": True, "reason": "import failed"}

    from scpn_control.core.gk_nonlinear import NonlinearGKConfig

    print(f"\n=== Nonlinear GK — {label} (16x16x64x16x8 x {n_steps}) ===")
    cfg = NonlinearGKConfig(
        n_kx=16,
        n_ky=16,
        n_theta=64,
        n_vpar=16,
        n_mu=8,
        dt=0.02,
        n_steps=n_steps,
        save_interval=50,
        R_L_Ti=6.9,
        R_L_Te=6.9,
        R_L_ne=2.2,
        q=1.4,
        s_hat=0.78,
        R0=2.78,
        a=1.0,
        B0=2.0,
        cfl_adapt=False,
        nonlinear=True,
        collisions=True,
        nu_collision=0.01,
        hyper_coeff=0.1,
    )
    solver = JaxNonlinearGKSolver(cfg)
    t0 = time.perf_counter()
    result = solver.run()
    elapsed = time.perf_counter() - t0
    print(f"  chi_i={result.chi_i:.6f}, converged={result.converged}")
    print(f"  Elapsed: {elapsed:.2f}s")
    return {
        "label": label,
        "chi_i": float(result.chi_i),
        "chi_e": float(result.chi_e),
        "converged": result.converged,
        "elapsed_s": elapsed,
        "n_steps": n_steps,
        "phi_rms": result.phi_rms_t.tolist(),
        "zonal_rms": result.zonal_rms_t.tolist(),
        "Q_i": result.Q_i_t.tolist(),
        "time": result.time.tolist(),
    }


def run_tglf_native() -> TGLFBenchmark:
    """Return native SAT1 CBC model diagnostics without an external TGLF executable.

    Fixed drives6.9/6.9/2.2,q1.4,shear0.78,R0=2.78m,a1m,B2T,epsilon0.18,
    temperatures2keV,n_e5e19m^-3; SAT1,ion ky16,theta64,other defaults retained.
    The native model supplies chi_i,chi_e,D_e in m^2/s, normalised gamma/ky and
    dominant_mode. Its linear subcall uses kinetic electrons and period1,
    differing from run_linear_benchmark; no independent physical reference is
    checked. Monotonic elapsed_s includes solver construction plus solve and
    excludes imports/params construction/console/writes. No file/state persists;
    provider/allocation errors propagate. Neither JAX nor GPU is required.
    """
    from scpn_control.core.gk_interface import GKLocalParams
    from scpn_control.core.gk_tglf_native import TGLFNativeConfig, TGLFNativeSolver

    print("\n=== Native TGLF — CBC ===")
    params = GKLocalParams(
        R_L_Ti=6.9,
        R_L_Te=6.9,
        R_L_ne=2.2,
        q=1.4,
        s_hat=0.78,
        R0=2.78,
        a=1.0,
        B0=2.0,
        epsilon=0.18,
        T_e_keV=2.0,
        T_i_keV=2.0,
        n_e=5.0,
    )
    t0 = time.perf_counter()
    solver = TGLFNativeSolver(TGLFNativeConfig(sat_model="SAT1", n_ky_ion=16, n_theta=64))
    result = solver.solve(params)
    elapsed = time.perf_counter() - t0
    print(f"  chi_i={result.chi_i:.6f}, chi_e={result.chi_e:.6f}")
    print(f"  dominant_mode={result.dominant_mode}")
    print(f"  Elapsed: {elapsed:.2f}s")
    return {
        "chi_i": float(result.chi_i),
        "chi_e": float(result.chi_e),
        "D_e": float(result.D_e),
        "dominant_mode": result.dominant_mode,
        "elapsed_s": elapsed,
        "gamma": result.gamma.tolist(),
        "k_y": result.k_y.tolist(),
    }


def main() -> None:
    """Run the unchanged six-stage campaign and write guarded cwd-relative raw JSON.

    No argv/help/reduced mode is parsed. First require_recorded_campaign binds
    gpu_results/gk_nonlinear_cbc_gpu.json to this source's REPO_ROOT. Persistent
    repository destinations require the recorded runner; outside-root scratch
    follows its existing exemption. This is not reference/result authentication.
    mkdir(exist_ok=True) then linear,TGLF,NumPy500,JAX500,NumPy2000,JAX2000 run
    sequentially. Each nonlinear call uses the same original fixed grid; these
    are fresh runs, not a retained-state warmup or increasing-resolution study.
    One UTC host timestamp and all raw maps are written only after all calls.
    Legacy indent2/allow_nan=True and platform text encoding remain; output
    replaces existing bytes and may be partial on I/O failure. No producer
    locking, alias guard, atomic replacement or schema/finite/reference check
    is added. Provider/I/O/guard errors propagate; earlier in-memory stages are
    lost on later failure. Missing JAX yields skipped reports, not script failure.
    main returns None; standalone success exits0. The standalone import/install
    block precedes main's guard and may provision JAX. Long CPU/GPU workloads,
    run-clock observations and model reports cannot establish physical admission.
    """
    out_dir = Path("gpu_results")
    out_path = out_dir / "gk_nonlinear_cbc_gpu.json"
    require_recorded_campaign(out_path, repository_root=REPO_ROOT)
    out_dir.mkdir(exist_ok=True)

    report: dict[str, object] = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # Phase 1: Linear benchmark
    report["linear_cbc"] = run_linear_benchmark()

    # Phase 2: TGLF native
    report["tglf_native_cbc"] = run_tglf_native()

    # Phase 3: Nonlinear NumPy (500 steps for timing)
    report["nonlinear_numpy_500"] = run_nonlinear_numpy(500, "numpy_500")

    # Phase 4: Fresh nonlinear JAX run (500 requested steps)
    report["nonlinear_jax_500"] = run_nonlinear_jax(500, "jax_500")

    # Phase 5: Longer NumPy run on the same fixed grid (2000 steps)
    report["nonlinear_numpy_2000"] = run_nonlinear_numpy(2000, "numpy_2000")

    # Phase 6: Longer JAX run on the same fixed grid (2000 steps)
    report["nonlinear_jax_2000"] = run_nonlinear_jax(2000, "jax_2000")

    # Save
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n=== Report saved to {out_path} ===")
    print(json.dumps({k: v.get("elapsed_s", "?") if isinstance(v, dict) else v for k, v in report.items()}, indent=2))


if __name__ == "__main__":
    # Install JAX if needed
    try:
        import jax

        print(f"JAX already installed: {jax.__version__}, devices: {jax.devices()}")
    except ImportError:
        install_jax()

    main()

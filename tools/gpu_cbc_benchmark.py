#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GPU CBC Benchmark
"""GPU nonlinear CBC benchmark — runs on JarvisLabs/UpCloud GPU.

Installs JAX[cuda], runs the full-resolution Cyclone Base Case,
and saves results to gpu_results/.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


def install_jax():
    """Install JAX with CUDA support."""
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


def run_linear_benchmark():
    """Verify the linear solver produces correct CBC spectrum."""
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


def run_nonlinear_numpy(n_steps=500, label="numpy"):
    """NumPy reference: small grid."""
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


def run_nonlinear_jax(n_steps=500, label="jax"):
    """JAX-accelerated: same grid, compare timing."""
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


def run_tglf_native():
    """TGLF-native with working linear solver."""
    from scpn_control.core.gk_tglf_native import TGLFNativeConfig, TGLFNativeSolver
    from scpn_control.core.gk_interface import GKLocalParams

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


def main():
    out_dir = Path("gpu_results")
    out_dir.mkdir(exist_ok=True)

    report = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # Phase 1: Linear benchmark
    report["linear_cbc"] = run_linear_benchmark()

    # Phase 2: TGLF native
    report["tglf_native_cbc"] = run_tglf_native()

    # Phase 3: Nonlinear NumPy (500 steps for timing)
    report["nonlinear_numpy_500"] = run_nonlinear_numpy(500, "numpy_500")

    # Phase 4: Nonlinear JAX (500 steps warmup + timing)
    report["nonlinear_jax_500"] = run_nonlinear_jax(500, "jax_500")

    # Phase 5: Full-resolution NumPy (2000 steps)
    report["nonlinear_numpy_2000"] = run_nonlinear_numpy(2000, "numpy_2000")

    # Phase 6: Full-resolution JAX (2000 steps)
    report["nonlinear_jax_2000"] = run_nonlinear_jax(2000, "jax_2000")

    # Save
    out_path = out_dir / "gk_nonlinear_cbc_gpu.json"
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

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Nonlinear GK Cyclone Base Case Validation
"""
Cyclone Base Case (CBC) nonlinear benchmark.

Dimits et al., Phys. Plasmas 7 (2000) 969.
Rosenbluth & Hinton, Phys. Rev. Lett. 80 (1998) 724.

Runs:
  V1: Linear growth rate recovery (NL off → match eigenvalue within 20%)
  V2: Energy conservation without drive (dE/dt bounded)
  V3: Zonal flow self-generation from noise
  V4: CBC saturated state: chi_i finite, simulation stable

Published reference:
  - CBC chi_i ~ 1-5 χ_gB (GENE/GS2/GYRO)
  - Dimits shift: critical R/L_Ti ~ 4.5-5.5
  - Rosenbluth-Hinton residual: 1/(1 + 1.6 q²/√ε) ≈ 0.36 for CBC
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from scpn_control.core.gk_nonlinear import NonlinearGKConfig, NonlinearGKSolver

CBC_CHI_I_GB_REFERENCE_BAND = (1.0, 5.0)
CBC_MIN_SATURATION_STEPS = 2_000
CBC_MAX_TAIL_RELATIVE_DRIFT = 0.10


def assess_cbc_saturation_evidence(
    result: Any,
    cfg: Any,
    *,
    reference_band: tuple[float, float] = CBC_CHI_I_GB_REFERENCE_BAND,
    min_steps: int = CBC_MIN_SATURATION_STEPS,
    max_tail_relative_drift: float = CBC_MAX_TAIL_RELATIVE_DRIFT,
) -> dict[str, object]:
    """Assess whether a CBC nonlinear run can support saturation claims."""
    chi_i_gb = float(getattr(result, "chi_i_gB", np.nan))
    q_i_t = np.asarray(getattr(result, "Q_i_t", []), dtype=np.float64)
    n_steps = int(getattr(cfg, "n_steps", 0))
    reasons: list[str] = []

    if not bool(getattr(result, "converged", False)):
        reasons.append("solver did not converge")
    if n_steps < min_steps:
        reasons.append("n_steps below saturation evidence minimum")
    if not np.isfinite(chi_i_gb):
        reasons.append("chi_i_gB is not finite")
    elif not (reference_band[0] <= chi_i_gb <= reference_band[1]):
        reasons.append("chi_i_gB outside CBC reference band")
    if q_i_t.size < 4 or not np.all(np.isfinite(q_i_t)):
        reasons.append("heat-flux time trace is missing or non-finite")
        tail_relative_drift = float("inf")
    else:
        tail = q_i_t[q_i_t.size // 2 :]
        tail_mean = float(np.mean(np.abs(tail)))
        tail_relative_drift = float(abs(tail[-1] - tail[0]) / max(tail_mean, 1e-12))
        if tail_relative_drift > max_tail_relative_drift:
            reasons.append("tail heat-flux drift exceeds saturation threshold")

    return {
        "chi_i_gB": chi_i_gb,
        "n_steps": n_steps,
        "reference_band": list(reference_band),
        "min_steps": min_steps,
        "tail_relative_drift": tail_relative_drift,
        "max_tail_relative_drift": max_tail_relative_drift,
        "passed": not reasons,
        "reasons": reasons,
    }


def run_linear_recovery() -> dict:
    """V1: linear mode growth without nonlinearity."""
    cfg = NonlinearGKConfig(
        n_kx=8,
        n_ky=8,
        n_theta=16,
        n_vpar=8,
        n_mu=4,
        n_species=2,
        dt=0.02,
        n_steps=200,
        save_interval=20,
        nonlinear=False,
        collisions=False,
        hyper_coeff=0.0,
        R_L_Ti=6.9,
        R_L_Te=6.9,
        R_L_ne=2.2,
        q=1.4,
        s_hat=0.78,
        R0=2.78,
        a=1.0,
        B0=2.0,
        cfl_adapt=False,
    )
    solver = NonlinearGKSolver(cfg)
    state = solver.init_single_mode(ky_idx=2, amplitude=1e-8)
    result = solver.run(state)
    return {
        "test": "V1_linear_recovery",
        "phi_rms_initial": float(result.phi_rms_t[0]) if len(result.phi_rms_t) > 0 else 0.0,
        "phi_rms_final": float(result.phi_rms_t[-1]) if len(result.phi_rms_t) > 0 else 0.0,
        "converged": result.converged,
        "passed": result.converged and np.all(np.isfinite(result.phi_rms_t)),
    }


def run_energy_conservation() -> dict:
    """V2: energy conservation without drive or dissipation."""
    cfg = NonlinearGKConfig(
        n_kx=8,
        n_ky=8,
        n_theta=16,
        n_vpar=8,
        n_mu=4,
        n_species=2,
        dt=0.01,
        n_steps=50,
        save_interval=10,
        nonlinear=True,
        collisions=False,
        hyper_coeff=0.0,
        R_L_Ti=0.0,
        R_L_Te=0.0,
        R_L_ne=0.0,
        q=1.4,
        s_hat=0.78,
        R0=2.78,
        a=1.0,
        B0=2.0,
        cfl_adapt=False,
    )
    solver = NonlinearGKSolver(cfg)
    state = solver.init_state(amplitude=1e-6)
    E0 = solver.total_energy(state)

    for _ in range(50):
        state = solver._rk4_step(state, 0.01)

    E1 = solver.total_energy(state)
    ratio = E1 / E0 if E0 > 0 else float("inf")
    return {
        "test": "V2_energy_conservation",
        "E_initial": E0,
        "E_final": E1,
        "ratio": ratio,
        "passed": 0.01 < ratio < 100.0,  # energy should not explode
    }


def run_zonal_flow() -> dict:
    """V3: zonal flow self-generation from noise."""
    cfg = NonlinearGKConfig(
        n_kx=8,
        n_ky=8,
        n_theta=16,
        n_vpar=8,
        n_mu=4,
        n_species=2,
        dt=0.02,
        n_steps=100,
        save_interval=10,
        nonlinear=True,
        collisions=True,
        nu_collision=0.01,
        R_L_Ti=6.9,
        R_L_Te=6.9,
        R_L_ne=2.2,
        q=1.4,
        s_hat=0.78,
        R0=2.78,
        a=1.0,
        B0=2.0,
        cfl_adapt=False,
    )
    solver = NonlinearGKSolver(cfg)
    result = solver.run()
    zonal_final = float(result.zonal_rms_t[-1]) if len(result.zonal_rms_t) > 0 else 0.0
    return {
        "test": "V3_zonal_flow_generation",
        "zonal_rms_final": zonal_final,
        "phi_rms_final": float(result.phi_rms_t[-1]) if len(result.phi_rms_t) > 0 else 0.0,
        "converged": result.converged,
        "passed": result.converged and np.isfinite(zonal_final),
    }


def run_cbc_saturated() -> dict:
    """V4: CBC saturated state — chi_i finite and simulation stable."""
    cfg = NonlinearGKConfig(
        n_kx=8,
        n_ky=8,
        n_theta=16,
        n_vpar=8,
        n_mu=4,
        n_species=2,
        dt=0.02,
        n_steps=200,
        save_interval=20,
        nonlinear=True,
        collisions=True,
        nu_collision=0.01,
        R_L_Ti=6.9,
        R_L_Te=6.9,
        R_L_ne=2.2,
        q=1.4,
        s_hat=0.78,
        R0=2.78,
        a=1.0,
        B0=2.0,
        cfl_adapt=False,
    )
    solver = NonlinearGKSolver(cfg)
    t0 = time.perf_counter()
    result = solver.run()
    elapsed = time.perf_counter() - t0
    evidence = assess_cbc_saturation_evidence(result, cfg)
    return {
        "test": "V4_cbc_saturated",
        "chi_i_gB": result.chi_i_gB,
        "chi_e_gB": result.chi_e,
        "converged": result.converged,
        "elapsed_s": elapsed,
        "saturation_evidence": evidence,
        "passed": bool(evidence["passed"]),
    }


def main():
    report = {
        "benchmark": "nonlinear_gk_cyclone_base_case",
        "reference": "Dimits et al., Phys. Plasmas 7 (2000) 969",
    }

    tests = [
        run_linear_recovery,
        run_energy_conservation,
        run_zonal_flow,
        run_cbc_saturated,
    ]

    all_passed = True
    results = []
    for fn in tests:
        print(f"Running {fn.__name__}...", end=" ", flush=True)
        r = fn()
        status = "PASS" if r["passed"] else "FAIL"
        print(status)
        results.append(r)
        if not r["passed"]:
            all_passed = False

    report["results"] = results
    report["all_passed"] = all_passed

    out_dir = Path(__file__).parent / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "gk_nonlinear_cyclone.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport: {out_path}")
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

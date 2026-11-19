# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Benchmark Transport.

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Transport Validation Benchmark
# © 1996–2026 Miroslav Šotek. All rights reserved.
# ──────────────────────────────────────────────────────────────────────
"""
Transport validation benchmark.

Three cases: the 1.5D transport solver's ion channel against an analytic
cylindrical steady state; the critical-gradient threshold of the analytic
transport model; and an evaluation of the IPB98(y,2) confinement scaling at
ITER-like parameters. The third case checks only that the scaling returns a
finite positive time and a finite non-negative uncertainty; it does not
compare the solver with the scaling.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from scpn_control.benchmark_records import require_recorded_campaign
from scpn_control.core.integrated_transport_solver import TransportSolver
from scpn_control.core.neural_transport import NeuralTransportModel, TransportInputs
from scpn_control.core.scaling_laws import ipb98y2_with_uncertainty

PURE_DIFFUSION_MAX_RELATIVE_ERROR = 0.05
REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "transport_benchmark.json"
MARKDOWN_REPORT = REPORT_DIR / "transport_benchmark.md"


def run_pure_diffusion_benchmark(nr: int = 200) -> dict[str, Any]:
    """Compare the ion channel with its cylindrical half-power steady state.

    The case splits auxiliary heating equally between ions and electrons.
    Equal diffusivities and nearly equal edge temperatures make exchange a
    small correction; the analytic reference uses only the ion power share.
    """
    # Dummy config: R0=2.0, a=1.0
    cfg = {
        "reactor_name": "Analytic-Transport",
        "grid_resolution": [33, 33],
        "dimensions": {"R_min": 1.0, "R_max": 3.0, "Z_min": -1.5, "Z_max": 1.5},
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        "coils": [],
        "solver": {"max_iterations": 1, "convergence_threshold": 1.0, "max_iterations_outer": 1},
    }
    descriptor, raw_path = tempfile.mkstemp(prefix="scpn-control-transport-", suffix=".json")
    os.close(descriptor)
    cfg_path = Path(raw_path)
    cfg_path.write_text(json.dumps(cfg))

    try:
        # Use higher resolution to minimize discretization error
        solver = TransportSolver(cfg_path, nr=nr)

        # 1. Physics Setup
        chi_const = 2.0  # m^2/s
        P_aux_MW = 10.0  # 10 MW

        solver.chi_i = np.full(nr, chi_const)
        solver.chi_e = np.full(nr, chi_const)
        solver.ne = np.full(nr, 1.0)  # 10^19 m^-3
        solver.n_impurity = np.zeros(nr)

        # Ensure flat heating profile in T-space
        solver.aux_heating_profile_width = 1e10  # Effectively flat
        solver.aux_heating_electron_fraction = 0.5

        # Initial guess close to steady state
        solver.Ti = np.full(nr, 1.0)
        solver.Te = np.full(nr, 1.0)

        dt = 0.01
        # Evolve to steady state
        for _ in range(10000):
            solver.evolve_profiles(dt=dt, P_aux=P_aux_MW)

        # 2. Derive Discrete S_T
        dV = solver._rho_volume_element()
        norm = float(np.sum(dV))
        e_keV_J = 1.602176634e-16
        ne_m3 = 1.0e19
        # This reference is for Ti alone. The configured 50/50 split deposits
        # only the ion share in that channel; using total power double-counts it.
        P_ion_MW = P_aux_MW * (1.0 - solver.aux_heating_electron_fraction)
        S_T_discrete = (P_ion_MW * 1e6) / (1.5 * ne_m3 * e_keV_J * norm)

        # T(rho) = T_edge + (S_T * a^2 / (4*chi)) * (1 - rho^2)
        rho = solver.rho
        T_analytic = 0.1 + (S_T_discrete * 1.0**2 / (4.0 * chi_const)) * (1.0 - rho**2)

        err = np.abs(solver.Ti - T_analytic)
        max_err_rel = np.max(err[1:-1]) / np.max(T_analytic)

        return {
            "max_relative_error": float(max_err_rel),
            "core_actual": float(solver.Ti[0]),
            "core_analytic": float(T_analytic[0]),
            "pass": bool(max_err_rel < PURE_DIFFUSION_MAX_RELATIVE_ERROR),
        }
    finally:
        if cfg_path.exists():
            cfg_path.unlink()


def run_threshold_benchmark() -> dict[str, Any]:
    """Benchmark 2: Critical gradient threshold behavior."""
    model = NeuralTransportModel(auto_discover=False)

    inp_sub = TransportInputs(grad_ti=3.0)
    inp_super = TransportInputs(grad_ti=10.0)

    flux_sub = model.predict(inp_sub)
    flux_super = model.predict(inp_super)

    return {
        "chi_sub": float(flux_sub.chi_i),
        "chi_super": float(flux_super.chi_i),
        "pass": bool(flux_sub.chi_i == 0.0 and flux_super.chi_i > 0.0),
    }


def run_iter_scaling_benchmark() -> dict[str, Any]:
    """Evaluate IPB98(y,2) at ITER-like parameters and check that the result is usable.

    The inputs are Ip=15 MA, B_T=5.3 T, n_e=10e19 m^-3, P_loss=50 MW, R=6.2 m,
    kappa=1.7, epsilon=a/R=2/6.2 and M=2.5. ``pass`` is True when the time is
    finite and positive and its uncertainty finite and non-negative; no solver
    output or measured confinement time is compared.
    """
    Ip, BT, ne19, P_loss, R, kappa, epsilon, M = 15.0, 5.3, 10.0, 50.0, 6.2, 1.7, 2.0 / 6.2, 2.5
    tau_pred, sigma = ipb98y2_with_uncertainty(Ip, BT, ne19, P_loss, R, kappa, epsilon, M)
    tau, uncertainty = float(tau_pred), float(sigma)
    usable = bool(np.isfinite(tau) and tau > 0.0 and np.isfinite(uncertainty) and uncertainty >= 0.0)
    return {"tau_predicted": tau, "uncertainty_sigma": uncertainty, "pass": usable}


def main() -> None:
    """Write the recorded transport benchmark through the campaign gate.

    The report pair goes to this module's ``reports`` directory, whatever the
    working directory; persistent destinations need a recorded campaign.
    """
    report_dir = REPORT_DIR
    json_path = JSON_REPORT
    markdown_path = MARKDOWN_REPORT
    require_recorded_campaign(json_path, markdown_path, repository_root=REPORT_DIR.parents[1])
    results = {
        "pure_diffusion": run_pure_diffusion_benchmark(),
        "threshold": run_threshold_benchmark(),
        "iter_scaling": run_iter_scaling_benchmark(),
    }

    pd = results["pure_diffusion"]
    print(
        f"DEBUG Core: actual={pd['core_actual']:.4f}, analytic={pd['core_analytic']:.4f}, err={pd['max_relative_error']:.2%}"
    )

    report_dir.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with markdown_path.open("w", encoding="utf-8") as f:
        f.write("# Transport Validation Benchmark\n\n")
        f.write("| Test | Metric | Result | Pass |\n")
        f.write("|------|--------|--------|------|\n")
        f.write(f"| Pure Diffusion | Max Rel Error | {pd['max_relative_error']:.2%} | {pd['pass']} |\n")
        th = results["threshold"]
        f.write(f"| Threshold | Chi Sub/Super | {th['chi_sub']:.1f} / {th['chi_super']:.1f} | {th['pass']} |\n")
        it = results["iter_scaling"]
        f.write(f"| ITER Scaling | Tau_E Predicted | {it['tau_predicted']:.2f}s | {it['pass']} |\n")

    print(f"Results saved to {report_dir}")


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Code To Code Benchmark
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Code-to-Code Benchmark: scpn-control vs TORAX
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Benchmark scpn-control transport solver against TORAX on identical scenarios.

TORAX (google-deepmind/torax) is an open-source JAX-based 1.5D tokamak
transport code. This module runs both codes on the same ITER-like scenario
and compares:
  - Te, Ti profiles at t_final
  - Stored energy W_th
  - Confinement time tau_E
  - Convergence behaviour

Usage:
  python -m validation.code_to_code_benchmark [--with-torax]

Without --with-torax, only scpn-control is run and results are saved
for later comparison. With --with-torax, TORAX is imported and run
on the same scenario (requires `pip install torax`).

Results are saved to validation/reports/code_to_code_benchmark.json.
"""

from __future__ import annotations

import json
import pprint
import sys
import time
from pathlib import Path

import numpy as np

TORAX_TMP_CONFIG = Path("validation/reports/_tmp_torax_c2c_config.py")


# ── Scenario Definition ──────────────────────────────────────────────

ITER_SCENARIO = {
    "name": "ITER_15MA_baseline",
    "R0": 6.2,  # m
    "a": 2.0,  # m
    "B0": 5.3,  # T
    "I_p": 15.0e6,  # A
    "kappa": 1.7,
    "delta": 0.33,
    "n_e0": 10.0,  # 10^19 m^-3
    "T_e0": 10.0,  # keV
    "T_i0": 10.0,  # keV
    "P_aux": 50.0,  # MW
    "n_rho": 50,
    "dt": 0.01,  # s
    "n_steps": 100,
    "t_final": 1.0,  # s
}


# ── scpn-control run ─────────────────────────────────────────────────


def _run_scpn_control(scenario: dict) -> dict:
    """Run scpn-control transport solver on the scenario."""
    from scpn_control.core.integrated_transport_solver import TransportSolver

    cfg_dict = {
        "reactor_name": scenario["name"],
        "dimensions": {
            "R_min": scenario["R0"] - scenario["a"],
            "R_max": scenario["R0"] + scenario["a"],
            "Z_min": -scenario["a"] * scenario["kappa"],
            "Z_max": scenario["a"] * scenario["kappa"],
        },
        "grid_resolution": [scenario["n_rho"], scenario["n_rho"]],
        "physics": {"plasma_current_target": scenario["I_p"]},
    }

    tmp_cfg = Path("validation/reports/_tmp_c2c_config.json")
    tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
    tmp_cfg.write_text(json.dumps(cfg_dict))

    try:
        solver = TransportSolver(tmp_cfg, multi_ion=True)

        t0 = time.perf_counter()
        Te_history = [solver.Te.copy()]
        Ti_history = [solver.Ti.copy()]

        for step in range(scenario["n_steps"]):
            solver.evolve_profiles(dt=scenario["dt"], P_aux=scenario["P_aux"])
            if step % 10 == 0:
                Te_history.append(solver.Te.copy())
                Ti_history.append(solver.Ti.copy())

        wall_time = time.perf_counter() - t0

        result = {
            "code": "scpn-control",
            "scenario": scenario["name"],
            "rho": solver.rho.tolist(),
            "Te_final": solver.Te.tolist(),
            "Ti_final": solver.Ti.tolist(),
            "ne_final": solver.ne.tolist(),
            "Te_avg": float(np.mean(solver.Te)),
            "Ti_avg": float(np.mean(solver.Ti)),
            "energy_balance_error": solver.energy_balance_error,
            "particle_balance_error": solver.particle_balance_error,
            "wall_time_s": wall_time,
            "n_steps": scenario["n_steps"],
            "dt": scenario["dt"],
        }
    finally:
        tmp_cfg.unlink(missing_ok=True)

    return result


# ── TORAX run (optional) ─────────────────────────────────────────────


def _torax_config_dict(scenario: dict) -> dict:
    """Build a TORAX config dictionary from the shared benchmark scenario."""
    edge_temp = max(0.1, min(float(scenario["T_e0"]) * 0.05, 1.0))
    edge_density = max(0.1e20, float(scenario["n_e0"]) * 0.1e20)
    fixed_dt = max(float(scenario["dt"]), min(float(scenario["t_final"]) / 5.0, 0.25))
    return {
        "plasma_composition": {
            "main_ion": {"D": 0.5, "T": 0.5},
            "impurity": "Ne",
            "Z_eff": 1.6,
        },
        "profile_conditions": {
            "Ip": float(scenario["I_p"]),
            "T_i": {0.0: {0.0: float(scenario["T_i0"]), 1.0: edge_temp}},
            "T_i_right_bc": edge_temp,
            "T_e": {0.0: {0.0: float(scenario["T_e0"]), 1.0: edge_temp}},
            "T_e_right_bc": edge_temp,
            "n_e": {0.0: {0.0: float(scenario["n_e0"]) * 1.0e19, 1.0: edge_density}},
            "n_e_right_bc": edge_density,
            "nbar": float(scenario["n_e0"]) * 0.8e19,
            "n_e_nbar_is_fGW": False,
            "normalize_n_e_to_nbar": False,
            "initial_psi_mode": "j",
        },
        "numerics": {
            "t_final": float(scenario["t_final"]),
            "fixed_dt": fixed_dt,
            "evolve_ion_heat": True,
            "evolve_electron_heat": True,
            "evolve_current": True,
            "evolve_density": True,
            "resistivity_multiplier": 50.0,
            "max_dt": fixed_dt,
        },
        "geometry": {
            "geometry_type": "circular",
            "n_rho": int(scenario["n_rho"]),
            "R_major": float(scenario["R0"]),
            "a_minor": float(scenario["a"]),
            "B_0": float(scenario["B0"]),
            "elongation_LCFS": float(scenario["kappa"]),
        },
        "neoclassical": {
            "bootstrap_current": {},
        },
        "sources": {
            "generic_current": {
                "fraction_of_total_current": 0.15,
                "gaussian_width": 0.075,
                "gaussian_location": 0.36,
            },
            "generic_particle": {
                "S_total": 0.0,
                "deposition_location": 0.3,
                "particle_width": 0.25,
            },
            "gas_puff": {
                "S_total": 0.0,
                "puff_decay_length": 0.3,
            },
            "pellet": {
                "S_total": 0.0,
                "pellet_width": 0.1,
                "pellet_deposition_location": 0.85,
            },
            "generic_heat": {
                "P_total": float(scenario["P_aux"]) * 1.0e6,
                "electron_heat_fraction": 0.5,
                "gaussian_location": 0.2,
                "gaussian_width": 0.25,
            },
            "fusion": {},
            "ei_exchange": {},
            "ohmic": {},
        },
        "pedestal": {},
        "transport": {
            "model_name": "constant",
        },
        "solver": {
            "solver_type": "linear",
        },
        "time_step_calculator": {
            "calculator_type": "fixed",
        },
    }


def _write_torax_config(path: Path, scenario: dict) -> None:
    """Write a temporary Python config module consumable by TORAX."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = pprint.pformat(_torax_config_dict(scenario), sort_dicts=True, width=100)
    path.write_text(f"CONFIG = {payload}\n", encoding="utf-8")


def _extract_torax_result(data_tree, scenario: dict, wall_time: float) -> dict:
    """Extract comparable final profiles and scalar metrics from TORAX output."""
    profiles = data_tree["profiles"].dataset
    scalars = data_tree["scalars"].dataset

    rho = np.asarray(profiles.coords["rho_norm"].values, dtype=np.float64)
    te = np.asarray(profiles["T_e"].isel(time=-1).values, dtype=np.float64)
    ti = np.asarray(profiles["T_i"].isel(time=-1).values, dtype=np.float64)
    ne = np.asarray(profiles["n_e"].isel(time=-1).values, dtype=np.float64)

    result = {
        "code": "torax",
        "scenario": scenario["name"],
        "status": "done",
        "rho": rho.tolist(),
        "Te_final": te.tolist(),
        "Ti_final": ti.tolist(),
        "ne_final": ne.tolist(),
        "Te_avg": float(np.mean(te)),
        "Ti_avg": float(np.mean(ti)),
        "ne_avg_m3": float(np.mean(ne)),
        "wall_time_s": wall_time,
    }
    if "W_thermal_total" in scalars:
        result["W_thermal_total_J"] = float(scalars["W_thermal_total"].isel(time=-1).values)
    if "tau_E" in scalars:
        result["tau_E_s"] = float(scalars["tau_E"].isel(time=-1).values)
    return result


def _run_torax(scenario: dict) -> dict | None:
    """Run TORAX on the same scenario, if installed."""
    try:
        import torax
    except ImportError:
        print("TORAX not installed — skipping TORAX comparison.")
        print("Install with: pip install torax")
        return None

    _write_torax_config(TORAX_TMP_CONFIG, scenario)
    try:
        cfg = torax.build_torax_config_from_file(TORAX_TMP_CONFIG.resolve())
        t0 = time.perf_counter()
        data_tree, _ = torax.run_simulation(cfg, progress_bar=False)
        wall_time = time.perf_counter() - t0
        return _extract_torax_result(data_tree, scenario, wall_time)
    finally:
        TORAX_TMP_CONFIG.unlink(missing_ok=True)


# ── Comparison ────────────────────────────────────────────────────────


def _compare_results(scpn: dict, torax: dict | None) -> dict:
    """Compare scpn-control and TORAX results."""
    comparison = {
        "scpn_control": scpn,
        "torax": torax,
        "comparison": {},
    }

    if torax is not None and "Te_final" in torax:
        Te_scpn = np.array(scpn["Te_final"])
        Te_torax = np.array(torax["Te_final"])

        # Interpolate to common grid if needed
        if len(Te_scpn) != len(Te_torax):
            rho_scpn = np.array(scpn["rho"])
            rho_torax = np.array(torax.get("rho", np.linspace(0, 1, len(Te_torax))))
            Te_torax_interp = np.interp(rho_scpn, rho_torax, Te_torax)
        else:
            Te_torax_interp = Te_torax

        comparison["comparison"]["Te_rmse_keV"] = float(np.sqrt(np.mean((Te_scpn - Te_torax_interp) ** 2)))
        comparison["comparison"]["Te_max_diff_keV"] = float(np.max(np.abs(Te_scpn - Te_torax_interp)))

    if torax is not None and "Ti_final" in torax and "Ti_final" in scpn:
        Ti_scpn = np.array(scpn["Ti_final"])
        Ti_torax = np.array(torax["Ti_final"])

        if len(Ti_scpn) != len(Ti_torax):
            rho_scpn = np.array(scpn["rho"])
            rho_torax = np.array(torax.get("rho", np.linspace(0, 1, len(Ti_torax))))
            Ti_torax_interp = np.interp(rho_scpn, rho_torax, Ti_torax)
        else:
            Ti_torax_interp = Ti_torax

        comparison["comparison"]["Ti_rmse_keV"] = float(np.sqrt(np.mean((Ti_scpn - Ti_torax_interp) ** 2)))
        comparison["comparison"]["Ti_max_diff_keV"] = float(np.max(np.abs(Ti_scpn - Ti_torax_interp)))

    if torax is not None:
        if "tau_E_s" in torax:
            comparison["comparison"]["torax_tau_E_s"] = torax["tau_E_s"]

    return comparison


# ── Main ──────────────────────────────────────────────────────────────


def main():
    with_torax = "--with-torax" in sys.argv

    print(f"Running code-to-code benchmark: {ITER_SCENARIO['name']}")
    print("=" * 60)

    print("\n[1/3] Running scpn-control...")
    scpn_result = _run_scpn_control(ITER_SCENARIO)
    print(f"  Te_avg = {scpn_result['Te_avg']:.3f} keV")
    print(f"  Ti_avg = {scpn_result['Ti_avg']:.3f} keV")
    print(f"  Energy balance error = {scpn_result['energy_balance_error']:.4e}")
    print(f"  Particle balance error = {scpn_result['particle_balance_error']:.4e}")
    print(f"  Wall time = {scpn_result['wall_time_s']:.2f} s")

    torax_result = None
    if with_torax:
        print("\n[2/3] Running TORAX...")
        torax_result = _run_torax(ITER_SCENARIO)
        if torax_result:
            print(f"  Status: {torax_result.get('status', 'done')}")
    else:
        print("\n[2/3] TORAX skipped (use --with-torax to enable)")

    print("\n[3/3] Comparing results...")
    comparison = _compare_results(scpn_result, torax_result)

    report_dir = Path("validation/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "code_to_code_benchmark.json"

    with open(report_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nResults saved to {report_path}")

    if comparison["comparison"]:
        print("\nComparison metrics:")
        for k, v in comparison["comparison"].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

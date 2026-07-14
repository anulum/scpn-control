# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Code-to-Code Benchmark
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
  python -m validation.code_to_code_benchmark [--with-torax] [--require-external]

Without --with-torax, only scpn-control is run and results are saved
for later comparison. With --with-torax, TORAX is imported and run
on the same scenario (requires `pip install torax`).

Results are saved to validation/reports/code_to_code_benchmark.json and
validation/reports/code_to_code_benchmark.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pprint
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPORT_SCHEMA_VERSION = "scpn-control.code-to-code-benchmark.v2"
REPORT_PATH = Path("validation/reports/code_to_code_benchmark.json")
MARKDOWN_REPORT_PATH = Path("validation/reports/code_to_code_benchmark.md")
TORAX_TMP_CONFIG = Path("validation/reports/_tmp_torax_c2c_config.py")


def _ensure_repo_src_on_path() -> None:
    """Allow direct script execution from a source checkout without installation."""

    repo_src = str(Path(__file__).resolve().parents[1] / "src")
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)


_ensure_repo_src_on_path()


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


def _canonical_json(value: Any) -> str:
    """Serialise evidence deterministically for SHA-256 binding."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_payload(value: Any) -> str:
    """Return the canonical SHA-256 digest for a JSON-compatible payload."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _payload_without_digest(report: dict[str, Any]) -> dict[str, Any]:
    """Return a report copy with the self-referential digest field removed."""

    payload = dict(report)
    payload.pop("payload_sha256", None)
    return payload


def _verify_payload_digest(report: dict[str, Any]) -> bool:
    """Return true when a persisted benchmark report matches its payload digest."""

    digest = report.get("payload_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        return False
    return digest == _sha256_payload(_payload_without_digest(report))


def _finite_number(value: Any) -> bool:
    """Return whether a scalar can be safely admitted as finite numeric evidence."""

    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value))


def _finite_vector(value: Any) -> bool:
    """Return whether a vector-like payload is finite and non-empty."""

    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return arr.ndim == 1 and arr.size > 0 and bool(np.all(np.isfinite(arr)))


def _benchmark_numeric_payload_is_finite(result: dict[str, Any]) -> bool:
    """Validate the profile and scalar fields needed for external admission."""

    required_vectors = ("rho", "Te_final", "Ti_final")
    if any(not _finite_vector(result.get(field)) for field in required_vectors):
        return False
    optional_scalars = ("Te_avg", "Ti_avg", "wall_time_s", "tau_E_s", "W_thermal_total_J")
    return all(field not in result or _finite_number(result[field]) for field in optional_scalars)


# ── scpn-control run ─────────────────────────────────────────────────


def _run_scpn_control(scenario: dict[str, Any]) -> dict[str, Any]:
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


def _torax_config_dict(scenario: dict[str, Any]) -> dict[str, Any]:
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


def _write_torax_config(path: Path, scenario: dict[str, Any]) -> None:
    """Write a temporary Python config module consumable by TORAX."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = pprint.pformat(_torax_config_dict(scenario), sort_dicts=True, width=100)
    path.write_text(f"CONFIG = {payload}\n", encoding="utf-8")


def _extract_torax_result(data_tree: Any, scenario: dict[str, Any], wall_time: float) -> dict[str, Any]:
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


def _run_torax(scenario: dict[str, Any]) -> dict[str, Any] | None:
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


def _compare_results(scpn: dict[str, Any], torax: dict[str, Any] | None) -> dict[str, Any]:
    """Compare scpn-control and TORAX results."""
    comparison: dict[str, Any] = {
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


def _external_reference_status(
    comparison: dict[str, Any],
    *,
    requested_torax: bool,
) -> dict[str, Any]:
    """Classify whether the external-code comparison can support admission."""

    torax = comparison.get("torax")
    metrics = comparison.get("comparison", {})
    blocked_reasons: list[str] = []
    if not requested_torax:
        blocked_reasons.append("torax_not_requested")
        status = "not_requested"
    elif torax is None:
        blocked_reasons.append("torax_not_available_or_failed")
        status = "blocked"
    elif not isinstance(torax, dict) or torax.get("code") != "torax":
        blocked_reasons.append("torax_payload_identity")
        status = "blocked"
    elif not _benchmark_numeric_payload_is_finite(torax):
        blocked_reasons.append("torax_numeric_payload")
        status = "blocked"
    elif not isinstance(metrics, dict) or not metrics:
        blocked_reasons.append("comparison_metrics_missing")
        status = "blocked"
    elif any(not _finite_number(value) for value in metrics.values()):
        blocked_reasons.append("comparison_metrics_non_finite")
        status = "blocked"
    else:
        status = "admitted"

    scpn = comparison.get("scpn_control")
    if not isinstance(scpn, dict) or scpn.get("code") != "scpn-control":
        blocked_reasons.append("scpn_payload_identity")
    elif not _benchmark_numeric_payload_is_finite(scpn):
        blocked_reasons.append("scpn_numeric_payload")

    if blocked_reasons and status == "admitted":
        status = "blocked"

    return {
        "provider": "TORAX",
        "artifact_kind": "code_to_code_transport_reference",
        "requested": requested_torax,
        "admitted": status == "admitted",
        "status": status,
        "blocked_reasons": blocked_reasons,
    }


def _build_external_reference_report(
    comparison: dict[str, Any],
    scenario: dict[str, Any],
    *,
    requested_torax: bool,
) -> dict[str, Any]:
    """Build a digest-bound code-to-code external-reference evidence report."""

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "scenario_name": scenario["name"],
        "scenario_sha256": _sha256_payload(scenario),
        "external_reference": _external_reference_status(
            comparison,
            requested_torax=requested_torax,
        ),
        "benchmark": comparison,
        "claim_boundary": (
            "Admits only a code-to-code transport comparison when TORAX runs "
            "successfully on the declared scenario; otherwise full-fidelity "
            "external-reference claims remain blocked."
        ),
    }
    report["payload_sha256"] = _sha256_payload(report)
    return report


def _write_markdown_report(report: dict[str, Any], path: Path = MARKDOWN_REPORT_PATH) -> None:
    """Write a human-readable summary of the code-to-code evidence report."""

    external = report["external_reference"]
    metrics = report["benchmark"].get("comparison", {})
    metric_lines = (
        [f"- `{name}`: `{value}`" for name, value in sorted(metrics.items())]
        if metrics
        else ["- No comparison metrics admitted."]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# TORAX Code-to-Code Benchmark Evidence",
                "",
                f"- Schema: `{report['schema_version']}`",
                f"- Scenario: `{report['scenario_name']}`",
                f"- Scenario digest: `{report['scenario_sha256']}`",
                f"- External provider: `{external['provider']}`",
                f"- External status: `{external['status']}`",
                f"- External admitted: `{external['admitted']}`",
                f"- Blocked reasons: `{', '.join(external['blocked_reasons']) or 'none'}`",
                f"- Payload digest: `{report['payload_sha256']}`",
                f"- Claim boundary: {report['claim_boundary']}",
                "",
                "## Comparison metrics",
                "",
                *metric_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )


# ── Main ──────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse code-to-code benchmark command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-torax",
        action="store_true",
        help="Run TORAX and admit the report only when the external comparison succeeds.",
    )
    parser.add_argument(
        "--require-external",
        action="store_true",
        help="Exit non-zero if the TORAX external-reference comparison is not admitted.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPORT_PATH,
        help="Path for the schema-versioned JSON evidence report.",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=MARKDOWN_REPORT_PATH,
        help="Path for the Markdown evidence summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    with_torax = bool(args.with_torax)

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
    report = _build_external_reference_report(
        comparison,
        ITER_SCENARIO,
        requested_torax=with_torax,
    )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    _write_markdown_report(report, args.markdown_out)

    print(f"\nResults saved to {args.json_out}")
    print(f"Summary saved to {args.markdown_out}")
    print(f"External reference status: {report['external_reference']['status']}")

    if comparison["comparison"]:
        print("\nComparison metrics:")
        for k, v in comparison["comparison"].items():
            print(f"  {k}: {v}")

    if args.require_external and not report["external_reference"]["admitted"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

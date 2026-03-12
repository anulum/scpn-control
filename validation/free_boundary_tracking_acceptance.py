# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Free-Boundary Tracking Acceptance Campaign
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Deterministic real-kernel acceptance campaign for free-boundary control."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from scpn_control.control.free_boundary_tracking import run_free_boundary_tracking
from scpn_control.core.fusion_kernel import CoilSet, FusionKernel

ROOT = Path(__file__).resolve().parents[1]

NOMINAL_THRESHOLDS = {
    "max_final_tracking_error_norm": 0.02,
    "max_shape_rms": 0.015,
    "require_objective_converged": True,
}
KICK_THRESHOLDS = {
    "max_final_tracking_error_norm": 0.02,
    "max_shape_rms": 0.015,
    "max_abs_coil_current": 5.0e4,
    "require_objective_converged": True,
}
MEASUREMENT_THRESHOLDS = {
    "min_measured_true_gap": 0.02,
    "min_measurement_offset": 0.04,
    "max_true_tracking_error_norm": 0.02,
}
CORRECTED_THRESHOLDS = {
    "max_measured_true_gap": 1.0e-10,
    "max_measurement_offset": 1.0e-10,
    "max_final_tracking_error_norm": 0.02,
    "max_shape_rms": 0.015,
    "require_objective_converged": True,
}


def _write_tracking_config(path: Path, *, tracking_cfg: dict[str, Any] | None = None) -> Path:
    cfg: dict[str, Any] = {
        "reactor_name": "Free-Boundary-Acceptance",
        "grid_resolution": [12, 12],
        "dimensions": {"R_min": 2.0, "R_max": 6.0, "Z_min": -3.0, "Z_max": 3.0},
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        "coils": [
            {"name": "PF1", "r": 3.0, "z": 4.0, "current": 2.0},
            {"name": "PF2", "r": 5.0, "z": -4.0, "current": -1.0},
        ],
        "solver": {
            "max_iterations": 10,
            "convergence_threshold": 1e-3,
            "relaxation_factor": 0.15,
            "solver_method": "sor",
            "boundary_variant": "free_boundary",
        },
        "free_boundary": {
            "current_limits": [5.0e4, 5.0e4],
            "target_flux_points": [[3.5, 0.0], [4.0, 0.5]],
            "objective_tolerances": {"shape_rms": 0.25, "shape_max_abs": 0.35},
        },
    }
    if tracking_cfg is not None:
        cfg["free_boundary_tracking"] = tracking_cfg
    path.write_text(json.dumps(cfg), encoding="utf-8")

    kernel = FusionKernel(path)
    coils = kernel.build_coilset_from_config()
    kernel.solve_free_boundary(
        coils,
        max_outer_iter=2,
        tol=1.0e-2,
        optimize_shape=False,
    )
    flux_targets = kernel._sample_flux_at_points(coils.target_flux_points)
    cfg["free_boundary"]["target_flux_values"] = [float(v) for v in flux_targets]
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def _coil_kick_disturbance(kernel: FusionKernel, coils: CoilSet, step: int) -> None:
    del kernel
    if step != 1:
        return
    kick = np.array([2000.0, -1500.0], dtype=np.float64)
    limits = np.array([5.0e4, 5.0e4], dtype=np.float64)
    coils.currents = np.clip(np.asarray(coils.currents, dtype=np.float64) + kick, -limits, limits)


def _run_nominal(config_path: Path) -> dict[str, Any]:
    return run_free_boundary_tracking(
        config_file=str(config_path),
        shot_steps=4,
        gain=0.6,
        verbose=False,
        kernel_factory=FusionKernel,
        stop_on_convergence=False,
    )


def _run_kick(config_path: Path) -> dict[str, Any]:
    return run_free_boundary_tracking(
        config_file=str(config_path),
        shot_steps=4,
        gain=0.6,
        verbose=False,
        kernel_factory=FusionKernel,
        disturbance_callback=_coil_kick_disturbance,
        stop_on_convergence=False,
    )


def _run_measurement_fault(config_path: Path) -> dict[str, Any]:
    return run_free_boundary_tracking(
        config_file=str(config_path),
        shot_steps=4,
        gain=0.6,
        verbose=False,
        kernel_factory=FusionKernel,
        stop_on_convergence=False,
    )


def _evaluate_nominal(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "final_tracking_error_norm": bool(
            float(summary["final_tracking_error_norm"]) <= NOMINAL_THRESHOLDS["max_final_tracking_error_norm"]
        ),
        "shape_rms": bool(float(summary["shape_rms"]) <= NOMINAL_THRESHOLDS["max_shape_rms"]),
        "objective_converged": bool(summary["objective_converged"] is NOMINAL_THRESHOLDS["require_objective_converged"]),
    }
    return {
        "thresholds": NOMINAL_THRESHOLDS.copy(),
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _evaluate_kick(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "final_tracking_error_norm": bool(
            float(summary["final_tracking_error_norm"]) <= KICK_THRESHOLDS["max_final_tracking_error_norm"]
        ),
        "shape_rms": bool(float(summary["shape_rms"]) <= KICK_THRESHOLDS["max_shape_rms"]),
        "max_abs_coil_current": bool(float(summary["max_abs_coil_current"]) <= KICK_THRESHOLDS["max_abs_coil_current"]),
        "objective_converged": bool(summary["objective_converged"] is KICK_THRESHOLDS["require_objective_converged"]),
    }
    return {
        "thresholds": KICK_THRESHOLDS.copy(),
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _evaluate_measurement_fault(summary: dict[str, Any]) -> dict[str, Any]:
    measured_true_gap = abs(float(summary["final_tracking_error_norm"]) - float(summary["final_true_tracking_error_norm"]))
    checks = {
        "measured_true_gap": bool(measured_true_gap >= MEASUREMENT_THRESHOLDS["min_measured_true_gap"]),
        "measurement_offset": bool(float(summary["max_abs_measurement_offset"]) >= MEASUREMENT_THRESHOLDS["min_measurement_offset"]),
        "true_tracking_error_norm": bool(
            float(summary["final_true_tracking_error_norm"]) <= MEASUREMENT_THRESHOLDS["max_true_tracking_error_norm"]
        ),
    }
    return {
        "thresholds": MEASUREMENT_THRESHOLDS.copy(),
        "checks": checks,
        "passes_thresholds": all(checks.values()),
        "measured_true_gap": float(measured_true_gap),
    }


def _evaluate_corrected(summary: dict[str, Any]) -> dict[str, Any]:
    measured_true_gap = abs(float(summary["final_tracking_error_norm"]) - float(summary["final_true_tracking_error_norm"]))
    checks = {
        "measured_true_gap": bool(measured_true_gap <= CORRECTED_THRESHOLDS["max_measured_true_gap"]),
        "measurement_offset": bool(float(summary["max_abs_measurement_offset"]) <= CORRECTED_THRESHOLDS["max_measurement_offset"]),
        "final_tracking_error_norm": bool(
            float(summary["final_tracking_error_norm"]) <= CORRECTED_THRESHOLDS["max_final_tracking_error_norm"]
        ),
        "shape_rms": bool(float(summary["shape_rms"]) <= CORRECTED_THRESHOLDS["max_shape_rms"]),
        "objective_converged": bool(
            summary["objective_converged"] is CORRECTED_THRESHOLDS["require_objective_converged"]
        ),
    }
    return {
        "thresholds": CORRECTED_THRESHOLDS.copy(),
        "checks": checks,
        "passes_thresholds": all(checks.values()),
        "measured_true_gap": float(measured_true_gap),
    }


def run_campaign() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="scpn_free_boundary_acceptance_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        nominal_cfg = _write_tracking_config(tmp_path / "nominal.json")
        nominal = _run_nominal(nominal_cfg)

        kick_cfg = _write_tracking_config(tmp_path / "kick.json")
        kick = _run_kick(kick_cfg)

        measurement_cfg = _write_tracking_config(
            tmp_path / "measurement.json",
            tracking_cfg={
                "measurement_bias": {"shape_flux": [0.03, -0.02]},
                "measurement_drift_per_step": {"shape_flux": [0.004, -0.003]},
            },
        )
        measurement_fault = _run_measurement_fault(measurement_cfg)

        corrected_cfg = _write_tracking_config(
            tmp_path / "measurement_corrected.json",
            tracking_cfg={
                "measurement_bias": {"shape_flux": [0.03, -0.02]},
                "measurement_drift_per_step": {"shape_flux": [0.004, -0.003]},
                "measurement_correction_bias": {"shape_flux": [0.03, -0.02]},
                "measurement_correction_drift_per_step": {"shape_flux": [0.004, -0.003]},
            },
        )
        corrected = _run_measurement_fault(corrected_cfg)

    scenarios = {
        "nominal": {
            "summary": nominal,
            **_evaluate_nominal(nominal),
        },
        "coil_kick": {
            "summary": kick,
            **_evaluate_kick(kick),
        },
        "measurement_fault_uncorrected": {
            "summary": measurement_fault,
            **_evaluate_measurement_fault(measurement_fault),
        },
        "measurement_fault_corrected": {
            "summary": corrected,
            **_evaluate_corrected(corrected),
        },
    }
    passes_thresholds = all(bool(entry["passes_thresholds"]) for entry in scenarios.values())
    return {
        "benchmark": "free_boundary_tracking_acceptance",
        "steps_per_scenario": 4,
        "scenarios": scenarios,
        "passes_thresholds": passes_thresholds,
    }


def generate_report() -> dict[str, Any]:
    t0 = time.perf_counter()
    campaign = run_campaign()
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": float(time.perf_counter() - t0),
        "free_boundary_tracking_acceptance": campaign,
    }


def render_markdown(report: dict[str, Any]) -> str:
    campaign = report["free_boundary_tracking_acceptance"]
    lines = [
        "# Free-Boundary Tracking Acceptance",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{report['runtime_seconds']:.3f} s`",
        f"- Steps per scenario: `{campaign['steps_per_scenario']}`",
        f"- Pass: `{'YES' if campaign['passes_thresholds'] else 'NO'}`",
        "",
        "## Scenarios",
        "",
    ]
    for name, data in campaign["scenarios"].items():
        summary = data["summary"]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Pass: `{'YES' if data['passes_thresholds'] else 'NO'}`",
                f"- Final tracking error: `{summary['final_tracking_error_norm']:.6e}`",
                f"- Final true tracking error: `{summary['final_true_tracking_error_norm']:.6e}`",
                f"- Shape RMS: `{summary['shape_rms']:.6e}`",
                f"- Max coil current: `{summary['max_abs_coil_current']:.6e}`",
                f"- Max measurement offset: `{summary['max_abs_measurement_offset']:.6e}`",
                "",
            ]
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run free-boundary tracking acceptance campaign.")
    parser.add_argument(
        "--json-out",
        default=str(ROOT / "validation" / "reports" / "free_boundary_tracking_acceptance.json"),
        help="Path to write JSON report.",
    )
    parser.add_argument(
        "--md-out",
        default=str(ROOT / "validation" / "reports" / "free_boundary_tracking_acceptance.md"),
        help="Path to write Markdown report.",
    )
    args = parser.parse_args(argv)

    report = generate_report()
    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

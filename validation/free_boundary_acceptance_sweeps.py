# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Generic acceptance severity sweeps.

"""Run fixed shape-measurement, delay, slew and kick sweeps on the real kernel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scpn_control.control.free_boundary_tracking import run_free_boundary_tracking
from scpn_control.core.fusion_kernel import FusionKernel
from validation.free_boundary_acceptance_evaluations import (
    _is_monotone_non_decreasing,
    _is_monotone_non_increasing,
)
from validation.free_boundary_acceptance_presets import (
    ACTUATOR_SLEW_LIMIT_SWEEP,
    COIL_KICK_SCALE_SWEEP,
    CORRECTED_THRESHOLDS,
    LATENCY_CORRECTED_THRESHOLDS,
    LATENCY_STEP_SWEEP,
    LATENCY_THRESHOLDS,
    MEASUREMENT_SWEEP_SCALES,
    NOMINAL_THRESHOLDS,
    _latency_tracking_cfg,
    _make_coil_kick_disturbance,
    _measurement_tracking_cfg,
    _run_actuator_limited_kick,
    _run_latency_fault,
    _run_measurement_fault,
    _write_tracking_config,
)


def _run_measurement_sweep(tmp_path: Path, *, template_cfg: dict[str, Any]) -> dict[str, Any]:
    """Run four fixed bias scales on real tracking and check increasing measured/true gap and offset."""
    entries: list[dict[str, Any]] = []
    for scale in MEASUREMENT_SWEEP_SCALES:
        cfg = _write_tracking_config(
            tmp_path / f"measurement_sweep_{scale:.1f}.json",
            template_cfg=template_cfg,
            tracking_cfg=_measurement_tracking_cfg(scale, corrected=False),
        )
        summary = _run_measurement_fault(cfg)
        measured_true_gap = abs(
            float(summary["final_tracking_error_norm"]) - float(summary["final_true_tracking_error_norm"])
        )
        entries.append(
            {
                "scale": float(scale),
                "final_tracking_error_norm": float(summary["final_tracking_error_norm"]),
                "final_true_tracking_error_norm": float(summary["final_true_tracking_error_norm"]),
                "measured_true_gap": float(measured_true_gap),
                "max_abs_measurement_offset": float(summary["max_abs_measurement_offset"]),
                "shape_rms": float(summary["shape_rms"]),
                "true_shape_rms": float(summary["true_shape_rms"]),
            }
        )
    measured_true_gaps = [float(entry["measured_true_gap"]) for entry in entries]
    measurement_offsets = [float(entry["max_abs_measurement_offset"]) for entry in entries]
    checks = {
        "measured_true_gap_monotone": _is_monotone_non_decreasing(measured_true_gaps),
        "measurement_offset_monotone": _is_monotone_non_decreasing(measurement_offsets),
    }
    return {
        "entries": entries,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _run_corrected_measurement_sweep(tmp_path: Path, *, template_cfg: dict[str, Any]) -> dict[str, Any]:
    """Run four exact known-bias corrections and check collapsed gaps/offsets and constant tracking."""
    entries: list[dict[str, Any]] = []
    for scale in MEASUREMENT_SWEEP_SCALES:
        cfg = _write_tracking_config(
            tmp_path / f"measurement_corrected_sweep_{scale:.1f}.json",
            template_cfg=template_cfg,
            tracking_cfg=_measurement_tracking_cfg(scale, corrected=True),
        )
        summary = _run_measurement_fault(cfg)
        measured_true_gap = abs(
            float(summary["final_tracking_error_norm"]) - float(summary["final_true_tracking_error_norm"])
        )
        entries.append(
            {
                "scale": float(scale),
                "final_tracking_error_norm": float(summary["final_tracking_error_norm"]),
                "final_true_tracking_error_norm": float(summary["final_true_tracking_error_norm"]),
                "measured_true_gap": float(measured_true_gap),
                "max_abs_measurement_offset": float(summary["max_abs_measurement_offset"]),
                "shape_rms": float(summary["shape_rms"]),
                "true_shape_rms": float(summary["true_shape_rms"]),
            }
        )
    measured_true_gaps = [float(entry["measured_true_gap"]) for entry in entries]
    measurement_offsets = [float(entry["max_abs_measurement_offset"]) for entry in entries]
    final_tracking_error = [float(entry["final_tracking_error_norm"]) for entry in entries]
    checks = {
        "max_measured_true_gap": bool(max(measured_true_gaps) <= CORRECTED_THRESHOLDS["max_measured_true_gap"]),
        "max_measurement_offset": bool(max(measurement_offsets) <= CORRECTED_THRESHOLDS["max_measurement_offset"]),
        "tracking_error_constant": bool(
            max(final_tracking_error) - min(final_tracking_error) <= CORRECTED_THRESHOLDS["max_measured_true_gap"]
        ),
    }
    return {
        "entries": entries,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _run_latency_step_sweep(tmp_path: Path, *, template_cfg: dict[str, Any]) -> dict[str, Any]:
    """Run delays zero through three with real kicks and check delay activation, tracking and convergence."""
    entries: list[dict[str, Any]] = []
    for latency_steps in LATENCY_STEP_SWEEP:
        cfg = _write_tracking_config(
            tmp_path / f"latency_sweep_{latency_steps}.json",
            template_cfg=template_cfg,
            tracking_cfg=_latency_tracking_cfg(latency_steps, corrected=False),
        )
        summary = _run_latency_fault(cfg)
        entries.append(
            {
                "latency_steps": int(latency_steps),
                "delayed_observation_error_norm": float(summary["max_delayed_observation_error_norm"]),
                "estimated_observation_error_norm": float(summary["max_estimated_observation_error_norm"]),
                "final_true_tracking_error_norm": float(summary["final_true_tracking_error_norm"]),
                "objective_converged": bool(summary["objective_converged"]),
            }
        )
    delayed_error = [float(entry["delayed_observation_error_norm"]) for entry in entries]
    final_true_tracking_error = [float(entry["final_true_tracking_error_norm"]) for entry in entries]
    checks = {
        "delayed_observation_error_monotone": _is_monotone_non_decreasing(delayed_error, atol=1.0e-6),
        "delayed_observation_error_active": bool(
            max(delayed_error) >= LATENCY_THRESHOLDS["min_delayed_observation_error_norm"]
        ),
        "final_true_tracking_error_bounded": all(
            value <= LATENCY_THRESHOLDS["max_true_tracking_error_norm"] for value in final_true_tracking_error
        ),
        "objective_converged_all": all(bool(entry["objective_converged"]) for entry in entries),
    }
    return {
        "entries": entries,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _run_corrected_latency_step_sweep(tmp_path: Path, *, template_cfg: dict[str, Any]) -> dict[str, Any]:
    """Run four compensated delays and check estimated/true errors and objective convergence."""
    entries: list[dict[str, Any]] = []
    for latency_steps in LATENCY_STEP_SWEEP:
        cfg = _write_tracking_config(
            tmp_path / f"latency_corrected_sweep_{latency_steps}.json",
            template_cfg=template_cfg,
            tracking_cfg=_latency_tracking_cfg(latency_steps, corrected=True),
        )
        summary = _run_latency_fault(cfg)
        entries.append(
            {
                "latency_steps": int(latency_steps),
                "delayed_observation_error_norm": float(summary["max_delayed_observation_error_norm"]),
                "estimated_observation_error_norm": float(summary["max_estimated_observation_error_norm"]),
                "final_true_tracking_error_norm": float(summary["final_true_tracking_error_norm"]),
                "objective_converged": bool(summary["objective_converged"]),
            }
        )
    estimated_error = [float(entry["estimated_observation_error_norm"]) for entry in entries]
    final_true_tracking_error = [float(entry["final_true_tracking_error_norm"]) for entry in entries]
    checks = {
        "estimated_observation_error_bounded": all(
            value <= LATENCY_CORRECTED_THRESHOLDS["max_estimated_observation_error_norm"] for value in estimated_error
        ),
        "final_true_tracking_error_bounded": all(
            value <= LATENCY_CORRECTED_THRESHOLDS["max_final_true_tracking_error_norm"]
            for value in final_true_tracking_error
        ),
        "objective_converged_all": all(bool(entry["objective_converged"]) for entry in entries),
    }
    return {
        "entries": entries,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _run_actuator_slew_sweep(tmp_path: Path, *, template_cfg: dict[str, Any]) -> dict[str, Any]:
    """Run five decreasing slew limits and check increasing lag and decreasing maximum current."""
    entries: list[dict[str, Any]] = []
    for slew_limit in ACTUATOR_SLEW_LIMIT_SWEEP:
        cfg = _write_tracking_config(
            tmp_path / f"actuator_slew_{slew_limit}.json",
            template_cfg=template_cfg,
        )
        summary = _run_actuator_limited_kick(cfg, coil_slew_limits=float(slew_limit))
        entries.append(
            {
                "coil_slew_limit": float(slew_limit),
                "max_abs_actuator_lag": float(summary["max_abs_actuator_lag"]),
                "mean_abs_actuator_lag": float(summary["mean_abs_actuator_lag"]),
                "max_abs_coil_current": float(summary["max_abs_coil_current"]),
                "final_tracking_error_norm": float(summary["final_tracking_error_norm"]),
            }
        )
    max_lag = [float(entry["max_abs_actuator_lag"]) for entry in entries]
    mean_lag = [float(entry["mean_abs_actuator_lag"]) for entry in entries]
    max_coil_current = [float(entry["max_abs_coil_current"]) for entry in entries]
    checks = {
        "max_abs_actuator_lag_monotone": _is_monotone_non_decreasing(max_lag),
        "mean_abs_actuator_lag_monotone": _is_monotone_non_decreasing(mean_lag),
        "max_abs_coil_current_monotone": _is_monotone_non_increasing(max_coil_current),
    }
    return {
        "entries": entries,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _run_coil_kick_scale_sweep(tmp_path: Path, *, template_cfg: dict[str, Any]) -> dict[str, Any]:
    """Run six increasing kicks and check current authority, bounded tracking and convergence."""
    entries: list[dict[str, Any]] = []
    for scale in COIL_KICK_SCALE_SWEEP:
        cfg = _write_tracking_config(
            tmp_path / f"coil_kick_scale_{scale:.1f}.json",
            template_cfg=template_cfg,
        )
        summary = run_free_boundary_tracking(
            config_file=str(cfg),
            shot_steps=4,
            gain=0.6,
            verbose=False,
            kernel_factory=FusionKernel,
            disturbance_callback=_make_coil_kick_disturbance(scale),
            stop_on_convergence=False,
        )
        entries.append(
            {
                "kick_scale": float(scale),
                "final_tracking_error_norm": float(summary["final_tracking_error_norm"]),
                "mean_tracking_error_norm": float(summary["mean_tracking_error_norm"]),
                "shape_rms": float(summary["shape_rms"]),
                "max_abs_coil_current": float(summary["max_abs_coil_current"]),
                "objective_converged": bool(summary["objective_converged"]),
            }
        )
    max_coil_current = [float(entry["max_abs_coil_current"]) for entry in entries]
    final_tracking_error = [float(entry["final_tracking_error_norm"]) for entry in entries]
    objective_converged = [bool(entry["objective_converged"]) for entry in entries]
    checks = {
        "max_abs_coil_current_monotone": _is_monotone_non_decreasing(max_coil_current),
        "objective_converged_all": all(objective_converged),
        "final_tracking_error_bounded": max(final_tracking_error)
        <= NOMINAL_THRESHOLDS["max_final_tracking_error_norm"],
    }
    return {
        "entries": entries,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }

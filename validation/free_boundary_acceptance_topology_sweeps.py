# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Topology acceptance severity sweeps.

"""Run fixed X-point/divertor kick, slew, calibration and supervisor sweeps.

All entries use the actual four-step tracking API and same-model targets.
Known-error corrections and monotonicity do not admit independent physics.
"""

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
    MEASUREMENT_SWEEP_SCALES,
    SUPERVISOR_FALLBACK_THRESHOLDS,
    TOPOLOGY_CORRECTED_THRESHOLDS,
    TOPOLOGY_KICK_SCALE_SWEEP,
    TOPOLOGY_THRESHOLDS,
    _make_coil_kick_disturbance,
    _run_measurement_fault,
    _run_supervisor_fallback_kick,
    _topology_measurement_tracking_cfg,
    _write_tracking_config,
)


def _run_topology_kick_scale_sweep(tmp_path: Path, *, topology_template_cfg: dict[str, Any]) -> dict[str, Any]:
    """Run three topology kicks and check monotone flux residuals with bounded topology errors."""
    entries: list[dict[str, Any]] = []
    for scale in TOPOLOGY_KICK_SCALE_SWEEP:
        cfg = _write_tracking_config(
            tmp_path / f"topology_kick_scale_{scale:.1f}.json",
            template_cfg=topology_template_cfg,
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
                "x_point_position_error": float(summary["x_point_position_error"]),
                "x_point_flux_error": float(summary["x_point_flux_error"]),
                "divertor_rms": float(summary["divertor_rms"]),
                "divertor_max_abs": float(summary["divertor_max_abs"]),
                "objective_converged": bool(summary["objective_converged"]),
                "max_abs_coil_current": float(summary["max_abs_coil_current"]),
            }
        )
    max_coil_current = [float(entry["max_abs_coil_current"]) for entry in entries]
    x_point_flux_error = [float(entry["x_point_flux_error"]) for entry in entries]
    divertor_rms = [float(entry["divertor_rms"]) for entry in entries]
    divertor_max_abs = [float(entry["divertor_max_abs"]) for entry in entries]
    objective_converged = [bool(entry["objective_converged"]) for entry in entries]
    checks = {
        "max_abs_coil_current_monotone": _is_monotone_non_decreasing(max_coil_current),
        "x_point_flux_error_monotone": _is_monotone_non_decreasing(x_point_flux_error),
        "divertor_rms_monotone": _is_monotone_non_decreasing(divertor_rms),
        "divertor_max_abs_monotone": _is_monotone_non_decreasing(divertor_max_abs),
        "objective_converged_all": all(objective_converged),
        "topology_errors_bounded": bool(
            max(float(entry["final_tracking_error_norm"]) for entry in entries)
            <= TOPOLOGY_THRESHOLDS["max_final_tracking_error_norm"]
            and max(float(entry["x_point_position_error"]) for entry in entries)
            <= TOPOLOGY_THRESHOLDS["max_x_point_position_error"]
            and max(float(entry["x_point_flux_error"]) for entry in entries)
            <= TOPOLOGY_THRESHOLDS["max_x_point_flux_error"]
            and max(float(entry["divertor_rms"]) for entry in entries) <= TOPOLOGY_THRESHOLDS["max_divertor_rms"]
            and max(float(entry["divertor_max_abs"]) for entry in entries)
            <= TOPOLOGY_THRESHOLDS["max_divertor_max_abs"]
        ),
    }
    return {
        "entries": entries,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _run_topology_actuator_slew_sweep(tmp_path: Path, *, topology_template_cfg: dict[str, Any]) -> dict[str, Any]:
    """Run five topology slew limits and check lag/error tradeoffs plus current and topology bounds."""
    entries: list[dict[str, Any]] = []
    for slew_limit in ACTUATOR_SLEW_LIMIT_SWEEP:
        cfg = _write_tracking_config(
            tmp_path / f"topology_actuator_slew_{slew_limit}.json",
            template_cfg=topology_template_cfg,
        )
        summary = run_free_boundary_tracking(
            config_file=str(cfg),
            shot_steps=4,
            gain=8.0,
            verbose=False,
            kernel_factory=FusionKernel,
            disturbance_callback=_make_coil_kick_disturbance(2.0),
            control_dt_s=0.1,
            coil_actuator_tau_s=0.05,
            coil_slew_limits=float(slew_limit),
            stop_on_convergence=False,
        )
        entries.append(
            {
                "coil_slew_limit": float(slew_limit),
                "max_abs_actuator_lag": float(summary["max_abs_actuator_lag"]),
                "x_point_flux_error": float(summary["x_point_flux_error"]),
                "divertor_rms": float(summary["divertor_rms"]),
                "divertor_max_abs": float(summary["divertor_max_abs"]),
                "objective_converged": bool(summary["objective_converged"]),
                "max_abs_coil_current": float(summary["max_abs_coil_current"]),
                "final_tracking_error_norm": float(summary["final_tracking_error_norm"]),
            }
        )
    max_abs_actuator_lag = [float(entry["max_abs_actuator_lag"]) for entry in entries]
    x_point_flux_error = [float(entry["x_point_flux_error"]) for entry in entries]
    divertor_rms = [float(entry["divertor_rms"]) for entry in entries]
    divertor_max_abs = [float(entry["divertor_max_abs"]) for entry in entries]
    max_abs_coil_current = [float(entry["max_abs_coil_current"]) for entry in entries]
    objective_converged = [bool(entry["objective_converged"]) for entry in entries]
    checks = {
        "max_abs_actuator_lag_monotone": _is_monotone_non_decreasing(max_abs_actuator_lag),
        "x_point_flux_error_monotone": _is_monotone_non_decreasing(x_point_flux_error),
        "divertor_rms_monotone": _is_monotone_non_decreasing(divertor_rms),
        "divertor_max_abs_monotone": _is_monotone_non_decreasing(divertor_max_abs),
        "max_abs_coil_current_monotone": _is_monotone_non_increasing(max_abs_coil_current),
        "objective_converged_all": all(objective_converged),
        "topology_errors_bounded": bool(
            max(float(entry["final_tracking_error_norm"]) for entry in entries)
            <= TOPOLOGY_THRESHOLDS["max_final_tracking_error_norm"]
            and max(float(entry["x_point_flux_error"]) for entry in entries)
            <= TOPOLOGY_THRESHOLDS["max_x_point_flux_error"]
            and max(float(entry["divertor_rms"]) for entry in entries) <= TOPOLOGY_THRESHOLDS["max_divertor_rms"]
            and max(float(entry["divertor_max_abs"]) for entry in entries)
            <= TOPOLOGY_THRESHOLDS["max_divertor_max_abs"]
        ),
    }
    return {
        "entries": entries,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _run_topology_measurement_sweep(tmp_path: Path, *, topology_template_cfg: dict[str, Any]) -> dict[str, Any]:
    """Run four topology bias scales and check increasing observed/true residual gaps."""
    entries: list[dict[str, Any]] = []
    for scale in MEASUREMENT_SWEEP_SCALES:
        cfg = _write_tracking_config(
            tmp_path / f"topology_measurement_scale_{scale:.1f}.json",
            template_cfg=topology_template_cfg,
            tracking_cfg=None if scale == 0.0 else _topology_measurement_tracking_cfg(scale, corrected=False),
        )
        summary = _run_measurement_fault(cfg)
        entries.append(
            {
                "scale": float(scale),
                "x_point_position_gap": abs(
                    float(summary["x_point_position_error"]) - float(summary["true_x_point_position_error"])
                ),
                "x_point_flux_gap": abs(
                    float(summary["x_point_flux_error"]) - float(summary["true_x_point_flux_error"])
                ),
                "divertor_rms_gap": abs(float(summary["divertor_rms"]) - float(summary["true_divertor_rms"])),
                "divertor_max_abs_gap": abs(
                    float(summary["divertor_max_abs"]) - float(summary["true_divertor_max_abs"])
                ),
                "max_abs_measurement_offset": float(summary["max_abs_measurement_offset"]),
                "objective_converged": bool(summary["objective_converged"]),
            }
        )
    x_point_position_gap = [float(entry["x_point_position_gap"]) for entry in entries]
    x_point_flux_gap = [float(entry["x_point_flux_gap"]) for entry in entries]
    divertor_rms_gap = [float(entry["divertor_rms_gap"]) for entry in entries]
    divertor_max_abs_gap = [float(entry["divertor_max_abs_gap"]) for entry in entries]
    measurement_offset = [float(entry["max_abs_measurement_offset"]) for entry in entries]
    checks = {
        "x_point_position_gap_monotone": _is_monotone_non_decreasing(x_point_position_gap),
        "x_point_flux_gap_monotone": _is_monotone_non_decreasing(x_point_flux_gap),
        "divertor_rms_gap_monotone": _is_monotone_non_decreasing(divertor_rms_gap),
        "divertor_max_abs_gap_monotone": _is_monotone_non_decreasing(divertor_max_abs_gap),
        "measurement_offset_monotone": _is_monotone_non_decreasing(measurement_offset),
    }
    return {
        "entries": entries,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _run_topology_corrected_measurement_sweep(
    tmp_path: Path, *, topology_template_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Run four known topology corrections and check collapsed gaps, offset and convergence."""
    entries: list[dict[str, Any]] = []
    for scale in MEASUREMENT_SWEEP_SCALES:
        cfg = _write_tracking_config(
            tmp_path / f"topology_measurement_corrected_scale_{scale:.1f}.json",
            template_cfg=topology_template_cfg,
            tracking_cfg=_topology_measurement_tracking_cfg(scale, corrected=True),
        )
        summary = _run_measurement_fault(cfg)
        entries.append(
            {
                "scale": float(scale),
                "x_point_position_gap": abs(
                    float(summary["x_point_position_error"]) - float(summary["true_x_point_position_error"])
                ),
                "x_point_flux_gap": abs(
                    float(summary["x_point_flux_error"]) - float(summary["true_x_point_flux_error"])
                ),
                "divertor_rms_gap": abs(float(summary["divertor_rms"]) - float(summary["true_divertor_rms"])),
                "divertor_max_abs_gap": abs(
                    float(summary["divertor_max_abs"]) - float(summary["true_divertor_max_abs"])
                ),
                "max_abs_measurement_offset": float(summary["max_abs_measurement_offset"]),
                "objective_converged": bool(summary["objective_converged"]),
            }
        )
    checks = {
        "max_x_point_position_gap": bool(
            max(float(entry["x_point_position_gap"]) for entry in entries)
            <= TOPOLOGY_CORRECTED_THRESHOLDS["max_x_point_position_gap"]
        ),
        "max_x_point_flux_gap": bool(
            max(float(entry["x_point_flux_gap"]) for entry in entries)
            <= TOPOLOGY_CORRECTED_THRESHOLDS["max_x_point_flux_gap"]
        ),
        "max_divertor_rms_gap": bool(
            max(float(entry["divertor_rms_gap"]) for entry in entries)
            <= TOPOLOGY_CORRECTED_THRESHOLDS["max_divertor_rms_gap"]
        ),
        "max_divertor_max_abs_gap": bool(
            max(float(entry["divertor_max_abs_gap"]) for entry in entries)
            <= TOPOLOGY_CORRECTED_THRESHOLDS["max_divertor_max_abs_gap"]
        ),
        "max_measurement_offset": bool(
            max(float(entry["max_abs_measurement_offset"]) for entry in entries)
            <= TOPOLOGY_CORRECTED_THRESHOLDS["max_measurement_offset"]
        ),
        "objective_converged_all": all(bool(entry["objective_converged"]) for entry in entries),
    }
    return {
        "entries": entries,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _run_topology_supervisor_measurement_sweep(
    tmp_path: Path, *, topology_template_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Run four biased supervisor fixtures and check fault visibility, interventions and bounded lag."""
    entries: list[dict[str, Any]] = []
    for scale in MEASUREMENT_SWEEP_SCALES:
        tracking_cfg: dict[str, Any] = {
            "fallback_currents": [0.0, 0.0],
            "supervisor_limits": {"max_abs_actuator_lag": 2.0},
            "hold_steps_after_reject": 2,
        }
        if scale != 0.0:
            tracking_cfg.update(_topology_measurement_tracking_cfg(scale, corrected=False))
        cfg = _write_tracking_config(
            tmp_path / f"topology_supervisor_measurement_scale_{scale:.1f}.json",
            template_cfg=topology_template_cfg,
            tracking_cfg=tracking_cfg,
        )
        summary = _run_supervisor_fallback_kick(cfg)
        entries.append(
            {
                "scale": float(scale),
                "x_point_position_gap": abs(
                    float(summary["x_point_position_error"]) - float(summary["true_x_point_position_error"])
                ),
                "x_point_flux_gap": abs(
                    float(summary["x_point_flux_error"]) - float(summary["true_x_point_flux_error"])
                ),
                "divertor_rms_gap": abs(float(summary["divertor_rms"]) - float(summary["true_divertor_rms"])),
                "divertor_max_abs_gap": abs(
                    float(summary["divertor_max_abs"]) - float(summary["true_divertor_max_abs"])
                ),
                "max_abs_measurement_offset": float(summary["max_abs_measurement_offset"]),
                "objective_converged": bool(summary["objective_converged"]),
                "supervisor_active": bool(summary["supervisor_active"]),
                "supervisor_safe": bool(summary["supervisor_safe"]),
                "supervisor_intervention_count": int(summary["supervisor_intervention_count"]),
                "fallback_active_steps": int(summary["fallback_active_steps"]),
                "max_abs_actuator_lag": float(summary["max_abs_actuator_lag"]),
            }
        )
    checks = {
        "x_point_position_gap_monotone": _is_monotone_non_decreasing(
            [float(entry["x_point_position_gap"]) for entry in entries]
        ),
        "x_point_flux_gap_monotone": _is_monotone_non_decreasing(
            [float(entry["x_point_flux_gap"]) for entry in entries]
        ),
        "divertor_rms_gap_monotone": _is_monotone_non_decreasing(
            [float(entry["divertor_rms_gap"]) for entry in entries]
        ),
        "divertor_max_abs_gap_monotone": _is_monotone_non_decreasing(
            [float(entry["divertor_max_abs_gap"]) for entry in entries]
        ),
        "measurement_offset_monotone": _is_monotone_non_decreasing(
            [float(entry["max_abs_measurement_offset"]) for entry in entries]
        ),
        "supervisor_active_all": all(bool(entry["supervisor_active"]) for entry in entries),
        "supervisor_safe_all": all(bool(entry["supervisor_safe"]) for entry in entries),
        "intervention_all": all(int(entry["supervisor_intervention_count"]) >= 1 for entry in entries),
        "fallback_all": all(int(entry["fallback_active_steps"]) >= 1 for entry in entries),
        "lag_bounded_all": all(
            float(entry["max_abs_actuator_lag"]) <= SUPERVISOR_FALLBACK_THRESHOLDS["max_abs_actuator_lag"]
            for entry in entries
        ),
    }
    return {
        "entries": entries,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _run_topology_supervisor_corrected_measurement_sweep(
    tmp_path: Path, *, topology_template_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Run four known-corrected supervisor fixtures and check collapsed gaps, intervention and bounded lag."""
    entries: list[dict[str, Any]] = []
    for scale in MEASUREMENT_SWEEP_SCALES:
        cfg = _write_tracking_config(
            tmp_path / f"topology_supervisor_measurement_corrected_scale_{scale:.1f}.json",
            template_cfg=topology_template_cfg,
            tracking_cfg={
                "fallback_currents": [0.0, 0.0],
                "supervisor_limits": {"max_abs_actuator_lag": 2.0},
                "hold_steps_after_reject": 2,
                **_topology_measurement_tracking_cfg(scale, corrected=True),
            },
        )
        summary = _run_supervisor_fallback_kick(cfg)
        entries.append(
            {
                "scale": float(scale),
                "x_point_position_gap": abs(
                    float(summary["x_point_position_error"]) - float(summary["true_x_point_position_error"])
                ),
                "x_point_flux_gap": abs(
                    float(summary["x_point_flux_error"]) - float(summary["true_x_point_flux_error"])
                ),
                "divertor_rms_gap": abs(float(summary["divertor_rms"]) - float(summary["true_divertor_rms"])),
                "divertor_max_abs_gap": abs(
                    float(summary["divertor_max_abs"]) - float(summary["true_divertor_max_abs"])
                ),
                "max_abs_measurement_offset": float(summary["max_abs_measurement_offset"]),
                "objective_converged": bool(summary["objective_converged"]),
                "supervisor_active": bool(summary["supervisor_active"]),
                "supervisor_safe": bool(summary["supervisor_safe"]),
                "supervisor_intervention_count": int(summary["supervisor_intervention_count"]),
                "fallback_active_steps": int(summary["fallback_active_steps"]),
                "max_abs_actuator_lag": float(summary["max_abs_actuator_lag"]),
            }
        )
    checks = {
        "max_x_point_position_gap": bool(
            max(float(entry["x_point_position_gap"]) for entry in entries)
            <= TOPOLOGY_CORRECTED_THRESHOLDS["max_x_point_position_gap"]
        ),
        "max_x_point_flux_gap": bool(
            max(float(entry["x_point_flux_gap"]) for entry in entries)
            <= TOPOLOGY_CORRECTED_THRESHOLDS["max_x_point_flux_gap"]
        ),
        "max_divertor_rms_gap": bool(
            max(float(entry["divertor_rms_gap"]) for entry in entries)
            <= TOPOLOGY_CORRECTED_THRESHOLDS["max_divertor_rms_gap"]
        ),
        "max_divertor_max_abs_gap": bool(
            max(float(entry["divertor_max_abs_gap"]) for entry in entries)
            <= TOPOLOGY_CORRECTED_THRESHOLDS["max_divertor_max_abs_gap"]
        ),
        "max_measurement_offset": bool(
            max(float(entry["max_abs_measurement_offset"]) for entry in entries)
            <= TOPOLOGY_CORRECTED_THRESHOLDS["max_measurement_offset"]
        ),
        "objective_converged_all": all(bool(entry["objective_converged"]) for entry in entries),
        "supervisor_active_all": all(bool(entry["supervisor_active"]) for entry in entries),
        "supervisor_safe_all": all(bool(entry["supervisor_safe"]) for entry in entries),
        "intervention_all": all(int(entry["supervisor_intervention_count"]) >= 1 for entry in entries),
        "fallback_all": all(int(entry["fallback_active_steps"]) >= 1 for entry in entries),
        "lag_bounded_all": all(
            float(entry["max_abs_actuator_lag"]) <= SUPERVISOR_FALLBACK_THRESHOLDS["max_abs_actuator_lag"]
            for entry in entries
        ),
    }
    return {
        "entries": entries,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }

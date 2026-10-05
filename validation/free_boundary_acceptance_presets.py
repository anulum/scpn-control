# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Free-boundary acceptance presets.

"""Construct fixed normalized free-boundary fixtures and real tracking invocations.

Permeability/current target are 1.0 on a 12-by-12 grid. Targets are sampled
from the same FusionKernel; exact supplied measurement errors are subtracted
in corrected cases. Limits/scales remain the original mutable declarations.
Coordinates use the model's metre labels, currents its configured convention,
and flux residuals its normalization, not independently calibrated SI flux.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from scpn_control.control.free_boundary_tracking import run_free_boundary_tracking
from scpn_control.core.fusion_kernel import CoilSet, FusionKernel

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


LATENCY_THRESHOLDS = {
    "min_delayed_observation_error_norm": 0.005,
    "max_true_tracking_error_norm": 0.02,
}


LATENCY_CORRECTED_THRESHOLDS = {
    "max_estimated_observation_error_norm": 0.01,
    "max_final_true_tracking_error_norm": 0.02,
    "require_objective_converged": True,
}


TOPOLOGY_THRESHOLDS = {
    "max_final_tracking_error_norm": 0.02,
    "max_shape_rms": 0.015,
    "max_x_point_position_error": 0.1,
    "max_x_point_flux_error": 0.01,
    "max_divertor_rms": 0.01,
    "max_divertor_max_abs": 0.01,
    "require_objective_converged": True,
}


TOPOLOGY_MEASUREMENT_THRESHOLDS = {
    "min_x_point_position_gap": 0.1,
    "min_x_point_flux_gap": 0.03,
    "min_divertor_rms_gap": 0.03,
    "min_divertor_max_abs_gap": 0.04,
    "min_measurement_offset": 0.08,
    "max_true_tracking_error_norm": 0.02,
    "max_true_x_point_position_error": TOPOLOGY_THRESHOLDS["max_x_point_position_error"],
    "max_true_x_point_flux_error": TOPOLOGY_THRESHOLDS["max_x_point_flux_error"],
    "max_true_divertor_rms": TOPOLOGY_THRESHOLDS["max_divertor_rms"],
    "max_true_divertor_max_abs": TOPOLOGY_THRESHOLDS["max_divertor_max_abs"],
    "require_objective_converged": False,
}


TOPOLOGY_LATENCY_MEASUREMENT_THRESHOLDS = {
    **TOPOLOGY_MEASUREMENT_THRESHOLDS,
    "min_divertor_max_abs_gap": 0.039,
}


TOPOLOGY_CORRECTED_THRESHOLDS = {
    "max_x_point_position_gap": 1.0e-10,
    "max_x_point_flux_gap": 1.0e-10,
    "max_divertor_rms_gap": 1.0e-10,
    "max_divertor_max_abs_gap": 1.0e-10,
    "max_measurement_offset": 1.0e-10,
    "max_final_tracking_error_norm": TOPOLOGY_THRESHOLDS["max_final_tracking_error_norm"],
    "max_x_point_position_error": TOPOLOGY_THRESHOLDS["max_x_point_position_error"],
    "max_x_point_flux_error": TOPOLOGY_THRESHOLDS["max_x_point_flux_error"],
    "max_divertor_rms": TOPOLOGY_THRESHOLDS["max_divertor_rms"],
    "max_divertor_max_abs": TOPOLOGY_THRESHOLDS["max_divertor_max_abs"],
    "require_objective_converged": True,
}


TOPOLOGY_SUPERVISOR_LATENCY_CORRECTED_THRESHOLDS = {
    **TOPOLOGY_CORRECTED_THRESHOLDS,
    "max_x_point_flux_gap": 1.0e-7,
    "max_divertor_rms_gap": 1.0e-7,
    "max_divertor_max_abs_gap": 1.0e-7,
    "max_estimated_observation_error_norm": 0.03,
    "max_final_true_tracking_error_norm": 0.02,
}


SUPERVISOR_FALLBACK_THRESHOLDS = {
    "min_supervisor_intervention_count": 1,
    "min_fallback_active_steps": 1,
    "max_abs_actuator_lag": 2.0,
    "min_lag_reduction_factor": 100.0,
    "max_final_tracking_error_norm": 0.02,
    "max_x_point_position_error": TOPOLOGY_THRESHOLDS["max_x_point_position_error"],
    "max_x_point_flux_error": TOPOLOGY_THRESHOLDS["max_x_point_flux_error"],
    "max_divertor_rms": TOPOLOGY_THRESHOLDS["max_divertor_rms"],
    "max_divertor_max_abs": TOPOLOGY_THRESHOLDS["max_divertor_max_abs"],
    "require_supervisor_active": True,
    "require_supervisor_safe": True,
    "require_objective_converged": True,
}


MEASUREMENT_SWEEP_SCALES = (0.0, 0.5, 1.0, 1.5)


LATENCY_STEP_SWEEP = (0, 1, 2, 3)


ACTUATOR_SLEW_LIMIT_SWEEP = (1.0e3, 1.0e2, 1.0e1, 1.0, 0.1)


COIL_KICK_SCALE_SWEEP = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)


TOPOLOGY_KICK_SCALE_SWEEP = (1.0, 2.0, 4.0)


TOPOLOGY_DIVERTOR_STRIKE_POINTS = (
    (3.2, -2.0),
    (4.8, -2.0),
)


def _base_tracking_config() -> dict[str, Any]:
    """Return fresh normalized 12-by-12 SOR fixtures with permeability/current target 1."""
    return {
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


def _build_tracking_template(tmp_path: Path) -> dict[str, Any]:
    """Write a temporary fixture and sample shape targets from its own two-outer-step solver."""
    cfg = _base_tracking_config()
    template_path = tmp_path / "template.json"
    template_path.write_text(json.dumps(cfg), encoding="utf-8")
    kernel = FusionKernel(template_path)
    coils = kernel.build_coilset_from_config()
    kernel.solve_free_boundary(
        coils,
        max_outer_iter=2,
        tol=1.0e-2,
        optimize_shape=False,
    )
    target_flux_points = coils.target_flux_points
    if target_flux_points is None:
        raise ValueError("coilset produced no target flux points to sample")
    flux_targets = kernel._sample_flux_at_points(target_flux_points)
    cfg["free_boundary"]["target_flux_values"] = [float(v) for v in flux_targets]
    return cfg


def _build_topology_tracking_template(tmp_path: Path, *, template_cfg: dict[str, Any]) -> dict[str, Any]:
    """Copy the template and sample X-point/divertor targets from the same real kernel."""
    cfg = deepcopy(template_cfg)
    topology_path = tmp_path / "topology_template.json"
    topology_path.write_text(json.dumps(cfg), encoding="utf-8")
    kernel = FusionKernel(topology_path)
    coils = kernel.build_coilset_from_config()
    kernel.solve_free_boundary(
        coils,
        max_outer_iter=2,
        tol=1.0e-2,
        optimize_shape=False,
    )
    x_pos, _ = kernel.find_x_point(kernel.Psi)
    x_target = np.asarray(x_pos, dtype=np.float64).reshape(2)
    divertor_points = np.asarray(TOPOLOGY_DIVERTOR_STRIKE_POINTS, dtype=np.float64)
    objective_tolerances = deepcopy(cfg["free_boundary"].get("objective_tolerances", {}))
    objective_tolerances.update(
        {
            "x_point_position": TOPOLOGY_THRESHOLDS["max_x_point_position_error"],
            "x_point_flux": TOPOLOGY_THRESHOLDS["max_x_point_flux_error"],
            "divertor_rms": TOPOLOGY_THRESHOLDS["max_divertor_rms"],
            "divertor_max_abs": TOPOLOGY_THRESHOLDS["max_divertor_max_abs"],
        }
    )
    cfg["free_boundary"].update(
        {
            "x_point_target": [float(x_target[0]), float(x_target[1])],
            "x_point_flux_target": float(kernel._interp_psi(float(x_target[0]), float(x_target[1]))),
            "divertor_strike_points": divertor_points.tolist(),
            "divertor_flux_values": [float(v) for v in kernel._sample_flux_at_points(divertor_points)],
            "objective_tolerances": objective_tolerances,
        }
    )
    return cfg


def _write_tracking_config(
    path: Path,
    *,
    template_cfg: dict[str, Any],
    tracking_cfg: dict[str, Any] | None = None,
) -> Path:
    """Copy template metadata, add tracking fields when supplied, and replace the given JSON path."""
    cfg = deepcopy(template_cfg)
    if tracking_cfg is not None:
        cfg["free_boundary_tracking"] = tracking_cfg
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def _merge_tracking_cfg(*parts: dict[str, Any] | None) -> dict[str, Any] | None:
    """Merge truthy optional dictionaries left to right; later keys replace earlier values."""
    merged: dict[str, Any] = {}
    for part in parts:
        if part:
            merged.update(part)
    return merged or None


def _make_coil_kick_disturbance(scale: float = 1.0) -> Callable[[FusionKernel, CoilSet, int], None]:
    """Build a two-coil step-one kick callback with declared +/-50000 current clipping."""
    kick = np.array([2000.0, -1500.0], dtype=np.float64) * float(scale)
    limits = np.array([5.0e4, 5.0e4], dtype=np.float64)

    def disturbance(kernel: FusionKernel, coils: CoilSet, step: int) -> None:
        """Apply the declared kick only at step one; mutate currents in the supplied real coilset."""
        del kernel
        if step != 1:
            return
        coils.currents = np.clip(np.asarray(coils.currents, dtype=np.float64) + kick, -limits, limits)

    return disturbance


def _run_nominal(config_path: Path) -> dict[str, Any]:
    """Execute all four real tracking steps at gain 0.6, without early convergence stopping."""
    return run_free_boundary_tracking(
        config_file=str(config_path),
        shot_steps=4,
        gain=0.6,
        verbose=False,
        kernel_factory=FusionKernel,
        stop_on_convergence=False,
    )


def _run_kick(config_path: Path) -> dict[str, Any]:
    """Execute four real steps at gain 0.6 with the unit step-one current kick."""
    return run_free_boundary_tracking(
        config_file=str(config_path),
        shot_steps=4,
        gain=0.6,
        verbose=False,
        kernel_factory=FusionKernel,
        disturbance_callback=_make_coil_kick_disturbance(),
        stop_on_convergence=False,
    )


def _run_topology_kick(config_path: Path) -> dict[str, Any]:
    """Execute four real steps at gain 0.6 with twice the declared current kick."""
    return run_free_boundary_tracking(
        config_file=str(config_path),
        shot_steps=4,
        gain=0.6,
        verbose=False,
        kernel_factory=FusionKernel,
        disturbance_callback=_make_coil_kick_disturbance(2.0),
        stop_on_convergence=False,
    )


def _run_measurement_fault(config_path: Path) -> dict[str, Any]:
    """Execute four gain-0.6 steps using the configured measurement bias, drift and correction."""
    return run_free_boundary_tracking(
        config_file=str(config_path),
        shot_steps=4,
        gain=0.6,
        verbose=False,
        kernel_factory=FusionKernel,
        stop_on_convergence=False,
    )


def _run_latency_fault(config_path: Path) -> dict[str, Any]:
    """Execute four gain-0.6 steps with a 32-fold kick and configured observation delay."""
    return run_free_boundary_tracking(
        config_file=str(config_path),
        shot_steps=4,
        gain=0.6,
        verbose=False,
        kernel_factory=FusionKernel,
        disturbance_callback=_make_coil_kick_disturbance(32.0),
        stop_on_convergence=False,
    )


def _run_actuator_limited_kick(config_path: Path, *, coil_slew_limits: float) -> dict[str, Any]:
    """Execute four gain-8 steps with 0.1 s control, 0.05 s actuator tau and supplied slew limit."""
    return run_free_boundary_tracking(
        config_file=str(config_path),
        shot_steps=4,
        gain=8.0,
        verbose=False,
        kernel_factory=FusionKernel,
        disturbance_callback=_make_coil_kick_disturbance(),
        control_dt_s=0.1,
        coil_actuator_tau_s=0.05,
        coil_slew_limits=coil_slew_limits,
        stop_on_convergence=False,
    )


def _run_supervisor_fallback_kick(config_path: Path) -> dict[str, Any]:
    """Execute four gain-8 steps with an eightfold kick and fixed 0.05 current-unit/s slew limit."""
    return run_free_boundary_tracking(
        config_file=str(config_path),
        shot_steps=4,
        gain=8.0,
        verbose=False,
        kernel_factory=FusionKernel,
        disturbance_callback=_make_coil_kick_disturbance(8.0),
        control_dt_s=0.1,
        coil_actuator_tau_s=0.05,
        coil_slew_limits=0.05,
        stop_on_convergence=False,
    )


def _measurement_tracking_cfg(scale: float, *, corrected: bool) -> dict[str, Any] | None:
    """Declare scaled shape bias/drift; corrected fixtures subtract the identical known error."""
    scale_value = float(scale)
    if scale_value == 0.0 and not corrected:
        return None
    tracking_cfg: dict[str, Any] = {
        "measurement_bias": {"shape_flux": [0.03 * scale_value, -0.02 * scale_value]},
        "measurement_drift_per_step": {"shape_flux": [0.004 * scale_value, -0.003 * scale_value]},
    }
    if corrected:
        tracking_cfg["measurement_correction_bias"] = tracking_cfg["measurement_bias"]
        tracking_cfg["measurement_correction_drift_per_step"] = tracking_cfg["measurement_drift_per_step"]
    return tracking_cfg


def _latency_tracking_cfg(latency_steps: int, *, corrected: bool) -> dict[str, Any] | None:
    """Declare integer step delay; positive corrected delays use unit compensation and 0.5 rate clipping."""
    steps = int(latency_steps)
    if steps == 0 and not corrected:
        return None
    tracking_cfg: dict[str, Any] = {
        "measurement_latency_steps": steps,
    }
    if corrected and steps > 0:
        tracking_cfg["latency_compensation_gain"] = 1.0
        tracking_cfg["latency_rate_max_abs"] = 0.5
    return tracking_cfg


def _topology_measurement_tracking_cfg(scale: float = 1.0, *, corrected: bool) -> dict[str, Any]:
    """Declare scaled X-point/divertor errors and, when requested, exact known-error subtraction."""
    scale_value = float(scale)
    tracking_cfg: dict[str, Any] = {
        "measurement_bias": {
            "x_point_position": [0.06 * scale_value, -0.05 * scale_value],
            "x_point_flux": 0.025 * scale_value,
            "divertor_flux": [0.03 * scale_value, -0.02 * scale_value],
        },
        "measurement_drift_per_step": {
            "x_point_position": [0.01 * scale_value, -0.008 * scale_value],
            "x_point_flux": 0.004 * scale_value,
            "divertor_flux": [0.005 * scale_value, -0.003 * scale_value],
        },
    }
    if corrected:
        tracking_cfg["measurement_correction_bias"] = tracking_cfg["measurement_bias"]
        tracking_cfg["measurement_correction_drift_per_step"] = tracking_cfg["measurement_drift_per_step"]
    return tracking_cfg

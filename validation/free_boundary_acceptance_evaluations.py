# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Free-boundary acceptance evaluations.

"""Compare trusted fixed-campaign summaries to unchanged local diagnostic limits.

These private evaluators coerce declared scalars/counts and compare selected
literal flags. They do not authenticate summaries or admit physical evidence.
Only topology-presence checks explicitly reject absent/nonfinite values.
Merged check/threshold names retain the original later-value replacement rule.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from validation.free_boundary_acceptance_presets import (
    CORRECTED_THRESHOLDS,
    KICK_THRESHOLDS,
    LATENCY_CORRECTED_THRESHOLDS,
    LATENCY_THRESHOLDS,
    MEASUREMENT_THRESHOLDS,
    NOMINAL_THRESHOLDS,
    SUPERVISOR_FALLBACK_THRESHOLDS,
    TOPOLOGY_CORRECTED_THRESHOLDS,
    TOPOLOGY_MEASUREMENT_THRESHOLDS,
    TOPOLOGY_THRESHOLDS,
)


def _evaluate_nominal(summary: dict[str, Any]) -> dict[str, Any]:
    """Compare trusted tracking/shape summaries to nominal limits and literal objective convergence."""
    checks = {
        "final_tracking_error_norm": bool(
            float(summary["final_tracking_error_norm"]) <= NOMINAL_THRESHOLDS["max_final_tracking_error_norm"]
        ),
        "shape_rms": bool(float(summary["shape_rms"]) <= NOMINAL_THRESHOLDS["max_shape_rms"]),
        "objective_converged": bool(
            summary["objective_converged"] is NOMINAL_THRESHOLDS["require_objective_converged"]
        ),
    }
    return {
        "thresholds": NOMINAL_THRESHOLDS.copy(),
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _evaluate_kick(summary: dict[str, Any]) -> dict[str, Any]:
    """Compare trusted tracking, shape and current summaries to fixed kick limits."""
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
    """Require the declared measured/true norm gap and offset, with bounded true tracking error."""
    measured_true_gap = abs(
        float(summary["final_tracking_error_norm"]) - float(summary["final_true_tracking_error_norm"])
    )
    checks = {
        "measured_true_gap": bool(measured_true_gap >= MEASUREMENT_THRESHOLDS["min_measured_true_gap"]),
        "measurement_offset": bool(
            float(summary["max_abs_measurement_offset"]) >= MEASUREMENT_THRESHOLDS["min_measurement_offset"]
        ),
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
    """Bound known-correction gap, offset, tracking and shape error, with literal convergence."""
    measured_true_gap = abs(
        float(summary["final_tracking_error_norm"]) - float(summary["final_true_tracking_error_norm"])
    )
    checks = {
        "measured_true_gap": bool(measured_true_gap <= CORRECTED_THRESHOLDS["max_measured_true_gap"]),
        "measurement_offset": bool(
            float(summary["max_abs_measurement_offset"]) <= CORRECTED_THRESHOLDS["max_measurement_offset"]
        ),
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


def _evaluate_latency_fault(summary: dict[str, Any]) -> dict[str, Any]:
    """Require active delayed observation error while bounding true tracking error."""
    checks = {
        "delayed_observation_error_norm": bool(
            float(summary["max_delayed_observation_error_norm"])
            >= LATENCY_THRESHOLDS["min_delayed_observation_error_norm"]
        ),
        "true_tracking_error_norm": bool(
            float(summary["final_true_tracking_error_norm"]) <= LATENCY_THRESHOLDS["max_true_tracking_error_norm"]
        ),
    }
    return {
        "thresholds": LATENCY_THRESHOLDS.copy(),
        "checks": checks,
        "passes_thresholds": all(checks.values()),
        "delayed_observation_error_norm": float(summary["max_delayed_observation_error_norm"]),
        "estimated_observation_error_norm": float(summary["max_estimated_observation_error_norm"]),
    }


def _evaluate_latency_corrected(
    summary: dict[str, Any],
    *,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bound estimated observation and final true errors using supplied or default correction limits."""
    threshold_set = LATENCY_CORRECTED_THRESHOLDS if thresholds is None else thresholds
    checks = {
        "estimated_observation_error_norm": bool(
            float(summary["max_estimated_observation_error_norm"])
            <= threshold_set["max_estimated_observation_error_norm"]
        ),
        "final_true_tracking_error_norm": bool(
            float(summary["final_true_tracking_error_norm"]) <= threshold_set["max_final_true_tracking_error_norm"]
        ),
        "objective_converged": bool(summary["objective_converged"] is threshold_set["require_objective_converged"]),
    }
    return {
        "thresholds": dict(threshold_set),
        "checks": checks,
        "passes_thresholds": all(checks.values()),
        "delayed_observation_error_norm": float(summary["max_delayed_observation_error_norm"]),
        "estimated_observation_error_norm": float(summary["max_estimated_observation_error_norm"]),
    }


def _evaluate_topology(summary: dict[str, Any]) -> dict[str, Any]:
    """Require present finite X-point/divertor errors and all fixed topology/tracking limits."""
    checks = {
        "final_tracking_error_norm": bool(
            float(summary["final_tracking_error_norm"]) <= TOPOLOGY_THRESHOLDS["max_final_tracking_error_norm"]
        ),
        "shape_rms": bool(float(summary["shape_rms"]) <= TOPOLOGY_THRESHOLDS["max_shape_rms"]),
        "x_point_position_error": bool(
            summary["x_point_position_error"] is not None
            and np.isfinite(float(summary["x_point_position_error"]))
            and float(summary["x_point_position_error"]) <= TOPOLOGY_THRESHOLDS["max_x_point_position_error"]
        ),
        "x_point_flux_error": bool(
            summary["x_point_flux_error"] is not None
            and np.isfinite(float(summary["x_point_flux_error"]))
            and float(summary["x_point_flux_error"]) <= TOPOLOGY_THRESHOLDS["max_x_point_flux_error"]
        ),
        "divertor_rms": bool(
            summary["divertor_rms"] is not None
            and np.isfinite(float(summary["divertor_rms"]))
            and float(summary["divertor_rms"]) <= TOPOLOGY_THRESHOLDS["max_divertor_rms"]
        ),
        "divertor_max_abs": bool(
            summary["divertor_max_abs"] is not None
            and np.isfinite(float(summary["divertor_max_abs"]))
            and float(summary["divertor_max_abs"]) <= TOPOLOGY_THRESHOLDS["max_divertor_max_abs"]
        ),
        "objective_converged": bool(
            summary["objective_converged"] is TOPOLOGY_THRESHOLDS["require_objective_converged"]
        ),
    }
    return {
        "thresholds": TOPOLOGY_THRESHOLDS.copy(),
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _evaluate_topology_measurement_fault(
    summary: dict[str, Any],
    *,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Require topology error gaps and offset while bounding true topology and tracking errors."""
    threshold_set = TOPOLOGY_MEASUREMENT_THRESHOLDS if thresholds is None else thresholds
    x_point_position_gap = abs(float(summary["x_point_position_error"]) - float(summary["true_x_point_position_error"]))
    x_point_flux_gap = abs(float(summary["x_point_flux_error"]) - float(summary["true_x_point_flux_error"]))
    divertor_rms_gap = abs(float(summary["divertor_rms"]) - float(summary["true_divertor_rms"]))
    divertor_max_abs_gap = abs(float(summary["divertor_max_abs"]) - float(summary["true_divertor_max_abs"]))
    checks = {
        "x_point_position_gap": bool(x_point_position_gap >= threshold_set["min_x_point_position_gap"]),
        "x_point_flux_gap": bool(x_point_flux_gap >= threshold_set["min_x_point_flux_gap"]),
        "divertor_rms_gap": bool(divertor_rms_gap >= threshold_set["min_divertor_rms_gap"]),
        "divertor_max_abs_gap": bool(divertor_max_abs_gap >= threshold_set["min_divertor_max_abs_gap"]),
        "measurement_offset": bool(
            float(summary["max_abs_measurement_offset"]) >= threshold_set["min_measurement_offset"]
        ),
        "true_tracking_error_norm": bool(
            float(summary["final_true_tracking_error_norm"]) <= threshold_set["max_true_tracking_error_norm"]
        ),
        "true_x_point_position_error": bool(
            float(summary["true_x_point_position_error"]) <= threshold_set["max_true_x_point_position_error"]
        ),
        "true_x_point_flux_error": bool(
            float(summary["true_x_point_flux_error"]) <= threshold_set["max_true_x_point_flux_error"]
        ),
        "true_divertor_rms": bool(float(summary["true_divertor_rms"]) <= threshold_set["max_true_divertor_rms"]),
        "true_divertor_max_abs": bool(
            float(summary["true_divertor_max_abs"]) <= threshold_set["max_true_divertor_max_abs"]
        ),
        "objective_converged": bool(summary["objective_converged"] is threshold_set["require_objective_converged"]),
    }
    return {
        "thresholds": dict(threshold_set),
        "checks": checks,
        "passes_thresholds": all(checks.values()),
        "x_point_position_gap": x_point_position_gap,
        "x_point_flux_gap": x_point_flux_gap,
        "divertor_rms_gap": divertor_rms_gap,
        "divertor_max_abs_gap": divertor_max_abs_gap,
    }


def _evaluate_topology_corrected(
    summary: dict[str, Any],
    *,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bound known-correction topology gaps, offset and tracking using selected limits."""
    threshold_set = TOPOLOGY_CORRECTED_THRESHOLDS if thresholds is None else thresholds
    x_point_position_gap = abs(float(summary["x_point_position_error"]) - float(summary["true_x_point_position_error"]))
    x_point_flux_gap = abs(float(summary["x_point_flux_error"]) - float(summary["true_x_point_flux_error"]))
    divertor_rms_gap = abs(float(summary["divertor_rms"]) - float(summary["true_divertor_rms"]))
    divertor_max_abs_gap = abs(float(summary["divertor_max_abs"]) - float(summary["true_divertor_max_abs"]))
    checks = {
        "x_point_position_gap": bool(x_point_position_gap <= threshold_set["max_x_point_position_gap"]),
        "x_point_flux_gap": bool(x_point_flux_gap <= threshold_set["max_x_point_flux_gap"]),
        "divertor_rms_gap": bool(divertor_rms_gap <= threshold_set["max_divertor_rms_gap"]),
        "divertor_max_abs_gap": bool(divertor_max_abs_gap <= threshold_set["max_divertor_max_abs_gap"]),
        "measurement_offset": bool(
            float(summary["max_abs_measurement_offset"]) <= threshold_set["max_measurement_offset"]
        ),
        "final_tracking_error_norm": bool(
            float(summary["final_tracking_error_norm"]) <= threshold_set["max_final_tracking_error_norm"]
        ),
        "x_point_position_error": bool(
            float(summary["x_point_position_error"]) <= threshold_set["max_x_point_position_error"]
        ),
        "x_point_flux_error": bool(float(summary["x_point_flux_error"]) <= threshold_set["max_x_point_flux_error"]),
        "divertor_rms": bool(float(summary["divertor_rms"]) <= threshold_set["max_divertor_rms"]),
        "divertor_max_abs": bool(float(summary["divertor_max_abs"]) <= threshold_set["max_divertor_max_abs"]),
        "objective_converged": bool(summary["objective_converged"] is threshold_set["require_objective_converged"]),
    }
    return {
        "thresholds": dict(threshold_set),
        "checks": checks,
        "passes_thresholds": all(checks.values()),
        "x_point_position_gap": x_point_position_gap,
        "x_point_flux_gap": x_point_flux_gap,
        "divertor_rms_gap": divertor_rms_gap,
        "divertor_max_abs_gap": divertor_max_abs_gap,
    }


def _evaluate_supervisor_only(summary: dict[str, Any]) -> dict[str, Any]:
    """Check declared intervention/hold counts, actuator lag and literal active/safe flags."""
    checks = {
        "supervisor_intervention_count": bool(
            int(summary["supervisor_intervention_count"])
            >= SUPERVISOR_FALLBACK_THRESHOLDS["min_supervisor_intervention_count"]
        ),
        "fallback_active_steps": bool(
            int(summary["fallback_active_steps"]) >= SUPERVISOR_FALLBACK_THRESHOLDS["min_fallback_active_steps"]
        ),
        "max_abs_actuator_lag": bool(
            float(summary["max_abs_actuator_lag"]) <= SUPERVISOR_FALLBACK_THRESHOLDS["max_abs_actuator_lag"]
        ),
        "supervisor_active": bool(
            summary["supervisor_active"] is SUPERVISOR_FALLBACK_THRESHOLDS["require_supervisor_active"]
        ),
        "supervisor_safe": bool(
            summary["supervisor_safe"] is SUPERVISOR_FALLBACK_THRESHOLDS["require_supervisor_safe"]
        ),
    }
    return {
        "thresholds": {
            "min_supervisor_intervention_count": SUPERVISOR_FALLBACK_THRESHOLDS["min_supervisor_intervention_count"],
            "min_fallback_active_steps": SUPERVISOR_FALLBACK_THRESHOLDS["min_fallback_active_steps"],
            "max_abs_actuator_lag": SUPERVISOR_FALLBACK_THRESHOLDS["max_abs_actuator_lag"],
            "require_supervisor_active": SUPERVISOR_FALLBACK_THRESHOLDS["require_supervisor_active"],
            "require_supervisor_safe": SUPERVISOR_FALLBACK_THRESHOLDS["require_supervisor_safe"],
        },
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _combine_evaluations(*evaluations: dict[str, Any]) -> dict[str, Any]:
    """Merge thresholds and checks left to right and recompute all checks; duplicate keys are replaced."""
    thresholds: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    extras: dict[str, Any] = {}
    for evaluation in evaluations:
        thresholds.update(dict(evaluation["thresholds"]))
        checks.update(dict(evaluation["checks"]))
        for key, value in evaluation.items():
            if key not in {"thresholds", "checks", "passes_thresholds"}:
                extras[key] = value
    return {
        "thresholds": thresholds,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
        **extras,
    }


def _evaluate_supervisor_fallback(
    summary: dict[str, Any],
    *,
    unsupervised_reference: dict[str, Any],
) -> dict[str, Any]:
    """Compare safe/reference lag and require bounded topology plus declared supervisor intervention."""
    safe_lag = float(summary["max_abs_actuator_lag"])
    reference_lag = float(unsupervised_reference["max_abs_actuator_lag"])
    lag_reduction_factor = float(reference_lag / max(safe_lag, 1.0e-12))
    checks = {
        "supervisor_intervention_count": bool(
            int(summary["supervisor_intervention_count"])
            >= SUPERVISOR_FALLBACK_THRESHOLDS["min_supervisor_intervention_count"]
        ),
        "fallback_active_steps": bool(
            int(summary["fallback_active_steps"]) >= SUPERVISOR_FALLBACK_THRESHOLDS["min_fallback_active_steps"]
        ),
        "max_abs_actuator_lag": bool(safe_lag <= SUPERVISOR_FALLBACK_THRESHOLDS["max_abs_actuator_lag"]),
        "lag_reduction_factor": bool(
            lag_reduction_factor >= SUPERVISOR_FALLBACK_THRESHOLDS["min_lag_reduction_factor"]
        ),
        "final_tracking_error_norm": bool(
            float(summary["final_tracking_error_norm"])
            <= SUPERVISOR_FALLBACK_THRESHOLDS["max_final_tracking_error_norm"]
        ),
        "x_point_position_error": bool(
            summary["x_point_position_error"] is not None
            and np.isfinite(float(summary["x_point_position_error"]))
            and float(summary["x_point_position_error"]) <= SUPERVISOR_FALLBACK_THRESHOLDS["max_x_point_position_error"]
        ),
        "x_point_flux_error": bool(
            summary["x_point_flux_error"] is not None
            and np.isfinite(float(summary["x_point_flux_error"]))
            and float(summary["x_point_flux_error"]) <= SUPERVISOR_FALLBACK_THRESHOLDS["max_x_point_flux_error"]
        ),
        "divertor_rms": bool(
            summary["divertor_rms"] is not None
            and np.isfinite(float(summary["divertor_rms"]))
            and float(summary["divertor_rms"]) <= SUPERVISOR_FALLBACK_THRESHOLDS["max_divertor_rms"]
        ),
        "divertor_max_abs": bool(
            summary["divertor_max_abs"] is not None
            and np.isfinite(float(summary["divertor_max_abs"]))
            and float(summary["divertor_max_abs"]) <= SUPERVISOR_FALLBACK_THRESHOLDS["max_divertor_max_abs"]
        ),
        "supervisor_active": bool(
            summary["supervisor_active"] is SUPERVISOR_FALLBACK_THRESHOLDS["require_supervisor_active"]
        ),
        "supervisor_safe": bool(
            summary["supervisor_safe"] is SUPERVISOR_FALLBACK_THRESHOLDS["require_supervisor_safe"]
        ),
        "objective_converged": bool(
            summary["objective_converged"] is SUPERVISOR_FALLBACK_THRESHOLDS["require_objective_converged"]
        ),
    }
    return {
        "thresholds": SUPERVISOR_FALLBACK_THRESHOLDS.copy(),
        "checks": checks,
        "passes_thresholds": all(checks.values()),
        "lag_reduction_factor": lag_reduction_factor,
    }


def _is_monotone_non_decreasing(values: list[float], *, atol: float = 1.0e-12) -> bool:
    """Compare adjacent float-coercible values with absolute tolerance; empty/single lists pass."""
    return all(float(b) + atol >= float(a) for a, b in zip(values[:-1], values[1:]))


def _is_monotone_non_increasing(values: list[float], *, atol: float = 1.0e-12) -> bool:
    """Compare adjacent float-coercible values in reverse order with absolute tolerance."""
    return all(float(b) <= float(a) + atol for a, b in zip(values[:-1], values[1:]))

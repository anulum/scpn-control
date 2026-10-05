# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Free-boundary acceptance campaign.

"""Assemble the complete same-model free-boundary regression campaign.

All eighteen scenarios and twelve sweeps use the original real solver and
fixed thresholds. Same-solver targets and exact known-error corrections
provide local diagnostic evidence without independent physical admission.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from validation.free_boundary_acceptance_evaluations import (
    _combine_evaluations,
    _evaluate_corrected,
    _evaluate_kick,
    _evaluate_latency_corrected,
    _evaluate_latency_fault,
    _evaluate_measurement_fault,
    _evaluate_nominal,
    _evaluate_supervisor_fallback,
    _evaluate_supervisor_only,
    _evaluate_topology,
    _evaluate_topology_corrected,
    _evaluate_topology_measurement_fault,
)
from validation.free_boundary_acceptance_presets import (
    TOPOLOGY_LATENCY_MEASUREMENT_THRESHOLDS,
    TOPOLOGY_SUPERVISOR_LATENCY_CORRECTED_THRESHOLDS,
    _build_topology_tracking_template,
    _build_tracking_template,
    _latency_tracking_cfg,
    _merge_tracking_cfg,
    _run_kick,
    _run_latency_fault,
    _run_measurement_fault,
    _run_nominal,
    _run_supervisor_fallback_kick,
    _run_topology_kick,
    _topology_measurement_tracking_cfg,
    _write_tracking_config,
)
from validation.free_boundary_acceptance_sweeps import (
    _run_actuator_slew_sweep,
    _run_coil_kick_scale_sweep,
    _run_corrected_latency_step_sweep,
    _run_corrected_measurement_sweep,
    _run_latency_step_sweep,
    _run_measurement_sweep,
)
from validation.free_boundary_acceptance_topology_sweeps import (
    _run_topology_actuator_slew_sweep,
    _run_topology_corrected_measurement_sweep,
    _run_topology_kick_scale_sweep,
    _run_topology_measurement_sweep,
    _run_topology_supervisor_corrected_measurement_sweep,
    _run_topology_supervisor_measurement_sweep,
)


def run_campaign() -> dict[str, Any]:
    """Execute the fixed real-kernel diagnostic cohort and return nested declarations.

    Returns
    -------
    dict
        Eighteen four-step scenarios, twelve severity sweeps, individual checks
        and their conjunction in passes_thresholds. Summaries contain mutable
        dictionaries/lists; no source authentication is established.

    Notes
    -----
        Temporary JSON configurations are deleted on exit. The grid is 12 by 12,
        permeability and plasma-current target are both 1.0. Shape, X-point and
        divertor targets are sampled from the same solver before the cohort;
        corrected measurement fixtures subtract the exact supplied bias/drift.
        Tracking norms combine configured objectives, not one universal SI unit.
    This is same-model regression evidence, not independent equilibrium,
        observer, actuator, facility or safety validation. Existing numerical
        kernel/tracking exceptions propagate; no fallback solver is substituted.
    """
    with tempfile.TemporaryDirectory(prefix="scpn_free_boundary_acceptance_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        template_cfg = _build_tracking_template(tmp_path)

        nominal_cfg = _write_tracking_config(
            tmp_path / "nominal.json",
            template_cfg=template_cfg,
        )
        nominal = _run_nominal(nominal_cfg)

        kick_cfg = _write_tracking_config(
            tmp_path / "kick.json",
            template_cfg=template_cfg,
        )
        kick = _run_kick(kick_cfg)

        measurement_cfg = _write_tracking_config(
            tmp_path / "measurement.json",
            template_cfg=template_cfg,
            tracking_cfg={
                "measurement_bias": {"shape_flux": [0.03, -0.02]},
                "measurement_drift_per_step": {"shape_flux": [0.004, -0.003]},
            },
        )
        measurement_fault = _run_measurement_fault(measurement_cfg)

        corrected_cfg = _write_tracking_config(
            tmp_path / "measurement_corrected.json",
            template_cfg=template_cfg,
            tracking_cfg={
                "measurement_bias": {"shape_flux": [0.03, -0.02]},
                "measurement_drift_per_step": {"shape_flux": [0.004, -0.003]},
                "measurement_correction_bias": {"shape_flux": [0.03, -0.02]},
                "measurement_correction_drift_per_step": {"shape_flux": [0.004, -0.003]},
            },
        )
        corrected = _run_measurement_fault(corrected_cfg)

        latency_cfg = _write_tracking_config(
            tmp_path / "latency.json",
            template_cfg=template_cfg,
            tracking_cfg=_latency_tracking_cfg(2, corrected=False),
        )
        latency_fault = _run_latency_fault(latency_cfg)

        latency_corrected_cfg = _write_tracking_config(
            tmp_path / "latency_corrected.json",
            template_cfg=template_cfg,
            tracking_cfg=_latency_tracking_cfg(2, corrected=True),
        )
        latency_corrected = _run_latency_fault(latency_corrected_cfg)

        topology_template_cfg = _build_topology_tracking_template(tmp_path, template_cfg=template_cfg)
        topology_cfg = _write_tracking_config(
            tmp_path / "topology_kick.json",
            template_cfg=topology_template_cfg,
        )
        topology_kick = _run_topology_kick(topology_cfg)

        topology_supervisor_measurement_fault_cfg = _write_tracking_config(
            tmp_path / "topology_supervisor_measurement_fault.json",
            template_cfg=topology_template_cfg,
            tracking_cfg={
                "fallback_currents": [0.0, 0.0],
                "supervisor_limits": {"max_abs_actuator_lag": 2.0},
                "hold_steps_after_reject": 2,
                **_topology_measurement_tracking_cfg(corrected=False),
            },
        )
        topology_supervisor_measurement_fault = _run_supervisor_fallback_kick(topology_supervisor_measurement_fault_cfg)

        topology_supervisor_measurement_corrected_cfg = _write_tracking_config(
            tmp_path / "topology_supervisor_measurement_corrected.json",
            template_cfg=topology_template_cfg,
            tracking_cfg={
                "fallback_currents": [0.0, 0.0],
                "supervisor_limits": {"max_abs_actuator_lag": 2.0},
                "hold_steps_after_reject": 2,
                **_topology_measurement_tracking_cfg(corrected=True),
            },
        )
        topology_supervisor_measurement_corrected = _run_supervisor_fallback_kick(
            topology_supervisor_measurement_corrected_cfg
        )

        topology_supervisor_measurement_latency_cfg = _write_tracking_config(
            tmp_path / "topology_supervisor_measurement_latency.json",
            template_cfg=topology_template_cfg,
            tracking_cfg=_merge_tracking_cfg(
                {
                    "fallback_currents": [0.0, 0.0],
                    "supervisor_limits": {"max_abs_actuator_lag": 2.0},
                    "hold_steps_after_reject": 2,
                },
                _topology_measurement_tracking_cfg(corrected=False),
                _latency_tracking_cfg(2, corrected=False),
            ),
        )
        topology_supervisor_measurement_latency_fault = _run_supervisor_fallback_kick(
            topology_supervisor_measurement_latency_cfg
        )

        topology_supervisor_measurement_latency_corrected_cfg = _write_tracking_config(
            tmp_path / "topology_supervisor_measurement_latency_corrected.json",
            template_cfg=topology_template_cfg,
            tracking_cfg=_merge_tracking_cfg(
                {
                    "fallback_currents": [0.0, 0.0],
                    "supervisor_limits": {"max_abs_actuator_lag": 2.0},
                    "hold_steps_after_reject": 2,
                },
                _topology_measurement_tracking_cfg(corrected=True),
                _latency_tracking_cfg(2, corrected=True),
            ),
        )
        topology_supervisor_measurement_latency_corrected = _run_supervisor_fallback_kick(
            topology_supervisor_measurement_latency_corrected_cfg
        )

        topology_combined_measurement_cfg = _write_tracking_config(
            tmp_path / "topology_combined_measurement.json",
            template_cfg=topology_template_cfg,
            tracking_cfg=_topology_measurement_tracking_cfg(corrected=False),
        )
        topology_combined_measurement_fault = _run_topology_kick(topology_combined_measurement_cfg)

        topology_combined_corrected_cfg = _write_tracking_config(
            tmp_path / "topology_combined_measurement_corrected.json",
            template_cfg=topology_template_cfg,
            tracking_cfg=_topology_measurement_tracking_cfg(corrected=True),
        )
        topology_combined_measurement_corrected = _run_topology_kick(topology_combined_corrected_cfg)

        topology_measurement_cfg = _write_tracking_config(
            tmp_path / "topology_measurement.json",
            template_cfg=topology_template_cfg,
            tracking_cfg=_topology_measurement_tracking_cfg(corrected=False),
        )
        topology_measurement_fault = _run_measurement_fault(topology_measurement_cfg)

        topology_corrected_cfg = _write_tracking_config(
            tmp_path / "topology_measurement_corrected.json",
            template_cfg=topology_template_cfg,
            tracking_cfg=_topology_measurement_tracking_cfg(corrected=True),
        )
        topology_measurement_corrected = _run_measurement_fault(topology_corrected_cfg)

        topology_measurement_latency_cfg = _write_tracking_config(
            tmp_path / "topology_measurement_latency.json",
            template_cfg=topology_template_cfg,
            tracking_cfg=_merge_tracking_cfg(
                _topology_measurement_tracking_cfg(corrected=False),
                _latency_tracking_cfg(2, corrected=False),
            ),
        )
        topology_measurement_latency_fault = _run_measurement_fault(topology_measurement_latency_cfg)

        topology_measurement_latency_corrected_cfg = _write_tracking_config(
            tmp_path / "topology_measurement_latency_corrected.json",
            template_cfg=topology_template_cfg,
            tracking_cfg=_merge_tracking_cfg(
                _topology_measurement_tracking_cfg(corrected=True),
                _latency_tracking_cfg(2, corrected=True),
            ),
        )
        topology_measurement_latency_corrected = _run_measurement_fault(topology_measurement_latency_corrected_cfg)

        unsupervised_safety_cfg = _write_tracking_config(
            tmp_path / "unsupervised_safety_reference.json",
            template_cfg=topology_template_cfg,
        )
        unsupervised_safety_reference = _run_supervisor_fallback_kick(unsupervised_safety_cfg)

        supervisor_fallback_cfg = _write_tracking_config(
            tmp_path / "supervisor_fallback.json",
            template_cfg=topology_template_cfg,
            tracking_cfg={
                "fallback_currents": [0.0, 0.0],
                "supervisor_limits": {"max_abs_actuator_lag": 2.0},
                "hold_steps_after_reject": 2,
            },
        )
        supervisor_fallback = _run_supervisor_fallback_kick(supervisor_fallback_cfg)

        measurement_sweep = _run_measurement_sweep(tmp_path, template_cfg=template_cfg)
        corrected_measurement_sweep = _run_corrected_measurement_sweep(tmp_path, template_cfg=template_cfg)
        latency_step_sweep = _run_latency_step_sweep(tmp_path, template_cfg=template_cfg)
        corrected_latency_step_sweep = _run_corrected_latency_step_sweep(tmp_path, template_cfg=template_cfg)
        actuator_slew_sweep = _run_actuator_slew_sweep(tmp_path, template_cfg=template_cfg)
        coil_kick_scale_sweep = _run_coil_kick_scale_sweep(tmp_path, template_cfg=template_cfg)
        topology_kick_scale_sweep = _run_topology_kick_scale_sweep(
            tmp_path,
            topology_template_cfg=topology_template_cfg,
        )
        topology_actuator_slew_sweep = _run_topology_actuator_slew_sweep(
            tmp_path,
            topology_template_cfg=topology_template_cfg,
        )
        topology_measurement_sweep = _run_topology_measurement_sweep(
            tmp_path,
            topology_template_cfg=topology_template_cfg,
        )
        topology_corrected_measurement_sweep = _run_topology_corrected_measurement_sweep(
            tmp_path,
            topology_template_cfg=topology_template_cfg,
        )
        topology_supervisor_measurement_sweep = _run_topology_supervisor_measurement_sweep(
            tmp_path,
            topology_template_cfg=topology_template_cfg,
        )
        topology_supervisor_corrected_measurement_sweep = _run_topology_supervisor_corrected_measurement_sweep(
            tmp_path,
            topology_template_cfg=topology_template_cfg,
        )

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
        "measurement_latency_uncorrected": {
            "summary": latency_fault,
            **_evaluate_latency_fault(latency_fault),
        },
        "measurement_latency_corrected": {
            "summary": latency_corrected,
            **_evaluate_latency_corrected(latency_corrected),
        },
        "x_point_divertor_kick": {
            "summary": topology_kick,
            **_evaluate_topology(topology_kick),
        },
        "x_point_divertor_supervisor_measurement_fault_uncorrected": {
            "summary": topology_supervisor_measurement_fault,
            **_combine_evaluations(
                _evaluate_topology_measurement_fault(topology_supervisor_measurement_fault),
                _evaluate_supervisor_only(topology_supervisor_measurement_fault),
            ),
        },
        "x_point_divertor_supervisor_measurement_fault_corrected": {
            "summary": topology_supervisor_measurement_corrected,
            **_combine_evaluations(
                _evaluate_topology_corrected(topology_supervisor_measurement_corrected),
                _evaluate_supervisor_only(topology_supervisor_measurement_corrected),
            ),
        },
        "x_point_divertor_supervisor_measurement_latency_uncorrected": {
            "summary": topology_supervisor_measurement_latency_fault,
            **_combine_evaluations(
                _evaluate_topology_measurement_fault(
                    topology_supervisor_measurement_latency_fault,
                    thresholds=TOPOLOGY_LATENCY_MEASUREMENT_THRESHOLDS,
                ),
                _evaluate_supervisor_only(topology_supervisor_measurement_latency_fault),
                _evaluate_latency_fault(topology_supervisor_measurement_latency_fault),
            ),
        },
        "x_point_divertor_supervisor_measurement_latency_corrected": {
            "summary": topology_supervisor_measurement_latency_corrected,
            **_combine_evaluations(
                _evaluate_topology_corrected(
                    topology_supervisor_measurement_latency_corrected,
                    thresholds=TOPOLOGY_SUPERVISOR_LATENCY_CORRECTED_THRESHOLDS,
                ),
                _evaluate_supervisor_only(topology_supervisor_measurement_latency_corrected),
                _evaluate_latency_corrected(
                    topology_supervisor_measurement_latency_corrected,
                    thresholds=TOPOLOGY_SUPERVISOR_LATENCY_CORRECTED_THRESHOLDS,
                ),
            ),
        },
        "x_point_divertor_combined_fault_uncorrected": {
            "summary": topology_combined_measurement_fault,
            **_evaluate_topology_measurement_fault(topology_combined_measurement_fault),
        },
        "x_point_divertor_combined_fault_corrected": {
            "summary": topology_combined_measurement_corrected,
            **_evaluate_topology_corrected(topology_combined_measurement_corrected),
        },
        "x_point_divertor_measurement_fault_uncorrected": {
            "summary": topology_measurement_fault,
            **_evaluate_topology_measurement_fault(topology_measurement_fault),
        },
        "x_point_divertor_measurement_fault_corrected": {
            "summary": topology_measurement_corrected,
            **_evaluate_topology_corrected(topology_measurement_corrected),
        },
        "x_point_divertor_measurement_latency_uncorrected": {
            "summary": topology_measurement_latency_fault,
            **_combine_evaluations(
                _evaluate_topology_measurement_fault(
                    topology_measurement_latency_fault,
                    thresholds=TOPOLOGY_LATENCY_MEASUREMENT_THRESHOLDS,
                ),
                _evaluate_latency_fault(topology_measurement_latency_fault),
            ),
        },
        "x_point_divertor_measurement_latency_corrected": {
            "summary": topology_measurement_latency_corrected,
            **_combine_evaluations(
                _evaluate_topology_corrected(topology_measurement_latency_corrected),
                _evaluate_latency_corrected(topology_measurement_latency_corrected),
            ),
        },
        "supervisor_fallback_kick": {
            "summary": supervisor_fallback,
            "unsupervised_reference": unsupervised_safety_reference,
            **_evaluate_supervisor_fallback(
                supervisor_fallback,
                unsupervised_reference=unsupervised_safety_reference,
            ),
        },
    }
    sweeps = {
        "measurement_fault_scale": measurement_sweep,
        "measurement_fault_corrected_scale": corrected_measurement_sweep,
        "measurement_latency_steps": latency_step_sweep,
        "measurement_latency_corrected_steps": corrected_latency_step_sweep,
        "actuator_slew_limit": actuator_slew_sweep,
        "coil_kick_scale": coil_kick_scale_sweep,
        "topology_kick_scale": topology_kick_scale_sweep,
        "topology_actuator_slew_limit": topology_actuator_slew_sweep,
        "topology_measurement_fault_scale": topology_measurement_sweep,
        "topology_measurement_corrected_scale": topology_corrected_measurement_sweep,
        "topology_supervisor_measurement_fault_scale": topology_supervisor_measurement_sweep,
        "topology_supervisor_measurement_corrected_scale": topology_supervisor_corrected_measurement_sweep,
    }
    passes_thresholds = all(bool(entry["passes_thresholds"]) for entry in scenarios.values()) and all(
        bool(entry["passes_thresholds"]) for entry in sweeps.values()
    )
    return {
        "benchmark": "free_boundary_tracking_acceptance",
        "steps_per_scenario": 4,
        "scenarios": scenarios,
        "sweeps": sweeps,
        "passes_thresholds": passes_thresholds,
    }

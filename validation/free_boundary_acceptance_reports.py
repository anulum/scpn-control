# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Free-boundary acceptance reports.

"""Describe and render normalized real-kernel regression observations.

Local campaign timing excludes output rendering/writes. Schema-v2 limits are
self-declarations; persistent origin custody belongs to the recorded runner.
The historical scientific report files are not rewritten by importing this module.
"""

from __future__ import annotations

import time
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from validation.free_boundary_acceptance_campaign import (
    run_campaign,
)

MODEL_CONTRACT = {
    "kind": "normalized_same_model_free_boundary_diagnostic",
    "grid_shape": [12, 12],
    "vacuum_permeability": 1.0,
    "plasma_current_target": 1.0,
    "steps_per_scenario": 4,
    "targets": "sampled from the same FusionKernel before cohort execution",
    "measurement_correction": "exact supplied bias/drift subtraction, not inferred calibration",
    "flux_units": "configured permeability=1 normalization; not calibrated SI flux",
    "tracking_norm": "weighted configured objectives; no universal physical unit",
    "position_units": "configured metre coordinate labels",
    "current_units": "configured coil-current convention",
    "runtime_seconds": "campaign and UTC timestamp construction; excludes rendering and writes",
    "physical_reference_admitted": False,
}


def generate_report() -> dict[str, Any]:
    """Run the fixed campaign and attach UTC time, elapsed runtime and model limits.

    Returns
    -------
    dict
        Fresh schema-v2 report with the legacy campaign/timestamp/runtime fields,
        a copied model contract and physical_reference_admitted=False. Nested
        summaries remain mutable declarations without origin authentication.

    Notes
    -----
        runtime_seconds measures campaign and timestamp construction through
        perf_counter, excluding Markdown/JSON rendering and output writes.
        Temporary fixture lifetime and numerical errors follow run_campaign.
    """
    t0 = time.perf_counter()
    campaign = run_campaign()
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": float(time.perf_counter() - t0),
        "free_boundary_tracking_acceptance": campaign,
        "schema_version": "scpn-control.free-boundary-tracking-acceptance.v2",
        "model_contract": deepcopy(MODEL_CONTRACT),
        "physical_reference_admitted": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a declared campaign with normalization, calibration and timing limits.

    Parameters
    ----------
    report : dict
        Legacy or schema-v2 report with timestamp, numeric runtime_seconds and
        the complete campaign scenario/sweep mappings.

    Returns
    -------
    str
        Markdown ending in a newline. No files are written and no campaign,
        checksum, source authentication or independent reference is rerun.

    Raises
    ------
    KeyError, TypeError, ValueError
        Missing/malformed required declarations or unsupported numeric formats.
    """
    campaign = report["free_boundary_tracking_acceptance"]
    lines = [
        "# Free-Boundary Tracking Acceptance",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{report['runtime_seconds']:.3f} s`",
        f"- Steps per scenario: `{campaign['steps_per_scenario']}`",
        f"- Pass: `{'YES' if campaign['passes_thresholds'] else 'NO'}`",
        "",
        "This is a normalized 12-by-12 same-model diagnostic with permeability and plasma-current target 1.0. Shape, X-point and divertor targets come from the same solver. Corrected measurements subtract the exact injected bias/drift; no independent observer calibration is established.",
        "",
        "Tracking norms combine configured objectives. Flux residuals use the configured permeability=1 normalization, not calibrated SI flux. Positions and currents follow the model coordinate/current conventions.",
        "",
        "Runtime includes campaign and UTC timestamp construction, excluding rendering and writes. No independent physical-reference, facility-control, safety or controlled performance admission is established.",
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
                f"- Supervisor interventions: `{summary['supervisor_intervention_count']}`",
                f"- Fallback active steps: `{summary['fallback_active_steps']}`",
                "",
            ]
        )
        if "lag_reduction_factor" in data:
            lines.extend(
                [
                    f"- Lag reduction factor vs unsupervised: `{data['lag_reduction_factor']:.2f}`",
                    "",
                ]
            )
    lines.extend(
        [
            "## Sweeps",
            "",
            "### measurement_fault_scale",
            "",
        ]
    )
    for entry in campaign["sweeps"]["measurement_fault_scale"]["entries"]:
        lines.append(
            "- "
            f"scale `{entry['scale']:.1f}`: gap `{entry['measured_true_gap']:.6e}`, "
            f"offset `{entry['max_abs_measurement_offset']:.6e}`"
        )
    lines.extend(
        [
            "",
            "### actuator_slew_limit",
            "",
        ]
    )
    for entry in campaign["sweeps"]["actuator_slew_limit"]["entries"]:
        lines.append(
            "- "
            f"slew `{entry['coil_slew_limit']:.3e}`: max lag `{entry['max_abs_actuator_lag']:.6e}`, "
            f"mean lag `{entry['mean_abs_actuator_lag']:.6e}`"
        )
    lines.extend(
        [
            "",
            "### measurement_fault_corrected_scale",
            "",
        ]
    )
    for entry in campaign["sweeps"]["measurement_fault_corrected_scale"]["entries"]:
        lines.append(
            "- "
            f"scale `{entry['scale']:.1f}`: gap `{entry['measured_true_gap']:.6e}`, "
            f"offset `{entry['max_abs_measurement_offset']:.6e}`"
        )
    lines.extend(
        [
            "",
            "### measurement_latency_steps",
            "",
        ]
    )
    for entry in campaign["sweeps"]["measurement_latency_steps"]["entries"]:
        lines.append(
            "- "
            f"steps `{entry['latency_steps']}`: delayed err `{entry['delayed_observation_error_norm']:.6e}`, "
            f"estimated err `{entry['estimated_observation_error_norm']:.6e}`"
        )
    lines.extend(
        [
            "",
            "### measurement_latency_corrected_steps",
            "",
        ]
    )
    for entry in campaign["sweeps"]["measurement_latency_corrected_steps"]["entries"]:
        lines.append(
            "- "
            f"steps `{entry['latency_steps']}`: delayed err `{entry['delayed_observation_error_norm']:.6e}`, "
            f"estimated err `{entry['estimated_observation_error_norm']:.6e}`"
        )
    lines.extend(
        [
            "",
            "### coil_kick_scale",
            "",
        ]
    )
    for entry in campaign["sweeps"]["coil_kick_scale"]["entries"]:
        lines.append(
            "- "
            f"scale `{entry['kick_scale']:.1f}`: max coil `{entry['max_abs_coil_current']:.6e}`, "
            f"final err `{entry['final_tracking_error_norm']:.6e}`"
        )
    lines.extend(
        [
            "",
            "### topology_kick_scale",
            "",
        ]
    )
    for entry in campaign["sweeps"]["topology_kick_scale"]["entries"]:
        lines.append(
            "- "
            f"scale `{entry['kick_scale']:.1f}`: x-flux err `{entry['x_point_flux_error']:.6e}`, "
            f"divertor rms `{entry['divertor_rms']:.6e}`, max coil `{entry['max_abs_coil_current']:.6e}`"
        )
    lines.extend(
        [
            "",
            "### topology_actuator_slew_limit",
            "",
        ]
    )
    for entry in campaign["sweeps"]["topology_actuator_slew_limit"]["entries"]:
        lines.append(
            "- "
            f"slew `{entry['coil_slew_limit']:.3e}`: max lag `{entry['max_abs_actuator_lag']:.6e}`, "
            f"x-flux err `{entry['x_point_flux_error']:.6e}`, divertor rms `{entry['divertor_rms']:.6e}`"
        )
    lines.extend(
        [
            "",
            "### topology_measurement_fault_scale",
            "",
        ]
    )
    for entry in campaign["sweeps"]["topology_measurement_fault_scale"]["entries"]:
        lines.append(
            "- "
            f"scale `{entry['scale']:.1f}`: x-pos gap `{entry['x_point_position_gap']:.6e}`, "
            f"x-flux gap `{entry['x_point_flux_gap']:.6e}`, divertor rms gap `{entry['divertor_rms_gap']:.6e}`"
        )
    lines.extend(
        [
            "",
            "### topology_measurement_corrected_scale",
            "",
        ]
    )
    for entry in campaign["sweeps"]["topology_measurement_corrected_scale"]["entries"]:
        lines.append(
            "- "
            f"scale `{entry['scale']:.1f}`: x-pos gap `{entry['x_point_position_gap']:.6e}`, "
            f"x-flux gap `{entry['x_point_flux_gap']:.6e}`, divertor rms gap `{entry['divertor_rms_gap']:.6e}`"
        )
    lines.extend(
        [
            "",
            "### topology_supervisor_measurement_fault_scale",
            "",
        ]
    )
    for entry in campaign["sweeps"]["topology_supervisor_measurement_fault_scale"]["entries"]:
        lines.append(
            "- "
            f"scale `{entry['scale']:.1f}`: x-pos gap `{entry['x_point_position_gap']:.6e}`, "
            f"x-flux gap `{entry['x_point_flux_gap']:.6e}`, lag `{entry['max_abs_actuator_lag']:.6e}`"
        )
    lines.extend(
        [
            "",
            "### topology_supervisor_measurement_corrected_scale",
            "",
        ]
    )
    for entry in campaign["sweeps"]["topology_supervisor_measurement_corrected_scale"]["entries"]:
        lines.append(
            "- "
            f"scale `{entry['scale']:.1f}`: x-pos gap `{entry['x_point_position_gap']:.6e}`, "
            f"x-flux gap `{entry['x_point_flux_gap']:.6e}`, lag `{entry['max_abs_actuator_lag']:.6e}`"
        )
    return "\n".join(lines) + "\n"

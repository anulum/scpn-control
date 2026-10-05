# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disturbance reports.

"""Render reduced-model observations with truthful horizons and provider limits."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from validation.disturbance_runtime import ScenarioMetrics, TraceData

EXPECTED_CONTROLLERS = ("PID", "H-infinity", "MPC", "SNN")
MODEL_CONTRACT = {
    "kind": "synthetic_two_state_vertical_proxy",
    "gamma_coefficient_s^-1": 100.0,
    "damping_s^-1": 10.0,
    "disturbance_acceleration_gain": 0.5,
    "integration": "explicit Euler",
    "position_bound_m": 10.0,
    "error_convention": "target_z minus position",
    "hinf_measurement": "negative error; exact measured position at zero target",
    "MPC": "zero-start approximate one-step-sensitivity gradient, not optimal MPC",
    "SNN": "real provider samples; plant dt does not set its neuron clock",
    "density_and_ELM": "acceleration forcings only; no density/beta/energy state",
    "physical_reference_admitted": False,
}


def generate_stdout_table(all_metrics: Sequence[ScenarioMetrics]) -> str:
    """Format observations with stable/completion status and SI metric units.

    Parameters
    ----------
    all_metrics : sequence of ScenarioMetrics
        Observed diagnostic rows; caller declarations are not authenticated.

    Returns
    -------
    str
        Markdown table. Settled is separate from finite/bounded completion.
    """
    lines = [
        "| Controller | Scenario | ISE (m² s) | Settle (s) | Peak (m) | Effort (m/s) | Observed (s) | Stable | Settled |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for m in all_metrics:
        lines.append(
            f"| {m.controller} | {m.scenario} | {m.ise:.6e} | {m.settling_time_s:.6g} | {m.peak_overshoot:.6e} | {m.control_effort:.6e} | {m.completed_duration_s:.6g} | {m.stable} | {m.settled} |"
        )
    return "\n".join(lines)


def generate_json_results(all_metrics: Sequence[ScenarioMetrics]) -> dict[str, object]:
    """Declare actual per-run horizons and incomplete controller coverage.

    Parameters
    ----------
    all_metrics : sequence of ScenarioMetrics
        Actual run rows, or explicitly unverified caller declarations.

    Returns
    -------
    dict
        Schema-v2 finite JSON data, synthetic model contract and missing
        controller names. No static registry duration substitutes actual runs.

    Raises
    ------
    ValueError
        A caller row includes nonfinite numeric metadata; JSON forbids NaN.
    """
    active = list(dict.fromkeys(m.controller for m in all_metrics))
    result: dict[str, object] = dict(
        schema_version=2,
        benchmark="disturbance_rejection",
        generated_at_utc=datetime.now(UTC).isoformat(),
        model_contract=dict(MODEL_CONTRACT),
        actual_controllers=active,
        missing_controllers=[n for n in EXPECTED_CONTROLLERS if n not in active],
        complete_controller_cohort=all(n in active for n in EXPECTED_CONTROLLERS),
        all_runs_completed_bounded=bool(all_metrics) and all(m.stable for m in all_metrics),
        results=[m.to_dict() for m in all_metrics],
    )
    json.dumps(result, allow_nan=False)
    return result


def generate_markdown_report(all_metrics: Sequence[ScenarioMetrics]) -> str:
    """Describe the actual observed horizons and reduced-model limitations.

    Parameters
    ----------
    all_metrics : sequence of ScenarioMetrics
        Per-run metrics, including explicit requested/completed intervals.

    Returns
    -------
    str
        UTF-8-ready Markdown with units, band semantics and no ITER claim.
    """
    result = generate_json_results(all_metrics)
    lines = [
        "# Synthetic disturbance rejection benchmark",
        "",
        "The two-state Euler model evolves position and velocity only. Density ramp and ELM names select acceleration forcing; density, beta_N and energy are not evolved.",
        "",
        "No physical-reference, facility control, safety or controlled speedup admission is established. Local wall times are execution observations.",
        "",
        "The H-infinity adapter converts target-minus-position error to its defining positive measurement convention. Nonzero-target tracking is not admitted. The MPC gradient is approximate. SNN uses the actual provider clock per call; plant dt does not calibrate neuron time.",
        "",
        "ISE integrates recorded error squared in m² s. Effort sums absolute held control over actual intervals in m/s. Peak is maximum absolute error, not signed overshoot.",
        "",
        "Settling band is threshold × max(initial absolute error, first 100 observed absolute errors, 0.001 m). Settled means the terminal sample is inside the band after finite/bounded completion; settling is assessed only over the observed horizon.",
        "",
        "Trace samples include the real initial and terminal state. Unstable runs stop at |z|>10 m and contain no fabricated tail.",
        "",
        f"Missing controllers: {result['missing_controllers']}",
        "",
        generate_stdout_table(all_metrics),
        "",
        "| Controller | Scenario | Requested (s) | Completed steps / requested | Termination | Band (m) |",
        "|---|---|---:|---:|---|---:|",
    ]
    lines.extend(
        f"| {m.controller} | {m.scenario} | {m.requested_duration_s:.6g} | {m.completed_steps}/{m.requested_steps} | {m.termination} | {m.settling_band_m:.6e} |"
        for m in all_metrics
    )
    return "\n".join(lines) + "\n"


def save_overlay_plots(
    all_traces: Mapping[tuple[str, str], TraceData], all_metrics: Sequence[ScenarioMetrics], output_dir: Path
) -> list[str]:
    """Write actual position/error/action overlays if matplotlib can be imported.

    Parameters
    ----------
    all_traces : Mapping
        Actual aligned traces keyed by (controller, scenario).
    all_metrics : sequence of ScenarioMetrics
        Matching actual rows, including the target reference in metres.
    output_dir : Path
        Existing caller-selected destination directory. Existing plot names
        are replaced; the CLI performs custody and source-alias checks first.

    Returns
    -------
    list of str
        Written PNG paths, or empty when matplotlib is unavailable.

    Notes
    -----
    Plots show each trace's actual termination time. This renderer has no
    numerical/provider substitution and establishes no physics admission.
    Every created figure is closed after rendering or writing, including
    failures. Earlier written plots remain when a later plot fails.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    if any((m.controller, m.scenario) not in all_traces for m in all_metrics):
        raise ValueError("every plotted metric requires its actual trace")
    saved: list[str] = []
    for scenario in dict.fromkeys(m.scenario for m in all_metrics):
        figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        try:
            for m in (m for m in all_metrics if m.scenario == scenario):
                trace = all_traces[(m.controller, scenario)]
                stride = max(1, len(trace.times) // 5000)
                for axis, values in zip(axes, [trace.positions, trace.errors, trace.controls], strict=True):
                    axis.plot(trace.times[::stride], values[::stride], label=m.controller)
                axes[0].axhline(m.target_z_m, color="black", alpha=0.2)
            for axis, label in zip(axes, ["Position (m)", "Error (m)", "Control (m/s²)"], strict=True):
                axis.set_ylabel(label)
                axis.grid(True, alpha=0.3)
                axis.legend()
            axes[-1].set_xlabel("Observed time (s)")
            figure.suptitle(f"Synthetic vertical proxy: {scenario}")
            figure.tight_layout()
            path = output_dir / ("benchmark_" + re.sub(r"[^a-z0-9_]", "_", scenario.lower().replace(" ", "_")) + ".png")
            figure.savefig(path, dpi=150)
        finally:
            plt.close(figure)
        saved.append(str(path))
    return saved

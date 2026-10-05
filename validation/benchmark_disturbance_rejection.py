# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Benchmark disturbance rejection.

"""Run four actual controllers on synthetic vertical acceleration disturbances.

Public legacy imports remain available here. Implementation responsibilities
live in disturbance_inputs, disturbance_controllers, disturbance_runtime and
disturbance_reports. There is no ITER geometry, density or beta_N evolution.

Usage::

    python validation/benchmark_disturbance_rejection.py --output-dir /tmp/run
    python validation/benchmark_disturbance_rejection.py --output-dir /tmp/run --duration-scale 0.001 --require-complete

Outputs use schema v2 and actual requested/completed horizons. Ordinary exit
zero means reports were written; --require-complete exits two after reports
when a controller is missing or a trajectory fails bounded completion.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _directory in [REPO_ROOT, REPO_ROOT / "src"]:
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from scpn_control.benchmark_records import require_recorded_campaign
from validation.disturbance_controllers import (
    ControllerProtocol as ControllerProtocol,
)
from validation.disturbance_controllers import (
    HInfinityErrorController,
)
from validation.disturbance_controllers import (
    MPCController as MPCController,
)
from validation.disturbance_controllers import (
    PIDController as PIDController,
)
from validation.disturbance_controllers import (
    SNNControllerWrapper as SNNControllerWrapper,
)
from validation.disturbance_controllers import (
    build_controllers as build_controllers,
)
from validation.disturbance_inputs import _positive
from validation.disturbance_reports import (
    generate_json_results as generate_json_results,
)
from validation.disturbance_reports import (
    generate_markdown_report as generate_markdown_report,
)
from validation.disturbance_reports import (
    generate_stdout_table as generate_stdout_table,
)
from validation.disturbance_reports import (
    save_overlay_plots as save_overlay_plots,
)
from validation.disturbance_runtime import (
    DT as DT,
)
from validation.disturbance_runtime import (
    SCENARIO_DURATIONS as SCENARIO_DURATIONS,
)
from validation.disturbance_runtime import (
    SCENARIOS as SCENARIOS,
)
from validation.disturbance_runtime import (
    LinearPlant as LinearPlant,
)
from validation.disturbance_runtime import (
    ScenarioMetrics as ScenarioMetrics,
)
from validation.disturbance_runtime import (
    TraceData as TraceData,
)
from validation.disturbance_runtime import (
    _compute_settling_time as _compute_settling_time,
)
from validation.disturbance_runtime import (
    _disturbance_density_ramp as _disturbance_density_ramp,
)
from validation.disturbance_runtime import (
    _disturbance_elm_pacing as _disturbance_elm_pacing,
)
from validation.disturbance_runtime import (
    _disturbance_vde as _disturbance_vde,
)
from validation.disturbance_runtime import (
    run_scenario as run_scenario,
)
from validation.report_output_paths import checked_report_destination

_build_hinf_controller = HInfinityErrorController


def main(output_dir: str | None = None, *, duration_scale: float = 1.0, require_complete: bool = False) -> None:
    """Run the real controller cohort and write truthful finite JSON/Markdown/PNG.

    Parameters
    ----------
    output_dir : str or None
        Caller path relative to cwd; default is repo artifacts, subject to
        recorded-campaign custody. JSON/Markdown/plots are sequential writes.
    duration_scale : float
        Positive common duration multiplier. Each duration must remain a
        positive integral number of fixed 100 microsecond Euler intervals.
    require_complete : bool
        If true, raise SystemExit(2) after reports when any expected controller
        is absent or any trajectory fails finite/bounded completion.

    Raises
    ------
    ValueError, RuntimeError, OSError
        Invalid configuration/output alias, campaign custody refusal,
        numerical/provider failure or filesystem error. No snapshot, lock,
        atomic report-pair write or physical-reference admission is promised.
    """
    scale = _positive(duration_scale, "duration_scale")
    if not isinstance(require_complete, bool):
        raise ValueError("require_complete must be boolean")
    out = REPO_ROOT / "artifacts" if output_dir is None else Path(output_dir)
    outputs = [
        out / "benchmark_disturbance_rejection.json",
        out / "benchmark_disturbance_rejection.md",
        *[out / ("benchmark_" + name.lower().replace(" ", "_") + ".png") for name in SCENARIOS],
    ]
    sources = [
        Path(__file__),
        *[
            REPO_ROOT / "validation" / (name + ".py")
            for name in ["disturbance_inputs", "disturbance_controllers", "disturbance_runtime", "disturbance_reports"]
        ],
        REPO_ROOT / "src/scpn_control/control/h_infinity_controller.py",
        REPO_ROOT / "src/scpn_control/control/neuro_cybernetic_controller.py",
    ]
    for output in outputs:
        checked_report_destination(output, inputs=[*sources, *[p for p in outputs if p != output]])
    require_recorded_campaign(*outputs, repository_root=REPO_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    controllers = build_controllers()
    metrics: list[ScenarioMetrics] = []
    traces: dict[tuple[str, str], TraceData] = {}
    print("Synthetic disturbance rejection; actual controllers: " + ", ".join(controllers))
    for name, cfg in SCENARIOS.items():
        selected = dict(cfg)
        selected["duration_s"] = float(SCENARIO_DURATIONS[name]) * scale
        for label, controller in controllers.items():
            row, trace = run_scenario(label, controller, name, selected)
            metrics.append(row)
            traces[(label, name)] = trace
    report = generate_json_results(metrics)
    report["controller_runtime"] = {
        label: {
            "implementation": type(controller).__module__ + "." + type(controller).__name__,
            "backend": controller.backend if isinstance(controller, SNNControllerWrapper) else "python",
        }
        for label, controller in controllers.items()
    }
    report["duration_scale"] = scale
    outputs[0].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    outputs[1].write_text(generate_markdown_report(metrics), encoding="utf-8")
    save_overlay_plots(traces, metrics, out)
    print(generate_stdout_table(metrics))
    print("Reports written to: " + str(out))
    if require_complete and not (report["complete_controller_cohort"] and report["all_runs_completed_bounded"]):
        raise SystemExit(2)


def cli(argv: Sequence[str] | None = None) -> int:
    """Parse the actual command line; return a fixed refusal for run/output errors.

    Parameters
    ----------
    argv : sequence of str or None
        Explicit arguments, or the process command line.

    Returns
    -------
    int
        Zero after ordinary reporting; two for authored run/output refusal.
        Argparse and strict-cohort SystemExit(2) retain their real exit code.
    """
    parser = argparse.ArgumentParser(description="Synthetic vertical-proxy disturbance benchmark.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    try:
        main(args.output_dir, duration_scale=args.duration_scale, require_complete=args.require_complete)
    except (ValueError, RuntimeError, OSError):
        print("Benchmark configuration, execution or output destination was refused.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(cli())

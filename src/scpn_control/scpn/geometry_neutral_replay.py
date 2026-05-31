# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Geometry-Neutral Replay
"""Compact geometry-neutral stellarator replay through the SCPN controller."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np

from scpn_control.core.stellarator_geometry import (
    StellaratorConfig,
    effective_ripple,
    stellarator_flux_surface,
    w7x_config,
)
from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.contracts import ControlScales, ControlTargets, FeatureAxisSpec
from scpn_control.scpn.controller import NeuroSymbolicController
from scpn_control.scpn.geometry_neutral_contracts import (
    ActuatorChannel,
    ActuatorSet,
    ControlObjective,
    DiagnosticChannel,
    DiagnosticFrame,
    MagneticConfiguration,
    ReplayScenario,
)
from scpn_control.scpn.structure import StochasticPetriNet

SCHEMA_VERSION = "scpn-control.geometry-neutral-replay.v1"
GEOMETRY_NEUTRAL_REPLAY_SCHEMA_VERSION = SCHEMA_VERSION
MANIFEST_SCHEMA_VERSION = "scpn-control.geometry-neutral-replay-manifest.v1"
GEOMETRY_NEUTRAL_REPLAY_MANIFEST_SCHEMA_VERSION = MANIFEST_SCHEMA_VERSION
DEFAULT_THRESHOLDS: dict[str, float] = {
    "max_final_fieldline_spread": 0.026,
    "min_improvement_fraction": 0.20,
    "max_abs_current_A": 1200.0,
    "max_p95_latency_us": 1000.0,
}


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _signature(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()[:16]


def _digest(payload: Any) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _require_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_manifest_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _require_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _fieldline_spread(config: StellaratorConfig, current_A: float) -> float:
    _, _, field = stellarator_flux_surface(config, s=0.72, n_theta=32, n_phi=40)
    base = float(np.std(field / np.mean(field))) + 0.43 * effective_ripple(config, 0.72)
    controlled = base - 0.000041 * abs(float(current_A))
    return float(max(0.009, controlled))


def _build_controller() -> NeuroSymbolicController:
    net = StochasticPetriNet()
    net.add_place("spread_low", initial_tokens=0.0)
    net.add_place("spread_high", initial_tokens=0.0)
    net.add_place("trim_pos", initial_tokens=0.0)
    net.add_place("trim_neg", initial_tokens=0.0)
    net.add_transition("increase_trim", threshold=0.01)
    net.add_transition("decrease_trim", threshold=0.01)
    net.add_arc("spread_high", "increase_trim", weight=1.0)
    net.add_arc("increase_trim", "trim_pos", weight=1.0)
    net.add_arc("spread_low", "decrease_trim", weight=1.0)
    net.add_arc("decrease_trim", "trim_neg", weight=1.0)
    net.compile()

    artifact = (
        FusionCompiler(bitstream_length=512, seed=101)
        .compile(net, firing_mode="binary")
        .export_artifact(
            name="geometry_neutral_stellarator_replay",
            dt_control_s=0.001,
            injection_config=[
                {
                    "place_id": 0,
                    "source": "spread_low",
                    "scale": 1.0,
                    "offset": 0.0,
                    "clamp_0_1": True,
                },
                {
                    "place_id": 1,
                    "source": "spread_high",
                    "scale": 1.0,
                    "offset": 0.0,
                    "clamp_0_1": True,
                },
            ],
            readout_config={
                "actions": [{"name": "helical_trim_A", "pos_place": 2, "neg_place": 3}],
                "gains": [1200.0],
                "abs_max": [1200.0],
                "slew_per_s": [4.0e5],
            },
        )
    )
    return NeuroSymbolicController(
        artifact=artifact,
        seed_base=314159,
        targets=ControlTargets(R_target_m=0.015, Z_target_m=0.0),
        scales=ControlScales(R_scale_m=0.012, Z_scale_m=1.0),
        feature_axes=[
            FeatureAxisSpec(
                obs_key="fieldline_spread",
                target=0.015,
                scale=0.012,
                pos_key="spread_low",
                neg_key="spread_high",
            )
        ],
        runtime_profile="deterministic",
        sc_binary_margin=0.0,
    )


def _scenario(*, steps: int, seed: int) -> ReplayScenario:
    steps_i = _require_int("steps", steps)
    seed_i = _require_int("seed", seed)
    config = w7x_config()
    actuator = ActuatorChannel(
        name="helical_trim_A",
        unit="A",
        min_value=-1200.0,
        max_value=1200.0,
        slew_rate_per_s=4.0e5,
        latency_steps=1,
        failure_mode="stuck_supported",
    )
    return ReplayScenario(
        name="geometry_neutral_public_stellarator_replay",
        seed=seed_i,
        steps=steps_i,
        dt_s=0.001,
        magnetic_configuration=MagneticConfiguration(
            name="public_w7x_like_reduced_order",
            device_class="stellarator",
            field_periods=config.N_fp,
            coordinate_system="boozer",
            reference="public synthetic W7-X-like reduced-order fixture",
        ),
        actuator_set=ActuatorSet(channels=(actuator,)),
        objective=ControlObjective(
            target_metrics={"fieldline_spread": 0.015},
            weights={"fieldline_spread": 1.0, "actuator_margin": 0.10},
            constraints={"max_abs_current_A": 1200.0},
        ),
        initial_frame=DiagnosticFrame(
            step=0,
            time_s=0.0,
            channels=(
                DiagnosticChannel(
                    name="fieldline_spread",
                    value=_fieldline_spread(config, 0.0),
                    unit="rad",
                    sigma=0.002,
                    provenance="public_synthetic",
                ),
                DiagnosticChannel(
                    name="effective_ripple",
                    value=effective_ripple(config, 0.72),
                    unit="dimensionless",
                    sigma=0.0008,
                    provenance="public_synthetic",
                ),
            ),
        ),
        fault_schedule={max(3, (2 * steps_i) // 3): {"helical_trim_A": "stuck"}},
    )


def _run_once(scenario: ReplayScenario) -> dict[str, Any]:
    config = w7x_config()
    controller = _build_controller()
    actuator = scenario.actuator_set.by_name("helical_trim_A")
    rng = np.random.default_rng(int(scenario.seed))
    previous_requested = 0.0
    applied_current = 0.0
    delayed = [0.0 for _ in range(int(actuator.latency_steps) + 1)]
    stuck_value: float | None = None
    trace: list[dict[str, Any]] = []

    for step in range(int(scenario.steps)):
        fault = scenario.fault_schedule.get(step, {}).get("helical_trim_A")
        if fault == "stuck" and stuck_value is None:
            stuck_value = applied_current

        fieldline_spread = _fieldline_spread(config, applied_current)
        fieldline_spread = float(max(0.0, fieldline_spread + rng.normal(0.0, 0.0001)))
        obs = {
            "fieldline_spread": fieldline_spread,
            "effective_ripple": effective_ripple(config, 0.72),
        }
        raw_action = float(cast(Mapping[str, float], controller.step(obs, step)).get("helical_trim_A", 0.0))
        if stuck_value is None:
            requested = actuator.apply_slew(
                previous=previous_requested,
                requested=raw_action,
                dt_s=float(scenario.dt_s),
            )
        else:
            requested = stuck_value
        previous_requested = requested
        delayed.append(requested)
        applied_current = float(delayed.pop(0))
        latency_us = float(120.0 + 4.0 * (step % 4) + 0.006 * abs(applied_current))
        trace.append(
            {
                "step": int(step),
                "time_s": round(float(step * scenario.dt_s), 9),
                "fieldline_spread": round(fieldline_spread, 9),
                "effective_ripple": round(float(obs["effective_ripple"]), 9),
                "requested_current_A": round(float(requested), 6),
                "applied_current_A": round(float(applied_current), 6),
                "latency_us": round(latency_us, 6),
                "fault_active": stuck_value is not None,
            }
        )

    signature = _signature({"scenario": scenario.to_dict(), "trace": trace})
    initial = float(trace[0]["fieldline_spread"])
    final = float(trace[-1]["fieldline_spread"])
    max_abs_current = max(abs(float(row["applied_current_A"])) for row in trace)
    p95_latency = float(np.percentile([float(row["latency_us"]) for row in trace], 95))
    return {
        "trace": trace,
        "signature": signature,
        "metrics": {
            "initial_fieldline_spread": round(initial, 9),
            "final_fieldline_spread": round(final, 9),
            "improvement_fraction": round((initial - final) / max(initial, 1e-12), 9),
            "max_abs_current_A": round(float(max_abs_current), 6),
            "p95_latency_us": round(p95_latency, 6),
        },
    }


def _passes(metrics: Mapping[str, float], thresholds: Mapping[str, float]) -> bool:
    return bool(
        float(metrics["final_fieldline_spread"]) <= float(thresholds["max_final_fieldline_spread"])
        and float(metrics["improvement_fraction"]) >= float(thresholds["min_improvement_fraction"])
        and float(metrics["max_abs_current_A"]) <= float(thresholds["max_abs_current_A"])
        and float(metrics["p95_latency_us"]) <= float(thresholds["max_p95_latency_us"])
    )


def _build_manifest(
    *,
    scenario_payload: Mapping[str, Any],
    trace: list[dict[str, Any]],
    metrics: Mapping[str, float],
    thresholds: Mapping[str, float],
    deterministic: bool,
    passes_thresholds: bool,
) -> dict[str, Any]:
    magnetic_configuration = _require_mapping(
        "scenario.magnetic_configuration",
        scenario_payload["magnetic_configuration"],
    )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "scenario_digest": _digest(scenario_payload),
        "trace_digest": _digest(trace),
        "metrics_digest": _digest(metrics),
        "thresholds_digest": _digest(thresholds),
        "acceptance": {
            "deterministic": bool(deterministic),
            "passes_thresholds": bool(passes_thresholds),
        },
        "provenance": {
            "magnetic_configuration_reference": magnetic_configuration["reference"],
            "actuator_calibration": "declared bounded helical-trim current contract",
            "latency_model": "deterministic affine replay latency model",
            "fault_model": "declared stuck-actuator schedule",
            "acceptance_threshold_source": "repository geometry-neutral replay thresholds",
        },
    }


def generate_report(*, steps: int = 12, seed: int = 314159) -> dict[str, Any]:
    """Generate a deterministic compact SCPN-control replay report."""
    steps_i = _require_int("steps", steps)
    seed_i = _require_int("seed", seed)
    if steps_i < 4:
        raise ValueError("steps must be >= 4.")
    scenario = _scenario(steps=steps_i, seed=seed_i)
    run_a = _run_once(scenario)
    run_b = _run_once(scenario)
    deterministic = run_a["signature"] == run_b["signature"]
    metrics = run_a["metrics"]
    scenario_payload = scenario.to_dict()
    thresholds = dict(DEFAULT_THRESHOLDS)
    passes_thresholds = bool(deterministic and _passes(metrics, DEFAULT_THRESHOLDS))
    manifest = _build_manifest(
        scenario_payload=scenario_payload,
        trace=run_a["trace"],
        metrics=metrics,
        thresholds=thresholds,
        deterministic=bool(deterministic),
        passes_thresholds=passes_thresholds,
    )
    report = {
        "geometry_neutral_replay": {
            "schema_version": SCHEMA_VERSION,
            "scenario": scenario_payload,
            "replay": {
                "deterministic": bool(deterministic),
                "signature": run_a["signature"],
                "trace": run_a["trace"],
            },
            "magnetic_configuration": scenario.magnetic_configuration.to_dict(),
            "metrics": metrics,
            "thresholds": thresholds,
            "passes_thresholds": passes_thresholds,
            "manifest": manifest,
            "limitations": [
                "This compact replay is not a production PCS.",
                "No external company data is used.",
                "The replay validates geometry-neutral SCPN control plumbing and actuator constraints.",
            ],
        }
    }
    validate_report(report)
    return report


def generate_geometry_neutral_report(*, steps: int = 12, seed: int = 314159) -> dict[str, Any]:
    """Public package alias for :func:`generate_report`."""
    return generate_report(steps=steps, seed=seed)


def validate_report(report: Mapping[str, Any]) -> None:
    """Validate the compact replay report contract without external packages."""
    if "geometry_neutral_replay" not in report:
        raise ValueError("missing geometry_neutral_replay")
    bench = report["geometry_neutral_replay"]
    required = (
        "schema_version",
        "scenario",
        "replay",
        "magnetic_configuration",
        "metrics",
        "thresholds",
        "passes_thresholds",
        "manifest",
        "limitations",
    )
    for key in required:
        if key not in bench:
            raise ValueError(f"missing geometry_neutral_replay.{key}")
    if bench["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unexpected schema_version")
    replay = _require_mapping("geometry_neutral_replay.replay", bench["replay"])
    scenario = _require_mapping("geometry_neutral_replay.scenario", bench["scenario"])
    metrics = _require_mapping("geometry_neutral_replay.metrics", bench["metrics"])
    thresholds = _require_mapping("geometry_neutral_replay.thresholds", bench["thresholds"])
    manifest = _require_mapping("geometry_neutral_replay.manifest", bench["manifest"])
    if not replay["deterministic"]:
        raise ValueError("replay must be deterministic")
    for key in (
        "initial_fieldline_spread",
        "final_fieldline_spread",
        "improvement_fraction",
        "max_abs_current_A",
        "p95_latency_us",
    ):
        value = float(metrics[key])
        if not np.isfinite(value):
            raise ValueError(f"metric must be finite: {key}")
    if not _passes(cast(Mapping[str, float], metrics), cast(Mapping[str, float], thresholds)):
        raise ValueError("metrics do not satisfy declared thresholds")
    if bool(bench["passes_thresholds"]) is not True:
        raise ValueError("passes_thresholds must be true for an admissible replay report")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unexpected manifest schema_version")
    expected_digests = {
        "scenario_digest": _digest(scenario),
        "trace_digest": _digest(replay["trace"]),
        "metrics_digest": _digest(metrics),
        "thresholds_digest": _digest(thresholds),
    }
    for key, expected in expected_digests.items():
        if manifest.get(key) != expected:
            label = key.replace("_", " ")
            raise ValueError(f"manifest {label} mismatch")
    acceptance = _require_mapping("geometry_neutral_replay.manifest.acceptance", manifest["acceptance"])
    if bool(acceptance.get("deterministic")) != bool(replay["deterministic"]):
        raise ValueError("manifest acceptance deterministic mismatch")
    if bool(acceptance.get("passes_thresholds")) != bool(bench["passes_thresholds"]):
        raise ValueError("manifest acceptance threshold mismatch")
    provenance = _require_mapping("geometry_neutral_replay.manifest.provenance", manifest["provenance"])
    for key in (
        "magnetic_configuration_reference",
        "actuator_calibration",
        "latency_model",
        "fault_model",
        "acceptance_threshold_source",
    ):
        _require_manifest_text(f"manifest provenance {key}", provenance[key])


def validate_geometry_neutral_report(report: Mapping[str, Any]) -> None:
    """Public package alias for :func:`validate_report`."""
    validate_report(report)


def render_markdown(report: Mapping[str, Any]) -> str:
    bench = report["geometry_neutral_replay"]
    metrics = bench["metrics"]
    lines = [
        "# Geometry-Neutral Stellarator Replay",
        "",
        f"- Schema: `{bench['schema_version']}`",
        f"- Deterministic replay: `{bench['replay']['deterministic']}`",
        f"- Replay signature: `{bench['replay']['signature']}`",
        f"- Threshold pass: `{'YES' if bench['passes_thresholds'] else 'NO'}`",
        "",
        "## Metrics",
        "",
        f"- Initial field-line spread: `{metrics['initial_fieldline_spread']:.6f}`",
        f"- Final field-line spread: `{metrics['final_fieldline_spread']:.6f}`",
        f"- Improvement fraction: `{metrics['improvement_fraction']:.6f}`",
        f"- Max absolute current: `{metrics['max_abs_current_A']:.3f} A`",
        f"- P95 latency: `{metrics['p95_latency_us']:.3f} us`",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in bench["limitations"])
    lines.append("")
    return "\n".join(lines)


def render_geometry_neutral_markdown(report: Mapping[str, Any]) -> str:
    """Public package alias for :func:`render_markdown`."""
    return render_markdown(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--output-json", default="artifacts/geometry_neutral_replay.json")
    parser.add_argument("--output-md", default="artifacts/geometry_neutral_replay.md")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(steps=args.steps, seed=args.seed)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    bench = report["geometry_neutral_replay"]
    metrics = bench["metrics"]
    print("Geometry-neutral stellarator replay complete.")
    print(
        "deterministic={deterministic}, final_fieldline_spread={final:.6f}, "
        "max_abs_current_A={current:.3f}, passes_thresholds={passes}".format(
            deterministic=bench["replay"]["deterministic"],
            final=metrics["final_fieldline_spread"],
            current=metrics["max_abs_current_A"],
            passes=bench["passes_thresholds"],
        )
    )
    if args.strict and not bench["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Geometry-neutral replay reports and schema admission.
"""Compact geometry-neutral stellarator replay through the SCPN controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
SCHEMA_VERSION_V1_1 = "scpn-control.geometry-neutral-replay.v1.1"
GEOMETRY_NEUTRAL_REPLAY_SCHEMA_VERSION = SCHEMA_VERSION
GEOMETRY_NEUTRAL_REPLAY_SCHEMA_VERSION_V1_1 = SCHEMA_VERSION_V1_1
SUPPORTED_REPLAY_SCHEMA_VERSIONS = (SCHEMA_VERSION, SCHEMA_VERSION_V1_1)
MANIFEST_SCHEMA_VERSION = "scpn-control.geometry-neutral-replay-manifest.v1"
GEOMETRY_NEUTRAL_REPLAY_MANIFEST_SCHEMA_VERSION = MANIFEST_SCHEMA_VERSION
EVIDENCE_SCHEMA_VERSION = "scpn-control.geometry-neutral-replay-evidence.v1"
GEOMETRY_NEUTRAL_REPLAY_EVIDENCE_SCHEMA_VERSION = EVIDENCE_SCHEMA_VERSION
AER_ADMISSION_SCHEMA_VERSION = "scpn-control.geometry-neutral-replay-aer-admission.v1"
GEOMETRY_NEUTRAL_REPLAY_AER_ADMISSION_SCHEMA_VERSION = AER_ADMISSION_SCHEMA_VERSION
GEOMETRY_NEUTRAL_REPLAY_BOUNDED = "bounded_synthetic_geometry_neutral_replay_only"
GEOMETRY_NEUTRAL_REPLAY_QUALIFIED = "qualified_geometry_neutral_replay_evidence"
DEFAULT_THRESHOLDS: dict[str, float] = {
    "max_final_fieldline_spread": 0.026,
    "min_improvement_fraction": 0.20,
    "max_abs_current_A": 1200.0,
    "max_p95_latency_us": 1000.0,
}


@dataclass(frozen=True)
class GeometryNeutralReplayEvidence:
    """Tamper-evident geometry-neutral replay evidence admission object."""

    schema_version: str
    generated_utc: str
    replay_schema_version: str
    replay_report_sha256: str
    scenario_digest: str
    trace_digest: str
    metrics_digest: str
    thresholds_digest: str
    magnetic_configuration_reference: str
    actuator_calibration: str
    latency_model: str
    fault_model: str
    final_fieldline_spread: float
    improvement_fraction: float
    max_abs_current_A: float
    p95_latency_us: float
    deterministic: bool
    passes_thresholds: bool
    measured_or_benchmark_artefact_sha256: str | None
    device_claim_allowed: bool
    claim_status: str
    payload_sha256: str


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _signature(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()[:16]


def _digest(payload: Any) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned["payload_sha256"] = ""
    return _digest(unsigned)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate JSON key in geometry-neutral replay evidence: {key}")
        seen.add(key)
        out[key] = value
    return out


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _require_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _require_finite_nonnegative(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and non-negative")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _require_finite(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_manifest_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _require_optional_sha256(name: str, value: Any) -> str | None:
    if value is None:
        return None
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return str(value)


def _require_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _require_nonnegative_int(name: str, value: Any) -> int:
    result = _require_int(name, value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _require_positive_int(name: str, value: Any) -> int:
    result = _require_int(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _require_uuid_text(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid UUID") from exc
    return str(parsed)


def load_replay_schema(version: str) -> dict[str, Any]:
    """Load a bundled geometry-neutral replay JSON Schema document."""
    if version == SCHEMA_VERSION:
        filename = "v1.json"
    elif version == SCHEMA_VERSION_V1_1:
        filename = "v1_1.json"
    else:
        raise ValueError("unsupported geometry-neutral replay schema version")
    path = Path(__file__).with_name("replay_schemas") / filename
    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(payload, dict):
        raise ValueError("replay schema document must be a JSON object")
    return payload


def register_v1_1_schema() -> dict[str, Any]:
    """Return the bundled v1.1 replay schema after self-identification checks."""
    schema = load_replay_schema(SCHEMA_VERSION_V1_1)
    if schema.get("$id") != SCHEMA_VERSION_V1_1:
        raise ValueError("v1.1 replay schema id mismatch")
    return schema


def assert_v1_replay_loadable_under_v1_1_schema_bundle(report: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate a v1 replay report while the v1.1 schema bundle is installed."""
    register_v1_1_schema()
    validate_report(report)
    bench = _require_mapping("geometry_neutral_replay", report["geometry_neutral_replay"])
    if bench.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("expected a v1 replay report for v1.1 back-compatibility admission")
    return report


def build_aer_admission_metadata(
    *,
    admission_report: Mapping[str, Any],
    decode_strategy: str,
    decode_window_ns: int,
    n_features: int,
    feature_normalisation: str = "unit",
    require_monotonic: bool = False,
    feature_vector: object | None = None,
) -> dict[str, Any]:
    """Build replay-safe AER admission metadata from a decoded observation."""

    report = _require_mapping("admission_report", admission_report)
    payload: dict[str, Any] = {
        "schema_version": AER_ADMISSION_SCHEMA_VERSION,
        "capacity": _require_positive_int("aer_admission.capacity", report.get("capacity")),
        "retained_events": _require_nonnegative_int(
            "aer_admission.retained_events",
            report.get("retained_events"),
        ),
        "overflowed": _require_bool("aer_admission.overflowed", report.get("overflowed")),
        "monotonic_input": _require_bool(
            "aer_admission.monotonic_input",
            report.get("monotonic_input"),
        ),
        "out_of_order_event_count": _require_nonnegative_int(
            "aer_admission.out_of_order_event_count",
            report.get("out_of_order_event_count"),
        ),
        "decode_strategy": str(decode_strategy),
        "decode_window_ns": _require_positive_int("aer_admission.decode_window_ns", decode_window_ns),
        "n_features": _require_positive_int("aer_admission.n_features", n_features),
        "feature_normalisation": str(feature_normalisation),
        "require_monotonic": _require_bool("aer_admission.require_monotonic", require_monotonic),
    }
    if feature_vector is not None:
        features = np.asarray(feature_vector, dtype=np.float64)
        if features.shape != (payload["n_features"],):
            raise ValueError("aer_admission.feature_vector must match n_features")
        if not np.all(np.isfinite(features)):
            raise ValueError("aer_admission.feature_vector must be finite")
        payload["feature_count"] = int(features.size)
        payload["feature_vector_sha256"] = _digest([float(value) for value in features.tolist()])
    return _validate_aer_admission_metadata(payload)


def attach_aer_admission_metadata(
    report: Mapping[str, Any],
    aer_admission: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach digest-bound AER admission metadata to a replay v1.1 report."""

    validate_report(report)
    payload = deepcopy(dict(report))
    bench = _require_mapping("geometry_neutral_replay", payload["geometry_neutral_replay"])
    bench_mut = cast(dict[str, Any], bench)
    metadata = _validate_aer_admission_metadata(aer_admission)
    bench_mut["schema_version"] = SCHEMA_VERSION_V1_1
    bench_mut["aer_admission"] = metadata
    manifest = _require_mapping("geometry_neutral_replay.manifest", bench_mut["manifest"])
    cast(dict[str, Any], manifest)["aer_admission_digest"] = _digest(metadata)
    validate_report(payload)
    return payload


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


def _validate_aer_admission_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(_require_mapping("aer_admission", value))
    if metadata.get("schema_version") != AER_ADMISSION_SCHEMA_VERSION:
        raise ValueError("aer_admission.schema_version is unsupported")
    capacity = _require_positive_int("aer_admission.capacity", metadata.get("capacity"))
    retained_events = _require_nonnegative_int(
        "aer_admission.retained_events",
        metadata.get("retained_events"),
    )
    if retained_events > capacity:
        raise ValueError("aer_admission.retained_events must not exceed capacity")
    overflowed = _require_bool("aer_admission.overflowed", metadata.get("overflowed"))
    monotonic_input = _require_bool(
        "aer_admission.monotonic_input",
        metadata.get("monotonic_input"),
    )
    out_of_order_event_count = _require_nonnegative_int(
        "aer_admission.out_of_order_event_count",
        metadata.get("out_of_order_event_count"),
    )
    if monotonic_input and out_of_order_event_count != 0:
        raise ValueError("aer_admission.monotonic_input conflicts with out_of_order_event_count")
    if (not monotonic_input) and out_of_order_event_count == 0:
        raise ValueError("aer_admission non-monotonic streams must record out_of_order_event_count")
    decode_strategy = metadata.get("decode_strategy")
    if decode_strategy not in ("rate", "temporal", "isi"):
        raise ValueError("aer_admission.decode_strategy must be one of: rate, temporal, isi")
    decode_window_ns = _require_positive_int(
        "aer_admission.decode_window_ns",
        metadata.get("decode_window_ns"),
    )
    n_features = _require_positive_int("aer_admission.n_features", metadata.get("n_features"))
    feature_normalisation = metadata.get("feature_normalisation")
    if feature_normalisation not in ("unit", "max", "zscore"):
        raise ValueError("aer_admission.feature_normalisation must be one of: unit, max, zscore")
    require_monotonic = _require_bool(
        "aer_admission.require_monotonic",
        metadata.get("require_monotonic"),
    )
    if require_monotonic and not monotonic_input:
        raise ValueError("aer_admission strict monotonic replay cannot admit non-monotonic input")
    feature_count = metadata.get("feature_count")
    if feature_count is not None:
        if _require_nonnegative_int("aer_admission.feature_count", feature_count) != n_features:
            raise ValueError("aer_admission.feature_count must match n_features")
    feature_digest = _require_optional_sha256(
        "aer_admission.feature_vector_sha256",
        metadata.get("feature_vector_sha256"),
    )
    if feature_digest is not None and feature_count is None:
        raise ValueError("aer_admission.feature_vector_sha256 requires feature_count")
    return {
        "schema_version": AER_ADMISSION_SCHEMA_VERSION,
        "capacity": capacity,
        "retained_events": retained_events,
        "overflowed": overflowed,
        "monotonic_input": monotonic_input,
        "out_of_order_event_count": out_of_order_event_count,
        "decode_strategy": str(decode_strategy),
        "decode_window_ns": decode_window_ns,
        "n_features": n_features,
        "feature_normalisation": str(feature_normalisation),
        "require_monotonic": require_monotonic,
        **({} if feature_count is None else {"feature_count": int(feature_count)}),
        **({} if feature_digest is None else {"feature_vector_sha256": feature_digest}),
    }


def _validate_v1_1_extensions(bench: Mapping[str, Any]) -> None:
    if "pulse_id" in bench:
        _require_uuid_text("pulse_id", bench["pulse_id"])

    if "capacitor_state_initial_J" in bench:
        _require_finite_nonnegative(
            "capacitor_state_initial_J",
            bench["capacitor_state_initial_J"],
        )

    if "trigger_timestamp_ns" in bench:
        _require_nonnegative_int("trigger_timestamp_ns", bench["trigger_timestamp_ns"])

    if "energy_recovered_J" in bench:
        _require_finite_nonnegative("energy_recovered_J", bench["energy_recovered_J"])

    if "shot_phase_log" in bench:
        phase_log = bench["shot_phase_log"]
        if not isinstance(phase_log, list):
            raise ValueError("shot_phase_log must be a list")
        previous_t = -math.inf
        for index, row in enumerate(phase_log):
            if not isinstance(row, Mapping):
                raise ValueError(f"shot_phase_log[{index}] must be an object")
            t = _require_finite(f"shot_phase_log[{index}].t", row.get("t"))
            if t < 0.0:
                raise ValueError(f"shot_phase_log[{index}].t must be non-negative")
            if t < previous_t:
                raise ValueError("shot_phase_log entries must be sorted by t")
            previous_t = t
            state = row.get("state")
            if not isinstance(state, str) or not state.strip():
                raise ValueError(f"shot_phase_log[{index}].state must be non-empty text")
            reason = row.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"shot_phase_log[{index}].reason must be non-empty text")

    if "frc_diagnostics" in bench:
        diagnostics = _require_mapping("frc_diagnostics", bench["frc_diagnostics"])
        if "s_parameter_at_burn" in diagnostics:
            _require_finite_nonnegative(
                "frc_diagnostics.s_parameter_at_burn",
                diagnostics["s_parameter_at_burn"],
            )
        if "mrti_peak_amplitude_m" in diagnostics:
            _require_finite_nonnegative(
                "frc_diagnostics.mrti_peak_amplitude_m",
                diagnostics["mrti_peak_amplitude_m"],
            )
        if "tilt_growth_rate_s_inv" in diagnostics:
            _require_finite(
                "frc_diagnostics.tilt_growth_rate_s_inv",
                diagnostics["tilt_growth_rate_s_inv"],
            )

    manifest = _require_mapping("geometry_neutral_replay.manifest", bench["manifest"])
    if "aer_admission" in bench:
        metadata = _validate_aer_admission_metadata(bench["aer_admission"])
        expected_digest = _digest(metadata)
        if manifest.get("aer_admission_digest") != expected_digest:
            raise ValueError("manifest aer admission digest mismatch")
    elif "aer_admission_digest" in manifest:
        raise ValueError("manifest aer admission digest requires aer_admission metadata")


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
    schema_version = bench["schema_version"]
    if schema_version not in SUPPORTED_REPLAY_SCHEMA_VERSIONS:
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
    if schema_version == SCHEMA_VERSION_V1_1:
        _validate_v1_1_extensions(bench)


def validate_geometry_neutral_report(report: Mapping[str, Any]) -> None:
    """Public package alias for :func:`validate_report`."""
    validate_report(report)


def save_geometry_neutral_replay_report(
    report: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Persist a validated geometry-neutral replay report as stable JSON."""
    validate_report(report)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_stable_json(report), encoding="utf-8")
    return path


def load_geometry_neutral_replay_report(path: str | Path) -> dict[str, Any]:
    """Load and validate a geometry-neutral replay report with duplicate-key checks."""
    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(payload, dict):
        raise ValueError("geometry-neutral replay report must be a JSON object")
    validate_report(payload)
    return payload


def _validate_geometry_neutral_replay_evidence_payload(
    payload: Mapping[str, Any],
    *,
    require_device_claim: bool,
) -> GeometryNeutralReplayEvidence:
    if payload.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise ValueError("geometry-neutral replay evidence schema_version is unsupported")
    declared_digest = payload.get("payload_sha256")
    if not _is_sha256(declared_digest):
        raise ValueError("geometry-neutral replay evidence payload_sha256 must be a SHA-256 hex digest")
    if declared_digest != _payload_sha256(payload):
        raise ValueError("geometry-neutral replay evidence payload_sha256 does not match payload")
    generated_utc = payload.get("generated_utc")
    if not isinstance(generated_utc, str) or not generated_utc.endswith("Z"):
        raise ValueError("geometry-neutral replay evidence generated_utc must be a UTC timestamp ending in Z")
    replay_schema_version = payload.get("replay_schema_version")
    if replay_schema_version not in SUPPORTED_REPLAY_SCHEMA_VERSIONS:
        raise ValueError("geometry-neutral replay evidence replay_schema_version is unsupported")
    for name in (
        "replay_report_sha256",
        "scenario_digest",
        "trace_digest",
        "metrics_digest",
        "thresholds_digest",
    ):
        if not _is_sha256(payload.get(name)):
            raise ValueError(f"{name} must be a SHA-256 hex digest")
    measured_digest = payload.get("measured_or_benchmark_artefact_sha256")
    if measured_digest is not None and not _is_sha256(measured_digest):
        raise ValueError("measured_or_benchmark_artefact_sha256 must be a SHA-256 hex digest")
    magnetic_reference = _require_manifest_text(
        "magnetic_configuration_reference",
        payload.get("magnetic_configuration_reference"),
    )
    actuator_calibration = _require_manifest_text("actuator_calibration", payload.get("actuator_calibration"))
    latency_model = _require_manifest_text("latency_model", payload.get("latency_model"))
    fault_model = _require_manifest_text("fault_model", payload.get("fault_model"))
    final_fieldline_spread = _require_finite_nonnegative(
        "final_fieldline_spread",
        payload.get("final_fieldline_spread"),
    )
    improvement_fraction = _require_finite_nonnegative(
        "improvement_fraction",
        payload.get("improvement_fraction"),
    )
    max_abs_current = _require_finite_nonnegative("max_abs_current_A", payload.get("max_abs_current_A"))
    p95_latency = _require_finite_nonnegative("p95_latency_us", payload.get("p95_latency_us"))
    deterministic = _require_bool("deterministic", payload.get("deterministic"))
    passes_thresholds = _require_bool("passes_thresholds", payload.get("passes_thresholds"))
    device_claim_allowed = _require_bool("device_claim_allowed", payload.get("device_claim_allowed"))
    expected_status = GEOMETRY_NEUTRAL_REPLAY_QUALIFIED if device_claim_allowed else GEOMETRY_NEUTRAL_REPLAY_BOUNDED
    if payload.get("claim_status") != expected_status:
        raise ValueError("geometry-neutral replay evidence claim_status does not match device claim state")
    if require_device_claim or device_claim_allowed:
        if not device_claim_allowed:
            raise ValueError("geometry-neutral replay evidence is bounded-only and cannot support device claims")
        if measured_digest is None:
            raise ValueError("device replay claims require measured or benchmark artefact evidence")
        if not deterministic or not passes_thresholds:
            raise ValueError("device replay claims require deterministic threshold-passing replay evidence")
        if "synthetic" in magnetic_reference.lower():
            raise ValueError("device replay claims cannot rely on synthetic magnetic-configuration provenance")
    return GeometryNeutralReplayEvidence(
        schema_version=str(payload["schema_version"]),
        generated_utc=generated_utc,
        replay_schema_version=str(replay_schema_version),
        replay_report_sha256=str(payload["replay_report_sha256"]),
        scenario_digest=str(payload["scenario_digest"]),
        trace_digest=str(payload["trace_digest"]),
        metrics_digest=str(payload["metrics_digest"]),
        thresholds_digest=str(payload["thresholds_digest"]),
        magnetic_configuration_reference=magnetic_reference,
        actuator_calibration=actuator_calibration,
        latency_model=latency_model,
        fault_model=fault_model,
        final_fieldline_spread=final_fieldline_spread,
        improvement_fraction=improvement_fraction,
        max_abs_current_A=max_abs_current,
        p95_latency_us=p95_latency,
        deterministic=deterministic,
        passes_thresholds=passes_thresholds,
        measured_or_benchmark_artefact_sha256=None if measured_digest is None else str(measured_digest),
        device_claim_allowed=device_claim_allowed,
        claim_status=str(payload["claim_status"]),
        payload_sha256=str(declared_digest),
    )


def geometry_neutral_replay_evidence(
    report: Mapping[str, Any],
    *,
    generated_utc: str | None = None,
    measured_or_benchmark_artefact_sha256: str | None = None,
    device_claim_allowed: bool = False,
) -> GeometryNeutralReplayEvidence:
    """Build tamper-evident replay evidence over an admitted replay report."""

    validate_report(report)
    bench = _require_mapping("geometry_neutral_replay", report["geometry_neutral_replay"])
    replay = _require_mapping("geometry_neutral_replay.replay", bench["replay"])
    metrics = _require_mapping("geometry_neutral_replay.metrics", bench["metrics"])
    manifest = _require_mapping("geometry_neutral_replay.manifest", bench["manifest"])
    provenance = _require_mapping("geometry_neutral_replay.manifest.provenance", manifest["provenance"])
    measured_digest = (
        None if measured_or_benchmark_artefact_sha256 is None else str(measured_or_benchmark_artefact_sha256).lower()
    )
    payload: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "generated_utc": generated_utc or _utc_now(),
        "replay_schema_version": str(bench["schema_version"]),
        "replay_report_sha256": _digest(bench),
        "scenario_digest": str(manifest["scenario_digest"]),
        "trace_digest": str(manifest["trace_digest"]),
        "metrics_digest": str(manifest["metrics_digest"]),
        "thresholds_digest": str(manifest["thresholds_digest"]),
        "magnetic_configuration_reference": str(provenance["magnetic_configuration_reference"]),
        "actuator_calibration": str(provenance["actuator_calibration"]),
        "latency_model": str(provenance["latency_model"]),
        "fault_model": str(provenance["fault_model"]),
        "final_fieldline_spread": float(metrics["final_fieldline_spread"]),
        "improvement_fraction": float(metrics["improvement_fraction"]),
        "max_abs_current_A": float(metrics["max_abs_current_A"]),
        "p95_latency_us": float(metrics["p95_latency_us"]),
        "deterministic": bool(replay["deterministic"]),
        "passes_thresholds": bool(bench["passes_thresholds"]),
        "measured_or_benchmark_artefact_sha256": measured_digest,
        "device_claim_allowed": bool(device_claim_allowed),
        "claim_status": (
            GEOMETRY_NEUTRAL_REPLAY_QUALIFIED if device_claim_allowed else GEOMETRY_NEUTRAL_REPLAY_BOUNDED
        ),
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return _validate_geometry_neutral_replay_evidence_payload(
        payload,
        require_device_claim=bool(device_claim_allowed),
    )


def assert_geometry_neutral_replay_claim_admissible(
    evidence: GeometryNeutralReplayEvidence,
) -> GeometryNeutralReplayEvidence:
    """Fail closed unless replay evidence supports a device-control claim."""

    if not isinstance(evidence, GeometryNeutralReplayEvidence):
        raise ValueError("evidence must be GeometryNeutralReplayEvidence")
    return _validate_geometry_neutral_replay_evidence_payload(asdict(evidence), require_device_claim=True)


def save_geometry_neutral_replay_evidence(
    evidence: GeometryNeutralReplayEvidence,
    output_path: str | Path,
) -> None:
    """Persist geometry-neutral replay evidence as sorted JSON."""

    if not isinstance(evidence, GeometryNeutralReplayEvidence):
        raise ValueError("evidence must be GeometryNeutralReplayEvidence")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_geometry_neutral_replay_evidence(
    path: str | Path,
    *,
    require_device_claim: bool = False,
) -> GeometryNeutralReplayEvidence:
    """Load geometry-neutral replay evidence with duplicate-key and digest admission."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(payload, dict):
        raise ValueError("geometry-neutral replay evidence must be a JSON object")
    return _validate_geometry_neutral_replay_evidence_payload(
        payload,
        require_device_claim=require_device_claim,
    )


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
    ]
    if "aer_admission" in bench:
        aer_admission = _require_mapping("geometry_neutral_replay.aer_admission", bench["aer_admission"])
        lines.extend(
            [
                "## AER Admission",
                "",
                f"- Decode strategy: `{aer_admission['decode_strategy']}`",
                f"- Decode window: `{aer_admission['decode_window_ns']} ns`",
                f"- Feature count: `{aer_admission['n_features']}`",
                f"- Monotonic input: `{aer_admission['monotonic_input']}`",
                f"- Out-of-order events: `{aer_admission['out_of_order_event_count']}`",
                f"- Overflowed: `{aer_admission['overflowed']}`",
                f"- Strict monotonic required: `{aer_admission['require_monotonic']}`",
                "",
            ]
        )
    lines.extend(["## Limitations", ""])
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

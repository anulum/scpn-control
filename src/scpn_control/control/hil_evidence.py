# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Hardware-in-the-Loop Test Harness

"""HIL replay evidence builder and admission."""

from __future__ import annotations

import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from scpn_control.control.hil_evidence_contracts import (
    _LOCAL_CLAIM_STATUS,
    _TARGET_HARDWARE_CLAIM_STATUS,
    HIL_REPLAY_EVIDENCE_BOUNDARY,
    HIL_REPLAY_EVIDENCE_SCHEMA_VERSION,
    _is_hex_sha256,
    _reject_duplicate_json_keys,
    _require_finite_float,
    _require_mapping,
    _require_non_empty_text,
    _require_non_negative_int,
    _require_positive_int,
    _require_qualified_hardware,
    _sha256_json,
    _utc_now_iso,
)
from scpn_control.control.hil_fpga import FPGARegisterMap
from scpn_control.control.hil_loop import ControlLoopMetrics


@dataclass
class HILBenchmarkResult:
    """Benchmark result for HIL sub-ms timing validation."""

    control_metrics: ControlLoopMetrics
    sensor_latency_us: float
    controller_latency_us: float
    actuator_latency_us: float
    total_loop_latency_us: float
    passes_sub_ms: bool
    passes_1khz: bool
    fpga_register_map: FPGARegisterMap | None


def _control_metrics_from_evidence_source(
    metrics: ControlLoopMetrics | HILBenchmarkResult,
) -> ControlLoopMetrics:
    if isinstance(metrics, HILBenchmarkResult):
        return metrics.control_metrics
    if isinstance(metrics, ControlLoopMetrics):
        return metrics
    raise TypeError("metrics must be ControlLoopMetrics or HILBenchmarkResult")


def _hil_timing_payload(metrics: ControlLoopMetrics) -> dict[str, Any]:
    iterations = _require_positive_int(metrics.iterations, "metrics.iterations")
    target_dt_us = _require_finite_float(metrics.target_dt_us, "metrics.target_dt_us", positive=True)
    overrun_count = _require_non_negative_int(metrics.overrun_count, "metrics.overrun_count")
    if overrun_count > iterations:
        raise ValueError("metrics.overrun_count cannot exceed metrics.iterations")
    overrun_fraction = _require_finite_float(metrics.overrun_fraction, "metrics.overrun_fraction")
    expected_fraction = overrun_count / iterations
    if abs(overrun_fraction - expected_fraction) > 1e-12:
        raise ValueError("metrics.overrun_fraction must equal overrun_count / iterations")

    p50 = _require_finite_float(metrics.p50_latency_us, "metrics.p50_latency_us")
    p95 = _require_finite_float(metrics.p95_latency_us, "metrics.p95_latency_us")
    p99 = _require_finite_float(metrics.p99_latency_us, "metrics.p99_latency_us")
    max_latency = _require_finite_float(metrics.max_latency_us, "metrics.max_latency_us")
    min_latency = _require_finite_float(metrics.min_latency_us, "metrics.min_latency_us")
    mean_latency = _require_finite_float(metrics.mean_latency_us, "metrics.mean_latency_us")
    jitter = _require_finite_float(metrics.jitter_std_us, "metrics.jitter_std_us")
    if not (min_latency <= p50 <= p95 <= p99 <= max_latency):
        raise ValueError("latency percentiles must satisfy min <= p50 <= p95 <= p99 <= max")
    if not (min_latency <= mean_latency <= max_latency):
        raise ValueError("mean latency must lie within min/max latency bounds")
    expected_sub_ms = p95 < 1000.0
    if bool(metrics.sub_ms_achieved) is not expected_sub_ms:
        raise ValueError("metrics.sub_ms_achieved must match p95_latency_us < 1000")

    return {
        "iterations": iterations,
        "target_dt_us": target_dt_us,
        "p50_latency_us": p50,
        "p95_latency_us": p95,
        "p99_latency_us": p99,
        "max_latency_us": max_latency,
        "min_latency_us": min_latency,
        "mean_latency_us": mean_latency,
        "jitter_std_us": jitter,
        "overrun_count": overrun_count,
        "overrun_fraction": overrun_fraction,
        "sub_ms_achieved": bool(metrics.sub_ms_achieved),
    }


def hil_replay_evidence(
    metrics: ControlLoopMetrics | HILBenchmarkResult,
    *,
    controller_id: str,
    target_hardware_id: str = "local-unqualified-host",
    target_hardware_class: str = "local-userspace-replay",
    rt_kernel: str = "generic-userspace",
    clock_source: str = "time.perf_counter_ns",
    interlock_events: int = 0,
    backpressure_events: int = 0,
    deployment_claim_allowed: bool = False,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build local HIL replay evidence with claim admission metadata.

    Local workstation timing is useful regression evidence, but it is not
    production deployment evidence. Caller-supplied target hardware metadata
    cannot qualify a deployment claim without independent provenance.
    """
    control_metrics = _control_metrics_from_evidence_source(metrics)
    timing = _hil_timing_payload(control_metrics)
    target_rate_hz = 1e6 / timing["target_dt_us"]
    interlocks = _require_non_negative_int(interlock_events, "interlock_events")
    backpressure = _require_non_negative_int(backpressure_events, "backpressure_events")
    controller = _require_non_empty_text(controller_id, "controller_id")
    hardware = {
        "target_hardware_id": _require_non_empty_text(target_hardware_id, "target_hardware_id"),
        "target_hardware_class": _require_non_empty_text(
            target_hardware_class,
            "target_hardware_class",
        ),
        "rt_kernel": _require_non_empty_text(rt_kernel, "rt_kernel"),
        "clock_source": _require_non_empty_text(clock_source, "clock_source"),
        "target_rate_hz": target_rate_hz,
    }
    claim_status = _TARGET_HARDWARE_CLAIM_STATUS if deployment_claim_allowed else _LOCAL_CLAIM_STATUS
    payload: dict[str, Any] = {
        "schema_version": HIL_REPLAY_EVIDENCE_SCHEMA_VERSION,
        "evidence_boundary": HIL_REPLAY_EVIDENCE_BOUNDARY,
        "generated_at": generated_at or _utc_now_iso(),
        "controller_id": controller,
        "target_hardware": hardware,
        "timing": timing,
        "safety_events": {
            "interlock_events": interlocks,
            "backpressure_events": backpressure,
        },
        "admission": {
            "deployment_claim_allowed": bool(deployment_claim_allowed),
            "claim_status": claim_status,
        },
    }
    payload["replay_digest"] = _sha256_json(
        {
            "controller_id": payload["controller_id"],
            "target_hardware": payload["target_hardware"],
            "timing": payload["timing"],
            "safety_events": payload["safety_events"],
            "admission": payload["admission"],
        }
    )
    payload["payload_sha256"] = _sha256_json(payload)
    return assert_hil_replay_evidence_admissible(
        payload,
        require_target_hardware=bool(deployment_claim_allowed),
    )


def assert_hil_replay_evidence_admissible(
    evidence: Mapping[str, Any],
    *,
    require_target_hardware: bool = False,
) -> dict[str, Any]:
    """Validate a HIL replay evidence payload and return an admitted copy."""
    payload = dict(_require_mapping(evidence, "evidence"))
    supplied_payload_sha = payload.get("payload_sha256")
    if not _is_hex_sha256(supplied_payload_sha):
        raise ValueError("payload_sha256 must be a lowercase SHA-256 hex digest")
    payload_without_sha = dict(payload)
    del payload_without_sha["payload_sha256"]
    expected_payload_sha = _sha256_json(payload_without_sha)
    if not hmac.compare_digest(str(supplied_payload_sha), expected_payload_sha):
        raise ValueError("payload_sha256 does not match HIL replay evidence payload")

    if payload.get("schema_version") != HIL_REPLAY_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported HIL replay evidence schema_version")
    if payload.get("evidence_boundary") != HIL_REPLAY_EVIDENCE_BOUNDARY:
        raise ValueError("unsupported HIL replay evidence boundary")
    _require_non_empty_text(payload.get("generated_at"), "generated_at")
    _require_non_empty_text(payload.get("controller_id"), "controller_id")

    target_hardware = _require_mapping(payload.get("target_hardware"), "target_hardware")
    timing = _require_mapping(payload.get("timing"), "timing")
    safety_events = _require_mapping(payload.get("safety_events"), "safety_events")
    admission = _require_mapping(payload.get("admission"), "admission")

    iterations = _require_positive_int(timing.get("iterations"), "timing.iterations")
    target_dt_us = _require_finite_float(timing.get("target_dt_us"), "timing.target_dt_us", positive=True)
    p50 = _require_finite_float(timing.get("p50_latency_us"), "timing.p50_latency_us")
    p95 = _require_finite_float(timing.get("p95_latency_us"), "timing.p95_latency_us")
    p99 = _require_finite_float(timing.get("p99_latency_us"), "timing.p99_latency_us")
    max_latency = _require_finite_float(timing.get("max_latency_us"), "timing.max_latency_us")
    min_latency = _require_finite_float(timing.get("min_latency_us"), "timing.min_latency_us")
    mean_latency = _require_finite_float(timing.get("mean_latency_us"), "timing.mean_latency_us")
    _require_finite_float(timing.get("jitter_std_us"), "timing.jitter_std_us")
    if not (min_latency <= p50 <= p95 <= p99 <= max_latency):
        raise ValueError("latency percentiles must satisfy min <= p50 <= p95 <= p99 <= max")
    if not (min_latency <= mean_latency <= max_latency):
        raise ValueError("mean latency must lie within min/max latency bounds")

    overrun_count = _require_non_negative_int(timing.get("overrun_count"), "timing.overrun_count")
    if overrun_count > iterations:
        raise ValueError("timing.overrun_count cannot exceed timing.iterations")
    overrun_fraction = _require_finite_float(timing.get("overrun_fraction"), "timing.overrun_fraction")
    if abs(overrun_fraction - overrun_count / iterations) > 1e-12:
        raise ValueError("timing.overrun_fraction must equal overrun_count / iterations")
    expected_sub_ms = p95 < 1000.0
    if not isinstance(timing.get("sub_ms_achieved"), bool):
        raise ValueError("timing.sub_ms_achieved must be boolean")
    if timing["sub_ms_achieved"] is not expected_sub_ms:
        raise ValueError("timing.sub_ms_achieved must match p95_latency_us < 1000")

    target_hardware_id = _require_non_empty_text(
        target_hardware.get("target_hardware_id"),
        "target_hardware.target_hardware_id",
    )
    target_hardware_class = _require_non_empty_text(
        target_hardware.get("target_hardware_class"),
        "target_hardware.target_hardware_class",
    )
    rt_kernel = _require_non_empty_text(target_hardware.get("rt_kernel"), "target_hardware.rt_kernel")
    _require_non_empty_text(target_hardware.get("clock_source"), "target_hardware.clock_source")
    target_rate_hz = _require_finite_float(
        target_hardware.get("target_rate_hz"),
        "target_hardware.target_rate_hz",
        positive=True,
    )
    expected_rate_hz = 1e6 / target_dt_us
    if abs(target_rate_hz - expected_rate_hz) > max(1e-9, expected_rate_hz * 1e-12):
        raise ValueError("target_hardware.target_rate_hz must match timing.target_dt_us")

    interlocks = _require_non_negative_int(
        safety_events.get("interlock_events"),
        "safety_events.interlock_events",
    )
    backpressure = _require_non_negative_int(
        safety_events.get("backpressure_events"),
        "safety_events.backpressure_events",
    )

    if not isinstance(admission.get("deployment_claim_allowed"), bool):
        raise ValueError("admission.deployment_claim_allowed must be boolean")
    deployment_claim_allowed = bool(admission["deployment_claim_allowed"])
    expected_status = _TARGET_HARDWARE_CLAIM_STATUS if deployment_claim_allowed else _LOCAL_CLAIM_STATUS
    if admission.get("claim_status") != expected_status:
        raise ValueError("admission.claim_status is inconsistent with deployment_claim_allowed")
    if require_target_hardware and not deployment_claim_allowed:
        raise ValueError("qualified target hardware evidence is required")
    if deployment_claim_allowed:
        _require_qualified_hardware(target_hardware_id, "target_hardware.target_hardware_id")
        _require_qualified_hardware(target_hardware_class, "target_hardware.target_hardware_class")
        _require_qualified_hardware(rt_kernel, "target_hardware.rt_kernel")
        if overrun_count != 0:
            raise ValueError("target hardware deployment evidence must have zero overruns")
        if max_latency > target_dt_us:
            raise ValueError("target hardware deployment evidence must meet max latency target")
        if interlocks != 0:
            raise ValueError("target hardware deployment evidence must have zero interlock events")
        if backpressure != 0:
            raise ValueError("target hardware deployment evidence must have zero backpressure events")

    supplied_replay_digest = payload.get("replay_digest")
    if not _is_hex_sha256(supplied_replay_digest):
        raise ValueError("replay_digest must be a lowercase SHA-256 hex digest")
    expected_replay_digest = _sha256_json(
        {
            "controller_id": payload["controller_id"],
            "target_hardware": target_hardware,
            "timing": timing,
            "safety_events": safety_events,
            "admission": admission,
        }
    )
    if not hmac.compare_digest(str(supplied_replay_digest), expected_replay_digest):
        raise ValueError("replay_digest does not match HIL replay evidence metrics")

    if deployment_claim_allowed:
        raise ValueError("HIL deployment claim requires independent target hardware provenance")

    return cast(dict[str, Any], json.loads(json.dumps(payload, sort_keys=True)))


def save_hil_replay_evidence(
    evidence: Mapping[str, Any],
    path: str | Path,
    *,
    require_target_hardware: bool = False,
) -> Path:
    """Persist an admitted HIL replay evidence payload as canonical JSON."""
    admitted = assert_hil_replay_evidence_admissible(
        evidence,
        require_target_hardware=require_target_hardware,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(admitted, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def load_hil_replay_evidence(
    path: str | Path,
    *,
    require_target_hardware: bool = False,
) -> dict[str, Any]:
    """Load and admit a HIL replay evidence payload from JSON."""
    raw = Path(path).read_text(encoding="utf-8")
    payload = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
    return assert_hil_replay_evidence_admissible(
        payload,
        require_target_hardware=require_target_hardware,
    )

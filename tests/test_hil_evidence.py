# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — HIL Harness Tests

"""HIL replay evidence contract and admission tests."""

import copy
import json
from pathlib import Path
from typing import Any, cast

import pytest

import scpn_control.control.hil_harness as hil
from scpn_control.control.hil_harness import (
    ControlLoopMetrics,
    HILBenchmarkResult,
    assert_hil_replay_evidence_admissible,
    hil_replay_evidence,
    load_hil_replay_evidence,
    save_hil_replay_evidence,
)


class TestHILReplayEvidence:
    """Exercise h i l replay evidence behavior."""

    @staticmethod
    def _metrics(
        *,
        overrun_count: int = 0,
        target_dt_us: float = 1000.0,
        max_latency_us: float = 40.0,
    ) -> ControlLoopMetrics:
        return ControlLoopMetrics(
            iterations=10,
            target_dt_us=target_dt_us,
            measured_dt_us=[10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 30.0, 35.0, 40.0],
            p50_latency_us=21.0,
            p95_latency_us=37.75,
            p99_latency_us=39.55,
            max_latency_us=max_latency_us,
            min_latency_us=10.0,
            mean_latency_us=22.7,
            jitter_std_us=9.117,
            overrun_count=overrun_count,
            overrun_fraction=overrun_count / 10,
            sub_ms_achieved=True,
        )

    def test_local_replay_evidence_round_trips_and_rejects_tampering(self, tmp_path: Path) -> None:
        """Check local replay evidence round trips and rejects tampering."""
        evidence = hil_replay_evidence(
            self._metrics(),
            controller_id="tests.test_hil_harness.pid-vde",
            generated_at="2026-05-24T00:00:00Z",
        )
        assert evidence["admission"]["deployment_claim_allowed"] is False
        assert evidence["admission"]["claim_status"] == "bounded_local_hil_replay_only"
        assert_hil_replay_evidence_admissible(evidence)

        path = save_hil_replay_evidence(evidence, tmp_path / "hil_replay_evidence.json")
        assert load_hil_replay_evidence(path) == evidence

        tampered = json.loads(path.read_text(encoding="utf-8"))
        tampered["timing"]["p95_latency_us"] = 1.0
        path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
        with pytest.raises(ValueError, match="payload_sha256"):
            load_hil_replay_evidence(path)

    def test_deployment_claim_requires_qualified_target_hardware(self) -> None:
        """Check deployment claim requires qualified target hardware."""
        with pytest.raises(ValueError, match="target_hardware"):
            hil_replay_evidence(
                self._metrics(),
                controller_id="pid-vde",
                deployment_claim_allowed=True,
                generated_at="2026-05-24T00:00:00Z",
            )

    def test_deployment_claim_rejects_runtime_safety_events(self) -> None:
        """Check deployment claim rejects runtime safety events."""
        with pytest.raises(ValueError, match="backpressure"):
            hil_replay_evidence(
                self._metrics(),
                controller_id="pid-vde",
                target_hardware_id="jetson-orin-lab-01",
                target_hardware_class="jetson-orin-preempt-rt",
                rt_kernel="linux-rt-6.8.0-lab",
                backpressure_events=1,
                deployment_claim_allowed=True,
                generated_at="2026-05-24T00:00:00Z",
            )

    def test_deployment_claim_requires_independent_target_origin(self) -> None:
        """Plausible target identifiers do not qualify caller-owned replay metrics."""
        with pytest.raises(ValueError, match="independent"):
            hil_replay_evidence(
                self._metrics(),
                controller_id="pid-vde",
                target_hardware_id="jetson-orin-lab-01",
                target_hardware_class="jetson-orin-preempt-rt",
                rt_kernel="linux-rt-6.8.0-lab",
                deployment_claim_allowed=True,
                generated_at="2026-05-24T00:00:00Z",
            )


# ── Validator-helper, timing-payload and replay-admission branch contracts ────


def _full_metrics(**overrides: Any) -> ControlLoopMetrics:
    base: dict[str, Any] = dict(
        iterations=10,
        target_dt_us=1000.0,
        measured_dt_us=[10.0] * 10,
        p50_latency_us=21.0,
        p95_latency_us=37.75,
        p99_latency_us=39.55,
        max_latency_us=40.0,
        min_latency_us=10.0,
        mean_latency_us=22.7,
        jitter_std_us=9.117,
        overrun_count=0,
        overrun_fraction=0.0,
        sub_ms_achieved=True,
    )
    base.update(overrides)
    return ControlLoopMetrics(**base)


def _local_evidence() -> dict[str, Any]:
    return hil_replay_evidence(
        _full_metrics(),
        controller_id="tests.pid-vde",
        generated_at="2026-05-24T00:00:00Z",
    )


def _qualified_evidence() -> dict[str, Any]:
    payload = hil_replay_evidence(
        _full_metrics(),
        controller_id="tests.pid-vde",
        target_hardware_id="jetson-orin-lab-01",
        target_hardware_class="jetson-orin-preempt-rt",
        rt_kernel="linux-rt-6.8.0-lab",
        deployment_claim_allowed=False,
        generated_at="2026-05-24T00:00:00Z",
    )
    payload["admission"]["deployment_claim_allowed"] = True
    payload["admission"]["claim_status"] = "qualified_target_hardware_deployment_evidence"
    return _reseal(payload)


def _reseal(payload: dict[str, Any]) -> dict[str, Any]:
    payload["replay_digest"] = hil._sha256_json(
        {
            "controller_id": payload["controller_id"],
            "target_hardware": payload["target_hardware"],
            "timing": payload["timing"],
            "safety_events": payload["safety_events"],
            "admission": payload["admission"],
        }
    )
    payload["payload_sha256"] = hil._sha256_json({k: v for k, v in payload.items() if k != "payload_sha256"})
    return payload


def test_utc_now_iso_is_z_suffixed() -> None:
    """Check utc now iso is z suffixed."""
    assert hil._utc_now_iso().endswith("Z")


def test_reject_duplicate_json_keys() -> None:
    """Check reject duplicate json keys."""
    with pytest.raises(ValueError, match="duplicate JSON key"):
        hil._reject_duplicate_json_keys([("a", 1), ("a", 2)])


def test_require_mapping_rejects_non_mapping() -> None:
    """Check require mapping rejects non mapping."""
    with pytest.raises(ValueError, match="must be an object"):
        hil._require_mapping([], "field")


@pytest.mark.parametrize(
    ("fn_name", "args", "match"),
    [
        ("_require_non_empty_text", ("  ", "field"), "non-empty string"),
        ("_require_positive_int", (0, "field"), "positive integer"),
        ("_require_non_negative_int", (-1, "field"), "non-negative integer"),
    ],
)
def test_scalar_validators_reject(fn_name: Any, args: Any, match: Any) -> None:
    """Check scalar validators reject."""
    with pytest.raises(ValueError, match=match):
        getattr(hil, fn_name)(*args)


@pytest.mark.parametrize(
    ("value", "kwargs", "match"),
    [
        (True, {}, "finite float"),
        ("not-a-number", {}, "finite float"),
        (float("inf"), {}, "must be finite"),
        (0.0, {"positive": True}, "must be positive"),
        (-1.0, {}, "must be non-negative"),
    ],
)
def test_require_finite_float_rejects(value: Any, kwargs: Any, match: Any) -> None:
    """Check require finite float rejects."""
    with pytest.raises(ValueError, match=match):
        hil._require_finite_float(value, "field", **kwargs)


@pytest.mark.parametrize("bad", ["unknown-rig", "placeholder-node"])
def test_require_qualified_hardware_rejects_placeholders(bad: Any) -> None:
    """Check require qualified hardware rejects placeholders."""
    with pytest.raises(ValueError, match="must not be a placeholder"):
        hil._require_qualified_hardware(bad, "field")


def test_control_metrics_from_benchmark_result() -> None:
    """Check control metrics from benchmark result."""
    metrics = _full_metrics()
    result = HILBenchmarkResult(
        control_metrics=metrics,
        sensor_latency_us=1.0,
        controller_latency_us=2.0,
        actuator_latency_us=3.0,
        total_loop_latency_us=6.0,
        passes_sub_ms=True,
        passes_1khz=True,
        fpga_register_map=None,
    )
    assert hil._control_metrics_from_evidence_source(result) is metrics


def test_control_metrics_from_evidence_source_rejects_bad_type() -> None:
    """Check control metrics from evidence source rejects bad type."""
    with pytest.raises(TypeError, match="ControlLoopMetrics or HILBenchmarkResult"):
        hil._control_metrics_from_evidence_source(cast(ControlLoopMetrics, object()))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"overrun_count": 20, "overrun_fraction": 2.0}, "overrun_count cannot exceed"),
        ({"overrun_count": 2, "overrun_fraction": 0.0}, "overrun_fraction must equal"),
        ({"p50_latency_us": 38.0}, "latency percentiles must satisfy"),
        ({"mean_latency_us": 9999.0}, "mean latency must lie within"),
        ({"sub_ms_achieved": False}, "sub_ms_achieved must match"),
    ],
)
def test_hil_timing_payload_rejects(overrides: Any, match: Any) -> None:
    """Check hil timing payload rejects."""
    with pytest.raises(ValueError, match=match):
        hil._hil_timing_payload(_full_metrics(**overrides))


def test_admissible_rejects_non_hex_payload_digest() -> None:
    """Check admissible rejects non hex payload digest."""
    payload = _local_evidence()
    payload["payload_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="payload_sha256 must be a lowercase SHA-256"):
        assert_hil_replay_evidence_admissible(payload)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda p: p.__setitem__("schema_version", "bad"), "schema_version"),
        (lambda p: p.__setitem__("evidence_boundary", "bad"), "evidence boundary"),
        (lambda p: p["timing"].__setitem__("p50_latency_us", 38.0), "latency percentiles"),
        (lambda p: p["timing"].__setitem__("mean_latency_us", 9999.0), "mean latency"),
        (lambda p: p["timing"].__setitem__("overrun_count", 20), "overrun_count cannot exceed"),
        (
            lambda p: (p["timing"].__setitem__("overrun_count", 2), p["timing"].__setitem__("overrun_fraction", 0.0)),
            "overrun_fraction must equal",
        ),
        (lambda p: p["timing"].__setitem__("sub_ms_achieved", "yes"), "sub_ms_achieved must be boolean"),
        (lambda p: p["timing"].__setitem__("sub_ms_achieved", False), "sub_ms_achieved must match"),
        (lambda p: p["target_hardware"].__setitem__("target_rate_hz", 999.0), "target_rate_hz must match"),
        (lambda p: p["admission"].__setitem__("deployment_claim_allowed", "yes"), "deployment_claim_allowed must be"),
        (lambda p: p["admission"].__setitem__("claim_status", "wrong"), "claim_status is inconsistent"),
    ],
)
def test_admissible_rejects_resealed_mutation(mutate: Any, match: Any) -> None:
    """Check admissible rejects resealed mutation."""
    payload = copy.deepcopy(_local_evidence())
    mutate(payload)
    _reseal(payload)
    with pytest.raises(ValueError, match=match):
        assert_hil_replay_evidence_admissible(payload)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda p: (p["timing"].__setitem__("overrun_count", 3), p["timing"].__setitem__("overrun_fraction", 0.3)),
            "must have zero overruns",
        ),
        (lambda p: p["timing"].__setitem__("max_latency_us", 5000.0), "must meet max latency target"),
        (lambda p: p["safety_events"].__setitem__("interlock_events", 1), "zero interlock events"),
    ],
)
def test_qualified_admissible_rejects(mutate: Any, match: Any) -> None:
    """Check qualified admissible rejects."""
    payload = copy.deepcopy(_qualified_evidence())
    mutate(payload)
    _reseal(payload)
    with pytest.raises(ValueError, match=match):
        assert_hil_replay_evidence_admissible(payload, require_target_hardware=True)


def test_admissible_requires_qualified_hardware_when_demanded() -> None:
    """Check admissible requires qualified hardware when demanded."""
    payload = _local_evidence()
    with pytest.raises(ValueError, match="qualified target hardware evidence is required"):
        assert_hil_replay_evidence_admissible(payload, require_target_hardware=True)


def test_admissible_rejects_non_hex_replay_digest() -> None:
    """Check admissible rejects non hex replay digest."""
    payload = _local_evidence()
    payload["replay_digest"] = "not-a-digest"
    payload["payload_sha256"] = hil._sha256_json({k: v for k, v in payload.items() if k != "payload_sha256"})
    with pytest.raises(ValueError, match="replay_digest must be a lowercase SHA-256"):
        assert_hil_replay_evidence_admissible(payload)


def test_admissible_rejects_replay_digest_mismatch() -> None:
    """Check admissible rejects replay digest mismatch."""
    payload = _local_evidence()
    payload["replay_digest"] = "a" * 64
    payload["payload_sha256"] = hil._sha256_json({k: v for k, v in payload.items() if k != "payload_sha256"})
    with pytest.raises(ValueError, match="replay_digest does not match"):
        assert_hil_replay_evidence_admissible(payload)

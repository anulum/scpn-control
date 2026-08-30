# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Test Digital Twin Ingest Validation.

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Digital twin ingest validation edge path tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Exercise validation boundaries in the digital-twin ingest pipeline.

The cases cover invalid emulated streams, bounded buffers, short horizons, and
sessions without control plans.
"""

from __future__ import annotations

import pytest

from scpn_control.control.digital_twin_ingest import (
    RealtimeTwinHook,
    TelemetryPacket,
    _build_snn_planner,
    generate_emulated_stream,
    run_realtime_twin_session,
)


def test_planner_uses_descriptive_artifact_identity() -> None:
    """The runtime artifact exposes its responsibility, not a workstream code."""
    assert _build_snn_planner().artifact.meta.name == "digital-twin-ingest-controller"


class TestGenerateEmulatedStreamValidation:
    """Exercise emulated-stream sample-count validation."""

    def test_samples_below_minimum_raises(self):
        """Samples < 32 raises ValueError (line 100)."""
        with pytest.raises(ValueError, match="samples must be >= 32"):
            generate_emulated_stream("SPARC", samples=16)

    def test_dt_ms_below_minimum_raises(self):
        """dt_ms < 1 raises ValueError (line 103)."""
        with pytest.raises(ValueError, match="dt_ms must be >= 1"):
            generate_emulated_stream("SPARC", dt_ms=0)


class TestRealtimeTwinHookEdgePaths:
    """Exercise real-time twin buffer and horizon boundaries."""

    def test_buffer_overflow_trims(self):
        """Ingesting > max_buffer packets trims to tail (line 148)."""
        hook = RealtimeTwinHook("SPARC", max_buffer=64)
        for i in range(70):
            pkt = TelemetryPacket(
                t_ms=i * 5,
                machine="SPARC",
                ip_ma=8.7,
                beta_n=1.65,
                q95=3.9,
                density_1e19=8.2,
            )
            hook.ingest(pkt)
        assert len(hook.buffer) == 64

    def test_empty_buffer_scenario_plan_raises(self):
        """scenario_plan on empty buffer raises RuntimeError (line 162)."""
        hook = RealtimeTwinHook("SPARC")
        with pytest.raises(RuntimeError, match="No telemetry"):
            hook.scenario_plan()

    def test_short_horizon_raises(self):
        """Horizon < 4 raises ValueError (line 165)."""
        hook = RealtimeTwinHook("SPARC")
        pkt = TelemetryPacket(
            t_ms=0,
            machine="SPARC",
            ip_ma=8.7,
            beta_n=1.65,
            q95=3.9,
            density_1e19=8.2,
        )
        hook.ingest(pkt)
        with pytest.raises(ValueError, match="horizon must be >= 4"):
            hook.scenario_plan(horizon=2)


class TestRunSessionEmptyPlans:
    """Exercise a real-time twin session without control plans."""

    def test_plan_every_exceeds_samples(self):
        """plan_every > samples yields no plans → fallback dict (line 322)."""
        result = run_realtime_twin_session(
            "SPARC",
            samples=32,
            plan_every=999,
            seed=42,
        )
        assert result["samples"] == 32
        assert result.get("plan_count", 0) == 0


def test_scenario_plan_records_no_safe_step_when_risk_stays_high(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rollout whose predicted disruption risk stays >= 0.85 records no safe steps (branch 279->251)."""
    monkeypatch.setattr(
        "scpn_control.control.digital_twin_ingest._predict_disruption_risk",
        lambda *_a, **_k: 0.9,
    )
    hook = RealtimeTwinHook("SPARC")
    hook.ingest(TelemetryPacket(t_ms=0, machine="SPARC", ip_ma=8.7, beta_n=1.65, q95=3.9, density_1e19=8.2))
    plan = hook.scenario_plan(horizon=4)
    assert plan["safe_horizon_rate"] == 0.0


@pytest.mark.parametrize("field", ["ip_ma", "beta_n", "q95", "density_1e19"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_telemetry_packet_rejects_non_finite_fields(field, bad):
    # A non-finite telemetry field would poison the risk signal (max(nan, 0)
    # is nan; nan comparisons fail open in the mitigation gate), so it must fail closed at
    # construction rather than reach risk scoring.
    """Reject non-finite telemetry packet fields."""
    kwargs = {"t_ms": 0, "machine": "SPARC", "ip_ma": 8.7, "beta_n": 1.65, "q95": 3.9, "density_1e19": 8.2}
    kwargs[field] = bad
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        TelemetryPacket(**kwargs)


def test_telemetry_packet_accepts_finite_fields():
    """Accept a telemetry packet whose fields are all finite."""
    packet = TelemetryPacket(t_ms=0, machine="SPARC", ip_ma=8.7, beta_n=1.65, q95=3.9, density_1e19=8.2)
    assert packet.beta_n == 1.65

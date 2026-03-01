# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Digital twin ingest validation edge path tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for generate_emulated_stream validation (100, 103),
RealtimeTwinHook buffer overflow (148), empty buffer (162),
short horizon (165), run_realtime_twin_session empty plans (322)."""

from __future__ import annotations

import pytest

from scpn_control.control.digital_twin_ingest import (
    RealtimeTwinHook,
    TelemetryPacket,
    generate_emulated_stream,
    run_realtime_twin_session,
)


class TestGenerateEmulatedStreamValidation:
    def test_samples_below_minimum_raises(self):
        """samples < 32 raises ValueError (line 100)."""
        with pytest.raises(ValueError, match="samples must be >= 32"):
            generate_emulated_stream("SPARC", samples=16)

    def test_dt_ms_below_minimum_raises(self):
        """dt_ms < 1 raises ValueError (line 103)."""
        with pytest.raises(ValueError, match="dt_ms must be >= 1"):
            generate_emulated_stream("SPARC", dt_ms=0)


class TestRealtimeTwinHookEdgePaths:
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
        """horizon < 4 raises ValueError (line 165)."""
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

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Digital twin telemetry admission and replay tests.

"""Exercise telemetry identity, ordering and replay through the public hook."""

from typing import cast

import pytest

from scpn_control.control.digital_twin_ingest import RealtimeTwinHook, run_realtime_twin_session
from scpn_control.control.digital_twin_telemetry import TelemetryPacket, generate_emulated_stream


def _packet(t_ms: int, machine: str = "SPARC") -> TelemetryPacket:
    return TelemetryPacket(t_ms, machine, 8.7, 1.65, 3.9, 8.2)


@pytest.mark.parametrize("timestamp", [-1, True])
def test_packet_rejects_invalid_timestamp(timestamp: int) -> None:
    """A physical sample needs a nonnegative integral timestamp."""
    with pytest.raises(ValueError, match="t_ms"):
        _packet(timestamp)


def test_packet_rejects_unknown_machine() -> None:
    """The packet identity must be a known canonical machine."""
    with pytest.raises(ValueError, match="machine"):
        _packet(0, "ITER")


def test_packet_rejects_noncanonical_or_nontext_machine() -> None:
    """Packets retain a canonical source identity throughout the ingest path."""
    with pytest.raises(ValueError, match="canonical"):
        _packet(0, "sparc")
    with pytest.raises(ValueError, match="machine"):
        _packet(0, cast(str, 7))


def test_ingest_rejects_other_machine_and_out_of_order_sample() -> None:
    """A rejected sample must leave the accepted replay buffer unchanged."""
    hook = RealtimeTwinHook("SPARC")
    first = _packet(5)
    hook.ingest(first)
    with pytest.raises(ValueError, match="machine"):
        hook.ingest(_packet(10, "NSTX-U"))
    with pytest.raises(ValueError, match="later"):
        hook.ingest(_packet(5))
    with pytest.raises(ValueError, match="later"):
        hook.ingest(_packet(4))
    assert hook.buffer == [first]


def test_ingest_rejects_nonpacket_without_changing_buffer() -> None:
    """An object without telemetry provenance cannot enter the replay buffer."""
    hook = RealtimeTwinHook("SPARC")
    with pytest.raises(TypeError, match="TelemetryPacket"):
        hook.ingest(cast(TelemetryPacket, object()))
    assert hook.buffer == []


def test_same_buffer_yields_same_plan_without_consuming_hypothetical_future() -> None:
    """Planning twice over the same accepted observations must replay equally."""
    hook = RealtimeTwinHook("SPARC", seed=19)
    hook.ingest(_packet(0))
    first = hook.scenario_plan(horizon=4)
    second = hook.scenario_plan(horizon=4)
    first.pop("latency_wall_ms")
    second.pop("latency_wall_ms")
    assert first == second


def test_public_integer_domains_do_not_truncate_fractional_or_boolean_values() -> None:
    """Session and stream counts must retain their exact declared domains."""
    with pytest.raises(ValueError, match="samples"):
        generate_emulated_stream("SPARC", samples=cast(int, 32.5))
    with pytest.raises(ValueError, match="horizon"):
        run_realtime_twin_session("SPARC", horizon=True)
    with pytest.raises(ValueError, match="max_buffer"):
        RealtimeTwinHook("SPARC", max_buffer=cast(int, 64.5))


def test_session_without_plans_does_not_invent_risk_or_latency() -> None:
    """No plan means no observed plan-risk or latency distribution."""
    result = run_realtime_twin_session("SPARC", samples=32, plan_every=100)
    assert result["plan_count"] == 0
    assert result["mean_risk"] is None
    assert result["p95_latency_ms"] is None
    assert result["passes_thresholds"] is False

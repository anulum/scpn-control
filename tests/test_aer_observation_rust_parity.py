# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — AER PyO3 parity tests.
"""Optional parity tests for Rust AER spike-buffer bindings."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.scpn.observation import SpikeEvent, decode_isi, decode_rate, decode_temporal

rust = pytest.importorskip("scpn_control_rs")

pytestmark = pytest.mark.skipif(
    not hasattr(rust, "aer_decode_rate"),
    reason="installed scpn_control_rs extension does not expose AER decoder bindings",
)


def _events() -> list[SpikeEvent]:
    return [SpikeEvent(0, 10), SpikeEvent(0, 30), SpikeEvent(1, 70), SpikeEvent(3, 80)]


def _tuples(events: list[SpikeEvent]) -> list[tuple[int, int]]:
    return [(event.neuron_id, event.timestamp_ns) for event in events]


def test_rust_decoders_match_python_vectors() -> None:
    events = _events()
    event_tuples = _tuples(events)
    assert np.array_equal(rust.aer_decode_rate(event_tuples, 100, 4), decode_rate(events, 100, 4))
    assert np.allclose(
        rust.aer_decode_temporal(event_tuples, 100, 4, 100),
        decode_temporal(events, 100, 4, now_ns=100),
        atol=1.0e-12,
    )
    assert np.allclose(rust.aer_decode_isi(event_tuples, 100, 4), decode_isi(events, 100, 4), atol=1.0e-12)


def test_rust_spike_buffer_matches_python_overflow_and_drain_semantics() -> None:
    buffer = rust.PySpikeBuffer(2)
    buffer.push(1, 10)
    buffer.push(2, 20)
    buffer.push(3, 30)
    assert buffer.snapshot() == [(2, 20), (3, 30)]
    assert buffer.overflowed is True
    assert buffer.drain_window(15, 25) == [(2, 20)]
    assert buffer.snapshot() == [(3, 30)]


def test_rust_spike_buffer_exposes_admission_metadata() -> None:
    buffer = rust.PySpikeBuffer(4)
    buffer.push(1, 100)
    buffer.push(1, 90)
    buffer.push(1, 95)
    buffer.push(1, 110)

    assert buffer.out_of_order_event_count == 2
    assert buffer.monotonic_input is False
    assert buffer.admission_report() == {
        "capacity": 4,
        "retained_events": 4,
        "overflowed": False,
        "out_of_order_event_count": 2,
        "monotonic_input": False,
    }

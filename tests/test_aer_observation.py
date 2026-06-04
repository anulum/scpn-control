# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — AER control-observation tests.
"""Tests for AER spike observation buffers and adapters."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.scpn.observation import AERControlObservation, SpikeBuffer, SpikeEvent


def test_spike_event_rejects_negative_timestamp() -> None:
    with pytest.raises(ValueError, match="timestamp_ns"):
        SpikeEvent(neuron_id=1, timestamp_ns=-1)


def test_spike_buffer_drops_oldest_and_latches_overflow() -> None:
    buffer = SpikeBuffer(capacity=2)
    buffer.push(SpikeEvent(1, 10))
    buffer.push(SpikeEvent(2, 20))
    buffer.push(SpikeEvent(3, 30))
    assert buffer.snapshot() == [SpikeEvent(2, 20), SpikeEvent(3, 30)]
    assert buffer.overflowed is True
    assert len(buffer) == 2


def test_spike_buffer_latches_out_of_order_admission() -> None:
    buffer = SpikeBuffer(capacity=4)
    buffer.push(SpikeEvent(1, 100))
    buffer.push(SpikeEvent(1, 90))
    buffer.push(SpikeEvent(1, 95))
    buffer.push(SpikeEvent(1, 110))

    assert buffer.out_of_order_event_count == 2
    assert buffer.monotonic_input is False
    assert buffer.admission_report() == {
        "capacity": 4,
        "retained_events": 4,
        "overflowed": False,
        "out_of_order_event_count": 2,
        "monotonic_input": False,
    }


def test_spike_buffer_clear_resets_admission_latches() -> None:
    buffer = SpikeBuffer(capacity=4)
    buffer.push(SpikeEvent(1, 100))
    buffer.push(SpikeEvent(1, 90))

    buffer.clear()

    assert buffer.out_of_order_event_count == 0
    assert buffer.monotonic_input is True
    assert buffer.overflowed is False
    assert buffer.snapshot() == []


def test_drain_window_discards_old_events_and_retains_future_events() -> None:
    buffer = SpikeBuffer(capacity=4)
    buffer.extend([SpikeEvent(0, 10), SpikeEvent(1, 20), SpikeEvent(2, 40)])
    drained = buffer.drain_window(window_ns=15, now_ns=25)
    assert drained == [SpikeEvent(0, 10), SpikeEvent(1, 20)]
    assert buffer.snapshot() == [SpikeEvent(2, 40)]


def test_aer_control_observation_to_features_drains_active_window() -> None:
    buffer = SpikeBuffer(capacity=8)
    buffer.extend([SpikeEvent(0, 10), SpikeEvent(0, 30), SpikeEvent(1, 70), SpikeEvent(3, 80)])
    obs = AERControlObservation(
        timestamp_ns=100,
        spike_stream=buffer,
        decode_window_ns=100,
        decode_strategy="rate",
        n_features=4,
    )
    features = obs.to_features()
    assert features.shape == (4,)
    assert features.dtype == np.float64
    assert np.all((features >= 0.0) & (features <= 1.0))
    assert features.tolist() == pytest.approx([0.5, 0.25, 0.0, 0.25])
    assert len(buffer) == 0


def test_aer_control_observation_strict_monotonic_mode_rejects_violation() -> None:
    buffer = SpikeBuffer(capacity=8)
    buffer.push(SpikeEvent(1, 100))
    buffer.push(SpikeEvent(1, 90))
    obs = AERControlObservation(
        timestamp_ns=120,
        spike_stream=buffer,
        decode_window_ns=50,
        n_features=4,
        require_monotonic=True,
    )

    with pytest.raises(ValueError, match="monotonic timestamp"):
        obs.to_features()

    assert buffer.snapshot() == [SpikeEvent(1, 100), SpikeEvent(1, 90)]


def test_aer_control_observation_reports_admission_state() -> None:
    buffer = SpikeBuffer(capacity=8)
    buffer.push(SpikeEvent(1, 100))
    obs = AERControlObservation(timestamp_ns=120, spike_stream=buffer, decode_window_ns=50, n_features=4)

    assert obs.admission_report()["monotonic_input"] is True


def test_aer_control_observation_feature_mapping_uses_prefix() -> None:
    buffer = SpikeBuffer(capacity=4)
    buffer.extend([SpikeEvent(2, 20), SpikeEvent(2, 30)])
    obs = AERControlObservation(timestamp_ns=40, spike_stream=buffer, decode_window_ns=40, n_features=4)
    mapping = obs.to_feature_mapping(prefix="neu_")
    assert set(mapping) == {"neu_0", "neu_1", "neu_2", "neu_3"}
    assert mapping["neu_2"] == pytest.approx(1.0)


def test_aer_control_observation_is_deterministic_for_fixed_input() -> None:
    events = [SpikeEvent(0, 10), SpikeEvent(0, 30), SpikeEvent(1, 70), SpikeEvent(3, 80)]
    first = SpikeBuffer(capacity=8)
    second = SpikeBuffer(capacity=8)
    first.extend(events)
    second.extend(events)
    obs_a = AERControlObservation(timestamp_ns=100, spike_stream=first, decode_window_ns=100, n_features=4)
    obs_b = AERControlObservation(timestamp_ns=100, spike_stream=second, decode_window_ns=100, n_features=4)
    assert np.array_equal(obs_a.to_features(), obs_b.to_features())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — AER Observation Internal Branch Tests
"""Branch coverage for the AER observation buffer, decode dispatch, feature
normalisation modes, and integer validators."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.scpn.observation import (
    AERControlObservation,
    SpikeBuffer,
    SpikeEvent,
    _non_negative_int,
    _normalise_features,
    _positive_int,
)


def _buffer(timestamps: list[tuple[int, int]]) -> SpikeBuffer:
    buffer = SpikeBuffer(capacity=16)
    buffer.extend([SpikeEvent(neuron_id=nid, timestamp_ns=ts) for nid, ts in timestamps])
    return buffer


class TestSpikeBuffer:
    def test_capacity_property_reports_configured_capacity(self) -> None:
        assert SpikeBuffer(capacity=8).capacity == 8


class TestObservationValidation:
    def test_rejects_unknown_decode_strategy(self) -> None:
        with pytest.raises(ValueError, match="decode_strategy must be one of"):
            AERControlObservation(
                timestamp_ns=10,
                spike_stream=_buffer([(0, 1)]),
                decode_window_ns=100,
                decode_strategy="bogus",  # type: ignore[arg-type]
            )

    def test_rejects_unknown_feature_normalisation(self) -> None:
        with pytest.raises(ValueError, match="feature_normalisation must be one of"):
            AERControlObservation(
                timestamp_ns=10,
                spike_stream=_buffer([(0, 1)]),
                decode_window_ns=100,
                feature_normalisation="bogus",  # type: ignore[arg-type]
            )

    def test_rejects_non_boolean_require_monotonic(self) -> None:
        with pytest.raises(ValueError, match="require_monotonic must be a boolean"):
            AERControlObservation(
                timestamp_ns=10,
                spike_stream=_buffer([(0, 1)]),
                decode_window_ns=100,
                require_monotonic="yes",  # type: ignore[arg-type]
            )


class TestDecodeDispatch:
    def test_to_features_temporal_branch(self) -> None:
        observation = AERControlObservation(
            timestamp_ns=100,
            spike_stream=_buffer([(0, 90), (1, 50)]),
            decode_window_ns=100,
            decode_strategy="temporal",
            n_features=4,
        )
        features = observation.to_features()
        assert features.shape == (4,)
        # neuron 1 spiked earlier in the window than neuron 0, so it maps higher.
        assert features[1] > features[0]

    def test_to_features_isi_branch(self) -> None:
        observation = AERControlObservation(
            timestamp_ns=100,
            spike_stream=_buffer([(0, 10), (0, 20), (0, 30)]),
            decode_window_ns=100,
            decode_strategy="isi",
            n_features=4,
        )
        features = observation.to_features()
        assert features.shape == (4,)
        assert 0.0 <= features[0] <= 1.0

    def test_to_feature_mapping_rejects_empty_prefix(self) -> None:
        observation = AERControlObservation(
            timestamp_ns=100,
            spike_stream=_buffer([(0, 90)]),
            decode_window_ns=100,
        )
        with pytest.raises(ValueError, match="prefix must not be empty"):
            observation.to_feature_mapping(prefix="")


class TestNormaliseFeatures:
    def test_rejects_non_one_dimensional(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            _normalise_features(np.zeros((2, 2)), "unit")

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            _normalise_features(np.array([np.nan, 0.0]), "unit")

    def test_max_mode_scales_to_unit_peak(self) -> None:
        out = _normalise_features(np.array([0.2, 0.4, 0.8]), "max")
        assert float(np.max(out)) == pytest.approx(1.0)

    def test_max_mode_with_zero_peak_returns_clipped(self) -> None:
        out = _normalise_features(np.zeros(3), "max")
        assert np.array_equal(out, np.zeros(3))

    def test_zscore_mode_maps_through_sigmoid(self) -> None:
        out = _normalise_features(np.array([0.1, 0.5, 0.9]), "zscore")
        assert out.shape == (3,)
        assert np.all((out >= 0.0) & (out <= 1.0))

    def test_zscore_mode_with_zero_variance_returns_zeros(self) -> None:
        out = _normalise_features(np.full(4, 0.5), "zscore")
        assert np.array_equal(out, np.zeros(4))

    def test_zscore_mode_empty_returns_empty(self) -> None:
        out = _normalise_features(np.zeros(0), "zscore")
        assert out.size == 0

    def test_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="feature_normalisation must be one of"):
            _normalise_features(np.array([0.5]), "bogus")  # type: ignore[arg-type]


class TestIntegerValidators:
    def test_positive_int_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            _positive_int("capacity", 0)

    def test_non_negative_int_rejects_non_integer(self) -> None:
        with pytest.raises(ValueError, match="must be an integer"):
            _non_negative_int("capacity", 1.5)  # type: ignore[arg-type]

    def test_non_negative_int_rejects_boolean(self) -> None:
        with pytest.raises(ValueError, match="must be an integer"):
            _non_negative_int("capacity", True)

    def test_non_negative_int_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            _non_negative_int("capacity", -1)


def test_drain_window_drops_events_before_window() -> None:
    """Events older than the window lower bound are dropped (branch 109->106).

    An event whose timestamp precedes now-window is neither in the future
    (retained) nor inside the window (drained), so the loop skips it and the
    event is discarded entirely — absent from both the returned batch and the
    buffer on the next drain.
    """
    buffer = _buffer([(0, 5), (1, 100), (2, 200)])

    drained = buffer.drain_window(window_ns=50, now_ns=120)  # lower = 70
    assert [event.timestamp_ns for event in drained] == [100]

    remaining = buffer.drain_window(window_ns=1000, now_ns=1000)
    assert [event.timestamp_ns for event in remaining] == [200]

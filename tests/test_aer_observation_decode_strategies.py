# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Project: SCPN Control
# Description: AER decoder strategy tests.
"""Analytical tests for AER decode strategies."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.scpn.observation import SpikeEvent, decode_isi, decode_rate, decode_temporal


def _events() -> list[SpikeEvent]:
    return [
        SpikeEvent(0, 10),
        SpikeEvent(0, 30),
        SpikeEvent(1, 70),
        SpikeEvent(3, 80),
        SpikeEvent(99, 90),
    ]


def test_decode_rate_returns_event_fractions() -> None:
    features = decode_rate(_events(), window_ns=100, n_features=4)
    assert features.tolist() == pytest.approx([0.4, 0.2, 0.0, 0.2])
    assert features.dtype == np.float64


def test_decode_temporal_maps_earlier_first_spikes_higher() -> None:
    features = decode_temporal(_events(), window_ns=100, n_features=4, now_ns=100)
    assert features.tolist() == pytest.approx([0.9, 0.3, 0.0, 0.2])


def test_decode_isi_maps_shorter_mean_interval_higher() -> None:
    features = decode_isi(_events(), window_ns=100, n_features=4)
    assert features.tolist() == pytest.approx([0.8, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("decoder", [decode_rate, decode_isi])
def test_decoders_reject_zero_window(decoder: object) -> None:
    with pytest.raises(ValueError, match="window_ns"):
        decoder(_events(), window_ns=0, n_features=4)


def test_decode_temporal_rejects_zero_window() -> None:
    with pytest.raises(ValueError, match="window_ns"):
        decode_temporal(_events(), window_ns=0, n_features=4, now_ns=100)

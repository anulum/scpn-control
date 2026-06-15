# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — AER control-observation adapter.
"""Address-event spike observation adapters for SCPN control loops."""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

DecodeStrategy = Literal["rate", "temporal", "isi"]
FeatureNormalisation = Literal["unit", "max", "zscore"]


@dataclass(frozen=True)
class SpikeEvent:
    """Single Address-Event Representation spike event."""

    neuron_id: int
    timestamp_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "neuron_id", _non_negative_int("neuron_id", self.neuron_id))
        object.__setattr__(self, "timestamp_ns", _non_negative_int("timestamp_ns", self.timestamp_ns))


class SpikeBuffer:
    """Bounded FIFO ring buffer for AER spike events.

    Overflow drops the oldest event and latches ``overflowed`` until ``clear``
    is called. This keeps memory bounded under bursty neuromorphic input.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = _positive_int("capacity", capacity)
        self._events: list[SpikeEvent] = []
        self._overflowed = False
        self._last_timestamp_ns: int | None = None
        self._out_of_order_event_count = 0

    @property
    def capacity(self) -> int:
        """Maximum number of retained spike events."""
        return self._capacity

    @property
    def overflowed(self) -> bool:
        """Return whether the buffer has dropped at least one event."""
        return self._overflowed

    @property
    def out_of_order_event_count(self) -> int:
        """Return count of events admitted after a later timestamp was seen."""
        return self._out_of_order_event_count

    @property
    def monotonic_input(self) -> bool:
        """Return whether admitted timestamps were non-decreasing."""
        return self._out_of_order_event_count == 0

    def __len__(self) -> int:
        return len(self._events)

    def push(self, event: SpikeEvent) -> None:
        """Append ``event`` and drop the oldest event if the buffer is full."""
        if self._last_timestamp_ns is None:
            self._last_timestamp_ns = event.timestamp_ns
        elif event.timestamp_ns < self._last_timestamp_ns:
            self._out_of_order_event_count += 1
        else:
            self._last_timestamp_ns = event.timestamp_ns

        if len(self._events) >= self._capacity:
            self._events.pop(0)
            self._overflowed = True
        self._events.append(event)

    def extend(self, events: list[SpikeEvent]) -> None:
        """Append a sequence of spike events in order."""
        for event in events:
            self.push(event)

    def snapshot(self) -> list[SpikeEvent]:
        """Return retained events without mutating the buffer."""
        return list(self._events)

    def drain_window(self, window_ns: int, now_ns: int) -> list[SpikeEvent]:
        """Drain events with timestamps in ``[now_ns - window_ns, now_ns]``.

        Events older than the window are discarded. Events newer than ``now_ns``
        are retained for the next control tick.
        """
        window = _positive_int("window_ns", window_ns)
        now = _non_negative_int("now_ns", now_ns)
        lower = max(0, now - window)
        drained: list[SpikeEvent] = []
        retained: list[SpikeEvent] = []
        for event in self._events:
            if event.timestamp_ns > now:
                retained.append(event)
            elif event.timestamp_ns >= lower:
                drained.append(event)
        self._events = retained
        return drained

    def clear(self) -> None:
        """Clear buffered events and reset the overflow latch."""
        self._events.clear()
        self._overflowed = False
        self._last_timestamp_ns = None
        self._out_of_order_event_count = 0

    def admission_report(self) -> dict[str, int | bool]:
        """Return bounded-buffer admission metadata for safety gating."""
        return {
            "capacity": self._capacity,
            "retained_events": len(self._events),
            "overflowed": self._overflowed,
            "out_of_order_event_count": self._out_of_order_event_count,
            "monotonic_input": self.monotonic_input,
        }


@dataclass(frozen=True)
class AERControlObservation:
    """Control observation that decodes an AER spike stream to feature arrays."""

    timestamp_ns: int
    spike_stream: SpikeBuffer
    decode_window_ns: int
    decode_strategy: DecodeStrategy = "rate"
    n_features: int = 64
    feature_normalisation: FeatureNormalisation = "unit"
    require_monotonic: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_ns", _non_negative_int("timestamp_ns", self.timestamp_ns))
        object.__setattr__(self, "decode_window_ns", _positive_int("decode_window_ns", self.decode_window_ns))
        object.__setattr__(self, "n_features", _positive_int("n_features", self.n_features))
        if self.decode_strategy not in ("rate", "temporal", "isi"):
            raise ValueError("decode_strategy must be one of: rate, temporal, isi")
        if self.feature_normalisation not in ("unit", "max", "zscore"):
            raise ValueError("feature_normalisation must be one of: unit, max, zscore")
        if not isinstance(self.require_monotonic, bool):
            raise ValueError("require_monotonic must be a boolean")

    def to_features(self) -> NDArray[np.float64]:
        """Drain the active window and return ``float64`` features in ``[0, 1]``."""
        if self.require_monotonic and not self.spike_stream.monotonic_input:
            raise ValueError("AER spike stream violated monotonic timestamp admission")

        events = self.spike_stream.drain_window(self.decode_window_ns, self.timestamp_ns)
        if self.decode_strategy == "rate":
            features = decode_rate(events, self.decode_window_ns, self.n_features)
        elif self.decode_strategy == "temporal":
            features = decode_temporal(events, self.decode_window_ns, self.n_features, now_ns=self.timestamp_ns)
        else:
            features = decode_isi(events, self.decode_window_ns, self.n_features)
        return _normalise_features(features, self.feature_normalisation)

    def to_feature_mapping(self, prefix: str = "aer_") -> dict[str, float]:
        """Return decoded features as a dictionary for existing mapping consumers."""
        if not prefix:
            raise ValueError("prefix must not be empty")
        features = self.to_features()
        return {f"{prefix}{idx}": float(value) for idx, value in enumerate(features)}

    def admission_report(self) -> dict[str, int | bool]:
        """Return spike-buffer admission metadata for this observation source."""
        return self.spike_stream.admission_report()


def decode_rate(events: list[SpikeEvent], window_ns: int, n_features: int) -> NDArray[np.float64]:
    """Decode AER events as per-feature event fractions over the active window."""
    _positive_int("window_ns", window_ns)
    n = _positive_int("n_features", n_features)
    counts = np.zeros(n, dtype=np.float64)
    for event in events:
        if 0 <= event.neuron_id < n:
            counts[event.neuron_id] += 1.0
    denominator = max(1.0, float(len(events)))
    return _clip01(counts / denominator)


def decode_temporal(
    events: list[SpikeEvent],
    window_ns: int,
    n_features: int,
    *,
    now_ns: int | None = None,
) -> NDArray[np.float64]:
    """Decode by first-spike timing; earlier spikes map closer to one."""
    window = _positive_int("window_ns", window_ns)
    n = _positive_int("n_features", n_features)
    now = (
        max((event.timestamp_ns for event in events), default=0)
        if now_ns is None
        else _non_negative_int("now_ns", now_ns)
    )
    lower = max(0, now - window)
    first: list[int | None] = [None] * n
    for event in events:
        if 0 <= event.neuron_id < n and lower <= event.timestamp_ns <= now:
            current = first[event.neuron_id]
            if current is None or event.timestamp_ns < current:
                first[event.neuron_id] = event.timestamp_ns
    features = np.zeros(n, dtype=np.float64)
    for idx, timestamp in enumerate(first):
        if timestamp is not None:
            features[idx] = 1.0 - ((timestamp - lower) / window)
    return _clip01(features)


def decode_isi(events: list[SpikeEvent], window_ns: int, n_features: int) -> NDArray[np.float64]:
    """Decode by mean inter-spike interval; faster repeated spikes map higher."""
    window = _positive_int("window_ns", window_ns)
    n = _positive_int("n_features", n_features)
    by_neuron: list[list[int]] = [[] for _ in range(n)]
    for event in events:
        if 0 <= event.neuron_id < n:
            by_neuron[event.neuron_id].append(event.timestamp_ns)
    features = np.zeros(n, dtype=np.float64)
    for idx, timestamps in enumerate(by_neuron):
        if len(timestamps) < 2:
            continue
        ordered = sorted(timestamps)
        intervals = [later - earlier for earlier, later in zip(ordered, ordered[1:])]
        mean_isi = sum(intervals) / len(intervals)
        features[idx] = 1.0 - (mean_isi / window)
    return _clip01(features)


def _normalise_features(features: NDArray[np.float64], mode: FeatureNormalisation) -> NDArray[np.float64]:
    values = np.asarray(features, dtype=np.float64)
    if values.shape != (values.size,):
        raise ValueError("features must be a one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("features must be finite")
    clipped = _clip01(values)
    if mode == "unit":
        return clipped
    if mode == "max":
        max_value = float(np.max(clipped)) if clipped.size else 0.0
        if max_value <= 0.0:
            return clipped
        return _clip01(clipped / max_value)
    if mode == "zscore":
        if clipped.size == 0:
            return clipped
        std = float(np.std(clipped))
        if std <= 1.0e-12:
            return np.zeros_like(clipped)
        z = (clipped - float(np.mean(clipped))) / std
        return _clip01(1.0 / (1.0 + np.exp(-z)))
    raise ValueError("feature_normalisation must be one of: unit, max, zscore")


def _clip01(values: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.asarray(np.clip(values, 0.0, 1.0), dtype=np.float64)
    return clipped


def _positive_int(name: str, value: int) -> int:
    number = _non_negative_int(name, value)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


__all__ = [
    "AERControlObservation",
    "DecodeStrategy",
    "FeatureNormalisation",
    "SpikeBuffer",
    "SpikeEvent",
    "decode_isi",
    "decode_rate",
    "decode_temporal",
]

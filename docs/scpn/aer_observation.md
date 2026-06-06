<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — AER control-observation guide. -->

# AER Control Observation

`scpn_control.scpn.observation` adapts Address-Event Representation spike
streams into the existing SCPN controller feature contract. The adapter keeps
the old `ControlObservation` mapping path intact: existing consumers can keep
reading dictionary or array features, while neuromorphic ingress can decode an
asynchronous spike stream into a bounded `float64` vector.

## Contract

The public Python surface provides:

- `SpikeEvent(neuron_id, timestamp_ns)` for one AER event.
- `SpikeBuffer(capacity)` for a bounded FIFO ring buffer.
- `SpikeBuffer.admission_report()` for bounded ingress evidence.
- `decode_rate(events, window_ns, n_features)` for event-fraction features.
- `decode_temporal(events, window_ns, n_features, now_ns=...)` for first-spike
  timing features.
- `decode_isi(events, window_ns, n_features)` for repeated-spike interval
  features.
- `AERControlObservation.to_features()` for `(n_features,)` `float64` vectors in
  `[0, 1]`.
- `AERControlObservation.to_feature_mapping(prefix="aer_")` for dictionary
  consumers that already use `extract_features(..., passthrough_keys=...)`.

The Rust parity surface lives in `control_core::spike_buffer`. When the optional
PyO3 extension is built, Python can call `scpn_control_rs.PySpikeBuffer`,
`scpn_control_rs.aer_decode_rate`, `scpn_control_rs.aer_decode_temporal`, and
`scpn_control_rs.aer_decode_isi`.

## Buffer semantics

`SpikeBuffer` is bounded by construction. When the buffer is full, `push()` drops
the oldest event and latches `overflowed=True`. `drain_window(window_ns, now_ns)`
returns events with timestamps in `[now_ns - window_ns, now_ns]`, discards older
events, and keeps future events. This makes burst handling explicit and prevents
unbounded memory growth under a noisy neuromorphic input stream.

The buffer also latches timestamp-order admission evidence. `monotonic_input`
remains true only while admitted timestamps are non-decreasing.
`out_of_order_event_count` increments when an event arrives after a later
timestamp has already been observed. `admission_report()` returns capacity,
retained event count, overflow state, monotonic state, and out-of-order count.

Out-of-order events are retained by default. That preserves forensic replay and
keeps the decoder deterministic even when upstream hardware delivers delayed
events. Safety-critical ingress can opt into fail-closed behavior with
`AERControlObservation(require_monotonic=True)`. Strict mode checks the latched
admission state before draining, so rejected events remain available for
post-mortem inspection.

## Decode equations

Rate decoding uses per-feature event fractions:

```text
x_i = count_i / max(1, N_events)
```

Temporal decoding maps the first spike in the active window to a bounded timing
feature:

```text
x_i = 1 - (t_first_i - (t_now - window)) / window
```

ISI decoding maps repeated spikes to a bounded high-frequency feature:

```text
x_i = 1 - mean(diff(t_i)) / window
```

All decoder outputs are clipped to `[0, 1]`. `AERControlObservation` can apply
`unit`, `max`, or bounded `zscore` normalisation after decoding.

## Claim boundary

This adapter is a control-ingress contract, not a facility neuromorphic bus
driver. It does not claim electrical AER signal integrity, FPGA timing closure,
Loihi/Brainchip deployment, radiation tolerance, or PCS hardware admission.
Those claims require separate hardware evidence, binary packet-schema
certification, and target-device latency reports.

## Example

```python
from scpn_control.scpn.observation import AERControlObservation, SpikeBuffer, SpikeEvent

buffer = SpikeBuffer(capacity=128)
buffer.push(SpikeEvent(neuron_id=0, timestamp_ns=10))
buffer.push(SpikeEvent(neuron_id=0, timestamp_ns=30))
buffer.push(SpikeEvent(neuron_id=7, timestamp_ns=80))

obs = AERControlObservation(
    timestamp_ns=100,
    spike_stream=buffer,
    decode_window_ns=100,
    decode_strategy="rate",
    n_features=8,
    require_monotonic=True,
)

features = obs.to_features()
mapping = obs.to_feature_mapping(prefix="aer_")
report = obs.admission_report()
```

## Practical use and scope

Use this page for Address-Event Representation observation wiring in control loops.

- Apply this when introducing or changing neuromorphic observation adapters.
- Keep stream-format assumptions synchronized with downstream feature-contract checks.
- Validate with deterministic traces before declaring production interoperability.


// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Project: SCPN Control
// Description: AER spike-buffer ring and decoders.
//! Address-event spike buffers and decoders for SCPN control observations.

use std::collections::VecDeque;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Single Address-Event Representation spike event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpikeEvent {
    /// Neuron or feature address.
    pub neuron_id: usize,
    /// Event timestamp in nanoseconds.
    pub timestamp_ns: u64,
}

impl SpikeEvent {
    /// Construct a spike event.
    pub fn new(neuron_id: usize, timestamp_ns: u64) -> Self {
        Self {
            neuron_id,
            timestamp_ns,
        }
    }
}

/// Bounded FIFO ring buffer. Overflow drops the oldest event.
#[derive(Clone, Debug)]
pub struct SpikeBuffer {
    capacity: usize,
    events: VecDeque<SpikeEvent>,
    overflowed: bool,
}

impl SpikeBuffer {
    /// Construct a bounded spike buffer.
    pub fn new(capacity: usize) -> Result<Self, SpikeBufferError> {
        if capacity == 0 {
            return Err(SpikeBufferError::NonPositive("capacity"));
        }
        Ok(Self {
            capacity,
            events: VecDeque::with_capacity(capacity),
            overflowed: false,
        })
    }

    /// Maximum retained events.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Current retained event count.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Return true if no events are retained.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Return whether at least one event has been dropped.
    pub fn overflowed(&self) -> bool {
        self.overflowed
    }

    /// Append an event, dropping the oldest if full.
    pub fn push(&mut self, event: SpikeEvent) {
        if self.events.len() >= self.capacity {
            let _ = self.events.pop_front();
            self.overflowed = true;
        }
        self.events.push_back(event);
    }

    /// Return retained events without mutation.
    pub fn snapshot(&self) -> Vec<SpikeEvent> {
        self.events.iter().copied().collect()
    }

    /// Drain events in [now_ns - window_ns, now_ns], discard older events, retain future events.
    pub fn drain_window(
        &mut self,
        window_ns: u64,
        now_ns: u64,
    ) -> Result<Vec<SpikeEvent>, SpikeBufferError> {
        if window_ns == 0 {
            return Err(SpikeBufferError::NonPositive("window_ns"));
        }
        let lower = now_ns.saturating_sub(window_ns);
        let mut drained = Vec::new();
        let mut retained = VecDeque::with_capacity(self.capacity);
        while let Some(event) = self.events.pop_front() {
            if event.timestamp_ns > now_ns {
                retained.push_back(event);
            } else if event.timestamp_ns >= lower {
                drained.push(event);
            }
        }
        self.events = retained;
        Ok(drained)
    }

    /// Clear retained events and reset overflow latch.
    pub fn clear(&mut self) {
        self.events.clear();
        self.overflowed = false;
    }
}

/// Decode AER events as per-feature event fractions.
pub fn decode_rate(
    events: &[SpikeEvent],
    window_ns: u64,
    n_features: usize,
) -> Result<Vec<f64>, SpikeBufferError> {
    validate_window_and_features(window_ns, n_features)?;
    let mut counts = vec![0.0_f64; n_features];
    for event in events {
        if event.neuron_id < n_features {
            counts[event.neuron_id] += 1.0;
        }
    }
    let denominator = (events.len() as f64).max(1.0);
    for value in &mut counts {
        *value = clip01(*value / denominator);
    }
    Ok(counts)
}

/// Decode by first-spike timing; earlier spikes map closer to one.
pub fn decode_temporal(
    events: &[SpikeEvent],
    window_ns: u64,
    n_features: usize,
    now_ns: u64,
) -> Result<Vec<f64>, SpikeBufferError> {
    validate_window_and_features(window_ns, n_features)?;
    let lower = now_ns.saturating_sub(window_ns);
    let mut first = vec![None::<u64>; n_features];
    for event in events {
        if event.neuron_id < n_features
            && event.timestamp_ns >= lower
            && event.timestamp_ns <= now_ns
        {
            match first[event.neuron_id] {
                Some(current) if current <= event.timestamp_ns => {}
                _ => first[event.neuron_id] = Some(event.timestamp_ns),
            }
        }
    }
    let mut features = vec![0.0_f64; n_features];
    for (idx, timestamp) in first.into_iter().enumerate() {
        if let Some(ts) = timestamp {
            features[idx] = clip01(1.0 - ((ts - lower) as f64 / window_ns as f64));
        }
    }
    Ok(features)
}

/// Decode by mean inter-spike interval; faster repeated spikes map higher.
pub fn decode_isi(
    events: &[SpikeEvent],
    window_ns: u64,
    n_features: usize,
) -> Result<Vec<f64>, SpikeBufferError> {
    validate_window_and_features(window_ns, n_features)?;
    let mut by_neuron = vec![Vec::<u64>::new(); n_features];
    for event in events {
        if event.neuron_id < n_features {
            by_neuron[event.neuron_id].push(event.timestamp_ns);
        }
    }
    let mut features = vec![0.0_f64; n_features];
    for (idx, timestamps) in by_neuron.iter_mut().enumerate() {
        if timestamps.len() < 2 {
            continue;
        }
        timestamps.sort_unstable();
        let mut total = 0_u64;
        let mut count = 0_u64;
        for pair in timestamps.windows(2) {
            total += pair[1] - pair[0];
            count += 1;
        }
        let mean_isi = total as f64 / count as f64;
        features[idx] = clip01(1.0 - (mean_isi / window_ns as f64));
    }
    Ok(features)
}

/// Errors raised by the spike-buffer adapter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SpikeBufferError {
    /// Integer parameter must be positive.
    NonPositive(&'static str),
}

impl Display for SpikeBufferError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonPositive(field) => write!(f, "{field} must be positive"),
        }
    }
}

impl Error for SpikeBufferError {}

fn validate_window_and_features(window_ns: u64, n_features: usize) -> Result<(), SpikeBufferError> {
    if window_ns == 0 {
        return Err(SpikeBufferError::NonPositive("window_ns"));
    }
    if n_features == 0 {
        return Err(SpikeBufferError::NonPositive("n_features"));
    }
    Ok(())
}

fn clip01(value: f64) -> f64 {
    if value < 0.0 {
        0.0
    } else if value > 1.0 {
        1.0
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_drops_oldest_on_overflow() {
        let mut buffer = SpikeBuffer::new(2).expect("valid buffer");
        buffer.push(SpikeEvent::new(1, 10));
        buffer.push(SpikeEvent::new(2, 20));
        buffer.push(SpikeEvent::new(3, 30));
        assert_eq!(
            buffer.snapshot(),
            vec![SpikeEvent::new(2, 20), SpikeEvent::new(3, 30)]
        );
        assert!(buffer.overflowed());
    }

    #[test]
    fn drain_window_retains_future_events() {
        let mut buffer = SpikeBuffer::new(4).expect("valid buffer");
        buffer.push(SpikeEvent::new(0, 10));
        buffer.push(SpikeEvent::new(1, 20));
        buffer.push(SpikeEvent::new(2, 40));
        let drained = buffer.drain_window(15, 25).expect("drain");
        assert_eq!(
            drained,
            vec![SpikeEvent::new(0, 10), SpikeEvent::new(1, 20)]
        );
        assert_eq!(buffer.snapshot(), vec![SpikeEvent::new(2, 40)]);
    }

    #[test]
    fn decoders_match_analytical_vectors() {
        let events = vec![
            SpikeEvent::new(0, 10),
            SpikeEvent::new(0, 30),
            SpikeEvent::new(1, 70),
            SpikeEvent::new(3, 80),
        ];
        assert_eq!(
            decode_rate(&events, 100, 4).expect("rate"),
            vec![0.5, 0.25, 0.0, 0.25]
        );
        assert_vec_close(
            &decode_temporal(&events, 100, 4, 100).expect("temporal"),
            &[0.9, 0.3, 0.0, 0.2],
        );
        assert_eq!(
            decode_isi(&events, 100, 4).expect("isi"),
            vec![0.8, 0.0, 0.0, 0.0]
        );
    }

    fn assert_vec_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (left, right) in actual.iter().zip(expected) {
            assert!((left - right).abs() <= 1e-12);
        }
    }
}

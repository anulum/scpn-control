// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — PyO3 AER spike-buffer bindings.
//! Python bindings for AER spike buffers and decoders.

use control_core::spike_buffer::{
    decode_isi, decode_rate, decode_temporal, SpikeBuffer, SpikeEvent,
};
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Python wrapper for the Rust spike buffer.
#[pyclass(name = "PySpikeBuffer")]
pub(crate) struct PySpikeBuffer {
    inner: SpikeBuffer,
}

#[pymethods]
impl PySpikeBuffer {
    /// Construct a bounded spike buffer.
    #[new]
    fn new(capacity: usize) -> PyResult<Self> {
        SpikeBuffer::new(capacity)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Append a spike event.
    fn push(&mut self, neuron_id: usize, timestamp_ns: u64) {
        self.inner.push(SpikeEvent::new(neuron_id, timestamp_ns));
    }

    /// Drain events in [now_ns - window_ns, now_ns].
    fn drain_window(&mut self, window_ns: u64, now_ns: u64) -> PyResult<Vec<(usize, u64)>> {
        self.inner
            .drain_window(window_ns, now_ns)
            .map(|events| {
                events
                    .into_iter()
                    .map(|event| (event.neuron_id, event.timestamp_ns))
                    .collect()
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Return retained events without mutation.
    fn snapshot(&self) -> Vec<(usize, u64)> {
        self.inner
            .snapshot()
            .into_iter()
            .map(|event| (event.neuron_id, event.timestamp_ns))
            .collect()
    }

    /// Clear retained events and reset overflow latch.
    fn clear(&mut self) {
        self.inner.clear();
    }

    /// Return retained event count.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Return buffer capacity.
    #[getter]
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Return whether the buffer has dropped events.
    #[getter]
    fn overflowed(&self) -> bool {
        self.inner.overflowed()
    }

    /// Return count of out-of-order admitted events.
    #[getter]
    fn out_of_order_event_count(&self) -> u64 {
        self.inner.out_of_order_event_count()
    }

    /// Return whether admitted timestamps were non-decreasing.
    #[getter]
    fn monotonic_input(&self) -> bool {
        self.inner.monotonic_input()
    }

    /// Return bounded-buffer admission metadata for safety gating.
    fn admission_report<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let report = self.inner.admission_report();
        let dict = PyDict::new(py);
        dict.set_item("capacity", report.capacity)?;
        dict.set_item("retained_events", report.retained_events)?;
        dict.set_item("overflowed", report.overflowed)?;
        dict.set_item("out_of_order_event_count", report.out_of_order_event_count)?;
        dict.set_item("monotonic_input", report.monotonic_input)?;
        Ok(dict)
    }
}

/// Decode AER events as per-feature event fractions.
#[pyfunction]
pub(crate) fn aer_decode_rate<'py>(
    py: Python<'py>,
    events: Vec<(usize, u64)>,
    window_ns: u64,
    n_features: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let spike_events = events_to_spikes(events);
    decode_rate(&spike_events, window_ns, n_features)
        .map(|values| Array1::from_vec(values).into_pyarray(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Decode by first-spike timing.
#[pyfunction]
pub(crate) fn aer_decode_temporal<'py>(
    py: Python<'py>,
    events: Vec<(usize, u64)>,
    window_ns: u64,
    n_features: usize,
    now_ns: u64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let spike_events = events_to_spikes(events);
    decode_temporal(&spike_events, window_ns, n_features, now_ns)
        .map(|values| Array1::from_vec(values).into_pyarray(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Decode by mean inter-spike interval.
#[pyfunction]
pub(crate) fn aer_decode_isi<'py>(
    py: Python<'py>,
    events: Vec<(usize, u64)>,
    window_ns: u64,
    n_features: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let spike_events = events_to_spikes(events);
    decode_isi(&spike_events, window_ns, n_features)
        .map(|values| Array1::from_vec(values).into_pyarray(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Register AER spike-buffer bindings.
pub(crate) fn register_spike_buffer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySpikeBuffer>()?;
    m.add_function(wrap_pyfunction!(aer_decode_rate, m)?)?;
    m.add_function(wrap_pyfunction!(aer_decode_temporal, m)?)?;
    m.add_function(wrap_pyfunction!(aer_decode_isi, m)?)?;
    Ok(())
}

fn events_to_spikes(events: Vec<(usize, u64)>) -> Vec<SpikeEvent> {
    events
        .into_iter()
        .map(|(neuron_id, timestamp_ns)| SpikeEvent::new(neuron_id, timestamp_ns))
        .collect()
}

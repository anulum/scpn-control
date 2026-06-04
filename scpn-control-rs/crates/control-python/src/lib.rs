// SPDX-License-Identifier: AGPL-3.0-or-later
// ──────────────────────────────────────────────────────────────────────
// SCPN Control — Rust Crate
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// ──────────────────────────────────────────────────────────────────────

//! PyO3 Python bindings for SCPN Control (Modern Bound API).

use ndarray::{Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use transport_bridge::PyUdpTransportBridge;

use control_control::analytic;
use control_control::digital_twin::Plasma2D;
use control_control::h_infinity::HInfController;
use control_control::mpc::{MPController, NeuralSurrogate};
use control_control::snn::{NeuroCyberneticController, SpikingControllerPool};
use control_control::spi::SPIMitigation;
use control_core::amr_kernel::{AmrKernelConfig, AmrKernelSolver};
use control_core::bfield;
use control_core::bout_interface::{self, BoutGridConfig};
use control_core::ignition;
use control_core::kernel::FusionKernel;
use control_core::particles::{self, ChargedParticle};
use control_core::transport::TransportSolver;
use control_core::vmec_interface::{self, VmecBoundaryState, VmecFourierMode, VmecSolverConfig};
use control_core::xpoint;
use control_math::kuramoto;
use control_math::multigrid::{multigrid_solve, MultigridConfig};
use control_math::tridiag;
use control_types::state::Grid2D;

mod slab;
mod transport_bridge;

// ─── Equilibrium solver ───

#[pyclass]
struct PyFusionKernel {
    inner: FusionKernel,
}

#[pyclass]
#[derive(Clone)]
struct PyEquilibriumResult {
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    residual: f64,
    #[pyo3(get)]
    axis_r: f64,
    #[pyo3(get)]
    axis_z: f64,
    #[pyo3(get)]
    x_point_r: f64,
    #[pyo3(get)]
    x_point_z: f64,
    #[pyo3(get)]
    psi_axis: f64,
    #[pyo3(get)]
    psi_boundary: f64,
    #[pyo3(get)]
    solve_time_ms: f64,
}

#[pymethods]
impl PyFusionKernel {
    #[new]
    fn new(config_path: &str) -> PyResult<Self> {
        let inner =
            FusionKernel::from_file(config_path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(PyFusionKernel { inner })
    }

    fn solve_equilibrium<'py>(&mut self, _py: Python<'py>) -> PyResult<PyEquilibriumResult> {
        let result = self
            .inner
            .solve_equilibrium()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyEquilibriumResult {
            converged: result.converged,
            iterations: result.iterations,
            residual: result.residual,
            axis_r: result.axis_position.0,
            axis_z: result.axis_position.1,
            x_point_r: result.x_point_position.0,
            x_point_z: result.x_point_position.1,
            psi_axis: result.psi_axis,
            psi_boundary: result.psi_boundary,
            solve_time_ms: result.solve_time_ms,
        })
    }

    fn get_psi<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.psi().clone().into_pyarray(py)
    }

    fn get_r<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.grid().r.clone().into_pyarray(py)
    }

    fn get_z<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.grid().z.clone().into_pyarray(py)
    }

    #[allow(clippy::type_complexity)]
    fn compute_b_field<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let (br, bz) = bfield::compute_b_field(self.inner.psi(), self.inner.grid())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((br.into_pyarray(py), bz.into_pyarray(py)))
    }

    fn find_x_point(&self, z_threshold: f64) -> PyResult<((f64, f64), f64)> {
        xpoint::find_x_point(self.inner.psi(), self.inner.grid(), z_threshold)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Calculate the vacuum magnetic flux from the external coil set.
    fn calculate_vacuum_field<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mu0 = 4.0 * std::f64::consts::PI * 1e-7; // Default mu0
        let psi_vac = control_core::vacuum::calculate_vacuum_field(
            self.inner.grid(),
            self.inner.coils(),
            mu0,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(psi_vac.into_pyarray(py))
    }

    fn sample_psi_at(&self, r: f64, z: f64) -> PyResult<f64> {
        self.inner
            .sample_psi_at(r, z)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn calculate_thermodynamics<'py>(
        &self,
        py: Python<'py>,
        p_aux_mw: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = ignition::calculate_thermodynamics(&self.inner, p_aux_mw)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dict = PyDict::new(py);
        dict.set_item("p_fusion_mw", result.p_fusion_mw)?;
        dict.set_item("p_alpha_mw", result.p_alpha_mw)?;
        dict.set_item("p_loss_mw", result.p_loss_mw)?;
        dict.set_item("p_aux_mw", result.p_aux_mw)?;
        dict.set_item("net_mw", result.net_mw)?;
        dict.set_item("q_factor", result.q_factor)?;
        dict.set_item("t_peak_kev", result.t_peak_kev)?;
        dict.set_item("w_thermal_mj", result.w_thermal_mj)?;
        Ok(dict)
    }
}

// ─── SCPN kernels ───

#[pyfunction]
fn scpn_dense_activations<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray2<'py, f64>,
    marking: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let w = weights.as_array();
    let m = marking.as_array();
    let out = w.dot(&m);
    Ok(out.to_owned().into_pyarray(py))
}

#[pyfunction]
fn scpn_marking_update<'py>(
    py: Python<'py>,
    marking: PyReadonlyArray1<'py, f64>,
    wi: PyReadonlyArray2<'py, f64>,
    wo: PyReadonlyArray2<'py, f64>,
    firing: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let m = marking.as_array();
    let f = firing.as_array();
    let wi_arr = wi.as_array();
    let wo_arr = wo.as_array();
    let cons = wi_arr.t().dot(&f);
    let prod = wo_arr.dot(&f);
    let out = &m - &cons + &prod;
    Ok(out.to_owned().into_pyarray(py))
}

// ─── Control ───

#[pyfunction]
fn shafranov_bv<'py>(
    py: Python<'py>,
    r_major: f64,
    a_minor: f64,
    ip_ma: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let res = analytic::shafranov_bv(r_major, a_minor, ip_ma)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("bv_required", res.bv_required)?;
    dict.set_item("term_log", res.term_log)?;
    dict.set_item("term_physics", res.term_physics)?;
    Ok(dict)
}

#[pyfunction]
fn solve_coil_currents<'py>(
    py: Python<'py>,
    response_matrix: PyReadonlyArray2<'py, f64>,
    target_bv: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let resp = response_matrix.as_array().to_owned();
    let flat_resp = resp
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("response_matrix not contiguous"))?;
    let currents = analytic::solve_coil_currents(flat_resp, target_bv)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(Array1::from_vec(currents).into_pyarray(py))
}

#[pyclass]
struct PySnnPool {
    inner: SpikingControllerPool,
}

#[pymethods]
impl PySnnPool {
    #[new]
    #[pyo3(signature = (n_neurons=50, gain=10.0, window_size=20))]
    fn new(n_neurons: usize, gain: f64, window_size: usize) -> PyResult<Self> {
        let inner = SpikingControllerPool::new(n_neurons, gain, window_size)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PySnnPool { inner })
    }

    fn step(&mut self, error: f64) -> PyResult<f64> {
        self.inner
            .step(error)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn n_neurons(&self) -> usize {
        self.inner.n_neurons
    }

    #[getter]
    fn gain(&self) -> f64 {
        self.inner.gain
    }
}

#[pyclass]
struct PySnnController {
    inner: NeuroCyberneticController,
}

#[pymethods]
impl PySnnController {
    #[new]
    fn new(target_r: f64, target_z: f64) -> PyResult<Self> {
        let inner = NeuroCyberneticController::new(target_r, target_z)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PySnnController { inner })
    }

    fn step(&mut self, r_err: f64, z_err: f64) -> PyResult<(f64, f64)> {
        self.inner
            .step(r_err, z_err)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn target_r(&self) -> f64 {
        self.inner.target_r
    }

    #[getter]
    fn target_z(&self) -> f64 {
        self.inner.target_z
    }
}

#[pyclass]
struct PySpikingControllerPool {
    n_neurons: usize,
    seed: u64,
    brain_r: SpikingControllerPool,
    brain_z: SpikingControllerPool,
    target_r: f64,
    target_z: f64,
    pid_kp: f64,
    pid_ki: f64,
    pid_kd: f64,
    pid_integral_r: f64,
    pid_integral_z: f64,
    pid_prev_error_r: f64,
    pid_prev_error_z: f64,
    transport_endpoint: String,
    transport_port: u16,
    transport_ttl: u8,
    transport_max_queue: usize,
    transport_backend: String,
    transport_heartbeat_port: u16,
    transport_heartbeat_timeout_ms: u64,
    core_snn: usize,
    core_z3: usize,
    core_net: usize,
    core_hb: usize,
    max_iterations: Option<usize>,
    tick_interval_s: f64,
    initial_state: (f64, f64),
    plant_gain: f64,
    u_min: Option<f64>,
    u_max: Option<f64>,
    kuramoto_weights: HashMap<String, f64>,
    itpa_constraints: HashMap<String, f64>,
    acados_targets: HashMap<String, f64>,
    steps_executed: usize,
    dropped: usize,
    publish_failures: usize,
    max_publish_failures: usize,
    last_cycle_ns: u128,
    total_cycle_ns: u128,
    last_acados_time_ns: u64,
    last_snn_time_ns: u64,
    is_running: bool,
    stop_flag: Arc<AtomicBool>,
}

#[pymethods]
impl PySpikingControllerPool {
    #[new]
    #[pyo3(signature = (n_neurons=64, seed=7))]
    fn new(n_neurons: usize, seed: u64) -> PyResult<Self> {
        if n_neurons == 0 {
            return Err(PyValueError::new_err("n_neurons must be positive"));
        }

        let brain_r = SpikingControllerPool::new(n_neurons, 10.0, 20)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let brain_z = SpikingControllerPool::new(n_neurons, 20.0, 20)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self {
            n_neurons,
            seed,
            brain_r,
            brain_z,
            target_r: 6.2,
            target_z: 0.0,
            pid_kp: 1.0,
            pid_ki: 0.05,
            pid_kd: 0.01,
            pid_integral_r: 0.0,
            pid_integral_z: 0.0,
            pid_prev_error_r: 0.0,
            pid_prev_error_z: 0.0,
            transport_endpoint: "239.0.0.1".to_string(),
            transport_port: 5555,
            transport_ttl: 1,
            transport_max_queue: 4,
            transport_backend: "std".to_string(),
            transport_heartbeat_port: 0,
            transport_heartbeat_timeout_ms: 3,
            core_snn: 1,
            core_z3: 2,
            core_net: 3,
            core_hb: 4,
            max_iterations: None,
            tick_interval_s: 0.001,
            initial_state: (6.2, 0.0),
            plant_gain: 0.0,
            u_min: None,
            u_max: None,
            kuramoto_weights: HashMap::new(),
            itpa_constraints: HashMap::new(),
            acados_targets: HashMap::new(),
            steps_executed: 0,
            dropped: 0,
            publish_failures: 0,
            max_publish_failures: 128,
            last_cycle_ns: 0,
            total_cycle_ns: 0,
            last_acados_time_ns: 0,
            last_snn_time_ns: 0,
            is_running: false,
            stop_flag: Arc::new(AtomicBool::new(false)),
        })
    }

    #[pyo3(signature = (endpoint, port, ttl, max_queue, backend, heartbeat_port, heartbeat_timeout_ms))]
    fn set_transport_settings(
        &mut self,
        endpoint: &str,
        port: u16,
        ttl: u8,
        max_queue: usize,
        backend: &str,
        heartbeat_port: u16,
        heartbeat_timeout_ms: u64,
    ) -> PyResult<()> {
        if port == 0 {
            return Err(PyValueError::new_err("transport port must be positive"));
        }
        if max_queue == 0 {
            return Err(PyValueError::new_err(
                "transport max_queue must be positive",
            ));
        }
        if heartbeat_timeout_ms == 0 {
            return Err(PyValueError::new_err(
                "heartbeat_timeout_ms must be positive",
            ));
        }

        let backend_lc = backend.trim().to_ascii_lowercase();
        if !matches!(
            backend_lc.as_str(),
            "std" | "udp" | "io-uring" | "io_uring" | "ioring"
        ) {
            return Err(PyValueError::new_err(
                "transport backend must be std, udp, or io-uring",
            ));
        }

        self.transport_endpoint = endpoint.to_string();
        self.transport_port = port;
        self.transport_ttl = ttl;
        self.transport_max_queue = max_queue;
        self.transport_backend = if matches!(backend_lc.as_str(), "io_uring" | "ioring") {
            "io-uring".to_string()
        } else {
            backend_lc
        };
        self.transport_heartbeat_port = heartbeat_port;
        self.transport_heartbeat_timeout_ms = heartbeat_timeout_ms;
        Ok(())
    }

    #[pyo3(signature = (endpoint, port, ttl, max_queue, backend, heartbeat_port, heartbeat_timeout_ms))]
    fn configure_transport(
        &mut self,
        endpoint: &str,
        port: u16,
        ttl: u8,
        max_queue: usize,
        backend: &str,
        heartbeat_port: u16,
        heartbeat_timeout_ms: u64,
    ) -> PyResult<()> {
        self.set_transport_settings(
            endpoint,
            port,
            ttl,
            max_queue,
            backend,
            heartbeat_port,
            heartbeat_timeout_ms,
        )
    }

    #[pyo3(signature = (core_snn, core_z3, core_net, core_hb))]
    fn set_execution_affinity(
        &mut self,
        core_snn: usize,
        core_z3: usize,
        core_net: usize,
        core_hb: usize,
    ) {
        self.core_snn = core_snn;
        self.core_z3 = core_z3;
        self.core_net = core_net;
        self.core_hb = core_hb;
    }

    fn set_kuramoto_weights(&mut self, weights: &Bound<'_, PyDict>) -> PyResult<()> {
        self.kuramoto_weights = parse_f64_dict(weights)?;
        Ok(())
    }

    fn configure_itpa_gyro_bohm(&mut self, constraints: &Bound<'_, PyDict>) -> PyResult<()> {
        self.itpa_constraints = parse_f64_dict(constraints)?;
        if let Some(c_gb) = self.itpa_constraints.get("c_gB") {
            self.acados_targets.insert("c_gB".to_string(), *c_gb);
        }
        Ok(())
    }

    fn set_nmpc_targets(&mut self, targets: &Bound<'_, PyDict>) -> PyResult<()> {
        self.apply_acados_targets(targets)
    }

    fn set_acados_targets(&mut self, targets: &Bound<'_, PyDict>) -> PyResult<()> {
        self.apply_acados_targets(targets)
    }

    fn configure_acados_targets(&mut self, targets: &Bound<'_, PyDict>) -> PyResult<()> {
        self.apply_acados_targets(targets)
    }

    #[pyo3(signature = (max_iterations=None, tick_interval=0.001, initial_state=None, plant_gain=0.0))]
    fn configure_runtime_budget(
        &mut self,
        max_iterations: Option<usize>,
        tick_interval: f64,
        initial_state: Option<Vec<f64>>,
        plant_gain: f64,
    ) -> PyResult<()> {
        if !tick_interval.is_finite() || tick_interval < 0.0 {
            return Err(PyValueError::new_err(
                "tick_interval must be finite and >= 0",
            ));
        }
        if !plant_gain.is_finite() {
            return Err(PyValueError::new_err("plant_gain must be finite"));
        }

        if let Some(values) = initial_state {
            if values.len() != 2 {
                return Err(PyValueError::new_err(
                    "initial_state must contain exactly two floats",
                ));
            }
            if !values[0].is_finite() || !values[1].is_finite() {
                return Err(PyValueError::new_err("initial_state must be finite"));
            }
            self.initial_state = (values[0], values[1]);
        }

        self.max_iterations = max_iterations;
        self.tick_interval_s = tick_interval;
        self.plant_gain = plant_gain;
        Ok(())
    }

    #[pyo3(signature = (arg1=None, arg2=None, arg3=None))]
    fn start(
        &mut self,
        py: Python<'_>,
        arg1: Option<usize>,
        arg2: Option<usize>,
        arg3: Option<usize>,
    ) -> PyResult<()> {
        if self.is_running {
            return Ok(());
        }

        let mut max_iterations = self.max_iterations;
        match (arg1, arg2, arg3) {
            (Some(steps), None, None) => {
                max_iterations = Some(steps);
            }
            (Some(core_snn), Some(core_z3), None) => {
                self.core_snn = core_snn;
                self.core_z3 = core_z3;
            }
            (Some(core_snn), Some(core_z3), Some(steps)) => {
                self.core_snn = core_snn;
                self.core_z3 = core_z3;
                max_iterations = Some(steps);
            }
            _ => {}
        }

        self.stop_flag.store(false, Ordering::Release);
        self.steps_executed = 0;
        self.dropped = 0;
        self.publish_failures = 0;
        self.last_cycle_ns = 0;
        self.total_cycle_ns = 0;
        self.last_acados_time_ns = 0;
        self.last_snn_time_ns = 0;
        self.is_running = true;

        let result = py.allow_threads(|| self.run_native_loop(max_iterations));
        self.is_running = false;
        result
    }

    fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Release);
        self.is_running = false;
    }

    fn force_shutdown(&mut self) {
        self.stop();
    }

    fn set_max_publish_failures(&mut self, max_publish_failures: usize) {
        self.max_publish_failures = max_publish_failures;
    }

    fn extract_slab_telemetry<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.telemetry(py)
    }

    fn campaign_summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.telemetry(py)
    }
}

impl PySpikingControllerPool {
    fn apply_acados_targets(&mut self, targets: &Bound<'_, PyDict>) -> PyResult<()> {
        let parsed = parse_f64_dict(targets)?;
        if let Some(target_r) = parsed
            .get("target_r")
            .or_else(|| parsed.get("rho_tor_target"))
            .or_else(|| parsed.get("R_target"))
        {
            self.target_r = *target_r;
        }
        if let Some(target_z) = parsed
            .get("target_z")
            .or_else(|| parsed.get("z_tor_target"))
            .or_else(|| parsed.get("Z_target"))
        {
            self.target_z = *target_z;
        }
        if let Some(u_min) = parsed.get("u_min") {
            self.u_min = Some(*u_min);
        }
        if let Some(u_max) = parsed.get("u_max") {
            self.u_max = Some(*u_max);
        }
        self.acados_targets.extend(parsed);
        Ok(())
    }

    fn run_native_loop(&mut self, max_iterations: Option<usize>) -> PyResult<()> {
        let mut bridge = PyUdpTransportBridge::new(
            &self.transport_endpoint,
            self.transport_port,
            self.transport_ttl,
            self.transport_max_queue,
            &self.transport_backend,
            self.transport_heartbeat_port,
            self.transport_heartbeat_timeout_ms,
        )?;
        bridge.start()?;

        let mut measured_r = self.initial_state.0;
        let mut measured_z = self.initial_state.1;
        let loop_result = loop {
            if self.stop_flag.load(Ordering::Acquire) {
                break Ok(());
            }

            if let Some(limit) = max_iterations {
                if self.steps_executed >= limit {
                    break Ok(());
                }
            }

            if bridge.heartbeat_expired() {
                break Err(PyRuntimeError::new_err("transport heartbeat timeout"));
            }

            let step_start = Instant::now();
            let error_r = self.target_r - measured_r;
            let error_z = self.target_z - measured_z;

            let snn_start = Instant::now();
            let r_snn = self
                .brain_r
                .step(error_r)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let z_snn = self
                .brain_z
                .step(error_z)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            self.last_snn_time_ns = snn_start.elapsed().as_nanos() as u64;

            let pid_start = Instant::now();
            self.pid_integral_r =
                (self.pid_integral_r + error_r * self.tick_interval_s).clamp(-1.0e6, 1.0e6);
            self.pid_integral_z =
                (self.pid_integral_z + error_z * self.tick_interval_s).clamp(-1.0e6, 1.0e6);
            let d_error_r = error_r - self.pid_prev_error_r;
            let d_error_z = error_z - self.pid_prev_error_z;
            self.pid_prev_error_r = error_r;
            self.pid_prev_error_z = error_z;

            let r_pid =
                self.pid_kp * error_r + self.pid_ki * self.pid_integral_r + self.pid_kd * d_error_r;
            let z_pid =
                self.pid_kp * error_z + self.pid_ki * self.pid_integral_z + self.pid_kd * d_error_z;
            self.last_acados_time_ns = pid_start.elapsed().as_nanos() as u64;

            let mut r_command = r_snn + r_pid;
            let mut z_command = z_snn + z_pid;
            if let Some(u_min) = self.u_min {
                r_command = r_command.max(u_min);
                z_command = z_command.max(u_min);
            }
            if let Some(u_max) = self.u_max {
                r_command = r_command.min(u_max);
                z_command = z_command.min(u_max);
            }

            match bridge.publish(
                error_r,
                error_z,
                r_command,
                z_command,
                self.last_acados_time_ns,
                self.last_snn_time_ns,
                0,
            ) {
                Ok(true) => {
                    self.publish_failures = 0;
                }
                Ok(false) => {
                    self.dropped = self.dropped.saturating_add(1);
                    self.publish_failures = self.publish_failures.saturating_add(1);
                    if self.publish_failures > self.max_publish_failures {
                        break Err(PyRuntimeError::new_err(
                            "transport publish backpressure exceeded tolerance",
                        ));
                    }
                }
                Err(error) => {
                    break Err(error);
                }
            }

            measured_r += self.plant_gain * r_command * self.tick_interval_s;
            measured_z += self.plant_gain * z_command * self.tick_interval_s;
            self.steps_executed = self.steps_executed.saturating_add(1);
            self.last_cycle_ns = step_start.elapsed().as_nanos();
            self.total_cycle_ns = self.total_cycle_ns.saturating_add(self.last_cycle_ns);

            if self.tick_interval_s > 0.0 {
                let target_ns = (self.tick_interval_s * 1_000_000_000.0) as u128;
                if target_ns > self.last_cycle_ns {
                    std::thread::sleep(Duration::from_nanos(
                        (target_ns - self.last_cycle_ns) as u64,
                    ));
                }
            }
        };

        let stop_result = bridge.stop();
        if loop_result.is_ok() {
            stop_result?;
        }
        loop_result
    }

    fn telemetry<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("running", self.is_running)?;
        dict.set_item("n_neurons", self.n_neurons)?;
        dict.set_item("seed", self.seed)?;
        dict.set_item("steps", self.steps_executed)?;
        dict.set_item("dropped", self.dropped)?;
        dict.set_item("publish_failures", self.publish_failures)?;
        dict.set_item("last_cycle_ns", self.last_cycle_ns.to_string())?;
        dict.set_item("total_cycle_ns", self.total_cycle_ns.to_string())?;
        dict.set_item("last_acados_time_ns", self.last_acados_time_ns)?;
        dict.set_item("last_snn_time_ns", self.last_snn_time_ns)?;
        dict.set_item("target_r", self.target_r)?;
        dict.set_item("target_z", self.target_z)?;
        dict.set_item("transport_backend", self.transport_backend.clone())?;
        dict.set_item("transport_endpoint", self.transport_endpoint.clone())?;
        dict.set_item("transport_port", self.transport_port)?;
        dict.set_item("heartbeat_enabled", self.transport_heartbeat_port > 0)?;

        let execution = PyDict::new(py);
        execution.set_item("core_snn", self.core_snn)?;
        execution.set_item("core_z3", self.core_z3)?;
        execution.set_item("core_net", self.core_net)?;
        execution.set_item("core_hb", self.core_hb)?;
        execution.set_item("mode", "native")?;
        dict.set_item("execution", execution)?;
        Ok(dict)
    }
}

fn parse_f64_dict(dict: &Bound<'_, PyDict>) -> PyResult<HashMap<String, f64>> {
    let mut parsed = HashMap::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let key = key.extract::<String>()?;
        if let Ok(number) = value.extract::<f64>() {
            if number.is_finite() {
                parsed.insert(key, number);
            }
        }
    }
    Ok(parsed)
}

#[pyclass]
struct PyMpcController {
    inner: MPController,
}

#[pymethods]
impl PyMpcController {
    #[new]
    fn new(
        weights: PyReadonlyArray2<'_, f64>,
        target: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let model = NeuralSurrogate {
            b_matrix: weights.as_array().to_owned(),
        };
        let inner = MPController::new(model, target.as_array().to_owned())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyMpcController { inner })
    }

    fn plan<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let s = state.as_array().to_owned();
        let u = self
            .inner
            .plan(&s)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(u.into_pyarray(py))
    }
}

#[pyclass]
struct PyPlasma2D {
    inner: Plasma2D,
}

#[pymethods]
impl PyPlasma2D {
    #[new]
    fn new() -> Self {
        PyPlasma2D {
            inner: Plasma2D::new(),
        }
    }

    fn step(&mut self, action: f64) -> PyResult<(f64, f64)> {
        self.inner
            .step(action)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

#[pyclass]
struct PyTransportSolver {
    inner: TransportSolver,
}

#[pymethods]
impl PyTransportSolver {
    #[new]
    fn new() -> Self {
        PyTransportSolver {
            inner: TransportSolver::new(),
        }
    }

    fn step(&mut self, dt: f64, p_heat: f64) -> PyResult<()> {
        self.inner
            .step(dt, p_heat)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn get_te<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.profiles.te.clone().into_pyarray(py)
    }
}

#[pyclass]
struct PyHInfController {
    inner: HInfController,
}

#[pymethods]
impl PyHInfController {
    #[new]
    fn new(
        a: PyReadonlyArray2<'_, f64>,
        b2: PyReadonlyArray2<'_, f64>,
        c2: PyReadonlyArray2<'_, f64>,
        gamma: f64,
        dt: f64,
    ) -> PyResult<Self> {
        let plant = control_control::h_infinity::HInfPlant::new(
            a.as_array().to_owned(),
            Array2::zeros((a.shape()[0], 1)),
            b2.as_array().to_owned(),
            Array2::zeros((1, a.shape()[0])),
            c2.as_array().to_owned(),
        )
        .map_err(PyValueError::new_err)?;
        Ok(PyHInfController {
            inner: HInfController::new(plant, gamma, 1e6, dt),
        })
    }

    fn step(&mut self, y: f64, dt: f64) -> f64 {
        self.inner.step(y, dt)
    }

    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma
    }

    #[getter]
    fn u_max(&self) -> f64 {
        self.inner.u_max
    }
}

// ─── Math Solvers ───

#[pyfunction]
fn py_thomas_solve<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    c: PyReadonlyArray1<'py, f64>,
    d: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = tridiag::thomas_solve(a.as_slice()?, b.as_slice()?, c.as_slice()?, d.as_slice()?);
    Ok(Array1::from_vec(x).into_pyarray(py))
}

#[pyclass]
struct PyAmrSolver {
    inner: AmrKernelSolver,
}

#[pymethods]
impl PyAmrSolver {
    #[new]
    #[pyo3(signature = (max_levels=2, refinement_threshold=0.1, omega=1.8, coarse_iters=400, patch_iters=300, blend=0.5))]
    fn new(
        max_levels: usize,
        refinement_threshold: f64,
        omega: f64,
        coarse_iters: usize,
        patch_iters: usize,
        blend: f64,
    ) -> PyResult<Self> {
        let config = AmrKernelConfig {
            max_levels,
            refinement_threshold,
            omega,
            coarse_iters,
            patch_iters,
            blend,
        };
        let inner =
            AmrKernelSolver::new(config).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyAmrSolver { inner })
    }

    fn solve_with_hierarchy<'py>(
        &self,
        py: Python<'py>,
        psi: PyReadonlyArray2<'py, f64>,
        r: PyReadonlyArray1<'py, f64>,
        z: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let rs = r.as_slice()?;
        let zs = z.as_slice()?;
        let nr = r.shape()[0];
        let nz = z.shape()[0];
        let grid = Grid2D::new(nr, nz, rs[0], rs[nr - 1], zs[0], zs[nz - 1]);
        let psi_out = psi.as_array().to_owned();
        self.inner
            .solve_with_hierarchy(&grid, &psi_out)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(psi_out.into_pyarray(py))
    }
}

#[pyclass]
struct PyVmecSolver {
    config: VmecSolverConfig,
}

#[pymethods]
impl PyVmecSolver {
    #[new]
    #[pyo3(signature = (m_pol=6, n_tor=0, ns=25, ntheta=32, nzeta=1, max_iter=500, tol=1e-8, step_size=5e-3))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        m_pol: usize,
        n_tor: usize,
        ns: usize,
        ntheta: usize,
        nzeta: usize,
        max_iter: usize,
        tol: f64,
        step_size: f64,
    ) -> Self {
        PyVmecSolver {
            config: VmecSolverConfig {
                m_pol,
                n_tor,
                ns,
                ntheta,
                nzeta,
                max_iter,
                tol,
                step_size,
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_fixed_boundary<'py>(
        &self,
        py: Python<'py>,
        r_axis: f64,
        z_axis: f64,
        a_minor: f64,
        kappa: f64,
        triangularity: f64,
        modes: Vec<(i32, i32, f64, f64, f64, f64)>,
        pressure: PyReadonlyArray1<'py, f64>,
        iota: PyReadonlyArray1<'py, f64>,
        phi_edge: f64,
    ) -> PyResult<PyObject> {
        let fourier_modes: Vec<VmecFourierMode> = modes
            .into_iter()
            .map(|(m, n, r_cos, r_sin, z_cos, z_sin)| VmecFourierMode {
                m,
                n,
                r_cos,
                r_sin,
                z_cos,
                z_sin,
            })
            .collect();
        let boundary = VmecBoundaryState {
            r_axis,
            z_axis,
            a_minor,
            kappa,
            triangularity,
            nfp: 1,
            modes: fourier_modes,
        };
        let result = vmec_interface::vmec_fixed_boundary_solve(
            &boundary,
            &self.config,
            pressure.as_slice()?,
            iota.as_slice()?,
            phi_edge,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let dict = PyDict::new(py);
        dict.set_item("rmnc", result.rmnc.into_pyarray(py))?;
        dict.set_item("zmns", result.zmns.into_pyarray(py))?;
        dict.set_item("iterations", result.iterations)?;
        dict.set_item("force_residual", result.force_residual)?;
        dict.set_item("converged", result.converged)?;
        Ok(dict.into_any().unbind())
    }
}

#[pyclass]
struct PyBoutInterface {
    config: BoutGridConfig,
}

#[pymethods]
impl PyBoutInterface {
    #[new]
    #[pyo3(signature = (nx=36, ny=64, nz=32, psi_inner=0.1, psi_outer=0.95))]
    fn new(nx: usize, ny: usize, nz: usize, psi_inner: f64, psi_outer: f64) -> Self {
        PyBoutInterface {
            config: BoutGridConfig {
                nx,
                ny,
                nz,
                psi_inner,
                psi_outer,
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_grid<'py>(
        &self,
        py: Python<'py>,
        psi: PyReadonlyArray2<'py, f64>,
        r: PyReadonlyArray1<'py, f64>,
        z: PyReadonlyArray1<'py, f64>,
        psi_axis: f64,
        psi_boundary: f64,
        b_toroidal: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let rs = r.as_slice()?;
        let zs = z.as_slice()?;
        let nr = r.shape()[0];
        let nz = z.shape()[0];
        let grid_2d = Grid2D::new(nr, nz, rs[0], rs[nr - 1], zs[0], zs[nz - 1]);
        let psi_arr = psi.as_array().to_owned();

        let result = bout_interface::generate_bout_grid(
            &psi_arr,
            grid_2d.r.as_slice().unwrap(),
            grid_2d.z.as_slice().unwrap(),
            psi_axis,
            psi_boundary,
            b_toroidal,
            &self.config,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let dict = PyDict::new(py);
        dict.set_item("psi_n", result.psi_n.into_pyarray(py))?;
        dict.set_item("g_xx", result.g_xx.into_pyarray(py))?;
        dict.set_item("g_yy", result.g_yy.into_pyarray(py))?;
        dict.set_item("g_zz", result.g_zz.into_pyarray(py))?;
        dict.set_item("jacobian", result.jacobian.into_pyarray(py))?;
        dict.set_item("b_mag", result.b_mag.into_pyarray(py))?;
        Ok(dict)
    }
}

#[pyclass]
struct PySPIMitigation {
    inner: SPIMitigation,
}

#[pymethods]
impl PySPIMitigation {
    #[new]
    #[pyo3(signature = (w_th_mj=300.0, ip_ma=15.0, te_kev=20.0))]
    fn new(w_th_mj: f64, ip_ma: f64, te_kev: f64) -> PyResult<Self> {
        let inner = SPIMitigation::new(w_th_mj, ip_ma, te_kev)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PySPIMitigation { inner })
    }

    fn run<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let history = self
            .inner
            .run()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let list = PyList::empty(py);
        for snap in history {
            let dict = PyDict::new(py);
            dict.set_item("time", snap.time)?;
            dict.set_item("w_th_mj", snap.w_th_mj)?;
            dict.set_item("ip_ma", snap.ip_ma)?;
            dict.set_item("te_kev", snap.te_kev)?;
            list.append(dict)?;
        }
        Ok(list)
    }
}

#[pyfunction]
fn boris_push_step(
    mut particle: ChargedParticle,
    b_field: (f64, f64, f64),
    e_field: (f64, f64, f64),
    dt: f64,
) -> PyResult<ChargedParticle> {
    let b = [b_field.0, b_field.1, b_field.2];
    let e = [e_field.0, e_field.1, e_field.2];
    particles::boris_push_step(&mut particle, b, e, dt)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(particle)
}

#[pyfunction]
fn deposit_toroidal_current<'py>(
    py: Python<'py>,
    particles: Vec<ChargedParticle>,
    r: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rs = r.as_slice()?;
    let zs = z.as_slice()?;
    let nr = r.shape()[0];
    let nz = z.shape()[0];
    let grid = Grid2D::new(nr, nz, rs[0], rs[nr - 1], zs[0], zs[nz - 1]);
    let j_phi = particles::deposit_toroidal_current_density(&particles, &grid)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(j_phi.into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn kuramoto_step<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'py, f64>,
    omega: PyReadonlyArray1<'py, f64>,
    dt: f64,
    k: f64,
    alpha: f64,
    zeta: f64,
    psi_external: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let th = theta.as_slice()?;
    let om = omega.as_slice()?;
    let res = kuramoto::kuramoto_sakaguchi_step(th, om, dt, k, alpha, zeta, psi_external);
    let dict = PyDict::new(py);
    dict.set_item("theta", res.theta.into_pyarray(py))?;
    dict.set_item("r", res.r)?;
    dict.set_item("psi_r", res.psi_r)?;
    dict.set_item("psi_global", res.psi_global)?;
    Ok(dict)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn kuramoto_run<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'py, f64>,
    omega: PyReadonlyArray1<'py, f64>,
    n_steps: usize,
    dt: f64,
    k: f64,
    alpha: f64,
    zeta: f64,
    psi_external: Option<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Vec<f64>)> {
    let th = theta.as_slice()?;
    let om = omega.as_slice()?;
    let (final_th, r_hist) =
        kuramoto::kuramoto_sakaguchi_run(th, om, n_steps, dt, k, alpha, zeta, psi_external);
    Ok((final_th.into_pyarray(py), r_hist))
}

#[pyfunction]
#[pyo3(signature = (theta, omega, n_steps, dt, k, alpha=0.0, zeta=0.0, psi_external=None))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn kuramoto_run_lyapunov<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'py, f64>,
    omega: PyReadonlyArray1<'py, f64>,
    n_steps: usize,
    dt: f64,
    k: f64,
    alpha: f64,
    zeta: f64,
    psi_external: Option<f64>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    f64,
)> {
    let th = theta.as_slice()?;
    let om = omega.as_slice()?;
    let (final_th, r_hist, v_hist, lyap) =
        kuramoto::kuramoto_run_lyapunov(th, om, n_steps, dt, k, alpha, zeta, psi_external);
    Ok((
        final_th.into_pyarray(py),
        Array1::from_vec(r_hist).into_pyarray(py),
        Array1::from_vec(v_hist).into_pyarray(py),
        lyap,
    ))
}

#[pyfunction]
fn py_multigrid_solve<'py>(
    py: Python<'py>,
    psi: PyReadonlyArray2<'py, f64>,
    source: PyReadonlyArray2<'py, f64>,
    r: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rs = r.as_slice()?;
    let zs = z.as_slice()?;
    let nr = r.shape()[0];
    let nz = z.shape()[0];
    let grid = Grid2D::new(nr, nz, rs[0], rs[nr - 1], zs[0], zs[nz - 1]);
    let mut psi_out = psi.as_array().to_owned();
    let config = MultigridConfig::default();
    multigrid_solve(
        &mut psi_out,
        &source.as_array().to_owned(),
        &grid,
        &config,
        100,
        1e-8,
    );
    Ok(psi_out.into_pyarray(py))
}

#[pyclass]
struct PyUpdeTick {
    #[pyo3(get)]
    theta_flat: Py<PyArray1<f64>>,
    #[pyo3(get)]
    r_layer: Py<PyArray1<f64>>,
    #[pyo3(get)]
    v_layer: Py<PyArray1<f64>>,
    #[pyo3(get)]
    r_global: f64,
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn upde_tick<'py>(
    py: Python<'py>,
    theta_flat: PyReadonlyArray1<'py, f64>,
    omega_flat: PyReadonlyArray1<'py, f64>,
    knm_flat: PyReadonlyArray1<'py, f64>,
    alpha_flat: PyReadonlyArray1<'py, f64>,
    zeta: PyReadonlyArray1<'py, f64>,
    n_layers: usize,
    n_per: usize,
    dt: f64,
    psi_driver: f64,
    pac_gamma: f64,
) -> PyResult<PyUpdeTick> {
    let th = theta_flat.as_slice()?;
    let om = omega_flat.as_slice()?;
    let knm = knm_flat.as_slice()?;
    let alph = alpha_flat.as_slice()?;
    let z = zeta.as_slice()?;

    let res = kuramoto::upde_tick(
        th, om, knm, alph, z, n_layers, n_per, dt, psi_driver, pac_gamma,
    );

    Ok(PyUpdeTick {
        theta_flat: Array1::from_vec(res.theta_flat).into_pyarray(py).unbind(),
        r_layer: Array1::from_vec(res.r_layer).into_pyarray(py).unbind(),
        v_layer: Array1::from_vec(res.v_layer).into_pyarray(py).unbind(),
        r_global: res.r_global,
    })
}

#[pyfunction]
fn scpn_sample_firing<'py>(
    py: Python<'py>,
    p_fire: PyReadonlyArray1<'py, f64>,
    n_passes: usize,
    seed: u64,
    antithetic: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let p = p_fire.as_array();
    let n = p.len();
    let mut counts = Vec::with_capacity(n);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    if antithetic && n_passes >= 2 {
        #[allow(clippy::manual_div_ceil)]
        let n_pairs = (n_passes + 1) / 2;
        let mut c = vec![0usize; n];
        for pair_idx in 0..n_pairs {
            for i in 0..n {
                let u: f64 = rng.random();
                if u < p[i] {
                    c[i] += 1;
                }
                // Antithetic sample
                #[allow(clippy::manual_is_multiple_of)]
                if (n_passes % 2 == 0 || pair_idx < n_pairs - 1) && (1.0 - u) < p[i] {
                    c[i] += 1;
                }
            }
        }
        for val in c {
            counts.push(val as f64 / n_passes as f64);
        }
    } else {
        for i in 0..n {
            let mut c = 0usize;
            for _ in 0..n_passes {
                if rng.random::<f64>() < p[i] {
                    c += 1;
                }
            }
            counts.push(c as f64 / n_passes as f64);
        }
    }

    Ok(Array1::from_vec(counts).into_pyarray(py))
}

// ─── Module registration ───

#[pymodule]
fn scpn_control_rs<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<PyFusionKernel>()?;
    m.add_class::<PyEquilibriumResult>()?;
    m.add_class::<PyUpdeTick>()?;
    m.add_function(wrap_pyfunction!(upde_tick, m)?)?;
    m.add_function(wrap_pyfunction!(scpn_dense_activations, m)?)?;
    m.add_function(wrap_pyfunction!(scpn_marking_update, m)?)?;
    m.add_function(wrap_pyfunction!(scpn_sample_firing, m)?)?;
    m.add_function(wrap_pyfunction!(shafranov_bv, m)?)?;
    m.add_function(wrap_pyfunction!(solve_coil_currents, m)?)?;
    m.add_class::<PySnnPool>()?;
    m.add_class::<PySnnController>()?;
    m.add_class::<PySpikingControllerPool>()?;
    m.add_class::<PyMpcController>()?;
    m.add_class::<PyPlasma2D>()?;
    m.add_class::<PyTransportSolver>()?;
    m.add_class::<PyHInfController>()?;
    m.add_function(wrap_pyfunction!(py_thomas_solve, m)?)?;
    m.add_class::<PyAmrSolver>()?;
    m.add_class::<PyVmecSolver>()?;
    m.add_class::<PyBoutInterface>()?;
    m.add_class::<ChargedParticle>()?;
    m.add_function(wrap_pyfunction!(boris_push_step, m)?)?;
    m.add_function(wrap_pyfunction!(deposit_toroidal_current, m)?)?;
    m.add_class::<PySPIMitigation>()?;
    m.add_function(wrap_pyfunction!(kuramoto_step, m)?)?;
    m.add_function(wrap_pyfunction!(kuramoto_run, m)?)?;
    m.add_function(wrap_pyfunction!(kuramoto_run_lyapunov, m)?)?;
    m.add_function(wrap_pyfunction!(py_multigrid_solve, m)?)?;
    m.add_class::<PyUdpTransportBridge>()?;
    Ok(())
}

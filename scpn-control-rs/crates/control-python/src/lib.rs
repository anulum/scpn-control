// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — PyO3 bindings for Rust kernels.
//! PyO3 Python bindings for SCPN Control (Modern Bound API).

use ndarray::{Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::hint::spin_loop;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use formal_worker::{
    NativeFormalConfig, NativeFormalMode, NativeFormalReport, NativeFormalRuntime, PetriNetSnapshot,
};
use spike_buffer::register_spike_buffer;
use transport_bridge::PyUdpTransportBridge;

use control_control::analytic;
use control_control::capacitor_bank::{
    free_response as control_capacitor_bank_free_response,
    CapacitorBank as ControlCapacitorBankModel, CapacitorBankSpec as ControlCapacitorBankSpecModel,
    CapacitorBankState as ControlCapacitorBankStateModel,
    EnergyReport as ControlCapacitorBankEnergyReport, PulseSpec as ControlCapacitorPulseSpec,
    PulseWaveform as ControlCapacitorPulseWaveform,
};
use control_control::digital_twin::Plasma2D;
use control_control::h_infinity::HInfController;
use control_control::mpc::{MPController, NeuralSurrogate};
use control_control::multi_shot_campaign::{
    CampaignShotPlan as ControlCampaignShotPlan, CampaignShotSample as ControlCampaignShotSample,
    MultiShotCampaignOrchestrator as ControlMultiShotCampaignOrchestrator,
    MultiShotCampaignReport as ControlMultiShotCampaignReport,
};
use control_control::pulsed_scenario::{
    CapacitorBankTelemetry as ControlCapacitorBankTelemetry,
    PulsedPlasmaTelemetry as ControlPulsedPlasmaTelemetry,
    PulsedScenarioScheduler as ControlPulsedScenarioScheduler,
    PulsedScenarioSpec as ControlPulsedScenarioSpec,
    PulsedScenarioState as ControlPulsedScenarioState,
};
use control_control::snn::{NeuroCyberneticController, SpikingControllerPool};
use control_control::spi::SPIMitigation;
use control_core::amr_kernel::{AmrKernelConfig, AmrKernelSolver};
use control_core::bfield;
use control_core::bout_interface::{self, BoutGridConfig};
use control_core::ignition;
use control_core::kernel::FusionKernel;
use control_core::particles::{self, ChargedParticle};
use control_core::transport::TransportSolver;

const MAX_SPIN_TICK_INTERVAL_S: f64 = 0.01;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativePacingMode {
    Sleep,
    Spin,
}

impl NativePacingMode {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "sleep" | "yield" | "scheduler" => Some(Self::Sleep),
            "spin" | "spin_loop" | "busy_wait" | "busywait" => Some(Self::Spin),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Sleep => "sleep",
            Self::Spin => "spin",
        }
    }
}

#[pyfunction]
#[pyo3(signature = (core_snn, core_z3, core_net, core_hb, tick_interval_s, pacing_mode))]
fn runtime_admission_snapshot<'py>(
    py: Python<'py>,
    core_snn: usize,
    core_z3: usize,
    core_net: usize,
    core_hb: usize,
    tick_interval_s: f64,
    pacing_mode: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let pacing = NativePacingMode::parse(pacing_mode)
        .ok_or_else(|| PyValueError::new_err("pacing_mode must be sleep or spin"))?;
    let requested = vec![core_snn, core_z3, core_net, core_hb];
    let mut sorted = requested.clone();
    sorted.sort_unstable();
    sorted.dedup();

    let kernel_release = std::fs::read_to_string("/proc/sys/kernel/osrelease")
        .map(|value| value.trim().to_string())
        .unwrap_or_else(|_| String::from("unknown"));
    let realtime_sysfs = std::fs::read_to_string("/sys/kernel/realtime")
        .ok()
        .map(|value| value.trim() == "1");
    let release_lc = kernel_release.to_ascii_lowercase();
    let preempt_rt = realtime_sysfs.unwrap_or(false)
        || release_lc.contains("preempt_rt")
        || release_lc.contains("-rt")
        || release_lc.contains("realtime");
    let available_parallelism = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    let max_requested_core = requested.iter().copied().max().unwrap_or(0);

    let dict = PyDict::new(py);
    dict.set_item("schema_version", "scpn-control.runtime-admission-native.v1")?;
    dict.set_item("kernel_release", kernel_release)?;
    dict.set_item("realtime_sysfs", realtime_sysfs)?;
    dict.set_item("preempt_rt", preempt_rt)?;
    dict.set_item("available_parallelism", available_parallelism)?;
    dict.set_item("requested_cores", requested)?;
    dict.set_item("requested_cores_distinct", sorted.len() == 4)?;
    dict.set_item(
        "requested_cores_in_range",
        max_requested_core < available_parallelism,
    )?;
    dict.set_item("tick_interval_s", tick_interval_s)?;
    dict.set_item("pacing_mode", pacing.label())?;
    dict.set_item(
        "spin_tick_admissible",
        pacing != NativePacingMode::Spin || tick_interval_s <= MAX_SPIN_TICK_INTERVAL_S,
    )?;
    Ok(dict)
}
use control_core::vmec_interface::{self, VmecBoundaryState, VmecFourierMode, VmecSolverConfig};
use control_core::xpoint;
use control_math::kuramoto;
use control_math::multigrid::{multigrid_solve, MultigridConfig};
use control_math::tridiag;
use control_types::state::Grid2D;

mod formal_worker;
mod slab;
mod spike_buffer;
mod transport_bridge;

// ─── Equilibrium solver ───

#[pyclass]
struct PyFusionKernel {
    inner: FusionKernel,
}

#[pyclass(from_py_object)]
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

type PySchedulerCommand = (f64, String, String, String, bool, f64);
type PyTransitionRecord = (f64, String, String, String);

#[pyclass(name = "PyPulsedScenarioSpec", from_py_object)]
#[derive(Clone, Copy)]
struct PyPulsedScenarioSpec {
    inner: ControlPulsedScenarioSpec,
}

#[pymethods]
impl PyPulsedScenarioSpec {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        min_precharge_energy_j: f64,
        ramp_current_a: f64,
        phase_tolerance_rad: f64,
        spatial_tolerance_m: f64,
        burn_temperature_ev: f64,
        min_fusion_power_w: f64,
        expansion_velocity_m_s: f64,
        dump_energy_floor_j: f64,
        recharge_voltage_fraction: f64,
        cooldown_temperature_ev: f64,
        cooldown_current_a: f64,
        min_burn_duration_s: f64,
    ) -> PyResult<Self> {
        ControlPulsedScenarioSpec::new(
            min_precharge_energy_j,
            ramp_current_a,
            phase_tolerance_rad,
            spatial_tolerance_m,
            burn_temperature_ev,
            min_fusion_power_w,
            expansion_velocity_m_s,
            dump_energy_floor_j,
            recharge_voltage_fraction,
            cooldown_temperature_ev,
            cooldown_current_a,
            min_burn_duration_s,
        )
        .map(|inner| Self { inner })
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass(name = "PyPulsedPlasmaTelemetry", from_py_object)]
#[derive(Clone, Copy)]
struct PyPulsedPlasmaTelemetry {
    inner: ControlPulsedPlasmaTelemetry,
}

#[pymethods]
impl PyPulsedPlasmaTelemetry {
    #[new]
    fn new(
        coil_current_a: f64,
        temperature_ev: f64,
        phase_lock_error_rad: f64,
        reference_error_m: f64,
        fusion_power_w: f64,
        radial_velocity_m_s: f64,
    ) -> PyResult<Self> {
        ControlPulsedPlasmaTelemetry::new(
            coil_current_a,
            temperature_ev,
            phase_lock_error_rad,
            reference_error_m,
            fusion_power_w,
            radial_velocity_m_s,
        )
        .map(|inner| Self { inner })
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass(name = "PyCapacitorBankTelemetry", from_py_object)]
#[derive(Clone, Copy)]
struct PyCapacitorBankTelemetry {
    inner: ControlCapacitorBankTelemetry,
}

#[pymethods]
impl PyCapacitorBankTelemetry {
    #[new]
    fn new(voltage_v: f64, voltage_max_v: f64, energy_j: f64) -> PyResult<Self> {
        ControlCapacitorBankTelemetry::new(voltage_v, voltage_max_v, energy_j)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn voltage_fraction(&self) -> f64 {
        self.inner.voltage_fraction()
    }
}

#[pyclass(name = "PyPulsedScenarioScheduler")]
struct PyPulsedScenarioScheduler {
    inner: ControlPulsedScenarioScheduler,
}

#[pymethods]
impl PyPulsedScenarioScheduler {
    #[new]
    fn new(spec: &PyPulsedScenarioSpec) -> Self {
        Self {
            inner: ControlPulsedScenarioScheduler::new(spec.inner),
        }
    }

    #[getter]
    fn state(&self) -> String {
        self.inner.state.as_str().to_string()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn step(
        &mut self,
        t_s: f64,
        plasma: &PyPulsedPlasmaTelemetry,
        bank: &PyCapacitorBankTelemetry,
    ) -> PyResult<PySchedulerCommand> {
        let command = self
            .inner
            .step(t_s, plasma.inner, bank.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((
            command.t_s,
            command.state.as_str().to_string(),
            command.action.as_str().to_string(),
            command.reason,
            command.transition,
            command.dwell_s,
        ))
    }

    fn transition_to(
        &mut self,
        next_state: &str,
        t_s: f64,
        reason: &str,
    ) -> PyResult<PyTransitionRecord> {
        let state = parse_pulsed_scenario_state(next_state)?;
        let record = self
            .inner
            .transition_to(state, t_s, reason)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((
            record.t_s,
            record.from_state.as_str().to_string(),
            record.to_state.as_str().to_string(),
            record.reason,
        ))
    }

    fn audit_log(&self) -> Vec<PyTransitionRecord> {
        self.inner
            .audit_log()
            .iter()
            .map(|record| {
                (
                    record.t_s,
                    record.from_state.as_str().to_string(),
                    record.to_state.as_str().to_string(),
                    record.reason.clone(),
                )
            })
            .collect()
    }
}

#[pyclass(name = "PyCapacitorBankSpec", from_py_object)]
#[derive(Clone, Copy)]
struct PyCapacitorBankSpecModel {
    inner: ControlCapacitorBankSpecModel,
}

#[pymethods]
impl PyCapacitorBankSpecModel {
    #[new]
    #[pyo3(signature = (
        capacitance_f,
        inductance_h,
        series_resistance_ohm,
        voltage_max_v,
        recharge_power_kw=0.0
    ))]
    fn new(
        capacitance_f: f64,
        inductance_h: f64,
        series_resistance_ohm: f64,
        voltage_max_v: f64,
        recharge_power_kw: f64,
    ) -> PyResult<Self> {
        ControlCapacitorBankSpecModel::new(
            capacitance_f,
            inductance_h,
            series_resistance_ohm,
            voltage_max_v,
            recharge_power_kw,
        )
        .map(|inner| Self { inner })
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn capacitance_f(&self) -> f64 {
        self.inner.capacitance_f
    }

    #[getter]
    fn inductance_h(&self) -> f64 {
        self.inner.inductance_h
    }

    #[getter]
    fn series_resistance_ohm(&self) -> f64 {
        self.inner.series_resistance_ohm
    }

    #[getter]
    fn voltage_max_v(&self) -> f64 {
        self.inner.voltage_max_v
    }

    #[getter]
    fn recharge_power_kw(&self) -> f64 {
        self.inner.recharge_power_kw
    }

    #[getter]
    fn regime(&self) -> String {
        self.inner.regime().as_str().to_string()
    }

    #[getter]
    fn natural_impedance_ohm(&self) -> f64 {
        self.inner.natural_impedance_ohm()
    }

    #[getter]
    fn damping_ratio(&self) -> f64 {
        self.inner.damping_ratio()
    }
}

#[pyclass(name = "PyCapacitorBankModel")]
struct PyCapacitorBankModel {
    inner: ControlCapacitorBankModel,
}

#[pymethods]
impl PyCapacitorBankModel {
    #[new]
    #[pyo3(signature = (
        capacitance_f,
        inductance_h,
        series_resistance_ohm,
        voltage_max_v,
        recharge_power_kw=0.0,
        initial_voltage_v=0.0,
        initial_current_a=0.0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        capacitance_f: f64,
        inductance_h: f64,
        series_resistance_ohm: f64,
        voltage_max_v: f64,
        recharge_power_kw: f64,
        initial_voltage_v: f64,
        initial_current_a: f64,
    ) -> PyResult<Self> {
        let spec = ControlCapacitorBankSpecModel::new(
            capacitance_f,
            inductance_h,
            series_resistance_ohm,
            voltage_max_v,
            recharge_power_kw,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        ControlCapacitorBankModel::new(spec, initial_voltage_v, initial_current_a)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        capacitor_state_to_dict(py, self.inner.state())
    }

    fn telemetry<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (voltage_v, voltage_max_v, energy_j) = self
            .inner
            .telemetry_fields()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dict = PyDict::new(py);
        dict.set_item("voltage_V", voltage_v)?;
        dict.set_item("voltage_max_V", voltage_max_v)?;
        dict.set_item("energy_J", energy_j)?;
        dict.set_item("voltage_fraction", voltage_v / voltage_max_v)?;
        Ok(dict)
    }

    #[pyo3(signature = (initial_voltage_v=0.0, initial_current_a=0.0))]
    fn reset<'py>(
        &mut self,
        py: Python<'py>,
        initial_voltage_v: f64,
        initial_current_a: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let state = self
            .inner
            .reset(initial_voltage_v, initial_current_a)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        capacitor_state_to_dict(py, state)
    }

    #[pyo3(signature = (dt, external_load_current_a=0.0))]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        dt: f64,
        external_load_current_a: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let state = self
            .inner
            .step(dt, external_load_current_a)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        capacitor_state_to_dict(py, state)
    }

    #[pyo3(signature = (peak_current_a, duration_s, waveform="half_sine", dt=1.0e-6, n_steps=1))]
    fn discharge<'py>(
        &mut self,
        py: Python<'py>,
        peak_current_a: f64,
        duration_s: f64,
        waveform: &str,
        dt: f64,
        n_steps: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let pulse = ControlCapacitorPulseSpec::new(
            peak_current_a,
            duration_s,
            parse_capacitor_waveform(waveform)?,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let report = self
            .inner
            .discharge(pulse, dt, n_steps)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        capacitor_energy_report_to_dict(py, report)
    }

    #[pyo3(signature = (peak_current_a, duration_s, waveform="half_sine"))]
    fn feasibility(
        &self,
        peak_current_a: f64,
        duration_s: f64,
        waveform: &str,
    ) -> PyResult<(bool, String)> {
        let pulse = ControlCapacitorPulseSpec::new(
            peak_current_a,
            duration_s,
            parse_capacitor_waveform(waveform)?,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner
            .feasibility(pulse)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn recharge_status<'py>(&self, py: Python<'py>, t: f64) -> PyResult<Bound<'py, PyDict>> {
        let status = self
            .inner
            .recharge_status(t)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dict = PyDict::new(py);
        dict.set_item("target_voltage_V", status.target_voltage_v)?;
        dict.set_item("projected_voltage_V", status.projected_voltage_v)?;
        dict.set_item("time_to_full_s", status.time_to_full_s)?;
        Ok(dict)
    }
}

#[pyfunction]
#[pyo3(signature = (
    capacitance_f,
    inductance_h,
    series_resistance_ohm,
    voltage_max_v,
    recharge_power_kw,
    v0,
    i0,
    t
))]
#[allow(clippy::too_many_arguments)]
fn capacitor_bank_free_response<'py>(
    py: Python<'py>,
    capacitance_f: f64,
    inductance_h: f64,
    series_resistance_ohm: f64,
    voltage_max_v: f64,
    recharge_power_kw: f64,
    v0: f64,
    i0: f64,
    t: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let spec = ControlCapacitorBankSpecModel::new(
        capacitance_f,
        inductance_h,
        series_resistance_ohm,
        voltage_max_v,
        recharge_power_kw,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let state = control_capacitor_bank_free_response(spec, v0, i0, t)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    capacitor_state_to_dict(py, state)
}

fn parse_capacitor_waveform(value: &str) -> PyResult<ControlCapacitorPulseWaveform> {
    ControlCapacitorPulseWaveform::parse(value).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn capacitor_state_to_dict<'py>(
    py: Python<'py>,
    state: ControlCapacitorBankStateModel,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("t", state.t)?;
    dict.set_item("voltage_V", state.voltage_v)?;
    dict.set_item("current_A", state.current_a)?;
    dict.set_item("di_dt_A_s", state.di_dt_a_s)?;
    dict.set_item("capacitance_F", state.capacitance_f)?;
    dict.set_item("energy_J", state.energy_j())?;
    Ok(dict)
}

fn capacitor_energy_report_to_dict<'py>(
    py: Python<'py>,
    report: ControlCapacitorBankEnergyReport,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("energy_delivered_J", report.energy_delivered_j)?;
    dict.set_item("energy_initial_J", report.energy_initial_j)?;
    dict.set_item("energy_remaining_J", report.energy_remaining_j)?;
    dict.set_item(
        "capacitor_energy_remaining_J",
        report.capacitor_energy_remaining_j,
    )?;
    dict.set_item(
        "inductor_energy_remaining_J",
        report.inductor_energy_remaining_j,
    )?;
    dict.set_item("resistive_loss_J", report.resistive_loss_j)?;
    dict.set_item("load_energy_J", report.load_energy_j)?;
    dict.set_item(
        "energy_balance_residual_J",
        report.energy_balance_residual_j,
    )?;
    dict.set_item(
        "energy_balance_relative_error",
        report.energy_balance_relative_error,
    )?;
    dict.set_item("energy_balance_passed", report.energy_balance_passed)?;
    dict.set_item("peak_voltage_V", report.peak_voltage_v)?;
    dict.set_item("peak_current_A", report.peak_current_a)?;
    dict.set_item("discharge_duration_s", report.discharge_duration_s)?;
    dict.set_item("rlc_regime", report.rlc_regime.as_str())?;
    Ok(dict)
}

fn parse_pulsed_scenario_state(value: &str) -> PyResult<ControlPulsedScenarioState> {
    match value.trim() {
        "idle" => Ok(ControlPulsedScenarioState::Idle),
        "ramp_up" => Ok(ControlPulsedScenarioState::RampUp),
        "flat_top" => Ok(ControlPulsedScenarioState::FlatTop),
        "burn" => Ok(ControlPulsedScenarioState::Burn),
        "expansion" => Ok(ControlPulsedScenarioState::Expansion),
        "dump" => Ok(ControlPulsedScenarioState::Dump),
        "recharge" => Ok(ControlPulsedScenarioState::Recharge),
        "cool_down" => Ok(ControlPulsedScenarioState::CoolDown),
        _ => Err(PyValueError::new_err("unknown pulsed scenario state")),
    }
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
    pacing_mode: NativePacingMode,
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
    formal_enabled: bool,
    formal_mode: NativeFormalMode,
    formal_max_marking: i64,
    formal_max_depth: usize,
    formal_dispatch_interval_steps: usize,
    formal_channel_capacity: usize,
    formal_last_report: NativeFormalReport,
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
            pacing_mode: NativePacingMode::Sleep,
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
            formal_enabled: true,
            formal_mode: NativeFormalMode::AsyncDrop,
            formal_max_marking: 100,
            formal_max_depth: 4,
            formal_dispatch_interval_steps: 30,
            formal_channel_capacity: 2,
            formal_last_report: NativeFormalReport::disabled(),
            is_running: false,
            stop_flag: Arc::new(AtomicBool::new(false)),
        })
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "PyO3 preserves stable Python keyword API"
    )]
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

    #[expect(
        clippy::too_many_arguments,
        reason = "PyO3 preserves stable Python keyword API"
    )]
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

    #[pyo3(signature = (max_iterations=None, tick_interval=0.001, initial_state=None, plant_gain=0.0, pacing_mode="sleep"))]
    fn configure_runtime_budget(
        &mut self,
        max_iterations: Option<usize>,
        tick_interval: f64,
        initial_state: Option<Vec<f64>>,
        plant_gain: f64,
        pacing_mode: &str,
    ) -> PyResult<()> {
        if !tick_interval.is_finite() || tick_interval < 0.0 {
            return Err(PyValueError::new_err(
                "tick_interval must be finite and >= 0",
            ));
        }
        let pacing = NativePacingMode::parse(pacing_mode)
            .ok_or_else(|| PyValueError::new_err("pacing_mode must be sleep or spin"))?;
        if pacing == NativePacingMode::Spin && tick_interval > MAX_SPIN_TICK_INTERVAL_S {
            return Err(PyValueError::new_err(
                "spin pacing requires tick_interval <= 0.01 seconds",
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
        self.pacing_mode = pacing;
        self.plant_gain = plant_gain;
        Ok(())
    }

    #[pyo3(signature = (enabled=true, max_marking=100, max_depth=4, dispatch_interval_steps=30, channel_capacity=2, mode="async_drop"))]
    fn configure_native_formal_verification(
        &mut self,
        enabled: bool,
        max_marking: i64,
        max_depth: usize,
        dispatch_interval_steps: usize,
        channel_capacity: usize,
        mode: &str,
    ) -> PyResult<()> {
        if max_marking <= 0 {
            return Err(PyValueError::new_err("max_marking must be positive"));
        }
        if !(1..=64).contains(&max_depth) {
            return Err(PyValueError::new_err("max_depth must be in [1, 64]"));
        }
        if dispatch_interval_steps == 0 {
            return Err(PyValueError::new_err(
                "dispatch_interval_steps must be positive",
            ));
        }
        if !(1..=1024).contains(&channel_capacity) {
            return Err(PyValueError::new_err(
                "channel_capacity must be in [1, 1024]",
            ));
        }
        let formal_mode = NativeFormalMode::parse(mode).ok_or_else(|| {
            PyValueError::new_err(
                "native formal verification mode must be async_drop, sync_stride, or aot_certificate",
            )
        })?;

        self.formal_enabled = enabled;
        self.formal_mode = formal_mode;
        self.formal_max_marking = max_marking;
        self.formal_max_depth = max_depth;
        self.formal_dispatch_interval_steps = dispatch_interval_steps;
        self.formal_channel_capacity = channel_capacity;
        self.formal_last_report = if enabled {
            NativeFormalReport {
                ..NativeFormalReport::default()
            }
        } else {
            NativeFormalReport::disabled()
        };
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
        self.formal_last_report = if self.formal_enabled {
            NativeFormalReport {
                ..NativeFormalReport::default()
            }
        } else {
            NativeFormalReport::disabled()
        };
        self.is_running = true;

        let result = py.detach(|| self.run_native_loop(max_iterations));
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
        let formal_config = NativeFormalConfig {
            mode: self.formal_mode,
            max_marking: self.formal_max_marking,
            max_depth: self.formal_max_depth,
            channel_capacity: self.formal_channel_capacity,
            core_z3: self.core_z3,
        };
        let mut formal_runtime = if self.formal_enabled {
            Some(
                NativeFormalRuntime::spawn(formal_config.clone(), Arc::clone(&self.stop_flag))
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )
        } else {
            None
        };

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
            if let Some(runtime) = formal_runtime.as_ref() {
                if runtime.violation_detected() {
                    break Err(PyRuntimeError::new_err(
                        "native formal verification contract violation",
                    ));
                }
            }

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

            if let Some(runtime) = formal_runtime.as_ref() {
                let next_step = self.steps_executed.saturating_add(1);
                if next_step.is_multiple_of(self.formal_dispatch_interval_steps) {
                    let snapshot = PetriNetSnapshot::from_control_outputs(
                        next_step as u64,
                        error_r,
                        error_z,
                        r_command,
                        z_command,
                    );
                    match self.formal_mode {
                        NativeFormalMode::AsyncDrop => {
                            let _ = runtime.try_submit_async(snapshot);
                        }
                        NativeFormalMode::SyncStride => {
                            if !runtime.submit_sync(snapshot) {
                                break Err(PyRuntimeError::new_err(
                                    "native formal verification contract violation",
                                ));
                            }
                        }
                        NativeFormalMode::AotCertificate => {
                            if !runtime.check_aot_certificate(snapshot) {
                                break Err(PyRuntimeError::new_err(
                                    "native formal verification contract violation",
                                ));
                            }
                        }
                    }
                }
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
                    match self.pacing_mode {
                        NativePacingMode::Sleep => {
                            std::thread::sleep(Duration::from_nanos(
                                (target_ns - self.last_cycle_ns) as u64,
                            ));
                        }
                        NativePacingMode::Spin => {
                            while step_start.elapsed().as_nanos() < target_ns {
                                spin_loop();
                            }
                        }
                    }
                }
            }
        };

        let formal_report = if let Some(runtime) = formal_runtime.take() {
            runtime.shutdown()
        } else {
            NativeFormalReport::disabled()
        };
        self.formal_last_report = formal_report;

        let stop_result = bridge.stop();
        if loop_result.is_ok() {
            stop_result?;
            if self.formal_last_report.failures > 0 {
                return Err(PyRuntimeError::new_err(
                    "native formal verification contract violation",
                ));
            }
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
        execution.set_item("pacing_mode", self.pacing_mode.label())?;
        dict.set_item("execution", execution)?;

        let formal = PyDict::new(py);
        formal.set_item("enabled", self.formal_enabled)?;
        formal.set_item("backend", self.formal_last_report.backend_label())?;
        formal.set_item("mode", self.formal_last_report.mode_label())?;
        formal.set_item("max_marking", self.formal_max_marking)?;
        formal.set_item("max_depth", self.formal_max_depth)?;
        formal.set_item(
            "dispatch_interval_steps",
            self.formal_dispatch_interval_steps,
        )?;
        formal.set_item("channel_capacity", self.formal_channel_capacity)?;
        formal.set_item("core_z3", self.core_z3)?;
        formal.set_item("generated", self.formal_last_report.generated)?;
        formal.set_item("submitted", self.formal_last_report.submitted)?;
        formal.set_item("dropped", self.formal_last_report.dropped)?;
        formal.set_item("checked", self.formal_last_report.checked)?;
        formal.set_item("failures", self.formal_last_report.failures)?;
        formal.set_item(
            "effective_verification_rate",
            self.formal_last_report.effective_verification_rate(),
        )?;
        formal.set_item(
            "last_checked_step",
            self.formal_last_report.last_checked_step,
        )?;
        formal.set_item("last_failed_step", self.formal_last_report.last_failed_step)?;
        formal.set_item("last_status", self.formal_last_report.status_label())?;
        formal.set_item("sync_wait_count", self.formal_last_report.sync_wait_count)?;
        formal.set_item(
            "sync_wait_total_ns",
            self.formal_last_report.sync_wait_total_ns,
        )?;
        formal.set_item("sync_wait_min_ns", self.formal_last_report.sync_wait_min_ns)?;
        formal.set_item("sync_wait_p50_ns", self.formal_last_report.sync_wait_p50_ns)?;
        formal.set_item("sync_wait_p95_ns", self.formal_last_report.sync_wait_p95_ns)?;
        formal.set_item("sync_wait_p99_ns", self.formal_last_report.sync_wait_p99_ns)?;
        formal.set_item("sync_wait_max_ns", self.formal_last_report.sync_wait_max_ns)?;
        formal.set_item(
            "certificate_admitted",
            self.formal_last_report.certificate_admitted,
        )?;
        formal.set_item(
            "certificate_schema_version",
            self.formal_last_report.certificate_schema_version.as_str(),
        )?;
        formal.set_item(
            "certificate_id",
            self.formal_last_report.certificate_id.as_str(),
        )?;
        formal.set_item(
            "certificate_contract",
            self.formal_last_report.certificate_contract.as_str(),
        )?;
        formal.set_item(
            "certificate_assumption_sha256",
            self.formal_last_report
                .certificate_assumption_sha256
                .as_str(),
        )?;
        dict.set_item("formal_verification", formal)?;
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

    #[allow(clippy::too_many_arguments)]
    fn plan_pulsed<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, f64>,
        scheduler_state: &str,
        bank_feasible: bool,
        burn_action_mask: PyReadonlyArray1<'py, bool>,
        safe_action: PyReadonlyArray1<'py, f64>,
        constraint_slack: f64,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyDict>)> {
        let s = state.as_array().to_owned();
        let mask: Vec<bool> = burn_action_mask.as_array().iter().copied().collect();
        let safe = safe_action.as_array().to_owned();
        let decision = self
            .inner
            .plan_pulsed(
                &s,
                scheduler_state,
                bank_feasible,
                &mask,
                &safe,
                constraint_slack,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let dict = PyDict::new(py);
        dict.set_item("mpc_objective", decision.mpc_objective)?;
        dict.set_item("constraint_slack", decision.constraint_slack)?;
        dict.set_item("scheduler_state", decision.scheduler_state)?;
        dict.set_item("bank_feasibility", decision.bank_feasibility)?;
        dict.set_item("reason", decision.reason)?;
        dict.set_item("bank_feasible", decision.bank_feasible)?;
        dict.set_item("safe_action_applied", decision.safe_action_applied)?;
        dict.set_item("burn_components_masked", decision.burn_components_masked)?;
        dict.set_item("peak_current_A", decision.peak_current_a)?;
        dict.set_item("evidence_schema_version", decision.evidence_schema_version)?;
        dict.set_item("action_sha256", decision.action_sha256)?;
        dict.set_item("safe_action_sha256", decision.safe_action_sha256)?;
        dict.set_item("burn_action_mask_sha256", decision.burn_action_mask_sha256)?;
        dict.set_item("admission_digest", decision.admission_digest)?;
        Ok((decision.action.into_pyarray(py), dict))
    }
}

#[pyclass(name = "PyMultiShotCampaignOrchestrator")]
struct PyMultiShotCampaignOrchestrator {
    inner: ControlMultiShotCampaignOrchestrator,
}

#[pymethods]
impl PyMultiShotCampaignOrchestrator {
    #[new]
    fn new(
        campaign_id: &str,
        scheduler_spec: &PyPulsedScenarioSpec,
        bank_spec: &PyCapacitorBankSpecModel,
        require_complete_lifecycle: bool,
    ) -> PyResult<Self> {
        let inner = ControlMultiShotCampaignOrchestrator::new(
            campaign_id,
            scheduler_spec.inner,
            bank_spec.inner,
            require_complete_lifecycle,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (
        shot_ids,
        sample_shot_index,
        sample_t_s,
        plasma,
        bank,
        initial_bank_voltage_v,
        pulsed_mpc_admission_digests=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn run_table<'py>(
        &self,
        py: Python<'py>,
        shot_ids: Vec<String>,
        sample_shot_index: PyReadonlyArray1<'py, usize>,
        sample_t_s: PyReadonlyArray1<'py, f64>,
        plasma: PyReadonlyArray2<'py, f64>,
        bank: PyReadonlyArray2<'py, f64>,
        initial_bank_voltage_v: PyReadonlyArray1<'py, f64>,
        pulsed_mpc_admission_digests: Option<Vec<Option<String>>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let indices = sample_shot_index.as_slice()?;
        let times = sample_t_s.as_slice()?;
        let initial_voltages = initial_bank_voltage_v.as_slice()?;
        let plasma_view = plasma.as_array();
        let bank_view = bank.as_array();
        if shot_ids.is_empty() {
            return Err(PyValueError::new_err("shot_ids must not be empty"));
        }
        if initial_voltages.len() != shot_ids.len() {
            return Err(PyValueError::new_err(
                "initial_bank_voltage_v length must equal shot_ids length",
            ));
        }
        if let Some(digests) = &pulsed_mpc_admission_digests {
            if digests.len() != shot_ids.len() {
                return Err(PyValueError::new_err(
                    "pulsed_mpc_admission_digests length must equal shot_ids length",
                ));
            }
        }
        if indices.len() != times.len()
            || plasma_view.nrows() != times.len()
            || bank_view.nrows() != times.len()
            || plasma_view.ncols() != 6
            || bank_view.ncols() != 3
        {
            return Err(PyValueError::new_err(
                "sample arrays must have matching rows; plasma must be N x 6 and bank N x 3",
            ));
        }
        let mut grouped: Vec<Vec<ControlCampaignShotSample>> = vec![Vec::new(); shot_ids.len()];
        for row in 0..times.len() {
            let shot_index = indices[row];
            if shot_index >= shot_ids.len() {
                return Err(PyValueError::new_err(
                    "sample_shot_index entry out of range",
                ));
            }
            let plasma_row = plasma_view.row(row);
            let bank_row = bank_view.row(row);
            let plasma_sample = ControlPulsedPlasmaTelemetry::new(
                plasma_row[0],
                plasma_row[1],
                plasma_row[2],
                plasma_row[3],
                plasma_row[4],
                plasma_row[5],
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let bank_sample =
                ControlCapacitorBankTelemetry::new(bank_row[0], bank_row[1], bank_row[2])
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            grouped[shot_index].push(
                ControlCampaignShotSample::new(times[row], plasma_sample, bank_sample)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            );
        }
        let mut shots = Vec::with_capacity(shot_ids.len());
        for (idx, shot_id) in shot_ids.iter().enumerate() {
            let mut shot = ControlCampaignShotPlan::new(
                shot_id,
                grouped[idx].clone(),
                initial_voltages[idx],
                0.0,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if let Some(digests) = &pulsed_mpc_admission_digests {
                if let Some(digest) = &digests[idx] {
                    shot = shot
                        .with_pulsed_mpc_admission_digest(digest)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                }
            }
            shots.push(shot);
        }
        let report = self
            .inner
            .run(&shots)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        multi_shot_report_to_dict(py, &report)
    }
}

fn multi_shot_report_to_dict<'py>(
    py: Python<'py>,
    report: &ControlMultiShotCampaignReport,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("schema_version", report.schema_version.as_str())?;
    dict.set_item("campaign_id", report.campaign_id.as_str())?;
    dict.set_item("shot_count", report.shot_count)?;
    dict.set_item("passed_count", report.passed_count)?;
    dict.set_item("failed_count", report.failed_count)?;
    dict.set_item(
        "require_complete_lifecycle",
        report.require_complete_lifecycle,
    )?;
    dict.set_item(
        "expected_transition_states",
        report.expected_transition_states.clone(),
    )?;
    dict.set_item(
        "pulsed_mpc_evidence_schema_version",
        report.pulsed_mpc_evidence_schema_version.as_str(),
    )?;
    dict.set_item(
        "pulsed_mpc_admission_digest_count",
        report.pulsed_mpc_admission_digest_count,
    )?;
    let shots = PyList::empty(py);
    for shot in &report.shots {
        let shot_dict = PyDict::new(py);
        shot_dict.set_item("shot_id", shot.shot_id.as_str())?;
        shot_dict.set_item("pulse_id", shot.pulse_id.as_str())?;
        shot_dict.set_item("passed", shot.passed)?;
        shot_dict.set_item("failure_reason", shot.failure_reason.clone())?;
        shot_dict.set_item("terminal_state", shot.terminal_state.as_str())?;
        shot_dict.set_item("transition_states", shot.transition_states.clone())?;
        shot_dict.set_item("capacitor_state_initial_J", shot.capacitor_state_initial_j)?;
        shot_dict.set_item("capacitor_state_final_J", shot.capacitor_state_final_j)?;
        shot_dict.set_item("energy_recovered_J", shot.energy_recovered_j)?;
        shot_dict.set_item("trigger_timestamp_ns", shot.trigger_timestamp_ns)?;
        shot_dict.set_item(
            "pulsed_mpc_evidence_schema_version",
            shot.pulsed_mpc_admission_digest
                .as_ref()
                .map(|_| report.pulsed_mpc_evidence_schema_version.as_str()),
        )?;
        shot_dict.set_item(
            "pulsed_mpc_admission_digest",
            shot.pulsed_mpc_admission_digest.clone(),
        )?;
        shots.append(shot_dict)?;
    }
    dict.set_item("shots", shots)?;
    dict.set_item("payload_sha256", report.payload_sha256.as_str())?;
    Ok(dict)
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

    fn measure_core_temp(&self, measurement_noise: f64) -> PyResult<f64> {
        self.inner
            .measure_core_temp(measurement_noise)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

#[pyclass]
struct PyRealtimeMonitor {
    knm_flat: Vec<f64>,
    alpha_flat: Vec<f64>,
    zeta: Vec<f64>,
    theta_flat: Vec<f64>,
    omega_flat: Vec<f64>,
    n_layers: usize,
    n_per: usize,
    dt: f64,
    psi_driver: f64,
    tick_count: usize,
}

#[pymethods]
impl PyRealtimeMonitor {
    #[new]
    #[pyo3(signature = (knm_flat, zeta_flat, theta_flat, omega_flat, n_layers, n_per, dt=1.0e-3, psi_driver=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        knm_flat: PyReadonlyArray1<'_, f64>,
        zeta_flat: PyReadonlyArray1<'_, f64>,
        theta_flat: PyReadonlyArray1<'_, f64>,
        omega_flat: PyReadonlyArray1<'_, f64>,
        n_layers: usize,
        n_per: usize,
        dt: f64,
        psi_driver: f64,
    ) -> PyResult<Self> {
        if n_layers == 0 || n_per == 0 {
            return Err(PyValueError::new_err("n_layers and n_per must be positive"));
        }
        if !dt.is_finite() || dt <= 0.0 {
            return Err(PyValueError::new_err("dt must be finite and > 0"));
        }
        if !psi_driver.is_finite() {
            return Err(PyValueError::new_err("psi_driver must be finite"));
        }

        let knm = knm_flat
            .as_slice()
            .map_err(|_| PyValueError::new_err("knm_flat must be contiguous"))?
            .to_vec();
        let zeta = zeta_flat
            .as_slice()
            .map_err(|_| PyValueError::new_err("zeta_flat must be contiguous"))?
            .to_vec();
        let theta = theta_flat
            .as_slice()
            .map_err(|_| PyValueError::new_err("theta_flat must be contiguous"))?
            .to_vec();
        let omega = omega_flat
            .as_slice()
            .map_err(|_| PyValueError::new_err("omega_flat must be contiguous"))?
            .to_vec();

        let phase_len = n_layers
            .checked_mul(n_per)
            .ok_or_else(|| PyValueError::new_err("n_layers * n_per overflowed"))?;
        if knm.len() != n_layers * n_layers {
            return Err(PyValueError::new_err(
                "knm_flat length must equal n_layers ** 2",
            ));
        }
        if zeta.len() != n_layers {
            return Err(PyValueError::new_err(
                "zeta_flat length must equal n_layers",
            ));
        }
        if theta.len() != phase_len || omega.len() != phase_len {
            return Err(PyValueError::new_err(
                "theta_flat and omega_flat lengths must equal n_layers * n_per",
            ));
        }
        if !knm.iter().all(|v| v.is_finite())
            || !zeta.iter().all(|v| v.is_finite())
            || !theta.iter().all(|v| v.is_finite())
            || !omega.iter().all(|v| v.is_finite())
        {
            return Err(PyValueError::new_err(
                "monitor arrays must contain only finite values",
            ));
        }

        Ok(Self {
            knm_flat: knm,
            alpha_flat: vec![0.0; n_layers * n_layers],
            zeta,
            theta_flat: theta,
            omega_flat: omega,
            n_layers,
            n_per,
            dt,
            psi_driver,
            tick_count: 0,
        })
    }

    #[getter]
    fn tick_count(&self) -> usize {
        self.tick_count
    }

    fn tick<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = kuramoto::upde_tick(
            &self.theta_flat,
            &self.omega_flat,
            &self.knm_flat,
            &self.alpha_flat,
            &self.zeta,
            self.n_layers,
            self.n_per,
            self.dt,
            self.psi_driver,
            0.0,
        );
        self.theta_flat.clone_from(&result.theta_flat);
        self.tick_count += 1;

        let dict = PyDict::new(py);
        dict.set_item("tick", self.tick_count)?;
        dict.set_item("theta_flat", result.theta_flat.into_pyarray(py))?;
        dict.set_item("R_layer", result.r_layer.into_pyarray(py))?;
        dict.set_item("R_global", result.r_global)?;
        dict.set_item("Psi_global", result.psi_global)?;
        dict.set_item("V_layer", result.v_layer.into_pyarray(py))?;
        dict.set_item("V_global", result.v_global)?;
        Ok(dict)
    }

    fn reset(&mut self, theta_flat: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let theta = theta_flat
            .as_slice()
            .map_err(|_| PyValueError::new_err("theta_flat must be contiguous"))?;
        if theta.len() != self.n_layers * self.n_per {
            return Err(PyValueError::new_err(
                "theta_flat length must equal n_layers * n_per",
            ));
        }
        if !theta.iter().all(|v| v.is_finite()) {
            return Err(PyValueError::new_err(
                "theta_flat must contain only finite values",
            ));
        }
        self.theta_flat.clear();
        self.theta_flat.extend_from_slice(theta);
        self.tick_count = 0;
        Ok(())
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
    ) -> PyResult<Py<PyAny>> {
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
    m.add_function(wrap_pyfunction!(runtime_admission_snapshot, m)?)?;
    m.add_function(wrap_pyfunction!(scpn_dense_activations, m)?)?;
    m.add_function(wrap_pyfunction!(scpn_marking_update, m)?)?;
    m.add_function(wrap_pyfunction!(scpn_sample_firing, m)?)?;
    m.add_function(wrap_pyfunction!(shafranov_bv, m)?)?;
    m.add_function(wrap_pyfunction!(solve_coil_currents, m)?)?;
    m.add_class::<PySnnPool>()?;
    m.add_class::<PySnnController>()?;
    m.add_class::<PyPulsedScenarioSpec>()?;
    m.add_class::<PyPulsedPlasmaTelemetry>()?;
    m.add_class::<PyCapacitorBankTelemetry>()?;
    m.add_class::<PyPulsedScenarioScheduler>()?;
    m.add_class::<PyCapacitorBankSpecModel>()?;
    m.add_class::<PyCapacitorBankModel>()?;
    m.add_function(wrap_pyfunction!(capacitor_bank_free_response, m)?)?;
    m.add_class::<PySpikingControllerPool>()?;
    m.add_class::<PyMpcController>()?;
    m.add_class::<PyMultiShotCampaignOrchestrator>()?;
    m.add_class::<PyPlasma2D>()?;
    m.add_class::<PyRealtimeMonitor>()?;
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
    register_spike_buffer(m)?;
    Ok(())
}

#[cfg(test)]
mod native_pacing_tests {
    #[cfg(not(feature = "extension-module"))]
    use super::runtime_admission_snapshot;
    use super::NativePacingMode;
    #[cfg(not(feature = "extension-module"))]
    use pyo3::types::{PyAnyMethods, PyDictMethods};
    #[cfg(not(feature = "extension-module"))]
    use pyo3::{prepare_freethreaded_python, Python};

    #[test]
    fn pacing_mode_parser_accepts_supported_aliases() {
        assert_eq!(
            NativePacingMode::parse("sleep"),
            Some(NativePacingMode::Sleep)
        );
        assert_eq!(
            NativePacingMode::parse("spin-loop"),
            Some(NativePacingMode::Spin)
        );
        assert_eq!(
            NativePacingMode::parse("busy_wait"),
            Some(NativePacingMode::Spin)
        );
        assert_eq!(NativePacingMode::parse("timer_magic"), None);
    }

    #[cfg(not(feature = "extension-module"))]
    #[test]
    fn runtime_admission_snapshot_binds_kernel_and_core_contract() {
        prepare_freethreaded_python();
        Python::with_gil(|py| {
            let snapshot = runtime_admission_snapshot(py, 4, 5, 6, 7, 0.0001, "spin")
                .expect("runtime admission snapshot should be constructible");

            let schema = snapshot
                .get_item("schema_version")
                .expect("schema lookup should not fail")
                .expect("schema should exist")
                .extract::<String>()
                .expect("schema should be a string");
            let cores_distinct = snapshot
                .get_item("requested_cores_distinct")
                .expect("distinct lookup should not fail")
                .expect("distinct flag should exist")
                .extract::<bool>()
                .expect("distinct flag should be bool");
            let pacing_mode = snapshot
                .get_item("pacing_mode")
                .expect("pacing lookup should not fail")
                .expect("pacing should exist")
                .extract::<String>()
                .expect("pacing should be a string");
            let spin_admissible = snapshot
                .get_item("spin_tick_admissible")
                .expect("spin lookup should not fail")
                .expect("spin admissibility should exist")
                .extract::<bool>()
                .expect("spin admissibility should be bool");

            assert_eq!(schema, "scpn-control.runtime-admission-native.v1");
            assert!(cores_distinct);
            assert_eq!(pacing_mode, "spin");
            assert!(spin_admissible);
        });
    }
}

//! PyO3 Python bindings for SCPN Control.
//!
//! Slim control-only bindings (~400 LOC). Exposes:
//! - Grad-Shafranov solver (FusionKernel)
//! - SCPN runtime kernels (dense_activations, marking_update, sample_firing)
//! - SNN controller (SpikingControllerPool, NeuroCyberneticController)
//! - MPC controller
//! - Plasma2D digital twin plant model
//! - Transport solver (Chang-Hinton, Sauter bootstrap)
//! - Analytic equilibrium (Shafranov Bv, coil currents)

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use control_control::digital_twin::Plasma2D;
use control_control::mpc::{MPController, NeuralSurrogate};
use control_control::pid::{IsoFluxController, PIDController};
use control_control::snn::{NeuroCyberneticController, SpikingControllerPool};
use control_core::kernel::FusionKernel;
use control_core::transport::{self, NeoclassicalParams, TransportSolver};
use control_math::kuramoto;

// ─── Equilibrium solver ───

/// Python-accessible Grad-Shafranov equilibrium solver.
#[pyclass]
struct PyFusionKernel {
    inner: FusionKernel,
}

#[pymethods]
impl PyFusionKernel {
    #[new]
    fn new(config_path: &str) -> PyResult<Self> {
        let inner = FusionKernel::from_file(config_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(PyFusionKernel { inner })
    }

    fn solve_equilibrium(&mut self) -> PyResult<PyEquilibriumResult> {
        let result = self
            .inner
            .solve_equilibrium()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
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

    fn get_j_phi<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.j_phi().clone().into_pyarray(py)
    }

    fn get_r<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.grid().r.clone().into_pyarray(py)
    }

    fn get_z<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.grid().z.clone().into_pyarray(py)
    }

    fn grid_shape(&self) -> (usize, usize) {
        (self.inner.grid().nr, self.inner.grid().nz)
    }

    fn set_solver_method(&mut self, method: &str) -> PyResult<()> {
        let m = match method.to_ascii_lowercase().as_str() {
            "sor" | "picard_sor" => control_core::kernel::SolverMethod::PicardSor,
            "multigrid" | "picard_multigrid" | "mg" => {
                control_core::kernel::SolverMethod::PicardMultigrid
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown solver method '{method}'. Use 'sor' or 'multigrid'."
                )))
            }
        };
        self.inner.set_solver_method(m);
        Ok(())
    }

    fn solver_method(&self) -> &str {
        match self.inner.solver_method() {
            control_core::kernel::SolverMethod::PicardSor => "sor",
            control_core::kernel::SolverMethod::PicardMultigrid => "multigrid",
        }
    }
}

// ─── Result types ───

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
impl PyEquilibriumResult {
    fn __repr__(&self) -> String {
        format!(
            "EquilibriumResult(converged={}, iters={}, residual={:.2e})",
            self.converged, self.iterations, self.residual
        )
    }
}

// ─── SCPN runtime kernels ───

#[pyfunction]
fn scpn_dense_activations<'py>(
    py: Python<'py>,
    arg1: &Bound<'py, PyAny>,
    arg2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (w, m) = if let (Ok(w), Ok(m)) = (
        arg1.extract::<PyReadonlyArray2<'py, f64>>(),
        arg2.extract::<PyReadonlyArray1<'py, f64>>(),
    ) {
        (w, m)
    } else if let (Ok(m), Ok(w)) = (
        arg1.extract::<PyReadonlyArray1<'py, f64>>(),
        arg2.extract::<PyReadonlyArray2<'py, f64>>(),
    ) {
        (w, m)
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "scpn_dense_activations expects (matrix, vector) or (vector, matrix)",
        ));
    };
    let w = w.as_array();
    let m = m.as_array();
    if w.ncols() != m.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "shape mismatch: w_in has {} columns, marking has length {}",
            w.ncols(),
            m.len()
        )));
    }
    Ok(w.dot(&m).into_pyarray(py))
}

#[pyfunction]
fn scpn_marking_update<'py>(
    py: Python<'py>,
    marking: PyReadonlyArray1<'py, f64>,
    w_in: &Bound<'py, PyAny>,
    w_out: &Bound<'py, PyAny>,
    firing: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let m = marking.as_array();
    let f = firing.as_array();

    if let (Ok(wi_2d), Ok(wo_2d)) = (
        w_in.extract::<PyReadonlyArray2<'py, f64>>(),
        w_out.extract::<PyReadonlyArray2<'py, f64>>(),
    ) {
        let wi = wi_2d.as_array();
        let wo = wo_2d.as_array();
        if wi.nrows() != f.len()
            || wo.ncols() != f.len()
            || wi.ncols() != m.len()
            || wo.nrows() != m.len()
        {
            return Err(pyo3::exceptions::PyValueError::new_err("shape mismatch"));
        }
        let cons = wi.t().dot(&f);
        let prod = wo.dot(&f);
        let out = (&m - &cons + &prod).mapv(|x| x.clamp(0.0, 1.0));
        return Ok(out.into_pyarray(py));
    }

    if let (Ok(pre_1d), Ok(post_1d)) = (
        w_in.extract::<PyReadonlyArray1<'py, f64>>(),
        w_out.extract::<PyReadonlyArray1<'py, f64>>(),
    ) {
        let pre = pre_1d.as_array();
        let post = post_1d.as_array();
        if pre.len() != m.len() || post.len() != m.len() || f.len() != m.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "vector mode length mismatch",
            ));
        }
        let out = &m - &(&pre * &f) + &(&post * &f);
        return Ok(out.into_pyarray(py));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "scpn_marking_update expects matrix or vector pre/post arguments",
    ))
}

#[pyfunction]
fn scpn_sample_firing<'py>(
    py: Python<'py>,
    p_fire: PyReadonlyArray1<'py, f64>,
    n_passes: usize,
    seed: u64,
    antithetic: bool,
) -> Bound<'py, PyArray1<f64>> {
    let probs = p_fire.as_array();
    let n_t = probs.len();
    let mut counts = vec![0_u32; n_t];
    let mut rng = StdRng::seed_from_u64(seed);

    if antithetic && n_passes >= 2 {
        let n_pairs = n_passes.div_ceil(2);
        let odd_trim = !n_passes.is_multiple_of(2);
        for pair_idx in 0..n_pairs {
            let use_mirror = !odd_trim || (pair_idx + 1 < n_pairs);
            for (i, p_raw) in probs.iter().enumerate() {
                let p = p_raw.clamp(0.0, 1.0);
                let u: f64 = rng.gen();
                if u < p {
                    counts[i] += 1;
                }
                if use_mirror && (1.0 - u) < p {
                    counts[i] += 1;
                }
            }
        }
    } else {
        for _ in 0..n_passes {
            for (i, p_raw) in probs.iter().enumerate() {
                let p = p_raw.clamp(0.0, 1.0);
                let u: f64 = rng.gen();
                if u < p {
                    counts[i] += 1;
                }
            }
        }
    }

    let denom = n_passes.max(1) as f64;
    let out = Array1::from_iter(counts.into_iter().map(|c| (c as f64) / denom));
    out.into_pyarray(py)
}

// ─── Control systems ───

#[pyfunction]
fn shafranov_bv(r_geo: f64, a_min: f64, ip_ma: f64) -> PyResult<(f64, f64, f64)> {
    let result = control_control::analytic::shafranov_bv(r_geo, a_min, ip_ma)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok((result.bv_required, result.term_log, result.term_physics))
}

#[pyfunction]
fn solve_coil_currents(green_func: Vec<f64>, target_bv: f64) -> PyResult<Vec<f64>> {
    control_control::analytic::solve_coil_currents(&green_func, target_bv)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

// ─── SNN controller ───

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
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PySnnPool { inner })
    }

    fn step(&mut self, error: f64) -> PyResult<f64> {
        self.inner
            .step(error)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
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
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PySnnController { inner })
    }

    fn step(&mut self, measured_r: f64, measured_z: f64) -> PyResult<(f64, f64)> {
        self.inner
            .step(measured_r, measured_z)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
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

// ─── MPC controller ───

#[pyclass]
struct PyMpcController {
    inner: MPController,
}

#[pymethods]
impl PyMpcController {
    #[new]
    fn new(
        b_matrix: PyReadonlyArray2<'_, f64>,
        target: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let surrogate = NeuralSurrogate::new(b_matrix.as_array().to_owned());
        let inner = MPController::new(surrogate, target.as_array().to_owned())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn plan<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let result = self
            .inner
            .plan(&state.as_array().to_owned())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }
}

// ─── Digital twin plant ───

#[pyclass]
struct PyPlasma2D {
    inner: Plasma2D,
}

#[pymethods]
impl PyPlasma2D {
    #[new]
    fn new() -> Self {
        Self {
            inner: Plasma2D::new(),
        }
    }

    fn step(&mut self, action: f64) -> PyResult<(f64, f64)> {
        self.inner
            .step(action)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn measure_core_temp(&self, noise: f64) -> PyResult<f64> {
        self.inner
            .measure_core_temp(noise)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

// ─── Transport solver ───

fn ensure_matching_length(name: &str, expected: usize, actual: usize) -> PyResult<()> {
    if expected != actual {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} length mismatch: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

#[pyclass]
struct PyTransportSolver {
    inner: TransportSolver,
}

#[pymethods]
impl PyTransportSolver {
    #[new]
    fn new() -> Self {
        Self {
            inner: TransportSolver::new(),
        }
    }

    fn evolve_profiles(&mut self, p_aux_mw: f64) -> PyResult<()> {
        self.inner
            .evolve_profiles(p_aux_mw)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn transport_step(&mut self, p_aux_mw: f64, dt: f64) -> PyResult<()> {
        transport::transport_step(&mut self.inner, p_aux_mw, dt)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn chang_hinton_chi_profile<'py>(
        &self,
        py: Python<'py>,
        rho: PyReadonlyArray1<'py, f64>,
        t_i_kev: PyReadonlyArray1<'py, f64>,
        n_e_19: PyReadonlyArray1<'py, f64>,
        q: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (rho_a, ti_a, ne_a, q_a) = (
            rho.as_array().to_owned(),
            t_i_kev.as_array().to_owned(),
            n_e_19.as_array().to_owned(),
            q.as_array().to_owned(),
        );
        let n = rho_a.len();
        ensure_matching_length("t_i_kev", n, ti_a.len())?;
        ensure_matching_length("n_e_19", n, ne_a.len())?;
        ensure_matching_length("q", n, q_a.len())?;
        let params = NeoclassicalParams {
            q_profile: q_a.clone(),
            ..NeoclassicalParams::default()
        };
        let chi = transport::chang_hinton_chi_profile(&rho_a, &ti_a, &ne_a, &q_a, &params);
        Ok(chi.into_pyarray(py))
    }

    #[allow(clippy::too_many_arguments)]
    fn sauter_bootstrap_profile<'py>(
        &self,
        py: Python<'py>,
        rho: PyReadonlyArray1<'py, f64>,
        t_e_kev: PyReadonlyArray1<'py, f64>,
        t_i_kev: PyReadonlyArray1<'py, f64>,
        n_e_19: PyReadonlyArray1<'py, f64>,
        q: PyReadonlyArray1<'py, f64>,
        epsilon: PyReadonlyArray1<'py, f64>,
        b_field: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (rho_a, te_a, ti_a, ne_a, q_a, eps_a) = (
            rho.as_array().to_owned(),
            t_e_kev.as_array().to_owned(),
            t_i_kev.as_array().to_owned(),
            n_e_19.as_array().to_owned(),
            q.as_array().to_owned(),
            epsilon.as_array().to_owned(),
        );
        let n = rho_a.len();
        ensure_matching_length("t_e_kev", n, te_a.len())?;
        ensure_matching_length("t_i_kev", n, ti_a.len())?;
        ensure_matching_length("n_e_19", n, ne_a.len())?;
        ensure_matching_length("q", n, q_a.len())?;
        ensure_matching_length("epsilon", n, eps_a.len())?;
        let j_bs = transport::sauter_bootstrap_current_profile(
            &rho_a, &te_a, &ti_a, &ne_a, &q_a, &eps_a, b_field,
        );
        Ok(j_bs.into_pyarray(py))
    }
}

// ─── Kuramoto–Sakaguchi phase sync kernel ───

#[pyfunction]
#[pyo3(signature = (theta, omega, dt, k, alpha=0.0, zeta=0.0, psi_external=None))]
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
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64, f64, f64)> {
    let th = theta.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("theta not contiguous: {e}"))
    })?;
    let om = omega.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("omega not contiguous: {e}"))
    })?;
    if th.len() != om.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "theta/omega length mismatch",
        ));
    }
    let res = kuramoto::kuramoto_sakaguchi_step(th, om, dt, k, alpha, zeta, psi_external);
    Ok((res.theta.into_pyarray(py), res.r, res.psi_r, res.psi_global))
}

#[pyfunction]
#[pyo3(signature = (theta, omega, n_steps, dt, k, alpha=0.0, zeta=0.0, psi_external=None))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
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
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let th = theta.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("theta not contiguous: {e}"))
    })?;
    let om = omega.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("omega not contiguous: {e}"))
    })?;
    if th.len() != om.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "theta/omega length mismatch",
        ));
    }
    let (final_theta, r_hist) =
        kuramoto::kuramoto_sakaguchi_run(th, om, n_steps, dt, k, alpha, zeta, psi_external);
    Ok((
        final_theta.into_pyarray(py),
        Array1::from_vec(r_hist).into_pyarray(py),
    ))
}

// ─── UPDE multi-layer realtime monitor ───

#[pyclass]
struct PyRealtimeMonitor {
    theta_flat: Vec<f64>,
    omega_flat: Vec<f64>,
    knm_flat: Vec<f64>,
    zeta: Vec<f64>,
    n_layers: usize,
    n_per: usize,
    dt: f64,
    psi_driver: f64,
    pac_gamma: f64,
    tick_count: u64,
}

#[pymethods]
impl PyRealtimeMonitor {
    #[new]
    #[pyo3(signature = (knm, zeta, theta, omega, n_layers, n_per, dt=1e-3, psi_driver=0.0, pac_gamma=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        knm: PyReadonlyArray1<'_, f64>,
        zeta: PyReadonlyArray1<'_, f64>,
        theta: PyReadonlyArray1<'_, f64>,
        omega: PyReadonlyArray1<'_, f64>,
        n_layers: usize,
        n_per: usize,
        dt: f64,
        psi_driver: f64,
        pac_gamma: f64,
    ) -> PyResult<Self> {
        let th = theta.as_slice()?.to_vec();
        let om = omega.as_slice()?.to_vec();
        let k = knm.as_slice()?.to_vec();
        let z = zeta.as_slice()?.to_vec();
        if th.len() != n_layers * n_per {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "theta length mismatch",
            ));
        }
        if om.len() != n_layers * n_per {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "omega length mismatch",
            ));
        }
        if k.len() != n_layers * n_layers {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "knm length mismatch",
            ));
        }
        if z.len() != n_layers {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "zeta length mismatch",
            ));
        }
        Ok(Self {
            theta_flat: th,
            omega_flat: om,
            knm_flat: k,
            zeta: z,
            n_layers,
            n_per,
            dt,
            psi_driver,
            pac_gamma,
            tick_count: 0,
        })
    }

    /// Advance one UPDE step. Returns dict with R_global, R_layer, V_global, V_layer, Psi_global, tick.
    fn tick<'py>(&mut self, py: Python<'py>) -> PyResult<PyObject> {
        let res = kuramoto::upde_tick(
            &self.theta_flat,
            &self.omega_flat,
            &self.knm_flat,
            &self.zeta,
            self.n_layers,
            self.n_per,
            self.dt,
            self.psi_driver,
            self.pac_gamma,
        );
        self.theta_flat = res.theta_flat;
        self.tick_count += 1;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("tick", self.tick_count)?;
        dict.set_item("R_global", res.r_global)?;
        dict.set_item("R_layer", res.r_layer.into_pyarray(py))?;
        dict.set_item("Psi_global", res.psi_global)?;
        dict.set_item("V_global", res.v_global)?;
        dict.set_item("V_layer", res.v_layer.into_pyarray(py))?;
        Ok(dict.into())
    }

    #[setter]
    fn set_psi_driver(&mut self, v: f64) {
        self.psi_driver = v;
    }

    #[getter]
    fn get_psi_driver(&self) -> f64 {
        self.psi_driver
    }

    #[setter]
    fn set_pac_gamma(&mut self, v: f64) {
        self.pac_gamma = v;
    }

    #[getter]
    fn get_pac_gamma(&self) -> f64 {
        self.pac_gamma
    }

    #[getter]
    fn tick_count(&self) -> u64 {
        self.tick_count
    }

    fn reset(&mut self, theta: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let th = theta.as_slice()?.to_vec();
        if th.len() != self.n_layers * self.n_per {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "theta length mismatch",
            ));
        }
        self.theta_flat = th;
        self.tick_count = 0;
        Ok(())
    }
}

// ─── PID controller ───

#[pyclass]
struct PyPIDController {
    inner: PIDController,
}

#[pymethods]
impl PyPIDController {
    #[new]
    fn new(kp: f64, ki: f64, kd: f64) -> PyResult<Self> {
        let inner = PIDController::new(kp, ki, kd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn radial() -> PyResult<Self> {
        let inner = PIDController::radial()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn vertical() -> PyResult<Self> {
        let inner = PIDController::vertical()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn step(&mut self, error: f64) -> PyResult<f64> {
        self.inner
            .step(error)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn kp(&self) -> f64 {
        self.inner.kp
    }
    #[getter]
    fn ki(&self) -> f64 {
        self.inner.ki
    }
    #[getter]
    fn kd(&self) -> f64 {
        self.inner.kd
    }
}

#[pyclass]
struct PyIsoFluxController {
    inner: IsoFluxController,
}

#[pymethods]
impl PyIsoFluxController {
    #[new]
    fn new(target_r: f64, target_z: f64) -> PyResult<Self> {
        let inner = IsoFluxController::new(target_r, target_z)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn step(&mut self, measured_r: f64, measured_z: f64) -> PyResult<(f64, f64)> {
        self.inner
            .step(measured_r, measured_z)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
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

// ─── Module registration ───

/// SCPN Control — Rust-accelerated control pipeline.
#[pymodule]
fn scpn_control_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Equilibrium
    m.add_class::<PyFusionKernel>()?;
    m.add_class::<PyEquilibriumResult>()?;
    // SCPN kernels
    m.add_function(wrap_pyfunction!(scpn_dense_activations, m)?)?;
    m.add_function(wrap_pyfunction!(scpn_marking_update, m)?)?;
    m.add_function(wrap_pyfunction!(scpn_sample_firing, m)?)?;
    // Control
    m.add_function(wrap_pyfunction!(shafranov_bv, m)?)?;
    m.add_function(wrap_pyfunction!(solve_coil_currents, m)?)?;
    m.add_class::<PySnnPool>()?;
    m.add_class::<PySnnController>()?;
    m.add_class::<PyMpcController>()?;
    m.add_class::<PyPlasma2D>()?;
    m.add_class::<PyTransportSolver>()?;
    m.add_class::<PyPIDController>()?;
    m.add_class::<PyIsoFluxController>()?;
    // Phase dynamics (Paper 27)
    m.add_function(wrap_pyfunction!(kuramoto_step, m)?)?;
    m.add_function(wrap_pyfunction!(kuramoto_run, m)?)?;
    m.add_function(wrap_pyfunction!(kuramoto_run_lyapunov, m)?)?;
    m.add_class::<PyRealtimeMonitor>()?;
    Ok(())
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
    let th = theta.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("theta not contiguous: {e}"))
    })?;
    let om = omega.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("omega not contiguous: {e}"))
    })?;
    if th.len() != om.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "theta/omega length mismatch",
        ));
    }
    let (final_theta, r_hist, v_hist, lyap_exp) =
        kuramoto::kuramoto_run_lyapunov(th, om, n_steps, dt, k, alpha, zeta, psi_external);
    Ok((
        final_theta.into_pyarray(py),
        Array1::from_vec(r_hist).into_pyarray(py),
        Array1::from_vec(v_hist).into_pyarray(py),
        lyap_exp,
    ))
}

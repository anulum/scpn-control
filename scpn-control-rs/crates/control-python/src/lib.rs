// ──────────────────────────────────────────────────────────────────────
// SCPN Control — Rust Crate
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: MIT OR Apache-2.0
// ──────────────────────────────────────────────────────────────────────

//! PyO3 Python bindings for SCPN Control (Modern Bound API).

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError, PyIOError};
use pyo3::types::{PyDict, PyList};

use control_control::analytic;
use control_control::digital_twin::Plasma2D;
use control_control::h_infinity::HInfController;
use control_control::mpc::{MPController, NeuralSurrogate};
use control_control::optimal;
use control_control::snn::{NeuroCyberneticController, SpikingControllerPool};
use control_control::spi::SPIMitigation;
use control_core::amr_kernel::{AmrKernelConfig, AmrKernelSolver};
use control_core::vmec_interface::{self, VmecBoundaryState, VmecSolverConfig, VmecFourierMode};
use control_core::bout_interface::{self, BoutGridConfig};
use control_core::bfield;
use control_core::ignition;
use control_core::kernel::FusionKernel;
use control_core::particles::{self, ChargedParticle};
use control_core::transport::TransportSolver;
use control_core::xpoint;
use control_math::kuramoto;
use control_math::tridiag;
use control_math::multigrid::{multigrid_solve, MultigridConfig};
use control_types::state::Grid2D;

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
        let inner = FusionKernel::from_file(config_path)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(PyFusionKernel { inner })
    }

    fn solve_equilibrium<'py>(&mut self, _py: Python<'py>) -> PyResult<PyEquilibriumResult> {
        let result = self.inner.solve_equilibrium()
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

    fn compute_b_field<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
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
        let psi_vac = control_core::vacuum::calculate_vacuum_field(self.inner.grid(), self.inner.coils(), mu0)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(psi_vac.into_pyarray(py))
    }

    fn sample_psi_at(&self, r: f64, z: f64) -> PyResult<f64> {
        self.inner
            .sample_psi_at(r, z)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn calculate_thermodynamics<'py>(&self, py: Python<'py>, p_aux_mw: f64) -> PyResult<Bound<'py, PyDict>> {
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
fn shafranov_bv<'py>(py: Python<'py>, r_major: f64, a_minor: f64, ip_ma: f64) -> PyResult<Bound<'py, PyDict>> {
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
    let flat_resp = resp.as_slice().ok_or_else(|| PyValueError::new_err("response_matrix not contiguous"))?;
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
    fn new(n_neurons: usize, gain: f64, window_size: usize) -> PyResult<Self> {
        let inner = SpikingControllerPool::new(n_neurons, gain, window_size)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PySnnPool { inner })
    }

    fn step(&mut self, error: f64) -> PyResult<f64> {
        self.inner.step(error).map_err(|e| PyRuntimeError::new_err(e.to_string()))
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
        self.inner.step(r_err, z_err).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

#[pyclass]
struct PyMpcController {
    inner: MPController,
}

#[pymethods]
impl PyMpcController {
    #[new]
    fn new(weights: PyReadonlyArray2<'_, f64>, target: PyReadonlyArray1<'_, f64>) -> PyResult<Self> {
        let model = NeuralSurrogate { b_matrix: weights.as_array().to_owned() };
        let inner = MPController::new(model, target.as_array().to_owned())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyMpcController { inner })
    }

    fn plan<'py>(&self, py: Python<'py>, state: PyReadonlyArray1<'py, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let s = state.as_array().to_owned();
        let u = self.inner.plan(&s).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
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
        PyPlasma2D { inner: Plasma2D::new() }
    }

    fn step(&mut self, action: f64) -> PyResult<(f64, f64)> {
        self.inner.step(action).map_err(|e| PyRuntimeError::new_err(e.to_string()))
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
        PyTransportSolver { inner: TransportSolver::new() }
    }

    fn step(&mut self, dt: f64, p_heat: f64) -> PyResult<()> {
        self.inner.step(dt, p_heat).map_err(|e| PyRuntimeError::new_err(e.to_string()))
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
    fn new(a: PyReadonlyArray2<'_, f64>, b2: PyReadonlyArray2<'_, f64>, c2: PyReadonlyArray2<'_, f64>, gamma: f64, dt: f64) -> PyResult<Self> {
        let plant = control_control::h_infinity::HInfPlant::new(
            a.as_array().to_owned(),
            Array2::zeros((a.shape()[0], 1)), 
            b2.as_array().to_owned(),
            Array2::zeros((1, a.shape()[0])), 
            c2.as_array().to_owned(),
        ).map_err(PyValueError::new_err)?;
        Ok(PyHInfController {
            inner: HInfController::new(plant, gamma, 1e6, dt),
        })
    }

    fn step(&mut self, y: f64, dt: f64) -> f64 {
        self.inner.step(y, dt)
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
    fn new(max_levels: usize, refinement_threshold: f64, omega: f64, coarse_iters: usize, patch_iters: usize, blend: f64) -> PyResult<Self> {
        let config = AmrKernelConfig { max_levels, refinement_threshold, omega, coarse_iters, patch_iters, blend };
        let inner = AmrKernelSolver::new(config).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyAmrSolver { inner })
    }

    fn solve_with_hierarchy<'py>(&self, py: Python<'py>, psi: PyReadonlyArray2<'py, f64>, r: PyReadonlyArray1<'py, f64>, z: PyReadonlyArray1<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let rs = r.as_slice()?;
        let zs = z.as_slice()?;
        let nr = r.shape()[0];
        let nz = z.shape()[0];
        let grid = Grid2D::new(nr, nz, rs[0], rs[nr-1], zs[0], zs[nz-1]);
        let mut psi_out = psi.as_array().to_owned();
        self.inner.solve_with_hierarchy(&grid, &mut psi_out).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
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
    fn new(m_pol: usize, n_tor: usize, ns: usize, ntheta: usize, nzeta: usize, max_iter: usize, tol: f64, step_size: f64) -> Self {
        PyVmecSolver { config: VmecSolverConfig { m_pol, n_tor, ns, ntheta, nzeta, max_iter, tol, step_size } }
    }

    fn solve_fixed_boundary<'py>(
        &self, py: Python<'py>, r_axis: f64, z_axis: f64, a_minor: f64, kappa: f64, triangularity: f64,
        modes: Vec<(i32, i32, f64, f64, f64, f64)>, pressure: PyReadonlyArray1<'py, f64>, iota: PyReadonlyArray1<'py, f64>, phi_edge: f64
    ) -> PyResult<PyObject> {
        let fourier_modes: Vec<VmecFourierMode> = modes.into_iter().map(|(m, n, r_cos, r_sin, z_cos, z_sin)| VmecFourierMode { m, n, r_cos, r_sin, z_cos, z_sin }).collect();
        let boundary = VmecBoundaryState { r_axis, z_axis, a_minor, kappa, triangularity, nfp: 1, modes: fourier_modes };
        let result = vmec_interface::vmec_fixed_boundary_solve(&boundary, &self.config, pressure.as_slice()?, iota.as_slice()?, phi_edge)
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
        let grid_2d = Grid2D::new(nr, nz, rs[0], rs[nr-1], zs[0], zs[nz-1]);
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
    fn new() -> Self {
        PySPIMitigation {
            inner: SPIMitigation::default(),
        }
    }

    fn run<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let history = self.inner.run().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
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
    let grid = Grid2D::new(nr, nz, rs[0], rs[nr-1], zs[0], zs[nz-1]);
    let j_phi = particles::deposit_toroidal_current_density(&particles, &grid)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(j_phi.into_pyarray(py))
}

#[pyfunction]
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
    let (final_th, r_hist) = kuramoto::kuramoto_sakaguchi_run(th, om, n_steps, dt, k, alpha, zeta, psi_external);
    Ok((final_th.into_pyarray(py), r_hist))
}

#[pyfunction]
#[pyo3(signature = (theta, omega, n_steps, dt, k, alpha=0.0, zeta=0.0, psi_external=None))]
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
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, f64)> {
    let th = theta.as_slice()?;
    let om = omega.as_slice()?;
    let (final_th, r_hist, v_hist, lyap) = kuramoto::kuramoto_run_lyapunov(th, om, n_steps, dt, k, alpha, zeta, psi_external);
    Ok((final_th.into_pyarray(py), Array1::from_vec(r_hist).into_pyarray(py), Array1::from_vec(v_hist).into_pyarray(py), lyap))
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
    let grid = Grid2D::new(nr, nz, rs[0], rs[nr-1], zs[0], zs[nz-1]);
    let mut psi_out = psi.as_array().to_owned();
    let config = MultigridConfig::default();
    multigrid_solve(&mut psi_out, &source.as_array().to_owned(), &grid, &config, 100, 1e-8);
    Ok(psi_out.into_pyarray(py))
}

// ─── Module registration ───

#[pymodule]
fn scpn_control_rs<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<PyFusionKernel>()?;
    m.add_class::<PyEquilibriumResult>()?;
    m.add_function(wrap_pyfunction!(scpn_dense_activations, m)?)?;
    m.add_function(wrap_pyfunction!(scpn_marking_update, m)?)?;
    m.add_function(wrap_pyfunction!(shafranov_bv, m)?)?;
    m.add_function(wrap_pyfunction!(solve_coil_currents, m)?)?;
    m.add_class::<PySnnPool>()?;
    m.add_class::<PySnnController>()?;
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
    Ok(())
}

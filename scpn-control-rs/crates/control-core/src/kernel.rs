// ──────────────────────────────────────────────────────────────────────
// SCPN Control — Fusion Kernel
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: MIT OR Apache-2.0
// ──────────────────────────────────────────────────────────────────────
//! Axisymmetric Grad-Shafranov equilibrium solver (Picard/Multigrid).

use std::time::Instant;

use ndarray::Array2;

use control_math::multigrid::{multigrid_solve, MultigridConfig};
use control_types::config::ProfileParams;
use control_types::config::ReactorConfig;
use control_types::error::FusionResult;
use control_types::state::{Grid2D, PlasmaState};

use crate::source;
use crate::xpoint;

/// Fixed constant for initial Gaussian current seed [m].
const SEED_GAUSSIAN_SIGMA: f64 = 0.3;

/// Red-Black SOR step for poloidal flux.
pub fn sor_step(psi: &mut Array2<f64>, source: &Array2<f64>, grid: &Grid2D, omega: f64) {
    let nz = grid.nz;
    let nr = grid.nr;
    let dr2 = grid.dr * grid.dr;
    let dz2 = grid.dz * grid.dz;
    let a_ns = 1.0 / dz2;
    let a_c = 2.0 / dr2 + 2.0 / dz2;

    for parity in [0_usize, 1_usize] {
        for iz in 1..nz - 1 {
            for ir in 1..nr - 1 {
                if (iz + ir) % 2 != parity {
                    continue;
                }
                let r = grid.r_at(iz, ir).max(1e-10);
                let a_e = 1.0 / dr2 + 1.0 / (2.0 * r * grid.dr);
                let a_w = 1.0 / dr2 - 1.0 / (2.0 * r * grid.dr);
                let gs_update = (a_e * psi[[iz, ir + 1]]
                    + a_w * psi[[iz, ir - 1]]
                    + a_ns * psi[[iz - 1, ir]]
                    + a_ns * psi[[iz + 1, ir]]
                    - source[[iz, ir]])
                    / a_c;
                let old = psi[[iz, ir]];
                psi[[iz, ir]] = (1.0 - omega) * old + omega * gs_update;
            }
        }
    }
}

/// Selects the inner linear solver used in Picard iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverMethod {
    /// Red-Black SOR inner solve (default, ω = 1.8).
    PicardSor,
    /// Multi-grid V-cycle solver.
    PicardMultigrid,
}

/// Grad-Shafranov equilibrium result.
#[derive(Debug, Clone)]
pub struct EquilibriumResult {
    pub converged: bool,
    pub iterations: usize,
    pub residual: f64,
    pub axis_position: (f64, f64),
    pub x_point_position: (f64, f64),
    pub psi_axis: f64,
    pub psi_boundary: f64,
    pub solve_time_ms: f64,
}

/// The core Grad-Shafranov solver kernel.
pub struct FusionKernel {
    pub config: ReactorConfig,
    pub grid: Grid2D,
    pub state: PlasmaState,
    pub solver_method: SolverMethod,
    pub external_profile_mode: bool,
    pub profile_params_p: Option<ProfileParams>,
    pub profile_params_ff: Option<ProfileParams>,
    pub particle_current_feedback: Option<Array2<f64>>,
    pub particle_feedback_coupling: f64,
}

impl FusionKernel {
    pub fn new(config: ReactorConfig) -> Self {
        let grid = config.create_grid();
        let nz = grid.nz;
        let nr = grid.nr;

        let state = PlasmaState {
            time: 0.0,
            psi: Array2::zeros((nz, nr)),
            j_phi: Array2::zeros((nz, nr)),
            b_r: None,
            b_z: None,
            b_phi: None,
            axis: None,
            psi_axis: 0.0,
            psi_boundary: 0.0,
        };

        FusionKernel {
            config,
            grid,
            state,
            solver_method: SolverMethod::PicardMultigrid,
            external_profile_mode: false,
            profile_params_p: None,
            profile_params_ff: None,
            particle_current_feedback: None,
            particle_feedback_coupling: 0.0,
        }
    }

    pub fn from_file(path: &str) -> FusionResult<Self> {
        let config = ReactorConfig::from_file(path)?;
        let grid = config.create_grid();
        let nz = grid.nz;
        let nr = grid.nr;

        let state = PlasmaState {
            time: 0.0,
            psi: Array2::zeros((nz, nr)),
            j_phi: Array2::zeros((nz, nr)),
            b_r: None,
            b_z: None,
            b_phi: None,
            axis: None,
            psi_axis: 0.0,
            psi_boundary: 0.0,
        };

        Ok(FusionKernel {
            config,
            grid,
            state,
            solver_method: SolverMethod::PicardMultigrid,
            external_profile_mode: false,
            profile_params_p: None,
            profile_params_ff: None,
            particle_current_feedback: None,
            particle_feedback_coupling: 0.0,
        })
    }

    pub fn psi(&self) -> &Array2<f64> {
        &self.state.psi
    }

    pub fn j_phi(&self) -> &Array2<f64> {
        &self.state.j_phi
    }

    pub fn grid(&self) -> &Grid2D {
        &self.grid
    }

    pub fn coils(&self) -> &[control_types::config::CoilConfig] {
        &self.config.coils
    }

    pub fn set_solver_method(&mut self, method: SolverMethod) {
        self.solver_method = method;
    }

    pub fn set_external_profiles(&mut self, params_p: ProfileParams, params_ff: ProfileParams) {
        self.external_profile_mode = true;
        self.profile_params_p = Some(params_p);
        self.profile_params_ff = Some(params_ff);
    }

    pub fn solve_equilibrium_with_profiles(
        &mut self,
        params_p: ProfileParams,
        params_ff: ProfileParams,
    ) -> FusionResult<EquilibriumResult> {
        self.set_external_profiles(params_p, params_ff);
        self.solve_equilibrium()
    }

    pub fn solve_equilibrium(&mut self) -> FusionResult<EquilibriumResult> {
        let t0 = Instant::now();
        let nz = self.grid.nz;
        let nr = self.grid.nr;
        let mu0 = self.config.physics.vacuum_permeability;

        // 1. Prepare vacuum field boundary
        let psi_vac = crate::vacuum::calculate_vacuum_field(&self.grid, &self.config.coils, mu0)?;
        self.state.psi.assign(&psi_vac);

        // 2. Initial current seed (Gaussian)
        let r_center = (self.config.dimensions.r_min + self.config.dimensions.r_max) / 2.0;
        let z_center = 0.0;
        for iz in 0..nz {
            for ir in 0..nr {
                let r = self.grid.r_at(iz, ir);
                let z = self.grid.z_at(iz, ir);
                let dist_sq = (r - r_center).powi(2) + (z - z_center).powi(2);
                self.state.j_phi[[iz, ir]] =
                    (-dist_sq / (2.0 * SEED_GAUSSIAN_SIGMA * SEED_GAUSSIAN_SIGMA)).exp();
            }
        }

        // Picard iteration loop
        let mut converged = false;
        let mut iterations = 0;
        let mut last_residual = 1.0;

        for k in 0..self.config.solver.max_iterations {
            iterations = k + 1;

            // Source = -μ₀ R J_phi
            let mut source = Array2::zeros((nz, nr));
            for iz in 0..nz {
                for ir in 0..nr {
                    source[[iz, ir]] = -mu0 * self.grid.r_at(iz, ir) * self.state.j_phi[[iz, ir]];
                }
            }

            let mut psi_new = self.state.psi.clone();
            match self.solver_method {
                SolverMethod::PicardSor => {
                    for _ in 0..10 {
                        sor_step(
                            &mut psi_new,
                            &source,
                            &self.grid,
                            self.config.solver.sor_omega,
                        );
                    }
                }
                SolverMethod::PicardMultigrid => {
                    let mg_config = MultigridConfig::default();
                    multigrid_solve(&mut psi_new, &source, &self.grid, &mg_config, 1, 1e-8);
                }
            }

            let diff = (&psi_new - &self.state.psi)
                .mapv(|v| v.abs())
                .fold(0.0, |acc, &x| f64::max(acc, x));
            self.state.psi.assign(&psi_new);
            last_residual = diff;

            if diff < self.config.solver.convergence_threshold {
                converged = true;
                break;
            }

            // 4. Update current density from flux (Picard step)
            let (_axis_pos, psi_axis) = xpoint::find_axis(&self.state.psi, &self.grid)?;
            let (_x_pos, psi_boundary) = xpoint::find_x_point(&self.state.psi, &self.grid, -1.0)?;

            if self.external_profile_mode {
                let params_p = self.profile_params_p.as_ref().unwrap();
                let params_ff = self.profile_params_ff.as_ref().unwrap();
                source::update_plasma_source_with_profiles(
                    &self.state.psi,
                    &mut self.state.j_phi,
                    &self.grid,
                    psi_axis,
                    psi_boundary,
                    params_p,
                    params_ff,
                    mu0,
                    self.config.physics.plasma_current_target,
                )?;
            } else {
                source::update_plasma_source_nonlinear(
                    &self.state.psi,
                    &mut self.state.j_phi,
                    &self.grid,
                    psi_axis,
                    psi_boundary,
                )?;
            }
        }

        // Post-processing: final find axis and X-point
        let (axis_pos, psi_axis) = xpoint::find_axis(&self.state.psi, &self.grid)?;
        let (x_pos, psi_boundary) = xpoint::find_x_point(&self.state.psi, &self.grid, -1.0)?;

        Ok(EquilibriumResult {
            converged,
            iterations,
            residual: last_residual,
            psi_axis,
            psi_boundary,
            axis_position: axis_pos,
            x_point_position: x_pos,
            solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
        })
    }

    pub fn sample_psi_at(&self, r: f64, z: f64) -> FusionResult<f64> {
        // Simple bilinear interpolation
        source::interpolate_2d(&self.state.psi, &self.grid.r, &self.grid.z, r, z)
    }

    pub fn sample_psi_at_probes(&self, probes_rz: &[(f64, f64)]) -> FusionResult<Vec<f64>> {
        let mut results = Vec::with_capacity(probes_rz.len());
        for &(r, z) in probes_rz {
            results.push(self.sample_psi_at(r, z)?);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn project_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
    }

    fn config_path(relative: &str) -> String {
        project_root().join(relative).to_string_lossy().to_string()
    }

    #[test]
    fn test_kernel_init() {
        let kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        assert_eq!(kernel.grid.nr, 129);
        assert_eq!(kernel.grid.nz, 129);
    }

    #[test]
    fn test_solve_equilibrium_runs() {
        let mut kernel = FusionKernel::from_file(&config_path("iter_config.json")).unwrap();
        kernel.config.solver.max_iterations = 5;
        let result = kernel.solve_equilibrium().expect("solver should run");
        assert!(result.iterations <= 5);
    }
}

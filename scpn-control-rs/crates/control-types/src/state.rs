// ─────────────────────────────────────────────────────────────────────
// SCPN Control — State
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: MIT OR Apache-2.0
// ─────────────────────────────────────────────────────────────────────
//! Physical state representations for the tokamak plasma.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// 2D rectangular grid in (R, Z) cylindrical coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Grid2D {
    pub nr: usize,
    pub nz: usize,
    pub r: Array1<f64>,
    pub z: Array1<f64>,
    pub dr: f64,
    pub dz: f64,
}

impl Grid2D {
    pub fn new(nr: usize, nz: usize, r_min: f64, r_max: f64, z_min: f64, z_max: f64) -> Self {
        let r = Array1::linspace(r_min, r_max, nr);
        let z = Array1::linspace(z_min, z_max, nz);
        let dr = (r_max - r_min) / (nr - 1) as f64;
        let dz = (z_max - z_min) / (nz - 1) as f64;

        Grid2D {
            nr,
            nz,
            r,
            z,
            dr,
            dz,
        }
    }

    /// Virtual coordinate R at index (iz, ir).
    #[inline]
    pub fn r_at(&self, _iz: usize, ir: usize) -> f64 {
        self.r[ir]
    }

    /// Virtual coordinate Z at index (iz, ir).
    #[inline]
    pub fn z_at(&self, iz: usize, _ir: usize) -> f64 {
        self.z[iz]
    }
}

/// 1D radial profiles on the normalized flux grid rho.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadialProfiles {
    /// Normalised radius [0, 1].
    pub rho: Array1<f64>,
    /// Electron temperature [keV].
    pub te: Array1<f64>,
    /// Ion temperature [keV].
    pub ti: Array1<f64>,
    /// Electron density [10^19 m^-3].
    pub ne: Array1<f64>,
    /// Impurity density [10^19 m^-3].
    pub n_impurity: Array1<f64>,
    /// Safety factor q.
    pub q: Array1<f64>,
    /// Poloidal current function F = R * B_phi.
    pub fpol: Array1<f64>,
}

/// Complete plasma state for a single time slice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasmaState {
    pub time: f64,
    /// Poloidal flux [nz, nr].
    pub psi: Array2<f64>,
    /// Toroidal current density [nz, nr].
    pub j_phi: Array2<f64>,
    /// Magnetic field R-component [nz, nr].
    pub b_r: Option<Array2<f64>>,
    /// Magnetic field Z-component [nz, nr].
    pub b_z: Option<Array2<f64>>,
    /// Magnetic field phi-component [nz, nr].
    pub b_phi: Option<Array2<f64>>,
    /// Magnetic axis position (R, Z).
    pub axis: Option<(f64, f64)>,
    /// Magnetic axis flux value.
    pub psi_axis: f64,
    /// Boundary flux value (separatrix).
    pub psi_boundary: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicsResult {
    pub p_fusion_mw: f64,
    pub p_alpha_mw: f64,
    pub p_loss_mw: f64,
    pub p_aux_mw: f64,
    pub net_mw: f64,
    pub q_factor: f64,
    pub t_peak_kev: f64,
    pub w_thermal_mj: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityResult {
    pub overall_stable: bool,
    pub n_criteria_checked: usize,
    pub n_criteria_stable: usize,
    pub eigenvalues: [f64; 2],
    pub eigenvectors: [[f64; 2]; 2],
    pub decay_index: f64,
    pub radial_force_mn: f64,
    pub vertical_force_mn: f64,
    pub is_stable: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_creation_65() {
        let grid = Grid2D::new(65, 65, 1.0, 9.0, -5.0, 5.0);
        assert_eq!(grid.nr, 65);
        assert_eq!(grid.nz, 65);
        assert!((grid.r[0] - 1.0).abs() < 1e-10);
        assert!((grid.r[64] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_grid_virtual_consistency() {
        let grid = Grid2D::new(128, 128, 1.0, 9.0, -5.0, 5.0);
        assert_eq!(grid.r_at(10, 20), grid.r[20]);
        assert_eq!(grid.z_at(10, 20), grid.z[10]);
    }

    #[test]
    fn test_plasma_state_initialization() {
        let state = PlasmaState {
            time: 0.0,
            psi: Array2::zeros((128, 128)),
            j_phi: Array2::zeros((128, 128)),
            b_r: None,
            b_z: None,
            b_phi: None,
            axis: None,
            psi_axis: 0.0,
            psi_boundary: 1.0,
        };
        assert_eq!(state.psi.shape(), &[128, 128]);
        assert_eq!(state.j_phi.shape(), &[128, 128]);
        assert!(state.b_phi.is_none());
    }
}

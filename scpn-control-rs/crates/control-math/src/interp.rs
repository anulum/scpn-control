// ─────────────────────────────────────────────────────────────────────
// SCPN Control — Interp
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Interpolation and spatial differentiation utilities.

use control_types::state::Grid2D;
use ndarray::Array2;

/// Bilinear interpolation of a 2D scalar field on a regular grid.
pub fn interp2d(field: &Array2<f64>, grid: &Grid2D, r: f64, z: f64) -> f64 {
    let nr = grid.nr;
    let nz = grid.nz;

    let ir = match grid
        .r
        .iter()
        .position(|&x| x >= r)
        .map(|idx| if idx == 0 { 0 } else { idx - 1 })
    {
        Some(idx) => idx.min(nr - 2),
        None => 0,
    };

    let iz = match grid
        .z
        .iter()
        .position(|&x| x >= z)
        .map(|idx| if idx == 0 { 0 } else { idx - 1 })
    {
        Some(idx) => idx.min(nz - 2),
        None => 0,
    };

    let dr = grid.r[ir + 1] - grid.r[ir];
    let dz = grid.z[iz + 1] - grid.z[iz];
    let fr = (r - grid.r[ir]) / dr;
    let fz = (z - grid.z[iz]) / dz;

    let v00 = field[[iz, ir]];
    let v10 = field[[iz, ir + 1]];
    let v01 = field[[iz + 1, ir]];
    let v11 = field[[iz + 1, ir + 1]];

    let v0 = v00 * (1.0 - fr) + v10 * fr;
    let v1 = v01 * (1.0 - fr) + v11 * fr;

    v0 * (1.0 - fz) + v1 * fz
}

/// Compute 2D central-difference gradients (df/dZ, df/dR).
pub fn gradient_2d(field: &Array2<f64>, grid: &Grid2D) -> (Array2<f64>, Array2<f64>) {
    let nr = grid.nr;
    let nz = grid.nz;
    let dr = grid.dr;
    let dz = grid.dz;

    let mut df_dz = Array2::zeros((nz, nr));
    let mut df_dr = Array2::zeros((nz, nr));

    // Interior points via central differences
    for iz in 1..nz - 1 {
        for ir in 1..nr - 1 {
            df_dz[[iz, ir]] = (field[[iz + 1, ir]] - field[[iz - 1, ir]]) / (2.0 * dz);
            df_dr[[iz, ir]] = (field[[iz, ir + 1]] - field[[iz, ir - 1]]) / (2.0 * dr);
        }
    }

    // Boundary points (simple forward/backward for now)
    for ir in 0..nr {
        df_dz[[0, ir]] = (field[[1, ir]] - field[[0, ir]]) / dz;
        df_dz[[nz - 1, ir]] = (field[[nz - 1, ir]] - field[[nz - 2, ir]]) / dz;
    }
    for iz in 0..nz {
        df_dr[[iz, 0]] = (field[[iz, 1]] - field[[iz, 0]]) / dr;
        df_dr[[iz, nr - 1]] = (field[[iz, nr - 1]] - field[[iz, nr - 2]]) / dr;
    }

    (df_dz, df_dr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interp2d_midpoint() {
        let grid = Grid2D::new(5, 5, 0.0, 4.0, 0.0, 4.0);
        // Constant field
        let field = Array2::from_elem((5, 5), 7.0);
        let val = interp2d(&field, &grid, 1.5, 2.5);
        assert!((val - 7.0).abs() < 1e-10, "Constant field interpolation");
    }

    #[test]
    fn test_interp2d_linear() {
        let grid = Grid2D::new(11, 11, 0.0, 10.0, 0.0, 10.0);
        // f(R, Z) = R + Z (linear)
        let field =
            Array2::from_shape_fn((11, 11), |(iz, ir)| grid.r_at(iz, ir) + grid.z_at(iz, ir));
        // At (R=3.5, Z=6.5), expected = 3.5 + 6.5 = 10.0
        let val = interp2d(&field, &grid, 3.5, 6.5);
        assert!((val - 10.0).abs() < 1e-10, "Linear interpolation: {val}");
    }

    #[test]
    fn test_gradient_2d_linear() {
        let grid = Grid2D::new(11, 11, 0.0, 10.0, 0.0, 10.0);
        // f(R, Z) = 2*R + 3*Z → df/dR = 2, df/dZ = 3
        let field = Array2::from_shape_fn((11, 11), |(iz, ir)| {
            2.0 * grid.r_at(iz, ir) + 3.0 * grid.z_at(iz, ir)
        });
        let (df_dz, df_dr) = gradient_2d(&field, &grid);

        // Check interior points (central differences are exact for linear)
        for iz in 1..10 {
            for ir in 1..10 {
                assert!(
                    (df_dr[[iz, ir]] - 2.0).abs() < 1e-10,
                    "df/dR at ({iz},{ir}) = {}",
                    df_dr[[iz, ir]]
                );
                assert!(
                    (df_dz[[iz, ir]] - 3.0).abs() < 1e-10,
                    "df/dZ at ({iz},{ir}) = {}",
                    df_dz[[iz, ir]]
                );
            }
        }
    }
}

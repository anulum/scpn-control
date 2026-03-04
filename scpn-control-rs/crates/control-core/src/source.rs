// ──────────────────────────────────────────────────────────────────────
// SCPN Control — Rust Crate
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: MIT OR Apache-2.0
// ──────────────────────────────────────────────────────────────────────
//! Non-linear source term calculation for Grad-Shafranov equilibrium.

use ndarray::{Array1, Array2};

use control_types::config::ProfileParams;
use control_types::error::{FusionError, FusionResult};
use control_types::state::Grid2D;

/// Bilinear interpolation of a 2D field.
pub fn interpolate_2d(
    field: &Array2<f64>,
    r_axis: &Array1<f64>,
    z_axis: &Array1<f64>,
    r: f64,
    z: f64,
) -> FusionResult<f64> {
    let nr = r_axis.len();
    let nz = z_axis.len();

    let ir = match r_axis
        .iter()
        .position(|&x| x >= r)
        .map(|idx| if idx == 0 { 0 } else { idx - 1 })
    {
        Some(idx) => idx.min(nr - 2),
        None => return Err(FusionError::ConfigError(format!("R={r} outside grid"))),
    };

    let iz = match z_axis
        .iter()
        .position(|&x| x >= z)
        .map(|idx| if idx == 0 { 0 } else { idx - 1 })
    {
        Some(idx) => idx.min(nz - 2),
        None => return Err(FusionError::ConfigError(format!("Z={z} outside grid"))),
    };

    let dr = r_axis[ir + 1] - r_axis[ir];
    let dz = z_axis[iz + 1] - z_axis[iz];
    let fr = (r - r_axis[ir]) / dr;
    let fz = (z - z_axis[iz]) / dz;

    let v00 = field[[iz, ir]];
    let v10 = field[[iz, ir + 1]];
    let v01 = field[[iz + 1, ir]];
    let v11 = field[[iz + 1, ir + 1]];

    let v0 = v00 * (1.0 - fr) + v10 * fr;
    let v1 = v01 * (1.0 - fr) + v11 * fr;

    Ok(v0 * (1.0 - fz) + v1 * fz)
}

/// Update plasma current density using a simplified profile model.
pub fn update_plasma_source_nonlinear(
    psi: &Array2<f64>,
    j_phi: &mut Array2<f64>,
    grid: &Grid2D,
    psi_axis: f64,
    psi_boundary: f64,
) -> FusionResult<()> {
    if psi.nrows() != grid.nz || psi.ncols() != grid.nr {
        return Err(FusionError::ConfigError(format!(
            "source update psi shape mismatch: expected ({}, {}), got ({}, {})",
            grid.nz,
            grid.nr,
            psi.nrows(),
            psi.ncols()
        )));
    }

    let nz = grid.nz;
    let nr = grid.nr;

    if psi.iter().any(|v| !v.is_finite()) || grid.r.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::ConfigError(
            "source update inputs must be finite".to_string(),
        ));
    }

    if !psi_axis.is_finite() || !psi_boundary.is_finite() {
        return Err(FusionError::ConfigError(
            "source update psi_axis/psi_boundary must be finite".to_string(),
        ));
    }

    let psi_range = psi_boundary - psi_axis;
    let denom = if psi_range.abs() < 1e-12 {
        1e-12
    } else {
        psi_range
    };

    for iz in 0..nz {
        for ir in 0..nr {
            let psi_val = psi[[iz, ir]];
            let psi_norm = (psi_val - psi_axis) / denom;

            // Only inside plasma (0 ≤ ψ_norm < 1)
            if (0.0..1.0).contains(&psi_norm) {
                let profile = 1.0 - psi_norm;
                let r = grid.r_at(iz, ir);
                if r <= 0.0 {
                    return Err(FusionError::ConfigError(format!(
                        "source update requires R > 0 inside plasma at ({iz}, {ir}), got {r}"
                    )));
                }
                // J_phi = (1 - ψ_norm) * R
                j_phi[[iz, ir]] = profile * r;
            } else {
                j_phi[[iz, ir]] = 0.0;
            }
        }
    }

    Ok(())
}

/// Update plasma current density using full mTanh profiles for P' and FF'.
#[allow(clippy::too_many_arguments)]
pub fn update_plasma_source_with_profiles(
    psi: &Array2<f64>,
    j_phi: &mut Array2<f64>,
    grid: &Grid2D,
    psi_axis: f64,
    psi_boundary: f64,
    params_p: &ProfileParams,
    params_ff: &ProfileParams,
    mu0: f64,
    i_target: f64,
) -> FusionResult<()> {
    let nz = grid.nz;
    let nr = grid.nr;
    let psi_range = psi_boundary - psi_axis;
    let denom = if psi_range.abs() < 1e-12 {
        1e-12
    } else {
        psi_range
    };

    let mut raw = Array2::zeros((nz, nr));
    let mut inside = vec![];

    for iz in 0..nz {
        for ir in 0..nr {
            let psi_n = (psi[[iz, ir]] - psi_axis) / denom;
            if (0.0..1.0).contains(&psi_n) {
                let r = grid.r_at(iz, ir).max(1e-6);
                let p = mtanh_profile(psi_n, params_p);
                let ff = mtanh_profile(psi_n, params_ff);

                // j_phi = SOURCE_BETA_MIX * R * P' + (1 - SOURCE_BETA_MIX) * (FF' / (mu0 * R))
                raw[[iz, ir]] = 0.5 * r * p + 0.5 * (ff / (mu0 * r));
                inside.push((iz, ir));
            } else {
                raw[[iz, ir]] = 0.0;
            }
        }
    }

    // Normalize to target current
    let i_raw = raw.iter().sum::<f64>() * grid.dr * grid.dz;
    if i_raw.abs() > 1e-14 {
        let scale = i_target / i_raw;
        j_phi.assign(&(raw * scale));
    } else {
        j_phi.fill(0.0);
    }

    Ok(())
}

/// Modified tanh profile (mTanh) for H-mode pedestals.
pub fn mtanh_profile(psi_norm: f64, params: &ProfileParams) -> f64 {
    let w = params.ped_width.abs().max(1e-8);
    let y = (params.ped_top - psi_norm) / w;
    let tanh_y = y.tanh();

    let core = if psi_norm < params.ped_top {
        params.core_alpha * (1.0 - (psi_norm / params.ped_top).powi(2))
    } else {
        0.0
    };

    0.5 * params.ped_height * (1.0 + tanh_y) + core
}

/// Partial derivatives of the mTanh profile with respect to its parameters.
pub fn mtanh_profile_derivatives(psi_norm: f64, params: &ProfileParams) -> [f64; 4] {
    let w = params.ped_width.abs().max(1e-8);
    let y = (params.ped_top - psi_norm) / w;
    let tanh_y = y.tanh();
    let sech2_y = 1.0 - tanh_y * tanh_y;

    // d(mTanh)/d(ped_top)
    let mut d_ped_top = 0.5 * params.ped_height * sech2_y / w;
    if psi_norm < params.ped_top && params.ped_top.abs() > 1e-8 {
        d_ped_top += 2.0 * params.core_alpha * psi_norm.powi(2) / params.ped_top.powi(3);
    }

    // d(mTanh)/d(ped_width)
    let d_ped_width = 0.5 * params.ped_height * sech2_y * (-y / w);

    // d(mTanh)/d(ped_height)
    let d_ped_height = 0.5 * (1.0 + tanh_y);

    // d(mTanh)/d(core_alpha)
    let d_core_alpha = if psi_norm < params.ped_top {
        1.0 - (psi_norm / params.ped_top).powi(2)
    } else {
        0.0
    };

    [d_ped_top, d_ped_width, d_ped_height, d_core_alpha]
}

/// Derivative of mTanh profile with respect to psi_norm.
pub fn mtanh_profile_dpsi_norm(
    psi_norm: f64,
    params: &ProfileParams,
    label: &str,
) -> FusionResult<f64> {
    let w = params.ped_width.abs().max(1e-8);
    let y = (params.ped_top - psi_norm) / w;
    let tanh_y = y.tanh();
    let sech2_y = 1.0 - tanh_y * tanh_y;

    let d_ped = -0.5 * params.ped_height * sech2_y / w;

    let d_core = if psi_norm < params.ped_top {
        -2.0 * params.core_alpha * psi_norm / (params.ped_top * params.ped_top)
    } else {
        0.0
    };

    let res = d_ped + d_core;
    if !res.is_finite() {
        return Err(FusionError::ConfigError(format!(
            "{label} gradient became non-finite"
        )));
    }
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolation_center() {
        let r_axis = Array1::linspace(0.0, 10.0, 11);
        let z_axis = Array1::linspace(0.0, 10.0, 11);
        let mut field = Array2::zeros((11, 11));
        field[[5, 5]] = 1.0;
        field[[5, 6]] = 1.0;
        field[[6, 5]] = 1.0;
        field[[6, 6]] = 1.0;

        let val = interpolate_2d(&field, &r_axis, &z_axis, 5.5, 5.5).unwrap();
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_source_update_zero_outside() {
        let grid = Grid2D::new(17, 17, 1.0, 9.0, -5.0, 5.0);
        let mut psi = Array2::zeros((17, 17));
        psi.fill(2.0); // Outside boundary
        let mut j_phi = Array2::zeros((17, 17));

        update_plasma_source_nonlinear(&psi, &mut j_phi, &grid, 0.0, 1.0).unwrap();
        assert!(j_phi.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_mtanh_at_pedestal() {
        let params = ProfileParams {
            ped_top: 0.9,
            ped_width: 0.05,
            ped_height: 1.0,
            core_alpha: 0.0,
        };
        // At ped_top, tanh(0) = 0, so value should be 0.5 * height
        let val = mtanh_profile(0.9, &params);
        assert!((val - 0.5).abs() < 1e-8);
    }
}

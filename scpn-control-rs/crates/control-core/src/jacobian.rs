// ─────────────────────────────────────────────────────────────────────
// SCPN Control — Analytical Jacobian Utilities
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: MIT OR Apache-2.0
// ─────────────────────────────────────────────────────────────────────
//! Analytical and finite-difference Jacobians for profile fitting.
//!
//! The forward model used in this crate-level inverse module maps probe-space
//! normalized flux samples to synthetic measurements:
//!   m_i = p(psi_i) + ff(psi_i)
//! where p and ff are mTanh profiles with independent parameter sets.

use crate::source::{mtanh_profile, mtanh_profile_derivatives};
use control_types::config::ProfileParams;
use control_types::error::{FusionError, FusionResult};

fn validate_profile_params(params: &ProfileParams, label: &str) -> FusionResult<()> {
    if !params.ped_top.is_finite() || params.ped_top <= 0.0 {
        return Err(FusionError::ConfigError(format!(
            "{label} ped_top must be finite and > 0, got {}",
            params.ped_top
        )));
    }
    if !params.ped_width.is_finite() || params.ped_width <= 0.0 {
        return Err(FusionError::ConfigError(format!(
            "{label} ped_width must be finite and > 0, got {}",
            params.ped_width
        )));
    }
    if !params.ped_height.is_finite() {
        return Err(FusionError::ConfigError(format!(
            "{label} ped_height must be finite, got {}",
            params.ped_height
        )));
    }
    if !params.core_alpha.is_finite() {
        return Err(FusionError::ConfigError(format!(
            "{label} core_alpha must be finite, got {}",
            params.core_alpha
        )));
    }
    Ok(())
}

/// Forward model used by inverse reconstruction.
pub fn forward_model_response(
    probe_psi_norm: &[f64],
    params_p: &ProfileParams,
    params_ff: &ProfileParams,
) -> FusionResult<Vec<f64>> {
    validate_profile_params(params_p, "params_p")?;
    validate_profile_params(params_ff, "params_ff")?;

    let mut results = Vec::with_capacity(probe_psi_norm.len());
    for &psi in probe_psi_norm {
        if !psi.is_finite() {
            return Err(FusionError::ConfigError(
                "non-finite psi_norm in probe".to_string(),
            ));
        }
        let p_val = mtanh_profile(psi, params_p);
        let ff_val = mtanh_profile(psi, params_ff);
        results.push(p_val + ff_val);
    }
    Ok(results)
}

/// Compute the 8-parameter Jacobian using analytical derivatives.
///
/// Parameters are ordered as: [ped_top_p, ped_width_p, ped_height_p, core_alpha_p,
///                             ped_top_ff, ped_width_ff, ped_height_ff, core_alpha_ff]
pub fn compute_analytical_jacobian(
    probe_psi_norm: &[f64],
    params_p: &ProfileParams,
    params_ff: &ProfileParams,
) -> FusionResult<Vec<Vec<f64>>> {
    validate_profile_params(params_p, "params_p")?;
    validate_profile_params(params_ff, "params_ff")?;

    let mut jac = Vec::with_capacity(probe_psi_norm.len());
    for &psi in probe_psi_norm {
        let dp = mtanh_profile_derivatives(psi, params_p);
        let df = mtanh_profile_derivatives(psi, params_ff);
        // Combined Jacobian rows: [dp/dx_p, df/dx_ff]
        jac.push(vec![dp[0], dp[1], dp[2], dp[3], df[0], df[1], df[2], df[3]]);
    }
    Ok(jac)
}

/// Compute the 8-parameter Jacobian using central finite differences.
pub fn compute_fd_jacobian(
    probe_psi_norm: &[f64],
    params_p: &ProfileParams,
    params_ff: &ProfileParams,
    eps: f64,
) -> FusionResult<Vec<Vec<f64>>> {
    let n_probes = probe_psi_norm.len();
    let mut jac = vec![vec![0.0; 8]; n_probes];

    let mut p_pert = *params_p;
    let mut ff_pert = *params_ff;

    for i in 0..4 {
        // Perturb p-parameters
        let original = match i {
            0 => params_p.ped_top,
            1 => params_p.ped_width,
            2 => params_p.ped_height,
            _ => params_p.core_alpha,
        };

        match i {
            0 => p_pert.ped_top = original + eps,
            1 => p_pert.ped_width = original + eps,
            2 => p_pert.ped_height = original + eps,
            _ => p_pert.core_alpha = original + eps,
        }
        let plus = forward_model_response(probe_psi_norm, &p_pert, params_ff)?;

        match i {
            0 => p_pert.ped_top = original - eps,
            1 => p_pert.ped_width = original - eps,
            2 => p_pert.ped_height = original - eps,
            _ => p_pert.core_alpha = original - eps,
        }
        let minus = forward_model_response(probe_psi_norm, &p_pert, params_ff)?;

        for j in 0..n_probes {
            jac[j][i] = (plus[j] - minus[j]) / (2.0 * eps);
        }
        // Reset
        p_pert = *params_p;

        // Perturb ff-parameters
        let original_ff = match i {
            0 => params_ff.ped_top,
            1 => params_ff.ped_width,
            2 => params_ff.ped_height,
            _ => params_ff.core_alpha,
        };

        match i {
            0 => ff_pert.ped_top = original_ff + eps,
            1 => ff_pert.ped_width = original_ff + eps,
            2 => ff_pert.ped_height = original_ff + eps,
            _ => ff_pert.core_alpha = original_ff + eps,
        }
        let plus_ff = forward_model_response(probe_psi_norm, params_p, &ff_pert)?;

        match i {
            0 => ff_pert.ped_top = original_ff - eps,
            1 => ff_pert.ped_width = original_ff - eps,
            2 => ff_pert.ped_height = original_ff - eps,
            _ => ff_pert.core_alpha = original_ff - eps,
        }
        let minus_ff = forward_model_response(probe_psi_norm, params_p, &ff_pert)?;

        for j in 0..n_probes {
            jac[j][i + 4] = (plus_ff[j] - minus_ff[j]) / (2.0 * eps);
        }
        // Reset
        ff_pert = *params_ff;
    }

    Ok(jac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jacobian_dimensions() {
        let probes = vec![0.1, 0.5, 0.9];
        let p = ProfileParams::default();
        let ff = ProfileParams::default();
        let jac = compute_analytical_jacobian(&probes, &p, &ff).unwrap();
        assert_eq!(jac.len(), 3);
        assert_eq!(jac[0].len(), 8);
    }

    #[test]
    fn test_fd_vs_analytical_parity() {
        let probes = vec![0.2, 0.8];
        let p = ProfileParams::default();
        let ff = ProfileParams::default();
        let jac_ana = compute_analytical_jacobian(&probes, &p, &ff).unwrap();
        let jac_fd = compute_fd_jacobian(&probes, &p, &ff, 1e-6).unwrap();

        for i in 0..2 {
            for j in 0..8 {
                let diff = (jac_ana[i][j] - jac_fd[i][j]).abs();
                assert!(
                    diff < 1e-5,
                    "Jacobian mismatch at ({i}, {j}): ana={}, fd={}",
                    jac_ana[i][j],
                    jac_fd[i][j]
                );
            }
        }
    }
}

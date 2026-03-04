// ─────────────────────────────────────────────────────────────────────
// SCPN Control — Ignition
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: MIT OR Apache-2.0
// ─────────────────────────────────────────────────────────────────────
//! Ignition physics: Bosch-Hale D-T reaction rate and thermodynamics.
//!
//! Port of `fusion_ignition_sim.py` lines 14-107.
//! Calculates fusion power, alpha heating, and Q-factor.

use crate::kernel::FusionKernel;
use control_types::constants::E_FUSION_DT;
use control_types::error::{FusionError, FusionResult};
pub use control_types::state::ThermodynamicsResult;

/// Peak density [m⁻³]. Python line 56.
const N_PEAK: f64 = 1.0e20;

/// Peak temperature [keV]. Python line 57.
const T_PEAK_KEV: f64 = 20.0;

/// Density profile exponent. Python line 63.
const DENSITY_EXPONENT: f64 = 0.5;

/// Temperature profile exponent. Python line 64.
const TEMPERATURE_EXPONENT: f64 = 1.0;

/// Alpha particle energy fraction (3.5/17.6 MeV ≈ 0.199).
/// Python line 82: `P_alpha = P_fusion * 0.2`
const ALPHA_FRACTION: f64 = 0.2;

/// Energy confinement time [s]. Python line 88.
const TAU_E: f64 = 3.0;

/// keV to Joules conversion: 1 keV = 1000 eV × 1.602e-19 J/eV
const KEV_TO_JOULES: f64 = 1.602e-16;

/// Bosch-Hale D-T fusion reaction rate ⟨σv⟩ in m³/s.
///
/// Uses NRL Plasma Formulary approximation:
///   σv = 3.68e-18 / T^(2/3) × exp(-19.94 / T^(1/3))
///
/// Valid for T < 100 keV. Caller must provide finite T > 0.
pub fn bosch_hale_dt(t_kev: f64) -> FusionResult<f64> {
    if !t_kev.is_finite() || t_kev <= 0.0 {
        return Err(FusionError::ConfigError(
            "ignition temperature must be finite and > 0 keV".to_string(),
        ));
    }
    let t13 = t_kev.powf(1.0 / 3.0);
    let sigv = (3.68e-18 / t_kev.powf(2.0 / 3.0)) * (-19.94 / t13).exp();
    Ok(sigv)
}

/// Calculate fusion performance metrics from current equilibrium state.
///
/// Integrates reaction rate and thermal energy over the plasma volume.
pub fn calculate_thermodynamics(
    kernel: &FusionKernel,
    p_aux_mw: f64,
) -> FusionResult<ThermodynamicsResult> {
    if !p_aux_mw.is_finite() || p_aux_mw < 0.0 {
        return Err(FusionError::ConfigError(format!(
            "p_aux must be finite and >= 0, got {p_aux_mw}"
        )));
    }

    let state = &kernel.state;
    let grid = &kernel.grid;
    let psi = &state.psi;
    let nz = grid.nz;
    let nr = grid.nr;

    let mut total_sigv = 0.0;
    let mut total_vol = 0.0;
    let mut total_therm = 0.0;

    let psi_range = state.psi_boundary - state.psi_axis;
    let denom = if psi_range.abs() < 1e-12 {
        1e-12
    } else {
        psi_range
    };

    for iz in 0..nz {
        for ir in 0..nr {
            let p_val = psi[[iz, ir]];
            let r = grid.r_at(iz, ir);
            let psi_norm = (p_val - state.psi_axis) / denom;

            // Only inside plasma boundary
            if (0.0..1.0).contains(&psi_norm) {
                // Profiles: n(ψ) = n_peak·(1 - ψ²)^0.5, T(ψ) = T_peak·(1 - ψ²)^1.0
                let p_factor = 1.0 - psi_norm.powi(2);
                let n_local = N_PEAK * p_factor.powf(DENSITY_EXPONENT);
                let t_local = T_PEAK_KEV * p_factor.powf(TEMPERATURE_EXPONENT);

                let sigv = bosch_hale_dt(t_local)?;
                let d_vol = 2.0 * std::f64::consts::PI * r * grid.dr * grid.dz;

                total_sigv += n_local * n_local * sigv * d_vol;
                total_vol += d_vol;
                total_therm += 1.5 * n_local * t_local * KEV_TO_JOULES * d_vol;
            }
        }
    }

    // P_fusion = 0.25 * n² * ⟨σv⟩ * E_fus (0.25 for 50/50 D-T mix)
    let p_fusion_mw = 0.25 * total_sigv * E_FUSION_DT * 1e-6;
    let p_alpha_mw = p_fusion_mw * ALPHA_FRACTION;
    let p_loss_mw = if total_vol > 0.0 {
        total_therm / (TAU_E * 1e6) // MW
    } else {
        0.0
    };

    let net_mw = p_alpha_mw + p_aux_mw - p_loss_mw;
    let q_factor = if p_aux_mw > 1e-3 {
        p_fusion_mw / p_aux_mw
    } else {
        0.0
    };

    Ok(ThermodynamicsResult {
        p_fusion_mw,
        p_alpha_mw,
        p_loss_mw,
        p_aux_mw,
        net_mw,
        q_factor,
        t_peak_kev: T_PEAK_KEV,
        w_thermal_mj: total_therm * 1e-6,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use control_types::state::Grid2D;
    use ndarray::Array2;
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
    fn test_bosch_hale_reference() {
        // T=10 keV should give ~1.1e-22 m³/s
        let sigv = bosch_hale_dt(10.0).unwrap();
        assert!(sigv > 1.0e-23);
        assert!(sigv < 1.0e-21);
    }

    #[test]
    fn test_thermodynamics_iter_baseline() {
        let kernel =
            FusionKernel::from_file(&config_path("validation/iter_validated_config.json")).unwrap();
        let result = calculate_thermodynamics(&kernel, 50.0).unwrap();
        assert!(result.p_fusion_mw > 0.0);
        assert!(result.q_factor > 0.0);
    }

    #[test]
    fn test_ignition_rejects_invalid_inputs() {
        let _grid = Grid2D::new(17, 17, 1.0, 9.0, -5.0, 5.0);
        let _psi = Array2::<f64>::zeros((17, 17));
        assert!(bosch_hale_dt(f64::NAN).is_err());
        assert!(bosch_hale_dt(-1.0).is_err());

        let kernel =
            FusionKernel::from_file(&config_path("validation/iter_validated_config.json")).unwrap();
        assert!(calculate_thermodynamics(&kernel, f64::NAN).is_err());
    }
}

//! Kuramoto–Sakaguchi mean-field solver with global field driver.
//!
//! Implements the phase reduction equation from SCPN Paper 27:
//!
//!   dθ_i/dt = ω_i + K·R·sin(ψ_r − θ_i − α) + ζ·sin(Ψ − θ_i)
//!
//! The ζ sin(Ψ−θ) term is the reviewer's "intention as carrier" injection
//! for plasma sync stability (no ˙Ψ equation — Ψ is exogenous).

use ndarray::Array1;
use rayon::prelude::*;

/// Kuramoto order parameter: R·exp(i·ψ_r) = (1/N)·Σ exp(i·θ_j).
/// Returns (R, ψ_r).
pub fn order_parameter(theta: &[f64]) -> (f64, f64) {
    let n = theta.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let (re, im) = theta.iter().fold((0.0_f64, 0.0_f64), |(re, im), &th| {
        (re + th.cos(), im + th.sin())
    });
    let inv_n = 1.0 / n as f64;
    let z_re = re * inv_n;
    let z_im = im * inv_n;
    let r = (z_re * z_re + z_im * z_im).sqrt();
    let psi = z_im.atan2(z_re);
    (r, psi)
}

/// Wrap phase to (-π, π].
#[inline]
pub fn wrap_phase(x: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let pi = std::f64::consts::PI;
    ((x + pi) % two_pi + two_pi) % two_pi - pi
}

/// Result of a single Kuramoto–Sakaguchi step.
pub struct KuramotoStepResult {
    pub theta: Array1<f64>,
    pub r: f64,
    pub psi_r: f64,
    pub psi_global: f64,
}

/// Single Euler step: Kuramoto–Sakaguchi + ζ sin(Ψ−θ).
///
/// When `psi_external` is `Some(Ψ)`, uses exogenous driver (no ˙Ψ).
/// When `None`, Ψ = ψ_r (mean-field mode).
pub fn kuramoto_sakaguchi_step(
    theta: &[f64],
    omega: &[f64],
    dt: f64,
    k: f64,
    alpha: f64,
    zeta: f64,
    psi_external: Option<f64>,
) -> KuramotoStepResult {
    let n = theta.len();
    assert_eq!(n, omega.len(), "theta and omega length mismatch");

    let (r, psi_r) = order_parameter(theta);
    let psi_global = psi_external.unwrap_or(psi_r);

    let kr_sin_base = k * r;

    let mut theta_out = Array1::zeros(n);

    // Hot loop — this is the sub-ms target
    theta_out
        .as_slice_mut()
        .unwrap()
        .par_chunks_mut(64)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let base = chunk_idx * 64;
            for (local_i, val) in chunk.iter_mut().enumerate() {
                let i = base + local_i;
                let th = theta[i];
                let om = omega[i];
                let mut dth = om + kr_sin_base * (psi_r - th - alpha).sin();
                if zeta != 0.0 {
                    dth += zeta * (psi_global - th).sin();
                }
                *val = wrap_phase(th + dt * dth);
            }
        });

    KuramotoStepResult {
        theta: theta_out,
        r,
        psi_r,
        psi_global,
    }
}

/// Run N steps and return final state + R trajectory.
#[allow(clippy::too_many_arguments)]
pub fn kuramoto_sakaguchi_run(
    theta_init: &[f64],
    omega: &[f64],
    n_steps: usize,
    dt: f64,
    k: f64,
    alpha: f64,
    zeta: f64,
    psi_external: Option<f64>,
) -> (Array1<f64>, Vec<f64>) {
    let mut theta = theta_init.to_vec();
    let mut r_hist = Vec::with_capacity(n_steps);

    for _ in 0..n_steps {
        let res = kuramoto_sakaguchi_step(&theta, omega, dt, k, alpha, zeta, psi_external);
        theta = res.theta.to_vec();
        r_hist.push(res.r);
    }

    (Array1::from_vec(theta), r_hist)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_parameter_synced() {
        let theta = vec![0.0; 100];
        let (r, psi) = order_parameter(&theta);
        assert!((r - 1.0).abs() < 1e-12);
        assert!(psi.abs() < 1e-12);
    }

    #[test]
    fn test_order_parameter_range() {
        let theta: Vec<f64> = (0..200).map(|i| i as f64 * 0.1).collect();
        let (r, _) = order_parameter(&theta);
        assert!(r >= 0.0);
        assert!(r <= 1.0);
    }

    #[test]
    fn test_wrap_phase_identity() {
        let x = 1.5;
        assert!((wrap_phase(x) - x).abs() < 1e-12);
    }

    #[test]
    fn test_wrap_phase_large() {
        let x = 7.0 * std::f64::consts::PI;
        let w = wrap_phase(x);
        assert!(w >= -std::f64::consts::PI);
        assert!(w <= std::f64::consts::PI);
    }

    #[test]
    fn test_step_preserves_count() {
        let theta = vec![0.1, 0.2, 0.3, 0.4];
        let omega = vec![1.0, 1.0, 1.0, 1.0];
        let res = kuramoto_sakaguchi_step(&theta, &omega, 0.01, 2.0, 0.0, 0.0, None);
        assert_eq!(res.theta.len(), 4);
    }

    #[test]
    fn test_zeta_pulls_toward_psi() {
        let n = 100;
        let theta: Vec<f64> = (0..n)
            .map(|i| -std::f64::consts::PI + (i as f64) * 0.06)
            .collect();
        let omega = vec![0.0; n];
        let psi_target = 0.5;

        let (final_theta, _) =
            kuramoto_sakaguchi_run(&theta, &omega, 500, 0.01, 0.0, 0.0, 3.0, Some(psi_target));

        let spread: f64 = final_theta
            .iter()
            .map(|&th| (th - psi_target).sin().powi(2))
            .sum::<f64>()
            / n as f64;
        assert!(spread < 0.1, "spread = {spread}, expected < 0.1");
    }

    #[test]
    fn test_run_returns_trajectory() {
        let theta = vec![0.0; 50];
        let omega: Vec<f64> = (0..50).map(|i| 0.01 * i as f64).collect();
        let (final_th, r_hist) =
            kuramoto_sakaguchi_run(&theta, &omega, 100, 0.01, 2.0, 0.0, 0.0, None);
        assert_eq!(final_th.len(), 50);
        assert_eq!(r_hist.len(), 100);
    }
}

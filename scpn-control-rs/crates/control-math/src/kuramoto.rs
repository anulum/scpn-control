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

/// Lyapunov stability candidate V(t) = (1/N) Σ (1 − cos(θ_i − Ψ)).
/// Returns (V, dV/dt_approx) where dV/dt is finite-differenced from last two V values.
pub fn lyapunov_v(theta: &[f64], psi: f64) -> f64 {
    let n = theta.len();
    if n == 0 {
        return 0.0;
    }
    theta.iter().map(|&th| 1.0 - (th - psi).cos()).sum::<f64>() / n as f64
}

/// Run N steps with Lyapunov tracking.
/// Returns (final_theta, r_hist, v_hist, lyapunov_exponent).
/// Lyapunov exponent λ = (1/T) · ln(V_final / V_initial).
/// λ < 0 ⟹ stable convergence toward Ψ.
#[allow(clippy::too_many_arguments)]
pub fn kuramoto_run_lyapunov(
    theta_init: &[f64],
    omega: &[f64],
    n_steps: usize,
    dt: f64,
    k: f64,
    alpha: f64,
    zeta: f64,
    psi_external: Option<f64>,
) -> (Array1<f64>, Vec<f64>, Vec<f64>, f64) {
    let mut theta = theta_init.to_vec();
    let mut r_hist = Vec::with_capacity(n_steps);
    let mut v_hist = Vec::with_capacity(n_steps);

    for _ in 0..n_steps {
        let res = kuramoto_sakaguchi_step(&theta, omega, dt, k, alpha, zeta, psi_external);
        let psi = psi_external.unwrap_or(res.psi_r);
        let v = lyapunov_v(res.theta.as_slice().unwrap(), psi);
        theta = res.theta.to_vec();
        r_hist.push(res.r);
        v_hist.push(v);
    }

    let t_total = n_steps as f64 * dt;
    let v0 = v_hist.first().copied().unwrap_or(1.0).max(1e-15);
    let vf = v_hist.last().copied().unwrap_or(1.0).max(1e-15);
    let lyap_exp = (vf / v0).ln() / t_total;

    (Array1::from_vec(theta), r_hist, v_hist, lyap_exp)
}

/// Result of a single UPDE multi-layer tick.
pub struct UpdeTick {
    /// Updated per-layer phases (concatenated: L × N_per).
    pub theta_flat: Vec<f64>,
    /// Per-layer order parameter R.
    pub r_layer: Vec<f64>,
    /// Global order parameter R.
    pub r_global: f64,
    /// Global mean phase Ψ.
    pub psi_global: f64,
    /// Per-layer Lyapunov V.
    pub v_layer: Vec<f64>,
    /// Global Lyapunov V.
    pub v_global: f64,
}

/// One multi-layer UPDE tick: per-layer Kuramoto + inter-layer Knm coupling.
///
/// `knm_flat` is the L×L coupling matrix in row-major order.
/// `theta_flat` / `omega_flat` are L × N_per concatenated layer arrays.
/// `zeta` is the per-layer ζ vector (length L).
#[allow(clippy::too_many_arguments)]
pub fn upde_tick(
    theta_flat: &[f64],
    omega_flat: &[f64],
    knm_flat: &[f64],
    zeta: &[f64],
    n_layers: usize,
    n_per: usize,
    dt: f64,
    psi_driver: f64,
    pac_gamma: f64,
) -> UpdeTick {
    assert_eq!(theta_flat.len(), n_layers * n_per);
    assert_eq!(omega_flat.len(), n_layers * n_per);
    assert_eq!(knm_flat.len(), n_layers * n_layers);
    assert_eq!(zeta.len(), n_layers);

    // Per-layer order parameters
    let mut r_layer = vec![0.0_f64; n_layers];
    let mut psi_layer = vec![0.0_f64; n_layers];
    for m in 0..n_layers {
        let start = m * n_per;
        let (r, psi) = order_parameter(&theta_flat[start..start + n_per]);
        r_layer[m] = r;
        psi_layer[m] = psi;
    }

    let psi_global = psi_driver;

    // Advance each layer
    let mut theta_out = vec![0.0_f64; n_layers * n_per];

    for m in 0..n_layers {
        let start = m * n_per;
        let z = zeta[m];

        for i in 0..n_per {
            let idx = start + i;
            let th = theta_flat[idx];
            let om = omega_flat[idx];

            // Intra-layer: K_mm * R_m * sin(ψ_m - θ)
            let k_mm = knm_flat[m * n_layers + m];
            let mut dth = om + k_mm * r_layer[m] * (psi_layer[m] - th).sin();

            // ζ sin(Ψ - θ)
            if z != 0.0 {
                dth += z * (psi_global - th).sin();
            }

            // Inter-layer coupling
            for n in 0..n_layers {
                if n == m {
                    continue;
                }
                let k_mn = knm_flat[m * n_layers + n];
                if k_mn == 0.0 {
                    continue;
                }
                let mut gain = k_mn * r_layer[n] * (psi_layer[n] - th).sin();
                // PAC gate: amplify when source layer is incoherent
                if pac_gamma != 0.0 {
                    gain *= 1.0 + pac_gamma * (1.0 - r_layer[n]);
                }
                dth += gain;
            }

            theta_out[idx] = wrap_phase(th + dt * dth);
        }
    }

    // Post-step order parameters
    let mut r_layer_out = vec![0.0_f64; n_layers];
    let mut v_layer = vec![0.0_f64; n_layers];
    for m in 0..n_layers {
        let start = m * n_per;
        let (r, _) = order_parameter(&theta_out[start..start + n_per]);
        r_layer_out[m] = r;
        v_layer[m] = lyapunov_v(&theta_out[start..start + n_per], psi_global);
    }
    let (r_global_out, _) = order_parameter(&theta_out);
    let v_global = lyapunov_v(&theta_out, psi_global);

    UpdeTick {
        theta_flat: theta_out,
        r_layer: r_layer_out,
        r_global: r_global_out,
        psi_global,
        v_layer,
        v_global,
    }
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

    #[test]
    fn test_lyapunov_v_synced_is_zero() {
        let theta = vec![0.5; 100];
        let v = lyapunov_v(&theta, 0.5);
        assert!(v.abs() < 1e-12);
    }

    #[test]
    fn test_upde_tick_shape() {
        let l = 4;
        let n = 20;
        let theta: Vec<f64> = (0..l * n).map(|i| (i as f64) * 0.05).collect();
        let omega = vec![0.0; l * n];
        // Identity-diagonal Knm (no inter-layer coupling)
        let mut knm = vec![0.0; l * l];
        for m in 0..l {
            knm[m * l + m] = 2.0;
        }
        let zeta = vec![0.5; l];
        let res = upde_tick(&theta, &omega, &knm, &zeta, l, n, 0.01, 0.3, 0.0);
        assert_eq!(res.theta_flat.len(), l * n);
        assert_eq!(res.r_layer.len(), l);
        assert_eq!(res.v_layer.len(), l);
        assert!(res.r_global >= 0.0 && res.r_global <= 1.0);
    }

    #[test]
    fn test_upde_tick_zeta_convergence() {
        let l = 4;
        let n = 30;
        let theta: Vec<f64> = (0..l * n)
            .map(|i| -std::f64::consts::PI + (i as f64) * 0.05)
            .collect();
        let omega = vec![0.0; l * n];
        let mut knm = vec![0.0; l * l];
        for m in 0..l {
            knm[m * l + m] = 1.0;
        }
        let zeta = vec![3.0; l];
        let psi = 0.5;

        let mut th = theta.clone();
        for _ in 0..300 {
            let res = upde_tick(&th, &omega, &knm, &zeta, l, n, 0.005, psi, 0.0);
            th = res.theta_flat;
        }
        let v = lyapunov_v(&th, psi);
        assert!(v < 0.3, "v = {v}, expected convergence toward Ψ");
    }

    #[test]
    fn test_lyapunov_exponent_negative_with_zeta() {
        let n = 100;
        let theta: Vec<f64> = (0..n)
            .map(|i| -std::f64::consts::PI + (i as f64) * 0.06)
            .collect();
        let omega = vec![0.0; n];
        let (_, _, v_hist, lyap_exp) =
            kuramoto_run_lyapunov(&theta, &omega, 500, 0.01, 0.0, 0.0, 3.0, Some(0.5));
        assert_eq!(v_hist.len(), 500);
        // λ < 0 ⟹ V is decreasing ⟹ stable convergence
        assert!(lyap_exp < 0.0, "lyap_exp = {lyap_exp}, expected < 0");
    }
}

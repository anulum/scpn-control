// ─────────────────────────────────────────────────────────────────────
// SCPN Control — H-infinity Robust Controller (Rust)
// © 1998–2026 Miroslav Šotek. All rights reserved.
// License: MIT OR Apache-2.0
// ─────────────────────────────────────────────────────────────────────
//! Doyle-Glover-Khargonekar H-infinity synthesis in Rust.
//!
//! Mirrors the Python `h_infinity_controller` module. Uses ndarray
//! for matrix ops; DARE solution requires ndarray-linalg (LAPACK).
//!
//! Status: types + interface defined; full DARE solver gated on
//! `ndarray-linalg` feature flag (not yet enabled).

use ndarray::{s, Array1, Array2, Axis};

/// Matrix exponential via Padé(6,6) scaling-and-squaring.
///
/// Higham 2005, "The Scaling and Squaring Method for the Matrix Exponential Revisited",
/// SIAM J. Matrix Anal. Appl. 26(4), 1179-1193.
///
/// Padé(6,6) coefficients: b_k = (12-k)! * 12! / ((12-k)! * k! * 12!)
/// simplified to the standard sequence for p=q=6.
fn matrix_exp(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "matrix_exp requires square matrix");

    if n == 0 {
        return Array2::zeros((0, 0));
    }

    // Padé(6,6) coefficients: c_k = (2p - k)! p! / ((2p)! k! (p-k)!)  for p=6
    const C: [f64; 7] = [
        1.0,                      // c0
        0.5,                      // c1 = 1/2
        0.113_636_363_636_363_63, // c2 = 5/44
        1.515_151_515_151_515e-2, // c3 = 1/66
        1.262_626_262_626_263e-3, // c4 = 5/3960
        6.313_131_313_131_313e-5, // c5 = 1/15840
        1.503_126_503_126_503e-6, // c6 = 1/665280
    ];

    // Scaling: find s such that ||A / 2^s||_inf <= 1
    let norm_a: f64 = (0..n)
        .map(|i| (0..n).map(|j| a[[i, j]].abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);

    let s = if norm_a > 0.0 {
        norm_a.log2().ceil() as u32
    } else {
        0
    };

    let scale = 2.0_f64.powi(-(s as i32));
    let a_scaled = a * scale;

    // Compute powers: A^2, A^3, ..., A^6
    let a2 = a_scaled.dot(&a_scaled);
    let a3 = a2.dot(&a_scaled);
    let a4 = a3.dot(&a_scaled);
    let a5 = a4.dot(&a_scaled);
    let a6 = a5.dot(&a_scaled);

    let eye = Array2::<f64>::eye(n);

    // U = A (c1 I + c3 A^2 + c5 A^4)
    let u_inner = &eye * C[1] + &a2 * C[3] + &a4 * C[5];
    let u = a_scaled.dot(&u_inner);

    // V = c0 I + c2 A^2 + c4 A^4 + c6 A^6
    let v = &eye * C[0] + &a2 * C[2] + &a4 * C[4] + &a6 * C[6];

    // expm(A_scaled) ≈ (V - U)^{-1} (V + U)
    let numer = &v + &u;
    let denom = &v - &u;

    // result = denom^{-1} * numer
    use ndarray_linalg::Inverse;
    let denom_inv = denom
        .inv()
        .expect("Padé denominator singular in matrix_exp");
    let mut result = denom_inv.dot(&numer);

    // Squaring: result = result^{2^s}
    for _ in 0..s {
        let tmp = result.clone();
        result = tmp.dot(&result);
    }

    result
}

/// ZOH discretization via matrix exponential.
///
/// Constructs the augmented matrix [A B; 0 0] * dt and extracts (Ad, Bd)
/// from its exponential, matching the Python `_zoh_discretize` exactly.
fn zoh_discretize(a: &Array2<f64>, b: &Array2<f64>, dt: f64) -> (Array2<f64>, Array2<f64>) {
    let n = a.nrows();
    let m = b.ncols();
    let nm = n + m;

    let mut aug = Array2::<f64>::zeros((nm, nm));
    aug.slice_mut(s![..n, ..n]).assign(&(a * dt));
    aug.slice_mut(s![..n, n..]).assign(&(b * dt));
    // Bottom rows stay zero

    let e_aug = matrix_exp(&aug);

    let ad = e_aug.slice(s![..n, ..n]).to_owned();
    let bd = e_aug.slice(s![..n, n..]).to_owned();

    (ad, bd)
}

/// Result of the H-infinity gamma bisection.
#[derive(Debug, Clone)]
pub struct HInfSynthesisResult {
    pub gamma: f64,
    pub gain_k: Array1<f64>,
    pub gain_l: Array1<f64>,
    pub is_stable: bool,
    pub robust_feasible: bool,
    pub gain_margin_db: f64,
}

/// Continuous-time plant matrices for H-infinity design.
#[derive(Debug, Clone)]
pub struct HInfPlant {
    pub a: Array2<f64>,
    pub b1: Array2<f64>,
    pub b2: Array2<f64>,
    pub c1: Array2<f64>,
    pub c2: Array2<f64>,
}

impl HInfPlant {
    pub fn new(
        a: Array2<f64>,
        b1: Array2<f64>,
        b2: Array2<f64>,
        c1: Array2<f64>,
        c2: Array2<f64>,
    ) -> Result<Self, String> {
        let n = a.nrows();
        if a.ncols() != n {
            return Err("A must be square".into());
        }
        if b1.nrows() != n {
            return Err("B1 row count must match A".into());
        }
        if b2.nrows() != n {
            return Err("B2 row count must match A".into());
        }
        if c1.ncols() != n {
            return Err("C1 column count must match A".into());
        }
        if c2.ncols() != n {
            return Err("C2 column count must match A".into());
        }
        Ok(Self { a, b1, b2, c1, c2 })
    }

    pub fn n_states(&self) -> usize {
        self.a.nrows()
    }
}

/// Discrete-time observer-based H-infinity controller state.
#[derive(Debug, Clone)]
pub struct HInfController {
    pub plant: HInfPlant,
    pub gamma: f64,
    pub u_max: f64,
    state: Array1<f64>,
    cached_dt: f64,
    // Discrete gains (recomputed on dt change)
    fd: Array2<f64>,
    kd: Array1<f64>,
    ld: Array1<f64>,
}

impl HInfController {
    /// Construct an H-infinity controller with LQR-approximated gains.
    ///
    /// Uses simple pole-placement gains as initial values until the
    /// full DARE solver is available via ndarray-linalg.
    pub fn new(plant: HInfPlant, gamma: f64, u_max: f64, dt: f64) -> Self {
        let n = plant.n_states();
        // Simple LQR-like gain: K = [gamma_growth, damping_target]
        // For 2-state VDE plant: u = -k1*z - k2*dz/dt
        let kd = if n == 2 {
            let a10 = plant.a[[1, 0]];
            let a11 = plant.a[[1, 1]];
            // Place closed-loop poles at -gamma (stable)
            let k1 = a10 + gamma * gamma;
            let k2 = -a11 + 2.0 * gamma;
            Array1::from_vec(vec![k1, k2])
        } else {
            Array1::zeros(n)
        };
        // Observer gain: dual of feedback
        let ld = kd.clone();
        // ZOH discretization: expm([A B; 0 0] * dt)
        let (fd, _bd) = zoh_discretize(&plant.a, &plant.b2, dt);
        Self {
            plant,
            gamma,
            u_max,
            state: Array1::zeros(n),
            cached_dt: dt,
            fd,
            kd,
            ld,
        }
    }

    /// Step the controller: measurement y at timestep dt → control u.
    ///
    /// Recomputes DARE gains if dt changes from the cached value.
    pub fn step(&mut self, y: f64, dt: f64) -> f64 {
        if (dt - self.cached_dt).abs() > 1e-12 * dt {
            self.update_discretization(dt);
        }

        let innovation = y - self.state.dot(&self.plant.c2.row(0));
        self.state = self.fd.dot(&self.state) + &self.ld * innovation;

        let u_raw = -self.kd.dot(&self.state);
        u_raw.clamp(-self.u_max, self.u_max)
    }

    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }

    fn update_discretization(&mut self, dt: f64) {
        use control_math::linalg::solve_dare;
        use ndarray_linalg::Inverse;

        let n = self.plant.a.nrows();

        // ZOH discretization via matrix exponential (Higham 2005)
        let (ad, bd_u) = zoh_discretize(&self.plant.a, &self.plant.b2, dt);
        let (_, bd_w) = zoh_discretize(&self.plant.a, &self.plant.b1, dt);

        self.fd = ad.clone();

        // 2. Solve Feedback DARE (Discrete H-infinity)
        // H-inf DARE has augmented B and R
        use ndarray::concatenate;
        let b_aug = concatenate![Axis(1), bd_u, bd_w.mapv(|v| v / self.gamma)];
        let r_aug = Array2::from_diag(&Array1::from_vec(vec![1.0, -1.0]));
        let q = self.plant.c1.t().dot(&self.plant.c1);

        if let Ok(x) = solve_dare(&ad, &b_aug, &q, &r_aug) {
            // kd = (R + B^T X B)^-1 B^T X A
            let r_u = Array2::eye(1);
            let k_term = (&r_u + &bd_u.t().dot(&x).dot(&bd_u)).inv().unwrap();
            let kd_mat = k_term.dot(&bd_u.t()).dot(&x).dot(&ad);
            self.kd = kd_mat.row(0).to_owned();
        }

        // 3. Solve Observer DARE (Dual)
        let q_obs = &bd_w.dot(&bd_w.t()) + Array2::eye(n) * 1e-6;
        let r_obs = Array2::eye(1);
        if let Ok(y) = solve_dare(
            &ad.t().to_owned(),
            &self.plant.c2.t().to_owned(),
            &q_obs,
            &r_obs,
        ) {
            // ld = A Y C^T (R + C Y C^T)^-1
            let r_y = Array2::eye(1);
            let l_term = (&r_y + &self.plant.c2.dot(&y).dot(&self.plant.c2.t()))
                .inv()
                .unwrap();
            let ld_mat = ad.dot(&y).dot(&self.plant.c2.t()).dot(&l_term);
            self.ld = ld_mat.column(0).to_owned();
        }

        self.cached_dt = dt;
    }
}

/// Factory: 2-state vertical-stability plant (eigenvalue at +gamma_growth).
pub fn radial_robust_plant(gamma_growth: f64, damping: f64) -> HInfPlant {
    let a = Array2::from_shape_vec(
        (2, 2),
        vec![0.0, 1.0, gamma_growth * gamma_growth, -damping],
    )
    .unwrap();
    let b1 = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
    let b2 = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
    let c1 = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap();
    let c2 = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap();
    HInfPlant { a, b1, b2, c1, c2 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plant_construction() {
        let plant = radial_robust_plant(100.0, 10.0);
        assert_eq!(plant.n_states(), 2);
    }

    #[test]
    fn test_plant_rejects_dimension_mismatch() {
        let a = Array2::eye(2);
        let b1 = Array2::zeros((3, 1)); // wrong rows
        let b2 = Array2::zeros((2, 1));
        let c1 = Array2::zeros((1, 2));
        let c2 = Array2::zeros((1, 2));
        assert!(HInfPlant::new(a, b1, b2, c1, c2).is_err());
    }

    #[test]
    fn test_controller_saturates_at_u_max() {
        let plant = radial_robust_plant(100.0, 10.0);
        let mut ctrl = HInfController::new(plant, 1.0, 5.0, 1e-3);
        let u = ctrl.step(1e6, 1e-3);
        assert!(u.abs() <= 5.0, "output {u} exceeds u_max=5.0");
    }

    #[test]
    fn test_reset_zeroes_state() {
        let plant = radial_robust_plant(100.0, 10.0);
        let mut ctrl = HInfController::new(plant, 1.0, 10.0, 1e-3);
        ctrl.step(1.0, 1e-3);
        ctrl.reset();
        assert!(ctrl.state.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_stable_under_zero_input() {
        let plant = radial_robust_plant(100.0, 10.0);
        let mut ctrl = HInfController::new(plant, 1.0, 10.0, 1e-3);
        for _ in 0..100 {
            let u = ctrl.step(0.0, 1e-3);
            assert!(u.is_finite(), "control diverged under zero input");
        }
        assert!(ctrl.state.iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn test_matrix_exp_zero_is_identity() {
        for n in [1, 2, 4, 8] {
            let z = Array2::<f64>::zeros((n, n));
            let result = matrix_exp(&z);
            let eye = Array2::<f64>::eye(n);
            for i in 0..n {
                for j in 0..n {
                    assert!(
                        (result[[i, j]] - eye[[i, j]]).abs() < 1e-14,
                        "expm(0)[{i},{j}] = {} != {}",
                        result[[i, j]],
                        eye[[i, j]]
                    );
                }
            }
        }
    }

    #[test]
    fn test_matrix_exp_diagonal() {
        // expm(diag(d)) = diag(exp(d))
        let d = [0.5, -1.0, 2.0, 0.0];
        let n = d.len();
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            a[[i, i]] = d[i];
        }
        let result = matrix_exp(&a);
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { d[i].exp() } else { 0.0 };
                assert!(
                    (result[[i, j]] - expected).abs() < 1e-10,
                    "expm(diag)[{i},{j}] = {}, expected {}",
                    result[[i, j]],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_matrix_exp_nilpotent() {
        // A = [[0, 1]; [0, 0]], A^2 = 0
        // expm(A) = I + A = [[1, 1]; [0, 1]]
        let a = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 0.0, 0.0]).unwrap();
        let result = matrix_exp(&a);
        let expected = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 0.0, 1.0]).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (result[[i, j]] - expected[[i, j]]).abs() < 1e-14,
                    "expm(nilpotent)[{i},{j}] = {}, expected {}",
                    result[[i, j]],
                    expected[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_matrix_exp_large_norm() {
        // Scaling-and-squaring should handle ||A|| >> 1.
        // A = diag(10, -10), expm(A) = diag(e^10, e^-10)
        let a = Array2::from_shape_vec((2, 2), vec![10.0, 0.0, 0.0, -10.0]).unwrap();
        let result = matrix_exp(&a);
        let rtol = 1e-8;
        assert!(
            ((result[[0, 0]] - 10.0_f64.exp()) / 10.0_f64.exp()).abs() < rtol,
            "expm(10) = {}, expected {}",
            result[[0, 0]],
            10.0_f64.exp()
        );
        assert!(
            ((result[[1, 1]] - (-10.0_f64).exp()) / (-10.0_f64).exp()).abs() < rtol,
            "expm(-10) = {}, expected {}",
            result[[1, 1]],
            (-10.0_f64).exp()
        );
        assert!(result[[0, 1]].abs() < 1e-12);
        assert!(result[[1, 0]].abs() < 1e-12);
    }

    #[test]
    fn test_zoh_matches_euler_small_dt() {
        // For small dt, ZOH ≈ Euler: Ad ≈ I + A*dt, Bd ≈ B*dt
        let a = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, -4.0, -2.0]).unwrap();
        let b = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        let dt = 1e-5;

        let (ad, bd) = zoh_discretize(&a, &b, dt);
        let ad_euler: Array2<f64> = Array2::eye(2) + &a * dt;
        let bd_euler: Array2<f64> = &b * dt;

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (ad[[i, j]] - ad_euler[[i, j]]).abs() < 1e-9,
                    "Ad[{i},{j}]: zoh={}, euler={}",
                    ad[[i, j]],
                    ad_euler[[i, j]]
                );
            }
            assert!(
                (bd[[i, 0]] - bd_euler[[i, 0]]).abs() < 1e-9,
                "Bd[{i},0]: zoh={}, euler={}",
                bd[[i, 0]],
                bd_euler[[i, 0]]
            );
        }
    }

    #[test]
    fn test_zoh_known_first_order() {
        // 1st-order system: dx/dt = -a*x + b*u
        // Ad = exp(-a*dt), Bd = (1 - exp(-a*dt))/a * b
        let a_val = 2.0;
        let b_val = 3.0;
        let dt = 0.1;

        let a = Array2::from_shape_vec((1, 1), vec![-a_val]).unwrap();
        let b = Array2::from_shape_vec((1, 1), vec![b_val]).unwrap();

        let (ad, bd) = zoh_discretize(&a, &b, dt);

        let ad_exact = (-a_val * dt).exp();
        let bd_exact = (1.0 - (-a_val * dt).exp()) / a_val * b_val;

        assert!(
            (ad[[0, 0]] - ad_exact).abs() < 1e-12,
            "Ad: got {}, expected {}",
            ad[[0, 0]],
            ad_exact
        );
        assert!(
            (bd[[0, 0]] - bd_exact).abs() < 1e-12,
            "Bd: got {}, expected {}",
            bd[[0, 0]],
            bd_exact
        );
    }
}

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

use ndarray::{Array1, Array2};

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
        // Euler discretization: Fd = I + A*dt
        let fd = Array2::eye(n) + &plant.a * dt;
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
        use ndarray::stack;
        use ndarray::Axis;
        use ndarray_linalg::Inverse;

        let n = self.plant.a.nrows();
        
        // 1. ZOH Discretization: Ad = exp(A*dt), Bd = (Ad - I) A^-1 B
        // Simple Euler approximation for now (matches Python ref)
        let ad = Array2::eye(n) + &self.plant.a * dt;
        let bd_u = &self.plant.b2 * dt;
        let bd_w = &self.plant.b1 * dt;
        
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
        if let Ok(y) = solve_dare(&ad.t().to_owned(), &self.plant.c2.t().to_owned(), &q_obs, &r_obs) {
            // ld = A Y C^T (R + C Y C^T)^-1
            let r_y = Array2::eye(1);
            let l_term = (&r_y + &self.plant.c2.dot(&y).dot(&self.plant.c2.t())).inv().unwrap();
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
}

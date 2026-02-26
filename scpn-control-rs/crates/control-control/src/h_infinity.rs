// ─────────────────────────────────────────────────────────────────────
// SCPN Control — H-infinity Robust Controller (Rust)
// © 1998–2026 Miroslav Šotek. All rights reserved.
// License: GNU AGPL v3 | Commercial licensing available
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

    fn update_discretization(&mut self, _dt: f64) {
        // TODO(gh-XXX): Implement ZOH + DARE discretization.
        // Requires ndarray-linalg for eigendecomposition / Schur.
        // For now, gains remain at the design-point values.
        self.cached_dt = _dt;
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
}

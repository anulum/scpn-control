// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — H-infinity controller-realization runtime (Rust).

//! Sampled runtime for an admitted continuous-time H-infinity controller.
//!
//! This crate does not claim an independent DGKF synthesis backend. The Python
//! owner validates the normalized standard plant, solves the two continuous
//! Riccati equations, checks strict feasibility and internal stability, and
//! supplies the central controller realization `(Ak, Bk, Ck)`. Rust executes
//! that exact realization using zero-order-hold discretization. This division
//! keeps Python and Rust latency measurements on the same algorithm and plant.
//!
//! The H-infinity bound applies to the unsaturated linear continuous-time
//! interconnection. Output clipping and sampled-data execution are separate
//! runtime boundaries and do not inherit that theorem automatically.

use ndarray::{s, Array1, Array2};
use ndarray_linalg::Solve;

fn all_finite(matrix: &Array2<f64>) -> bool {
    matrix.iter().all(|value| value.is_finite())
}

fn solve_matrix(left: &Array2<f64>, right: &Array2<f64>) -> Result<Array2<f64>, String> {
    if left.nrows() != left.ncols() || left.nrows() != right.nrows() {
        return Err("linear-system dimensions are incompatible".into());
    }
    let mut solution = Array2::<f64>::zeros(right.raw_dim());
    for column in 0..right.ncols() {
        let solved = left
            .solve_into(right.column(column).to_owned())
            .map_err(|error| format!("linear solve failed: {error}"))?;
        solution.column_mut(column).assign(&solved);
    }
    Ok(solution)
}

/// Matrix exponential via Padé(6,6) scaling and squaring.
///
/// The Padé denominator is solved column by column; no explicit matrix inverse
/// is formed. Non-finite inputs and results fail closed.
pub(crate) fn try_matrix_exp(matrix: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err("matrix_exp requires a square matrix".into());
    }
    if !all_finite(matrix) {
        return Err("matrix_exp requires finite entries".into());
    }
    if n == 0 {
        return Ok(Array2::zeros((0, 0)));
    }

    const COEFFICIENTS: [f64; 7] = [
        1.0,
        0.5,
        0.113_636_363_636_363_63,
        1.515_151_515_151_515e-2,
        1.262_626_262_626_263e-3,
        6.313_131_313_131_313e-5,
        1.503_126_503_126_503e-6,
    ];

    let infinity_norm = (0..n)
        .map(|row| {
            (0..n)
                .map(|column| matrix[[row, column]].abs())
                .sum::<f64>()
        })
        .fold(0.0_f64, f64::max);
    let squarings = if infinity_norm > 0.0 {
        (infinity_norm.log2().ceil().max(0.0) as u32).min(1100)
    } else {
        0
    };
    let scaled = matrix * 2.0_f64.powi(-(squarings as i32));
    let squared = scaled.dot(&scaled);
    let cubed = squared.dot(&scaled);
    let fourth = cubed.dot(&scaled);
    let fifth = fourth.dot(&scaled);
    let sixth = fifth.dot(&scaled);
    let identity = Array2::<f64>::eye(n);

    let numerator_odd = scaled.dot(
        &(&identity * COEFFICIENTS[1] + &squared * COEFFICIENTS[3] + &fourth * COEFFICIENTS[5]),
    );
    let numerator_even = &identity * COEFFICIENTS[0]
        + &squared * COEFFICIENTS[2]
        + &fourth * COEFFICIENTS[4]
        + &sixth * COEFFICIENTS[6];
    let mut result = solve_matrix(
        &(&numerator_even - &numerator_odd),
        &(&numerator_even + &numerator_odd),
    )?;

    for _ in 0..squarings {
        result = result.dot(&result);
        if !all_finite(&result) {
            return Err("matrix exponential overflowed to a non-finite result".into());
        }
    }
    Ok(result)
}

fn zoh_discretize(
    state_matrix: &Array2<f64>,
    input_matrix: &Array2<f64>,
    dt: f64,
) -> Result<(Array2<f64>, Array2<f64>), String> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err("dt must be finite and strictly positive".into());
    }
    let n_states = state_matrix.nrows();
    if state_matrix.ncols() != n_states || input_matrix.nrows() != n_states {
        return Err("controller realization dimensions are incompatible".into());
    }
    let n_inputs = input_matrix.ncols();
    let mut augmented = Array2::<f64>::zeros((n_states + n_inputs, n_states + n_inputs));
    augmented
        .slice_mut(s![..n_states, ..n_states])
        .assign(&(state_matrix * dt));
    augmented
        .slice_mut(s![..n_states, n_states..])
        .assign(&(input_matrix * dt));
    let exponential = try_matrix_exp(&augmented)?;
    Ok((
        exponential.slice(s![..n_states, ..n_states]).to_owned(),
        exponential.slice(s![..n_states, n_states..]).to_owned(),
    ))
}

/// Runtime state for a pre-synthesized SISO central H-infinity controller.
#[derive(Debug, Clone)]
pub struct HInfController {
    /// Continuous-time central-controller state matrix.
    pub ak: Array2<f64>,
    /// Continuous-time measurement-injection matrix.
    pub bk: Array2<f64>,
    /// Continuous-time control-output matrix.
    pub ck: Array2<f64>,
    /// Admitted continuous-time attenuation bound, retained as provenance.
    pub gamma: f64,
    /// Symmetric runtime output limit.
    pub u_max: f64,
    state: Array1<f64>,
    cached_dt: f64,
    ak_discrete: Array2<f64>,
    bk_discrete: Array2<f64>,
}

impl HInfController {
    /// Validate and construct a sampled executor for `(Ak, Bk, Ck)`.
    pub fn new(
        ak: Array2<f64>,
        bk: Array2<f64>,
        ck: Array2<f64>,
        gamma: f64,
        u_max: f64,
        dt: f64,
    ) -> Result<Self, String> {
        let n_states = ak.nrows();
        if n_states == 0 || ak.ncols() != n_states {
            return Err("Ak must be a non-empty square matrix".into());
        }
        if bk.dim() != (n_states, 1) {
            return Err("Bk must have shape (n, 1) for the SISO runtime".into());
        }
        if ck.dim() != (1, n_states) {
            return Err("Ck must have shape (1, n) for the SISO runtime".into());
        }
        if !all_finite(&ak) || !all_finite(&bk) || !all_finite(&ck) {
            return Err("controller realization matrices must be finite".into());
        }
        if !gamma.is_finite() || gamma <= 0.0 {
            return Err("gamma must be finite and strictly positive".into());
        }
        if !u_max.is_finite() || u_max <= 0.0 {
            return Err("u_max must be finite and strictly positive".into());
        }
        let (ak_discrete, bk_discrete) = zoh_discretize(&ak, &bk, dt)?;
        Ok(Self {
            ak,
            bk,
            ck,
            gamma,
            u_max,
            state: Array1::zeros(n_states),
            cached_dt: dt,
            ak_discrete,
            bk_discrete,
        })
    }

    /// Evaluate control before advancing state from one scalar measurement.
    pub fn step(&mut self, measurement: f64, dt: f64) -> Result<f64, String> {
        if !measurement.is_finite() {
            return Err("measurement must be finite".into());
        }
        if !dt.is_finite() || dt <= 0.0 {
            return Err("dt must be finite and strictly positive".into());
        }
        if dt != self.cached_dt {
            (self.ak_discrete, self.bk_discrete) = zoh_discretize(&self.ak, &self.bk, dt)?;
            self.cached_dt = dt;
        }

        let raw_control = self.ck.row(0).dot(&self.state);
        let control = raw_control.clamp(-self.u_max, self.u_max);
        self.state =
            self.ak_discrete.dot(&self.state) + self.bk_discrete.column(0).to_owned() * measurement;
        if !self.state.iter().all(|value| value.is_finite()) {
            return Err("controller state became non-finite".into());
        }
        Ok(control)
    }

    /// Reset the dynamic controller state to zero.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }

    /// Return a copy of the current controller state for parity diagnostics.
    pub fn state(&self) -> Array1<f64> {
        self.state.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_controller(dt: f64) -> HInfController {
        HInfController::new(
            Array2::from_shape_vec((1, 1), vec![-2.0]).unwrap(),
            Array2::from_shape_vec((1, 1), vec![3.0]).unwrap(),
            Array2::from_shape_vec((1, 1), vec![4.0]).unwrap(),
            5.0,
            100.0,
            dt,
        )
        .unwrap()
    }

    #[test]
    fn exact_scalar_zoh_and_output_before_update() {
        let dt = 0.1;
        let mut controller = scalar_controller(dt);
        assert_eq!(controller.step(1.0, dt).unwrap(), 0.0);
        let expected_state = 1.5 * (1.0 - (-2.0 * dt).exp());
        assert!((controller.state()[0] - expected_state).abs() < 1.0e-12);
        assert!((controller.step(0.0, dt).unwrap() - 4.0 * expected_state).abs() < 1.0e-12);
    }

    #[test]
    fn dt_change_recomputes_exact_realization() {
        let mut controller = scalar_controller(0.1);
        controller.step(1.0, 0.2).unwrap();
        let expected_state = 1.5 * (1.0 - (-0.4_f64).exp());
        assert!((controller.state()[0] - expected_state).abs() < 1.0e-12);
    }

    #[test]
    fn saturation_and_reset_are_effective() {
        let mut controller = HInfController::new(
            Array2::from_shape_vec((1, 1), vec![-1.0]).unwrap(),
            Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
            Array2::from_shape_vec((1, 1), vec![1.0e6]).unwrap(),
            2.0,
            5.0,
            0.1,
        )
        .unwrap();
        controller.step(1.0, 0.1).unwrap();
        assert_eq!(controller.step(0.0, 0.1).unwrap(), 5.0);
        controller.reset();
        assert_eq!(controller.state()[0], 0.0);
    }

    #[test]
    fn invalid_constructor_and_step_domains_fail_closed() {
        let ak = Array2::eye(2);
        let bk = Array2::zeros((2, 1));
        let ck = Array2::zeros((1, 2));
        assert!(HInfController::new(ak.clone(), bk.clone(), ck.clone(), 0.0, 1.0, 0.1).is_err());
        assert!(HInfController::new(ak.clone(), bk.clone(), ck.clone(), 1.0, 0.0, 0.1).is_err());
        assert!(HInfController::new(ak.clone(), Array2::zeros((1, 1)), ck, 1.0, 1.0, 0.1).is_err());
        let mut controller = scalar_controller(0.1);
        assert!(controller.step(f64::NAN, 0.1).is_err());
        assert!(controller.step(0.0, 0.0).is_err());
    }

    #[test]
    fn matrix_exponential_matches_known_cases() {
        let zero = Array2::<f64>::zeros((2, 2));
        assert_eq!(
            try_matrix_exp(&zero).expect("zero matrix has a finite exponential"),
            Array2::<f64>::eye(2)
        );
        let diagonal = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, -1.0]).unwrap();
        let exponential =
            try_matrix_exp(&diagonal).expect("finite diagonal matrix has an exponential");
        assert!((exponential[[0, 0]] - 1.0_f64.exp()).abs() < 1.0e-12);
        assert!((exponential[[1, 1]] - (-1.0_f64).exp()).abs() < 1.0e-12);
        assert!(try_matrix_exp(&Array2::zeros((2, 3))).is_err());
    }
}

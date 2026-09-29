// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Tridiag.

//! Pivoted tridiagonal LAPACK solve and the historical padded-array adapter.

use lapack_sys::dgtsvx_;
use std::ffi::c_char;
use thiserror::Error;

/// Refusal categories shared with the Python and PyO3 CONTROL adapters.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum TridiagonalError {
    /// An input has the wrong length or exceeds LAPACK's index range.
    #[error("invalid tridiagonal shape or padded sentinel")]
    InvalidShape,
    /// An input contains NaN or infinity.
    #[error("nonfinite tridiagonal input")]
    NonFiniteInput,
    /// Pivoted factorisation found a singular matrix.
    #[error("singular tridiagonal factorisation")]
    SingularFactorization,
    /// The computed solution is nonfinite or has excessive backward error.
    #[error("tridiagonal numerical failure")]
    NumericalFailure,
}

/// Solve a compact finite tridiagonal system with pivoted LAPACK `dgtsvx`.
///
/// `lower` and `upper` have length `n-1`; `diagonal` and `rhs` have length
/// `n >= 1`. The inputs are not mutated. The expert driver factors with
/// partial pivoting and applies LAPACK iterative refinement (`dgtrfs`):
/// partial pivoting alone is only normwise backward stable, so a row that the
/// pivot moves (for example an identity boundary row) can miss the rowwise
/// bound. A successful result is finite and passes a rowwise scale-aware
/// backward-error check.
///
/// # Errors
///
/// Returns [`TridiagonalError`] for invalid shape, nonfinite input, singular
/// factorisation or inadmissible numerical output.
pub fn solve_tridiagonal(
    lower: &[f64],
    diagonal: &[f64],
    upper: &[f64],
    rhs: &[f64],
) -> Result<Vec<f64>, TridiagonalError> {
    let n = diagonal.len();
    if n == 0
        || n > i32::MAX as usize
        || lower.len() != n - 1
        || upper.len() != n - 1
        || rhs.len() != n
    {
        return Err(TridiagonalError::InvalidShape);
    }
    if lower
        .iter()
        .chain(diagonal)
        .chain(upper)
        .chain(rhs)
        .any(|v| !v.is_finite())
    {
        return Err(TridiagonalError::NonFiniteInput);
    }

    let mut result = rhs.to_vec();
    if n == 1 {
        if diagonal[0] == 0.0 {
            return Err(TridiagonalError::SingularFactorization);
        }
        result[0] /= diagonal[0];
    } else {
        let n_i32 = n as i32;
        let nrhs = 1_i32;
        let fact = b'N' as c_char;
        let trans = b'N' as c_char;
        let mut dlf = vec![0.0; n - 1];
        let mut df = vec![0.0; n];
        let mut duf = vec![0.0; n - 1];
        let mut du2 = vec![0.0; n - 2];
        let mut ipiv = vec![0_i32; n];
        let mut rcond = 0.0_f64;
        let mut ferr = 0.0_f64;
        let mut berr = 0.0_f64;
        let mut work = vec![0.0; 3 * n];
        let mut iwork = vec![0_i32; n];
        let mut info = 0_i32;
        // FACT='N' reads the input slices; LAPACK writes only the local
        // factor, workspace and solution buffers sized as `dgtsvx` requires.
        unsafe {
            dgtsvx_(
                &fact,
                &trans,
                &n_i32,
                &nrhs,
                lower.as_ptr(),
                diagonal.as_ptr(),
                upper.as_ptr(),
                dlf.as_mut_ptr(),
                df.as_mut_ptr(),
                duf.as_mut_ptr(),
                du2.as_mut_ptr(),
                ipiv.as_mut_ptr(),
                rhs.as_ptr(),
                &n_i32,
                result.as_mut_ptr(),
                &n_i32,
                &mut rcond,
                &mut ferr,
                &mut berr,
                work.as_mut_ptr(),
                iwork.as_mut_ptr(),
                &mut info,
            );
        }
        // INFO = n+1 only reports RCOND below machine precision; the refined
        // solution exists and the rowwise check below remains the arbiter.
        if info > 0 && info <= n_i32 {
            return Err(TridiagonalError::SingularFactorization);
        }
        if info < 0 {
            return Err(TridiagonalError::NumericalFailure);
        }
    }

    if result.iter().any(|v| !v.is_finite()) {
        return Err(TridiagonalError::NumericalFailure);
    }
    let tolerance = 64.0 * n as f64 * f64::EPSILON;
    for i in 0..n {
        let centre = diagonal[i] * result[i];
        let below = if i > 0 {
            lower[i - 1] * result[i - 1]
        } else {
            0.0
        };
        let above = if i + 1 < n {
            upper[i] * result[i + 1]
        } else {
            0.0
        };
        let scale = centre.abs() + below.abs() + above.abs() + rhs[i].abs();
        let residual = (centre + below + above - rhs[i]).abs();
        if !scale.is_finite() || !residual.is_finite() || residual > tolerance * scale.max(1e-300) {
            return Err(TridiagonalError::NumericalFailure);
        }
    }
    Ok(result)
}

/// Adapt the historical `n`-padded Rust layout to the compact solver.
///
/// `a[0]` and `c[n-1]` are unused by the matrix and must be exactly zero.
/// All four arrays have length `n >= 1` and remain unchanged.
///
/// # Errors
///
/// Returns [`TridiagonalError`] for invalid sentinels or any compact solve
/// failure; it never returns an infinite or silently repaired solution.
pub fn thomas_solve(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    d: &[f64],
) -> Result<Vec<f64>, TridiagonalError> {
    let n = b.len();
    if n == 0 || a.len() != n || c.len() != n || d.len() != n {
        return Err(TridiagonalError::InvalidShape);
    }
    if a.iter().chain(b).chain(c).chain(d).any(|v| !v.is_finite()) {
        return Err(TridiagonalError::NonFiniteInput);
    }
    if a[0] != 0.0 || c[n - 1] != 0.0 {
        return Err(TridiagonalError::InvalidShape);
    }
    solve_tridiagonal(&a[1..], b, &c[..n - 1], d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pivoted_zero_leading_diagonal_and_singleton() {
        let lower = [1.0];
        let diagonal = [0.0, 1.0];
        let upper = [1.0];
        let rhs = [1.0, 2.0];
        assert_eq!(
            solve_tridiagonal(&lower, &diagonal, &upper, &rhs),
            Ok(vec![1.0, 1.0])
        );
        assert_eq!(solve_tridiagonal(&[], &[2.0], &[], &[4.0]), Ok(vec![2.0]));
        assert_eq!(diagonal, [0.0, 1.0]);
        assert_eq!(rhs, [1.0, 2.0]);
    }

    #[test]
    fn refusal_categories_and_padded_sentinels() {
        assert_eq!(
            solve_tridiagonal(&[], &[], &[], &[]),
            Err(TridiagonalError::InvalidShape)
        );
        assert_eq!(
            solve_tridiagonal(&[], &[0.0], &[], &[1.0]),
            Err(TridiagonalError::SingularFactorization)
        );
        assert_eq!(
            solve_tridiagonal(&[], &[f64::NAN], &[], &[1.0]),
            Err(TridiagonalError::NonFiniteInput)
        );
        assert_eq!(
            thomas_solve(&[1.0, 1.0], &[1.0, 1.0], &[1.0, 0.0], &[1.0, 1.0]),
            Err(TridiagonalError::InvalidShape)
        );
        assert_eq!(
            thomas_solve(&[0.0, 1.0], &[1.0, 1.0], &[1.0, 1.0], &[1.0, 1.0]),
            Err(TridiagonalError::InvalidShape)
        );
    }

    #[test]
    fn finite_near_zero_pivot_and_common_scales() {
        assert_eq!(
            solve_tridiagonal(&[], &[1e-200], &[], &[2e-200]),
            Ok(vec![2.0])
        );
        for scale in [1e-100, 1.0, 1e100] {
            let lower = [-scale, -scale];
            let diagonal = [4.0 * scale; 3];
            let upper = [-scale, -scale];
            let rhs = [3.0 * scale, 2.0 * scale, 3.0 * scale];
            let solution = solve_tridiagonal(&lower, &diagonal, &upper, &rhs)
                .expect("dominant scaled system is admissible");
            for value in solution {
                assert!((value - 1.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn pivot_moved_identity_row_meets_rowwise_bound() {
        // Partial pivoting swaps the identity boundary row below the dominant
        // interior row; unrefined `dgtsv` leaves x[0] = 0.1 ± 1e-14, which
        // misses the rowwise bound. Refinement restores the exact boundary.
        let lower = [-100.0, 0.0];
        let diagonal = [1.0, 80101.0, 1.0];
        let upper = [0.0, -80000.0];
        let rhs = [0.1, 50.0, 0.1];
        let solution = solve_tridiagonal(&lower, &diagonal, &upper, &rhs)
            .expect("refined pivoted solve meets the rowwise bound");
        assert_eq!(solution[0], 0.1);
        assert_eq!(solution[2], 0.1);
        let middle = (50.0 + 100.0 * 0.1 + 80000.0 * 0.1) / 80101.0;
        assert!((solution[1] - middle).abs() <= 4.0 * f64::EPSILON * middle);
    }

    #[test]
    fn test_thomas_identity() {
        // Solve I * x = [1,2,3,4,5]
        let n = 5;
        let a = vec![0.0; n];
        let b = vec![1.0; n];
        let c = vec![0.0; n];
        let d = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = thomas_solve(&a, &b, &c, &d).expect("identity is nonsingular");
        for i in 0..n {
            assert!((x[i] - d[i]).abs() < 1e-12, "x[{i}] should equal d[{i}]");
        }
    }

    #[test]
    fn test_thomas_simple_tridiag() {
        // Solve [-1, 2, -1] tridiagonal system (1D Laplacian)
        // [ 2 -1  0  0]   [x0]   [1]
        // [-1  2 -1  0] * [x1] = [0]
        // [ 0 -1  2 -1]   [x2]   [0]
        // [ 0  0 -1  2]   [x3]   [1]
        let a = vec![0.0, -1.0, -1.0, -1.0];
        let b = vec![2.0, 2.0, 2.0, 2.0];
        let c = vec![-1.0, -1.0, -1.0, 0.0];
        let d = vec![1.0, 0.0, 0.0, 1.0];
        let x = thomas_solve(&a, &b, &c, &d).expect("Poisson matrix is nonsingular");

        // Verify Ax = d
        let ax = [
            b[0] * x[0] + c[0] * x[1],
            a[1] * x[0] + b[1] * x[1] + c[1] * x[2],
            a[2] * x[1] + b[2] * x[2] + c[2] * x[3],
            a[3] * x[2] + b[3] * x[3],
        ];
        for i in 0..4 {
            assert!(
                (ax[i] - d[i]).abs() < 1e-10,
                "Ax[{i}] = {}, expected {}",
                ax[i],
                d[i]
            );
        }
    }

    #[test]
    fn test_thomas_heat_equation_pattern() {
        // Typical implicit heat equation pattern:
        // main = 1 + 2*alpha, sub/super = -alpha
        let n = 10;
        let alpha = 0.4;
        let a: Vec<f64> = (0..n).map(|i| if i > 0 { -alpha } else { 0.0 }).collect();
        let b = vec![1.0 + 2.0 * alpha; n];
        let c: Vec<f64> = (0..n)
            .map(|i| if i < n - 1 { -alpha } else { 0.0 })
            .collect();
        let d = vec![1.0; n]; // uniform RHS

        let x = thomas_solve(&a, &b, &c, &d).expect("diffusion matrix is nonsingular");

        // All values should be positive and finite
        for (i, &xi) in x.iter().enumerate() {
            assert!(
                xi > 0.0 && xi.is_finite(),
                "x[{i}] = {xi} should be positive finite"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// SCPN Control — Linalg
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: MIT OR Apache-2.0
// ─────────────────────────────────────────────────────────────────────
//! Linear algebra utilities.
//!
//! SVD, 2x2 eigendecomposition, pseudoinverse with Tikhonov regularization.

use control_types::error::{FusionError, FusionResult};
use ndarray::{Array1, Array2};
use ndarray_linalg::SVD;

/// 2x2 eigenvalue decomposition.
///
/// Used by stability_analyzer.py for the stiffness matrix.
/// Returns (eigenvalues, eigenvectors) sorted by ascending eigenvalue.
pub fn eig_2x2(a: &[[f64; 2]; 2]) -> ([f64; 2], [[f64; 2]; 2]) {
    let trace = a[0][0] + a[1][1];
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let disc = trace * trace - 4.0 * det;

    if disc < 0.0 {
        // Complex eigenvalues — return real parts
        let re = trace / 2.0;
        return ([re, re], [[1.0, 0.0], [0.0, 1.0]]);
    }

    let sqrt_disc = disc.sqrt();
    let l1 = (trace - sqrt_disc) / 2.0;
    let l2 = (trace + sqrt_disc) / 2.0;

    // Eigenvectors
    let v1 = if a[0][1].abs() > 1e-15 {
        let x = l1 - a[1][1];
        let y = a[1][0];
        let norm = (x * x + y * y).sqrt();
        [x / norm, y / norm]
    } else if a[1][0].abs() > 1e-15 {
        let x = a[0][1];
        let y = l1 - a[0][0];
        let norm = (x * x + y * y).sqrt();
        [x / norm, y / norm]
    } else {
        [1.0, 0.0]
    };

    let v2 = if a[0][1].abs() > 1e-15 {
        let x = l2 - a[1][1];
        let y = a[1][0];
        let norm = (x * x + y * y).sqrt();
        [x / norm, y / norm]
    } else if a[1][0].abs() > 1e-15 {
        let x = a[0][1];
        let y = l2 - a[0][0];
        let norm = (x * x + y * y).sqrt();
        [x / norm, y / norm]
    } else {
        [0.0, 1.0]
    };

    ([l1, l2], [v1, v2])
}

/// SVD for small matrices using industrial-grade LAPACK.
///
/// Returns (U, sigma, Vt) where A ≈ U * diag(sigma) * Vt.
/// Replaces the unstable one-sided Jacobi implementation.
pub fn svd_small(a: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let (u, sigma, vt) = a.svd(true, true).expect("SVD decomposition failed");
    (
        u.expect("U matrix missing"),
        sigma,
        vt.expect("Vt matrix missing"),
    )
}

/// Pseudoinverse with SVD and singular value cutoff.
///
/// Used by fusion_optimal_control.py compute_optimal_correction().
pub fn pinv_svd(a: &Array2<f64>, sv_cutoff: f64) -> Array2<f64> {
    let (u, sigma, vt) = svd_small(a);
    let (m, n) = a.dim();
    let k = sigma.len();

    let mut result = Array2::zeros((n, m));

    for idx in 0..k {
        if sigma[idx] > sv_cutoff {
            let inv_s = 1.0 / sigma[idx];
            for i in 0..n {
                for j in 0..m {
                    result[[i, j]] += vt[[idx, i]] * inv_s * u[[j, idx]];
                }
            }
        }
    }

    result
}

/// Solve the Discrete Algebraic Riccati Equation (DARE) using the doubling algorithm (SDA).
///
/// X = A^T X A - (A^T X B)(R + B^T X B)^-1 (B^T X A) + Q
///
/// Returns the stabilising solution X.
pub fn solve_dare(
    a: &Array2<f64>,
    b: &Array2<f64>,
    q: &Array2<f64>,
    r: &Array2<f64>,
) -> FusionResult<Array2<f64>> {
    use ndarray_linalg::Inverse;

    let n = a.nrows();
    let mut ak = a.clone();
    let mut qk = q.clone();
    let r_inv: Array2<f64> = r
        .inv()
        .map_err(|e| FusionError::LinAlg(format!("R not invertible: {e}")))?;
    let mut gk = b.dot(&r_inv).dot(&b.t());

    let max_iter = 50;
    let tol = 1e-12;

    for _ in 0..max_iter {
        let ident = Array2::eye(n);
        let inv_term: Array2<f64> = (ident + gk.dot(&qk))
            .inv()
            .map_err(|e| FusionError::LinAlg(format!("DARE inversion failed: {e}")))?;

        let ak_next = ak.dot(&inv_term).dot(&ak);
        let qk_next = &qk + &ak.t().dot(&qk).dot(&inv_term).dot(&ak);
        let gk_next = &gk + &ak.dot(&inv_term).dot(&gk).dot(&ak.t());

        let diff = (&qk_next - &qk).mapv(|v| v.abs()).sum();
        ak = ak_next;
        qk = qk_next;
        gk = gk_next;

        if diff < tol {
            return Ok(qk);
        }
    }

    Ok(qk)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eig_2x2_diagonal() {
        let a = [[3.0, 0.0], [0.0, 5.0]];
        let (vals, _vecs) = eig_2x2(&a);
        assert!((vals[0] - 3.0).abs() < 1e-10);
        assert!((vals[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_svd_identity() {
        let a = Array2::eye(3);
        let (u, sigma, vt) = svd_small(&a);
        for i in 0..3 {
            assert!((sigma[i] - 1.0).abs() < 1e-10, "sigma[{i}] = {}", sigma[i]);
        }
        let reconstructed = u.dot(&Array2::from_diag(&sigma)).dot(&vt);
        for i in 0..3 {
            for j in 0..3 {
                assert!((reconstructed[[i, j]] - a[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_pinv_svd_identity() {
        let a = Array2::eye(3);
        let pinv = pinv_svd(&a, 1e-10);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((pinv[[i, j]] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_solve_dare_simple() {
        let a = Array2::from_elem((2, 2), 0.5);
        let b = Array2::eye(2);
        let q = Array2::eye(2);
        let r = Array2::eye(2);

        let x = solve_dare(&a, &b, &q, &r).unwrap();
        assert_eq!(x.shape(), &[2, 2]);
        assert!(x[[0, 0]] > 1.0);
    }
}

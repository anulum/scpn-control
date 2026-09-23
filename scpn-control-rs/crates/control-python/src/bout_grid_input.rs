// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — BOUT grid input admission.
//! Validate BOUT grid inputs before native grid construction.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Refuse degenerate coordinates and mismatched flux grid dimensions.
pub(super) fn validate_bout_grid_inputs(
    rs: &[f64],
    zs: &[f64],
    psi_shape: &[usize],
) -> PyResult<()> {
    if rs.len() < 2 || zs.len() < 2 {
        return Err(PyValueError::new_err(
            "r and z each require at least two points",
        ));
    }
    if psi_shape != [zs.len(), rs.len()] {
        return Err(PyValueError::new_err("psi shape must be (len(z), len(r))"));
    }
    for (axis, points) in [("r", rs), ("z", zs)] {
        let first = points[0];
        let last = points[points.len() - 1];
        if !first.is_finite() || !last.is_finite() || first >= last {
            return Err(PyValueError::new_err(format!(
                "{axis} endpoints must be finite and increasing"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod bout_grid_input_tests {
    use super::validate_bout_grid_inputs;
    use pyo3::exceptions::PyValueError;

    #[cfg(not(feature = "extension-module"))]
    #[test]
    fn public_generate_grid_refuses_invalid_numpy_inputs() {
        use crate::PyBoutInterface;
        use ndarray::{Array1, Array2};
        use numpy::{IntoPyArray, PyArrayMethods};

        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let interface = PyBoutInterface::new(2, 2, 2, 0.0, 1.0);
            for (psi, r, z) in [
                (
                    Array2::zeros((2, 0)),
                    Array1::zeros(0),
                    Array1::from_vec(vec![0.0, 1.0]),
                ),
                (
                    Array2::zeros((2, 1)),
                    Array1::from_vec(vec![0.0]),
                    Array1::from_vec(vec![0.0, 1.0]),
                ),
                (
                    Array2::zeros((3, 2)),
                    Array1::from_vec(vec![0.0, 1.0]),
                    Array1::from_vec(vec![0.0, 1.0]),
                ),
            ] {
                let psi = psi.into_pyarray(py);
                let r = r.into_pyarray(py);
                let z = z.into_pyarray(py);
                let error = interface
                    .generate_grid(
                        py,
                        psi.readonly(),
                        r.readonly(),
                        z.readonly(),
                        0.0,
                        1.0,
                        1.0,
                    )
                    .unwrap_err();
                assert!(error.is_instance_of::<PyValueError>(py));
            }
        });
    }

    #[test]
    fn refuses_empty_singleton_and_shape_mismatch() {
        pyo3::Python::initialize();
        for (rs, zs, shape) in [
            (vec![], vec![0.0, 1.0], vec![2, 0]),
            (vec![0.0, 1.0], vec![], vec![0, 2]),
            (vec![0.0], vec![0.0, 1.0], vec![2, 1]),
            (vec![0.0, 1.0], vec![0.0], vec![1, 2]),
            (vec![0.0, 1.0], vec![0.0, 1.0], vec![2, 3]),
        ] {
            let error = validate_bout_grid_inputs(&rs, &zs, &shape).unwrap_err();
            pyo3::Python::attach(|py| assert!(error.is_instance_of::<PyValueError>(py)));
        }
    }

    #[test]
    fn refuses_nonfinite_or_nonincreasing_endpoints() {
        pyo3::Python::initialize();
        for rs in [
            [f64::NAN, 1.0],
            [0.0, f64::INFINITY],
            [1.0, 1.0],
            [2.0, 1.0],
        ] {
            assert!(validate_bout_grid_inputs(&rs, &[0.0, 1.0], &[2, 2]).is_err());
        }
        assert!(validate_bout_grid_inputs(&[0.0, 1.0], &[1.0, 0.0], &[2, 2]).is_err());
        assert!(validate_bout_grid_inputs(&[0.0, 1.0], &[0.0, 1.0], &[2, 2]).is_ok());
    }
}

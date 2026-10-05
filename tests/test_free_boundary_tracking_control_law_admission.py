# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Free-boundary coil-correction admission tests

"""Exercise malformed coil-correction requests through the public calculation."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_control.control.free_boundary_tracking_control_law as law


def test_compute_coil_correction_fail_closed_shapes() -> None:
    """Shape and parameter validation fails closed."""
    target = np.ones(2, dtype=np.float64)
    with pytest.raises(ValueError, match="match the free-boundary target"):
        law.compute_coil_correction(
            np.ones(3),
            target_vector=target,
            objective_bias_estimate=np.zeros(2),
            control_objective_weights=np.ones(2),
            response_matrix=np.eye(2),
            response_regularization=1e-3,
            correction_limit=1.0,
            control_mask=np.ones(2),
            coil_currents=np.zeros(2),
            coil_current_limits=np.ones(2),
        )
    with pytest.raises(ValueError, match="objective_bias_estimate must match"):
        law.compute_coil_correction(
            target,
            target_vector=target,
            objective_bias_estimate=np.zeros(3),
            control_objective_weights=np.ones(2),
            response_matrix=np.eye(2),
            response_regularization=1e-3,
            correction_limit=1.0,
            control_mask=np.ones(2),
            coil_currents=np.zeros(2),
            coil_current_limits=np.ones(2),
        )
    with pytest.raises(ValueError, match="control_objective_weights must match"):
        law.compute_coil_correction(
            target,
            target_vector=target,
            objective_bias_estimate=np.zeros(2),
            control_objective_weights=np.ones(3),
            response_matrix=np.eye(2),
            response_regularization=1e-3,
            correction_limit=1.0,
            control_mask=np.ones(2),
            coil_currents=np.zeros(2),
            coil_current_limits=np.ones(2),
        )
    with pytest.raises(ValueError, match="control_mask must match"):
        law.compute_coil_correction(
            target,
            target_vector=target,
            objective_bias_estimate=np.zeros(2),
            control_objective_weights=np.ones(2),
            response_matrix=np.eye(2),
            response_regularization=1e-3,
            correction_limit=1.0,
            control_mask=np.ones(3),
            coil_currents=np.zeros(2),
            coil_current_limits=np.ones(2),
        )
    with pytest.raises(ValueError, match="response_matrix must be"):
        law.compute_coil_correction(
            target,
            target_vector=target,
            objective_bias_estimate=np.zeros(2),
            control_objective_weights=np.ones(2),
            response_matrix=np.ones(2),
            response_regularization=1e-3,
            correction_limit=1.0,
            control_mask=np.ones(2),
            coil_currents=np.zeros(2),
            coil_current_limits=np.ones(2),
        )
    with pytest.raises(ValueError, match="correction_limit"):
        law.compute_coil_correction(
            target,
            target_vector=target,
            objective_bias_estimate=np.zeros(2),
            control_objective_weights=np.ones(2),
            response_matrix=np.eye(2),
            response_regularization=1e-3,
            correction_limit=0.0,
            control_mask=np.ones(2),
            coil_currents=np.zeros(2),
            coil_current_limits=np.ones(2),
        )
    with pytest.raises(ValueError, match="response_regularization"):
        law.compute_coil_correction(
            target,
            target_vector=target,
            objective_bias_estimate=np.zeros(2),
            control_objective_weights=np.ones(2),
            response_matrix=np.eye(2),
            response_regularization=-1.0,
            correction_limit=1.0,
            control_mask=np.ones(2),
            coil_currents=np.zeros(2),
            coil_current_limits=np.ones(2),
        )

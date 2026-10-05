# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Free-boundary response and activation diagnostics tests

"""Exercise free-boundary response diagnostics, objective masks and headroom."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_control.control.free_boundary_tracking_control_law as law
from scpn_control.control.free_boundary_tracking_observation import ObjectiveBlock


def _blocks_shape_and_xpoint() -> tuple[ObjectiveBlock, ...]:
    return (
        ObjectiveBlock("shape_flux", 0, 3),
        ObjectiveBlock("x_point_position", 3, 5),
        ObjectiveBlock("x_point_flux", 5, 6),
        ObjectiveBlock("divertor_flux", 6, 8),
    )


def test_compute_response_diagnostics_rank_and_degeneracy() -> None:
    """Well-conditioned matrix is non-degenerate; near-zero matrix is degenerate."""
    well = np.array([[1.0, 0.0], [0.0, 2.0], [0.5, 0.5]], dtype=np.float64)
    diag = law.compute_response_diagnostics(well)
    assert diag.rank == 2
    assert diag.degenerate is False
    assert diag.max_singular_value > 0.0
    assert np.isfinite(diag.condition_number)

    dead = np.ones((4, 2), dtype=np.float64) * 1e-16
    dead_diag = law.compute_response_diagnostics(dead)
    assert dead_diag.degenerate is True
    assert dead_diag.rank == 0
    assert not np.isfinite(dead_diag.condition_number) or dead_diag.condition_number == float("inf")


def test_build_control_activation_mask_zeros_satisfied_blocks() -> None:
    """Satisfied objective blocks are masked off; unsatisfied blocks stay active."""
    blocks = _blocks_shape_and_xpoint()
    tolerances = {
        "shape_rms": 0.1,
        "shape_max_abs": 0.2,
        "x_point_position": 0.05,
        "x_point_flux": 0.01,
        "divertor_rms": 0.1,
        "divertor_max_abs": 0.2,
    }
    metrics = {
        "objective_checks": {
            "shape_rms": True,
            "shape_max_abs": True,
            "x_point_position": False,
            "x_point_flux": True,
            "divertor_rms": True,
            "divertor_max_abs": True,
        }
    }
    mask = law.build_control_activation_mask(8, blocks, tolerances, metrics)
    np.testing.assert_allclose(mask[0:3], 0.0)
    np.testing.assert_allclose(mask[3:5], 1.0)
    np.testing.assert_allclose(mask[5:6], 0.0)
    np.testing.assert_allclose(mask[6:8], 0.0)


def test_build_control_activation_mask_unknown_block_fails_closed() -> None:
    """Unknown objective block names fail closed."""
    with pytest.raises(ValueError, match="Unknown objective block"):
        law.build_control_activation_mask(
            2,
            (ObjectiveBlock("not_a_real_block", 0, 2),),
            {},
            {"objective_checks": {}},
        )


def test_build_coil_penalties_headroom_and_unbounded() -> None:
    """Penalties grow near limits; non-finite limits stay unit penalty."""
    currents = np.array([9.0, 0.0, -8.0], dtype=np.float64)
    limits = np.array([10.0, 10.0, np.inf], dtype=np.float64)
    delta_hint = np.array([1.0, 0.0, -1.0], dtype=np.float64)
    penalties = law.build_coil_penalties(currents, limits, delta_hint)
    assert penalties.shape == (3,)
    assert float(penalties[0]) > float(penalties[1])
    assert float(penalties[2]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="match the number of coils"):
        law.build_coil_penalties(currents, np.array([1.0, 2.0]), delta_hint)

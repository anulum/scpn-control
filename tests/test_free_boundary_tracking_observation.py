# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for FB tracking observation vectors

"""Drive production free-boundary tracking observation vector builders."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_control.control.free_boundary_tracking_observation as obs
from scpn_control.core.fusion_kernel import CoilSet


def _coilset_with_shape_and_xpoint() -> CoilSet:
    return CoilSet(
        positions=[(3.0, 2.0), (5.0, -2.0)],
        currents=np.array([1.0e4, -1.0e4]),
        turns=[10, 10],
        target_flux_points=np.array([[3.5, 0.0], [4.0, 0.5], [4.5, -0.5]], dtype=float),
        target_flux_values=np.array([0.10, 0.20, 0.15], dtype=float),
        x_point_target=np.array([4.0, -1.5], dtype=float),
        x_point_flux_target=0.12,
        divertor_strike_points=np.array([[3.2, -2.5], [4.8, -2.5]], dtype=float),
        divertor_flux_values=np.array([0.11, 0.11], dtype=float),
    )


def test_build_target_vector_blocks_and_values() -> None:
    """Target vector stacks shape, X-point, and divertor blocks in order."""
    coils = _coilset_with_shape_and_xpoint()
    target, blocks = obs.build_target_vector(coils)
    names = [b.name for b in blocks]
    assert names == ["shape_flux", "x_point_position", "x_point_flux", "divertor_flux"]
    assert target.shape == (3 + 2 + 1 + 2,)
    np.testing.assert_allclose(target[:3], [0.10, 0.20, 0.15])
    np.testing.assert_allclose(target[3:5], [4.0, -1.5])
    assert target[5] == pytest.approx(0.12)
    np.testing.assert_allclose(target[6:], [0.11, 0.11])


def test_build_target_vector_fail_closed_x_point_flux_without_target() -> None:
    """x_point_flux_target without position target fails closed."""
    coils = CoilSet(x_point_flux_target=0.1)
    with pytest.raises(ValueError, match="x_point_flux_target requires x_point_target"):
        obs.build_target_vector(coils)


def test_resolve_measurement_vector_success_and_fail_closed() -> None:
    """Measurement vector resolution supports scalar broadcast and rejects bad input."""
    coils = _coilset_with_shape_and_xpoint()
    target, blocks = obs.build_target_vector(coils)
    zero = obs.resolve_measurement_vector(None, objective_blocks=blocks, target_size=target.size, name="m")
    np.testing.assert_allclose(zero, 0.0)
    broadcast = obs.resolve_measurement_vector(
        {"shape_flux": 0.01},
        objective_blocks=blocks,
        target_size=target.size,
        name="m",
    )
    np.testing.assert_allclose(broadcast[:3], 0.01)
    with pytest.raises(ValueError, match="mapping"):
        obs.resolve_measurement_vector("bad", objective_blocks=blocks, target_size=target.size, name="m")
    with pytest.raises(ValueError, match="Unknown"):
        obs.resolve_measurement_vector({"bogus": 1.0}, objective_blocks=blocks, target_size=target.size, name="m")
    with pytest.raises(ValueError, match="finite"):
        obs.resolve_measurement_vector(
            {"shape_flux": float("inf")},
            objective_blocks=blocks,
            target_size=target.size,
            name="m",
        )


def test_control_weights_and_measurement_offset() -> None:
    """Control weights follow tolerances; measurement offset combines channels."""
    coils = _coilset_with_shape_and_xpoint()
    target, blocks = obs.build_target_vector(coils)
    weights = obs.build_control_objective_weights(
        target.size,
        blocks,
        {"shape_rms": 0.1, "x_point_position": 0.05},
    )
    assert weights.shape == target.shape
    assert float(weights[0]) == pytest.approx(1.0 / 0.1)
    assert float(weights[3]) == pytest.approx(1.0 / 0.05)
    bias = np.ones_like(target)
    drift = np.full_like(target, 0.5)
    corr_bias = np.full_like(target, 0.25)
    corr_drift = np.full_like(target, 0.1)
    offset = obs.current_measurement_offset(bias, drift, corr_bias, corr_drift)
    np.testing.assert_allclose(offset, 1.0 + 0.5 - 0.25 - 0.1)

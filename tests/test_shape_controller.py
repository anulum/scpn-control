# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Plasma Shape Controller Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np

from scpn_control.control.shape_controller import (
    CoilSet,
    PlasmaShapeController,
    iter_lower_single_null_target,
)


def test_iter_target_generation():
    target = iter_lower_single_null_target()
    assert len(target.isoflux_points) == 30
    assert len(target.gap_points) == 3
    assert target.xpoint_target is not None


def test_jacobian_full_rank():
    target = iter_lower_single_null_target()
    coils = CoilSet(n_coils=10)
    ctrl = PlasmaShapeController(target, coils, kernel=None)

    J = ctrl.jacobian.compute()
    # Check that rank is equal to number of coils (since equations > coils typically)
    # The random mock ensures this is highly probable.
    rank = np.linalg.matrix_rank(J)
    assert rank == coils.n_coils


def test_shape_controller_step():
    target = iter_lower_single_null_target()
    coils = CoilSet(n_coils=10)
    ctrl = PlasmaShapeController(target, coils, kernel=None)

    psi = np.ones((33, 33))  # Mock psi that produces non-zero error
    currents = np.zeros(10)

    delta_I = ctrl.step(psi, currents)

    # Should recommend some change
    assert np.any(np.abs(delta_I) > 0.0)

    # Check rate limiting
    assert np.all(np.abs(delta_I) <= 1000.0)


def test_shape_controller_limits():
    target = iter_lower_single_null_target()
    coils = CoilSet(n_coils=10)
    # Reduce max current heavily
    coils.max_currents = np.ones(10) * 100.0
    ctrl = PlasmaShapeController(target, coils, kernel=None)

    # If the required jump is large but we start near max, it must clip
    currents = np.ones(10) * 99.0

    # Force a large error direction
    ctrl.K_shape = -np.ones_like(ctrl.K_shape) * 1e6
    psi = np.ones((33, 33))

    delta_I = ctrl.step(psi, currents)

    I_next = currents + delta_I
    assert np.all(I_next <= 100.0)
    assert np.all(I_next >= -100.0)


def test_shape_performance_metrics():
    target = iter_lower_single_null_target()
    coils = CoilSet(n_coils=10)
    ctrl = PlasmaShapeController(target, coils, kernel=None)

    psi = np.ones((33, 33))
    res = ctrl.evaluate_performance(psi)

    assert res.isoflux_error > 0.0
    assert len(res.gap_errors) == 3
    # min_gap should not be wildly negative for the mock error
    assert res.min_gap > 0.0
    assert res.xpoint_error > 0.0

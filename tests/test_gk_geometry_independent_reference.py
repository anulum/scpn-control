# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — independent Miller geometry reference tests
"""Offline tests for :mod:`validation.gk_geometry_independent_reference`."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.gk_geometry import miller_geometry
from validation.gk_geometry_independent_reference import (
    IndependentMillerMetric,
    independent_miller_metric,
)

_THETA = np.linspace(-np.pi, np.pi, 96, endpoint=False)


def _params(**overrides: float) -> dict[str, float]:
    base = {
        "R0": 2.78,
        "a": 1.0,
        "rho": 0.5,
        "kappa": 1.7,
        "delta": 0.3,
        "s_kappa": 0.0,
        "s_delta": 0.0,
        "q": 1.9,
        "dR_dr": -0.08,
        "B0": 2.0,
    }
    base.update(overrides)
    return base


def test_returns_metric_with_grid_shape() -> None:
    ref = independent_miller_metric(theta=_THETA, **_params())
    assert isinstance(ref, IndependentMillerMetric)
    for field in (
        ref.R,
        ref.Z,
        ref.jacobian,
        ref.g_rr,
        ref.g_rt,
        ref.g_tt,
        ref.B_toroidal,
        ref.b_dot_grad_theta,
        ref.dR_dr,
        ref.dZ_dr,
    ):
        assert field.shape == _THETA.shape
        assert np.all(np.isfinite(field))


@pytest.mark.parametrize(
    "case",
    [
        {"kappa": 1.0, "delta": 0.0, "s_kappa": 0.0, "s_delta": 0.0, "dR_dr": 0.0},
        {"kappa": 1.7, "delta": 0.3, "s_kappa": 0.0, "s_delta": 0.0, "dR_dr": -0.08},
        {"kappa": 1.7, "delta": 0.3, "s_kappa": 0.4, "s_delta": 0.3, "dR_dr": -0.08},
        {"kappa": 1.6, "delta": -0.2, "s_kappa": -0.35, "s_delta": -0.25, "dR_dr": -0.05},
    ],
)
def test_matches_production_metric(case: dict[str, float]) -> None:
    params = _params(**case)
    geom = miller_geometry(**params, n_theta=128, n_period=1)
    ref = independent_miller_metric(theta=np.asarray(geom.theta), **params)
    for prod, got in (
        (geom.g_rr, ref.g_rr),
        (geom.g_rt, ref.g_rt),
        (geom.g_tt, ref.g_tt),
        (geom.jacobian, ref.jacobian),
    ):
        np.testing.assert_allclose(got, prod, rtol=1e-6, atol=1e-8)


def test_metric_is_sensitive_to_shaping_shear() -> None:
    # The reference must actually depend on s_kappa / s_delta, else it could not
    # detect the production bug it exists to guard against.
    base = independent_miller_metric(theta=_THETA, **_params(s_kappa=0.0, s_delta=0.0))
    sheared = independent_miller_metric(theta=_THETA, **_params(s_kappa=0.5, s_delta=0.4))
    assert not np.allclose(base.g_rr, sheared.g_rr, rtol=1e-3)
    assert not np.allclose(base.g_tt, sheared.g_tt, rtol=1e-3)


def test_circular_toroidal_field_one_over_R() -> None:
    ref = independent_miller_metric(theta=_THETA, **_params(kappa=1.0, delta=0.0, dR_dr=0.0))
    np.testing.assert_allclose(ref.B_toroidal, 2.0 * 2.78 / ref.R, rtol=1e-12)


@pytest.mark.parametrize(
    "override, match",
    [
        ({"R0": 0.0}, "R0 must be positive"),
        ({"a": -1.0}, "a must be positive"),
        ({"R0": float("nan")}, "R0 must be finite"),
        ({"rho": 0.0}, r"rho must lie in \(0, 1\]"),
        ({"rho": 1.5}, r"rho must lie in \(0, 1\]"),
        ({"kappa": 0.0}, "kappa must be positive"),
        ({"delta": 1.0}, r"delta must lie in \(-1, 1\)"),
        ({"delta": float("inf")}, "delta must be finite"),
        ({"s_kappa": float("nan")}, "s_kappa must be finite"),
        ({"s_delta": float("inf")}, "s_delta must be finite"),
        ({"q": 0.0}, "q must be positive"),
        ({"dR_dr": float("nan")}, "dR_dr must be finite"),
        ({"B0": -2.0}, "B0 must be positive"),
        ({"eps_r_rel": 0.0}, "eps_r_rel must be positive"),
        ({"eps_theta": -1.0}, "eps_theta must be positive"),
    ],
)
def test_rejects_invalid_parameters(override: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        independent_miller_metric(theta=_THETA, **_params(**override))


def test_rejects_non_1d_theta() -> None:
    with pytest.raises(ValueError, match="theta must be a non-empty 1-D array"):
        independent_miller_metric(theta=np.zeros((2, 2)), **_params())


def test_rejects_empty_theta() -> None:
    with pytest.raises(ValueError, match="theta must be a non-empty 1-D array"):
        independent_miller_metric(theta=np.asarray([], dtype=np.float64), **_params())


def test_rejects_surface_crossing_axis() -> None:
    # Small R0 with unit local radius drives R(theta=pi) negative.
    with pytest.raises(ValueError, match="major radius must remain positive"):
        independent_miller_metric(
            theta=_THETA,
            R0=0.5,
            a=1.0,
            rho=1.0,
            kappa=1.0,
            delta=0.0,
            s_kappa=0.0,
            s_delta=0.0,
            q=1.4,
            dR_dr=0.0,
            B0=2.0,
        )

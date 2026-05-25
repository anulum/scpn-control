# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable transport tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core import differentiable_transport as dt


def _profiles(rho: np.ndarray) -> np.ndarray:
    te = 8.0 * np.exp(-((rho - 0.35) ** 2) / 0.03) + 0.2
    ti = 6.0 * np.exp(-((rho - 0.42) ** 2) / 0.04) + 0.2
    ne = 4.0 + 0.8 * (1.0 - rho**2)
    nz = 0.03 + 0.02 * np.exp(-((rho - 0.65) ** 2) / 0.02)
    return np.stack([te, ti, ne, nz])


def test_numpy_transport_step_diffuses_all_channels_and_applies_boundaries():
    rho = np.linspace(0.05, 1.0, 48)
    profiles = _profiles(rho)
    chi = np.stack(
        [
            0.22 + 0.03 * rho,
            0.18 + 0.02 * rho,
            0.05 + 0.01 * rho,
            0.015 + 0.002 * rho,
        ]
    )
    sources = np.zeros_like(profiles)
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])

    stepped = dt.differentiable_transport_step(
        profiles,
        chi,
        sources,
        rho,
        2.0e-3,
        edge_values,
        use_jax=False,
    )

    assert stepped.shape == profiles.shape
    assert np.all(np.isfinite(stepped))
    assert np.max(stepped[0, 1:-1]) < np.max(profiles[0, 1:-1])
    assert np.max(stepped[1, 1:-1]) < np.max(profiles[1, 1:-1])
    np.testing.assert_allclose(stepped[:, -1], edge_values, atol=1e-12)
    np.testing.assert_allclose(stepped[:, 0], stepped[:, 1], atol=1e-12)


def test_transport_tracking_loss_is_zero_for_exact_next_step_target():
    rho = np.linspace(0.05, 1.0, 32)
    profiles = _profiles(rho)
    chi = 0.04 * np.ones_like(profiles)
    sources = np.zeros_like(profiles)
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])
    target = dt.differentiable_transport_step(
        profiles,
        chi,
        sources,
        rho,
        1.0e-3,
        edge_values,
        use_jax=False,
    )

    loss = dt.transport_tracking_loss(
        profiles,
        chi,
        sources,
        target,
        rho,
        1.0e-3,
        edge_values,
        use_jax=False,
    )

    assert loss == pytest.approx(0.0, abs=1e-24)


def test_gradient_api_fails_closed_without_jax(monkeypatch):
    rho = np.linspace(0.05, 1.0, 16)
    profiles = _profiles(rho)
    chi = 0.04 * np.ones_like(profiles)
    sources = np.zeros_like(profiles)
    target = profiles.copy()
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])
    monkeypatch.setattr(dt, "_HAS_JAX", False)

    with pytest.raises(RuntimeError, match="transport_loss_gradient requires JAX"):
        dt.transport_loss_gradient(profiles, chi, sources, target, rho, 1.0e-3, edge_values)


@pytest.mark.skipif(not dt.has_jax(), reason="JAX optional dependency is not installed")
def test_transport_loss_gradient_is_finite_with_jax():
    rho = np.linspace(0.05, 1.0, 24)
    profiles = _profiles(rho)
    chi = np.stack(
        [
            0.20 + 0.02 * rho,
            0.16 + 0.02 * rho,
            0.04 + 0.005 * rho,
            0.012 + 0.001 * rho,
        ]
    )
    sources = np.zeros_like(profiles)
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])
    target = profiles.copy()
    target[0, 8:16] *= 0.97
    target[1, 8:16] *= 0.98
    weights = np.array([1.0, 1.0, 0.25, 0.1])

    loss, gradient = dt.transport_loss_gradient(
        profiles,
        chi,
        sources,
        target,
        rho,
        1.0e-3,
        edge_values,
        weights=weights,
    )

    assert np.isfinite(loss)
    assert gradient.shape == chi.shape
    assert np.all(np.isfinite(gradient))
    assert np.any(np.abs(gradient) > 0.0)

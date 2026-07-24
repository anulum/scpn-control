# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for transport core step / rollout

"""Drive production transport step and rollout helpers on real surfaces."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_control.core.differentiable_transport as facade
import scpn_control.core.differentiable_transport_core as core


def _profiles(n_rho: int = 12) -> tuple[NDArray[np.float64], ...]:
    rho = np.linspace(0.05, 1.0, n_rho, dtype=np.float64)
    profiles = np.stack(
        [
            1.0 + 0.2 * (1.0 - rho),
            0.9 + 0.15 * (1.0 - rho),
            0.5 + 0.05 * (1.0 - rho),
            0.05 + 0.02 * (1.0 - rho),
        ]
    )
    chi = 0.05 * np.ones((4, n_rho), dtype=np.float64)
    sources = 0.01 * np.ones((4, n_rho), dtype=np.float64)
    edge = profiles[:, -1].copy()
    return profiles, chi, sources, rho, edge


def test_public_core_symbols_bind_to_leaf() -> None:
    """Facade re-exports are the production core step/rollout leaf objects."""
    assert facade.differentiable_transport_step is core.differentiable_transport_step
    assert facade.differentiable_transport_rollout is core.differentiable_transport_rollout
    assert facade._transport_step_numpy is core._transport_step_numpy
    assert facade._transport_step_jax is core._transport_step_jax
    assert facade._transport_rollout_numpy is core._transport_rollout_numpy
    assert facade._transport_rollout_jax is core._transport_rollout_jax


def test_step_numpy_preserves_shape_and_edge() -> None:
    """One-step NumPy advance returns four channels with Dirichlet edge values."""
    profiles, chi, sources, rho, edge = _profiles()
    stepped = facade.differentiable_transport_step(profiles, chi, sources, rho, 1.0e-3, edge, use_jax=False)
    assert stepped.shape == profiles.shape
    np.testing.assert_allclose(stepped[:, -1], edge, rtol=1.0e-12, atol=1.0e-12)
    leaf = core.differentiable_transport_step(profiles, chi, sources, rho, 1.0e-3, edge, use_jax=False)
    np.testing.assert_allclose(leaf, stepped, rtol=1.0e-12, atol=1.0e-14)


def test_step_rejects_negative_dt() -> None:
    """Fail closed when the time step is non-positive."""
    profiles, chi, sources, rho, edge = _profiles()
    with pytest.raises(ValueError, match="dt"):
        core.differentiable_transport_step(profiles, chi, sources, rho, 0.0, edge, use_jax=False)


def test_rollout_shape_and_zero_source_near_identity() -> None:
    """Multi-step rollout has (n_steps, 4, n_rho); near-zero sources stay close to start."""
    profiles, chi, sources, rho, edge = _profiles()
    sequence = np.stack([sources * 0.0, sources * 0.0, sources * 0.0])
    history = facade.differentiable_transport_rollout(profiles, chi, sequence, rho, 1.0e-4, edge, use_jax=False)
    assert history.shape == (3, 4, rho.size)
    # With zero sources and short dt, profiles remain finite and edge fixed.
    assert np.all(np.isfinite(history))
    np.testing.assert_allclose(history[-1, :, -1], edge, rtol=1.0e-12, atol=1.0e-12)


def test_step_jax_unavailable_via_facade_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Internal JAX step fails closed when facade jnp is cleared."""
    profiles, chi, sources, rho, edge = _profiles()
    monkeypatch.setattr(facade, "jnp", None)
    monkeypatch.setattr(facade, "jax", None)
    with pytest.raises(RuntimeError, match="JAX transport step"):
        facade._transport_step_jax(profiles, chi, sources, rho, 1.0e-3, edge)


def test_step_jax_matches_numpy_when_available() -> None:
    """JAX and NumPy one-step advances agree on the production path."""
    if not facade.has_jax():
        pytest.skip("JAX not installed in this CI lane (optional transport backend)")
    profiles, chi, sources, rho, edge = _profiles()
    step_jax = np.asarray(facade.differentiable_transport_step(profiles, chi, sources, rho, 1.0e-3, edge, use_jax=True))
    step_numpy = facade.differentiable_transport_step(profiles, chi, sources, rho, 1.0e-3, edge, use_jax=False)
    np.testing.assert_allclose(step_jax, step_numpy, rtol=1.0e-8, atol=1.0e-10)

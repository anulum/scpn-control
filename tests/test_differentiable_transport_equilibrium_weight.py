# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for equilibrium-weighted transport AD

"""Drive production equilibrium-weighted loss and gradient helpers on real surfaces."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_control.core.differentiable_transport as facade
import scpn_control.core.differentiable_transport_equilibrium_weight as eq_weight


def _profiles(n_rho: int = 10) -> tuple[NDArray[np.float64], ...]:
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
    target = profiles + 0.03
    # Flux map with non-uniform radial mean so weights differ from unity.
    r = np.linspace(0.0, 1.0, n_rho)
    z = np.linspace(-0.5, 0.5, 8)
    rr, zz = np.meshgrid(r, z, indexing="xy")
    psi = (1.0 - rr) ** 2 * (1.0 + 0.2 * zz)
    return profiles, chi, sources, target, rho, edge, psi


def test_public_eq_weight_symbols_bind_to_leaf() -> None:
    """Facade re-exports are the production equilibrium-weight leaf objects."""
    assert facade.EquilibriumWeightedTransportGradient is eq_weight.EquilibriumWeightedTransportGradient
    assert facade.EquilibriumWeightedTransportRolloutGradient is eq_weight.EquilibriumWeightedTransportRolloutGradient
    assert facade.equilibrium_radial_weights is eq_weight.equilibrium_radial_weights
    assert facade.equilibrium_weighted_transport_tracking_loss is eq_weight.equilibrium_weighted_transport_tracking_loss
    assert facade.equilibrium_weighted_transport_loss_gradient is eq_weight.equilibrium_weighted_transport_loss_gradient
    assert (
        facade.equilibrium_weighted_transport_rollout_tracking_loss
        is eq_weight.equilibrium_weighted_transport_rollout_tracking_loss
    )
    assert (
        facade.equilibrium_weighted_transport_rollout_source_gradient
        is eq_weight.equilibrium_weighted_transport_rollout_source_gradient
    )


def test_equilibrium_radial_weights_mean_one_and_nonnegative() -> None:
    """Production radial weights from a flux map are non-negative with unit mean."""
    _profiles_pack = _profiles()
    psi = _profiles_pack[-1]
    n_rho = _profiles_pack[4].size
    weights = facade.equilibrium_radial_weights(psi, n_rho)
    assert weights.shape == (n_rho,)
    assert np.all(weights >= 0.0)
    assert float(np.mean(weights)) == pytest.approx(1.0, abs=1.0e-12)
    # Non-uniform flux must not collapse to all-ones (production geometry signal).
    assert not np.allclose(weights, np.ones_like(weights))
    leaf_weights = eq_weight.equilibrium_radial_weights(psi, n_rho)
    np.testing.assert_allclose(leaf_weights, weights, rtol=1.0e-12, atol=1.0e-14)


def test_equilibrium_radial_weights_reject_bad_n_rho() -> None:
    """Fail closed when n_rho is invalid."""
    psi = np.ones((4, 8))
    with pytest.raises(ValueError, match="n_rho"):
        eq_weight.equilibrium_radial_weights(psi, 2)
    with pytest.raises(ValueError, match="n_rho"):
        eq_weight.equilibrium_radial_weights(psi, True)


def test_validate_equilibrium_psi_rejects_bad_shape() -> None:
    """Fail closed when flux map is not a 2-D field of adequate size."""
    with pytest.raises(ValueError, match="equilibrium_psi"):
        eq_weight._validate_equilibrium_psi(np.ones(5))
    with pytest.raises(ValueError, match="equilibrium_psi"):
        eq_weight._validate_equilibrium_psi(np.ones((2, 2)))


def test_eq_weighted_tracking_loss_differs_from_uniform() -> None:
    """GS weighting changes the real one-step tracking loss vs uniform weights."""
    profiles, chi, sources, target, rho, edge, psi = _profiles()
    uniform = float(facade.transport_tracking_loss(profiles, chi, sources, target, rho, 1.0e-3, edge, use_jax=False))
    weighted = float(
        facade.equilibrium_weighted_transport_tracking_loss(
            profiles, chi, sources, target, rho, 1.0e-3, edge, psi, use_jax=False
        )
    )
    assert weighted != pytest.approx(uniform, abs=1.0e-14)
    leaf = float(
        eq_weight.equilibrium_weighted_transport_tracking_loss(
            profiles, chi, sources, target, rho, 1.0e-3, edge, psi, use_jax=False
        )
    )
    assert leaf == pytest.approx(weighted, abs=1.0e-14)


def test_eq_weighted_loss_gradient_fail_closed_without_jax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Facade JAX gate controls leaf equilibrium-weighted gradient admission."""
    profiles, chi, sources, target, rho, edge, psi = _profiles()
    monkeypatch.setattr(facade, "_HAS_JAX", False)
    with pytest.raises(RuntimeError, match="equilibrium_weighted_transport_loss_gradient requires JAX"):
        facade.equilibrium_weighted_transport_loss_gradient(profiles, chi, sources, target, rho, 1.0e-3, edge, psi)
    with pytest.raises(RuntimeError, match="equilibrium_weighted_transport_rollout_source_gradient requires JAX"):
        facade.equilibrium_weighted_transport_rollout_source_gradient(
            profiles,
            chi,
            np.stack([sources, sources]),
            np.stack([target, target]),
            rho,
            1.0e-3,
            edge,
            psi,
        )


def test_eq_weighted_loss_gradient_finite_via_facade() -> None:
    """Leaf and facade share finite JAX equilibrium-weighted chi gradients."""
    if not facade.has_jax():
        pytest.skip("JAX not installed in this CI lane (optional transport backend)")
    profiles, chi, sources, target, rho, edge, psi = _profiles()
    result = facade.equilibrium_weighted_transport_loss_gradient(profiles, chi, sources, target, rho, 1.0e-3, edge, psi)
    assert isinstance(result, facade.EquilibriumWeightedTransportGradient)
    assert np.all(np.isfinite(result.chi_gradient))
    assert np.all(np.isfinite(result.equilibrium_gradient))
    assert result.loss >= 0.0
    leaf = eq_weight.equilibrium_weighted_transport_loss_gradient(
        profiles, chi, sources, target, rho, 1.0e-3, edge, psi
    )
    np.testing.assert_allclose(leaf.chi_gradient, result.chi_gradient, rtol=1.0e-10, atol=1.0e-12)


def test_eq_weighted_rollout_source_gradient_finite_via_facade() -> None:
    """Multi-step GS-weighted source gradient is finite on the production path."""
    if not facade.has_jax():
        pytest.skip("JAX not installed in this CI lane (optional transport backend)")
    profiles, chi, sources, target, rho, edge, psi = _profiles()
    source_sequence = np.stack([sources, sources * 1.1, sources * 0.9])
    history = facade.differentiable_transport_rollout(profiles, chi, source_sequence, rho, 1.0e-3, edge, use_jax=False)
    target_history = np.asarray(history) + 0.02
    result = facade.equilibrium_weighted_transport_rollout_source_gradient(
        profiles, chi, source_sequence, target_history, rho, 1.0e-3, edge, psi
    )
    assert isinstance(result, facade.EquilibriumWeightedTransportRolloutGradient)
    assert np.all(np.isfinite(result.source_gradient))
    assert result.final_profiles.shape == profiles.shape
    assert result.loss >= 0.0

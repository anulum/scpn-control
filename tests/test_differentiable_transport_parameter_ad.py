# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for transport parameter AD

"""Drive production parameter-gradient and audit helpers on real surfaces."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_control.core.differentiable_transport as facade
import scpn_control.core.differentiable_transport_parameter_ad as parameter_ad


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
    target = profiles + 0.03
    edge = profiles[:, -1].copy()
    return profiles, chi, sources, target, rho, edge


def test_public_parameter_ad_symbols_bind_to_leaf() -> None:
    """Facade re-exports are the production parameter-AD leaf objects."""
    assert facade.TransportParameterGradients is parameter_ad.TransportParameterGradients
    assert facade.transport_tracking_loss is parameter_ad.transport_tracking_loss
    assert facade.transport_loss_gradient is parameter_ad.transport_loss_gradient
    assert facade.transport_parameter_gradients is parameter_ad.transport_parameter_gradients
    assert facade.audit_transport_parameter_gradients is parameter_ad.audit_transport_parameter_gradients
    assert (
        facade.assert_transport_parameter_gradients_consistent
        is parameter_ad.assert_transport_parameter_gradients_consistent
    )


def test_transport_tracking_loss_zero_for_exact_next_step() -> None:
    """Production tracking loss is zero when target equals the real next step."""
    profiles, chi, sources, _target, rho, edge = _profiles()
    next_profiles = facade.differentiable_transport_step(profiles, chi, sources, rho, 1.0e-3, edge, use_jax=False)
    loss = facade.transport_tracking_loss(
        profiles,
        chi,
        sources,
        next_profiles,
        rho,
        1.0e-3,
        edge,
        use_jax=False,
    )
    assert float(loss) == pytest.approx(0.0, abs=1.0e-12)
    leaf_loss = parameter_ad.transport_tracking_loss(
        profiles,
        chi,
        sources,
        next_profiles,
        rho,
        1.0e-3,
        edge,
        use_jax=False,
    )
    assert float(leaf_loss) == pytest.approx(float(loss), abs=1.0e-12)


def test_transport_tracking_loss_requires_target() -> None:
    """Fail closed when target profiles are omitted."""
    profiles, chi, sources, _target, rho, edge = _profiles()
    with pytest.raises(ValueError, match="target_profiles"):
        parameter_ad.transport_tracking_loss(profiles, chi, sources, None, rho, 1.0e-3, edge, use_jax=False)


def test_parameter_gradients_fail_closed_without_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Facade JAX gate controls leaf parameter-gradient admission."""
    profiles, chi, sources, target, rho, edge = _profiles()
    monkeypatch.setattr(facade, "_HAS_JAX", False)
    with pytest.raises(RuntimeError, match="transport_parameter_gradients requires JAX"):
        facade.transport_parameter_gradients(profiles, chi, sources, target, rho, 1.0e-3, edge)
    with pytest.raises(RuntimeError, match="transport_loss_gradient requires JAX"):
        facade.transport_loss_gradient(profiles, chi, sources, target, rho, 1.0e-3, edge)


def test_parameter_gradients_and_audit_via_facade() -> None:
    """Leaf and facade share finite JAX parameter gradients and pass FD audit."""
    if not facade.has_jax():
        pytest.skip("JAX not installed in this CI lane (optional transport backend)")
    profiles, chi, sources, target, rho, edge = _profiles()
    result = facade.transport_parameter_gradients(profiles, chi, sources, target, rho, 1.0e-3, edge)
    assert isinstance(result, facade.TransportParameterGradients)
    assert np.all(np.isfinite(result.chi_gradient))
    assert np.all(np.isfinite(result.source_gradient))
    assert result.loss >= 0.0
    leaf_result = parameter_ad.transport_parameter_gradients(profiles, chi, sources, target, rho, 1.0e-3, edge)
    np.testing.assert_allclose(leaf_result.chi_gradient, result.chi_gradient, rtol=1.0e-10, atol=1.0e-12)
    audit = facade.assert_transport_parameter_gradients_consistent(
        profiles,
        chi,
        sources,
        target,
        rho,
        1.0e-3,
        edge,
        sample_indices=((0, 2), (1, 5), (2, 3), (3, 4)),
    )
    assert audit.passed
    assert audit.checked_indices == ((0, 2), (1, 5), (2, 3), (3, 4))


def test_audit_rejects_invalid_epsilon() -> None:
    """Production audit fails closed on non-positive finite-difference step."""
    profiles, chi, sources, target, rho, edge = _profiles()
    with pytest.raises(ValueError, match="epsilon"):
        parameter_ad.audit_transport_parameter_gradients(profiles, chi, sources, target, rho, 1.0e-3, edge, epsilon=0.0)


def test_audit_keeps_target_guard_when_gradients_monkeypatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even with patched gradients, missing targets still fail closed on the real path."""
    profiles, chi, sources, _target, rho, edge = _profiles()

    def fake_parameter_gradients(*args: object, **kwargs: object) -> parameter_ad.TransportParameterGradients:
        return parameter_ad.TransportParameterGradients(
            loss=0.0,
            chi_gradient=np.zeros_like(chi),
            source_gradient=np.zeros_like(sources),
        )

    monkeypatch.setattr(facade, "transport_parameter_gradients", fake_parameter_gradients)
    with pytest.raises(ValueError, match="target_profiles is required"):
        facade.audit_transport_parameter_gradients(profiles, chi, sources, None, rho, 1.0e-3, edge)


def test_gradient_audit_indices_default_and_bounds() -> None:
    """Sample index helper covers default interior points and rejects OOB pairs."""
    shape = (4, 10)
    defaults = parameter_ad._gradient_audit_indices(shape, None)
    assert (0, 1) in defaults
    assert (0, 5) in defaults
    assert (0, 8) in defaults
    with pytest.raises(ValueError, match="out-of-bounds"):
        parameter_ad._gradient_audit_indices(shape, ((0, 99),))
    with pytest.raises(ValueError, match="at least one"):
        parameter_ad._gradient_audit_indices(shape, ())

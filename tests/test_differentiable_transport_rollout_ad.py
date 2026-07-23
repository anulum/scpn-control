# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for transport rollout source AD

"""Drive production rollout source-gradient and audit helpers on real surfaces."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_control.core.differentiable_transport as facade
import scpn_control.core.differentiable_transport_rollout_ad as rollout_ad


def _rollout_fixture(n_rho: int = 10, n_steps: int = 3) -> tuple[NDArray[np.float64], ...]:
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
    source_sequence = 0.01 * np.ones((n_steps, 4, n_rho), dtype=np.float64)
    edge = profiles[:, -1].copy()
    history = facade.differentiable_transport_rollout(profiles, chi, source_sequence, rho, 1.0e-3, edge, use_jax=False)
    target_history = np.asarray(history, dtype=np.float64)
    return profiles, chi, source_sequence, target_history, rho, edge


def test_public_rollout_ad_symbols_bind_to_leaf() -> None:
    """Facade re-exports are the production rollout-AD leaf objects."""
    assert facade.TransportRolloutSourceGradients is rollout_ad.TransportRolloutSourceGradients
    assert facade.transport_rollout_tracking_loss is rollout_ad.transport_rollout_tracking_loss
    assert facade.transport_rollout_source_gradients is rollout_ad.transport_rollout_source_gradients
    assert facade.audit_transport_rollout_source_gradients is rollout_ad.audit_transport_rollout_source_gradients
    assert (
        facade.assert_transport_rollout_source_gradients_consistent
        is rollout_ad.assert_transport_rollout_source_gradients_consistent
    )


def test_rollout_tracking_loss_zero_for_exact_history() -> None:
    """Production multi-step loss is zero when target equals the real rollout."""
    profiles, chi, source_sequence, target_history, rho, edge = _rollout_fixture()
    loss = facade.transport_rollout_tracking_loss(
        profiles,
        chi,
        source_sequence,
        target_history,
        rho,
        1.0e-3,
        edge,
        use_jax=False,
    )
    assert float(loss) == pytest.approx(0.0, abs=1.0e-12)
    leaf_loss = rollout_ad.transport_rollout_tracking_loss(
        profiles,
        chi,
        source_sequence,
        target_history,
        rho,
        1.0e-3,
        edge,
        use_jax=False,
    )
    assert float(leaf_loss) == pytest.approx(float(loss), abs=1.0e-12)


def test_rollout_tracking_loss_requires_target_history() -> None:
    """Fail closed when multi-step target history is omitted."""
    profiles, chi, source_sequence, _target, rho, edge = _rollout_fixture()
    with pytest.raises(ValueError, match="target_history"):
        rollout_ad.transport_rollout_tracking_loss(
            profiles, chi, source_sequence, None, rho, 1.0e-3, edge, use_jax=False
        )


def test_rollout_source_gradients_fail_closed_without_jax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Facade JAX gate controls leaf rollout source-gradient admission."""
    profiles, chi, source_sequence, target_history, rho, edge = _rollout_fixture()
    monkeypatch.setattr(facade, "_HAS_JAX", False)
    with pytest.raises(RuntimeError, match="transport_rollout_source_gradients requires JAX"):
        facade.transport_rollout_source_gradients(profiles, chi, source_sequence, target_history, rho, 1.0e-3, edge)


def test_rollout_source_gradients_and_audit_via_facade() -> None:
    """Leaf and facade share finite JAX rollout source gradients and pass FD audit."""
    assert facade.has_jax(), "project venv must provide JAX for rollout AD"
    profiles, chi, source_sequence, target_history, rho, edge = _rollout_fixture()
    # Use a non-exact target so source gradients are non-trivial.
    target_history = target_history + 0.02
    result = facade.transport_rollout_source_gradients(
        profiles, chi, source_sequence, target_history, rho, 1.0e-3, edge
    )
    assert isinstance(result, facade.TransportRolloutSourceGradients)
    assert np.all(np.isfinite(result.source_gradient))
    assert result.loss >= 0.0
    assert result.final_profiles.shape == profiles.shape
    leaf_result = rollout_ad.transport_rollout_source_gradients(
        profiles, chi, source_sequence, target_history, rho, 1.0e-3, edge
    )
    np.testing.assert_allclose(leaf_result.source_gradient, result.source_gradient, rtol=1.0e-10, atol=1.0e-12)
    sample_indices = ((0, 0, 1), (2, 1, 5), (1, 2, 3), (2, 3, 2))
    audit = facade.assert_transport_rollout_source_gradients_consistent(
        profiles,
        chi,
        source_sequence,
        target_history,
        rho,
        1.0e-3,
        edge,
        sample_indices=sample_indices,
    )
    assert audit.passed
    assert audit.checked_indices == sample_indices


def test_rollout_audit_rejects_invalid_epsilon() -> None:
    """Production rollout audit fails closed on non-positive FD step."""
    profiles, chi, source_sequence, target_history, rho, edge = _rollout_fixture()
    with pytest.raises(ValueError, match="epsilon"):
        rollout_ad.audit_transport_rollout_source_gradients(
            profiles,
            chi,
            source_sequence,
            target_history,
            rho,
            1.0e-3,
            edge,
            epsilon=0.0,
        )


def test_rollout_gradient_audit_indices_default_and_bounds() -> None:
    """Sample index helper covers defaults, dedupes, and rejects OOB triples."""
    shape = (3, 4, 8)
    defaults = rollout_ad._rollout_gradient_audit_indices(shape, None)
    assert len(defaults) >= 1
    assert all(len(index) == 3 for index in defaults)
    deduped = rollout_ad._rollout_gradient_audit_indices(shape, [(0, 0, 1), (0, 0, 1), (1, 2, 4)])
    assert deduped == ((0, 0, 1), (1, 2, 4))
    with pytest.raises(ValueError, match="out-of-range"):
        rollout_ad._rollout_gradient_audit_indices(shape, ((0, 0, 99),))
    with pytest.raises(ValueError, match="at least one"):
        rollout_ad._rollout_gradient_audit_indices(shape, [])
    with pytest.raises(ValueError, match="shape"):
        rollout_ad._rollout_gradient_audit_indices((4, 5), None)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for transport closure adapters

"""Drive the production closure leaf through public and leaf entry points."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_control.core.differentiable_transport as facade
import scpn_control.core.differentiable_transport_closures as closures


def _rho(n: int = 8) -> NDArray[np.float64]:
    return np.linspace(0.0, 1.0, n, dtype=np.float64)


def _neural_closure(n: int = 8) -> SimpleNamespace:
    rho = _rho(n)
    chi_e = 0.2 + 0.1 * rho
    chi_i = 0.3 + 0.05 * rho
    d_e = 0.1 + 0.02 * rho
    total = chi_e + chi_i + d_e
    weights = np.stack([chi_e / total, chi_i / total, d_e / total])
    return SimpleNamespace(chi_e=chi_e, chi_i=chi_i, d_e=d_e, channel_weights=weights)


class _GkModel:
    def evaluate_profile(
        self,
        rho: NDArray[np.float64],
        profiles: dict[str, object],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        del profiles
        chi_i = 0.4 * np.ones_like(rho)
        chi_e = 0.2 * np.ones_like(rho)
        d_e = 0.1 * np.ones_like(rho)
        return chi_i, chi_e, d_e


def test_neural_closure_maps_to_four_channels_via_leaf_and_facade() -> None:
    """Leaf and facade entry points agree on four-channel coefficient maps."""
    closure = _neural_closure()
    leaf = closures.transport_coefficients_from_neural_closure(
        closure, impurity_diffusivity_fraction=0.5, chi_floor=0.05
    )
    public = facade.transport_coefficients_from_neural_closure(
        closure, impurity_diffusivity_fraction=0.5, chi_floor=0.05
    )
    assert leaf.shape == (4, closure.chi_e.size)
    np.testing.assert_allclose(leaf, public)
    np.testing.assert_allclose(leaf[0], np.maximum(0.05, closure.chi_e))
    np.testing.assert_allclose(leaf[3], np.maximum(0.05, 0.5 * closure.d_e))


def test_gyrokinetic_closure_adapter_fail_closed_and_success() -> None:
    """GK adapter validates model contract and produces facade-compatible coefficients."""
    rho = _rho()
    with pytest.raises(ValueError, match="evaluate_profile"):
        closures.gyrokinetic_transport_closure_profiles(object(), rho, {})
    result = closures.gyrokinetic_transport_closure_profiles(_GkModel(), rho, {})
    assert result.source == "reduced_gyrokinetic"
    assert result.channel_weights.shape == (3, rho.size)
    chi = facade.transport_coefficients_from_gyrokinetic_closure(result)
    assert chi.shape == (4, rho.size)
    assert np.all(chi >= 0.0)


def test_neural_closure_rejects_invalid_fraction_and_shape() -> None:
    """Fail-closed guards for impurity fraction and profile shape remain on the leaf."""
    closure = _neural_closure()
    with pytest.raises(ValueError, match="impurity_diffusivity_fraction"):
        closures.transport_coefficients_from_neural_closure(closure, impurity_diffusivity_fraction=1.5)
    bad = _neural_closure()
    bad.chi_i = bad.chi_i[:-1]
    with pytest.raises(ValueError, match="same shape"):
        closures.transport_coefficients_from_neural_closure(bad)


def test_public_symbol_identity_on_facade() -> None:
    """Facade re-exports bind to the closure leaf production objects."""
    assert facade.GyrokineticTransportClosureResult is closures.GyrokineticTransportClosureResult
    assert facade.transport_coefficients_from_neural_closure is closures.transport_coefficients_from_neural_closure
    assert facade.gyrokinetic_transport_closure_profiles is closures.gyrokinetic_transport_closure_profiles


def test_as_float_array_rejects_non_finite() -> None:
    """Closure array coercion fails closed on NaN or infinite entries."""
    with pytest.raises(ValueError, match="finite"):
        closures._as_float_array("chi", np.array([1.0, np.nan]))

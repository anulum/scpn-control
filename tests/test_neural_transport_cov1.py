# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural transport COV-1 tests.
"""Focused COV-1 tests for neural transport public surfaces."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_control.core.neural_transport import NeuralTransportModel, TransportInputs, neural_transport_closure_profiles


def _flat_profile() -> NDArray[np.float64]:
    """Return a profile with no transport-driving gradients."""
    return np.full(5, 5.0, dtype=np.float64)


def _write_zero_scale_weights(path: Path) -> None:
    """Write a valid neural weight file whose heat-channel outputs are zero."""
    np.savez(
        path,
        w1=np.zeros((10, 2), dtype=np.float64),
        b1=np.zeros(2, dtype=np.float64),
        w2=np.zeros((2, 2), dtype=np.float64),
        b2=np.zeros(2, dtype=np.float64),
        w3=np.zeros((2, 3), dtype=np.float64),
        b3=np.zeros(3, dtype=np.float64),
        input_mean=np.zeros(10, dtype=np.float64),
        input_std=np.ones(10, dtype=np.float64),
        output_scale=np.zeros(3, dtype=np.float64),
        version=np.array(1, dtype=np.int64),
    )


def test_predict_reports_stable_channel_for_nonpositive_transport() -> None:
    """The public single-point predictor reports stable when all fallback fluxes vanish."""
    model = NeuralTransportModel(auto_discover=False)

    fluxes = model.predict(
        TransportInputs(
            grad_te=0.0,
            grad_ti=0.0,
            grad_ne=0.0,
        )
    )

    assert fluxes.channel == "stable"


def test_neural_predict_reports_stable_channel_for_zero_scaled_weights(tmp_path: Path) -> None:
    """The public neural-weight path reports stable when learned heat fluxes vanish."""
    weights_path = tmp_path / "zero_scale_transport_weights.npz"
    _write_zero_scale_weights(weights_path)
    model = NeuralTransportModel(weights_path=weights_path)

    fluxes = model.predict(TransportInputs())

    assert model.is_neural is True
    assert fluxes.channel == "stable"


def test_closure_channel_weights_remain_sum_safe_after_numpy_reload() -> None:
    """The public closure result returns channel weights whose ndarray reductions work."""
    rho = np.linspace(0.1, 0.9, 5, dtype=np.float64)
    flat = _flat_profile()

    result = neural_transport_closure_profiles(
        rho,
        flat,
        flat,
        flat,
        np.full(5, 2.0, dtype=np.float64),
        np.full(5, 0.5, dtype=np.float64),
        model=NeuralTransportModel(auto_discover=False),
    )

    assert np.allclose(result.channel_weights, 1.0 / 3.0)
    assert np.allclose(result.channel_weights.sum(axis=0), np.ones_like(rho))
    assert np.allclose(result.channel_weights.sum(axis=1), np.full(3, 5.0 / 3.0))
    assert result.channel_weights.sum() == pytest.approx(5.0)
    assert np.allclose(result.channel_weights.sum(axis=0, initial=1.0), np.full(5, 2.0))
    with pytest.raises(ValueError, match="default dtype"):
        result.channel_weights.sum(axis=0, dtype=np.float64)
    with pytest.raises(ValueError, match="axis None, 0, or 1"):
        result.channel_weights.sum(axis=2)

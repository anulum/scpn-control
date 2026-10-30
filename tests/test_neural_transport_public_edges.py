# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural transport public-edge tests.
"""Focused tests for neural transport public surfaces."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_control.core.neural_transport import NeuralTransportModel, TransportInputs


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


_RELOAD_PROBE = """
import importlib, json
import numpy as np
from scpn_control.core.neural_transport import NeuralTransportModel, neural_transport_closure_profiles

importlib.reload(np)
rho = np.linspace(0.1, 0.9, 5)
flat = np.full(5, 5.0)
weights = neural_transport_closure_profiles(
    rho, flat, flat, flat, np.full(5, 2.0), np.full(5, 0.5),
    model=NeuralTransportModel(auto_discover=False),
).channel_weights
print(json.dumps({
    "type": type(weights).__name__,
    "by_radius": weights.sum(axis=0).tolist(),
    "by_channel": weights.sum(axis=1).tolist(),
    "total": float(weights.sum()),
    "initial": weights.sum(axis=0, initial=1.0).tolist(),
    "float32": weights.sum(axis=0, dtype=np.float32).tolist(),
}))
"""


def test_closure_channel_weights_reduce_normally_after_numpy_reload() -> None:
    """Return a plain ndarray whose reductions work even after NumPy is reloaded.

    The reload runs in a fresh interpreter so it cannot disturb this session.
    """
    completed = subprocess.run(
        [sys.executable, "-W", "ignore::UserWarning", "-c", _RELOAD_PROBE],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    reductions = json.loads(completed.stdout.strip().splitlines()[-1])
    assert reductions["type"] == "ndarray"
    assert np.allclose(reductions["by_radius"], np.ones(5))
    assert np.allclose(reductions["by_channel"], np.full(3, 5.0 / 3.0))
    assert reductions["total"] == pytest.approx(5.0)
    assert np.allclose(reductions["initial"], np.full(5, 2.0))
    assert np.allclose(reductions["float32"], np.ones(5))

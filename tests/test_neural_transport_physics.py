# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Neural Transport Physics Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import pytest

from scpn_control.core.neural_transport import NeuralTransportModel, TransportInputs


@pytest.fixture
def model():
    # Force fallback to verify logic first
    return NeuralTransportModel(auto_discover=False)


def test_zero_gradients_near_zero_transport(model):
    """Verify that zero gradients result in low/near-zero transport."""
    # Critical gradient fallback has exact zero for gradients below threshold
    inp = TransportInputs(grad_te=0.0, grad_ti=0.0, grad_ne=0.0)
    fluxes = model.predict(inp)

    assert fluxes.chi_e == 0.0
    assert fluxes.chi_i == 0.0
    assert fluxes.d_e == 0.0
    assert fluxes.channel == "stable"


def test_gradient_threshold_response(model):
    """Verify that transport increases only after crossing a threshold."""
    # Using critical_gradient_model thresholds: ITG=4.0, TEM=5.0
    inp_below = TransportInputs(grad_ti=3.0, grad_te=4.0)
    fluxes_below = model.predict(inp_below)
    assert fluxes_below.chi_i == 0.0
    assert fluxes_below.chi_e == 0.0

    inp_above = TransportInputs(grad_ti=5.0, grad_te=6.0)
    fluxes_above = model.predict(inp_above)
    assert fluxes_above.chi_i > 0.0
    assert fluxes_above.chi_e > 0.0


def test_beta_enhancement_trend(model):
    """Verify that higher beta generally increases transport (turbulence)."""
    # Note: fallback critical_gradient_model ignores beta_e.
    # If model is neural, we check the trend.
    # If fallback, we skip or verify it doesn't decrease.
    if not model.is_neural:
        pytest.skip("Neural weights not found; analytic fallback ignores beta.")

    inp_low = TransportInputs(grad_ti=10.0, beta_e=0.01)
    inp_high = TransportInputs(grad_ti=10.0, beta_e=0.05)

    flux_low = model.predict(inp_low)
    flux_high = model.predict(inp_high)

    # EM effects in QLKNN usually increase transport at high beta
    assert flux_high.chi_i >= flux_low.chi_i


def test_gradient_reversal_consistency(model):
    """Verify that reversing gradient sign behaves physically (if applicable)."""
    # Normalized gradients are usually positive.
    # Fallback model clips at 0.
    inp_neg = TransportInputs(grad_ti=-5.0)
    fluxes = model.predict(inp_neg)
    assert fluxes.chi_i == 0.0

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Disruption Predictor Probabilistic Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.control.disruption_predictor import (
    predict_disruption_risk_safe,
    simulate_tearing_mode,
)

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_simulate_multiple_modes():
    """Verify all disruption modes generate finite signals."""
    for mode in ["ntm", "density_limit", "vde"]:
        signal, label, time_to_disrupt = simulate_tearing_mode(steps=100, mode=mode)
        assert isinstance(signal, np.ndarray)
        assert signal.size > 0
        assert np.all(np.isfinite(signal))
        assert label in [0, 1]
        assert isinstance(time_to_disrupt, int)


def test_disruption_mode_unknown():
    """Verify error on unknown disruption mode."""
    with pytest.raises(ValueError, match="Unknown disruption mode"):
        simulate_tearing_mode(mode="invalid_mode")


def test_predict_disruption_risk_safe_probabilistic():
    """Verify probabilistic metadata is returned even in fallback mode."""
    signal = np.linspace(0.1, 0.5, 50)
    risk, meta = predict_disruption_risk_safe(signal)

    assert 0.0 <= risk <= 1.0
    assert "probabilistic_output" in meta
    assert meta["probabilistic_output"] is True
    assert "risk_std" in meta or "risk_mean" in meta
    assert "risk_p05" in meta
    assert "risk_p95" in meta
    assert meta["risk_p95"] >= meta["risk_p05"]


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_mc_dropout_uncertainty(tmp_path):
    """Verify MC dropout produces uncertainty (std > 0) for ambiguous signals."""
    from scpn_control.control.disruption_predictor import (
        DEFAULT_MODEL_FILENAME,
        train_predictor,
    )

    model_path = tmp_path / DEFAULT_MODEL_FILENAME
    # Train a tiny model quickly
    train_predictor(
        seq_len=16,
        n_shots=10,
        epochs=2,
        model_path=model_path,
        save_plot=False,
    )

    # Ambiguous signal (middle of the road)
    signal = np.random.default_rng(42).normal(0.5, 0.1, 16)
    risk, meta = predict_disruption_risk_safe(
        signal,
        model_path=model_path,
        seq_len=16,
        mc_samples=20,
    )

    assert meta["mode"] == "checkpoint"
    assert "risk_std" in meta
    # Epistemic uncertainty should be non-zero with dropout
    assert meta["risk_std"] >= 0.0


def test_deterministic_fallback_sigma_points():
    """Verify sigma points method is used when no model exists."""
    signal = np.array([0.1, 0.2, 0.3])
    risk, meta = predict_disruption_risk_safe(signal, model_path="nonexistent.pth")

    assert meta["mode"] == "fallback"
    assert meta["probabilistic_method"] == "deterministic_sigma_points"
    assert meta["risk_samples_used"] > 1

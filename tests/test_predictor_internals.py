# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Disruption Predictor Internal Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for DisruptionTransformer forward validation, train_predictor
save_plot path, and predict_disruption_risk_safe inference failure path."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scpn_control.control.disruption_predictor import (
    DisruptionTransformer,
    predict_disruption_risk_safe,
    train_predictor,
)


class TestDisruptionTransformerForward:
    def test_valid_input(self):
        model = DisruptionTransformer(seq_len=32)
        model.eval()
        x = torch.randn(2, 32, 1)
        out = model(x)
        assert out.shape == (2, 1)
        assert (out >= 0).all() and (out <= 1).all()

    def test_rank2_raises(self):
        model = DisruptionTransformer(seq_len=32)
        with pytest.raises(ValueError, match="rank"):
            model(torch.randn(32, 1))

    def test_zero_length_raises(self):
        model = DisruptionTransformer(seq_len=32)
        with pytest.raises(ValueError, match="length must be >= 1"):
            model(torch.randn(1, 0, 1))

    def test_wrong_feature_dim_raises(self):
        model = DisruptionTransformer(seq_len=32)
        with pytest.raises(ValueError, match="feature dimension must be 1"):
            model(torch.randn(1, 16, 3))

    def test_exceeds_seq_len_raises(self):
        model = DisruptionTransformer(seq_len=16)
        with pytest.raises(ValueError, match="exceeds configured seq_len"):
            model(torch.randn(1, 32, 1))

    def test_shorter_than_seq_len_ok(self):
        model = DisruptionTransformer(seq_len=64)
        model.eval()
        out = model(torch.randn(1, 16, 1))
        assert out.shape == (1, 1)


class TestTrainPredictorPlot:
    def test_train_with_save_plot(self, tmp_path):
        model, info = train_predictor(
            seq_len=32, n_shots=8, epochs=2,
            model_path=str(tmp_path / "model.pth"),
            seed=0, save_plot=True,
        )
        assert model is not None
        assert info["epochs"] == 2

    def test_train_no_save_plot(self, tmp_path):
        model, info = train_predictor(
            seq_len=32, n_shots=8, epochs=2,
            model_path=str(tmp_path / "model.pth"),
            seed=0, save_plot=False,
        )
        assert model is not None


class TestPredictSafeInferenceFailure:
    def test_checkpoint_mode_with_model(self, tmp_path):
        """Train a model, then use it via predict_disruption_risk_safe."""
        model_path = str(tmp_path / "safe_test.pth")
        train_predictor(
            seq_len=32, n_shots=8, epochs=2,
            model_path=model_path, seed=0, save_plot=False,
        )
        signal = np.ones(100) * 0.5
        risk, meta = predict_disruption_risk_safe(
            signal, model_path=model_path, seq_len=32,
        )
        assert 0.0 <= risk <= 1.0
        assert meta["mode"] in ("checkpoint", "fallback")

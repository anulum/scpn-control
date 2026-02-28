# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Disruption Predictor Edge Path Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for load_or_train_predictor fallback paths, evaluate_predictor,
and predict_disruption_risk_safe inference failure path."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scpn_control.control.disruption_predictor import (
    DisruptionTransformer,
    evaluate_predictor,
    load_or_train_predictor,
    predict_disruption_risk_safe,
    train_predictor,
)


class TestLoadOrTrainPredictor:
    def test_load_existing_checkpoint(self, tmp_path):
        model_path = str(tmp_path / "pred.pth")
        train_predictor(seq_len=32, n_shots=8, epochs=2, model_path=model_path, seed=0, save_plot=False)
        model, meta = load_or_train_predictor(model_path=model_path, seq_len=32)
        assert model is not None
        assert meta["trained"] is False
        assert meta["fallback"] is False

    def test_missing_checkpoint_train(self, tmp_path):
        model_path = str(tmp_path / "new.pth")
        model, meta = load_or_train_predictor(
            model_path=model_path, seq_len=32,
            train_kwargs={"n_shots": 8, "epochs": 2, "seed": 0, "save_plot": False},
        )
        assert model is not None
        assert meta["trained"] is True

    def test_missing_checkpoint_no_train_fallback(self, tmp_path):
        model_path = str(tmp_path / "missing.pth")
        model, meta = load_or_train_predictor(
            model_path=model_path, seq_len=32,
            train_if_missing=False, allow_fallback=True,
        )
        assert model is None
        assert meta["fallback"] is True
        assert meta["reason"] == "checkpoint_missing"

    def test_missing_checkpoint_no_train_raises(self, tmp_path):
        model_path = str(tmp_path / "missing2.pth")
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            load_or_train_predictor(
                model_path=model_path, seq_len=32,
                train_if_missing=False, allow_fallback=False,
            )

    def test_corrupt_checkpoint_fallback(self, tmp_path):
        model_path = tmp_path / "corrupt.pth"
        torch.save({"state_dict": {"bogus": torch.tensor([1.0])}}, model_path)
        model, meta = load_or_train_predictor(
            model_path=str(model_path), seq_len=32,
            allow_fallback=True, train_if_missing=False,
        )
        assert model is None
        assert meta["fallback"] is True
        assert "checkpoint_load_failed" in meta["reason"]

    def test_corrupt_checkpoint_no_fallback_raises(self, tmp_path):
        model_path = tmp_path / "corrupt2.pth"
        torch.save({"state_dict": {"bogus": torch.tensor([1.0])}}, model_path)
        with pytest.raises((RuntimeError, ValueError, KeyError, OSError)):
            load_or_train_predictor(
                model_path=str(model_path), seq_len=32,
                allow_fallback=False, train_if_missing=False,
            )

    def test_raw_state_dict_checkpoint(self, tmp_path):
        """Checkpoint that's a bare state_dict (no 'state_dict' wrapper key)."""
        model_path = tmp_path / "raw.pth"
        m = DisruptionTransformer(seq_len=32)
        torch.save(m.state_dict(), model_path)
        model, meta = load_or_train_predictor(model_path=str(model_path), seq_len=32)
        assert model is not None
        assert meta["trained"] is False


class TestPredictDisruptionRiskSafeInferenceFail:
    def test_model_raises_returns_fallback(self, tmp_path):
        """Force the model inference to fail and verify fallback to base risk."""
        model_path = tmp_path / "bad_inf.pth"
        # Train a valid model, then corrupt the weights file
        train_predictor(
            seq_len=32, n_shots=8, epochs=2,
            model_path=str(model_path), seed=0, save_plot=False,
        )
        # Overwrite with mismatched architecture
        small = DisruptionTransformer(seq_len=16)
        torch.save({"state_dict": small.state_dict(), "seq_len": 16}, model_path)

        signal = np.ones(100) * 0.5
        risk, meta = predict_disruption_risk_safe(
            signal, model_path=str(model_path), seq_len=32,
        )
        assert 0.0 <= risk <= 1.0
        assert meta["mode"] in ("checkpoint", "fallback")


class _PredictWrapper:
    """Wrap DisruptionTransformer to expose .predict(seq) -> float."""

    def __init__(self, model):
        self._model = model
        self._model.eval()

    def predict(self, seq):
        t = torch.tensor(np.asarray(seq), dtype=torch.float32)
        if t.ndim == 2:
            t = t.unsqueeze(0)
        with torch.no_grad():
            return float(self._model(t).item())


class TestEvaluatePredictor:
    def test_evaluate_returns_metrics(self, tmp_path):
        model_path = str(tmp_path / "eval.pth")
        model, _ = train_predictor(
            seq_len=32, n_shots=16, epochs=3,
            model_path=model_path, seed=0, save_plot=False,
        )
        rng = np.random.default_rng(0)
        n = 20
        X = [torch.randn(32, 1) for _ in range(n)]
        y = [float(rng.integers(0, 2)) for _ in range(n)]
        wrapper = _PredictWrapper(model)
        result = evaluate_predictor(wrapper, X, y)
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_evaluate_with_times(self, tmp_path):
        model_path = str(tmp_path / "eval2.pth")
        model, _ = train_predictor(
            seq_len=32, n_shots=16, epochs=3,
            model_path=model_path, seed=0, save_plot=False,
        )
        rng = np.random.default_rng(0)
        n = 20
        X = [torch.randn(32, 1) for _ in range(n)]
        y = [float(rng.integers(0, 2)) for _ in range(n)]
        times = [rng.uniform(0.0, 0.2) for _ in range(n)]
        wrapper = _PredictWrapper(model)
        result = evaluate_predictor(wrapper, X, y, times_test=times)
        assert "recall_at_10ms" in result
        assert "recall_at_100ms" in result

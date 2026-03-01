# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Disruption predictor fallback path tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for train_predictor torch guard (549), load_or_train fallback
(704-707), predict_disruption_risk_safe inference fallback (764-769)."""

from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pytest

from scpn_control.control.disruption_predictor import (
    load_or_train_predictor,
    predict_disruption_risk_safe,
)


class TestTrainPredictorTorchGuard:
    def test_torch_missing_raises(self, monkeypatch):
        """train_predictor without torch raises RuntimeError (line 549)."""
        import scpn_control.control.disruption_predictor as dp

        original_torch = dp.torch
        monkeypatch.setattr(dp, "torch", None)
        monkeypatch.setattr(dp, "optim", None)
        try:
            with pytest.raises(RuntimeError, match="Torch is required"):
                dp.train_predictor(n_shots=8, epochs=1)
        finally:
            monkeypatch.setattr(dp, "torch", original_torch)


class TestLoadOrTrainFallback:
    def test_train_failure_with_fallback(self, tmp_path, monkeypatch):
        """Training failure with allow_fallback=True returns None (lines 704-707)."""
        import scpn_control.control.disruption_predictor as dp

        # Ensure torch appears available so the code reaches train_predictor
        if dp.torch is None:
            monkeypatch.setattr(dp, "torch", type(sys)("_fake_torch"))
        fake_path = tmp_path / "nonexistent.pt"
        with patch(
            "scpn_control.control.disruption_predictor.train_predictor",
            side_effect=RuntimeError("mock training failure"),
        ):
            model, info = load_or_train_predictor(
                model_path=fake_path,
                force_retrain=True,
                allow_fallback=True,
                train_if_missing=True,
            )
        assert model is None
        assert info["fallback"] is True
        assert "train_failed" in info["reason"]


class TestPredictDisruptionRiskSafeFallback:
    def test_inference_failure_returns_base_risk(self, tmp_path):
        """Model inference failure falls back to base risk (lines 764-769)."""
        signal = np.random.default_rng(42).normal(0.0, 0.1, size=100)
        risk, meta = predict_disruption_risk_safe(
            signal,
            model_path=tmp_path / "nonexistent.pt",
            train_if_missing=False,
        )
        assert 0.0 <= risk <= 1.0
        assert meta["mode"] == "fallback"

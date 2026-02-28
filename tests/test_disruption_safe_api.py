# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Disruption Safe API Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Tests for load_or_train_predictor and predict_disruption_risk_safe."""
from __future__ import annotations

import numpy as np
import pytest

import scpn_control.control.disruption_predictor as dp_mod
from scpn_control.control.disruption_predictor import (
    load_or_train_predictor,
    predict_disruption_risk_safe,
)


# ── Fallback tests: monkeypatch torch away ──────────────────────────

class TestLoadOrTrainPredictorFallback:
    def test_returns_none_model_when_torch_absent(self, monkeypatch):
        monkeypatch.setattr(dp_mod, "torch", None)
        model, meta = load_or_train_predictor(allow_fallback=True)
        assert model is None
        assert meta["fallback"] is True
        assert meta["reason"] == "torch_unavailable"

    def test_no_fallback_raises_when_torch_absent(self, monkeypatch):
        monkeypatch.setattr(dp_mod, "torch", None)
        with pytest.raises(RuntimeError, match="Torch is required"):
            load_or_train_predictor(allow_fallback=False)

    def test_custom_model_path_in_meta(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dp_mod, "torch", None)
        _, meta = load_or_train_predictor(
            model_path=str(tmp_path / "custom.pth"), allow_fallback=True,
        )
        assert "custom.pth" in meta["model_path"]

    def test_seq_len_propagated(self, monkeypatch):
        monkeypatch.setattr(dp_mod, "torch", None)
        _, meta = load_or_train_predictor(seq_len=64, allow_fallback=True)
        assert meta["seq_len"] == 64

    def test_none_path_uses_default(self, monkeypatch):
        monkeypatch.setattr(dp_mod, "torch", None)
        _, meta = load_or_train_predictor(model_path=None, allow_fallback=True)
        assert "disruption_model.pth" in meta["model_path"]


class TestLoadOrTrainMissingCheckpoint:
    def test_missing_checkpoint_no_train_fallback(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dp_mod, "torch", None)
        _, meta = load_or_train_predictor(
            model_path=str(tmp_path / "nonexistent.pth"),
            train_if_missing=False,
            allow_fallback=True,
        )
        assert meta["fallback"] is True
        assert meta["reason"] == "torch_unavailable"

    def test_missing_checkpoint_no_train_no_fallback_raises(self, tmp_path):
        """When torch is available but checkpoint missing and no fallback allowed."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            load_or_train_predictor(
                model_path=str(tmp_path / "nonexistent.pth"),
                train_if_missing=False,
                allow_fallback=False,
            )

    def test_missing_checkpoint_train_if_missing_false_fallback(self, tmp_path):
        """When torch available, checkpoint missing, train=False, fallback=True."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        model, meta = load_or_train_predictor(
            model_path=str(tmp_path / "nonexistent.pth"),
            train_if_missing=False,
            allow_fallback=True,
        )
        assert model is None
        assert meta["fallback"] is True
        assert meta["reason"] == "checkpoint_missing"


# ── predict_disruption_risk_safe: fallback path ─────────────────────

class TestPredictDisruptionRiskSafeFallback:
    def test_returns_bounded_risk_fallback(self, monkeypatch):
        monkeypatch.setattr(dp_mod, "torch", None)
        signal = np.ones(100) * 0.5
        risk, meta = predict_disruption_risk_safe(signal)
        assert 0.0 <= risk <= 1.0
        assert meta["mode"] == "fallback"
        assert meta["risk_source"] == "predict_disruption_risk"

    def test_with_toroidal_fallback(self, monkeypatch):
        monkeypatch.setattr(dp_mod, "torch", None)
        signal = np.linspace(0.1, 5.0, 100)
        toroidal = {"toroidal_n1_amp": 1.0, "toroidal_n2_amp": 0.5, "toroidal_n3_amp": 0.3}
        risk, meta = predict_disruption_risk_safe(signal, toroidal)
        assert 0.0 <= risk <= 1.0
        assert meta["fallback"] is True

    def test_custom_seq_len_fallback(self, monkeypatch):
        monkeypatch.setattr(dp_mod, "torch", None)
        signal = np.ones(200)
        risk, meta = predict_disruption_risk_safe(signal, seq_len=64)
        assert 0.0 <= risk <= 1.0
        assert meta["seq_len"] == 64


# ── predict_disruption_risk_safe: with torch (if available) ──────────

class TestPredictDisruptionRiskSafeWithTorch:
    def test_returns_bounded_risk(self):
        signal = np.ones(100) * 0.5
        risk, meta = predict_disruption_risk_safe(signal)
        assert 0.0 <= risk <= 1.0

    def test_deterministic(self):
        signal = np.linspace(0.5, 1.5, 80)
        r1, _ = predict_disruption_risk_safe(signal)
        r2, _ = predict_disruption_risk_safe(signal)
        assert r1 == r2

    def test_meta_contains_mode(self):
        signal = np.ones(100)
        _, meta = predict_disruption_risk_safe(signal)
        assert "mode" in meta
        assert meta["mode"] in ("fallback", "checkpoint")

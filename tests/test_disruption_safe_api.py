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
            model_path=str(tmp_path / "custom.pth"),
            allow_fallback=True,
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
            import torch  # noqa: F401
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
            import torch  # noqa: F401
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

    def test_fallback_returns_probabilistic_metadata(self, monkeypatch):
        monkeypatch.setattr(dp_mod, "torch", None)
        signal = np.linspace(0.2, 1.8, 96)
        risk, meta = predict_disruption_risk_safe(signal)
        assert 0.0 <= risk <= 1.0
        assert meta["probabilistic_output"] is True
        assert meta["probabilistic_method"] == "deterministic_sigma_points"
        assert 0.0 <= meta["risk_p05"] <= meta["risk_p95"] <= 1.0
        assert meta["risk_interval"][0] == pytest.approx(meta["risk_p05"])
        assert meta["risk_interval"][1] == pytest.approx(meta["risk_p95"])
        assert meta["risk_samples_used"] == len(dp_mod.PROBABILISTIC_SIGMA_LEVELS)
        assert meta["risk_std"] >= 0.0


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

    def test_checkpoint_like_path_returns_probabilistic_metadata(self, monkeypatch):
        class _FakeNoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        class _FakeTorch:
            float32 = np.float32

            @staticmethod
            def tensor(data, dtype=None):
                return np.asarray(data, dtype=np.float32 if dtype is not None else float)

            @staticmethod
            def no_grad():
                return _FakeNoGrad()

        class _FakeTensor:
            def __init__(self, value: float) -> None:
                self._value = value

            def item(self) -> float:
                return self._value

        class _FakeModel:
            def eval(self) -> None:
                return None

            def __call__(self, x):
                risk = float(np.clip(0.25 + 0.35 * float(np.mean(np.asarray(x))), 0.0, 1.0))
                return _FakeTensor(risk)

        monkeypatch.setattr(dp_mod, "torch", _FakeTorch())
        monkeypatch.setattr(
            dp_mod,
            "load_or_train_predictor",
            lambda **_: (_FakeModel(), {"seq_len": 64, "trained": True, "fallback": False}),
        )
        signal = np.linspace(0.1, 1.1, 90)
        risk, meta = predict_disruption_risk_safe(signal)
        assert 0.0 <= risk <= 1.0
        assert meta["mode"] == "checkpoint"
        assert meta["risk_source"] == "transformer_mc_dropout"
        assert meta["probabilistic_output"] is True
        assert meta["probabilistic_method"] == "transformer_mc_plus_sigma_points"
        assert 0.0 <= meta["risk_p05"] <= meta["risk_p95"] <= 1.0
        assert meta["risk_samples_used"] >= len(dp_mod.PROBABILISTIC_SIGMA_LEVELS) + 1
        assert meta["risk_std"] >= 0.0

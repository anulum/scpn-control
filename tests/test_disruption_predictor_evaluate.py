# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Disruption predictor evaluate + inference fallback tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Coverage for evaluate_predictor: times_test branch and zero-recall case."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from scpn_control.control.disruption_predictor import (
    evaluate_predictor,
    load_or_train_predictor,
    predict_disruption_risk_safe,
    run_anomaly_alarm_campaign,
)

try:
    import torch as _torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class _StubModel:
    """Minimal predict-capable model for evaluate_predictor tests."""

    def __init__(self, threshold: float = 0.5):
        self._threshold = threshold

    def predict(self, seq):
        return float(np.mean(seq))


class TestEvaluatePredictorTimesTest:
    def test_times_test_with_recall_at_t(self):
        """evaluate_predictor with times_test != None exercises recall@T."""
        model = _StubModel()
        X_test = [np.ones(10) * 0.8, np.ones(10) * 0.2, np.ones(10) * 0.9]
        y_test = [1, 0, 1]
        times_test = [0.025, 0.005, 0.050]
        result = evaluate_predictor(model, X_test, y_test, times_test=times_test, threshold=0.5)
        assert "recall_at_10ms" in result
        assert "recall_at_20ms" in result
        assert "recall_at_30ms" in result
        assert "recall_at_50ms" in result
        assert "recall_at_100ms" in result
        for T_ms in [10, 20, 30, 50, 100]:
            assert 0.0 <= result[f"recall_at_{T_ms}ms"] <= 1.0

    def test_times_test_none_omits_recall_at_keys(self):
        """evaluate_predictor without times_test skips recall@T keys."""
        model = _StubModel()
        X_test = [np.ones(10) * 0.8]
        y_test = [1]
        result = evaluate_predictor(model, X_test, y_test, times_test=None)
        assert "recall_at_10ms" not in result
        assert "accuracy" in result

    def test_zero_recall_when_mask_sum_zero(self):
        """When mask.sum() == 0 (no positive with early-enough warning), recall_at_T = 0."""
        model = _StubModel()
        X_test = [np.ones(10) * 0.8, np.ones(10) * 0.3]
        y_test = [1, 0]
        # time is too short for any T threshold
        times_test = [0.001, 0.001]
        result = evaluate_predictor(model, X_test, y_test, times_test=times_test, threshold=0.5)
        # For T=100ms, the disruption (y=1) at times_test=0.001 is < 0.1s, so mask is empty
        assert result["recall_at_100ms"] == 0.0
        assert result["recall_at_50ms"] == 0.0

    def test_all_predictions_positive_full_recall(self):
        """All positives predicted correctly with sufficient warning time."""
        model = _StubModel()
        X_test = [np.ones(10) * 0.9, np.ones(10) * 0.8]
        y_test = [1, 1]
        times_test = [0.200, 0.300]
        result = evaluate_predictor(model, X_test, y_test, times_test=times_test, threshold=0.5)
        assert result["recall_at_10ms"] == 1.0
        assert result["recall_at_100ms"] == 1.0
        assert result["recall"] == 1.0


class TestAnomalyCampaignPositiveLabel:
    def test_campaign_covers_positive_alarm(self):
        """run_anomaly_alarm_campaign with mock forcing true positives."""

        def _mock_tearing(steps=1000, *, rng=None):
            sig = np.ones(steps) * 5.0
            return sig, 1, np.zeros(steps)

        with patch(
            "scpn_control.control.disruption_predictor.simulate_tearing_mode",
            side_effect=_mock_tearing,
        ):
            result = run_anomaly_alarm_campaign(
                window=50,
                episodes=10,
                seed=42,
                threshold=0.1,
            )
        assert result["true_positive_rate"] > 0.0


class TestLoadOrTrainNoFallback:
    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
    def test_train_failure_no_fallback_raises(self, tmp_path):
        with (
            patch(
                "scpn_control.control.disruption_predictor.train_predictor",
                side_effect=RuntimeError("mock train failure"),
            ),
            pytest.raises(RuntimeError, match="mock train failure"),
        ):
            load_or_train_predictor(
                model_path=tmp_path / "x.pt",
                force_retrain=True,
                allow_fallback=False,
                train_if_missing=True,
            )


class TestPredictSafeInferenceFailure:
    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
    def test_model_inference_failure_falls_back(self):
        class _BrokenModel:
            def eval(self):
                pass

            def __call__(self, x):
                raise RuntimeError("inference exploded")

        signal = np.random.default_rng(42).normal(0.0, 0.1, size=100)
        with patch(
            "scpn_control.control.disruption_predictor.load_or_train_predictor",
            return_value=(_BrokenModel(), {"seq_len": 50, "trained": True}),
        ):
            risk, meta = predict_disruption_risk_safe(signal)

        assert 0.0 <= risk <= 1.0
        assert meta["mode"] == "fallback"
        assert "inference_failed" in meta["reason"]

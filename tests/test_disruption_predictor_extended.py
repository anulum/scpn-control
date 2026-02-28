# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Extended disruption predictor tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Tests for HybridAnomalyDetector, fault campaigns, signal helpers, and evaluate_predictor."""

import numpy as np
import pytest

from scpn_control.control.disruption_predictor import (
    HybridAnomalyDetector,
    _normalize_fault_campaign_inputs,
    _prepare_signal_window,
    _require_int,
    _synthetic_control_signal,
    default_model_path,
    evaluate_predictor,
    run_anomaly_alarm_campaign,
    run_fault_noise_campaign,
    simulate_tearing_mode,
)


# ── _require_int edge cases ──────────────────────────────────────────

class TestRequireInt:
    def test_accepts_int(self):
        assert _require_int("x", 5) == 5

    def test_accepts_numpy_int(self):
        assert _require_int("x", np.int64(7)) == 7

    def test_rejects_float(self):
        with pytest.raises(ValueError):
            _require_int("x", 3.5)

    def test_rejects_bool(self):
        with pytest.raises(ValueError):
            _require_int("x", True)

    def test_minimum_enforced(self):
        with pytest.raises(ValueError, match=">="):
            _require_int("x", 3, minimum=5)

    def test_rejects_string(self):
        with pytest.raises(ValueError):
            _require_int("x", "hello")

    def test_minimum_none_rejects_nonint(self):
        with pytest.raises(ValueError, match="must be an integer"):
            _require_int("x", 2.5)


# ── _prepare_signal_window ───────────────────────────────────────────

class TestPrepareSignalWindow:
    def test_exact_length(self):
        sig = np.arange(50, dtype=float)
        out = _prepare_signal_window(sig, 50)
        assert out.shape == (50,)
        np.testing.assert_array_equal(out, sig)

    def test_truncates_long_signal(self):
        sig = np.arange(100, dtype=float)
        out = _prepare_signal_window(sig, 30)
        assert out.shape == (30,)
        np.testing.assert_array_equal(out, sig[:30])

    def test_pads_short_signal(self):
        sig = np.array([1.0, 2.0, 3.0])
        out = _prepare_signal_window(sig, 10)
        assert out.shape == (10,)
        assert out[0] == 1.0
        assert out[-1] == 3.0  # edge-padded

    def test_rejects_bad_seq_len(self):
        with pytest.raises(ValueError):
            _prepare_signal_window(np.ones(10), 3)


# ── _synthetic_control_signal ────────────────────────────────────────

class TestSyntheticControlSignal:
    def test_shape(self):
        rng = np.random.default_rng(0)
        sig = _synthetic_control_signal(rng, 128)
        assert sig.shape == (128,)

    def test_positive(self):
        rng = np.random.default_rng(0)
        sig = _synthetic_control_signal(rng, 256)
        assert np.all(sig >= 0.01)

    def test_deterministic(self):
        a = _synthetic_control_signal(np.random.default_rng(42), 100)
        b = _synthetic_control_signal(np.random.default_rng(42), 100)
        np.testing.assert_array_equal(a, b)


# ── _normalize_fault_campaign_inputs ────────────────────────────────

class TestNormalizeFaultCampaignInputs:
    def test_valid_inputs(self):
        result = _normalize_fault_campaign_inputs(42, 10, 64, 0.03, 5, 4, 0.05)
        assert result == (42, 10, 64, 0.03, 5, 4, 0.05)

    def test_rejects_negative_noise(self):
        with pytest.raises(ValueError, match="noise_std"):
            _normalize_fault_campaign_inputs(0, 1, 16, -0.1, 1, 1, 0.01)

    def test_rejects_nonpositive_recovery_epsilon(self):
        with pytest.raises(ValueError, match="recovery_epsilon"):
            _normalize_fault_campaign_inputs(0, 1, 16, 0.01, 1, 1, 0.0)

    def test_rejects_small_window(self):
        with pytest.raises(ValueError, match=">="):
            _normalize_fault_campaign_inputs(0, 1, 8, 0.01, 1, 1, 0.01)


# ── HybridAnomalyDetector ───────────────────────────────────────────

class TestHybridAnomalyDetector:
    def test_score_returns_keys(self):
        det = HybridAnomalyDetector()
        result = det.score(np.ones(50) * 0.5)
        assert "supervised_score" in result
        assert "unsupervised_score" in result
        assert "anomaly_score" in result
        assert "alarm" in result

    def test_anomaly_score_bounded(self):
        det = HybridAnomalyDetector()
        for _ in range(20):
            r = det.score(np.random.default_rng(0).uniform(0, 5, size=50))
            assert 0.0 <= r["anomaly_score"] <= 1.0

    def test_first_call_unsupervised_zero(self):
        det = HybridAnomalyDetector()
        r = det.score(np.ones(30))
        assert r["unsupervised_score"] == 0.0

    def test_subsequent_call_has_unsupervised(self):
        det = HybridAnomalyDetector()
        det.score(np.ones(30))
        r = det.score(np.ones(30) * 10.0)
        assert r["unsupervised_score"] > 0.0

    def test_alarm_when_high_risk(self):
        det = HybridAnomalyDetector(threshold=0.3)
        sig = np.linspace(0.1, 10.0, 100)
        obs = {"toroidal_n1_amp": 2.0, "toroidal_n2_amp": 1.5, "toroidal_n3_amp": 1.0}
        r = det.score(sig, obs)
        assert r["alarm"] is True

    def test_rejects_invalid_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            HybridAnomalyDetector(threshold=1.5)

    def test_rejects_invalid_ema(self):
        with pytest.raises(ValueError, match="ema"):
            HybridAnomalyDetector(ema=0.0)

    def test_rejects_negative_ema(self):
        with pytest.raises(ValueError, match="ema"):
            HybridAnomalyDetector(ema=-0.1)


# ── run_fault_noise_campaign ────────────────────────────────────────

class TestRunFaultNoiseCampaign:
    def test_returns_expected_keys(self):
        result = run_fault_noise_campaign(seed=0, episodes=4, window=32)
        assert "mean_abs_risk_error" in result
        assert "p95_abs_risk_error" in result
        assert "recovery_success_rate" in result
        assert "passes_thresholds" in result
        assert "fault_count" in result

    def test_deterministic(self):
        a = run_fault_noise_campaign(seed=42, episodes=4, window=32)
        b = run_fault_noise_campaign(seed=42, episodes=4, window=32)
        assert a["mean_abs_risk_error"] == b["mean_abs_risk_error"]

    def test_fault_count_positive(self):
        r = run_fault_noise_campaign(seed=0, episodes=8, window=64, bit_flip_interval=5)
        assert r["fault_count"] > 0

    def test_error_bounded(self):
        r = run_fault_noise_campaign(seed=0, episodes=4, window=32)
        assert r["mean_abs_risk_error"] >= 0.0
        assert r["p95_abs_risk_error"] >= r["mean_abs_risk_error"]


# ── run_anomaly_alarm_campaign ───────────────────────────────────────

class TestRunAnomalyAlarmCampaign:
    def test_returns_expected_keys(self):
        r = run_anomaly_alarm_campaign(seed=0, episodes=4, window=64)
        assert "true_positive_rate" in r
        assert "false_positive_rate" in r
        assert "passes_thresholds" in r

    def test_deterministic(self):
        a = run_anomaly_alarm_campaign(seed=42, episodes=4, window=64)
        b = run_anomaly_alarm_campaign(seed=42, episodes=4, window=64)
        assert a["true_positive_rate"] == b["true_positive_rate"]

    def test_rates_bounded(self):
        r = run_anomaly_alarm_campaign(seed=0, episodes=8, window=64)
        assert 0.0 <= r["true_positive_rate"] <= 1.0
        assert 0.0 <= r["false_positive_rate"] <= 1.0

    def test_rejects_invalid_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            run_anomaly_alarm_campaign(threshold=1.5)


# ── simulate_tearing_mode edge cases ─────────────────────────────────

class TestSimulateTearingModeExtended:
    def test_disruption_returns_label_1(self):
        for seed in range(200):
            rng = np.random.default_rng(seed)
            sig, label, ttd = simulate_tearing_mode(2000, rng=rng)
            if label == 1:
                assert ttd >= 0
                assert len(sig) <= 2000
                return
        pytest.skip("no disruption found in 200 seeds")

    def test_safe_returns_label_0(self):
        for seed in range(100):
            rng = np.random.default_rng(seed)
            sig, label, ttd = simulate_tearing_mode(1000, rng=rng)
            if label == 0:
                assert ttd == -1
                assert len(sig) == 1000
                break
        else:
            pytest.skip("no safe shot found in 100 seeds")


# ── evaluate_predictor ───────────────────────────────────────────────

class _DummyModel:
    """Minimal model that thresholds on max(seq)."""
    def predict(self, seq):
        return float(np.max(seq))


class TestEvaluatePredictor:
    def test_returns_expected_keys(self):
        model = _DummyModel()
        X = [np.array([0.3]), np.array([0.8])]
        y = [0, 1]
        result = evaluate_predictor(model, X, y)
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert "confusion_matrix" in result

    def test_perfect_model(self):
        model = _DummyModel()
        X = [np.array([0.2]), np.array([0.3]), np.array([0.8]), np.array([0.9])]
        y = [0, 0, 1, 1]
        result = evaluate_predictor(model, X, y, threshold=0.5)
        assert result["accuracy"] == 1.0
        assert result["recall"] == 1.0
        assert result["false_positive_rate"] == 0.0

    def test_recall_at_T(self):
        model = _DummyModel()
        X = [np.array([0.8]), np.array([0.9])]
        y = [1, 1]
        times = [0.025, 0.060]
        result = evaluate_predictor(model, X, y, times_test=times, threshold=0.5)
        assert "recall_at_10ms" in result
        assert "recall_at_50ms" in result
        assert result["recall_at_50ms"] == 1.0

    def test_f1_computed(self):
        model = _DummyModel()
        X = [np.array([0.1]), np.array([0.9])]
        y = [1, 0]
        result = evaluate_predictor(model, X, y, threshold=0.5)
        assert 0.0 <= result["f1"] <= 1.0


# ── default_model_path ───────────────────────────────────────────────

class TestDefaultModelPath:
    def test_returns_path(self):
        p = default_model_path()
        assert p.name == "disruption_model.pth"
        assert "artifacts" in str(p)

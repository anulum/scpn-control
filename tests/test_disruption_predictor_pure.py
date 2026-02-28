# Tests for pure-function disruption predictor components (no PyTorch needed).

import numpy as np
import pytest

from scpn_control.control.disruption_predictor import (
    apply_bit_flip_fault,
    build_disruption_feature_vector,
    predict_disruption_risk,
    simulate_tearing_mode,
)


class TestSimulateTearingMode:
    def test_returns_triple(self):
        signal, label, ttd = simulate_tearing_mode(steps=200, rng=np.random.default_rng(0))
        assert signal.ndim == 1
        assert signal.size <= 200  # may terminate early on disruption
        assert label in (0, 1)
        assert isinstance(ttd, (int, np.integer))

    def test_deterministic_with_seed(self):
        s1, l1, t1 = simulate_tearing_mode(steps=100, rng=np.random.default_rng(42))
        s2, l2, t2 = simulate_tearing_mode(steps=100, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(s1, s2)
        assert l1 == l2

    def test_signal_finite(self):
        signal, _, _ = simulate_tearing_mode(steps=500, rng=np.random.default_rng(7))
        assert np.all(np.isfinite(signal))

    def test_rejects_bad_steps(self):
        with pytest.raises(ValueError):
            simulate_tearing_mode(steps=0)


class TestBuildDisruptionFeatureVector:
    def test_length_11(self):
        sig = np.ones(50)
        feats = build_disruption_feature_vector(sig)
        assert len(feats) == 11

    def test_with_toroidal(self):
        sig = np.random.default_rng(0).normal(size=100)
        toroidal = {
            "toroidal_n1_amp": 0.3,
            "toroidal_n2_amp": 0.2,
            "toroidal_n3_amp": 0.1,
            "toroidal_asymmetry_index": 0.4,
            "toroidal_radial_spread": 0.05,
        }
        feats = build_disruption_feature_vector(sig, toroidal)
        assert feats[6] == pytest.approx(0.3)  # n1
        assert feats[7] == pytest.approx(0.2)  # n2

    def test_empty_signal_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            build_disruption_feature_vector(np.array([]))

    def test_nan_signal_raises(self):
        with pytest.raises(ValueError, match="finite"):
            build_disruption_feature_vector(np.array([1.0, float("nan"), 2.0]))

    def test_no_toroidal_zeros(self):
        feats = build_disruption_feature_vector(np.ones(30))
        # Without toroidal, n1..spread should be 0
        assert feats[6] == 0.0
        assert feats[10] == 0.0


class TestPredictDisruptionRisk:
    def test_output_bounded(self):
        sig = np.ones(100)
        risk = predict_disruption_risk(sig)
        assert 0.0 <= risk <= 1.0

    def test_high_disturbance_higher_risk(self):
        rng = np.random.default_rng(42)
        sig_calm = np.ones(200) * 0.5
        sig_turb = np.ones(200) * 0.5 + rng.normal(0, 0.5, 200)
        risk_calm = predict_disruption_risk(sig_calm)
        risk_turb = predict_disruption_risk(sig_turb)
        assert risk_turb > risk_calm

    def test_toroidal_increases_risk(self):
        sig = np.ones(100)
        risk_no_tor = predict_disruption_risk(sig)
        risk_with_tor = predict_disruption_risk(sig, {
            "toroidal_n1_amp": 1.0,
            "toroidal_n2_amp": 0.5,
            "toroidal_n3_amp": 0.3,
            "toroidal_asymmetry_index": 1.2,
            "toroidal_radial_spread": 0.1,
        })
        assert risk_with_tor > risk_no_tor

    def test_deterministic(self):
        sig = np.linspace(0.5, 1.5, 100)
        r1 = predict_disruption_risk(sig)
        r2 = predict_disruption_risk(sig)
        assert r1 == r2


class TestApplyBitFlipFault:
    def test_flip_and_back(self):
        val = 3.14
        flipped = apply_bit_flip_fault(val, 10)
        assert flipped != val
        restored = apply_bit_flip_fault(flipped, 10)
        assert restored == pytest.approx(val)

    def test_bit_0_changes_lsb(self):
        val = 1.0
        flipped = apply_bit_flip_fault(val, 0)
        # Flipping LSB of mantissa changes value by ~2^-52
        assert abs(flipped - val) < 1e-14

    def test_rejects_bool_index(self):
        with pytest.raises(ValueError, match="integer"):
            apply_bit_flip_fault(1.0, True)

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="\\[0, 63\\]"):
            apply_bit_flip_fault(1.0, 64)
        with pytest.raises(ValueError, match="\\[0, 63\\]"):
            apply_bit_flip_fault(1.0, -1)

    def test_nan_result_returns_original(self):
        # Flipping sign+exponent bits can produce NaN; function returns original
        result = apply_bit_flip_fault(1.0, 63)
        assert np.isfinite(result)

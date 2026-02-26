"""Tests for scpn_control.control.disruption_predictor."""

import numpy as np
import pytest

from scpn_control.control.disruption_predictor import (
    simulate_tearing_mode,
    build_disruption_feature_vector,
    predict_disruption_risk,
    apply_bit_flip_fault,
)


class TestSimulateTearingMode:
    def test_returns_tuple_of_three(self):
        signal, label, ttd = simulate_tearing_mode(200, rng=np.random.default_rng(0))
        assert isinstance(signal, np.ndarray)
        assert label in (0, 1)
        assert isinstance(ttd, (int, float, np.integer))

    def test_deterministic_with_seed(self):
        a = simulate_tearing_mode(100, rng=np.random.default_rng(42))
        b = simulate_tearing_mode(100, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1] == b[1]

    def test_rejects_noninteger_steps(self):
        with pytest.raises(ValueError):
            simulate_tearing_mode(10.5)

    def test_signal_length_matches_or_shorter(self):
        signal, label, ttd = simulate_tearing_mode(500, rng=np.random.default_rng(1))
        assert len(signal) <= 500


class TestBuildDisruptionFeatureVector:
    def test_output_length(self):
        sig = np.ones(100)
        fv = build_disruption_feature_vector(sig)
        assert fv.shape == (11,)

    def test_rejects_empty_signal(self):
        with pytest.raises(ValueError, match="at least one"):
            build_disruption_feature_vector(np.array([]))

    def test_rejects_nan_signal(self):
        with pytest.raises(ValueError, match="finite"):
            build_disruption_feature_vector(np.array([1.0, float("nan"), 2.0]))

    def test_toroidal_observables_included(self):
        sig = np.ones(50)
        obs = {"toroidal_n1_amp": 0.5, "toroidal_n2_amp": 0.3, "toroidal_n3_amp": 0.1}
        fv = build_disruption_feature_vector(sig, obs)
        assert fv[6] == pytest.approx(0.5)
        assert fv[7] == pytest.approx(0.3)

    def test_rejects_nan_observables(self):
        with pytest.raises(ValueError, match="finite"):
            build_disruption_feature_vector(
                np.ones(10),
                {"toroidal_n1_amp": float("nan")},
            )


class TestPredictDisruptionRisk:
    def test_output_range(self):
        sig = np.ones(100) * 0.5
        risk = predict_disruption_risk(sig)
        assert 0.0 <= risk <= 1.0

    def test_high_risk_for_growing_signal(self):
        sig = np.linspace(0.1, 10.0, 200)
        obs = {"toroidal_n1_amp": 2.0, "toroidal_n2_amp": 1.5, "toroidal_n3_amp": 1.0}
        risk = predict_disruption_risk(sig, obs)
        assert risk > 0.5

    def test_low_risk_for_flat_signal(self):
        sig = np.ones(200) * 0.01
        risk = predict_disruption_risk(sig)
        assert risk < 0.5


class TestApplyBitFlipFault:
    def test_flips_and_returns_float(self):
        result = apply_bit_flip_fault(1.0, 0)
        assert isinstance(result, float)

    def test_double_flip_recovers(self):
        original = 3.14
        flipped = apply_bit_flip_fault(original, 10)
        recovered = apply_bit_flip_fault(flipped, 10)
        assert recovered == pytest.approx(original)

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            apply_bit_flip_fault(1.0, 64)
        with pytest.raises(ValueError):
            apply_bit_flip_fault(1.0, -1)

    def test_rejects_non_integer(self):
        with pytest.raises(ValueError):
            apply_bit_flip_fault(1.0, 3.5)

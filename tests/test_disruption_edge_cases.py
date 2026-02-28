# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Disruption Edge Case Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Edge case tests for disruption contracts and anomaly campaigns."""
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.control.advanced_soc_fusion_learning import FusionAIAgent
from scpn_control.control.disruption_contracts import (
    impurity_transport_response,
    mcnp_lite_tbr,
    post_disruption_halo_runaway,
    require_1d_array,
    require_fraction,
    require_int,
    run_real_shot_replay,
    synthetic_disruption_signal,
)
from scpn_control.control.disruption_predictor import (
    HybridAnomalyDetector,
    _require_int,
    build_disruption_feature_vector,
    predict_disruption_risk,
    run_anomaly_alarm_campaign,
    run_fault_noise_campaign,
)


# ── Small window_size triggers the signal_window.size < 8 continue branch ──

def _build_shot(n: int) -> dict:
    t = np.linspace(0.0, 0.01 * n, n, dtype=np.float64)
    return {
        "time_s": t,
        "Ip_MA": np.full(n, 12.0, dtype=np.float64),
        "beta_N": np.full(n, 2.0, dtype=np.float64),
        "n1_amp": np.full(n, 0.05, dtype=np.float64),
        "n2_amp": np.full(n, 0.02, dtype=np.float64),
        "dBdt_gauss_per_s": np.full(n, 0.3, dtype=np.float64),
        "is_disruption": False,
        "disruption_time_idx": -1,
    }


def test_replay_small_window_skips_early_steps():
    """window_size=8 means first loop iterations have signal_window.size < 8."""
    agent = FusionAIAgent(epsilon=0.05)
    shot_data = _build_shot(n=32)
    out = run_real_shot_replay(
        shot_data=shot_data,
        rl_agent=agent,
        risk_threshold=0.90,
        spi_trigger_risk=0.95,
        window_size=8,
    )
    assert out["n_steps"] == 32
    assert out["spi_triggered"] is False


def test_replay_minimum_window_size_8():
    """window_size must be >= 8."""
    agent = FusionAIAgent(epsilon=0.05)
    with pytest.raises(ValueError, match="window_size"):
        run_real_shot_replay(
            shot_data=_build_shot(n=32),
            rl_agent=agent,
            window_size=7,
        )


# ── Anomaly campaign edge cases ──

class TestAnomalyAlarmCampaignEdge:
    def test_minimal_episodes(self):
        r = run_anomaly_alarm_campaign(seed=0, episodes=1, window=32)
        assert r["episodes"] == 1
        assert 0.0 <= r["true_positive_rate"] <= 1.0

    def test_low_threshold_more_alarms(self):
        r_low = run_anomaly_alarm_campaign(seed=42, episodes=8, window=64, threshold=0.10)
        r_high = run_anomaly_alarm_campaign(seed=42, episodes=8, window=64, threshold=0.90)
        # Low threshold should alarm at least as often as high threshold
        assert r_low["true_positive_rate"] >= r_high["true_positive_rate"]


# ── Fault campaign edge cases ──

class TestFaultCampaignEdge:
    def test_single_episode(self):
        r = run_fault_noise_campaign(seed=0, episodes=1, window=32)
        assert r["episodes"] == 1
        assert r["mean_abs_risk_error"] >= 0.0

    def test_high_noise_higher_error(self):
        r_low = run_fault_noise_campaign(seed=42, episodes=4, window=32, noise_std=0.001)
        r_high = run_fault_noise_campaign(seed=42, episodes=4, window=32, noise_std=0.5)
        assert r_high["mean_abs_risk_error"] >= r_low["mean_abs_risk_error"]


# ── Feature vector computed asymmetry index ──

class TestFeatureVectorAsymmetry:
    def test_asymmetry_computed_from_n_amps_when_absent(self):
        sig = np.ones(50)
        obs = {"toroidal_n1_amp": 0.3, "toroidal_n2_amp": 0.4, "toroidal_n3_amp": 0.0}
        fv = build_disruption_feature_vector(sig, obs)
        expected_asym = np.sqrt(0.3**2 + 0.4**2)
        assert fv[9] == pytest.approx(expected_asym, rel=1e-6)

    def test_explicit_asymmetry_overrides_computed(self):
        sig = np.ones(50)
        obs = {
            "toroidal_n1_amp": 0.3,
            "toroidal_n2_amp": 0.4,
            "toroidal_n3_amp": 0.0,
            "toroidal_asymmetry_index": 99.0,
        }
        fv = build_disruption_feature_vector(sig, obs)
        assert fv[9] == pytest.approx(99.0)

    def test_single_sample_signal(self):
        fv = build_disruption_feature_vector(np.array([5.0]))
        assert fv[0] == 5.0  # mean
        assert fv[3] == 0.0  # slope (single sample)


# ── Anomaly detector EMA convergence ──

class TestAnomalyDetectorConvergence:
    def test_ema_tracks_mean(self):
        det = HybridAnomalyDetector(ema=0.5, threshold=0.99)
        for _ in range(50):
            det.score(np.ones(30) * 0.5)
        assert det.initialized is True
        assert det.mean > 0.0

    def test_var_stays_positive(self):
        det = HybridAnomalyDetector()
        for _ in range(20):
            det.score(np.ones(30))
        assert det.var > 0.0


# ── mcnp_lite_tbr edge: out-of-range inputs are clipped ──

class TestMcnpLiteTbrClipping:
    def test_enrichment_over_one_clipped(self):
        tbr1, _ = mcnp_lite_tbr(
            base_tbr=1.0, li6_enrichment=1.0, be_multiplier_fraction=0.5, reflector_albedo=0.5,
        )
        tbr2, _ = mcnp_lite_tbr(
            base_tbr=1.0, li6_enrichment=5.0, be_multiplier_fraction=0.5, reflector_albedo=0.5,
        )
        assert tbr1 == tbr2  # clipped to 1.0 in both cases

    def test_negative_enrichment_clipped_to_zero(self):
        tbr1, _ = mcnp_lite_tbr(
            base_tbr=1.0, li6_enrichment=0.0, be_multiplier_fraction=0.5, reflector_albedo=0.5,
        )
        tbr2, _ = mcnp_lite_tbr(
            base_tbr=1.0, li6_enrichment=-5.0, be_multiplier_fraction=0.5, reflector_albedo=0.5,
        )
        assert tbr1 == tbr2


# ── post_disruption_halo_runaway edge ──

class TestPostDisruptionEdge:
    def test_very_short_tau_cq(self):
        result = post_disruption_halo_runaway(
            pre_current_ma=15.0, tau_cq_s=0.0001,
            disturbance=0.5, mitigation_strength=0.5, zeff_eff=2.0,
        )
        for v in result.values():
            assert np.isfinite(v)

    def test_zero_disturbance_zero_mitigation(self):
        result = post_disruption_halo_runaway(
            pre_current_ma=10.0, tau_cq_s=0.010,
            disturbance=0.0, mitigation_strength=0.0, zeff_eff=1.0,
        )
        assert result["runaway_beam_ma"] >= 0.0
        assert result["halo_current_ma"] >= 0.0


# ── impurity_transport_response: xenon-heavy cocktail ──

class TestImpurityXenonHeavy:
    def test_xenon_increases_zeff(self):
        r_no_xe = impurity_transport_response(
            neon_quantity_mol=0.5, argon_quantity_mol=0.2,
            xenon_quantity_mol=0.0, disturbance=0.5, seed_shift=0,
        )
        r_xe = impurity_transport_response(
            neon_quantity_mol=0.5, argon_quantity_mol=0.2,
            xenon_quantity_mol=0.5, disturbance=0.5, seed_shift=0,
        )
        assert r_xe["zeff_eff"] > r_no_xe["zeff_eff"]


# ── require_int from disruption_predictor: minimum=None message ──

class TestRequireIntNoMinimum:
    def test_no_minimum_message(self):
        with pytest.raises(ValueError, match="must be an integer\\."):
            _require_int("x", 2.5)

    def test_with_minimum_message(self):
        with pytest.raises(ValueError, match="must be an integer >= 5"):
            _require_int("x", 2.5, minimum=5)


# ── synthetic_disruption_signal determinism ──

class TestSyntheticSignalDeterminism:
    def test_deterministic(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        s1, t1 = synthetic_disruption_signal(rng=rng1, disturbance=0.5)
        s2, t2 = synthetic_disruption_signal(rng=rng2, disturbance=0.5)
        np.testing.assert_array_equal(s1, s2)
        assert t1 == t2

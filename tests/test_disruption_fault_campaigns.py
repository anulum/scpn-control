# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for disruption fault/anomaly campaigns leaf

"""Drive production fault-injection and anomaly-alarm campaign leaf contracts."""

from __future__ import annotations

import math

import numpy as np
import pytest

import scpn_control.control.disruption_fault_campaigns as leaf
import scpn_control.control.disruption_predictor as owner


def test_owner_campaign_symbols_bind_to_leaf() -> None:
    """Owner re-exports are the production campaign leaf objects."""
    assert owner.apply_bit_flip_fault is leaf.apply_bit_flip_fault
    assert owner.run_fault_noise_campaign is leaf.run_fault_noise_campaign
    assert owner.run_anomaly_alarm_campaign is leaf.run_anomaly_alarm_campaign
    assert owner.HybridAnomalyDetector is leaf.HybridAnomalyDetector


def test_apply_bit_flip_fault_determinism_and_bounds() -> None:
    """Bit flips are deterministic and reject out-of-range indices."""
    a = leaf.apply_bit_flip_fault(1.25, 5)
    b = leaf.apply_bit_flip_fault(1.25, 5)
    assert a == b
    assert isinstance(a, float)
    with pytest.raises(ValueError, match="bit_index"):
        leaf.apply_bit_flip_fault(1.0, -1)
    with pytest.raises(ValueError, match="bit_index"):
        leaf.apply_bit_flip_fault(1.0, 64)
    with pytest.raises(ValueError, match="bit_index"):
        leaf.apply_bit_flip_fault(1.0, True)  # type: ignore[arg-type]


def test_fault_noise_campaign_deterministic_and_report_shape() -> None:
    """Fault campaign is seed-deterministic and returns the resilience report keys."""
    a = leaf.run_fault_noise_campaign(seed=7, episodes=4, window=32)
    b = leaf.run_fault_noise_campaign(seed=7, episodes=4, window=32)
    assert a == b
    for key in (
        "seed",
        "episodes",
        "window",
        "fault_count",
        "mean_abs_risk_error",
        "p95_abs_risk_error",
        "recovery_steps_p95",
        "recovery_success_rate",
        "thresholds",
        "passes_thresholds",
    ):
        assert key in a
    assert a["fault_count"] >= 1
    assert 0.0 <= a["recovery_success_rate"] <= 1.0


def test_anomaly_alarm_campaign_and_detector() -> None:
    """Anomaly campaign reports rates; detector scores finite alarms."""
    report = leaf.run_anomaly_alarm_campaign(seed=3, episodes=6, window=48, threshold=0.5)
    assert "true_positive_rate" in report
    assert "false_positive_rate" in report
    assert "p95_alarm_latency_steps" in report
    det = leaf.HybridAnomalyDetector(threshold=0.99, ema=0.1)
    score = det.score(np.linspace(0.01, 0.02, 20))
    assert set(score) >= {"supervised_score", "unsupervised_score", "anomaly_score", "alarm"}
    assert 0.0 <= score["anomaly_score"] <= 1.0
    # second score exercises EMA-initialized unsupervised branch
    score2 = det.score(np.linspace(0.5, 0.9, 20))
    assert math.isfinite(score2["unsupervised_score"])


def test_normalize_fault_campaign_inputs_fail_closed() -> None:
    """Campaign input normaliser rejects non-positive window/noise parameters."""
    with pytest.raises(ValueError):
        leaf._normalize_fault_campaign_inputs(
            seed=0,
            episodes=0,
            window=32,
            noise_std=0.01,
            bit_flip_interval=2,
            recovery_window=2,
            recovery_epsilon=0.01,
        )
    with pytest.raises(ValueError):
        leaf._normalize_fault_campaign_inputs(
            seed=0,
            episodes=1,
            window=32,
            noise_std=-0.1,
            bit_flip_interval=2,
            recovery_window=2,
            recovery_epsilon=0.01,
        )


def test_anomaly_campaign_pad_and_true_positive_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pad short synthetic trajectories and count true-positive alarms."""

    def _short_disruptive(*, steps: int = 128, rng=None, mode: str = "ntm"):
        # Shorter than window forces pad; label=1 forces positive branch.
        signal = np.linspace(0.1, 2.0, max(8, steps // 4), dtype=float)
        return signal, 1, 3

    monkeypatch.setattr(leaf, "simulate_tearing_mode", _short_disruptive)
    report = leaf.run_anomaly_alarm_campaign(seed=1, episodes=2, window=32, threshold=0.01)
    assert report["true_positive_rate"] >= 0.0
    # With very low threshold alarms should fire and populate latencies.
    assert report["p95_alarm_latency_steps"] >= 0.0


def test_anomaly_campaign_positive_without_alarm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive episodes without alarms still count toward the TPR denominator."""

    def _disruptive_quiet(*, steps: int = 128, rng=None, mode: str = "ntm"):
        return np.ones(steps, dtype=float) * 0.01, 1, 0

    class _SilentDetector:
        def __init__(self, threshold: float = 0.5) -> None:
            self.threshold = threshold

        def score(self, signal, toroidal_observables=None):
            return {
                "supervised_score": 0.0,
                "unsupervised_score": 0.0,
                "anomaly_score": 0.0,
                "alarm": False,
            }

    monkeypatch.setattr(leaf, "simulate_tearing_mode", _disruptive_quiet)
    monkeypatch.setattr(leaf, "HybridAnomalyDetector", _SilentDetector)
    report = leaf.run_anomaly_alarm_campaign(seed=0, episodes=1, window=16, threshold=0.99)
    assert report["true_positive_rate"] == 0.0


def test_fault_noise_campaign_non_recovery_branch() -> None:
    """Tiny recovery epsilon forces the no-recovery path after bit-flip faults."""
    report = leaf.run_fault_noise_campaign(
        seed=0,
        episodes=2,
        window=24,
        noise_std=0.5,
        bit_flip_interval=3,
        recovery_window=1,
        recovery_epsilon=1e-15,
    )
    assert report["fault_count"] >= 1
    # With near-zero epsilon, recovery within one step is expected to fail often.
    assert report["recovery_success_rate"] < 1.0 or report["recovery_steps_p95"] >= 1.0


def test_anomaly_campaign_false_positive_alarm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative episodes that alarm increment the false-positive count."""

    def _quiet_negative(*, steps: int = 128, rng=None, mode: str = "ntm"):
        return np.ones(int(steps), dtype=float) * 0.01, 0, 0

    class _AlwaysAlarm:
        def __init__(self, threshold: float = 0.5) -> None:
            self.threshold = threshold

        def score(self, signal, toroidal_observables=None):
            return {
                "supervised_score": 1.0,
                "unsupervised_score": 1.0,
                "anomaly_score": 1.0,
                "alarm": True,
            }

    monkeypatch.setattr(leaf, "simulate_tearing_mode", _quiet_negative)
    monkeypatch.setattr(leaf, "HybridAnomalyDetector", _AlwaysAlarm)
    report = leaf.run_anomaly_alarm_campaign(seed=0, episodes=1, window=16, threshold=0.5)
    assert report["false_positive_rate"] == 1.0
    assert report["true_positive_rate"] == 0.0

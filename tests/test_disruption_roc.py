# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tests for the disruption ROC and warning-time core
"""Multi-angle tests for :mod:`scpn_control.control.disruption_roc`."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.control.advanced_soc_fusion_learning import FusionAIAgent
from scpn_control.control.disruption_contracts import run_real_shot_replay
from scpn_control.control.disruption_roc import (
    ShotEvaluation,
    confusion_at_threshold,
    disruption_metrics,
    first_alarm_index,
    roc_auc_from_curve,
    roc_curve,
    score_risk_series,
    warning_time_recall,
)


def _evaluation(
    risk_series: list[float],
    *,
    label: int,
    disruption_time_idx: int,
    window_size: int = 1,
    dt_s: float = 0.001,
) -> ShotEvaluation:
    risk = np.asarray(risk_series, dtype=np.float64)
    time_s = np.arange(risk.shape[0], dtype=np.float64) * dt_s
    return ShotEvaluation(
        risk_series=risk,
        label=label,
        disruption_time_idx=disruption_time_idx,
        time_s=time_s,
        window_size=window_size,
    )


# --------------------------------------------------------------------------- #
# score_risk_series
# --------------------------------------------------------------------------- #
def test_score_risk_series_is_bounded_and_leading_zero() -> None:
    rng = np.random.default_rng(0)
    n = 200
    dbdt = rng.normal(0.0, 1.0, n)
    n1 = np.abs(rng.normal(0.0, 0.3, n))
    n2 = np.abs(rng.normal(0.0, 0.2, n))
    risk = score_risk_series(dbdt, n1, n2, window_size=32)
    assert risk.shape == (n,)
    assert np.all((risk >= 0.0) & (risk <= 1.0))
    assert np.all(risk[:32] == 0.0)
    assert np.all(np.isfinite(risk))


def test_score_risk_series_rejects_bad_window() -> None:
    arr = np.ones(10)
    with pytest.raises(ValueError, match="window_size must be >= 1"):
        score_risk_series(arr, arr, arr, window_size=0)


def test_score_risk_series_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        score_risk_series(np.ones(10), np.ones(9), np.ones(10), window_size=2)


@pytest.mark.parametrize("channel", ["dbdt", "n1", "n2"])
def test_score_risk_series_rejects_non_finite(channel: str) -> None:
    arrays = {"dbdt": np.ones(10), "n1": np.ones(10), "n2": np.ones(10)}
    arrays[channel] = arrays[channel].copy()
    arrays[channel][3] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        score_risk_series(arrays["dbdt"], arrays["n1"], arrays["n2"], window_size=2)


def test_score_risk_series_matches_replay_single_source_of_truth() -> None:
    """The scorer must reproduce run_real_shot_replay's internal risk_series."""
    shot = dict(
        np.load(
            "validation/reference_data/diiid/disruption_shots/shot_155916_locked_mode.npz",
            allow_pickle=False,
        )
    )
    out = run_real_shot_replay(shot_data=shot, rl_agent=FusionAIAgent(epsilon=0.05), window_size=128)
    replay_risk = np.asarray(out["risk_series"], dtype=np.float64)
    direct_risk = score_risk_series(
        np.asarray(shot["dBdt_gauss_per_s"], dtype=np.float64),
        np.asarray(shot["n1_amp"], dtype=np.float64),
        np.asarray(shot["n2_amp"], dtype=np.float64),
        window_size=128,
    )
    assert np.array_equal(replay_risk, direct_risk)


# --------------------------------------------------------------------------- #
# ShotEvaluation validation
# --------------------------------------------------------------------------- #
def test_shot_evaluation_accepts_safe_and_disruptive() -> None:
    safe = _evaluation([0.0, 0.1, 0.2], label=0, disruption_time_idx=-1)
    disruptive = _evaluation([0.0, 0.9, 0.9], label=1, disruption_time_idx=2)
    assert safe.label == 0
    assert disruptive.disruption_time_idx == 2


def test_shot_evaluation_rejects_bad_label() -> None:
    with pytest.raises(ValueError, match="label must be 0"):
        _evaluation([0.0, 0.1], label=2, disruption_time_idx=-1)


def test_shot_evaluation_rejects_bad_window() -> None:
    with pytest.raises(ValueError, match="window_size must be >= 1"):
        _evaluation([0.0, 0.1], label=0, disruption_time_idx=-1, window_size=0)


def test_shot_evaluation_rejects_time_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        ShotEvaluation(
            risk_series=np.zeros(3),
            label=0,
            disruption_time_idx=-1,
            time_s=np.zeros(2),
            window_size=1,
        )


@pytest.mark.parametrize("bad_idx", [-1, 5])
def test_shot_evaluation_rejects_bad_disruption_index(bad_idx: int) -> None:
    with pytest.raises(ValueError, match="disruption_time_idx"):
        _evaluation([0.0, 0.9, 0.9], label=1, disruption_time_idx=bad_idx)


# --------------------------------------------------------------------------- #
# first_alarm_index
# --------------------------------------------------------------------------- #
def test_first_alarm_index_finds_first_exceedance() -> None:
    risk = np.array([0.0, 0.2, 0.6, 0.9])
    assert first_alarm_index(risk, 0.5, start=0) == 2


def test_first_alarm_index_returns_minus_one_when_quiet() -> None:
    risk = np.array([0.0, 0.2, 0.3])
    assert first_alarm_index(risk, 0.5, start=0) == -1


def test_first_alarm_index_respects_start() -> None:
    risk = np.array([0.9, 0.1, 0.9])
    assert first_alarm_index(risk, 0.5, start=1) == 2


def test_first_alarm_index_clamps_negative_start() -> None:
    risk = np.array([0.9, 0.1])
    assert first_alarm_index(risk, 0.5, start=-5) == 0


# --------------------------------------------------------------------------- #
# confusion_at_threshold
# --------------------------------------------------------------------------- #
def test_confusion_counts_all_four_outcomes() -> None:
    evaluations = [
        _evaluation([0.0, 0.9, 0.1, 0.1], label=1, disruption_time_idx=3),  # TP (alarm@1 < 3)
        _evaluation([0.0, 0.1, 0.1, 0.9], label=1, disruption_time_idx=2),  # FN (alarm@3 >= 2)
        _evaluation([0.0, 0.1, 0.1], label=1, disruption_time_idx=2),  # FN (no alarm)
        _evaluation([0.0, 0.9, 0.1], label=0, disruption_time_idx=-1),  # FP
        _evaluation([0.0, 0.1, 0.1], label=0, disruption_time_idx=-1),  # TN
    ]
    confusion = confusion_at_threshold(evaluations, 0.5)
    assert (confusion["tp"], confusion["fn"], confusion["fp"], confusion["tn"]) == (1.0, 2.0, 1.0, 1.0)
    assert confusion["tpr"] == pytest.approx(1 / 3)
    assert confusion["fpr"] == pytest.approx(1 / 2)


def test_confusion_zero_denominators() -> None:
    confusion = confusion_at_threshold([], 0.5)
    assert confusion["tpr"] == 0.0
    assert confusion["fpr"] == 0.0


# --------------------------------------------------------------------------- #
# roc_curve + roc_auc_from_curve
# --------------------------------------------------------------------------- #
def test_roc_curve_returns_parallel_lists() -> None:
    evaluations = [
        _evaluation([0.0, 0.9, 0.1], label=1, disruption_time_idx=2),
        _evaluation([0.0, 0.1, 0.1], label=0, disruption_time_idx=-1),
    ]
    fpr, tpr = roc_curve(evaluations, [0.2, 0.8])
    assert len(fpr) == len(tpr) == 2


def test_roc_auc_perfect_classifier() -> None:
    assert roc_auc_from_curve([0.0], [1.0]) == pytest.approx(1.0)


def test_roc_auc_diagonal_is_half() -> None:
    assert roc_auc_from_curve([0.5], [0.5]) == pytest.approx(0.5)


def test_roc_auc_with_existing_endpoints() -> None:
    # Points already include fpr 0.0 and 1.0; a perfect step corner → AUC 1.0.
    assert roc_auc_from_curve([0.0, 0.0, 1.0], [0.0, 1.0, 1.0]) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# warning_time_recall
# --------------------------------------------------------------------------- #
def test_warning_time_recall_no_disruptive_shots() -> None:
    evaluations = [_evaluation([0.0, 0.1], label=0, disruption_time_idx=-1)]
    assert warning_time_recall(evaluations, 0.5, 10.0) == 0.0


def test_warning_time_recall_lead_threshold() -> None:
    # Alarm at index 5, disruption at index 10, dt 1 ms → 5 ms lead.
    risk = [0.0] * 5 + [0.9] * 6
    disruptive = _evaluation(risk, label=1, disruption_time_idx=10)
    assert warning_time_recall([disruptive], 0.5, 3.0) == 1.0
    assert warning_time_recall([disruptive], 0.5, 6.0) == 0.0


def test_warning_time_recall_skips_late_or_absent_alarm() -> None:
    late = _evaluation([0.0, 0.1, 0.1, 0.9], label=1, disruption_time_idx=2)  # alarm after disruption
    quiet = _evaluation([0.0, 0.1, 0.1], label=1, disruption_time_idx=2)  # no alarm
    assert warning_time_recall([late, quiet], 0.5, 1.0) == 0.0


# --------------------------------------------------------------------------- #
# disruption_metrics
# --------------------------------------------------------------------------- #
def test_disruption_metrics_bundle() -> None:
    evaluations = [
        _evaluation([0.0, 0.9, 0.1, 0.1], label=1, disruption_time_idx=3),
        _evaluation([0.0, 0.1, 0.1], label=0, disruption_time_idx=-1),
    ]
    metrics = disruption_metrics(
        evaluations,
        thresholds=list(np.linspace(0.0, 1.0, 11)),
        alarm_threshold=0.5,
        warning_ms=[10, 50],
    )
    assert set(metrics) == {
        "auc",
        "roc_fpr",
        "roc_tpr",
        "alarm_threshold",
        "confusion_at_alarm_threshold",
        "recall_at_warning_ms",
        "n_shots",
        "n_disruptive",
    }
    assert metrics["n_shots"] == 2
    assert metrics["n_disruptive"] == 1
    assert 0.0 <= float(metrics["auc"]) <= 1.0
    assert set(metrics["recall_at_warning_ms"]) == {10, 50}

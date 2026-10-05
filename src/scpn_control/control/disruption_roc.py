# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption ROC and warning-time analysis core
"""Reusable disruption-prediction ROC and warning-time analysis.

The public ``score_risk_series`` import delegates to its dedicated scoring
leaf. This module owns ROC, confusion, and warning-time metric assembly.

Warning-time metrics follow the DisruptionBench convention: an alarm on a
disruptive shot counts as a true positive only if it fires strictly before the
labelled disruption sample, and the warning lead time is measured against the
shot timebase.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpn_control.control._disruption_risk_series import score_risk_series

__all__ = [
    "ShotEvaluation",
    "score_risk_series",
    "first_alarm_index",
    "confusion_at_threshold",
    "roc_curve",
    "roc_auc_from_curve",
    "warning_time_recall",
    "disruption_metrics",
]


@dataclass(frozen=True)
class ShotEvaluation:
    """A scored shot ready for ROC and warning-time analysis.

    ``risk_series`` is the per-sample output of :func:`score_risk_series`;
    ``label`` is ``1`` for a disruptive shot and ``0`` for a safe shot;
    ``disruption_time_idx`` is the labelled disruption sample (``>= 0`` and
    required when ``label == 1``, ignored when ``label == 0``); ``time_s`` is the
    strictly increasing shot timebase used for warning-time leads; and
    ``window_size`` is the leading gap of zero-risk samples to skip when scanning
    for an alarm. The shot must contain at least one scored sample, finite
    probabilities in ``[0, 1]``, and a strictly increasing finite timebase.
    """

    risk_series: NDArray[np.float64]
    label: int
    disruption_time_idx: int
    time_s: NDArray[np.float64]
    window_size: int

    def __post_init__(self) -> None:
        """Validate the scored shot before it enters any report metric."""
        if self.risk_series.ndim != 1 or self.time_s.ndim != 1:
            raise ValueError("risk_series and time_s must be one-dimensional.")
        if self.label not in (0, 1):
            raise ValueError("label must be 0 (safe) or 1 (disruptive).")
        n = int(self.risk_series.shape[0])
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1.")
        if self.window_size >= n:
            raise ValueError("window_size must leave at least one scored sample.")
        if int(self.time_s.shape[0]) != n:
            raise ValueError("risk_series and time_s must share the same length.")
        if not bool(np.all(np.isfinite(self.risk_series))) or not bool(
            np.all((0.0 <= self.risk_series) & (self.risk_series <= 1.0))
        ):
            raise ValueError("risk_series must contain finite probabilities in [0, 1].")
        if not bool(np.all(np.isfinite(self.time_s))) or not bool(np.all(np.diff(self.time_s) > 0.0)):
            raise ValueError("time_s must be finite and strictly increasing.")
        if self.label == 1:
            if not 0 <= self.disruption_time_idx < n:
                raise ValueError("a disruptive shot needs 0 <= disruption_time_idx < n_samples.")


def first_alarm_index(risk_series: NDArray[np.float64], threshold: float, *, start: int) -> int:
    """Return the first sample index at/after ``start`` whose risk exceeds ``threshold``.

    Returns ``-1`` when the risk never rises above ``threshold``. Risk and
    threshold must be finite probabilities in ``[0, 1]``.
    """
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1].")
    if (
        risk_series.ndim != 1
        or not bool(np.all(np.isfinite(risk_series)))
        or not bool(np.all((0.0 <= risk_series) & (risk_series <= 1.0)))
    ):
        raise ValueError("risk_series must contain finite probabilities in [0, 1].")
    for t in range(max(start, 0), int(risk_series.shape[0])):
        if risk_series[t] > threshold:
            return t
    return -1


def _shot_alarm(evaluation: ShotEvaluation, threshold: float) -> int:
    return first_alarm_index(evaluation.risk_series, threshold, start=evaluation.window_size)


def confusion_at_threshold(evaluations: Sequence[ShotEvaluation], threshold: float) -> dict[str, float]:
    """Confusion counts and rates at a single alarm ``threshold``.

    A disruptive shot is a true positive only when an alarm fires strictly
    before its labelled disruption sample; a later or absent alarm is a false
    negative. A safe shot with any alarm is a false positive.
    """
    tp = fp = tn = fn = 0
    for evaluation in evaluations:
        alarm = _shot_alarm(evaluation, threshold)
        detected = alarm >= 0
        if evaluation.label == 1:
            if detected and alarm < evaluation.disruption_time_idx:
                tp += 1
            else:
                fn += 1
        elif detected:
            fp += 1
        else:
            tn += 1
    return {
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "tpr": tp / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
    }


def roc_curve(evaluations: Sequence[ShotEvaluation], thresholds: Sequence[float]) -> tuple[list[float], list[float]]:
    """Sweep ``thresholds`` and return parallel ``(fpr, tpr)`` lists."""
    fpr_list: list[float] = []
    tpr_list: list[float] = []
    for threshold in thresholds:
        confusion = confusion_at_threshold(evaluations, float(threshold))
        fpr_list.append(confusion["fpr"])
        tpr_list.append(confusion["tpr"])
    return fpr_list, tpr_list


def roc_auc_from_curve(fpr: Sequence[float], tpr: Sequence[float]) -> float:
    """Return the trapezoidal AUC of a ROC curve.

    The ``(0, 0)`` and ``(1, 1)`` endpoints are added when absent, the points are
    sorted by false-positive rate, and the area is integrated with the
    trapezoidal rule — the same assembly used by the synthetic ROC analysis.
    Empty, unmatched, nonfinite or out-of-range rate arrays refuse.
    """
    fpr_points = [float(x) for x in fpr]
    tpr_points = [float(x) for x in tpr]
    if not fpr_points or len(fpr_points) != len(tpr_points):
        raise ValueError("ROC arrays must be nonempty and equal length.")
    if not all(np.isfinite(x) and 0.0 <= x <= 1.0 for x in fpr_points + tpr_points):
        raise ValueError("ROC points must be finite rates in [0, 1].")
    if (0.0, 0.0) not in zip(fpr_points, tpr_points, strict=True):
        fpr_points.append(0.0)
        tpr_points.append(0.0)
    if (1.0, 1.0) not in zip(fpr_points, tpr_points, strict=True):
        fpr_points.append(1.0)
        tpr_points.append(1.0)
    order = np.lexsort((tpr_points, fpr_points))
    fpr_sorted = np.asarray(fpr_points, dtype=np.float64)[order]
    tpr_sorted = np.asarray(tpr_points, dtype=np.float64)[order]
    # Trapezoidal integral of TPR over FPR, computed natively (no scipy) so the
    # coverage tracer's NumPy reload cannot break scipy's array-API copy-mode.
    widths = np.diff(fpr_sorted)
    heights = (tpr_sorted[1:] + tpr_sorted[:-1]) / 2.0
    return float(np.sum(widths * heights))


def warning_time_recall(evaluations: Sequence[ShotEvaluation], threshold: float, warning_ms: float) -> float:
    """Fraction of disruptive shots alarmed at least ``warning_ms`` before disruption.

    Returns ``0.0`` when there are no disruptive shots. The lead time is measured
    on each shot's timebase between the labelled disruption sample and the first
    alarm sample. The lead requirement must be finite and nonnegative.
    """
    if not np.isfinite(warning_ms) or warning_ms < 0.0:
        raise ValueError("warning_ms must be finite and nonnegative.")
    disruptive = [e for e in evaluations if e.label == 1]
    if not disruptive:
        return 0.0
    hits = 0
    for evaluation in disruptive:
        alarm = _shot_alarm(evaluation, threshold)
        if alarm < 0 or alarm >= evaluation.disruption_time_idx:
            continue
        lead_ms = (evaluation.time_s[evaluation.disruption_time_idx] - evaluation.time_s[alarm]) * 1000.0
        if lead_ms >= warning_ms:
            hits += 1
    return hits / len(disruptive)


def disruption_metrics(
    evaluations: Sequence[ShotEvaluation],
    *,
    thresholds: Sequence[float],
    alarm_threshold: float,
    warning_ms: Sequence[float],
) -> dict[str, object]:
    """Assemble the full ROC + warning-time metric bundle for scored shots.

    Combines the threshold-swept ROC curve and AUC with the confusion matrix and
    warning-time recall evaluated at a single operating ``alarm_threshold``.
    ROC/AUC requires both safe and disruptive shots. Warning keys are unique
    nonnegative integer milliseconds.
    """
    labels = {evaluation.label for evaluation in evaluations}
    if labels != {0, 1}:
        raise ValueError("ROC metrics require both safe and disruptive shots.")
    if not thresholds or not all(np.isfinite(t) and 0.0 <= t <= 1.0 for t in thresholds):
        raise ValueError("thresholds must be nonempty finite probabilities in [0, 1].")
    if not np.isfinite(alarm_threshold) or not 0.0 <= alarm_threshold <= 1.0:
        raise ValueError("alarm_threshold must be finite and in [0, 1].")
    if not all(np.isfinite(w) and w >= 0.0 and float(w).is_integer() for w in warning_ms):
        raise ValueError("warning_ms must contain nonnegative integer millisecond keys.")
    if len({int(w) for w in warning_ms}) != len(warning_ms):
        raise ValueError("warning_ms keys must be unique.")
    fpr, tpr = roc_curve(evaluations, thresholds)
    auc = roc_auc_from_curve(fpr, tpr)
    recall = {int(w): warning_time_recall(evaluations, alarm_threshold, float(w)) for w in warning_ms}
    return {
        "auc": auc,
        "roc_fpr": [float(x) for x in fpr],
        "roc_tpr": [float(x) for x in tpr],
        "alarm_threshold": float(alarm_threshold),
        "confusion_at_alarm_threshold": confusion_at_threshold(evaluations, alarm_threshold),
        "recall_at_warning_ms": recall,
        "n_shots": len(evaluations),
        "n_disruptive": sum(1 for e in evaluations if e.label == 1),
    }

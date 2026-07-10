# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption ROC and warning-time analysis core
"""Reusable disruption-prediction ROC and warning-time analysis.

This module is the single source of truth for two pieces of disruption analysis
that were previously inlined in separate scripts:

* ``score_risk_series`` — the canonical per-window scoring convention (the
  ``dBdt`` signal window plus n=1/n=2/n=3 toroidal observables) shared by the
  real-shot replay in :mod:`scpn_control.control.disruption_contracts` and the
  FAIR-MAST evaluation harness.
* ``roc_auc_from_curve`` — the endpoint-forced, FPR-sorted trapezoidal AUC
  assembly shared with the synthetic ROC analysis.

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

from scpn_control.control.disruption_predictor import predict_disruption_risk

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

# Upper clip on toroidal mode amplitudes fed to the predictor, matching the
# real-shot replay convention in ``disruption_contracts.run_real_shot_replay``.
_TOROIDAL_AMP_CLIP = 10.0
# The n=3 amplitude is a bounded approximation of n=2 (0.4 * n=2; no dedicated
# n=3 diagnostic channel exists), so this scoring core is a bounded model.
_N3_FROM_N2 = 0.4


def score_risk_series(
    dbdt: NDArray[np.float64],
    n1_amp: NDArray[np.float64],
    n2_amp: NDArray[np.float64],
    *,
    window_size: int,
) -> NDArray[np.float64]:
    """Return the per-sample disruption risk over a shot.

    The signal window is the ``window_size``-sample history of ``dbdt`` ending at
    (but excluding) each sample; the toroidal observables are derived from the
    n=1/n=2 mode amplitudes with the n=3 amplitude approximated as
    ``0.4 * n2``. Samples before the first full window carry zero risk. This
    mirrors the scoring loop of
    :func:`scpn_control.control.disruption_contracts.run_real_shot_replay` and is
    its single source of truth.

    Parameters
    ----------
    dbdt, n1_amp, n2_amp : numpy.ndarray
        One-dimensional, equal-length, finite arrays: the magnetic pickup-coil
        time derivative and the n=1/n=2 toroidal mode amplitudes.
    window_size : int
        Sliding predictor window in samples (``>= 1``).

    Returns
    -------
    numpy.ndarray
        Risk in ``[0, 1]`` per sample, zero for the leading ``window_size``
        samples.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1.")
    n = int(dbdt.shape[0])
    if not n1_amp.shape[0] == n2_amp.shape[0] == n:
        raise ValueError("dbdt, n1_amp and n2_amp must share the same length.")
    for name, arr in (("dbdt", dbdt), ("n1_amp", n1_amp), ("n2_amp", n2_amp)):
        if not bool(np.all(np.isfinite(arr))):
            raise ValueError(f"{name} must be finite.")

    n3_amp = n2_amp * _N3_FROM_N2
    risk = np.zeros(n, dtype=np.float64)
    for t in range(window_size, n):
        window = dbdt[t - window_size : t]
        toroidal = {
            "toroidal_n1_amp": float(np.clip(n1_amp[t], 0.0, _TOROIDAL_AMP_CLIP)),
            "toroidal_n2_amp": float(np.clip(n2_amp[t], 0.0, _TOROIDAL_AMP_CLIP)),
            "toroidal_n3_amp": float(np.clip(n3_amp[t], 0.0, _TOROIDAL_AMP_CLIP)),
            "toroidal_asymmetry_index": float(np.sqrt(n1_amp[t] ** 2 + n2_amp[t] ** 2 + n3_amp[t] ** 2)),
            "toroidal_radial_spread": float(0.02 + 0.05 * n1_amp[t]),
        }
        risk[t] = float(np.clip(predict_disruption_risk(window, toroidal), 0.0, 1.0))
    return risk


@dataclass(frozen=True)
class ShotEvaluation:
    """A scored shot ready for ROC and warning-time analysis.

    ``risk_series`` is the per-sample output of :func:`score_risk_series`;
    ``label`` is ``1`` for a disruptive shot and ``0`` for a safe shot;
    ``disruption_time_idx`` is the labelled disruption sample (``>= 0`` and
    required when ``label == 1``, ignored when ``label == 0``); ``time_s`` is the
    strictly increasing shot timebase used for warning-time leads; and
    ``window_size`` is the leading gap of zero-risk samples to skip when scanning
    for an alarm.
    """

    risk_series: NDArray[np.float64]
    label: int
    disruption_time_idx: int
    time_s: NDArray[np.float64]
    window_size: int

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError("label must be 0 (safe) or 1 (disruptive).")
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1.")
        n = int(self.risk_series.shape[0])
        if int(self.time_s.shape[0]) != n:
            raise ValueError("risk_series and time_s must share the same length.")
        if self.label == 1:
            if not 0 <= self.disruption_time_idx < n:
                raise ValueError("a disruptive shot needs 0 <= disruption_time_idx < n_samples.")


def first_alarm_index(risk_series: NDArray[np.float64], threshold: float, *, start: int) -> int:
    """Return the first sample index at/after ``start`` whose risk exceeds ``threshold``.

    Returns ``-1`` when the risk never rises above ``threshold``.
    """
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
    """
    fpr_points = [float(x) for x in fpr]
    tpr_points = [float(x) for x in tpr]
    if not any(point == 0.0 for point in fpr_points):
        fpr_points.append(0.0)
        tpr_points.append(0.0)
    if not any(point == 1.0 for point in fpr_points):
        fpr_points.append(1.0)
        tpr_points.append(1.0)
    order = np.argsort(fpr_points)
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
    alarm sample.
    """
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
    """
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

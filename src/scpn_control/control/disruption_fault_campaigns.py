# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption fault/noise and anomaly-alarm campaigns

"""Deterministic fault-injection and hybrid anomaly-alarm campaign helpers.

This leaf owns bit-flip injection, synthetic control-signal generation,
fault/noise resilience campaigns, and the hybrid anomaly detector plus
alarm campaign (CTL-G07 R7-S3). Physics proxies and claim boundaries live
in sibling leaves; checkpoint integrity and torch training remain on the
owner.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scpn_control._typing import FloatArray
from scpn_control.control.disruption_physics_proxies import (
    _linear_percentile,
    predict_disruption_risk,
    simulate_tearing_mode,
)
from scpn_control.core._validators import (
    require_bounded_float,
    require_fraction,
    require_non_negative_float,
    require_positive_float,
)
from scpn_control.core._validators import (
    require_int as _require_int,
)


def apply_bit_flip_fault(value: float, bit_index: int) -> float:
    """Inject a deterministic single-bit fault into a float."""
    if isinstance(bit_index, bool) or not isinstance(bit_index, (int, np.integer)):
        raise ValueError("bit_index must be an integer in [0, 63].")
    bit = int(bit_index)
    if bit < 0 or bit > 63:
        raise ValueError("bit_index must be an integer in [0, 63].")
    raw = np.array([float(value)], dtype=np.float64).view(np.uint64)[0]
    flipped = np.uint64(raw ^ (np.uint64(1) << np.uint64(bit)))
    out = np.array([flipped], dtype=np.uint64).view(np.float64)[0]
    return float(out if np.isfinite(out) else value)


def _synthetic_control_signal(rng: np.random.Generator, length: int) -> FloatArray:
    t = np.linspace(0.0, 1.0, int(length), dtype=float)
    # 3 Hz ≈ rotating 2/1 NTM, 7 Hz ≈ 3/1 mode (synthetic test signal)
    base = 0.7 + 0.15 * np.sin(2.0 * np.pi * 3.0 * t) + 0.05 * np.cos(2.0 * np.pi * 7.0 * t)
    ramp = np.where(t > 0.65, (t - 0.65) * 0.9, 0.0)
    noise = rng.normal(0.0, 0.01, size=t.shape)
    return np.asarray(np.clip(base + ramp + noise, 0.01, None))


def _normalize_fault_campaign_inputs(
    seed: int,
    episodes: int,
    window: int,
    noise_std: float,
    bit_flip_interval: int,
    recovery_window: int,
    recovery_epsilon: float,
) -> tuple[int, int, int, float, int, int, float]:
    seed_i = _require_int("seed", seed, 0)
    episodes_i = _require_int("episodes", episodes, 1)
    window_i = _require_int("window", window, 16)
    noise = require_non_negative_float("noise_std", noise_std)
    bit_flip_i = _require_int("bit_flip_interval", bit_flip_interval, 1)
    recovery_window_i = _require_int("recovery_window", recovery_window, 1)
    recovery_eps = require_positive_float("recovery_epsilon", recovery_epsilon)

    return (
        seed_i,
        episodes_i,
        window_i,
        noise,
        bit_flip_i,
        recovery_window_i,
        recovery_eps,
    )


def run_fault_noise_campaign(
    seed: int = 42,
    episodes: int = 64,
    window: int = 128,
    noise_std: float = 0.03,
    bit_flip_interval: int = 11,
    recovery_window: int = 6,
    recovery_epsilon: float = 0.03,
) -> dict[str, Any]:
    """Run deterministic synthetic fault/noise campaign for disruption-risk resilience."""
    (
        seed_i,
        episodes_i,
        window_i,
        noise,
        bit_flip_i,
        recovery_window_i,
        recovery_eps,
    ) = _normalize_fault_campaign_inputs(
        seed,
        episodes,
        window,
        noise_std,
        bit_flip_interval,
        recovery_window,
        recovery_epsilon,
    )
    rng = np.random.default_rng(seed_i)

    abs_errors = []
    recovery_steps = []
    n_faults = 0

    for _ in range(episodes_i):
        signal = _synthetic_control_signal(rng, window_i)
        toroidal = {
            "toroidal_n1_amp": float(rng.uniform(0.05, 0.25)),
            "toroidal_n2_amp": float(rng.uniform(0.03, 0.18)),
            "toroidal_n3_amp": float(rng.uniform(0.01, 0.12)),
            "toroidal_radial_spread": float(rng.uniform(0.01, 0.08)),
        }
        toroidal["toroidal_asymmetry_index"] = float(
            np.sqrt(
                toroidal["toroidal_n1_amp"] ** 2 + toroidal["toroidal_n2_amp"] ** 2 + toroidal["toroidal_n3_amp"] ** 2
            )
        )

        baseline = np.array(
            [predict_disruption_risk(signal[: i + 1], toroidal) for i in range(window_i)],
            dtype=float,
        )

        faulty_signal = signal.copy()
        faulty_indices = []
        for i in range(window_i):
            faulty_signal[i] += float(rng.normal(0.0, noise))
            if i % bit_flip_i == 0:
                faulty_signal[i] = apply_bit_flip_fault(faulty_signal[i], int(rng.integers(0, 52)))
                faulty_indices.append(i)

        faulty_toroidal = dict(toroidal)
        faulty_toroidal["toroidal_n1_amp"] = max(
            0.0, faulty_toroidal["toroidal_n1_amp"] + float(rng.normal(0.0, noise * 0.5))
        )
        faulty_toroidal["toroidal_n2_amp"] = max(
            0.0, faulty_toroidal["toroidal_n2_amp"] + float(rng.normal(0.0, noise * 0.4))
        )
        faulty_toroidal["toroidal_asymmetry_index"] = float(
            np.sqrt(
                faulty_toroidal["toroidal_n1_amp"] ** 2
                + faulty_toroidal["toroidal_n2_amp"] ** 2
                + faulty_toroidal["toroidal_n3_amp"] ** 2
            )
        )

        perturbed = np.array(
            [predict_disruption_risk(faulty_signal[: i + 1], faulty_toroidal) for i in range(window_i)],
            dtype=float,
        )

        err = np.abs(perturbed - baseline)
        abs_errors.extend(err.tolist())

        for idx in faulty_indices:
            n_faults += 1
            stop = min(window_i, idx + recovery_window_i + 1)
            recover_idx = stop
            for j in range(idx, stop):
                if err[j] <= recovery_eps:
                    recover_idx = j
                    break
            recovery_steps.append(int(recover_idx - idx))

    errors_arr = np.asarray(abs_errors, dtype=float)
    rec_arr = np.asarray(
        recovery_steps if recovery_steps else [recovery_window_i + 1],
        dtype=float,
    )

    mean_abs_err = float(np.mean(errors_arr))
    p95_abs_err = _linear_percentile(errors_arr, 95.0)
    p95_recovery = _linear_percentile(rec_arr, 95.0)
    success_rate = float(np.mean(rec_arr <= recovery_window_i))

    thresholds = {
        "max_mean_abs_risk_error": 0.08,
        "max_p95_abs_risk_error": 0.22,
        "max_recovery_steps_p95": float(recovery_window_i),
        "min_recovery_success_rate": 0.80,
    }
    passes = bool(
        mean_abs_err <= thresholds["max_mean_abs_risk_error"]
        and p95_abs_err <= thresholds["max_p95_abs_risk_error"]
        and p95_recovery <= thresholds["max_recovery_steps_p95"]
        and success_rate >= thresholds["min_recovery_success_rate"]
    )

    return {
        "seed": seed_i,
        "episodes": episodes_i,
        "window": window_i,
        "noise_std": noise,
        "bit_flip_interval": bit_flip_i,
        "fault_count": int(n_faults),
        "mean_abs_risk_error": mean_abs_err,
        "p95_abs_risk_error": p95_abs_err,
        "recovery_steps_p95": p95_recovery,
        "recovery_success_rate": success_rate,
        "thresholds": thresholds,
        "passes_thresholds": passes,
    }


class HybridAnomalyDetector:
    """Supervised + unsupervised anomaly detector for early disruption alarms.

    Supervised term: ``predict_disruption_risk`` (Rea et al. 2019).
    Unsupervised term: online z-score novelty on recent risk stream.
    Combined alarm fires when anomaly_score ≥ threshold.
    """

    def __init__(self, threshold: float = 0.50, ema: float = 0.05) -> None:
        self.threshold = require_fraction("threshold", threshold)
        self.ema = require_bounded_float("ema", ema, low=0.0, high=1.0, low_exclusive=True)
        self.mean = 0.0
        self.var = 1.0
        self.initialized = False

    def score(self, signal: Any, toroidal_observables: dict[str, float] | None = None) -> dict[str, Any]:
        """Combine the supervised risk with an online novelty (anomaly) score.

        Parameters
        ----------
        signal
            The diagnostic signal scored by the supervised predictor.
        toroidal_observables
            Optional toroidal observables passed to the supervised predictor.

        Returns
        -------
        dict[str, Any]
            A mapping with the supervised disruption risk, the EMA-based
            unsupervised novelty score, and the running signal statistics.
        """
        supervised = predict_disruption_risk(signal, toroidal_observables)
        value = float(supervised)

        if self.initialized:
            z = abs(value - self.mean) / float(np.sqrt(self.var + 1e-9))
            unsupervised = float(1.0 - np.exp(-0.5 * z))
        else:
            unsupervised = 0.0
            self.initialized = True

        alpha = self.ema
        delta = value - self.mean
        self.mean += alpha * delta
        self.var = (1.0 - alpha) * self.var + alpha * delta * delta
        self.var = float(max(self.var, 1e-9))

        anomaly_score = float(np.clip(0.7 * supervised + 0.3 * unsupervised, 0.0, 1.0))
        return {
            "supervised_score": float(supervised),
            "unsupervised_score": float(unsupervised),
            "anomaly_score": anomaly_score,
            "alarm": bool(anomaly_score >= self.threshold),
        }


def run_anomaly_alarm_campaign(
    seed: int = 42,
    episodes: int = 32,
    window: int = 128,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Deterministic anomaly-alarm campaign under random perturbations."""
    seed_i = _require_int("seed", seed, 0)
    episodes_i = _require_int("episodes", episodes, 1)
    window_i = _require_int("window", window, 16)
    threshold_f = require_fraction("threshold", threshold)

    rng = np.random.default_rng(seed_i)
    detector = HybridAnomalyDetector(threshold=threshold_f)

    true_positives = 0
    false_positives = 0
    positives = 0
    negatives = 0
    latencies = []

    for ep in range(episodes_i):
        sim_rng = np.random.default_rng(int(seed_i + ep))
        signal, label, _ = simulate_tearing_mode(steps=window_i, rng=sim_rng)
        if signal.size < window_i:
            signal = np.pad(signal, (0, window_i - signal.size), mode="edge")
        signal = np.asarray(signal[:window_i], dtype=float)

        n1 = float(rng.uniform(0.03, 0.24))
        n2 = float(rng.uniform(0.02, 0.16))
        n3 = float(rng.uniform(0.01, 0.10))
        toroidal = {
            "toroidal_n1_amp": n1,
            "toroidal_n2_amp": n2,
            "toroidal_n3_amp": n3,
            "toroidal_asymmetry_index": float(np.sqrt(n1 * n1 + n2 * n2 + n3 * n3)),
            "toroidal_radial_spread": float(rng.uniform(0.01, 0.08)),
        }

        first_alarm = None
        for k in range(window_i):
            perturbed_signal = signal[: k + 1].copy()
            perturbed_signal[-1] += float(rng.normal(0.0, 0.02))
            score = detector.score(perturbed_signal, toroidal)
            if score["alarm"] and first_alarm is None:
                first_alarm = k

        is_positive = bool(label == 1)
        if is_positive:
            positives += 1
            if first_alarm is not None:
                true_positives += 1
                latencies.append(first_alarm)
        else:
            negatives += 1
            if first_alarm is not None:
                false_positives += 1

    tpr = float(true_positives / max(positives, 1))
    fpr = float(false_positives / max(negatives, 1))
    p95_latency = _linear_percentile(latencies, 95.0) if latencies else float(window_i)
    passes = bool(tpr >= 0.75 and fpr <= 0.35)

    return {
        "seed": seed_i,
        "episodes": episodes_i,
        "window": window_i,
        "threshold": threshold_f,
        "true_positive_rate": tpr,
        "false_positive_rate": fpr,
        "p95_alarm_latency_steps": p95_latency,
        "passes_thresholds": passes,
    }

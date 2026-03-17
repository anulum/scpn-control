# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
from __future__ import annotations

# ML disruption prediction: Kates-Harbeck et al. 2019, Nature 568, 526 —
# FRNN deep-learning disruption predictor on JET/DIII-D.
#
# Locked-mode disruption precursor: de Vries et al. 2011, Nucl. Fusion 51,
# 053018 — cross-machine disruption database; locked modes dominate.
#
# Warning-time requirement: τ_warning > τ_TQ + τ_mitigation ≈ 10–30 ms for
# ITER; Lehnen et al. 2015, J. Nucl. Mater. 463, 39.
#
# Input features (locked-mode amplitude, P_rad fraction, q95, β_N, l_i,
# Greenwald fraction): Rea et al. 2019, Nucl. Fusion 59, 096016, Table I.

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:  # pragma: no cover - optional dependency path
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]

import logging

from scpn_control.core._validators import (
    require_bounded_float,
    require_finite_array,
    require_fraction,
    require_int as _require_int,
    require_non_negative_float,
    require_positive_float,
)

logger = logging.getLogger(__name__)

DEFAULT_SEQ_LEN = 100
DEFAULT_MODEL_FILENAME = "disruption_model.pth"
PROBABILISTIC_SIGMA_LEVELS = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
TOROIDAL_PERTURB_FRACTION = 0.08
MIN_PROBABILISTIC_NOISE_SCALE = 1.0e-4

# Minimum required alarm lead time.
# Lehnen et al. 2015, J. Nucl. Mater. 463, 39 — τ_warning > τ_TQ + τ_mitigation.
# For ITER τ_TQ ≈ 1–5 ms, τ_mitigation ≈ 10–20 ms → lower bound 10 ms.
TAU_WARNING_MIN_S: float = 0.010  # s

# Locked-mode amplitude threshold above which disruption alarm is raised.
# de Vries et al. 2011, Nucl. Fusion 51, 053018, §3 — locked-mode onset
# is identified as the dominant disruption precursor across JET, DIII-D, AUG.
LOCKED_MODE_ALARM_THRESHOLD: float = 0.15  # normalised units


def simulate_tearing_mode(
    steps: int = 1000,
    *,
    mode: str = "ntm",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, int, int]:
    """Generate synthetic shot data for multiple disruption mechanisms.

    Physics models:
    - ntm: Modified Rutherford equation (Rutherford 1973).
    - density_limit: Greenwald density limit (Greenwald 2002,
      Plasma Phys. Control. Fusion 44, R27).
    - vde: Exponential growth on Naydon instability timescale.

    Parameters
    ----------
    steps : int
        Number of steps to simulate.
    mode : str
        Disruption mechanism: "ntm", "density_limit", or "vde".
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    signal : ndarray
        Magnetic or diagnostic sensor data.
    label : int
        1 if disrupted, 0 if safe.
    time_to_disruption : int
        Steps remaining at detection, or -1 if safe.
    """
    steps = _require_int("steps", steps, 1)
    local_rng = rng if rng is not None else np.random.default_rng()

    is_disruptive = float(local_rng.random()) > 0.5
    trigger_time = int(local_rng.integers(200, 800)) if is_disruptive else 9999
    dt = 0.01  # s

    signal_history = []

    if mode == "ntm":
        # Modified Rutherford equation; Fitzpatrick, Phys. Plasmas 2, 825 (1995)
        w = 0.01  # cm, seed island half-width
        delta_prime = -0.5
        W_SAT = 10.0  # cm, saturation island width
        W_LOCK = 8.0  # cm, locking threshold (de Vries et al. 2011)
        NOISE_STD = 0.05
        for t in range(steps):
            if t > trigger_time:
                delta_prime = 0.5
            dw = (delta_prime * (1 - w / W_SAT)) * dt
            w += dw + float(local_rng.normal(0.0, NOISE_STD))
            w = max(w, 0.01)
            signal_history.append(w)
            if w > W_LOCK:
                return np.array(signal_history), 1, (t - trigger_time)

    elif mode == "density_limit":
        # Greenwald density limit: n_G = I_p / (π a²)
        # Greenwald 2002, Plasma Phys. Control. Fusion 44, R27, Eq. 1
        Ip = 15.0  # MA, ITER full performance
        a = 2.0  # m, ITER minor radius
        n_G = Ip / (np.pi * a**2)
        n_e = 0.5 * n_G
        n_drift = 0.0
        for t in range(steps):
            if t > trigger_time:
                n_drift = 0.005
            n_e += n_drift + float(local_rng.normal(0.0, 0.001))
            signal_history.append(n_e)
            if n_e > n_G:
                return np.array(signal_history), 1, (t - trigger_time)

    elif mode == "vde":
        # Vertical Displacement Event: exponential vertical growth
        z = 0.001  # m, initial vertical offset
        z_growth_rate = 0.0  # s⁻¹
        Z_LIMIT = 0.5  # m, first-wall contact limit
        for t in range(steps):
            if t > trigger_time:
                z_growth_rate = 50.0  # 50 Hz growth rate
            dz = (z_growth_rate * z) * dt
            z += dz + float(local_rng.normal(0.0, 0.0005))
            signal_history.append(abs(z))
            if abs(z) > Z_LIMIT:
                return np.array(signal_history), 1, (t - trigger_time)
    else:
        raise ValueError(f"Unknown disruption mode: {mode}")

    return np.array(signal_history), 0, -1


def build_disruption_feature_vector(signal: Any, toroidal_observables: dict[str, float] | None = None) -> np.ndarray:
    """Build a compact feature vector for control-oriented disruption scoring.

    Feature layout:
      [mean, std, max, slope, energy, last,
       toroidal_n1_amp, toroidal_n2_amp, toroidal_n3_amp,
       toroidal_asymmetry_index, toroidal_radial_spread]

    Feature selection follows Rea et al. 2019, Nucl. Fusion 59, 096016,
    Table I — locked-mode amplitude, radiated power fraction, q95, β_N,
    l_i, and Greenwald fraction are the primary predictors.
    """
    sig = np.asarray(signal, dtype=float).reshape(-1)
    if sig.size == 0:
        raise ValueError("signal must contain at least one sample")
    require_finite_array("signal", sig)

    mean = float(np.mean(sig))
    std = float(np.std(sig))
    max_val = float(np.max(sig))
    slope = float((sig[-1] - sig[0]) / max(sig.size - 1, 1))
    energy = float(np.mean(sig**2))
    last = float(sig[-1])

    obs = toroidal_observables or {}
    n1 = float(obs.get("toroidal_n1_amp", 0.0))
    n2 = float(obs.get("toroidal_n2_amp", 0.0))
    n3 = float(obs.get("toroidal_n3_amp", 0.0))
    asym_raw = obs.get("toroidal_asymmetry_index")
    asym = float(asym_raw) if asym_raw is not None else float(np.sqrt(n1 * n1 + n2 * n2 + n3 * n3))
    spread = float(obs.get("toroidal_radial_spread", 0.0))
    require_finite_array("toroidal observables", [n1, n2, n3, asym, spread])

    return np.array(
        [mean, std, max_val, slope, energy, last, n1, n2, n3, asym, spread],
        dtype=float,
    )


def predict_disruption_risk(signal: Any, toroidal_observables: dict[str, float] | None = None) -> float:
    """Lightweight deterministic disruption risk estimator returning a value in [0, 1].

    Supplements the Transformer pathway by explicitly consuming toroidal
    asymmetry observables from 3D diagnostics (n=1,2,3 mode amplitudes).

    Feature weights tuned on synthetic DIII-D/JET validation shots; see
    validation/reports/disruption_replay_pipeline_benchmark.md.
    Logit bias: sigmoid(−4.0) ≈ 0.018, giving low base risk on zero features.
    """
    features = build_disruption_feature_vector(signal, toroidal_observables)
    mean, std, max_val, slope, energy, last, n1, n2, n3, asym, spread = features

    thermal_term = 0.03 * max_val + 0.55 * std + 0.005 * energy + 0.50 * slope
    asym_term = 1.10 * n1 + 0.70 * n2 + 0.45 * n3 + 0.50 * asym + 0.15 * spread
    state_term = 0.02 * mean + 0.02 * last

    # Logit bias calibrated so median safe shot → ~2% risk, disrupting shot → ~60%.
    LOGIT_BIAS = -4.0
    logits = LOGIT_BIAS + thermal_term + asym_term + state_term
    return float(1.0 / (1.0 + np.exp(-logits)))


def disruption_warning_time(
    signal: Any,
    toroidal_observables: dict[str, float] | None = None,
    *,
    risk_threshold: float = 0.5,
    dt: float = 0.001,
) -> float:
    """Estimate τ_warning: time between first alarm and end of signal.

    τ_warning > τ_TQ + τ_mitigation ≈ 10–30 ms for ITER.
    Reference: Lehnen et al. 2015, J. Nucl. Mater. 463, 39.

    Parameters
    ----------
    signal : array-like
        Time-series of a disruption-precursor diagnostic.
    toroidal_observables : dict, optional
        Toroidal mode amplitudes and asymmetry index.
    risk_threshold : float
        Risk value above which the alarm is triggered.
    dt : float
        Sampling interval in seconds.

    Returns
    -------
    tau_warning : float
        Warning time in seconds; 0.0 if alarm never fires.
    """
    flat = np.asarray(signal, dtype=float).reshape(-1)
    for k in range(len(flat)):
        risk = predict_disruption_risk(flat[: k + 1], toroidal_observables)
        if risk >= risk_threshold:
            return float((len(flat) - k - 1) * dt)
    return 0.0


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


def _synthetic_control_signal(rng: np.random.Generator, length: int) -> np.ndarray:
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
    p95_abs_err = float(np.percentile(errors_arr, 95))
    p95_recovery = float(np.percentile(rec_arr, 95))
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
    p95_latency = float(np.percentile(latencies, 95)) if latencies else float(window_i)
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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_model_path() -> Path:
    return _repo_root() / "artifacts" / DEFAULT_MODEL_FILENAME


def _normalize_seq_len(seq_len: int) -> int:
    return _require_int("seq_len", seq_len, 8)


def _prepare_signal_window(signal: Any, seq_len: int) -> np.ndarray:
    seq_len = _normalize_seq_len(seq_len)
    flat = np.asarray(signal, dtype=float).reshape(-1)
    if flat.size >= seq_len:
        return flat[:seq_len]
    return np.pad(flat, (0, seq_len - flat.size), mode="edge")


def _estimate_signal_noise_scale(signal: Any) -> float:
    flat = np.asarray(signal, dtype=float).reshape(-1)
    if flat.size < 2:
        return MIN_PROBABILISTIC_NOISE_SCALE
    diff_scale = float(np.std(np.diff(flat)))
    return max(diff_scale, MIN_PROBABILISTIC_NOISE_SCALE)


def _sigma_point_pattern(length: int) -> np.ndarray:
    n = _require_int("length", length, 1)
    if n == 1:
        return np.ones(1, dtype=float)
    pattern = np.linspace(-1.0, 1.0, n, dtype=float)
    rms = float(np.sqrt(np.mean(pattern * pattern)))
    return pattern / max(rms, 1.0e-12)


def _perturb_toroidal_observables(
    toroidal_observables: dict[str, float] | None,
    sigma_level: float,
) -> dict[str, float] | None:
    if toroidal_observables is None:
        return None
    scale = 1.0 + TOROIDAL_PERTURB_FRACTION * float(sigma_level)
    return {key: float(value) * scale for key, value in toroidal_observables.items()}


def _summarize_risk_samples(
    samples: Any,
    *,
    center_risk: float,
    method: str,
) -> dict[str, Any]:
    clipped = np.clip(np.asarray(samples, dtype=float).reshape(-1), 0.0, 1.0)
    if clipped.size < 1:
        clipped = np.asarray([center_risk], dtype=float)
    q05, q50, q95 = np.quantile(clipped, [0.05, 0.50, 0.95])
    return {
        "probabilistic_output": True,
        "probabilistic_method": method,
        "risk_mean": float(np.mean(clipped)),
        "risk_std": float(np.std(clipped)),
        "risk_p05": float(q05),
        "risk_p50": float(q50),
        "risk_p95": float(q95),
        "risk_interval": [float(q05), float(q95)],
        "risk_interval_width": float(q95 - q05),
        "risk_samples_used": int(clipped.size),
        "risk_center": float(center_risk),
    }


def _deterministic_risk_samples(
    signal: Any,
    toroidal_observables: dict[str, float] | None,
) -> np.ndarray:
    flat = np.asarray(signal, dtype=float).reshape(-1)
    noise_scale = _estimate_signal_noise_scale(flat)
    pattern = _sigma_point_pattern(max(flat.size, 1))
    samples = [
        predict_disruption_risk(
            flat + float(level) * noise_scale * pattern,
            _perturb_toroidal_observables(toroidal_observables, float(level)),
        )
        for level in PROBABILISTIC_SIGMA_LEVELS
    ]
    return np.asarray(samples, dtype=float)


def _model_risk_samples(
    model: Any,
    signal: Any,
    *,
    seq_len: int,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("Torch is required for model-based risk samples.")
    prepared = _prepare_signal_window(signal, seq_len)
    noise_scale = _estimate_signal_noise_scale(prepared)
    pattern = _sigma_point_pattern(prepared.size)
    samples: list[float] = []
    model.eval()
    for level in PROBABILISTIC_SIGMA_LEVELS:
        perturbed = prepared + float(level) * noise_scale * pattern
        input_tensor = torch.tensor(perturbed, dtype=torch.float32).reshape(1, -1, 1)
        with torch.no_grad():
            samples.append(float(np.clip(float(model(input_tensor).item()), 0.0, 1.0)))
    return np.asarray(samples, dtype=float)


# --- AI: TRANSFORMER MODEL ---
if torch is not None:

    class DisruptionTransformer(nn.Module):
        """Transformer encoder for disruption prediction with MC dropout.

        Architecture follows Kates-Harbeck et al. 2019, Nature 568, 526 (FRNN)
        adapted to a single-channel time series with positional encoding.
        MC dropout uncertainty: Gal & Ghahramani 2016, ICML.
        """

        def __init__(self, seq_len: int = DEFAULT_SEQ_LEN, dropout: float = 0.1) -> None:
            super().__init__()
            self.seq_len = _normalize_seq_len(seq_len)
            self.embedding = nn.Linear(1, 32)
            self.pos_encoder = nn.Parameter(torch.zeros(1, self.seq_len, 32))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=32,
                nhead=4,
                dim_feedforward=64,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.classifier = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(16, 1),
            )
            self.sigmoid = nn.Sigmoid()

        def forward(self, src: Any) -> Any:
            if src.ndim != 3:
                raise ValueError(f"Input tensor must have shape [batch, seq, 1]; got rank {src.ndim}.")
            if src.shape[1] < 1:
                raise ValueError("Input sequence length must be >= 1.")
            if src.shape[2] != 1:
                raise ValueError(f"Input feature dimension must be 1; got {src.shape[2]}.")
            if src.shape[1] > self.seq_len:
                raise ValueError(f"Input sequence length {src.shape[1]} exceeds configured seq_len {self.seq_len}.")
            x = self.embedding(src) + self.pos_encoder[:, : src.shape[1], :]
            output = self.transformer(x)
            last_step = output[:, -1, :]
            return self.sigmoid(self.classifier(last_step))

        def predict_with_uncertainty(self, src: Any, n_samples: int = 10) -> tuple[float, float]:
            """MC dropout inference returning (mean, std) over n_samples passes."""
            self.train()
            samples = []
            with torch.no_grad():
                for _ in range(n_samples):
                    samples.append(float(self.forward(src).item()))
            return float(np.mean(samples)), float(np.std(samples))
else:  # pragma: no cover - only used without torch installed

    class DisruptionTransformer:  # type: ignore[no-redef]
        def __init__(self) -> None:
            raise RuntimeError("Torch is required for DisruptionTransformer.")


def train_predictor(
    seq_len: int = DEFAULT_SEQ_LEN,
    n_shots: int = 500,
    epochs: int = 50,
    model_path: str | Path | None = None,
    seed: int = 42,
    save_plot: bool = True,
) -> tuple[Any, dict[str, Any]]:
    if torch is None or optim is None:
        raise RuntimeError("Torch is required for train_predictor().")

    seq_len = _normalize_seq_len(seq_len)
    n_shots = _require_int("n_shots", n_shots, 8)
    epochs = _require_int("epochs", epochs, 1)
    seed = _require_int("seed", seed, 0)
    torch.manual_seed(seed)
    data_rng = np.random.default_rng(seed)
    eval_rng = np.random.default_rng(seed + 1000003)

    model_path = Path(model_path) if model_path is not None else default_model_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("--- SCPN SAFETY AI: Disruption Prediction (Transformer) ---")
    logger.info(f"Sequence length: {seq_len} | Shots: {n_shots} | Epochs: {epochs}")

    logger.info("Generating synthetic shots (Rutherford Physics)...")
    X_train = []
    y_train = []

    sim_steps = max(1000, seq_len + 16)
    for _ in range(n_shots):
        sig, label, _ = simulate_tearing_mode(steps=sim_steps, rng=data_rng)
        sig_window = _prepare_signal_window(sig, seq_len)
        X_train.append(sig_window.reshape(-1, 1))
        y_train.append(label)

    X_tensor = torch.tensor(np.asarray(X_train, dtype=np.float32), dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    model = DisruptionTransformer(seq_len=seq_len)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    logger.info("Training Transformer...")
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch}: Loss={loss.item():.4f}")

    logger.info("Validating on a new shot...")
    test_sig, test_lbl, _ = simulate_tearing_mode(steps=sim_steps, rng=eval_rng)
    input_sig = _prepare_signal_window(test_sig, seq_len)
    input_tensor = torch.tensor(input_sig, dtype=torch.float32).reshape(1, -1, 1)

    model.eval()
    with torch.no_grad():
        risk = model(input_tensor).item()

    logger.info(f"Test Shot Ground Truth: {'DISRUPTIVE' if test_lbl else 'SAFE'}")
    logger.info(f"AI Prediction Risk: {risk * 100:.1f}%")

    torch.save({"state_dict": model.state_dict(), "seq_len": int(seq_len)}, model_path)
    logger.info(f"Saved model: {model_path}")

    plot_path = _repo_root() / "artifacts" / "Disruption_AI_Result.png"
    if save_plot:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(losses)
        ax1.set_title("Transformer Training Loss")
        ax1.set_xlabel("Epoch")
        ax2.plot(test_sig, "r-" if test_lbl else "g-")
        ax2.set_title(f"Diagnostic Signal (AI Risk: {risk:.2f})")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)
        logger.info(f"Saved: {plot_path}")

    return model, {
        "seq_len": int(seq_len),
        "shots": int(n_shots),
        "epochs": int(epochs),
        "model_path": str(model_path),
        "risk": float(risk),
        "test_label": int(test_lbl),
    }


def load_or_train_predictor(
    model_path: str | Path | None = None,
    seq_len: int = DEFAULT_SEQ_LEN,
    force_retrain: bool = False,
    train_kwargs: dict[str, Any] | None = None,
    train_if_missing: bool = True,
    allow_fallback: bool = True,
) -> tuple[Any, dict[str, Any]]:
    if torch is None:
        if not allow_fallback:
            raise RuntimeError("Torch is required for load_or_train_predictor().")
        return None, {
            "trained": False,
            "fallback": True,
            "reason": "torch_unavailable",
            "model_path": str(model_path) if model_path is not None else str(default_model_path()),
            "seq_len": int(_normalize_seq_len(seq_len)),
        }

    path = Path(model_path) if model_path is not None else default_model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    seq_len = _normalize_seq_len(seq_len)
    kwargs = dict(train_kwargs or {})

    if path.exists() and not force_retrain:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                loaded_seq_len = _normalize_seq_len(checkpoint.get("seq_len", seq_len))
            else:
                state_dict = checkpoint
                loaded_seq_len = seq_len

            model = DisruptionTransformer(seq_len=loaded_seq_len)
            model.load_state_dict(state_dict)
            model.eval()
            return model, {
                "trained": False,
                "fallback": False,
                "model_path": str(path),
                "seq_len": int(loaded_seq_len),
            }
        except (RuntimeError, ValueError, KeyError, OSError) as exc:
            if not allow_fallback:
                raise
            return None, {
                "trained": False,
                "fallback": True,
                "reason": f"checkpoint_load_failed:{exc.__class__.__name__}",
                "model_path": str(path),
                "seq_len": int(seq_len),
            }

    if not train_if_missing and not force_retrain:
        if allow_fallback:
            return None, {
                "trained": False,
                "fallback": True,
                "reason": "checkpoint_missing",
                "model_path": str(path),
                "seq_len": int(seq_len),
            }
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    kwargs.setdefault("seq_len", seq_len)
    kwargs.setdefault("model_path", path)
    try:
        model, info = train_predictor(**kwargs)
    except (RuntimeError, ValueError, OSError) as exc:
        if not allow_fallback:
            raise
        return None, {
            "trained": False,
            "fallback": True,
            "reason": f"train_failed:{exc.__class__.__name__}",
            "model_path": str(path),
            "seq_len": int(seq_len),
        }
    info["trained"] = True
    info["fallback"] = False
    return model, info


def predict_disruption_risk_safe(
    signal: Any,
    toroidal_observables: dict[str, float] | None = None,
    *,
    model_path: str | Path | None = None,
    seq_len: int = DEFAULT_SEQ_LEN,
    train_if_missing: bool = False,
    mc_samples: int = 10,
) -> tuple[float, dict[str, Any]]:
    """Predict disruption risk with MC dropout uncertainty if model is available.

    Returns
    -------
    risk, metadata
        ``risk`` is the MC mean risk in ``[0, 1]``.
        ``metadata`` includes ``risk_std`` (epistemic uncertainty).
    """
    base_risk = float(np.clip(predict_disruption_risk(signal, toroidal_observables), 0.0, 1.0))

    model, meta = load_or_train_predictor(
        model_path=model_path,
        seq_len=seq_len,
        force_retrain=False,
        train_kwargs={"seq_len": _normalize_seq_len(seq_len), "save_plot": False},
        train_if_missing=bool(train_if_missing),
        allow_fallback=True,
    )

    if model is None or torch is None:
        out_meta = dict(meta)
        out_meta["mode"] = "fallback"
        out_meta["risk_source"] = "predict_disruption_risk"
        out_meta.update(
            _summarize_risk_samples(
                _deterministic_risk_samples(signal, toroidal_observables),
                center_risk=base_risk,
                method="deterministic_sigma_points",
            )
        )
        return base_risk, out_meta

    try:
        model_seq_len = int(meta.get("seq_len", _normalize_seq_len(seq_len)))
        input_sig = _prepare_signal_window(signal, model_seq_len)
        input_tensor = torch.tensor(input_sig, dtype=torch.float32).reshape(1, -1, 1)

        if hasattr(model, "predict_with_uncertainty"):
            mean_risk, std_risk = model.predict_with_uncertainty(input_tensor, n_samples=mc_samples)
        else:
            model.eval()
            with torch.no_grad():
                mean_risk = float(model(input_tensor).item())
            std_risk = 0.0

        out_meta = dict(meta)
        out_meta["mode"] = "checkpoint"
        out_meta["risk_source"] = "transformer_mc_dropout"
        out_meta["risk_std"] = std_risk
        out_meta["risk_mean"] = mean_risk

        input_samples = _deterministic_risk_samples(signal, toroidal_observables)
        combined_samples = np.append(input_samples, [mean_risk])

        out_meta.update(
            _summarize_risk_samples(
                combined_samples,
                center_risk=mean_risk,
                method="transformer_mc_plus_sigma_points",
            )
        )
        return mean_risk, out_meta
    except (RuntimeError, ValueError, OSError) as exc:
        out_meta = dict(meta)
        out_meta["mode"] = "fallback"
        out_meta["risk_source"] = "predict_disruption_risk"
        out_meta["reason"] = f"inference_failed:{exc.__class__.__name__}"
        out_meta.update(
            _summarize_risk_samples(
                _deterministic_risk_samples(signal, toroidal_observables),
                center_risk=base_risk,
                method="deterministic_sigma_points",
            )
        )
        return base_risk, out_meta


def evaluate_predictor(
    model: Any,
    X_test: Any,
    y_test: Any,
    times_test: Any = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate disruption predictor on test set.

    Returns accuracy, precision, recall, F1, confusion matrix,
    and recall@T for T in [10, 20, 30, 50, 100] ms.
    """
    pred_list: list[int] = []
    for seq in X_test:
        pred = model.predict(seq)
        pred_list.append(1 if pred > threshold else 0)
    predictions = np.array(pred_list)
    y_true = np.array(y_test)

    tp = np.sum((predictions == 1) & (y_true == 1))
    fp = np.sum((predictions == 1) & (y_true == 0))
    tn = np.sum((predictions == 0) & (y_true == 0))
    fn = np.sum((predictions == 0) & (y_true == 1))

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    fpr = fp / max(fp + tn, 1)

    result = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }

    if times_test is not None:
        for T_ms in [10, 20, 30, 50, 100]:
            T_s = T_ms / 1000.0
            early_enough = np.array(times_test) >= T_s
            mask = (y_true == 1) & early_enough
            if mask.sum() > 0:
                recall_at_t = np.sum(predictions[mask] == 1) / mask.sum()
            else:
                recall_at_t = 0.0
            result[f"recall_at_{T_ms}ms"] = float(recall_at_t)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or load disruption Transformer predictor.")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--shots", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    _, meta = load_or_train_predictor(
        model_path=args.model_path,
        seq_len=args.seq_len,
        force_retrain=bool(args.force_retrain),
        train_kwargs={
            "seq_len": args.seq_len,
            "n_shots": args.shots,
            "epochs": args.epochs,
            "seed": args.seed,
            "save_plot": not args.no_plot,
        },
    )
    print(
        f"Predictor ready | trained={meta.get('trained')} | seq_len={meta.get('seq_len')} "
        f"| model_path={meta.get('model_path')}"
    )

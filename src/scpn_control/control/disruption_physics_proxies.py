# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption physics proxies (tearing, features, risk, warning)

"""Heuristic disruption physics proxies and synthetic precursor generators.

This leaf owns synthetic tearing-mode trajectories, feature construction,
fixed-weight risk scoring, and warning-time evaluation (CTL-G07 R7-S2). Claim
boundaries live in :mod:`disruption_risk_claims`; fault campaigns and
checkpointed torch models remain on the owner.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from scpn_control._typing import FloatArray
from scpn_control.control.disruption_risk_claims import (
    DISRUPTION_FEATURE_CONTRACT,
)
from scpn_control.core._validators import (
    require_bounded_float,
    require_finite_array,
)
from scpn_control.core._validators import (
    require_int as _require_int,
)


def _linear_percentile(values: Any, percentile: float) -> float:
    """Return a deterministic linear percentile without NumPy reduction sentinels."""
    p = require_bounded_float("percentile", percentile, low=0.0, high=100.0)
    flat = sorted(float(value) for value in np.asarray(values, dtype=float).reshape(-1))
    if not flat:
        raise ValueError("values must contain at least one sample")
    if len(flat) == 1:
        return flat[0]

    rank = (len(flat) - 1) * (p / 100.0)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return flat[lower]
    fraction = rank - float(lower)
    return flat[lower] * (1.0 - fraction) + flat[upper] * fraction


def simulate_tearing_mode(
    steps: int = 1000,
    *,
    mode: str = "ntm",
    rng: np.random.Generator | None = None,
) -> tuple[FloatArray, int, int]:
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


def build_disruption_feature_vector(signal: Any, toroidal_observables: dict[str, float] | None = None) -> FloatArray:
    """Build a compact feature vector for control-oriented disruption scoring.

    Feature layout is declared by ``DISRUPTION_FEATURE_CONTRACT``.

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

    values = [mean, std, max_val, slope, energy, last, n1, n2, n3, asym, spread]
    if len(values) != len(DISRUPTION_FEATURE_CONTRACT):
        raise RuntimeError("disruption feature vector length drifted from the public contract")
    # Finite inputs can still overflow to ±inf under squaring (energy), so
    # re-validate the computed features rather than emit a non-finite vector.
    return require_finite_array("disruption features", values)


def predict_disruption_risk(signal: Any, toroidal_observables: dict[str, float] | None = None) -> float:
    """Heuristic disruption-risk baseline returning a value in [0, 1].

    A transparent, deterministic fixed-weight logistic score over toroidal-asymmetry
    observables (n=1,2,3 mode amplitudes) from 3D diagnostics. This is **not** a
    model trained on a real disruption database: the feature weights and logit bias
    below are hand-chosen by inspection (sanity-checked against synthetic DIII-D/JET
    shots in validation/reports/disruption_replay_pipeline_benchmark.md), not fitted
    by an optimiser. It can be compared with the optional synthetic Transformer
    pathway as an interpretable baseline. Logit bias: sigmoid(−4.0) ≈ 0.018,
    giving low base risk on zero features.
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

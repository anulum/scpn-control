# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Federated Disruption.
"""MLP weights, forward pass, and gradients for disruption federation."""

from __future__ import annotations

import numpy as np

from scpn_control._typing import FloatArray

N_FEATURES = 8  # Ip, beta_N, q95, n/n_GW, li, dBp/dt, locked_mode_amp, n1_rms
FEATURE_NAMES = ("Ip", "beta_N", "q95", "n_nGW", "li", "dBp_dt", "locked_mode_amp", "n1_rms")


def _l2_norm(weights: dict[str, FloatArray]) -> float:
    return float(np.sqrt(sum(float(np.sum(np.asarray(value, dtype=np.float64) ** 2)) for value in weights.values())))


def _weight_delta(local_weights: dict[str, FloatArray], global_weights: dict[str, FloatArray]) -> dict[str, FloatArray]:
    return {key: np.asarray(local_weights[key], dtype=np.float64) - global_weights[key] for key in global_weights}


def _apply_weight_delta(global_weights: dict[str, FloatArray], delta: dict[str, FloatArray]) -> dict[str, FloatArray]:
    return {key: global_weights[key] + delta[key] for key in global_weights}


def _relu(x: FloatArray) -> FloatArray:
    return np.asarray(np.maximum(0.0, x), dtype=x.dtype)


def _sigmoid(x: FloatArray) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0))), dtype=x.dtype)


def _binary_cross_entropy(y_pred: FloatArray, y_true: FloatArray) -> float:
    p = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
    return -float(np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def _init_mlp_weights(rng: np.random.Generator) -> dict[str, FloatArray]:
    """Xavier initialisation for 8→32→16→1 MLP."""
    return {
        "w1": rng.normal(0, np.sqrt(2.0 / (N_FEATURES + 32)), (N_FEATURES, 32)),
        "b1": np.zeros(32),
        "w2": rng.normal(0, np.sqrt(2.0 / (32 + 16)), (32, 16)),
        "b2": np.zeros(16),
        "w3": rng.normal(0, np.sqrt(2.0 / (16 + 1)), (16, 1)),
        "b3": np.zeros(1),
    }


def _mlp_forward(x: FloatArray, weights: dict[str, FloatArray]) -> FloatArray:
    """Forward pass: 8→32→16→1 with ReLU hidden, sigmoid output."""
    h1 = _relu(x @ weights["w1"] + weights["b1"])
    h2 = _relu(h1 @ weights["w2"] + weights["b2"])
    return _sigmoid(h2 @ weights["w3"] + weights["b3"]).ravel()


def _mlp_gradients(x: FloatArray, y: FloatArray, weights: dict[str, FloatArray]) -> tuple[dict[str, FloatArray], float]:
    """Backprop for BCE loss. Returns (grads_dict, loss)."""
    n = x.shape[0]
    h1_pre = x @ weights["w1"] + weights["b1"]
    h1 = np.maximum(0.0, h1_pre)
    h2_pre = h1 @ weights["w2"] + weights["b2"]
    h2 = np.maximum(0.0, h2_pre)
    logits = (h2 @ weights["w3"] + weights["b3"]).ravel()
    y_pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))

    loss = _binary_cross_entropy(y_pred, y)

    # dL/d_logits for BCE with sigmoid output
    dl = (y_pred - y) / n  # (n,)

    grads: dict[str, FloatArray] = {}
    grads["b3"] = np.sum(dl, axis=0, keepdims=True).ravel()
    grads["w3"] = h2.T @ dl.reshape(-1, 1)

    dh2 = dl.reshape(-1, 1) @ weights["w3"].T
    dh2 = dh2 * (h2_pre > 0).astype(float)
    grads["b2"] = np.sum(dh2, axis=0)
    grads["w2"] = h1.T @ dh2

    dh1 = dh2 @ weights["w2"].T
    dh1 = dh1 * (h1_pre > 0).astype(float)
    grads["b1"] = np.sum(dh1, axis=0)
    grads["w1"] = x.T @ dh1

    return grads, loss

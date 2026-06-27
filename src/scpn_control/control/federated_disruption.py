# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Federated Disruption
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Federated Learning for Multi-Machine Disruption Prediction
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Federated learning framework for cross-machine disruption prediction.

Trains a shared MLP disruption classifier across heterogeneous tokamak
datasets (DIII-D, JET, KSTAR, EAST, SPARC) without centralising raw data.
Supports FedAvg (McMahan et al., AISTATS 2017) and FedProx (Li et al.,
MLSys 2020) aggregation strategies.

Disruption features: Ip, beta_N, q95, n/n_GW, li, dBp/dt,
locked_mode_amplitude, n1_rms — 8-dimensional input space whose
distributions differ across machines (JET: higher Ip; DIII-D: more
shaping; KSTAR: longer pulses).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from scpn_control._typing import FloatArray

logger = logging.getLogger(__name__)

N_FEATURES = 8  # Ip, beta_N, q95, n/n_GW, li, dBp/dt, locked_mode_amp, n1_rms
FEATURE_NAMES = ("Ip", "beta_N", "q95", "n_nGW", "li", "dBp_dt", "locked_mode_amp", "n1_rms")

# Machine-specific feature distribution parameters (mean, std) per feature.
# Derived from ITPA global confinement database ranges.
# Greenwald, NF 42 (2002); de Vries, NF 51 (2011).
MACHINE_PROFILES: dict[str, dict[str, tuple[float, float]]] = {
    "DIII-D": {
        "Ip": (1.2, 0.3),
        "beta_N": (2.2, 0.6),
        "q95": (4.5, 1.0),
        "n_nGW": (0.55, 0.15),
        "li": (0.90, 0.10),
        "dBp_dt": (0.8, 0.5),
        "locked_mode_amp": (0.3, 0.25),
        "n1_rms": (0.15, 0.12),
    },
    "JET": {
        "Ip": (2.8, 0.6),
        "beta_N": (1.8, 0.4),
        "q95": (3.8, 0.8),
        "n_nGW": (0.65, 0.12),
        "li": (0.85, 0.08),
        "dBp_dt": (1.2, 0.7),
        "locked_mode_amp": (0.5, 0.35),
        "n1_rms": (0.20, 0.15),
    },
    "KSTAR": {
        "Ip": (0.6, 0.15),
        "beta_N": (2.5, 0.7),
        "q95": (5.0, 1.2),
        "n_nGW": (0.45, 0.12),
        "li": (0.95, 0.12),
        "dBp_dt": (0.5, 0.3),
        "locked_mode_amp": (0.2, 0.15),
        "n1_rms": (0.10, 0.08),
    },
    "EAST": {
        "Ip": (0.5, 0.12),
        "beta_N": (2.0, 0.5),
        "q95": (5.5, 1.5),
        "n_nGW": (0.50, 0.18),
        "li": (0.88, 0.09),
        "dBp_dt": (0.4, 0.25),
        "locked_mode_amp": (0.18, 0.12),
        "n1_rms": (0.08, 0.06),
    },
    "SPARC": {
        "Ip": (8.7, 1.0),
        "beta_N": (1.5, 0.3),
        "q95": (3.5, 0.6),
        "n_nGW": (0.75, 0.10),
        "li": (0.80, 0.06),
        "dBp_dt": (2.0, 1.0),
        "locked_mode_amp": (0.6, 0.40),
        "n1_rms": (0.25, 0.18),
    },
}


def _require_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return int(value)


def _require_positive_float(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _l2_norm(weights: dict[str, FloatArray]) -> float:
    return float(np.sqrt(sum(float(np.sum(np.asarray(value, dtype=np.float64) ** 2)) for value in weights.values())))


def _weight_delta(local_weights: dict[str, FloatArray], global_weights: dict[str, FloatArray]) -> dict[str, FloatArray]:
    return {key: np.asarray(local_weights[key], dtype=np.float64) - global_weights[key] for key in global_weights}


def _apply_weight_delta(global_weights: dict[str, FloatArray], delta: dict[str, FloatArray]) -> dict[str, FloatArray]:
    return {key: global_weights[key] + delta[key] for key in global_weights}


@dataclass(frozen=True)
class DifferentialPrivacyConfig:
    """Facility-level differential privacy for federated client updates."""

    max_update_norm: float = 1.0
    noise_multiplier: float = 1.0
    delta: float = 1.0e-5
    seed: int = 20240531

    def __post_init__(self) -> None:
        _require_positive_float("max_update_norm", self.max_update_norm)
        _require_positive_float("noise_multiplier", self.noise_multiplier)
        delta_value = float(self.delta)
        if not np.isfinite(delta_value) or delta_value <= 0.0 or delta_value >= 1.0:
            raise ValueError("delta must be finite and in (0, 1)")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed:
            raise ValueError("seed must be an integer")


@dataclass(frozen=True)
class PrivacyLedgerEntry:
    """Per-round facility-level privacy accounting record."""

    round_index: int
    participating_clients: int
    epsilon_spent: float
    cumulative_epsilon: float
    delta: float
    max_update_norm: float
    noise_multiplier: float
    clipped_clients: tuple[str, ...]


@dataclass(frozen=True)
class FacilityBenchmarkSummary:
    """Deterministic benchmark summary for a federated disruption campaign."""

    aggregation: str
    machines: tuple[str, ...]
    n_rounds: int
    mean_accuracy: float
    mean_loss: float
    per_machine_accuracy: dict[str, float]
    privacy_epsilon: float | None
    privacy_delta: float | None
    evidence_kind: str


def gaussian_mechanism_epsilon(noise_multiplier: float, delta: float) -> float:
    """Return the conservative Gaussian-mechanism epsilon for one round."""
    sigma = _require_positive_float("noise_multiplier", noise_multiplier)
    delta_value = float(delta)
    if not np.isfinite(delta_value) or delta_value <= 0.0 or delta_value >= 1.0:
        raise ValueError("delta must be finite and in (0, 1)")
    return float(np.sqrt(2.0 * np.log(1.25 / delta_value)) / sigma)


def compose_privacy_epsilon(noise_multiplier: float, delta: float, n_rounds: int) -> float:
    """Linearly compose the conservative per-round Gaussian epsilon budget."""
    rounds = _require_positive_int("n_rounds", n_rounds)
    return float(rounds * gaussian_mechanism_epsilon(noise_multiplier, delta))


# ── MLP (numpy-only, same pattern as neural_transport.py) ────────────


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


# ── Differential privacy ─────────────────────────────────────────────


def differential_privacy_clip(
    gradients: dict[str, FloatArray],
    max_norm: float,
    noise_sigma: float,
    rng: np.random.Generator | None = None,
) -> dict[str, FloatArray]:
    """Clip per-parameter gradient norms and add Gaussian noise (DP-SGD).

    Reference: Abadi et al., "Deep Learning with Differential Privacy",
    CCS 2016.
    """
    rng = rng or np.random.default_rng()
    total_norm = np.sqrt(sum(float(np.sum(g**2)) for g in gradients.values()))
    clip_factor = min(1.0, max_norm / max(total_norm, 1e-12))
    clipped: dict[str, FloatArray] = {}
    for key, grad in gradients.items():
        clipped[key] = grad * clip_factor + rng.normal(0.0, noise_sigma, grad.shape)
    return clipped


# ── Data generation ──────────────────────────────────────────────────


def _generate_disruption_data(
    machine: str,
    n_samples: int,
    disruption_fraction: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    """Synthetic disruption dataset for a tokamak.

    Safe shots: features sampled from machine profile.
    Disruptive shots: elevated locked_mode_amp, dBp/dt, lower q95, higher n/n_GW.
    """
    if machine not in MACHINE_PROFILES:
        raise ValueError(f"Unknown machine {machine!r}; available: {sorted(MACHINE_PROFILES)}")
    profile = MACHINE_PROFILES[machine]
    X = np.empty((n_samples, N_FEATURES))
    y = np.zeros(n_samples)

    n_disrupt = int(n_samples * disruption_fraction)
    y[:n_disrupt] = 1.0

    for i, feat in enumerate(FEATURE_NAMES):
        mu, sigma = profile[feat]
        X[:, i] = rng.normal(mu, sigma, n_samples)

    # Shift disruptive shots toward instability boundaries
    X[:n_disrupt, 2] -= 0.8  # q95 drops (de Vries et al., NF 51 2011)
    X[:n_disrupt, 3] += 0.25  # n/n_GW rises toward Greenwald limit
    X[:n_disrupt, 5] *= 2.5  # dBp/dt spike
    X[:n_disrupt, 6] += 0.6  # locked mode growth
    X[:n_disrupt, 7] += 0.3  # n=1 RMS rise

    # Clamp non-negative features
    X[:, 0] = np.maximum(X[:, 0], 0.01)  # Ip > 0
    X[:, 3] = np.clip(X[:, 3], 0.01, 1.5)
    X[:, 4] = np.maximum(X[:, 4], 0.3)
    X[:, 6] = np.maximum(X[:, 6], 0.0)
    X[:, 7] = np.maximum(X[:, 7], 0.0)

    # Shuffle
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


# ── Config ───────────────────────────────────────────────────────────


@dataclass
class FederatedConfig:
    """Configuration for federated disruption prediction training."""

    n_rounds: int = 10
    local_epochs: int = 5
    learning_rate: float = 0.01
    aggregation: str = "fedavg"  # "fedavg" or "fedprox"
    mu_proximal: float = 0.01  # FedProx proximal term weight; Li et al. MLSys 2020
    min_clients: int = 2
    machines: list[str] = field(default_factory=lambda: ["DIII-D", "JET", "KSTAR"])
    dp_config: DifferentialPrivacyConfig | None = None

    def __post_init__(self) -> None:
        self.n_rounds = _require_positive_int("n_rounds", self.n_rounds)
        self.local_epochs = _require_positive_int("local_epochs", self.local_epochs)
        self.learning_rate = _require_positive_float("learning_rate", self.learning_rate)
        if self.aggregation not in ("fedavg", "fedprox"):
            raise ValueError(f"aggregation must be 'fedavg' or 'fedprox', got {self.aggregation!r}")
        self.mu_proximal = float(self.mu_proximal)
        if not np.isfinite(self.mu_proximal) or self.mu_proximal < 0:
            raise ValueError("mu_proximal must be >= 0")
        self.min_clients = _require_positive_int("min_clients", self.min_clients)
        for m in self.machines:
            if m not in MACHINE_PROFILES:
                raise ValueError(f"Unknown machine {m!r}; available: {sorted(MACHINE_PROFILES)}")


# ── Client ───────────────────────────────────────────────────────────


class MachineClient:
    """Local training client for a single tokamak."""

    def __init__(
        self,
        machine: str,
        X_train: FloatArray,
        y_train: FloatArray,
        X_test: FloatArray,
        y_test: FloatArray,
        learning_rate: float = 0.01,
    ) -> None:
        if machine not in MACHINE_PROFILES:
            raise ValueError(f"Unknown machine {machine!r}")
        self.machine = machine
        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.y_train = np.asarray(y_train, dtype=np.float64).ravel()
        self.X_test = np.asarray(X_test, dtype=np.float64)
        self.y_test = np.asarray(y_test, dtype=np.float64).ravel()
        self.learning_rate = _require_positive_float("learning_rate", learning_rate)
        self._validate_dataset_contract()
        self._weights: dict[str, FloatArray] = {}

    def _validate_dataset_contract(self) -> None:
        for name, x in (("X_train", self.X_train), ("X_test", self.X_test)):
            if x.ndim != 2 or x.shape[1] != N_FEATURES or not np.all(np.isfinite(x)):
                raise ValueError(f"{name} must be finite with shape (n, {N_FEATURES})")
            if x.shape[0] < 1:
                raise ValueError(f"{name} must contain at least one sample")
        for name, y, x in (("y_train", self.y_train, self.X_train), ("y_test", self.y_test, self.X_test)):
            if y.ndim != 1 or y.shape[0] != x.shape[0] or not np.all(np.isfinite(y)):
                raise ValueError(f"{name} must be finite with one label per sample")
            if not set(np.unique(y)).issubset({0.0, 1.0}):
                raise ValueError(f"{name} must contain binary disruption labels 0 or 1")

    def get_data_size(self) -> int:
        """Number of local training samples held by this federated client."""
        return int(self.X_train.shape[0])

    def local_train(
        self,
        global_weights: dict[str, FloatArray],
        n_epochs: int,
        mu_proximal: float = 0.0,
    ) -> dict[str, FloatArray]:
        """SGD on local data, starting from global_weights.

        When mu_proximal > 0, adds the FedProx penalty
        (mu/2)||w - w_global||^2 to the loss gradient.
        """
        w = {k: v.copy() for k, v in global_weights.items()}

        for _ in range(n_epochs):
            grads, _ = _mlp_gradients(self.X_train, self.y_train, w)
            for key in w:
                g = grads[key]
                if mu_proximal > 0:
                    g = g + mu_proximal * (w[key] - global_weights[key])
                w[key] = w[key] - self.learning_rate * g

        self._weights = w
        return {k: v.copy() for k, v in w.items()}

    def local_evaluate(self, weights: dict[str, FloatArray]) -> dict[str, float]:
        """Binary classification metrics on local test set."""
        y_pred_prob = _mlp_forward(self.X_test, weights)
        y_pred = (y_pred_prob >= 0.5).astype(float)
        y = self.y_test

        tp = float(np.sum((y_pred == 1) & (y == 1)))
        fp = float(np.sum((y_pred == 1) & (y == 0)))
        fn = float(np.sum((y_pred == 0) & (y == 1)))
        tn = float(np.sum((y_pred == 0) & (y == 0)))
        n = max(len(y), 1)

        accuracy = (tp + tn) / n
        precision = tp / max(tp + fp, 1e-12)
        recall = tp / max(tp + fn, 1e-12)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        loss = _binary_cross_entropy(y_pred_prob, y)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "loss": loss,
            "n_samples": int(n),
        }


# ── Server ───────────────────────────────────────────────────────────


class FederatedServer:
    """Orchestrates federated training across machine clients."""

    def __init__(self, config: FederatedConfig, seed: int = 42) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.dp_rng = np.random.default_rng(config.dp_config.seed if config.dp_config is not None else seed + 1)
        self.global_weights = _init_mlp_weights(self.rng)
        self.privacy_ledger: list[PrivacyLedgerEntry] = []

    def aggregate(self, client_updates: list[dict[str, Any]]) -> dict[str, FloatArray]:
        """FedAvg: weighted average of model weights by dataset size.

        McMahan et al., "Communication-Efficient Learning of Deep Networks
        from Decentralized Data", AISTATS 2017.
        """
        if not client_updates:
            raise ValueError("aggregate requires at least one client update")
        total = sum(u["n_samples"] for u in client_updates)
        if total <= 0:
            raise ValueError("aggregate requires positive client sample counts")
        avg: dict[str, FloatArray] = {}
        for key in client_updates[0]["weights"]:
            avg[key] = sum(u["weights"][key] * (u["n_samples"] / total) for u in client_updates)
        return avg

    def _privatise_client_updates(
        self,
        updates: list[dict[str, Any]],
        *,
        round_index: int,
    ) -> list[dict[str, Any]]:
        """Clip and noise full client model deltas for facility-level DP."""
        dp = self.config.dp_config
        if dp is None:
            return updates

        privatized: list[dict[str, Any]] = []
        clipped_clients: list[str] = []
        for update in updates:
            machine = str(update["machine"])
            delta = _weight_delta(update["weights"], self.global_weights)
            norm = _l2_norm(delta)
            clip = min(1.0, dp.max_update_norm / max(norm, 1.0e-12))
            if clip < 1.0:
                clipped_clients.append(machine)

            noised_delta: dict[str, FloatArray] = {}
            for key, value in delta.items():
                noise = self.dp_rng.normal(
                    0.0,
                    dp.noise_multiplier * dp.max_update_norm,
                    size=value.shape,
                )
                noised_delta[key] = value * clip + noise
            privatized.append(
                {
                    "weights": _apply_weight_delta(self.global_weights, noised_delta),
                    "n_samples": update["n_samples"],
                    "machine": machine,
                }
            )

        epsilon = gaussian_mechanism_epsilon(dp.noise_multiplier, dp.delta)
        cumulative = epsilon + (self.privacy_ledger[-1].cumulative_epsilon if self.privacy_ledger else 0.0)
        self.privacy_ledger.append(
            PrivacyLedgerEntry(
                round_index=round_index,
                participating_clients=len(updates),
                epsilon_spent=float(epsilon),
                cumulative_epsilon=float(cumulative),
                delta=float(dp.delta),
                max_update_norm=float(dp.max_update_norm),
                noise_multiplier=float(dp.noise_multiplier),
                clipped_clients=tuple(clipped_clients),
            )
        )
        return privatized

    def fedprox_aggregate(
        self,
        client_updates: list[dict[str, Any]],
        global_weights: dict[str, FloatArray],
        mu: float,
    ) -> dict[str, FloatArray]:
        """FedProx aggregation with proximal regularisation.

        The proximal term is applied during local training (not aggregation),
        so aggregation itself is weighted averaging — the difference from
        FedAvg is in the client-side gradient update. This method exists
        for API symmetry; the mu parameter documents the proximal weight used.
        """
        _ = mu  # applied during local_train, not aggregation
        return self.aggregate(client_updates)

    def run_round(self, clients: list[MachineClient]) -> dict[str, Any]:
        """Single federated round: distribute → local train → aggregate."""
        if len(clients) < self.config.min_clients:
            raise ValueError(f"Need >= {self.config.min_clients} clients, got {len(clients)}")

        mu = self.config.mu_proximal if self.config.aggregation == "fedprox" else 0.0
        updates: list[dict[str, Any]] = []
        client_metrics: list[dict[str, Any]] = []

        for client in clients:
            local_w = client.local_train(self.global_weights, self.config.local_epochs, mu)
            metrics = client.local_evaluate(local_w)
            updates.append({"weights": local_w, "n_samples": client.get_data_size(), "machine": client.machine})
            client_metrics.append({"machine": client.machine, **metrics})

        updates = self._privatise_client_updates(updates, round_index=len(self.privacy_ledger))
        if self.config.aggregation == "fedprox":
            self.global_weights = self.fedprox_aggregate(updates, self.global_weights, mu)
        else:
            self.global_weights = self.aggregate(updates)

        result: dict[str, Any] = {"client_metrics": client_metrics}
        if self.config.dp_config is not None:
            result["privacy"] = self.privacy_ledger[-1]
        return result

    def train(self, clients: list[MachineClient], n_rounds: int | None = None) -> list[dict[str, Any]]:
        """Full federated training loop.

        Returns per-round metrics including per-client accuracy, loss, n_samples.
        """
        rounds = n_rounds if n_rounds is not None else self.config.n_rounds
        history: list[dict[str, Any]] = []

        for r in range(rounds):
            round_result = self.run_round(clients)
            mean_loss = float(np.mean([m["loss"] for m in round_result["client_metrics"]]))
            mean_acc = float(np.mean([m["accuracy"] for m in round_result["client_metrics"]]))
            round_result["round"] = r
            round_result["mean_loss"] = mean_loss
            round_result["mean_accuracy"] = mean_acc
            history.append(round_result)
            logger.info("round %d  mean_loss=%.4f  mean_acc=%.3f", r, mean_loss, mean_acc)

        return history

    def privacy_summary(self) -> dict[str, float | int | None]:
        """Return cumulative facility-level DP spend for the current server."""
        if self.config.dp_config is None:
            return {"epsilon": None, "delta": None, "rounds": 0}
        epsilon = self.privacy_ledger[-1].cumulative_epsilon if self.privacy_ledger else 0.0
        return {
            "epsilon": float(epsilon),
            "delta": float(self.config.dp_config.delta),
            "rounds": len(self.privacy_ledger),
        }

    def get_state(self) -> dict[str, Any]:
        """Serialisable snapshot of server state."""
        dp_config = None
        if self.config.dp_config is not None:
            dp_config = {
                "max_update_norm": self.config.dp_config.max_update_norm,
                "noise_multiplier": self.config.dp_config.noise_multiplier,
                "delta": self.config.dp_config.delta,
                "seed": self.config.dp_config.seed,
            }
        return {
            "config": {
                "n_rounds": self.config.n_rounds,
                "local_epochs": self.config.local_epochs,
                "learning_rate": self.config.learning_rate,
                "aggregation": self.config.aggregation,
                "mu_proximal": self.config.mu_proximal,
                "min_clients": self.config.min_clients,
                "machines": list(self.config.machines),
                "dp_config": dp_config,
            },
            "weights": {k: v.tolist() for k, v in self.global_weights.items()},
            "privacy_ledger": [
                {
                    "round_index": entry.round_index,
                    "participating_clients": entry.participating_clients,
                    "epsilon_spent": entry.epsilon_spent,
                    "cumulative_epsilon": entry.cumulative_epsilon,
                    "delta": entry.delta,
                    "max_update_norm": entry.max_update_norm,
                    "noise_multiplier": entry.noise_multiplier,
                    "clipped_clients": list(entry.clipped_clients),
                }
                for entry in self.privacy_ledger
            ],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> FederatedServer:
        """Reconstruct server from serialised state."""
        cfg_payload = dict(state["config"])
        if isinstance(cfg_payload.get("dp_config"), dict):
            cfg_payload["dp_config"] = DifferentialPrivacyConfig(**cfg_payload["dp_config"])
        cfg = FederatedConfig(**cfg_payload)
        server = cls(cfg)
        server.global_weights = {k: np.asarray(v) for k, v in state["weights"].items()}
        server.privacy_ledger = [
            PrivacyLedgerEntry(
                round_index=int(entry["round_index"]),
                participating_clients=int(entry["participating_clients"]),
                epsilon_spent=float(entry["epsilon_spent"]),
                cumulative_epsilon=float(entry["cumulative_epsilon"]),
                delta=float(entry["delta"]),
                max_update_norm=float(entry["max_update_norm"]),
                noise_multiplier=float(entry["noise_multiplier"]),
                clipped_clients=tuple(entry["clipped_clients"]),
            )
            for entry in state.get("privacy_ledger", [])
        ]
        return server


# ── Factory ──────────────────────────────────────────────────────────


def create_machine_clients(
    machine_configs: list[dict[str, Any]],
    seed: int = 42,
) -> list[MachineClient]:
    """Create MachineClient instances with synthetic disruption data.

    Parameters
    ----------
    machine_configs : list of dicts
        Each dict must have "machine" (str) and optionally
        "n_train" (int, default 200), "n_test" (int, default 50),
        "disruption_fraction" (float, default 0.4),
        "learning_rate" (float, default 0.01).
    seed : int
        Base RNG seed; each machine gets seed + index.
    """
    clients: list[MachineClient] = []
    for i, cfg in enumerate(machine_configs):
        machine = cfg["machine"]
        n_train = cfg.get("n_train", 200)
        n_test = cfg.get("n_test", 50)
        frac = cfg.get("disruption_fraction", 0.4)
        lr = cfg.get("learning_rate", 0.01)
        rng = np.random.default_rng(seed + i)

        X_train, y_train = _generate_disruption_data(machine, n_train, frac, rng)
        X_test, y_test = _generate_disruption_data(machine, n_test, frac, rng)
        clients.append(MachineClient(machine, X_train, y_train, X_test, y_test, lr))

    return clients


def create_facility_clients_from_arrays(
    datasets: dict[str, dict[str, FloatArray]],
    *,
    learning_rate: float = 0.01,
) -> list[MachineClient]:
    """Create clients from per-facility arrays without centralising raw shots.

    Each facility payload must contain `X_train`, `y_train`, `X_test`, and
    `y_test`. The constructor enforces the shared 8-feature disruption
    contract and binary label boundary before the data can enter a federation.
    """
    if not datasets:
        raise ValueError("datasets must contain at least one facility")
    clients: list[MachineClient] = []
    for machine, payload in datasets.items():
        missing = {"X_train", "y_train", "X_test", "y_test"} - set(payload)
        if missing:
            raise ValueError(f"{machine} dataset missing required arrays: {sorted(missing)}")
        clients.append(
            MachineClient(
                machine,
                payload["X_train"],
                payload["y_train"],
                payload["X_test"],
                payload["y_test"],
                learning_rate=learning_rate,
            )
        )
    return clients


def run_synthetic_multifacility_benchmark(
    *,
    machines: list[str] | tuple[str, ...] = ("DIII-D", "JET", "KSTAR", "EAST"),
    n_rounds: int = 4,
    local_epochs: int = 3,
    aggregation: str = "fedprox",
    dp_config: DifferentialPrivacyConfig | None = None,
    seed: int = 20240531,
) -> FacilityBenchmarkSummary:
    """Run a deterministic synthetic multi-facility disruption benchmark.

    This benchmark exercises the production federation, heterogeneity, and
    privacy-accounting contracts. It is not measured cross-facility validation.
    """
    machine_list = list(machines)
    client_specs = [
        {
            "machine": machine,
            "n_train": 180,
            "n_test": 60,
            "disruption_fraction": 0.25 + 0.05 * (idx % 4),
            "learning_rate": 0.015,
        }
        for idx, machine in enumerate(machine_list)
    ]
    clients = create_machine_clients(client_specs, seed=seed)
    cfg = FederatedConfig(
        n_rounds=n_rounds,
        local_epochs=local_epochs,
        learning_rate=0.015,
        aggregation=aggregation,
        mu_proximal=0.05,
        min_clients=max(2, min(3, len(machine_list))),
        machines=machine_list,
        dp_config=dp_config,
    )
    server = FederatedServer(cfg, seed=seed)
    history = server.train(clients, n_rounds)
    final_metrics = history[-1]["client_metrics"]
    per_machine_accuracy = {str(metric["machine"]): float(metric["accuracy"]) for metric in final_metrics}
    privacy = server.privacy_summary()
    return FacilityBenchmarkSummary(
        aggregation=aggregation,
        machines=tuple(machine_list),
        n_rounds=n_rounds,
        mean_accuracy=float(history[-1]["mean_accuracy"]),
        mean_loss=float(history[-1]["mean_loss"]),
        per_machine_accuracy=per_machine_accuracy,
        privacy_epsilon=None if privacy["epsilon"] is None else float(privacy["epsilon"]),
        privacy_delta=None if privacy["delta"] is None else float(privacy["delta"]),
        evidence_kind="synthetic_multi_facility",
    )


__all__ = [
    "DifferentialPrivacyConfig",
    "FacilityBenchmarkSummary",
    "FederatedConfig",
    "FederatedServer",
    "MACHINE_PROFILES",
    "MachineClient",
    "N_FEATURES",
    "PrivacyLedgerEntry",
    "compose_privacy_epsilon",
    "create_facility_clients_from_arrays",
    "create_machine_clients",
    "differential_privacy_clip",
    "gaussian_mechanism_epsilon",
    "run_synthetic_multifacility_benchmark",
]

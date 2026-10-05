# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Federated Disruption.
"""Synthetic and array-backed machine clients for disruption federation."""

from __future__ import annotations

from typing import Any

import numpy as np

from scpn_control._typing import FloatArray
from scpn_control.control._federated_model import (
    FEATURE_NAMES,
    N_FEATURES,
    _binary_cross_entropy,
    _mlp_forward,
    _mlp_gradients,
)
from scpn_control.control._federated_privacy import _require_positive_float

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
        """Return the number of local training samples held by this federated client."""
        return int(self.X_train.shape[0])

    def local_train(
        self,
        global_weights: dict[str, FloatArray],
        n_epochs: int,
        mu_proximal: float = 0.0,
        *,
        learning_rate: float | None = None,
    ) -> dict[str, FloatArray]:
        """SGD on local data, starting from global_weights.

        When mu_proximal > 0, adds the FedProx penalty
        (mu/2)||w - w_global||^2 to the loss gradient.
        A server may supply its configured learning rate; direct local calls
        use this client's own rate.
        """
        rate = self.learning_rate if learning_rate is None else _require_positive_float("learning_rate", learning_rate)
        w = {k: v.copy() for k, v in global_weights.items()}

        for _ in range(n_epochs):
            grads, _ = _mlp_gradients(self.X_train, self.y_train, w)
            for key in w:
                g = grads[key]
                if mu_proximal > 0:
                    g = g + mu_proximal * (w[key] - global_weights[key])
                w[key] = w[key] - rate * g

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
    """Create in-process clients from arrays supplied by each facility.

    Each facility payload must contain `X_train`, `y_train`, `X_test`, and
    `y_test`. The constructor enforces the shared 8-feature disruption
    contract and binary label boundary. The caller already holds these arrays;
    this helper does not establish a remote data-isolation boundary.
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

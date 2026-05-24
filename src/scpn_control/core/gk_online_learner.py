# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Online Surrogate Retraining
"""
Online learning layer for surrogate transport model improvement.

Accumulates GK spot-check results as training data and periodically
fine-tunes the QLKNN surrogate.  Includes validation holdout and
automatic rollback if performance degrades.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """Single (input, target) pair from a GK spot-check."""

    input_10d: NDArray[np.float64]  # shape (10,)
    target_3d: NDArray[np.float64]  # shape (3,): chi_e, chi_i, D_e


@dataclass
class LearnerConfig:
    """Online learner parameters."""

    buffer_size: int = 100  # trigger retraining when buffer reaches this
    validation_fraction: float = 0.2
    n_epochs: int = 10
    learning_rate: float = 1e-4
    max_generations: int = 50  # max retraining cycles before stopping

    def __post_init__(self) -> None:
        _require_int_at_least("buffer_size", self.buffer_size, minimum=2)
        validation_fraction = float(self.validation_fraction)
        if not np.isfinite(validation_fraction) or not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be finite and within (0, 1)")
        self.validation_fraction = validation_fraction
        _require_int_at_least("n_epochs", self.n_epochs, minimum=1)
        self.learning_rate = _require_positive_float("learning_rate", self.learning_rate)
        _require_int_at_least("max_generations", self.max_generations, minimum=0)


class OnlineLearner:
    """Accumulate GK data and fine-tune the surrogate when ready."""

    def __init__(self, config: LearnerConfig | None = None) -> None:
        self.config = config or LearnerConfig()
        self.buffer: list[TrainingSample] = []
        self.generation: int = 0
        self._best_val_loss: float = float("inf")
        self._weights_backup: dict | None = None
        self.retrain_history: list[dict] = []

    def add_sample(self, input_10d: NDArray[np.float64], target_3d: NDArray[np.float64]) -> None:
        input_arr = np.asarray(input_10d, dtype=np.float64)
        target_arr = np.asarray(target_3d, dtype=np.float64)
        if input_arr.shape != (10,):
            raise ValueError(f"input_10d must have shape (10,), received {input_arr.shape}")
        if target_arr.shape != (3,):
            raise ValueError(f"target_3d must have shape (3,), received {target_arr.shape}")
        if not np.all(np.isfinite(input_arr)):
            raise ValueError("input_10d must contain only finite values")
        if not np.all(np.isfinite(target_arr)):
            raise ValueError("target_3d must contain only finite values")
        self.buffer.append(TrainingSample(input_10d=input_arr.copy(), target_3d=target_arr.copy()))

    @property
    def buffer_full(self) -> bool:
        return len(self.buffer) >= self.config.buffer_size

    def try_retrain(
        self,
        current_weights: dict | None = None,
    ) -> dict | None:
        """Attempt retraining if buffer is full.

        Parameters
        ----------
        current_weights : dict or None
            Current MLP weight arrays (w1, b1, ...). If None, trains
            from scratch on the buffer data.

        Returns
        -------
        dict or None
            Updated weights if retraining succeeded, None if skipped or
            rolled back due to validation loss increase.
        """
        if not self.buffer_full:
            return None

        if self.generation >= self.config.max_generations:
            _logger.info("Max retraining generations (%d) reached", self.config.max_generations)
            return None

        n = len(self.buffer)
        n_val = max(int(n * self.config.validation_fraction), 1)
        n_train = n - n_val

        # Shuffle and split
        rng = np.random.default_rng(self.generation)
        indices = rng.permutation(n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        X_train = np.array([self.buffer[i].input_10d for i in train_idx])
        Y_train = np.array([self.buffer[i].target_3d for i in train_idx])
        X_val = np.array([self.buffer[i].input_10d for i in val_idx])
        Y_val = np.array([self.buffer[i].target_3d for i in val_idx])

        # Simple gradient descent on MSE (no external ML framework needed)
        w1: NDArray[np.float64]
        b1: NDArray[np.float64]
        w2: NDArray[np.float64]
        b2: NDArray[np.float64]
        w3: NDArray[np.float64]
        b3: NDArray[np.float64]
        if current_weights is None:
            # Initialise small random weights
            w1 = rng.normal(0, 0.1, (10, 64))
            b1 = np.zeros(64)
            w2 = rng.normal(0, 0.1, (64, 32))
            b2 = np.zeros(32)
            w3 = rng.normal(0, 0.1, (32, 3))
            b3 = np.zeros(3)
        else:
            required = ("w1", "b1", "w2", "b2", "w3", "b3")
            missing = [name for name in required if name not in current_weights]
            if missing:
                raise ValueError(f"current_weights missing required keys: {missing}")
            w1 = np.asarray(current_weights["w1"], dtype=np.float64).copy()
            b1 = np.asarray(current_weights["b1"], dtype=np.float64).copy()
            w2 = np.asarray(current_weights["w2"], dtype=np.float64).copy()
            b2 = np.asarray(current_weights["b2"], dtype=np.float64).copy()
            w3 = np.asarray(current_weights["w3"], dtype=np.float64).copy()
            b3 = np.asarray(current_weights["b3"], dtype=np.float64).copy()
            if w1.shape != (10, 64):
                raise ValueError(f"current_weights['w1'] must have shape (10, 64), received {w1.shape}")
            if b1.shape != (64,):
                raise ValueError(f"current_weights['b1'] must have shape (64,), received {b1.shape}")
            if w2.shape != (64, 32):
                raise ValueError(f"current_weights['w2'] must have shape (64, 32), received {w2.shape}")
            if b2.shape != (32,):
                raise ValueError(f"current_weights['b2'] must have shape (32,), received {b2.shape}")
            if w3.shape != (32, 3):
                raise ValueError(f"current_weights['w3'] must have shape (32, 3), received {w3.shape}")
            if b3.shape != (3,):
                raise ValueError(f"current_weights['b3'] must have shape (3,), received {b3.shape}")
            if not (
                np.all(np.isfinite(w1))
                and np.all(np.isfinite(b1))
                and np.all(np.isfinite(w2))
                and np.all(np.isfinite(b2))
                and np.all(np.isfinite(w3))
                and np.all(np.isfinite(b3))
            ):
                raise ValueError("current_weights must contain only finite values")

        self._weights_backup = {
            "w1": w1.copy(),
            "b1": b1.copy(),
            "w2": w2.copy(),
            "b2": b2.copy(),
            "w3": w3.copy(),
            "b3": b3.copy(),
        }

        lr = self.config.learning_rate
        best_val = float("inf")
        best_weights = None

        for epoch in range(self.config.n_epochs):
            # Forward pass (train)
            h1 = np.maximum(0, X_train @ w1 + b1)
            h2 = np.maximum(0, h1 @ w2 + b2)
            pred = h2 @ w3 + b3
            loss_train = float(np.mean((pred - Y_train) ** 2))

            # Backprop (simplified, output layer only for stability)
            grad_out = 2.0 * (pred - Y_train) / n_train
            grad_w3 = h2.T @ grad_out
            grad_b3 = grad_out.sum(axis=0)

            w3 -= lr * grad_w3
            b3 -= lr * grad_b3

            # Validation loss
            h1_v = np.maximum(0, X_val @ w1 + b1)
            h2_v = np.maximum(0, h1_v @ w2 + b2)
            pred_v = h2_v @ w3 + b3
            loss_val = float(np.mean((pred_v - Y_val) ** 2))

            if loss_val < best_val:
                best_val = loss_val
                best_weights = {
                    "w1": w1.copy(),
                    "b1": b1.copy(),
                    "w2": w2.copy(),
                    "b2": b2.copy(),
                    "w3": w3.copy(),
                    "b3": b3.copy(),
                }

        # Rollback check
        if best_val >= self._best_val_loss:
            _logger.info("Validation loss did not improve (%.4f >= %.4f), rolling back", best_val, self._best_val_loss)
            self.retrain_history.append({"generation": self.generation, "accepted": False, "val_loss": best_val})
            return None

        self._best_val_loss = best_val
        self.generation += 1
        self.buffer.clear()
        self.retrain_history.append({"generation": self.generation, "accepted": True, "val_loss": best_val})
        _logger.info("Retraining gen %d accepted, val_loss=%.6f", self.generation, best_val)
        return best_weights

    def reset(self) -> None:
        self.buffer.clear()
        self.generation = 0
        self._best_val_loss = float("inf")
        self._weights_backup = None
        self.retrain_history.clear()


def _require_int_at_least(field: str, value: int, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _require_positive_float(field: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return scalar

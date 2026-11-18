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
fine-tunes a 10 → 64 → 32 → 3 ReLU surrogate:

* **Training** backpropagates through all three layers in coordinates
  standardised by the training split, so the loss weighs ``chi_e``, ``chi_i``
  and ``D_e`` equally and is dimensionless. Weights are returned in raw units,
  with the standardisation folded into the first and last layers.
* **Admission** compares the candidate with the incumbent weights on the *same*
  validation slice and rolls back unless the candidate improves on it.
* **Buffer policy:** an accepted generation consumes its window (the buffer is
  cleared and the window's information lives in the returned weights); a
  rejected attempt keeps the window, which grows until an attempt is accepted.
  Forgetting of data outside the current window is not measured.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_control._atomic import write_text_atomic
from scpn_control.core._gk_learner_mlp import (
    LAYER_SHAPES,
    Standardiser,
    Weights,
    loss_and_gradients,
    mean_squared_error,
)

_logger = logging.getLogger(__name__)

LOSS_SCALING = "mse_of_targets_standardised_by_train_split"
REPORT_SCHEMA_VERSION = "2.0"


@dataclass
class TrainingSample:
    """Single (input, target) pair from a GK spot-check."""

    input_10d: NDArray[np.float64]  # shape (10,)
    target_3d: NDArray[np.float64]  # shape (3,): chi_e, chi_i, D_e
    ood_score: float = 0.0
    source: str = "gk_spot_check"


@dataclass
class LearnerConfig:
    """Online learner parameters."""

    buffer_size: int = 100  # trigger retraining when buffer reaches this
    validation_fraction: float = 0.2
    n_epochs: int = 10
    learning_rate: float = 1e-4
    max_generations: int = 50  # max retraining cycles before stopping
    max_ood_score: float = 3.0
    min_validation_improvement: float = 0.0

    def __post_init__(self) -> None:
        """Validate buffer size, validation fraction, epochs and learning rate."""
        _require_int_at_least("buffer_size", self.buffer_size, minimum=2)
        validation_fraction = float(self.validation_fraction)
        if not np.isfinite(validation_fraction) or not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be finite and within (0, 1)")
        self.validation_fraction = validation_fraction
        _require_int_at_least("n_epochs", self.n_epochs, minimum=1)
        self.learning_rate = _require_positive_float("learning_rate", self.learning_rate)
        _require_int_at_least("max_generations", self.max_generations, minimum=0)
        self.max_ood_score = _require_nonnegative_float("max_ood_score", self.max_ood_score)
        self.min_validation_improvement = _require_nonnegative_float(
            "min_validation_improvement", self.min_validation_improvement
        )


@dataclass(frozen=True)
class RetrainDecision:
    """Auditable retraining admission decision."""

    generation: int
    accepted: bool
    val_loss: float
    incumbent_val_loss: float
    train_samples: int
    validation_samples: int
    buffer_size: int
    reason: str
    max_ood_score: float
    learning_rate: float
    n_epochs: int
    loss_scaling: str = LOSS_SCALING


class OnlineLearner:
    """Accumulate GK data and fine-tune the surrogate when ready."""

    def __init__(self, config: LearnerConfig | None = None) -> None:
        """Start with an empty sample buffer, generation zero and no retrain history."""
        self.config = config or LearnerConfig()
        self.buffer: list[TrainingSample] = []
        self.generation: int = 0
        self._last_accepted_val_loss: float = float("inf")
        self._weights_backup: dict[str, Any] | None = None
        self.retrain_history: list[dict[str, Any]] = []

    def add_sample(
        self,
        input_10d: NDArray[np.float64],
        target_3d: NDArray[np.float64],
        *,
        ood_score: float = 0.0,
        source: str = "gk_spot_check",
    ) -> None:
        """Admit one labelled GK spot-check sample into the replay buffer.

        Parameters
        ----------
        input_10d
            The 10-dimensional gyrokinetic input vector, shape ``(10,)``.
        target_3d
            The transport-coefficient target ``[chi_i, chi_e, D_e]``, shape
            ``(3,)``; values must be non-negative.
        ood_score
            Out-of-distribution score; must be non-negative and not exceed the
            configured admission threshold.
        source
            Non-empty provenance label for the sample.

        Raises
        ------
        ValueError
            If a shape, finiteness, non-negativity, OOD-threshold, or source
            constraint is violated.
        """
        input_arr = np.asarray(input_10d, dtype=np.float64)
        target_arr = np.asarray(target_3d, dtype=np.float64)
        ood_value = _require_nonnegative_float("ood_score", ood_score)
        if ood_value > self.config.max_ood_score:
            raise ValueError("ood_score exceeds configured admission threshold")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
        if input_arr.shape != (10,):
            raise ValueError(f"input_10d must have shape (10,), received {input_arr.shape}")
        if target_arr.shape != (3,):
            raise ValueError(f"target_3d must have shape (3,), received {target_arr.shape}")
        if not np.all(np.isfinite(input_arr)):
            raise ValueError("input_10d must contain only finite values")
        if not np.all(np.isfinite(target_arr)):
            raise ValueError("target_3d must contain only finite values")
        if np.any(target_arr < 0.0):
            raise ValueError("target_3d transport coefficients must be nonnegative")
        self.buffer.append(
            TrainingSample(
                input_10d=input_arr.copy(),
                target_3d=target_arr.copy(),
                ood_score=ood_value,
                source=source.strip(),
            )
        )

    @property
    def buffer_full(self) -> bool:
        """Whether the replay buffer has reached the configured size."""
        return len(self.buffer) >= self.config.buffer_size

    def try_retrain(
        self,
        current_weights: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Attempt retraining if the buffer is full.

        The buffer is split into a training and a validation slice (seeded by the
        generation). Inputs and targets are standardised by statistics of the
        training slice; all six weight arrays are trained by full-batch gradient
        descent with exact backpropagation, and the epoch with the lowest
        validation loss is kept. Losses are mean squared errors of the
        standardised targets, hence dimensionless.

        The candidate is admitted only if its validation loss is below the
        incumbent's loss *on the same validation slice* minus
        ``min_validation_improvement``. The incumbent is ``current_weights``;
        without them there is no incumbent and any finite candidate is admitted.

        Parameters
        ----------
        current_weights : dict or None
            Incumbent MLP weight arrays (w1, b1, ...) in raw units, used as the
            warm start and as the admission reference. If None, trains from a
            seeded random initialisation.

        Returns
        -------
        dict or None
            The admitted weights in raw units (raw inputs to raw targets), or None
            if skipped or rolled back.

        Raises
        ------
        ValueError
            If ``current_weights`` lacks an array or has a wrong shape or a
            non-finite value.
        """
        if not self.buffer_full:
            return None

        if self.generation >= self.config.max_generations:
            _logger.info("Max retraining generations (%d) reached", self.config.max_generations)
            self._append_decision(
                accepted=False,
                val_loss=float("nan"),
                incumbent_val_loss=float("nan"),
                train_samples=0,
                validation_samples=0,
                buffer_size=len(self.buffer),
                reason="max_generations_reached",
            )
            return None

        n = len(self.buffer)
        n_val = max(int(n * self.config.validation_fraction), 1)
        n_train = n - n_val

        rng = np.random.default_rng(self.generation)
        indices = rng.permutation(n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        inputs = np.array([sample.input_10d for sample in self.buffer])
        targets = np.array([sample.target_3d for sample in self.buffer])
        scaler = Standardiser.fit(inputs[train_idx], targets[train_idx])
        x_train, y_train = scaler.inputs(inputs[train_idx]), scaler.targets(targets[train_idx])
        x_val, y_val = scaler.inputs(inputs[val_idx]), scaler.targets(targets[val_idx])

        if current_weights is None:
            weights: Weights = {
                name: rng.normal(0.0, 0.1, shape) if name.startswith("w") else np.zeros(shape)
                for name, shape in LAYER_SHAPES.items()
            }
            incumbent_loss = float("inf")
            self._weights_backup = scaler.to_raw(weights)
        else:
            raw = _validated_weights(current_weights)
            self._weights_backup = {name: array.copy() for name, array in raw.items()}
            weights = scaler.to_standard(raw)
            incumbent_loss = mean_squared_error(weights, x_val, y_val)

        lr = self.config.learning_rate
        best_val = float("inf")
        best_weights: Weights | None = None
        # A divergent learning rate may overflow; a non-finite loss is never selected
        # and an all-non-finite run is rejected below.
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(self.config.n_epochs):
                _, gradients = loss_and_gradients(weights, x_train, y_train)
                weights = {name: weights[name] - lr * gradients[name] for name in LAYER_SHAPES}
                loss_val = mean_squared_error(weights, x_val, y_val)
                if loss_val < best_val:
                    best_val = loss_val
                    best_weights = {name: array.copy() for name, array in weights.items()}

        if best_weights is None:
            _logger.info("Every epoch produced a non-finite validation loss, rolling back")
            self._append_decision(
                accepted=False,
                val_loss=best_val,
                incumbent_val_loss=incumbent_loss,
                train_samples=n_train,
                validation_samples=n_val,
                buffer_size=n,
                reason="validation_loss_not_finite",
            )
            return None

        if not best_val < incumbent_loss - self.config.min_validation_improvement:
            _logger.info(
                "Validation loss did not improve on the incumbent (%.4f >= %.4f), rolling back",
                best_val,
                incumbent_loss,
            )
            self._append_decision(
                accepted=False,
                val_loss=best_val,
                incumbent_val_loss=incumbent_loss,
                train_samples=n_train,
                validation_samples=n_val,
                buffer_size=n,
                reason="validation_loss_not_improved",
            )
            return None

        self._last_accepted_val_loss = best_val
        self.generation += 1
        self.buffer.clear()
        self._append_decision(
            accepted=True,
            val_loss=best_val,
            incumbent_val_loss=incumbent_loss,
            train_samples=n_train,
            validation_samples=n_val,
            buffer_size=n,
            reason="accepted",
        )
        _logger.info("Retraining gen %d accepted, val_loss=%.6f", self.generation, best_val)
        return dict(scaler.to_raw(best_weights))

    def latest_decision(self) -> RetrainDecision | None:
        """Return the latest retraining decision, if any."""
        if not self.retrain_history:
            return None
        return RetrainDecision(**self.retrain_history[-1])

    def save_retrain_report(self, path: str | Path) -> None:
        """Persist retraining decisions and current learner state as JSON."""
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "generation": self.generation,
            "last_accepted_val_loss": self._last_accepted_val_loss,
            "loss_scaling": LOSS_SCALING,
            "buffer_samples": len(self.buffer),
            "config": {
                "buffer_size": self.config.buffer_size,
                "validation_fraction": self.config.validation_fraction,
                "n_epochs": self.config.n_epochs,
                "learning_rate": self.config.learning_rate,
                "max_generations": self.config.max_generations,
                "max_ood_score": self.config.max_ood_score,
                "min_validation_improvement": self.config.min_validation_improvement,
            },
            "decisions": self.retrain_history,
        }
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(output, json.dumps(report, indent=2, sort_keys=True) + "\n")

    def _append_decision(
        self,
        *,
        accepted: bool,
        val_loss: float,
        incumbent_val_loss: float,
        train_samples: int,
        validation_samples: int,
        buffer_size: int,
        reason: str,
    ) -> None:
        """Record one accepted or rejected retrain decision in the history.

        The recorded generation is the learner's generation after the decision:
        the new generation for an acceptance (counted before this call), the
        unchanged one for a rejection.
        """
        decision = RetrainDecision(
            generation=self.generation,
            accepted=accepted,
            val_loss=float(val_loss),
            incumbent_val_loss=float(incumbent_val_loss),
            train_samples=int(train_samples),
            validation_samples=int(validation_samples),
            buffer_size=int(buffer_size),
            reason=reason,
            max_ood_score=float(self.config.max_ood_score),
            learning_rate=float(self.config.learning_rate),
            n_epochs=int(self.config.n_epochs),
        )
        self.retrain_history.append(asdict(decision))

    def reset(self) -> None:
        """Clear the buffer, generation counter, weight backup, and history."""
        self.buffer.clear()
        self.generation = 0
        self._last_accepted_val_loss = float("inf")
        self._weights_backup = None
        self.retrain_history.clear()


def _validated_weights(current_weights: dict[str, Any]) -> Weights:
    """Return float copies of the six incumbent arrays after checking keys, shapes and finiteness."""
    missing = [name for name in LAYER_SHAPES if name not in current_weights]
    if missing:
        raise ValueError(f"current_weights missing required keys: {missing}")
    weights: Weights = {}
    for name, shape in LAYER_SHAPES.items():
        array = np.asarray(current_weights[name], dtype=np.float64).copy()
        if array.shape != shape:
            raise ValueError(f"current_weights[{name!r}] must have shape {shape}, received {array.shape}")
        weights[name] = array
    if not all(np.all(np.isfinite(array)) for array in weights.values()):
        raise ValueError("current_weights must contain only finite values")
    return weights


def _require_int_at_least(field: str, value: int, *, minimum: int) -> int:
    """Return an integer at least ``minimum``; refuse booleans and smaller values."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _require_positive_float(field: str, value: float) -> float:
    """Return a finite positive float; refuse anything else."""
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return scalar


def _require_nonnegative_float(field: str, value: float) -> float:
    """Return a finite non-negative float; refuse anything else."""
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return scalar


__all__ = [
    "LOSS_SCALING",
    "REPORT_SCHEMA_VERSION",
    "LearnerConfig",
    "OnlineLearner",
    "RetrainDecision",
    "TrainingSample",
]

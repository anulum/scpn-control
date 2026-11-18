# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Standardised two-hidden-layer MLP for the GK online learner.
"""Standardisation, exact backpropagation and weight folding for the GK online learner.

The learner's network maps 10 gyrokinetic inputs to 3 transport coefficients
through two ReLU hidden layers (10 → 64 → 32 → 3). Training runs in standardised
coordinates: inputs and targets are centred and scaled by statistics of the
training split, so the loss weighs ``chi_e``, ``chi_i`` and ``D_e`` equally and
is dimensionless. Weights are exchanged with callers in raw units; the affine
standardisation is folded into the first and last layers exactly, so a folded
network evaluated on raw inputs returns raw targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Weights = dict[str, FloatArray]

LAYER_SHAPES: dict[str, tuple[int, ...]] = {
    "w1": (10, 64),
    "b1": (64,),
    "w2": (64, 32),
    "b2": (32,),
    "w3": (32, 3),
    "b3": (3,),
}


@dataclass(frozen=True)
class Standardiser:
    """Per-column centring and scaling of inputs and targets, fitted on one split."""

    x_mean: FloatArray
    x_scale: FloatArray
    y_mean: FloatArray
    y_scale: FloatArray

    @classmethod
    def fit(cls, inputs: FloatArray, targets: FloatArray) -> Standardiser:
        """Fit means and standard deviations; a constant column keeps scale 1.

        Parameters
        ----------
        inputs
            Training inputs, shape ``(n, 10)``.
        targets
            Training targets, shape ``(n, 3)``.

        Returns
        -------
        Standardiser
            The fitted statistics.
        """
        x_std = inputs.std(axis=0)
        y_std = targets.std(axis=0)
        return cls(
            x_mean=inputs.mean(axis=0),
            x_scale=np.where(x_std > 0.0, x_std, 1.0),
            y_mean=targets.mean(axis=0),
            y_scale=np.where(y_std > 0.0, y_std, 1.0),
        )

    def inputs(self, values: FloatArray) -> FloatArray:
        """Return standardised inputs."""
        return (values - self.x_mean) / self.x_scale

    def targets(self, values: FloatArray) -> FloatArray:
        """Return standardised targets."""
        return (values - self.y_mean) / self.y_scale

    def to_standard(self, raw: Weights) -> Weights:
        """Return weights that act on standardised data as ``raw`` acts on raw data.

        Parameters
        ----------
        raw
            Weights that map raw inputs to raw targets.

        Returns
        -------
        Weights
            The equivalent weights in standardised coordinates.
        """
        return {
            "w1": self.x_scale[:, None] * raw["w1"],
            "b1": raw["b1"] + self.x_mean @ raw["w1"],
            "w2": raw["w2"].copy(),
            "b2": raw["b2"].copy(),
            "w3": raw["w3"] / self.y_scale[None, :],
            "b3": (raw["b3"] - self.y_mean) / self.y_scale,
        }

    def to_raw(self, standard: Weights) -> Weights:
        """Fold the standardisation into the first and last layers.

        Parameters
        ----------
        standard
            Weights that map standardised inputs to standardised targets.

        Returns
        -------
        Weights
            The equivalent weights that map raw inputs to raw targets.
        """
        return {
            "w1": standard["w1"] / self.x_scale[:, None],
            "b1": standard["b1"] - (self.x_mean / self.x_scale) @ standard["w1"],
            "w2": standard["w2"].copy(),
            "b2": standard["b2"].copy(),
            "w3": standard["w3"] * self.y_scale[None, :],
            "b3": standard["b3"] * self.y_scale + self.y_mean,
        }


def forward(weights: Weights, inputs: FloatArray) -> FloatArray:
    """Return the network output for a batch of inputs.

    Parameters
    ----------
    weights
        The six weight arrays of :data:`LAYER_SHAPES`.
    inputs
        Inputs, shape ``(n, 10)``.

    Returns
    -------
    FloatArray
        Outputs, shape ``(n, 3)``.
    """
    hidden_1 = np.maximum(0.0, inputs @ weights["w1"] + weights["b1"])
    hidden_2 = np.maximum(0.0, hidden_1 @ weights["w2"] + weights["b2"])
    output: FloatArray = hidden_2 @ weights["w3"] + weights["b3"]
    return output


def mean_squared_error(weights: Weights, inputs: FloatArray, targets: FloatArray) -> float:
    """Return the mean over samples and outputs of the squared prediction error."""
    return float(np.mean((forward(weights, inputs) - targets) ** 2))


def loss_and_gradients(weights: Weights, inputs: FloatArray, targets: FloatArray) -> tuple[float, Weights]:
    """Return :func:`mean_squared_error` and its exact gradient for all six arrays.

    The gradient is backpropagated through both ReLU layers; the ReLU derivative is
    taken as 0 at 0.

    Parameters
    ----------
    weights
        The six weight arrays.
    inputs
        Inputs, shape ``(n, 10)``.
    targets
        Targets, shape ``(n, 3)``.

    Returns
    -------
    tuple[float, Weights]
        The loss and one gradient array per weight array, with matching shapes.
    """
    pre_1 = inputs @ weights["w1"] + weights["b1"]
    hidden_1 = np.maximum(0.0, pre_1)
    pre_2 = hidden_1 @ weights["w2"] + weights["b2"]
    hidden_2 = np.maximum(0.0, pre_2)
    residual = hidden_2 @ weights["w3"] + weights["b3"] - targets
    loss = float(np.mean(residual**2))
    grad_output = 2.0 * residual / residual.size
    grad_pre_2 = (grad_output @ weights["w3"].T) * (pre_2 > 0.0)
    grad_pre_1 = (grad_pre_2 @ weights["w2"].T) * (pre_1 > 0.0)
    gradients: Weights = {
        "w1": inputs.T @ grad_pre_1,
        "b1": grad_pre_1.sum(axis=0),
        "w2": hidden_1.T @ grad_pre_2,
        "b2": grad_pre_2.sum(axis=0),
        "w3": hidden_2.T @ grad_output,
        "b3": grad_output.sum(axis=0),
    }
    return loss, gradients


__all__ = [
    "LAYER_SHAPES",
    "Standardiser",
    "Weights",
    "forward",
    "loss_and_gradients",
    "mean_squared_error",
]

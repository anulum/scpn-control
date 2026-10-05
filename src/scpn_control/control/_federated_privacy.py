# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Federated Disruption.
"""Facility-update clipping and bounded privacy accounting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scpn_control._typing import FloatArray


def _require_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return int(value)


def _require_positive_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be positive and finite")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _require_delta(value: float) -> float:
    """Require a numeric delta strictly between zero and one."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("delta must be finite and in (0, 1)")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("delta must be finite and in (0, 1)") from exc
    if not np.isfinite(result) or result <= 0.0 or result >= 1.0:
        raise ValueError("delta must be finite and in (0, 1)")
    return result


@dataclass(frozen=True)
class DifferentialPrivacyConfig:
    """Clipping and Gaussian noise parameters for facility model updates.

    The reported epsilon is a nominal single-mechanism calculation. It does
    not certify privacy for the complete training protocol or returned metrics.
    """

    max_update_norm: float = 1.0
    noise_multiplier: float = 1.0
    delta: float = 1.0e-5
    seed: int = 20240531

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_update_norm", _require_positive_float("max_update_norm", self.max_update_norm))
        object.__setattr__(self, "noise_multiplier", _require_positive_float("noise_multiplier", self.noise_multiplier))
        object.__setattr__(self, "delta", _require_delta(self.delta))
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("seed must be an integer >= 0")
        object.__setattr__(self, "seed", int(self.seed))


@dataclass(frozen=True)
class PrivacyLedgerEntry:
    """Per-round nominal Gaussian-mechanism accounting record."""

    round_index: int
    participating_clients: int
    epsilon_spent: float
    cumulative_epsilon: float
    delta: float
    max_update_norm: float
    noise_multiplier: float
    clipped_clients: tuple[str, ...]


def gaussian_mechanism_epsilon(noise_multiplier: float, delta: float) -> float:
    """Return the nominal Gaussian-mechanism epsilon formula for one round."""
    sigma = _require_positive_float("noise_multiplier", noise_multiplier)
    delta_value = _require_delta(delta)
    with np.errstate(over="ignore", divide="ignore"):
        epsilon = float(np.sqrt(2.0 * np.log(1.25 / delta_value)) / sigma)
    if not np.isfinite(epsilon):
        raise ValueError("nominal Gaussian epsilon must be finite")
    return epsilon


def compose_privacy_epsilon(noise_multiplier: float, delta: float, n_rounds: int) -> float:
    """Sum nominal per-round epsilon values without certifying composition."""
    rounds = _require_positive_int("n_rounds", n_rounds)
    epsilon = rounds * gaussian_mechanism_epsilon(noise_multiplier, delta)
    if not np.isfinite(epsilon):
        raise ValueError("composed nominal epsilon must be finite")
    return float(epsilon)


def differential_privacy_clip(
    gradients: dict[str, FloatArray],
    max_norm: float,
    noise_sigma: float,
    rng: np.random.Generator | None = None,
) -> dict[str, FloatArray]:
    """Clip an aggregate gradient dictionary and add Gaussian noise.

    This is not per-example DP-SGD and does not certify a privacy guarantee.
    """
    bound = _require_positive_float("max_norm", max_norm)
    if isinstance(noise_sigma, bool) or not isinstance(noise_sigma, (int, float, np.integer, np.floating)):
        raise ValueError("noise_sigma must be finite and nonnegative")
    sigma = float(noise_sigma)
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("noise_sigma must be finite and nonnegative")
    if not gradients:
        raise ValueError("gradients must contain at least one parameter")
    arrays: dict[str, FloatArray] = {}
    for key, grad in gradients.items():
        raw = np.asarray(grad)
        if raw.dtype.kind not in "iuf" or not np.all(np.isfinite(raw)):
            raise ValueError(f"gradients.{key} must be finite numeric values")
        arrays[key] = np.asarray(raw, dtype=np.float64)
    max_abs = max(float(np.max(np.abs(grad))) for grad in arrays.values())
    if max_abs == 0.0:
        clip_factor = 1.0
    else:
        scaled_norm = float(np.sqrt(sum(float(np.sum((grad / max_abs) ** 2)) for grad in arrays.values())))
        total_norm = max_abs * scaled_norm
        clip_factor = 1.0 if total_norm <= bound else (bound / max_abs) / scaled_norm
    generator = np.random.default_rng() if rng is None else rng
    clipped: dict[str, FloatArray] = {}
    for key, grad in arrays.items():
        value = grad * clip_factor + generator.normal(0.0, sigma, grad.shape)
        if not np.all(np.isfinite(value)):
            raise ValueError(f"gradients.{key} produced nonfinite noise output")
        clipped[key] = value
    return clipped

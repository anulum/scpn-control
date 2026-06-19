# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Out-of-Distribution Detector for GK Surrogates
"""
Detect when surrogate transport model inputs fall outside the
training distribution, triggering escalation to a full GK solver.

Three independent methods, any of which can flag OOD:
  1. Mahalanobis distance (multivariate Gaussian assumption)
  2. Ensemble disagreement (variance across K models)
  3. Input range checks (hard + soft bounds from training data)

Reference calibration: QLKNN-10D training distribution
(van de Plassche et al., Phys. Plasmas 27, 022310, 2020).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

# QLKNN-10D training distribution bounds (approximate)
# Order: R/L_Ti, R/L_Te, R/L_ne, q, s_hat, alpha_MHD, Te/Ti, Z_eff, nu_star, beta_e
_TRAINING_RANGES = np.array(
    [
        [0.0, 30.0],  # R/L_Ti
        [0.0, 30.0],  # R/L_Te
        [-5.0, 15.0],  # R/L_ne
        [1.0, 6.0],  # q
        [0.0, 5.0],  # s_hat
        [0.0, 2.0],  # alpha_MHD
        [0.3, 3.0],  # Te/Ti
        [1.0, 5.0],  # Z_eff
        [0.001, 10.0],  # nu_star
        [0.0, 0.1],  # beta_e
    ],
    dtype=np.float64,
)

# Approximate training mean and std for soft bounds
_TRAINING_MEAN = np.array(
    [6.0, 6.0, 2.0, 2.0, 1.0, 0.3, 1.0, 1.5, 0.5, 0.02],
    dtype=np.float64,
)
_TRAINING_STD = np.array(
    [5.0, 5.0, 3.0, 1.2, 1.0, 0.5, 0.5, 0.8, 1.5, 0.02],
    dtype=np.float64,
)


@dataclass
class OODResult:
    """Out-of-distribution detection result."""

    is_ood: bool
    confidence: float  # 0.0 = clearly in-distribution, 1.0 = clearly OOD
    method: str  # "mahalanobis" / "ensemble" / "range" / "combined"
    details: dict[str, Any]

    def __post_init__(self) -> None:
        confidence_value = float(self.confidence)
        if not np.isfinite(confidence_value) or confidence_value < 0.0 or confidence_value > 1.0:
            raise ValueError("confidence must be finite and within [0, 1]")
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("method must be a non-empty string")
        if not isinstance(self.details, dict):
            raise ValueError("details must be a dict")
        self.confidence = confidence_value


def _to_input_vector(
    R_L_Ti: float,
    R_L_Te: float,
    R_L_ne: float,
    q: float,
    s_hat: float,
    alpha_MHD: float,
    Te_Ti: float,
    Z_eff: float,
    nu_star: float,
    beta_e: float,
) -> NDArray[np.float64]:
    values = np.array(
        [R_L_Ti, R_L_Te, R_L_ne, q, s_hat, alpha_MHD, Te_Ti, Z_eff, nu_star, beta_e],
        dtype=np.float64,
    )
    _validate_input_vector(values)
    return values


class OODDetector:
    """Three-method OOD detector for surrogate transport models.

    Parameters
    ----------
    mahalanobis_threshold : float
        Mahalanobis distance above which an input is flagged OOD.
        Default 4.0 corresponds to ~99.99% chi-squared quantile for 10D.
    soft_sigma_threshold : float
        Number of training-set standard deviations for soft range check.
    ensemble_disagreement_threshold : float
        Relative std across ensemble predictions above which to flag OOD.
    """

    def __init__(
        self,
        *,
        mahalanobis_threshold: float = 4.0,
        soft_sigma_threshold: float = 2.0,
        ensemble_disagreement_threshold: float = 0.3,
        training_mean: NDArray[np.float64] | None = None,
        training_cov_inv: NDArray[np.float64] | None = None,
    ) -> None:
        _require_positive("mahalanobis_threshold", mahalanobis_threshold)
        _require_positive("soft_sigma_threshold", soft_sigma_threshold)
        _require_positive("ensemble_disagreement_threshold", ensemble_disagreement_threshold)
        self.mahalanobis_threshold = mahalanobis_threshold
        self.soft_sigma_threshold = soft_sigma_threshold
        self.ensemble_disagreement_threshold = ensemble_disagreement_threshold

        self._mean = np.asarray(training_mean if training_mean is not None else _TRAINING_MEAN, dtype=np.float64)
        if self._mean.shape != _TRAINING_MEAN.shape or not np.all(np.isfinite(self._mean)):
            raise ValueError("training_mean must be a finite 10-element vector")
        if training_cov_inv is not None:
            self._cov_inv = np.asarray(training_cov_inv, dtype=np.float64)
            if self._cov_inv.shape != (10, 10) or not np.all(np.isfinite(self._cov_inv)):
                raise ValueError("training_cov_inv must be a finite 10x10 matrix")
            _validate_covariance_inverse(self._cov_inv)
        else:
            # Diagonal approximation from training std
            self._cov_inv = np.diag(1.0 / np.maximum(_TRAINING_STD**2, 1e-12))

    def mahalanobis_distance(self, x: NDArray[np.float64]) -> float:
        """Compute Mahalanobis distance from training mean."""
        _validate_input_vector(x)
        diff = x - self._mean
        return float(np.sqrt(diff @ self._cov_inv @ diff))

    def check_mahalanobis(self, x: NDArray[np.float64]) -> OODResult:
        d_m = self.mahalanobis_distance(x)
        is_ood = d_m > self.mahalanobis_threshold
        confidence = min(d_m / self.mahalanobis_threshold, 1.0) if self.mahalanobis_threshold > 0 else 0.0
        return OODResult(
            is_ood=is_ood,
            confidence=confidence,
            method="mahalanobis",
            details={"distance": d_m, "threshold": self.mahalanobis_threshold},
        )

    def check_range(self, x: NDArray[np.float64]) -> OODResult:
        """Check hard bounds and soft sigma bounds."""
        _validate_input_vector(x)
        hard_violations = []
        soft_violations = []

        for i in range(min(len(x), len(_TRAINING_RANGES))):
            lo, hi = _TRAINING_RANGES[i]
            if x[i] < lo or x[i] > hi:
                hard_violations.append(i)
            sigma_dist = abs(x[i] - self._mean[i]) / max(_TRAINING_STD[i], 1e-12)
            if sigma_dist > self.soft_sigma_threshold:
                soft_violations.append(i)

        is_ood = len(hard_violations) > 0
        n_soft = len(soft_violations)
        confidence = 1.0 if hard_violations else min(n_soft / 3.0, 1.0)

        return OODResult(
            is_ood=is_ood,
            confidence=confidence,
            method="range",
            details={
                "hard_violations": hard_violations,
                "soft_violations": soft_violations,
            },
        )

    def check_ensemble(self, predictions: NDArray[np.float64]) -> OODResult:
        """Check disagreement across ensemble predictions.

        Parameters
        ----------
        predictions : array, shape (K, 3)
            Ensemble of [chi_i, chi_e, D_e] predictions from K models.
        """
        predictions = np.asarray(predictions, dtype=np.float64)
        if predictions.ndim != 2:
            raise ValueError(f"predictions must be a 2D array, received ndim={predictions.ndim}")
        if predictions.shape[1] != 3:
            raise ValueError(f"predictions must have shape (K, 3), received {predictions.shape}")
        if not np.all(np.isfinite(predictions)):
            raise ValueError("predictions must be finite")
        if np.any(predictions < 0.0):
            raise ValueError("predictions must be non-negative transport coefficients")
        if predictions.shape[0] < 2:
            return OODResult(
                is_ood=False,
                confidence=0.0,
                method="ensemble",
                details={"reason": "insufficient_models"},
            )

        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        safe_mean = np.maximum(np.abs(mean_pred), 1e-10)
        rel_std = std_pred / safe_mean
        max_rel_std = float(np.max(rel_std))

        is_ood = max_rel_std > self.ensemble_disagreement_threshold
        confidence = min(max_rel_std / self.ensemble_disagreement_threshold, 1.0)

        return OODResult(
            is_ood=is_ood,
            confidence=confidence,
            method="ensemble",
            details={
                "max_relative_std": max_rel_std,
                "per_channel_rel_std": rel_std.tolist(),
                "threshold": self.ensemble_disagreement_threshold,
            },
        )

    def check(
        self,
        x: NDArray[np.float64],
        ensemble_predictions: NDArray[np.float64] | None = None,
    ) -> OODResult:
        """Run all applicable checks and return combined result."""
        _validate_input_vector(x)
        results = [
            self.check_mahalanobis(x),
            self.check_range(x),
        ]
        if ensemble_predictions is not None:
            results.append(self.check_ensemble(ensemble_predictions))

        any_ood = any(r.is_ood for r in results)
        max_confidence = max(r.confidence for r in results)

        return OODResult(
            is_ood=any_ood,
            confidence=max_confidence,
            method="combined",
            details={
                "sub_results": {r.method: {"is_ood": r.is_ood, "confidence": r.confidence} for r in results},
            },
        )


def _require_positive(field: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field} must be finite and positive")


def _validate_covariance_inverse(cov_inv: NDArray[np.float64]) -> None:
    if not np.allclose(cov_inv, cov_inv.T, rtol=1e-12, atol=1e-12):
        raise ValueError("training_cov_inv must be symmetric")
    try:
        np.linalg.cholesky(cov_inv)
    except np.linalg.LinAlgError as exc:
        raise ValueError("training_cov_inv must be positive definite") from exc


def _validate_input_vector(x: NDArray[np.float64]) -> None:
    if x.shape != _TRAINING_MEAN.shape or not np.all(np.isfinite(x)):
        raise ValueError("x must be a finite 10-element vector")

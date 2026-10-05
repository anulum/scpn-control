# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK OOD declared numeric and metric domains

"""Check declared finite distribution/threshold/rate/metric domains without fitting or authenticating a covariance."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

ROOT = Path(__file__).resolve().parents[1]

_EXPECTED_FEATURE_SCHEMA = [
    "R_L_Ti",
    "R_L_Te",
    "R_L_ne",
    "q",
    "s_hat",
    "alpha_MHD",
    "Te_Ti",
    "Z_eff",
    "nu_star",
    "beta_e",
]
_THRESHOLD_FIELDS = ("mahalanobis", "soft_sigma", "ensemble_disagreement")
_ACCEPTANCE_FIELDS = (
    "false_positive_rate",
    "false_negative_rate",
    "max_false_positive_rate",
    "max_false_negative_rate",
    "ood_recall",
    "min_ood_recall",
)


def _validate_training_distribution(path: Path, payload: object, errors: list[dict[str, object]]) -> None:
    """Require nonblank dataset/positive integer count and finite signed means/nonnegative standard deviations.

    Both arrays contain exactly ten representable nonboolean numbers. Zero standard
    deviation remains valid descriptive metadata; no sample/covariance recomputation
    or calibration fitting occurs. Sample count is uncapped metadata only.
    """
    if not isinstance(payload, dict):
        errors.append(
            {
                "path": _portable_path(path),
                "field": "training_distribution",
                "error": "training_distribution must be an object",
            }
        )
        return
    if not isinstance(payload.get("dataset_id"), str) or not str(payload.get("dataset_id")).strip():
        errors.append(
            {"path": _portable_path(path), "field": "dataset_id", "error": "dataset_id must be a non-empty string"}
        )
    sample_count = payload.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        errors.append(
            {"path": _portable_path(path), "field": "sample_count", "error": "sample_count must be a positive integer"}
        )
    for field in ("mean", "std"):
        values = payload.get(field)
        if (
            not isinstance(values, list)
            or len(values) != len(_EXPECTED_FEATURE_SCHEMA)
            or not all(_is_number(value) for value in values)
        ):
            errors.append(
                {"path": _portable_path(path), "field": field, "error": "field must be a numeric 10-element array"}
            )
        elif field == "std" and any(float(value) < 0.0 for value in values):
            errors.append(
                {"path": _portable_path(path), "field": field, "error": "standard deviations must be non-negative"}
            )


def _validate_numeric_object(
    path: Path,
    payload: object,
    fields: tuple[str, ...],
    parent: str,
    errors: list[dict[str, object]],
) -> None:
    """Require finite positive detector thresholds or inclusive probability/rate bounds within zero and one.

    Finite/positive threshold semantics agree with the actual public OODDetector
    constructor. These declarations do not install thresholds, run held-out
    predictions or establish empirical calibration/physical deployment.
    """
    if not isinstance(payload, dict):
        errors.append({"path": _portable_path(path), "field": parent, "error": f"{parent} must be an object"})
        return
    for field in fields:
        value = payload.get(field)
        if not _is_number(value):
            errors.append({"path": _portable_path(path), "field": field, "error": "field must be numeric"})
        elif parent == "thresholds" and float(value) <= 0.0:
            errors.append({"path": _portable_path(path), "field": field, "error": "threshold must be positive"})
        elif parent == "acceptance" and not 0.0 <= float(value) <= 1.0:
            errors.append({"path": _portable_path(path), "field": field, "error": "rate must be within [0, 1]"})


def _validate_mahalanobis_metric(path: Path, payload: object, errors: list[dict[str, object]]) -> None:
    """Check method, ASCII64-hex covariance declaration, true positive_definite label and feature order.

    No covariance bytes are retrieved or matrix definiteness recomputed. These
    are author metadata checks, not provenance or actual SPD authentication.
    """
    if not isinstance(payload, dict):
        errors.append(
            {
                "path": _portable_path(path),
                "field": "mahalanobis_metric",
                "error": "mahalanobis_metric must be an object",
            }
        )
        return
    if not isinstance(payload.get("calibration_method"), str) or not str(payload.get("calibration_method")).strip():
        errors.append(
            {"path": _portable_path(path), "field": "calibration_method", "error": "field must be a non-empty string"}
        )
    covariance_sha = payload.get("covariance_inverse_sha256")
    if not isinstance(covariance_sha, str) or len(covariance_sha) != 64 or not _is_hex(covariance_sha):
        errors.append(
            {
                "path": _portable_path(path),
                "field": "covariance_inverse_sha256",
                "error": "field must be a SHA-256 hex digest",
            }
        )
    if payload.get("positive_definite") is not True:
        errors.append(
            {"path": _portable_path(path), "field": "positive_definite", "error": "metric must be positive definite"}
        )
    if payload.get("feature_order") != _EXPECTED_FEATURE_SCHEMA:
        errors.append(
            {
                "path": _portable_path(path),
                "field": "feature_order",
                "error": "feature_order must match the declared 10D GK OOD vector",
            }
        )


def _is_number(value: object) -> TypeGuard[int | float]:
    """Require a finite representable nonboolean number; overflow is a field refusal rather than a crash."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _is_hex(value: str) -> bool:
    """Require only ASCII hex characters; the public metric caller separately requires exactly64 characters."""
    return all(character in "0123456789abcdefABCDEF" for character in value)


def _portable_path(path: Path) -> str:
    """Keep original repo-relative/outside lexical path shape; filesystem resolution failures use lexical fallback."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except (ValueError, OSError, RuntimeError):
        return path.as_posix()

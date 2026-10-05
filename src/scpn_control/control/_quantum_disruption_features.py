# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Quantum Disruption Features

"""Quantum disruption features for the bounded quantum bridge."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.control._quantum_disruption_constants import (
    CONTROL_FEATURE_NAMES,
    CONTROL_TO_ITER_INDEX,
    EXTRA_ITER_INDEX,
    ITER_CENTRES,
    ITER_FEATURE_NAMES,
    ITER_MAXS,
    ITER_MINS,
    QuantumDisruptionBridgeConfig,
    QuantumFeatureMapping,
)
from scpn_control.control._quantum_disruption_utils import _bounded_score, _is_sha256, _payload_digest


def map_control_features_to_iter(
    control_features: Any,
    *,
    extra_iter_features: Mapping[str, float] | None = None,
    config: QuantumDisruptionBridgeConfig | None = None,
) -> QuantumFeatureMapping:
    """Map the CONTROL 8-feature disruption contract to the ITER 11-feature contract."""
    resolved_config = QuantumDisruptionBridgeConfig() if config is None else config
    control = _as_feature_vector("control_features", control_features, 8)
    raw = np.array(ITER_CENTRES, dtype=np.float64)
    for control_index, name in enumerate(CONTROL_FEATURE_NAMES):
        iter_index = CONTROL_TO_ITER_INDEX.get(name)
        if iter_index is not None:
            raw[iter_index] = float(control[control_index])

    supplied_extra = dict(extra_iter_features or {})
    for name, value in supplied_extra.items():
        if name not in EXTRA_ITER_INDEX:
            raise ValueError(f"unsupported extra ITER feature: {name}")
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"extra ITER feature {name} must be finite")
        raw[EXTRA_ITER_INDEX[name]] = scalar

    missing = tuple(name for name in EXTRA_ITER_INDEX if name not in supplied_extra)
    if missing and not resolved_config.allow_center_defaults:
        raise ValueError(
            "missing required ITER features; pass allow_center_defaults=True only for bounded fallback use: "
            + ", ".join(missing)
        )

    normalized = normalize_iter_features(raw)
    return QuantumFeatureMapping(
        raw_iter_features=raw,
        normalized_iter_features=normalized,
        control_feature_names=CONTROL_FEATURE_NAMES,
        iter_feature_names=ITER_FEATURE_NAMES,
        defaults_used=missing,
        claim_status=resolved_config.claim_status,
        publication_safe=False,
    )


def normalize_iter_features(raw_features: Any) -> FloatArray:
    """Normalise ITER feature values into the SCPN-QUANTUM-CONTROL 11-feature range."""
    raw = _as_feature_vector("raw_iter_features", raw_features, 11)
    denom = np.where(ITER_MAXS > ITER_MINS, ITER_MAXS - ITER_MINS, 1.0)
    return np.asarray(np.clip((raw - ITER_MINS) / denom, 0.0, 1.0), dtype=np.float64)


def _as_feature_vector(name: str, value: Any, expected_size: int) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != expected_size:
        raise ValueError(f"{name} must contain {expected_size} values")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _as_sample_matrix(name: str, value: Any) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != 8:
        raise ValueError(f"{name} must have shape (n, 8)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _amplitude_encode(values: AnyFloatArray) -> FloatArray:
    padded = np.zeros(16, dtype=np.float64)
    padded[: values.size] = values
    norm = float(np.linalg.norm(padded))
    if norm <= 1.0e-15:
        padded[0] = 1.0
        return padded
    return padded / norm


def _classical_baseline_score(normalized_iter_features: AnyFloatArray) -> float:
    q95_instability = 1.0 - normalized_iter_features[1]
    raw = (
        0.22 * normalized_iter_features[6]
        + 0.18 * normalized_iter_features[4]
        + 0.18 * normalized_iter_features[3]
        + 0.14 * q95_instability
        + 0.12 * normalized_iter_features[5]
        + 0.08 * abs(normalized_iter_features[10] - 0.5) * 2.0
        + 0.08 * normalized_iter_features[8]
    )
    return _bounded_score("classical_baseline_score", raw)


def _build_admission_evidence(
    *,
    control_features: Any,
    mapping: QuantumFeatureMapping,
    quantum_available: bool,
) -> dict[str, Any]:
    control = _as_feature_vector("control_features", control_features, 8)
    reasons = ["external_validation_required", "control_admission_blocked"]
    if mapping.defaults_used:
        reasons.append("center_defaults_used")
    if not quantum_available:
        reasons.append("quantum_backend_unavailable")
    return {
        "decision": "advisory_only",
        "publication_safe": False,
        "admitted_for_control": False,
        "defaults_used": list(mapping.defaults_used),
        "reasons": reasons,
        "required_external_evidence": [
            "measured_disruption_database",
            "quantum_backend_benchmark",
            "classical_baseline_comparison",
        ],
        "control_features_sha256": _payload_digest({"control_features": control.tolist()}),
        "normalized_iter_features_sha256": _payload_digest(
            {"normalized_iter_features": mapping.normalized_iter_features.tolist()}
        ),
        "feature_mapping_sha256": _payload_digest({"feature_mapping": mapping.payload()}),
    }


def _validate_mapping_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("quantum disruption bridge feature_mapping must be an object")
    raw = _as_feature_vector("feature_mapping.raw_iter_features", value.get("raw_iter_features"), 11)
    normalized = _as_feature_vector(
        "feature_mapping.normalized_iter_features", value.get("normalized_iter_features"), 11
    )
    if np.any(normalized < 0.0) or np.any(normalized > 1.0):
        raise ValueError("quantum disruption bridge normalized features must be in [0, 1]")
    if not np.allclose(normalize_iter_features(raw), normalized, atol=1.0e-12):
        raise ValueError("quantum disruption bridge normalized features do not match raw features")
    for key in ("control_feature_names", "iter_feature_names", "defaults_used", "unmapped_control_features"):
        if not isinstance(value.get(key), list) or any(not isinstance(item, str) for item in value[key]):
            raise ValueError(f"quantum disruption bridge feature_mapping {key} must be a list of strings")
    if tuple(value["control_feature_names"]) != CONTROL_FEATURE_NAMES:
        raise ValueError("quantum disruption bridge control feature names are unsupported")
    if tuple(value["iter_feature_names"]) != ITER_FEATURE_NAMES:
        raise ValueError("quantum disruption bridge ITER feature names are unsupported")
    if value.get("claim_status") not in {"bounded_model", "validation_gap"}:
        raise ValueError("quantum disruption bridge feature_mapping claim_status is unsupported")
    if value.get("publication_safe") is not False:
        raise ValueError("quantum disruption bridge feature_mapping publication_safe must be false")
    return value


def _validate_admission_evidence(
    value: object,
    *,
    feature_mapping: Mapping[str, Any],
    quantum_available: bool,
) -> None:
    if not isinstance(value, dict):
        raise ValueError("quantum disruption bridge admission_evidence must be an object")
    if value.get("decision") != "advisory_only":
        raise ValueError("quantum disruption bridge admission_evidence decision must be advisory_only")
    if value.get("publication_safe") is not False:
        raise ValueError("quantum disruption bridge admission_evidence publication_safe must be false")
    if value.get("admitted_for_control") is not False:
        raise ValueError("quantum disruption bridge admission_evidence admitted_for_control must be false")
    defaults_used = value.get("defaults_used")
    if not isinstance(defaults_used, list) or any(not isinstance(item, str) for item in defaults_used):
        raise ValueError("quantum disruption bridge admission_evidence defaults_used must be a list of strings")
    if defaults_used != feature_mapping["defaults_used"]:
        raise ValueError("quantum disruption bridge admission_evidence defaults_used must match feature_mapping")
    reasons = value.get("reasons")
    if not isinstance(reasons, list) or any(not isinstance(item, str) or not item for item in reasons):
        raise ValueError("quantum disruption bridge admission_evidence reasons must be non-empty strings")
    for required_reason in ("external_validation_required", "control_admission_blocked"):
        if required_reason not in reasons:
            raise ValueError(f"quantum disruption bridge admission_evidence missing reason {required_reason}")
    if defaults_used and "center_defaults_used" not in reasons:
        raise ValueError("quantum disruption bridge admission_evidence must record center_defaults_used")
    if not quantum_available and "quantum_backend_unavailable" not in reasons:
        raise ValueError("quantum disruption bridge admission_evidence must record quantum_backend_unavailable")
    required_external = value.get("required_external_evidence")
    if not isinstance(required_external, list) or any(
        not isinstance(item, str) or not item for item in required_external
    ):
        raise ValueError("quantum disruption bridge admission_evidence required_external_evidence must be strings")
    for required_evidence in (
        "measured_disruption_database",
        "quantum_backend_benchmark",
        "classical_baseline_comparison",
    ):
        if required_evidence not in required_external:
            raise ValueError(
                f"quantum disruption bridge admission_evidence missing external evidence {required_evidence}"
            )
    for key in ("control_features_sha256", "normalized_iter_features_sha256", "feature_mapping_sha256"):
        digest = value.get(key)
        if not isinstance(digest, str) or not _is_sha256(digest):
            raise ValueError(f"quantum disruption bridge admission_evidence {key} must be a SHA-256 hex digest")
    expected_normalized_digest = _payload_digest(
        {"normalized_iter_features": feature_mapping["normalized_iter_features"]}
    )
    if value["normalized_iter_features_sha256"] != expected_normalized_digest:
        raise ValueError("quantum disruption bridge admission_evidence normalized_iter_features_sha256 mismatch")
    expected_mapping_digest = _payload_digest({"feature_mapping": dict(feature_mapping)})
    if value["feature_mapping_sha256"] != expected_mapping_digest:
        raise ValueError("quantum disruption bridge admission_evidence feature_mapping_sha256 mismatch")

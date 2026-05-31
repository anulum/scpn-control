# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Quantum Disruption Bridge

"""Fail-closed bridge for optional quantum-enhanced disruption prediction.

SCPN-CONTROL owns this control-facing facade. The quantum implementation remains
owned by SCPN-QUANTUM-CONTROL and is imported only inside execution paths.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np

SCHEMA_VERSION = "scpn-control.quantum-disruption-bridge-report.v1"
KERNEL_SCHEMA_VERSION = "scpn-control.quantum-disruption-kernel-report.v1"
CERTIFICATE_SCHEMA_VERSION = "scpn-control.quantum-disruption-advisory-certificate.v1"
DEPENDENCY_CONTRACT_SCHEMA_VERSION = "scpn-control.quantum-disruption-dependency-contract.v1"
CONTROL_FACADE_OWNER = "scpn-control"
QUANTUM_BACKEND_OWNER = "scpn-quantum-control"
QUANTUM_MODULE = "scpn_quantum_control.control.q_disruption_iter"
CLAIM_BOUNDARY = (
    "advisory bounded-model quantum disruption bridge; not measured facility validation, "
    "not controller promotion, and not publication-safe evidence without external validation"
)
ITER_FEATURE_NAMES = (
    "I_p",
    "q95",
    "li",
    "n_GW",
    "beta_N",
    "P_rad",
    "locked_mode",
    "V_loop",
    "W_stored",
    "kappa",
    "dIp_dt",
)
CONTROL_FEATURE_NAMES = ("Ip", "beta_N", "q95", "n_nGW", "li", "dBp_dt", "locked_mode_amp", "n1_rms")
ITER_MINS = np.array([0.5, 1.5, 0.5, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 1.0, -5.0], dtype=np.float64)
ITER_MAXS = np.array([17.0, 8.0, 2.0, 1.5, 4.0, 100.0, 0.01, 5.0, 400.0, 2.2, 5.0], dtype=np.float64)
ITER_CENTRES = np.array([15.0, 3.0, 0.85, 0.85, 1.8, 30.0, 0.0001, 0.3, 350.0, 1.7, 0.0], dtype=np.float64)
CONTROL_TO_ITER_INDEX = {"Ip": 0, "q95": 1, "li": 2, "n_nGW": 3, "beta_N": 4, "locked_mode_amp": 6}
EXTRA_ITER_INDEX = {"P_rad": 5, "V_loop": 7, "W_stored": 8, "kappa": 9, "dIp_dt": 10}
REQUIRED_DOWNSTREAM_POLICY = (
    "do_not_admit_control_action",
    "do_not_publish_as_facility_validation",
    "require_external_evidence",
)
QUANTUM_CORE_DEPENDENCIES = (
    "qiskit>=2.2,<3.0",
    "qiskit-" + "a" + "er>=0.15,<1.0",
    "qiskit-qasm3-import>=0.6,<1.0",
)
QUANTUM_OPTIONAL_PROVIDER_DEPENDENCIES = (
    "qiskit-ibm-runtime>=0.40,<1.0",
    "amazon-bra" + "k" + "et-sdk>=1.117,<2.0",
    "azure-quantum>=3.9,<4.0",
    "qbraid>=0.12,<1.0",
    "cirq-core>=1.6,<2.0",
    "pennylane>=0.40,<1.0",
    "requests>=2.22,<3.0",
    "oqc-qcaas-client>=3.22,<4.0",
    "pulser-core>=1.8,<2.0",
    "perceval-quandela>=1.1,<2.0",
    "pytket-quantinuum>=0.59,<1.0",
    "pyquil>=4.17,<5.0",
)


@dataclass(frozen=True)
class QuantumDisruptionBridgeConfig:
    """Configuration for the optional quantum disruption bridge."""

    allow_center_defaults: bool = False
    require_quantum_backend: bool = False
    seed: int = 20240531
    backend_profile: str = "statevector"
    quantum_module: str = QUANTUM_MODULE
    claim_status: str = "bounded_model"
    source_mode: str = "control-facade"

    def __post_init__(self) -> None:
        if not isinstance(self.allow_center_defaults, bool):
            raise ValueError("allow_center_defaults must be a bool")
        if not isinstance(self.require_quantum_backend, bool):
            raise ValueError("require_quantum_backend must be a bool")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed:
            raise ValueError("seed must be an integer")
        _require_non_empty("backend_profile", self.backend_profile)
        _require_non_empty("quantum_module", self.quantum_module)
        if self.claim_status not in {"bounded_model", "validation_gap"}:
            raise ValueError("claim_status must be bounded_model or validation_gap")
        _require_non_empty("source_mode", self.source_mode)


@dataclass(frozen=True)
class QuantumFeatureMapping:
    """CONTROL-to-ITER feature mapping with provenance."""

    raw_iter_features: np.ndarray
    normalized_iter_features: np.ndarray
    control_feature_names: tuple[str, ...]
    iter_feature_names: tuple[str, ...]
    defaults_used: tuple[str, ...] = field(default_factory=tuple)
    unmapped_control_features: tuple[str, ...] = ("dBp_dt", "n1_rms")
    claim_status: str = "bounded_model"
    publication_safe: bool = False

    def payload(self) -> dict[str, Any]:
        """Return a JSON-safe mapping payload."""

        return {
            "raw_iter_features": self.raw_iter_features.tolist(),
            "normalized_iter_features": self.normalized_iter_features.tolist(),
            "control_feature_names": list(self.control_feature_names),
            "iter_feature_names": list(self.iter_feature_names),
            "defaults_used": list(self.defaults_used),
            "unmapped_control_features": list(self.unmapped_control_features),
            "claim_status": self.claim_status,
            "publication_safe": self.publication_safe,
        }


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


def normalize_iter_features(raw_features: Any) -> np.ndarray:
    """Normalise ITER feature values into the SCPN-QUANTUM-CONTROL 11-feature range."""

    raw = _as_feature_vector("raw_iter_features", raw_features, 11)
    denom = np.where(ITER_MAXS > ITER_MINS, ITER_MAXS - ITER_MINS, 1.0)
    return np.asarray(np.clip((raw - ITER_MINS) / denom, 0.0, 1.0), dtype=np.float64)


def quantum_disruption_dependency_contract() -> dict[str, Any]:
    """Return the CONTROL-to-QUANTUM disruption bridge dependency contract."""

    payload: dict[str, Any] = {
        "schema_version": DEPENDENCY_CONTRACT_SCHEMA_VERSION,
        "control_facade_owner": CONTROL_FACADE_OWNER,
        "quantum_backend_owner": QUANTUM_BACKEND_OWNER,
        "control_package": "scpn-control",
        "quantum_package": "scpn-quantum-control",
        "quantum_module": QUANTUM_MODULE,
        "report_schema_versions": {
            "bridge": SCHEMA_VERSION,
            "kernel": KERNEL_SCHEMA_VERSION,
            "certificate": CERTIFICATE_SCHEMA_VERSION,
        },
        "required_public_surface": {
            "classifier_class": "QuantumDisruptionClassifier",
            "constructor_kwargs": ["seed"],
            "predict_method": "predict",
            "predict_input": {
                "shape": [11],
                "feature_names": list(ITER_FEATURE_NAMES),
                "normalised_range": [0.0, 1.0],
                "dtype": "float64-compatible",
            },
            "predict_output": {
                "type": "scalar-float",
                "range": [0.0, 1.0],
            },
        },
        "feature_contract": {
            "control_feature_names": list(CONTROL_FEATURE_NAMES),
            "iter_feature_names": list(ITER_FEATURE_NAMES),
            "extra_iter_features": list(EXTRA_ITER_INDEX),
            "centre_defaults_allowed_only_when_declared": True,
        },
        "dependency_groups": {
            "control_runtime": ["numpy"],
            "quantum_core": list(QUANTUM_CORE_DEPENDENCIES),
            "quantum_optional_providers": list(QUANTUM_OPTIONAL_PROVIDER_DEPENDENCIES),
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "required_downstream_policy": list(REQUIRED_DOWNSTREAM_POLICY),
        "admitted_for_control": False,
        "publication_safe": False,
    }
    payload["contract_sha256"] = _contract_digest(payload)
    return validate_quantum_disruption_dependency_contract(payload)


def validate_quantum_disruption_dependency_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the CONTROL-to-QUANTUM disruption bridge dependency contract."""

    if not isinstance(payload, dict):
        raise ValueError("quantum disruption dependency contract must be an object")
    if payload.get("schema_version") != DEPENDENCY_CONTRACT_SCHEMA_VERSION:
        raise ValueError("quantum disruption dependency contract schema_version is unsupported")
    if payload.get("control_facade_owner") != CONTROL_FACADE_OWNER:
        raise ValueError("quantum disruption dependency contract control_facade_owner is unsupported")
    if payload.get("quantum_backend_owner") != QUANTUM_BACKEND_OWNER:
        raise ValueError("quantum disruption dependency contract quantum_backend_owner is unsupported")
    if payload.get("quantum_module") != QUANTUM_MODULE:
        raise ValueError("quantum disruption dependency contract quantum_module is unsupported")
    if payload.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ValueError("quantum disruption dependency contract claim_boundary is unsupported")
    if payload.get("admitted_for_control") is not False:
        raise ValueError("quantum disruption dependency contract admitted_for_control must be false")
    if payload.get("publication_safe") is not False:
        raise ValueError("quantum disruption dependency contract publication_safe must be false")
    _validate_contract_report_schemas(payload.get("report_schema_versions"))
    _validate_contract_public_surface(payload.get("required_public_surface"))
    _validate_contract_feature_contract(payload.get("feature_contract"))
    _validate_contract_dependency_groups(payload.get("dependency_groups"))
    policy = payload.get("required_downstream_policy")
    if not isinstance(policy, list) or any(not isinstance(item, str) or not item for item in policy):
        raise ValueError("quantum disruption dependency contract required_downstream_policy must be strings")
    for required_policy in REQUIRED_DOWNSTREAM_POLICY:
        if required_policy not in policy:
            raise ValueError(f"quantum disruption dependency contract missing downstream policy {required_policy}")
    declared_digest = payload.get("contract_sha256")
    if not isinstance(declared_digest, str) or not _is_sha256(declared_digest):
        raise ValueError("quantum disruption dependency contract contract_sha256 must be a SHA-256 hex digest")
    if _contract_digest(payload) != declared_digest.lower():
        raise ValueError("quantum disruption dependency contract contract_sha256 does not match payload")
    return payload


def quantum_disruption_kernel_matrix(
    samples_a: Any,
    samples_b: Any | None = None,
    *,
    config: QuantumDisruptionBridgeConfig | None = None,
) -> dict[str, Any]:
    """Return a bounded amplitude-encoding kernel report for CONTROL feature samples."""

    resolved_config = QuantumDisruptionBridgeConfig() if config is None else config
    a = _as_sample_matrix("samples_a", samples_a)
    b = a if samples_b is None else _as_sample_matrix("samples_b", samples_b)
    mapped_a = [map_control_features_to_iter(row, config=resolved_config).normalized_iter_features for row in a]
    mapped_b = [map_control_features_to_iter(row, config=resolved_config).normalized_iter_features for row in b]
    amp_a = np.vstack([_amplitude_encode(row) for row in mapped_a])
    amp_b = np.vstack([_amplitude_encode(row) for row in mapped_b])
    kernel = np.asarray(np.clip((amp_a @ amp_b.T) ** 2, 0.0, 1.0), dtype=np.float64)
    payload: dict[str, Any] = {
        "schema_version": KERNEL_SCHEMA_VERSION,
        "status": "advisory-kernel",
        "created_at": _utc_now(),
        "claim_boundary": CLAIM_BOUNDARY,
        "control_facade_owner": CONTROL_FACADE_OWNER,
        "quantum_backend_owner": QUANTUM_BACKEND_OWNER,
        "backend_profile": resolved_config.backend_profile,
        "feature_map": "amplitude-encoding-fidelity",
        "samples_a_count": int(a.shape[0]),
        "samples_b_count": int(b.shape[0]),
        "kernel_matrix": kernel.tolist(),
        "dependency_contract": quantum_disruption_dependency_contract(),
        "config": _config_payload(resolved_config),
        "admitted_for_control": False,
    }
    payload["report_certificate"] = _build_report_certificate(payload, report_kind="kernel-advisory")
    payload["payload_sha256"] = _payload_digest(payload)
    return validate_quantum_disruption_kernel_report(payload)


def run_quantum_disruption_bridge(
    control_features: Any,
    *,
    extra_iter_features: Mapping[str, float] | None = None,
    config: QuantumDisruptionBridgeConfig | None = None,
) -> dict[str, Any]:
    """Run the optional quantum disruption bridge and return an advisory report."""

    resolved_config = QuantumDisruptionBridgeConfig() if config is None else config
    mapping = map_control_features_to_iter(
        control_features,
        extra_iter_features=extra_iter_features,
        config=resolved_config,
    )
    classical_score = _classical_baseline_score(mapping.normalized_iter_features)
    quantum_score: float | None = None
    quantum_available = False
    unavailable_reason: str | None = None
    try:
        module = importlib.import_module(resolved_config.quantum_module)
        classifier_type = module.QuantumDisruptionClassifier
        classifier = classifier_type(seed=resolved_config.seed)
        quantum_score = _bounded_score("quantum_score", classifier.predict(mapping.normalized_iter_features))
        quantum_available = True
        status = "advisory"
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        if resolved_config.require_quantum_backend:
            raise RuntimeError("quantum disruption backend is required but unavailable") from exc
        unavailable_reason = f"{type(exc).__name__}: {exc}"
        status = "quantum-unavailable"

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "created_at": _utc_now(),
        "claim_boundary": CLAIM_BOUNDARY,
        "claim_status": mapping.claim_status,
        "control_facade_owner": CONTROL_FACADE_OWNER,
        "quantum_backend_owner": QUANTUM_BACKEND_OWNER,
        "quantum_module": resolved_config.quantum_module,
        "backend_profile": resolved_config.backend_profile,
        "quantum_available": quantum_available,
        "unavailable_reason": unavailable_reason,
        "feature_mapping": mapping.payload(),
        "admission_evidence": _build_admission_evidence(
            control_features=control_features,
            mapping=mapping,
            quantum_available=quantum_available,
        ),
        "classical_baseline_score": classical_score,
        "quantum_score": quantum_score,
        "risk_score": quantum_score if quantum_score is not None else classical_score,
        "admitted_for_control": False,
        "human_review_required": True,
        "dependency_contract": quantum_disruption_dependency_contract(),
        "config": _config_payload(resolved_config),
    }
    payload["report_certificate"] = _build_report_certificate(payload, report_kind="bridge-advisory")
    payload["payload_sha256"] = _payload_digest(payload)
    return validate_quantum_disruption_bridge_report(payload)


def validate_quantum_disruption_bridge_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a tamper-evident quantum disruption bridge report."""

    if not isinstance(payload, dict):
        raise ValueError("quantum disruption bridge report must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("quantum disruption bridge report schema_version is unsupported")
    if payload.get("status") not in {"advisory", "quantum-unavailable"}:
        raise ValueError("quantum disruption bridge report status is unsupported")
    if payload.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ValueError("quantum disruption bridge report claim_boundary is unsupported")
    if payload.get("control_facade_owner") != CONTROL_FACADE_OWNER:
        raise ValueError("quantum disruption bridge report control_facade_owner is unsupported")
    if payload.get("quantum_backend_owner") != QUANTUM_BACKEND_OWNER:
        raise ValueError("quantum disruption bridge report quantum_backend_owner is unsupported")
    if not isinstance(payload.get("created_at"), str) or not payload["created_at"]:
        raise ValueError("quantum disruption bridge report created_at must be non-empty")
    if payload.get("claim_status") not in {"bounded_model", "validation_gap"}:
        raise ValueError("quantum disruption bridge report claim_status is unsupported")
    if not isinstance(payload.get("quantum_available"), bool):
        raise ValueError("quantum disruption bridge report quantum_available must be a bool")
    if payload["status"] == "quantum-unavailable" and payload["quantum_available"]:
        raise ValueError("quantum disruption bridge unavailable status conflicts with quantum_available")
    if payload["status"] == "advisory" and not payload["quantum_available"]:
        raise ValueError("quantum disruption bridge advisory status requires quantum_available")
    if not isinstance(payload.get("admitted_for_control"), bool) or payload["admitted_for_control"]:
        raise ValueError("quantum disruption bridge report is not allowed to admit control action")
    if payload.get("human_review_required") is not True:
        raise ValueError("quantum disruption bridge report requires human review")
    feature_mapping = _validate_mapping_payload(payload.get("feature_mapping"))
    _validate_admission_evidence(
        payload.get("admission_evidence"),
        feature_mapping=feature_mapping,
        quantum_available=payload["quantum_available"],
    )
    _bounded_score("classical_baseline_score", payload.get("classical_baseline_score"))
    quantum_score = payload.get("quantum_score")
    if quantum_score is not None:
        _bounded_score("quantum_score", quantum_score)
    _bounded_score("risk_score", payload.get("risk_score"))
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("quantum disruption bridge report config must be an object")
    for key in ("quantum_module", "backend_profile"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"quantum disruption bridge report {key} must be non-empty")
    _validate_report_dependency_contract(payload)
    _validate_report_certificate(payload, report_kind="bridge-advisory")
    declared_digest = payload.get("payload_sha256")
    if not isinstance(declared_digest, str) or not _is_sha256(declared_digest):
        raise ValueError("quantum disruption bridge report payload_sha256 must be a SHA-256 hex digest")
    if _payload_digest(payload) != declared_digest.lower():
        raise ValueError("quantum disruption bridge report payload_sha256 does not match payload")
    return payload


def validate_quantum_disruption_kernel_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a tamper-evident quantum disruption kernel report."""

    if not isinstance(payload, dict):
        raise ValueError("quantum disruption kernel report must be an object")
    if payload.get("schema_version") != KERNEL_SCHEMA_VERSION:
        raise ValueError("quantum disruption kernel report schema_version is unsupported")
    if payload.get("status") != "advisory-kernel":
        raise ValueError("quantum disruption kernel report status is unsupported")
    if payload.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ValueError("quantum disruption kernel report claim_boundary is unsupported")
    if payload.get("control_facade_owner") != CONTROL_FACADE_OWNER:
        raise ValueError("quantum disruption kernel report control_facade_owner is unsupported")
    if payload.get("quantum_backend_owner") != QUANTUM_BACKEND_OWNER:
        raise ValueError("quantum disruption kernel report quantum_backend_owner is unsupported")
    if not isinstance(payload.get("admitted_for_control"), bool) or payload["admitted_for_control"]:
        raise ValueError("quantum disruption kernel report is not allowed to admit control action")
    matrix = np.asarray(payload.get("kernel_matrix"), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape != (payload.get("samples_a_count"), payload.get("samples_b_count")):
        raise ValueError("quantum disruption kernel report matrix shape is inconsistent")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("quantum disruption kernel report matrix must be finite")
    if np.any(matrix < -1.0e-12) or np.any(matrix > 1.0 + 1.0e-12):
        raise ValueError("quantum disruption kernel report matrix values must be in [0, 1]")
    if matrix.shape[0] == matrix.shape[1]:
        if not np.allclose(matrix, matrix.T, atol=1.0e-12):
            raise ValueError("quantum disruption kernel report square matrix must be symmetric")
        if not np.allclose(np.diag(matrix), np.ones(matrix.shape[0]), atol=1.0e-12):
            raise ValueError("quantum disruption kernel report diagonal must be one")
    _validate_report_dependency_contract(payload)
    _validate_report_certificate(payload, report_kind="kernel-advisory")
    declared_digest = payload.get("payload_sha256")
    if not isinstance(declared_digest, str) or not _is_sha256(declared_digest):
        raise ValueError("quantum disruption kernel report payload_sha256 must be a SHA-256 hex digest")
    if _payload_digest(payload) != declared_digest.lower():
        raise ValueError("quantum disruption kernel report payload_sha256 does not match payload")
    return payload


def _as_feature_vector(name: str, value: Any, expected_size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != expected_size:
        raise ValueError(f"{name} must contain {expected_size} values")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _as_sample_matrix(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != 8:
        raise ValueError(f"{name} must have shape (n, 8)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _amplitude_encode(values: np.ndarray) -> np.ndarray:
    padded = np.zeros(16, dtype=np.float64)
    padded[: values.size] = values
    norm = float(np.linalg.norm(padded))
    if norm <= 1.0e-15:
        padded[0] = 1.0
        return padded
    return padded / norm


def _classical_baseline_score(normalized_iter_features: np.ndarray) -> float:
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


def _build_report_certificate(payload: Mapping[str, Any], *, report_kind: str) -> dict[str, Any]:
    certificate: dict[str, Any] = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "report_kind": report_kind,
        "report_schema_version": payload.get("schema_version"),
        "control_facade_owner": CONTROL_FACADE_OWNER,
        "quantum_backend_owner": QUANTUM_BACKEND_OWNER,
        "dependency_contract_schema_version": _dependency_contract_schema_version(payload),
        "dependency_contract_sha256": _dependency_contract_digest(payload),
        "claim_boundary_sha256": _payload_digest({"claim_boundary": payload.get("claim_boundary")}),
        "admitted_for_control": False,
        "publication_safe": False,
        "external_validation_required": True,
        "required_downstream_policy": list(REQUIRED_DOWNSTREAM_POLICY),
        "content_sha256": _report_content_digest(payload),
    }
    certificate["certificate_sha256"] = _certificate_digest(certificate)
    return certificate


def _validate_report_dependency_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = payload.get("dependency_contract")
    if not isinstance(value, dict):
        raise ValueError("quantum disruption report dependency_contract must be an object")
    return validate_quantum_disruption_dependency_contract(value)


def _dependency_contract_schema_version(payload: Mapping[str, Any]) -> str:
    return str(_validate_report_dependency_contract(payload)["schema_version"])


def _dependency_contract_digest(payload: Mapping[str, Any]) -> str:
    return str(_validate_report_dependency_contract(payload)["contract_sha256"])


def _validate_report_certificate(payload: Mapping[str, Any], *, report_kind: str) -> None:
    value = payload.get("report_certificate")
    if not isinstance(value, dict):
        raise ValueError("quantum disruption report_certificate must be an object")
    if value.get("schema_version") != CERTIFICATE_SCHEMA_VERSION:
        raise ValueError("quantum disruption report_certificate schema_version is unsupported")
    if value.get("report_kind") != report_kind:
        raise ValueError("quantum disruption report_certificate report_kind is unsupported")
    if value.get("report_schema_version") != payload.get("schema_version"):
        raise ValueError("quantum disruption report_certificate report_schema_version mismatch")
    if value.get("control_facade_owner") != CONTROL_FACADE_OWNER:
        raise ValueError("quantum disruption report_certificate control_facade_owner is unsupported")
    if value.get("quantum_backend_owner") != QUANTUM_BACKEND_OWNER:
        raise ValueError("quantum disruption report_certificate quantum_backend_owner is unsupported")
    if value.get("dependency_contract_schema_version") != _dependency_contract_schema_version(payload):
        raise ValueError("quantum disruption report_certificate dependency_contract_schema_version mismatch")
    dependency_digest = value.get("dependency_contract_sha256")
    if not isinstance(dependency_digest, str) or not _is_sha256(dependency_digest):
        raise ValueError(
            "quantum disruption report_certificate dependency_contract_sha256 must be a SHA-256 hex digest"
        )
    if dependency_digest != _dependency_contract_digest(payload):
        raise ValueError("quantum disruption report_certificate dependency_contract_sha256 mismatch")
    if value.get("admitted_for_control") is not False:
        raise ValueError("quantum disruption report_certificate admitted_for_control must be false")
    if value.get("publication_safe") is not False:
        raise ValueError("quantum disruption report_certificate publication_safe must be false")
    if value.get("external_validation_required") is not True:
        raise ValueError("quantum disruption report_certificate external_validation_required must be true")
    policy = value.get("required_downstream_policy")
    if not isinstance(policy, list) or any(not isinstance(item, str) or not item for item in policy):
        raise ValueError("quantum disruption report_certificate required_downstream_policy must be strings")
    for required_policy in REQUIRED_DOWNSTREAM_POLICY:
        if required_policy not in policy:
            raise ValueError(f"quantum disruption report_certificate missing downstream policy {required_policy}")
    claim_digest = value.get("claim_boundary_sha256")
    if not isinstance(claim_digest, str) or not _is_sha256(claim_digest):
        raise ValueError("quantum disruption report_certificate claim_boundary_sha256 must be a SHA-256 hex digest")
    if claim_digest != _payload_digest({"claim_boundary": payload.get("claim_boundary")}):
        raise ValueError("quantum disruption report_certificate claim_boundary_sha256 mismatch")
    content_digest = value.get("content_sha256")
    if not isinstance(content_digest, str) or not _is_sha256(content_digest):
        raise ValueError("quantum disruption report_certificate content_sha256 must be a SHA-256 hex digest")
    if content_digest != _report_content_digest(payload):
        raise ValueError("quantum disruption report_certificate content_sha256 mismatch")
    certificate_digest = value.get("certificate_sha256")
    if not isinstance(certificate_digest, str) or not _is_sha256(certificate_digest):
        raise ValueError("quantum disruption report_certificate certificate_sha256 must be a SHA-256 hex digest")
    if certificate_digest.lower() != _certificate_digest(value):
        raise ValueError("quantum disruption report_certificate certificate_sha256 mismatch")


def _validate_contract_report_schemas(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("quantum disruption dependency contract report_schema_versions must be an object")
    expected = {
        "bridge": SCHEMA_VERSION,
        "kernel": KERNEL_SCHEMA_VERSION,
        "certificate": CERTIFICATE_SCHEMA_VERSION,
    }
    if value != expected:
        raise ValueError("quantum disruption dependency contract report_schema_versions are unsupported")


def _validate_contract_public_surface(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("quantum disruption dependency contract required_public_surface must be an object")
    if value.get("classifier_class") != "QuantumDisruptionClassifier":
        raise ValueError("quantum disruption dependency contract classifier_class is unsupported")
    if value.get("constructor_kwargs") != ["seed"]:
        raise ValueError("quantum disruption dependency contract constructor_kwargs are unsupported")
    if value.get("predict_method") != "predict":
        raise ValueError("quantum disruption dependency contract predict_method is unsupported")
    predict_input = value.get("predict_input")
    if not isinstance(predict_input, dict):
        raise ValueError("quantum disruption dependency contract predict_input must be an object")
    if predict_input.get("shape") != [11]:
        raise ValueError("quantum disruption dependency contract predict_input shape is unsupported")
    if predict_input.get("feature_names") != list(ITER_FEATURE_NAMES):
        raise ValueError("quantum disruption dependency contract predict_input feature_names are unsupported")
    if predict_input.get("normalised_range") != [0.0, 1.0]:
        raise ValueError("quantum disruption dependency contract predict_input normalised_range is unsupported")
    if predict_input.get("dtype") != "float64-compatible":
        raise ValueError("quantum disruption dependency contract predict_input dtype is unsupported")
    predict_output = value.get("predict_output")
    if not isinstance(predict_output, dict):
        raise ValueError("quantum disruption dependency contract predict_output must be an object")
    if predict_output.get("type") != "scalar-float":
        raise ValueError("quantum disruption dependency contract predict_output type is unsupported")
    if predict_output.get("range") != [0.0, 1.0]:
        raise ValueError("quantum disruption dependency contract predict_output range is unsupported")


def _validate_contract_feature_contract(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("quantum disruption dependency contract feature_contract must be an object")
    if value.get("control_feature_names") != list(CONTROL_FEATURE_NAMES):
        raise ValueError("quantum disruption dependency contract control_feature_names are unsupported")
    if value.get("iter_feature_names") != list(ITER_FEATURE_NAMES):
        raise ValueError("quantum disruption dependency contract iter_feature_names are unsupported")
    if value.get("extra_iter_features") != list(EXTRA_ITER_INDEX):
        raise ValueError("quantum disruption dependency contract extra_iter_features are unsupported")
    if value.get("centre_defaults_allowed_only_when_declared") is not True:
        raise ValueError("quantum disruption dependency contract centre default policy is unsupported")


def _validate_contract_dependency_groups(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("quantum disruption dependency contract dependency_groups must be an object")
    if value.get("control_runtime") != ["numpy"]:
        raise ValueError("quantum disruption dependency contract control_runtime dependencies are unsupported")
    if value.get("quantum_core") != list(QUANTUM_CORE_DEPENDENCIES):
        raise ValueError("quantum disruption dependency contract quantum_core dependencies are unsupported")
    if value.get("quantum_optional_providers") != list(QUANTUM_OPTIONAL_PROVIDER_DEPENDENCIES):
        raise ValueError(
            "quantum disruption dependency contract quantum_optional_providers dependencies are unsupported"
        )


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


def _config_payload(config: QuantumDisruptionBridgeConfig) -> dict[str, Any]:
    return asdict(config)


def _bounded_score(name: str, value: object) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    score = float(value)
    if score < 0.0 or score > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return score


def _require_non_empty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _payload_digest(payload: Mapping[str, Any]) -> str:
    digest_payload = {key: value for key, value in payload.items() if key != "payload_sha256"}
    encoded = json.dumps(_jsonable(digest_payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _report_content_digest(payload: Mapping[str, Any]) -> str:
    content = {key: value for key, value in payload.items() if key not in {"payload_sha256", "report_certificate"}}
    return _payload_digest(content)


def _certificate_digest(certificate: Mapping[str, Any]) -> str:
    content = {key: value for key, value in certificate.items() if key != "certificate_sha256"}
    return _payload_digest({"report_certificate": content})


def _contract_digest(contract: Mapping[str, Any]) -> str:
    content = {key: value for key, value in contract.items() if key != "contract_sha256"}
    return _payload_digest({"dependency_contract": content})


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.floating | np.integer):
        return value.item()
    return value


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

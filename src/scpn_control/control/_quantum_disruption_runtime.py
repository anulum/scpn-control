# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Quantum Disruption Runtime

"""Quantum disruption runtime for the bounded quantum bridge."""

from __future__ import annotations

import importlib
from typing import Any, Mapping

import numpy as np

from scpn_control.control._quantum_disruption_backend import _build_backend_contract_attestation
from scpn_control.control._quantum_disruption_constants import (
    CLAIM_BOUNDARY,
    CONTROL_FACADE_OWNER,
    KERNEL_SCHEMA_VERSION,
    QUANTUM_BACKEND_OWNER,
    SCHEMA_VERSION,
    QuantumDisruptionBridgeConfig,
)
from scpn_control.control._quantum_disruption_contract import quantum_disruption_dependency_contract
from scpn_control.control._quantum_disruption_features import (
    _amplitude_encode,
    _as_sample_matrix,
    _build_admission_evidence,
    _classical_baseline_score,
    map_control_features_to_iter,
)
from scpn_control.control._quantum_disruption_reports import (
    _build_advisory_decision,
    _build_report_certificate,
    validate_quantum_disruption_bridge_report,
    validate_quantum_disruption_kernel_report,
)
from scpn_control.control._quantum_disruption_utils import (
    _bounded_score,
    _config_payload,
    _payload_digest,
    _utc_now,
)


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
        "backend_contract_attestation": _build_backend_contract_attestation(
            module=None,
            expected_contract=quantum_disruption_dependency_contract(),
            status="not_evaluated",
        ),
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
    dependency_contract = quantum_disruption_dependency_contract()
    backend_contract_attestation: dict[str, Any] | None = None
    try:
        module = importlib.import_module(resolved_config.quantum_module)
        backend_contract_attestation = _build_backend_contract_attestation(
            module=module,
            expected_contract=dependency_contract,
            status="available",
        )
        classifier_type = module.QuantumDisruptionClassifier
        classifier = classifier_type(seed=resolved_config.seed)
        quantum_score = _bounded_score("quantum_score", classifier.predict(mapping.normalized_iter_features))
        quantum_available = True
        status = "advisory"
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        if resolved_config.require_quantum_backend:
            raise RuntimeError("quantum disruption backend is required but unavailable") from exc
        unavailable_reason = f"{type(exc).__name__}: {exc}"
        backend_contract_attestation = _build_backend_contract_attestation(
            module=None,
            expected_contract=dependency_contract,
            status="backend_unavailable",
        )
        status = "quantum-unavailable"

    risk_score = quantum_score if quantum_score is not None else classical_score
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
        "risk_score": risk_score,
        "admitted_for_control": False,
        "human_review_required": True,
        "dependency_contract": dependency_contract,
        "backend_contract_attestation": backend_contract_attestation,
        "advisory_decision": _build_advisory_decision(
            classical_baseline_score=classical_score,
            quantum_score=quantum_score,
            backend_contract_attestation=backend_contract_attestation,
        ),
        "config": _config_payload(resolved_config),
    }
    payload["report_certificate"] = _build_report_certificate(payload, report_kind="bridge-advisory")
    payload["payload_sha256"] = _payload_digest(payload)
    return validate_quantum_disruption_bridge_report(payload)

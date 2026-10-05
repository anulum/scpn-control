# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Quantum Disruption Backend

"""Quantum disruption backend for the bounded quantum bridge."""

from __future__ import annotations

from typing import Any, Mapping

from scpn_control.control._quantum_disruption_contract import (
    _dependency_contract_digest,
    validate_quantum_disruption_dependency_contract,
)
from scpn_control.control._quantum_disruption_utils import (
    _backend_attestation_payload_digest,
    _is_sha256,
)


def _build_backend_contract_attestation(
    *,
    module: object | None,
    expected_contract: Mapping[str, Any],
    status: str,
) -> dict[str, Any]:
    expected_digest = str(expected_contract["contract_sha256"])
    if status == "backend_unavailable":
        return _seal_backend_contract_attestation(
            {
                "status": "backend_unavailable",
                "backend_contract_validated": False,
                "expected_contract_sha256": expected_digest,
                "observed_contract_sha256": None,
                "reasons": ["quantum_backend_unavailable"],
            }
        )
    if status == "not_evaluated":
        return _seal_backend_contract_attestation(
            {
                "status": "not_evaluated",
                "backend_contract_validated": False,
                "expected_contract_sha256": expected_digest,
                "observed_contract_sha256": None,
                "reasons": ["quantum_backend_not_imported_for_kernel_report"],
            }
        )
    if module is None:
        raise RuntimeError("quantum disruption backend contract attestation requires a module")
    contract_factory = getattr(module, "scpn_control_bridge_dependency_contract", None)
    if contract_factory is None:
        return _seal_backend_contract_attestation(
            {
                "status": "not_exposed",
                "backend_contract_validated": False,
                "expected_contract_sha256": expected_digest,
                "observed_contract_sha256": None,
                "reasons": ["backend_contract_not_exposed"],
            }
        )
    if not callable(contract_factory):
        raise RuntimeError("quantum disruption backend contract factory is not callable")
    try:
        observed_contract = validate_quantum_disruption_dependency_contract(contract_factory())
    except ValueError as exc:
        raise RuntimeError("quantum disruption backend contract validation failed") from exc
    observed_digest = str(observed_contract["contract_sha256"])
    if observed_digest != expected_digest:
        raise RuntimeError("quantum disruption backend contract mismatch")
    return _seal_backend_contract_attestation(
        {
            "status": "matched",
            "backend_contract_validated": True,
            "expected_contract_sha256": expected_digest,
            "observed_contract_sha256": observed_digest,
            "reasons": ["backend_contract_matched"],
        }
    )


def _seal_backend_contract_attestation(payload: dict[str, Any]) -> dict[str, Any]:
    payload["attestation_sha256"] = _backend_attestation_payload_digest(payload)
    return payload


def _validate_backend_contract_attestation(value: object, *, payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("quantum disruption backend_contract_attestation must be an object")
    status = value.get("status")
    if status not in {"matched", "not_exposed", "backend_unavailable", "not_evaluated"}:
        raise ValueError("quantum disruption backend_contract_attestation status is unsupported")
    if not isinstance(value.get("backend_contract_validated"), bool):
        raise ValueError("quantum disruption backend_contract_attestation backend_contract_validated must be a bool")
    expected_digest = value.get("expected_contract_sha256")
    if not isinstance(expected_digest, str) or not _is_sha256(expected_digest):
        raise ValueError(
            "quantum disruption backend_contract_attestation expected_contract_sha256 must be a SHA-256 hex digest"
        )
    if expected_digest != _dependency_contract_digest(payload):
        raise ValueError("quantum disruption backend_contract_attestation expected_contract_sha256 mismatch")
    observed_digest = value.get("observed_contract_sha256")
    if observed_digest is not None and (not isinstance(observed_digest, str) or not _is_sha256(observed_digest)):
        raise ValueError(
            "quantum disruption backend_contract_attestation observed_contract_sha256 must be null or SHA-256"
        )
    if status == "matched":
        if value["backend_contract_validated"] is not True:
            raise ValueError("quantum disruption backend_contract_attestation matched status must be validated")
        if observed_digest != expected_digest:
            raise ValueError("quantum disruption backend_contract_attestation observed_contract_sha256 mismatch")
    else:
        if value["backend_contract_validated"] is not False:
            raise ValueError("quantum disruption backend_contract_attestation non-matched status must not validate")
        if observed_digest is not None:
            raise ValueError("quantum disruption backend_contract_attestation non-matched status must not observe")
    reasons = value.get("reasons")
    if not isinstance(reasons, list) or any(not isinstance(item, str) or not item for item in reasons):
        raise ValueError("quantum disruption backend_contract_attestation reasons must be non-empty strings")
    required_reason = {
        "matched": "backend_contract_matched",
        "not_exposed": "backend_contract_not_exposed",
        "backend_unavailable": "quantum_backend_unavailable",
        "not_evaluated": "quantum_backend_not_imported_for_kernel_report",
    }[status]
    if required_reason not in reasons:
        raise ValueError("quantum disruption backend_contract_attestation required reason missing")
    attestation_digest = value.get("attestation_sha256")
    if not isinstance(attestation_digest, str) or not _is_sha256(attestation_digest):
        raise ValueError("quantum disruption backend_contract_attestation attestation_sha256 must be a SHA-256")
    if attestation_digest.lower() != _backend_attestation_payload_digest(value):
        raise ValueError("quantum disruption backend_contract_attestation attestation_sha256 mismatch")
    return value


def _backend_contract_attestation_digest(payload: Mapping[str, Any]) -> str:
    attestation = _validate_backend_contract_attestation(payload.get("backend_contract_attestation"), payload=payload)
    return str(attestation["attestation_sha256"])

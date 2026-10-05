# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Quantum Disruption Utils

"""Quantum disruption utils for the bounded quantum bridge."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, Mapping

import numpy as np

from scpn_control.control._quantum_disruption_constants import QuantumDisruptionBridgeConfig


def _config_payload(config: QuantumDisruptionBridgeConfig) -> dict[str, Any]:
    return asdict(config)


def _bounded_score(name: str, value: object) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    score = float(value)
    if score < 0.0 or score > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return score


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


def _backend_attestation_payload_digest(attestation: Mapping[str, Any]) -> str:
    content = {key: value for key, value in attestation.items() if key != "attestation_sha256"}
    return _payload_digest({"backend_contract_attestation": content})


def _advisory_decision_payload_digest(decision: Mapping[str, Any]) -> str:
    content = {key: value for key, value in decision.items() if key != "decision_sha256"}
    return _payload_digest({"advisory_decision": content})


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
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")

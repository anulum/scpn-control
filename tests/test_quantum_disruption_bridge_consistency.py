# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Quantum disruption report consistency tests
"""Exercise public report validation against internally resealed inconsistent scores."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pytest

from scpn_control.control.quantum_disruption_bridge import (
    QuantumDisruptionBridgeConfig,
    run_quantum_disruption_bridge,
    validate_quantum_disruption_bridge_report,
)


def _digest(payload: dict[str, Any]) -> str:
    """Calculate the documented canonical JSON digest independently of production helpers."""
    content = {key: value for key, value in payload.items() if key != "payload_sha256"}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reseal(report: dict[str, Any]) -> None:
    """Recompute all public digest layers after a caller edits advisory content."""
    decision = report["advisory_decision"]
    decision["decision_sha256"] = _digest(
        {"advisory_decision": {key: value for key, value in decision.items() if key != "decision_sha256"}}
    )
    certificate = report["report_certificate"]
    certificate["advisory_decision_sha256"] = decision["decision_sha256"]
    certificate["content_sha256"] = _digest(
        {key: value for key, value in report.items() if key not in {"payload_sha256", "report_certificate"}}
    )
    certificate["certificate_sha256"] = _digest(
        {"report_certificate": {key: value for key, value in certificate.items() if key != "certificate_sha256"}}
    )
    report["payload_sha256"] = _digest(report)


def test_public_validator_rejects_resealed_risk_score_unbound_to_source_score() -> None:
    """A self-consistent digest chain must not change the score basis relationship."""
    report = run_quantum_disruption_bridge(
        np.array([1.2, 2.4, 3.4, 0.7, 0.9, 0.15, 0.004, 0.2]),
        config=QuantumDisruptionBridgeConfig(allow_center_defaults=True, quantum_module="missing.quantum.backend"),
    )
    assert report["risk_score"] != 1.0
    report["risk_score"] = 1.0
    report["advisory_decision"]["risk_score"] = 1.0
    report["advisory_decision"]["risk_band"] = "high"
    _reseal(report)

    with pytest.raises(ValueError, match="risk_score must match selected source score"):
        validate_quantum_disruption_bridge_report(report)


def test_public_validator_rejects_resealed_classical_score_unbound_to_features() -> None:
    """A caller cannot invent a classical score after the feature map is fixed."""
    report = run_quantum_disruption_bridge(
        np.array([1.2, 2.4, 3.4, 0.7, 0.9, 0.15, 0.004, 0.2]),
        config=QuantumDisruptionBridgeConfig(allow_center_defaults=True, quantum_module="missing.quantum.backend"),
    )
    assert report["classical_baseline_score"] != 0.0
    report["classical_baseline_score"] = 0.0
    report["risk_score"] = 0.0
    report["advisory_decision"]["risk_score"] = 0.0
    report["advisory_decision"]["risk_band"] = "low"
    _reseal(report)

    with pytest.raises(ValueError, match="classical_baseline_score must match feature mapping"):
        validate_quantum_disruption_bridge_report(report)


def test_public_validator_rejects_available_backend_without_quantum_score() -> None:
    """An advisory backend cannot be marked available without its score."""
    report = run_quantum_disruption_bridge(
        np.array([1.2, 2.4, 3.4, 0.7, 0.9, 0.15, 0.004, 0.2]),
        config=QuantumDisruptionBridgeConfig(allow_center_defaults=True, quantum_module="missing.quantum.backend"),
    )
    report["status"] = "advisory"
    report["quantum_available"] = True
    _reseal(report)

    with pytest.raises(ValueError, match="quantum_available must match quantum_score presence"):
        validate_quantum_disruption_bridge_report(report)


def test_public_validator_rejects_backend_unavailable_attestation_with_quantum_score() -> None:
    """An unobserved backend cannot claim a quantum score through re-sealing."""
    report = run_quantum_disruption_bridge(
        np.array([1.2, 2.4, 3.4, 0.7, 0.9, 0.15, 0.004, 0.2]),
        config=QuantumDisruptionBridgeConfig(allow_center_defaults=True, quantum_module="missing.quantum.backend"),
    )
    report["status"] = "advisory"
    report["quantum_available"] = True
    report["quantum_score"] = 0.6
    report["risk_score"] = 0.6
    report["advisory_decision"]["risk_score"] = 0.6
    report["advisory_decision"]["score_basis"] = "quantum_score"
    report["advisory_decision"]["risk_band"] = "elevated"
    _reseal(report)

    with pytest.raises(ValueError, match="backend attestation must match quantum availability"):
        validate_quantum_disruption_bridge_report(report)

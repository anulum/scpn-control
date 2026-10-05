# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Quantum Disruption Constants

"""Quantum disruption constants for the bounded quantum bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from scpn_control._typing import AnyFloatArray

SCHEMA_VERSION = "scpn-control.quantum-disruption-bridge-report.v1"
KERNEL_SCHEMA_VERSION = "scpn-control.quantum-disruption-kernel-report.v1"
CERTIFICATE_SCHEMA_VERSION = "scpn-control.quantum-disruption-advisory-certificate.v1"
DEPENDENCY_CONTRACT_SCHEMA_VERSION = "scpn-control.quantum-disruption-dependency-contract.v1"
ADVISORY_DECISION_SCHEMA_VERSION = "scpn-control.quantum-disruption-advisory-decision.v1"
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
RISK_BAND_THRESHOLDS = {"elevated": 0.4, "high": 0.7}
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

    raw_iter_features: AnyFloatArray
    normalized_iter_features: AnyFloatArray
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


def _require_non_empty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value

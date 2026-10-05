# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Quantum Disruption Bridge
"""Stable public facade for the bounded quantum disruption bridge."""

from __future__ import annotations

from scpn_control.control._quantum_disruption_backend import (
    _build_backend_contract_attestation as _build_backend_contract_attestation,
)
from scpn_control.control._quantum_disruption_constants import (
    ADVISORY_DECISION_SCHEMA_VERSION,
    CERTIFICATE_SCHEMA_VERSION,
    CLAIM_BOUNDARY,
    CONTROL_FACADE_OWNER,
    CONTROL_FEATURE_NAMES,
    CONTROL_TO_ITER_INDEX,
    DEPENDENCY_CONTRACT_SCHEMA_VERSION,
    EXTRA_ITER_INDEX,
    ITER_CENTRES,
    ITER_FEATURE_NAMES,
    ITER_MAXS,
    ITER_MINS,
    KERNEL_SCHEMA_VERSION,
    QUANTUM_BACKEND_OWNER,
    QUANTUM_CORE_DEPENDENCIES,
    QUANTUM_MODULE,
    QUANTUM_OPTIONAL_PROVIDER_DEPENDENCIES,
    REQUIRED_DOWNSTREAM_POLICY,
    RISK_BAND_THRESHOLDS,
    SCHEMA_VERSION,
    QuantumDisruptionBridgeConfig,
    QuantumFeatureMapping,
)
from scpn_control.control._quantum_disruption_contract import (
    quantum_disruption_dependency_contract,
    validate_quantum_disruption_dependency_contract,
)
from scpn_control.control._quantum_disruption_features import (
    _amplitude_encode as _amplitude_encode,
)
from scpn_control.control._quantum_disruption_features import (
    map_control_features_to_iter,
    normalize_iter_features,
)
from scpn_control.control._quantum_disruption_reports import (
    _risk_band as _risk_band,
)
from scpn_control.control._quantum_disruption_reports import (
    validate_quantum_disruption_bridge_report,
    validate_quantum_disruption_kernel_report,
)
from scpn_control.control._quantum_disruption_runtime import (
    quantum_disruption_kernel_matrix,
    run_quantum_disruption_bridge,
)
from scpn_control.control._quantum_disruption_utils import _jsonable as _jsonable

__all__ = [
    "ADVISORY_DECISION_SCHEMA_VERSION",
    "CERTIFICATE_SCHEMA_VERSION",
    "CLAIM_BOUNDARY",
    "CONTROL_FACADE_OWNER",
    "CONTROL_FEATURE_NAMES",
    "CONTROL_TO_ITER_INDEX",
    "DEPENDENCY_CONTRACT_SCHEMA_VERSION",
    "EXTRA_ITER_INDEX",
    "ITER_CENTRES",
    "ITER_FEATURE_NAMES",
    "ITER_MAXS",
    "ITER_MINS",
    "KERNEL_SCHEMA_VERSION",
    "QUANTUM_BACKEND_OWNER",
    "QUANTUM_CORE_DEPENDENCIES",
    "QUANTUM_MODULE",
    "QUANTUM_OPTIONAL_PROVIDER_DEPENDENCIES",
    "QuantumDisruptionBridgeConfig",
    "QuantumFeatureMapping",
    "REQUIRED_DOWNSTREAM_POLICY",
    "RISK_BAND_THRESHOLDS",
    "SCHEMA_VERSION",
    "annotations",
    "map_control_features_to_iter",
    "normalize_iter_features",
    "quantum_disruption_dependency_contract",
    "quantum_disruption_kernel_matrix",
    "run_quantum_disruption_bridge",
    "validate_quantum_disruption_bridge_report",
    "validate_quantum_disruption_dependency_contract",
    "validate_quantum_disruption_kernel_report",
]

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Quantum Disruption Contract

"""Quantum disruption contract for the bounded quantum bridge."""

from __future__ import annotations

from typing import Any, Mapping

from scpn_control.control._quantum_disruption_constants import (
    CERTIFICATE_SCHEMA_VERSION,
    CLAIM_BOUNDARY,
    CONTROL_FACADE_OWNER,
    CONTROL_FEATURE_NAMES,
    DEPENDENCY_CONTRACT_SCHEMA_VERSION,
    EXTRA_ITER_INDEX,
    ITER_FEATURE_NAMES,
    KERNEL_SCHEMA_VERSION,
    QUANTUM_BACKEND_OWNER,
    QUANTUM_CORE_DEPENDENCIES,
    QUANTUM_MODULE,
    QUANTUM_OPTIONAL_PROVIDER_DEPENDENCIES,
    REQUIRED_DOWNSTREAM_POLICY,
    SCHEMA_VERSION,
)
from scpn_control.control._quantum_disruption_utils import _contract_digest, _is_sha256


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


def _validate_report_dependency_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = payload.get("dependency_contract")
    if not isinstance(value, dict):
        raise ValueError("quantum disruption report dependency_contract must be an object")
    return validate_quantum_disruption_dependency_contract(value)


def _dependency_contract_schema_version(payload: Mapping[str, Any]) -> str:
    return str(_validate_report_dependency_contract(payload)["schema_version"])


def _dependency_contract_digest(payload: Mapping[str, Any]) -> str:
    return str(_validate_report_dependency_contract(payload)["contract_sha256"])


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

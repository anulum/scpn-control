# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Package Root

"""scpn-control: Neuro-symbolic Stochastic Petri Net controller."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

try:
    __version__ = version("scpn-control")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"

__all__ = [
    "__version__",
    "FusionKernel",
    "RUST_BACKEND",
    "TokamakConfig",
    "StochasticPetriNet",
    "FusionCompiler",
    "CompiledNet",
    "NeuroSymbolicController",
    "FormalPetriNetVerifier",
    "verify_formal_contracts",
    "SafetyCertificatePolicy",
    "SafetyCertificateBundlePolicy",
    "admit_safety_certificate_bundle_artifact",
    "build_safety_certificate_bundle_artifact",
    "CTLFormula",
    "LTLFormula",
    "build_safety_certificate_bundle_payload",
    "build_safety_certificate_payload",
    "generate_safety_certificate",
    "validate_safety_certificate_bundle_payload",
    "validate_safety_certificate_payload",
    "validate_safety_certificate_bundle_artifact",
    "write_safety_certificate_bundle",
    "write_safety_certificate",
    "kuramoto_sakaguchi_step",
    "order_parameter",
    "KnmSpec",
    "build_knm_paper27",
    "UPDESystem",
    "LyapunovGuard",
    "RealtimeMonitor",
    "HTTPChatProvider",
    "PhysicsDebugAssistant",
    "PhysicsDebugEvidence",
    "PhysicsDebugGap",
    "ProviderPolicy",
    "build_local_provider",
    "build_physics_debug_report",
    "run_provider_quorum",
    "validate_physics_debug_report",
    "validate_physics_debug_quorum_report",
    "write_physics_debug_quorum_report",
    "write_physics_debug_report",
]

_EXPORT_MODULES = {
    "FusionKernel": "scpn_control.core",
    "RUST_BACKEND": "scpn_control.core",
    "TokamakConfig": "scpn_control.core",
    "StochasticPetriNet": "scpn_control.scpn",
    "FusionCompiler": "scpn_control.scpn",
    "CompiledNet": "scpn_control.scpn",
    "NeuroSymbolicController": "scpn_control.scpn",
    "FormalPetriNetVerifier": "scpn_control.scpn",
    "verify_formal_contracts": "scpn_control.scpn",
    "SafetyCertificatePolicy": "scpn_control.scpn",
    "SafetyCertificateBundlePolicy": "scpn_control.scpn",
    "admit_safety_certificate_bundle_artifact": "scpn_control.scpn",
    "build_safety_certificate_bundle_artifact": "scpn_control.scpn",
    "CTLFormula": "scpn_control.scpn",
    "LTLFormula": "scpn_control.scpn",
    "build_safety_certificate_bundle_payload": "scpn_control.scpn",
    "build_safety_certificate_payload": "scpn_control.scpn",
    "generate_safety_certificate": "scpn_control.scpn",
    "validate_safety_certificate_bundle_payload": "scpn_control.scpn",
    "validate_safety_certificate_payload": "scpn_control.scpn",
    "validate_safety_certificate_bundle_artifact": "scpn_control.scpn",
    "write_safety_certificate_bundle": "scpn_control.scpn",
    "write_safety_certificate": "scpn_control.scpn",
    "kuramoto_sakaguchi_step": "scpn_control.phase",
    "order_parameter": "scpn_control.phase",
    "KnmSpec": "scpn_control.phase",
    "build_knm_paper27": "scpn_control.phase",
    "UPDESystem": "scpn_control.phase",
    "LyapunovGuard": "scpn_control.phase",
    "RealtimeMonitor": "scpn_control.phase",
    "HTTPChatProvider": "scpn_control.physics_debug",
    "PhysicsDebugAssistant": "scpn_control.physics_debug",
    "PhysicsDebugEvidence": "scpn_control.physics_debug",
    "PhysicsDebugGap": "scpn_control.physics_debug",
    "ProviderPolicy": "scpn_control.physics_debug",
    "build_local_provider": "scpn_control.physics_debug",
    "build_physics_debug_report": "scpn_control.physics_debug",
    "run_provider_quorum": "scpn_control.physics_debug",
    "validate_physics_debug_report": "scpn_control.physics_debug",
    "validate_physics_debug_quorum_report": "scpn_control.physics_debug",
    "write_physics_debug_quorum_report": "scpn_control.physics_debug",
    "write_physics_debug_report": "scpn_control.physics_debug",
}


def __getattr__(name: str) -> Any:
    """Load public exports lazily so submodule imports stay dependency-light."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module 'scpn_control' has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return package attributes including lazy public exports."""
    return sorted(set(globals()) | set(__all__))

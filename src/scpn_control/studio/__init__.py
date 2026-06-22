# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Studio vertical
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# ──────────────────────────────────────────────────────────────────────
"""CONTROL's studio vertical, built on the locked ``scpn-studio-platform`` SDK.

This package is CONTROL's federated studio surface for the SCPN STUDIO Hub. It
consumes the domain-neutral platform SDK (it never forks it): it declares CONTROL's
verbs as platform :class:`~scpn_studio_platform.verbs.Verb` records
(:mod:`scpn_control.studio.verbs`), maps CONTROL's provenance-graded result surfaces
onto platform :class:`~scpn_studio_platform.evidence.EvidenceBundle` records
(:mod:`scpn_control.studio.evidence`), and authors the schema-A
:class:`~scpn_studio_platform.manifest.CapabilityManifest`
(:mod:`scpn_control.studio.manifest`).

The platform SDK is an optional dependency (install the ``studio`` extra); importing
this package without it raises :class:`ModuleNotFoundError` at import time, which is
the intended fail-closed behaviour for an optional studio surface.
"""

from __future__ import annotations

from .evidence import (
    ControllerLatencyResult,
    EfitReconstructionResult,
    SafetyCertificateResult,
    canonical_digest,
    controller_latency_evidence,
    efit_reconstruction_evidence,
    safety_certificate_evidence,
)
from .manifest import build_manifest, declared_surface
from .verbs import (
    CONTROL_VERBS,
    STUDIO_ID,
    core_verbs,
    domain_verbs,
    evidence_schemas,
)

__all__ = [
    "CONTROL_VERBS",
    "STUDIO_ID",
    "ControllerLatencyResult",
    "EfitReconstructionResult",
    "SafetyCertificateResult",
    "build_manifest",
    "canonical_digest",
    "controller_latency_evidence",
    "core_verbs",
    "declared_surface",
    "domain_verbs",
    "efit_reconstruction_evidence",
    "evidence_schemas",
    "safety_certificate_evidence",
]

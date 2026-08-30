# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Reactor semantic admission facade

"""Public review-only admission of portable SPO reactor semantics."""

from .admission import (
    ReactorSemanticAdmissionPolicy,
    admit_reactor_semantic_handoff,
)
from .decision import (
    ADMISSION_SCHEMA,
    ADMISSION_SCHEMA_VERSION,
    MAX_ADMISSION_BYTES,
    REFUSAL_CODES,
    ReactorSemanticAdmissionDecision,
    ReactorSemanticAdmissionStatus,
    admission_decision_digest,
    admission_decision_from_bytes,
    admission_decision_to_bytes,
)

__all__ = [
    "ADMISSION_SCHEMA",
    "ADMISSION_SCHEMA_VERSION",
    "MAX_ADMISSION_BYTES",
    "REFUSAL_CODES",
    "ReactorSemanticAdmissionDecision",
    "ReactorSemanticAdmissionPolicy",
    "ReactorSemanticAdmissionStatus",
    "admission_decision_digest",
    "admission_decision_from_bytes",
    "admission_decision_to_bytes",
    "admit_reactor_semantic_handoff",
]

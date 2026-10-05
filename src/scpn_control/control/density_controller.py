# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Density-control plant and particle-source model

"""Stable public facade for density transport, control, evidence, and estimation."""

from __future__ import annotations

from scpn_control.control._density_claim_evidence import (
    DensityControlClaimEvidence,
    assert_density_control_facility_claim_admissible,
    density_control_claim_evidence,
    save_density_control_claim_evidence,
)
from scpn_control.control._density_claim_evidence import (
    _non_empty_text as _non_empty_text,
)
from scpn_control.control._density_control_runtime import ActuatorCommand, DensityController
from scpn_control.control._density_estimator import KalmanDensityEstimator
from scpn_control.control._density_fueling import FuelingOptimizer, PelletSchedule
from scpn_control.control._density_transport import ParticleTransportModel

__all__ = [
    "ActuatorCommand",
    "DensityControlClaimEvidence",
    "DensityController",
    "FuelingOptimizer",
    "KalmanDensityEstimator",
    "ParticleTransportModel",
    "PelletSchedule",
    "assert_density_control_facility_claim_admissible",
    "density_control_claim_evidence",
    "save_density_control_claim_evidence",
]

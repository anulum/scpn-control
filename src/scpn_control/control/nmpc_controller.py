# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Nonlinear Model Predictive Controller
# SCPN Control — Nonlinear Model Predictive Controller
"""Stable public NMPC controller imports."""

from __future__ import annotations

from .nmpc_hessian import NMPCHessian
from .nmpc_types import (
    AcadosOcpFactory,
    AcadosSolverFactory,
    AcadosSymbolicDynamics,
    CostHessianAudit,
    NMPCConfig,
    RTILatencyReport,
    RTIStepResult,
    _as_finite_vector,
    _as_spd_matrix,
    _percentile_ms,
)

__all__ = [
    "AcadosOcpFactory",
    "AcadosSolverFactory",
    "AcadosSymbolicDynamics",
    "CostHessianAudit",
    "NMPCConfig",
    "NonlinearMPC",
    "RTILatencyReport",
    "RTIStepResult",
    "_as_finite_vector",
    "_as_spd_matrix",
    "_percentile_ms",
]


class NonlinearMPC(NMPCHessian):
    """SQP-based NMPC with validated plant linearization contracts."""

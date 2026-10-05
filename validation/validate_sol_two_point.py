# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Public SOL algebraic diagnostic API and CLI.

"""Expose the legacy SOL diagnostic names from cohesive checkout-only owners.

The CLI and Python API check shared production formulas and sealed report
consistency. A passing result or SHA-256 seal admits no independent scientific,
facility, safety, controller or training evidence. See docs/validation.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _source_root in (ROOT, ROOT / "src"):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

from validation.sol_two_point_contracts.command import main
from validation.sol_two_point_contracts.evidence import (
    SOL_TWO_POINT_SCHEMA_VERSION,
    build_evidence,
    validate_evidence_payload,
)
from validation.sol_two_point_contracts.models import (
    DetachmentBoundary,
    ScalingCheck,
    SOLConfig,
    SOLValidationResult,
    conduction_integral_rel_error,
    connection_length_rel_error,
    default_config,
    detachment_boundary,
    eich_scaling_checks,
    flux_mapping_rel_error,
    peak_heat_flux_rel_error,
    pressure_balance_rel_error,
    validate_sol_two_point,
)

__all__ = [
    "SOL_TWO_POINT_SCHEMA_VERSION",
    "SOLConfig",
    "ScalingCheck",
    "DetachmentBoundary",
    "SOLValidationResult",
    "default_config",
    "connection_length_rel_error",
    "flux_mapping_rel_error",
    "conduction_integral_rel_error",
    "pressure_balance_rel_error",
    "eich_scaling_checks",
    "peak_heat_flux_rel_error",
    "detachment_boundary",
    "validate_sol_two_point",
    "build_evidence",
    "validate_evidence_payload",
    "main",
]

if __name__ == "__main__":
    raise SystemExit(main())

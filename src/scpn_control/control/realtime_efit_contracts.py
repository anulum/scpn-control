# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-time equilibrium reconstruction utilities.

"""Data contracts and physical constants for realtime EFIT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from scpn_control._typing import AnyFloatArray

MU0 = 4.0e-7 * np.pi


@dataclass
class MagneticDiagnostics:
    """Layout of magnetic sensors."""

    flux_loops: list[tuple[float, float]]  # (R, Z) positions
    b_probes: list[tuple[float, float, str]]  # (R, Z, direction 'R' or 'Z')
    rogowski_radius: float


@dataclass
class ShapeParams:
    """Reconstructed macroscopic parameters."""

    R0: float
    a: float
    kappa: float
    delta_upper: float
    delta_lower: float
    q95: float
    beta_pol: float
    li: float
    Ip_reconstructed: float


@dataclass
class ReconstructionResult:
    """Equilibrium reconstruction output from real-time EFIT.

    Attributes
    ----------
    psi
        Reconstructed poloidal-flux map.
    p_prime_coeffs
        Fitted ``p'(ψ)`` basis coefficients.
    ff_prime_coeffs
        Fitted ``FF'(ψ)`` basis coefficients.
    shape
        Derived plasma-shape parameters.
    chi_squared
        Goodness-of-fit chi-squared against the diagnostics.
    n_iterations
        Number of Picard iterations performed.
    wall_time_ms
        Reconstruction wall time in milliseconds.
    coil_currents
        Fitted external-coil currents [A] for a free-boundary reconstruction;
        ``None`` for a fixed-boundary fit (no coils supplied).
    iteration_status
        Whether the Picard tolerance was reached, its cap was exhausted, or
        the geometric single-solve mode was used. Directly constructed results
        remain ``unreported``.
    final_relative_change
        Last Picard relative-flux change, or ``None`` for geometric/unreported
        results.
    """

    psi: AnyFloatArray
    p_prime_coeffs: AnyFloatArray
    ff_prime_coeffs: AnyFloatArray
    shape: ShapeParams
    chi_squared: float
    n_iterations: int
    wall_time_ms: float
    # Keyword-only with a default so subclasses (e.g. KineticReconstructionResult)
    # can still add required positional fields without a dataclass ordering clash.
    coil_currents: AnyFloatArray | None = field(default=None, kw_only=True)
    iteration_status: Literal["unreported", "geometric_solve", "picard_converged", "picard_limit"] = field(
        default="unreported", kw_only=True
    )
    final_relative_change: float | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class EFITLiteClaimEvidence:
    """Serialisable admission evidence for EFIT-lite reconstruction claims."""

    schema_version: int
    source: str
    source_id: str
    diagnostic_source: str
    model_id: str
    grid_shape: tuple[int, int]
    n_flux_loops: int
    n_b_probes: int
    rogowski_radius_m: float
    chi_squared: float
    n_iterations: int
    iteration_status: Literal["unreported", "geometric_solve", "picard_converged", "picard_limit"]
    final_relative_change: float | None
    wall_time_ms: float
    ip_reconstructed_A: float
    q95: float
    beta_pol: float
    li: float
    psi_relative_error: float | None
    ip_relative_error: float | None
    q95_abs_error: float | None
    beta_pol_abs_error: float | None
    li_abs_error: float | None
    psi_relative_tolerance: float
    ip_relative_tolerance: float
    q95_abs_tolerance: float
    beta_pol_abs_tolerance: float
    li_abs_tolerance: float
    facility_claim_allowed: bool
    claim_status: str

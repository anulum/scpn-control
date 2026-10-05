# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Shared rigid vertical configuration and result values.

"""Immutable values shared by the rigid-model validator and its report decoder.

Geometry uses metres, current MA, field tesla and inertia kg. Configuration
construction rejects nonfinite/nonpositive values and a >= R0. Result, scaling
and wall containers are frozen but do not validate their verdicts on construction;
the evidence decoder checks those domains and their mutual consistency. These
values do not establish a measured reference or facility-control admission.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_MU_0 = 4.0e-7 * math.pi


@dataclass(frozen=True)
class VerticalConfig:
    """Frozen geometry, current and inertia for the rigid vertical reference.

    Attributes
    ----------
    r0, a : float
        Major and minor radii in metres, with 0 < a < r0.
    kappa : float
        Positive dimensionless elongation.
    ip_ma, b0, m_eff_kg : float
        Positive plasma current in MA, field in tesla and effective inertia in kg.

    Raises
    ------
    ValueError
        A value is boolean, nonnumeric, nonfinite, nonpositive or a >= r0.
    """

    r0: float
    a: float
    kappa: float
    ip_ma: float
    b0: float
    m_eff_kg: float

    def __post_init__(self) -> None:
        _positive_float("r0", self.r0)
        _positive_float("a", self.a)
        _positive_float("kappa", self.kappa)
        _positive_float("ip_ma", self.ip_ma)
        _positive_float("b0", self.b0)
        _positive_float("m_eff_kg", self.m_eff_kg)
        if self.a >= self.r0:
            raise ValueError("a must be smaller than r0 for tokamak ordering")

    def curvature_spring(self, n_index: float) -> float:
        """Return ``K = n mu_0 Ip^2 / (4 pi R_0)`` in N/m without mutation.

        ``n_index`` is the signed dimensionless decay index. This arithmetic
        method does not validate its argument or floating-point representability;
        the public validation operations enforce their own index domains.
        """
        ip_amps = self.ip_ma * 1.0e6
        return n_index * _MU_0 * ip_amps**2 / (4.0 * math.pi * self.r0)


@dataclass(frozen=True)
class ScalingCheck:
    """Frozen scaling observation with a label and three dimensionless numbers.

    ``measured_ratio`` and ``expected_ratio`` compare growth rates; ``rel_error``
    declares their relative difference. Construction is unchecked. The report
    decoder verifies the known law, ratio domains and computed relative error.
    """

    name: str
    measured_ratio: float
    expected_ratio: float
    rel_error: float


@dataclass(frozen=True)
class WallStabilisation:
    """Frozen passive-wall comparison with growth rates in s^-1.

    ``wall_slows_growth`` and ``with_wall_finite`` are declared boolean checks.
    Construction is unchecked; serialization validates finite rates and whether
    the declarations agree with the comparison. This is a bounded rigid model.
    """

    no_wall_growth_rate: float
    with_wall_growth_rate: float
    wall_slows_growth: bool
    with_wall_finite: bool


@dataclass(frozen=True)
class RzipValidationResult:
    """Frozen result of the rigid-model validation, without constructor admission.

    ``config`` carries geometry/current/inertia. Index tuples are dimensionless;
    maximum relative errors, scaling observations and ``exact_tol`` are also
    dimensionless. ``marginal_growth_rate``, ``marginal_tol`` and wall growth
    rates use s^-1. Individual ``*_passed`` values and ``passed`` are declarations.
    The report decoder validates their agreement with metrics and thresholds.
    No array or mutable model state is retained, and no facility claim is granted.
    """

    config: VerticalConfig
    unstable_indices: tuple[float, ...]
    stable_indices: tuple[float, ...]
    max_growth_rel_error: float
    max_frequency_rel_error: float
    max_growth_time_rel_error: float
    marginal_growth_rate: float
    scaling: tuple[ScalingCheck, ...]
    max_scaling_rel_error: float
    wall: WallStabilisation
    exact_tol: float
    marginal_tol: float
    growth_passed: bool
    frequency_passed: bool
    growth_time_passed: bool
    marginal_passed: bool
    scaling_passed: bool
    wall_passed: bool
    passed: bool


def _finite_float(name: str, value: object) -> float:
    """Require a finite scalar number without boolean or text coercion."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(name: str, value: object) -> float:
    """Require a finite strictly positive scalar."""
    result = _finite_float(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _negative_float(name: str, value: object) -> float:
    """Require a finite strictly negative scalar."""
    result = _finite_float(name, value)
    if result >= 0.0:
        raise ValueError(f"{name} must be negative")
    return result

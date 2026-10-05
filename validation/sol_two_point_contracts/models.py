# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — SOL algebraic diagnostic inputs and numerical checks.

"""Check production SOL formulas algebraically without physical evidence admission.

These checks use the same Eich implementation/constants as the production model.
Agreement establishes bounded algebraic consistency, not independent numerical,
experimental, facility, safety or training evidence. No state is trained or stored.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from scpn_control.core.sol_model import (
    DETACHMENT_ONSET_EV,
    GAMMA_SHEATH,
    KAPPA_0_ELECTRON,
    TwoPointSOL,
    detachment_threshold,
    eich_heat_flux_width,
    peak_target_heat_flux,
)

_E_CHARGE = 1.602e-19
_M_ION = 2.0 * 1.6726e-27


@dataclass(frozen=True)
class SOLConfig:
    """Hold positive finite geometry and field values for a local SOL diagnostic.

    Parameters
    ----------
    r0, a : float
        Major/minor radius in metres, with ``0 < a < r0``.
    q95 : float
        Dimensionless edge safety factor.
    b_pol : float
        Poloidal field in tesla.

    Raises
    ------
    ValueError
        A value is nonnumeric, boolean, nonfinite, nonpositive or violates
        the radius ordering. Numeric strings are refused; values are immutable.

    Notes
    -----
    This geometry declaration carries no facility provenance or measured data.
    """

    r0: float
    a: float
    q95: float
    b_pol: float

    def __post_init__(self) -> None:
        _positive_float("r0", self.r0)
        _positive_float("a", self.a)
        _positive_float("q95", self.q95)
        _positive_float("b_pol", self.b_pol)
        if self.a >= self.r0:
            raise ValueError("a must be smaller than r0 for tokamak ordering")
        _positive_float("epsilon", self.epsilon)

    @property
    def epsilon(self) -> float:
        """Return dimensionless ``a/r0`` in the open interval (0, 1)."""
        return self.a / self.r0

    def model(self) -> TwoPointSOL:
        """Construct a fresh production model; retain no mutable solver state."""
        return TwoPointSOL(self.r0, self.a, self.q95, self.b_pol)


def default_config() -> SOLConfig:
    """Return illustrative R0=1.7 m, a=0.5 m, q95=3.5, B_pol=0.4 T inputs."""
    return SOLConfig(r0=1.7, a=0.5, q95=3.5, b_pol=0.4)


def connection_length_rel_error(config: SOLConfig) -> float:
    """Return dimensionless connection-length error against ``pi*q95*r0``.

    ``config`` supplies metres/dimensionless geometry. A fresh production
    model is compared with the same algebraic formula. ValueError refuses
    nonfinite or nonpositive derived lengths; no external reference is used.
    """
    analytic = _positive_float("connection length", math.pi * config.q95 * config.r0)
    return abs(config.model().L_par - analytic) / analytic


def _reconstructed_q_par(config: SOLConfig, p_sol_mw: float) -> float:
    """Analytic parallel heat flux ``P_SOL/(4 pi R0 lambda_q)(q95/eps)`` [W/m^2]."""
    lambda_q_m = eich_heat_flux_width(p_sol_mw, config.r0, config.b_pol, config.epsilon) * 1e-3
    return _positive_float(
        "parallel heat flux",
        (p_sol_mw * 1e6) / (4.0 * math.pi * config.r0 * lambda_q_m) * (config.q95 / config.epsilon),
    )


def flux_mapping_rel_error(config: SOLConfig, p_sol_mw: float, n_u_19: float) -> float:
    """Return dimensionless parallel-flux formula error for MW and 1e19 m^-3 inputs.

    Config uses metres/tesla/dimensionless values. Positive finite numeric
    power/density are required; booleans and strings raise ValueError. The
    production solve and reconstruction share the Eich width implementation.
    Arithmetic overflow/domain errors propagate for extreme derived quantities.
    """
    p_sol_mw = _positive_float("P_SOL_MW", p_sol_mw)
    n_u_19 = _positive_float("n_u_19", n_u_19)
    solution = config.model().solve(p_sol_mw, n_u_19)
    analytic = _reconstructed_q_par(config, p_sol_mw)
    return abs(solution.q_parallel_MW_m2 * 1e6 - analytic) / analytic


def conduction_integral_rel_error(config: SOLConfig, p_sol_mw: float, n_u_19: float) -> float:
    """Relative error of the Spitzer-Härm upstream conduction integral.

    Power is in MW; upstream density is in 1e19 m^-3. Both must be positive
    finite numbers. Invalid declared inputs raise ValueError, and extreme
    derived arithmetic errors propagate. The same production constants and
    Eich width are used, so agreement is not independent physics validation.

    The solved upstream temperature must satisfy
    ``q_par = kappa_0 T_u^{7/2} / (3.5 L_par)``.
    """
    p_sol_mw = _positive_float("P_SOL_MW", p_sol_mw)
    n_u_19 = _positive_float("n_u_19", n_u_19)
    model = config.model()
    solution = model.solve(p_sol_mw, n_u_19)
    q_par_from_tu = KAPPA_0_ELECTRON * solution.T_upstream_eV**3.5 / (3.5 * model.L_par)
    analytic = _reconstructed_q_par(config, p_sol_mw)
    return float(abs(q_par_from_tu - analytic) / analytic)


def pressure_balance_rel_error(config: SOLConfig, p_sol_mw: float, n_u_19: float) -> float:
    """Return dimensionless ``n_u*T_u = 2*n_t*T_t`` pressure-balance error.

    Power in MW and density in 1e19 m^-3 must be positive finite numbers;
    booleans/strings raise ValueError. A fresh production solve supplies eV
    temperatures and target density. Derived arithmetic/domain errors propagate.
    """
    p_sol_mw = _positive_float("P_SOL_MW", p_sol_mw)
    n_u_19 = _positive_float("n_u_19", n_u_19)
    solution = config.model().solve(p_sol_mw, n_u_19)
    upstream_pressure = n_u_19 * 1e19 * solution.T_upstream_eV
    target_pressure = 2.0 * solution.n_target_19 * 1e19 * solution.T_target_eV
    return abs(upstream_pressure - target_pressure) / upstream_pressure


@dataclass(frozen=True)
class ScalingCheck:
    """Store an exponent label and dimensionless measured/expected/error ratios.

    Values describe local algebraic observations. The evidence builder checks
    that ratios are finite/positive and the nonnegative error is consistent.
    """

    name: str
    measured_ratio: float
    expected_ratio: float
    rel_error: float


def eich_scaling_checks(config: SOLConfig, p_sol_mw: float) -> tuple[ScalingCheck, ...]:
    """Check four dimensionless Eich width ratios at positive finite MW power.

    Power, radius and field double. Epsilon doubles below 0.5, otherwise
    halves so every probe remains inside the production ``0 < epsilon < 1``
    domain. Names/order retain power, radius, field and epsilon exponents.
    Config values are metres/tesla/dimensionless. Invalid or overflowing
    input/probe values raise ValueError or the production arithmetic error.
    """
    p_sol_mw = _positive_float("P_SOL_MW", p_sol_mw)
    epsilon_factor = 2.0 if config.epsilon < 0.5 else 0.5
    base = eich_heat_flux_width(p_sol_mw, config.r0, config.b_pol, config.epsilon)
    specs = (
        ("power_-0.02", eich_heat_flux_width(2.0 * p_sol_mw, config.r0, config.b_pol, config.epsilon), 2.0**-0.02),
        ("major_radius_0.04", eich_heat_flux_width(p_sol_mw, 2.0 * config.r0, config.b_pol, config.epsilon), 2.0**0.04),
        ("b_pol_-0.92", eich_heat_flux_width(p_sol_mw, config.r0, 2.0 * config.b_pol, config.epsilon), 2.0**-0.92),
        (
            "epsilon_0.42",
            eich_heat_flux_width(p_sol_mw, config.r0, config.b_pol, epsilon_factor * config.epsilon),
            epsilon_factor**0.42,
        ),
    )
    return tuple(
        ScalingCheck(
            name=name,
            measured_ratio=value / base,
            expected_ratio=expected,
            rel_error=abs(value / base - expected) / expected,
        )
        for name, value, expected in specs
    )


def peak_heat_flux_rel_error(config: SOLConfig, p_sol_mw: float) -> float:
    """Return dimensionless peak-flux formula error at positive finite MW power.

    Use fresh production Eich width, fixed expansion factor 5 and incidence
    angle 3 degrees. Geometry is metres/tesla/dimensionless. Invalid input
    raises ValueError; extreme derived arithmetic errors propagate. This
    shared-formula check supplies no independent heat-load measurement.
    """
    p_sol_mw = _positive_float("P_SOL_MW", p_sol_mw)
    lambda_q_m = eich_heat_flux_width(p_sol_mw, config.r0, config.b_pol, config.epsilon) * 1e-3
    f_expansion, alpha_deg = 5.0, 3.0
    measured = peak_target_heat_flux(p_sol_mw, config.r0, lambda_q_m, f_expansion=f_expansion, alpha_deg=alpha_deg)
    analytic = p_sol_mw / (4.0 * math.pi * config.r0 * lambda_q_m * f_expansion) * math.sin(math.radians(alpha_deg))
    return abs(measured - analytic) / analytic


@dataclass(frozen=True)
class DetachmentBoundary:
    """Store positive critical density in 1e19 m^-3 and two local boolean probes.

    Probes use 0.99 and 1.01 times the critical density. No shot, uncertainty,
    clock, measured detachment boundary or facility admission is represented.
    """

    critical_density_19: float
    detached_below_critical: bool
    detached_above_critical: bool


def detachment_boundary(q_par_mw_m2: float, l_par: float) -> DetachmentBoundary:
    """Locate the analytic critical density where the sheath target reaches 5 eV.

    ``T_t = (2 q_par/(gamma n_u T_u e))^2 m_i/e`` decreases with ``n_u``; the onset
    ``T_t = 5 eV`` fixes the critical density, below which the target is attached
    and above which it is detached. Inputs are parallel heat flux in MW/m^2
    and connection length in metres, both positive finite numbers; booleans
    and strings raise ValueError. Nonfinite derived critical density refuses
    with ValueError; extreme arithmetic errors propagate. Constants are
    the same deuteron/sheath values as production, not independent evidence.
    """
    q_par_mw_m2 = _positive_float("q_par_mw_m2", q_par_mw_m2)
    l_par = _positive_float("l_par", l_par)
    q_par = q_par_mw_m2 * 1e6
    t_u = (3.5 * l_par * q_par / KAPPA_0_ELECTRON) ** (2.0 / 7.0)
    # Solve T_t(n_u) = DETACHMENT_ONSET_EV for n_u.
    n_crit = (2.0 * q_par) / (GAMMA_SHEATH * t_u * _E_CHARGE) / math.sqrt(DETACHMENT_ONSET_EV * _E_CHARGE / _M_ION)
    n_crit_19 = _positive_float("critical density", n_crit / 1e19)
    return DetachmentBoundary(
        critical_density_19=n_crit_19,
        detached_below_critical=detachment_threshold(0.99 * n_crit_19, q_par_mw_m2, l_par),
        detached_above_critical=detachment_threshold(1.01 * n_crit_19, q_par_mw_m2, l_par),
    )


@dataclass(frozen=True)
class SOLValidationResult:
    """Store immutable local inputs, dimensionless errors and seven check flags.

    Operating points are ordered ``(power_MW, upstream_density_1e19_m^-3)``
    pairs. Geometry/scaling/detachment use their dataclass units. Six error
    gates use strict ``error < exact_tol``; detachment requires attached below
    and detached above the analytic boundary. ``passed`` is their conjunction.
    The timestamp-free result is deterministic and has no physical admission.
    """

    config: SOLConfig
    operating_points: tuple[tuple[float, float], ...]
    connection_length_rel_error: float
    max_flux_mapping_rel_error: float
    max_conduction_rel_error: float
    max_pressure_balance_rel_error: float
    scaling: tuple[ScalingCheck, ...]
    max_scaling_rel_error: float
    peak_heat_flux_rel_error: float
    detachment: DetachmentBoundary
    exact_tol: float
    connection_passed: bool
    flux_mapping_passed: bool
    conduction_passed: bool
    pressure_passed: bool
    scaling_passed: bool
    peak_flux_passed: bool
    detachment_passed: bool
    passed: bool


def validate_sol_two_point(
    *,
    config: SOLConfig | None = None,
    operating_points: Sequence[tuple[float, float]] = ((10.0, 3.0), (20.0, 5.0), (5.0, 1.5)),
    detachment_q_par_mw_m2: float = 100.0,
    detachment_l_par: float = 20.0,
    exact_tol: float = 1e-9,
) -> SOLValidationResult:
    """Check the production two-point SOL model against its own algebraic forms.

    The connection length, parallel-flux mapping, Spitzer-Härm upstream conduction
    integral, pressure balance, Eich scaling exponents, peak heat flux, and the
    detachment density boundary must all hold to ``exact_tol``.

    Parameters
    ----------
    config
        Immutable metre/tesla/dimensionless geometry; None uses illustrative
        defaults, with no facility provenance.
    operating_points
        Nonempty sequence of length-two power-MW/density-1e19-m^-3 pairs.
        All values must be positive finite numbers, excluding booleans/strings.
    detachment_q_par_mw_m2, detachment_l_par
        Positive finite parallel heat flux in MW/m^2 and length in metres.
    exact_tol
        Positive finite dimensionless strict relative-error threshold.

    Returns
    -------
    SOLValidationResult
        Deterministic timestamp-free local diagnostic; finite nonnegative
        relative errors and seven explicit flags. There is no state mutation.

    Raises
    ------
    ValueError
        Invalid declared inputs, pair shapes or nonfinite derived diagnostics.
    OverflowError, ZeroDivisionError
        Extreme finite values exceed the production arithmetic domain.

    Notes
    -----
    This shares production constants/Eich width, and establishes algebraic
    consistency only. External numerical/physical/facility admission is absent.

    Examples
    --------
    >>> result = validate_sol_two_point(operating_points=((10.0, 3.0),))
    >>> result.passed, len(result.scaling)
    (True, 4)
    >>> validate_sol_two_point(exact_tol=1e-30).passed
    False
    """
    exact_tol = _positive_float("exact_tol", exact_tol)
    if config is None:
        config = default_config()
    if not isinstance(config, SOLConfig):
        raise ValueError("config must be SOLConfig or None")
    if isinstance(operating_points, (str, bytes)) or not isinstance(operating_points, Sequence):
        raise ValueError("operating_points must be a nonempty sequence of numeric pairs")
    normalized = []
    for point in operating_points:
        if isinstance(point, (str, bytes)) or not isinstance(point, Sequence) or len(point) != 2:
            raise ValueError("each operating point must contain exactly two numbers")
        normalized.append((_positive_float("P_SOL_MW", point[0]), _positive_float("n_u_19", point[1])))
    points = tuple(normalized)
    if not points:
        raise ValueError("at least one operating point is required")

    conn_err = connection_length_rel_error(config)
    max_flux = max(flux_mapping_rel_error(config, p, n) for p, n in points)
    max_cond = max(conduction_integral_rel_error(config, p, n) for p, n in points)
    max_press = max(pressure_balance_rel_error(config, p, n) for p, n in points)
    scaling = eich_scaling_checks(config, points[0][0])
    max_scaling = max(check.rel_error for check in scaling)
    peak_err = peak_heat_flux_rel_error(config, points[0][0])
    detach = detachment_boundary(detachment_q_par_mw_m2, detachment_l_par)

    for name, value in (
        ("connection error", conn_err),
        ("flux error", max_flux),
        ("conduction error", max_cond),
        ("pressure error", max_press),
        ("scaling error", max_scaling),
        ("peak error", peak_err),
    ):
        _finite_float(name, value)

    connection_passed = conn_err < exact_tol
    flux_mapping_passed = max_flux < exact_tol
    conduction_passed = max_cond < exact_tol
    pressure_passed = max_press < exact_tol
    scaling_passed = max_scaling < exact_tol
    peak_flux_passed = peak_err < exact_tol
    detachment_passed = (not detach.detached_below_critical) and detach.detached_above_critical

    passed = (
        connection_passed
        and flux_mapping_passed
        and conduction_passed
        and pressure_passed
        and scaling_passed
        and peak_flux_passed
        and detachment_passed
    )
    return SOLValidationResult(
        config=config,
        operating_points=points,
        connection_length_rel_error=conn_err,
        max_flux_mapping_rel_error=max_flux,
        max_conduction_rel_error=max_cond,
        max_pressure_balance_rel_error=max_press,
        scaling=scaling,
        max_scaling_rel_error=max_scaling,
        peak_heat_flux_rel_error=peak_err,
        detachment=detach,
        exact_tol=exact_tol,
        connection_passed=connection_passed,
        flux_mapping_passed=flux_mapping_passed,
        conduction_passed=conduction_passed,
        pressure_passed=pressure_passed,
        scaling_passed=scaling_passed,
        peak_flux_passed=peak_flux_passed,
        detachment_passed=detachment_passed,
        passed=passed,
    )


def _finite_float(name: str, value: object) -> float:
    """Refuse boolean/coercible/nonfinite/overflowing values as ValueError."""
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
    """Return a strictly positive finite Python number without string coercion."""
    result = _finite_float(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result

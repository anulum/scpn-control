#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Two-point scrape-off-layer model analytic validation
"""Validate the two-point scrape-off-layer model against exact closed forms.

The scrape-off-layer (SOL) model (``src/scpn_control/core/sol_model.py``) gives
the upstream and target conditions from the Stangeby two-point model, the Eich
heat-flux width regression, and the divertor peak-heat-flux and detachment
relations. Every relation is an exact algebraic closed form, so the model can be
validated without any measured shot or external code — the validation is fully
self-contained.

Exact references checked against the production methods:

1. **Connection length**: ``L_par = pi q95 R0``.
2. **Parallel heat-flux mapping**: ``q_par = P_SOL / (4 pi R0 lambda_q) (q95/eps)``.
3. **Upstream conduction integral** (Spitzer-Härm): the solved upstream
   temperature satisfies ``q_par = kappa_0 T_u^{7/2} / (7/2 L_par)``.
4. **Pressure balance**: ``n_u T_u = 2 n_t T_t``.
5. **Eich regression exponents**: ``lambda_q`` scales as ``P^{-0.02}``,
   ``R^{0.04}``, ``B_pol^{-0.92}``, and ``eps^{0.42}``.
6. **Peak target heat flux**: ``q_target = P_SOL/(4 pi R0 lambda_q f_exp) sin alpha``.
7. **Detachment threshold**: the sheath target temperature
   ``T_t = (2 q_par/(gamma n_u T_u e))^2 m_i/e`` crosses the 5 eV onset at the
   analytic critical density.

References:
  Stangeby P. C. (2000) *The Plasma Boundary of Magnetic Fusion Devices*, IoP.
  Eich T. et al. (2013) *Nucl. Fusion* 53, 093031.
  Lipschultz B. et al. (1999) *Plasma Phys. Control. Fusion* 41, A585.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from scpn_control.core.sol_model import (
    DETACHMENT_ONSET_EV,
    GAMMA_SHEATH,
    KAPPA_0_ELECTRON,
    TwoPointSOL,
    detachment_threshold,
    eich_heat_flux_width,
    peak_target_heat_flux,
)

SOL_TWO_POINT_SCHEMA_VERSION = "scpn-control.sol-two-point-validation.v1"

# Charge and deuteron mass exactly as used in sol_model.py.
_E_CHARGE = 1.602e-19
_M_ION = 2.0 * 1.6726e-27


@dataclass(frozen=True)
class SOLConfig:
    """Geometry and edge-safety-factor inputs for the two-point SOL model."""

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

    @property
    def epsilon(self) -> float:
        return self.a / self.r0

    def model(self) -> TwoPointSOL:
        return TwoPointSOL(self.r0, self.a, self.q95, self.b_pol)


def default_config() -> SOLConfig:
    """An ITER-like SOL geometry with q95 = 3.5 and a 0.4 T poloidal field."""
    return SOLConfig(r0=1.7, a=0.5, q95=3.5, b_pol=0.4)


def connection_length_rel_error(config: SOLConfig) -> float:
    """Relative error of ``L_par`` against ``pi q95 R0``."""
    analytic = math.pi * config.q95 * config.r0
    return abs(config.model().L_par - analytic) / analytic


def _reconstructed_q_par(config: SOLConfig, p_sol_mw: float) -> float:
    """Analytic parallel heat flux ``P_SOL/(4 pi R0 lambda_q)(q95/eps)`` [W/m^2]."""
    lambda_q_m = eich_heat_flux_width(p_sol_mw, config.r0, config.b_pol, config.epsilon) * 1e-3
    return (p_sol_mw * 1e6) / (4.0 * math.pi * config.r0 * lambda_q_m) * (config.q95 / config.epsilon)


def flux_mapping_rel_error(config: SOLConfig, p_sol_mw: float, n_u_19: float) -> float:
    """Relative error of the reported parallel heat flux against its closed form."""
    solution = config.model().solve(p_sol_mw, n_u_19)
    analytic = _reconstructed_q_par(config, p_sol_mw)
    return abs(solution.q_parallel_MW_m2 * 1e6 - analytic) / analytic


def conduction_integral_rel_error(config: SOLConfig, p_sol_mw: float, n_u_19: float) -> float:
    """Relative error of the Spitzer-Härm upstream conduction integral.

    The solved upstream temperature must satisfy
    ``q_par = kappa_0 T_u^{7/2} / (3.5 L_par)``.
    """
    model = config.model()
    solution = model.solve(p_sol_mw, n_u_19)
    q_par_from_tu = KAPPA_0_ELECTRON * solution.T_upstream_eV**3.5 / (3.5 * model.L_par)
    analytic = _reconstructed_q_par(config, p_sol_mw)
    return float(abs(q_par_from_tu - analytic) / analytic)


def pressure_balance_rel_error(config: SOLConfig, p_sol_mw: float, n_u_19: float) -> float:
    """Relative error of the two-point pressure balance ``n_u T_u = 2 n_t T_t``."""
    solution = config.model().solve(p_sol_mw, n_u_19)
    upstream_pressure = n_u_19 * 1e19 * solution.T_upstream_eV
    target_pressure = 2.0 * solution.n_target_19 * 1e19 * solution.T_target_eV
    return abs(upstream_pressure - target_pressure) / upstream_pressure


@dataclass(frozen=True)
class ScalingCheck:
    """One Eich-regression scaling-exponent observation."""

    name: str
    measured_ratio: float
    expected_ratio: float
    rel_error: float


def eich_scaling_checks(config: SOLConfig, p_sol_mw: float) -> tuple[ScalingCheck, ...]:
    """Verify the Eich ``lambda_q`` regression exponents via factor-of-two ratios."""
    base = eich_heat_flux_width(p_sol_mw, config.r0, config.b_pol, config.epsilon)
    specs = (
        ("power_-0.02", eich_heat_flux_width(2.0 * p_sol_mw, config.r0, config.b_pol, config.epsilon), 2.0**-0.02),
        ("major_radius_0.04", eich_heat_flux_width(p_sol_mw, 2.0 * config.r0, config.b_pol, config.epsilon), 2.0**0.04),
        ("b_pol_-0.92", eich_heat_flux_width(p_sol_mw, config.r0, 2.0 * config.b_pol, config.epsilon), 2.0**-0.92),
        ("epsilon_0.42", eich_heat_flux_width(p_sol_mw, config.r0, config.b_pol, 2.0 * config.epsilon), 2.0**0.42),
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
    """Relative error of the divertor peak heat flux against its closed form."""
    lambda_q_m = eich_heat_flux_width(p_sol_mw, config.r0, config.b_pol, config.epsilon) * 1e-3
    f_expansion, alpha_deg = 5.0, 3.0
    measured = peak_target_heat_flux(p_sol_mw, config.r0, lambda_q_m, f_expansion=f_expansion, alpha_deg=alpha_deg)
    analytic = p_sol_mw / (4.0 * math.pi * config.r0 * lambda_q_m * f_expansion) * math.sin(math.radians(alpha_deg))
    return abs(measured - analytic) / analytic


@dataclass(frozen=True)
class DetachmentBoundary:
    """Detachment-onset boundary against the analytic critical density."""

    critical_density_19: float
    detached_below_critical: bool
    detached_above_critical: bool


def detachment_boundary(q_par_mw_m2: float, l_par: float) -> DetachmentBoundary:
    """Locate the analytic critical density where the sheath target reaches 5 eV.

    ``T_t = (2 q_par/(gamma n_u T_u e))^2 m_i/e`` decreases with ``n_u``; the onset
    ``T_t = 5 eV`` fixes the critical density, below which the target is attached
    and above which it is detached.
    """
    _positive_float("q_par_mw_m2", q_par_mw_m2)
    _positive_float("l_par", l_par)
    q_par = q_par_mw_m2 * 1e6
    t_u = (3.5 * l_par * q_par / KAPPA_0_ELECTRON) ** (2.0 / 7.0)
    # Solve T_t(n_u) = DETACHMENT_ONSET_EV for n_u.
    n_crit = (2.0 * q_par) / (GAMMA_SHEATH * t_u * _E_CHARGE) / math.sqrt(DETACHMENT_ONSET_EV * _E_CHARGE / _M_ION)
    n_crit_19 = n_crit / 1e19
    return DetachmentBoundary(
        critical_density_19=n_crit_19,
        detached_below_critical=detachment_threshold(0.99 * n_crit_19, q_par_mw_m2, l_par),
        detached_above_critical=detachment_threshold(1.01 * n_crit_19, q_par_mw_m2, l_par),
    )


@dataclass(frozen=True)
class SOLValidationResult:
    """Outcome of the two-point SOL model validation."""

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
    """Validate the production two-point SOL model against its exact closed forms.

    The connection length, parallel-flux mapping, Spitzer-Härm upstream conduction
    integral, pressure balance, Eich scaling exponents, peak heat flux, and the
    detachment density boundary must all hold to ``exact_tol``.
    """
    config = config or default_config()
    points = tuple((_positive_float("P_SOL_MW", p), _positive_float("n_u_19", n)) for p, n in operating_points)
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


def build_evidence(result: SOLValidationResult, *, target_id: str) -> dict[str, Any]:
    """Build a tamper-evident, schema-versioned validation evidence payload."""
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    payload: dict[str, Any] = {
        "schema_version": SOL_TWO_POINT_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "target_id": target_id,
        "config": {
            "r0": result.config.r0,
            "a": result.config.a,
            "q95": result.config.q95,
            "b_pol": result.config.b_pol,
        },
        "operating_points": [list(point) for point in result.operating_points],
        "exact_tol": result.exact_tol,
        "connection_length_rel_error": result.connection_length_rel_error,
        "max_flux_mapping_rel_error": result.max_flux_mapping_rel_error,
        "max_conduction_rel_error": result.max_conduction_rel_error,
        "max_pressure_balance_rel_error": result.max_pressure_balance_rel_error,
        "scaling": [
            {
                "name": check.name,
                "measured_ratio": check.measured_ratio,
                "expected_ratio": check.expected_ratio,
                "rel_error": check.rel_error,
            }
            for check in result.scaling
        ],
        "max_scaling_rel_error": result.max_scaling_rel_error,
        "peak_heat_flux_rel_error": result.peak_heat_flux_rel_error,
        "detachment": {
            "critical_density_19": result.detachment.critical_density_19,
            "detached_below_critical": result.detachment.detached_below_critical,
            "detached_above_critical": result.detachment.detached_above_critical,
        },
        "connection_passed": result.connection_passed,
        "flux_mapping_passed": result.flux_mapping_passed,
        "conduction_passed": result.conduction_passed,
        "pressure_passed": result.pressure_passed,
        "scaling_passed": result.scaling_passed,
        "peak_flux_passed": result.peak_flux_passed,
        "detachment_passed": result.detachment_passed,
        "passed": result.passed,
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Return ``True`` when a payload is well-formed, sealed, and passing."""
    if payload.get("schema_version") != SOL_TWO_POINT_SCHEMA_VERSION:
        raise ValueError("unsupported sol two-point evidence schema_version")
    declared = payload.get("payload_sha256")
    if not _is_sha256(declared):
        raise ValueError("payload_sha256 must be a SHA-256 hex digest")
    if declared != _payload_sha256(payload):
        raise ValueError("payload_sha256 does not match payload")
    return bool(payload.get("passed"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned["payload_sha256"] = ""
    return hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(name: str, value: object) -> float:
    result = _finite_float(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _write_report(evidence: Mapping[str, Any], json_path: Path) -> None:
    """Persist the sealed JSON evidence and a human-readable Markdown summary."""
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = json_path.with_suffix(".md")
    config = evidence["config"]
    detach = evidence["detachment"]
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "",
        "# Two-Point Scrape-Off-Layer Model Validation",
        "",
        f"- Schema: `{evidence['schema_version']}`",
        f"- Generated (UTC): {evidence['generated_utc']}",
        f"- Target: `{evidence['target_id']}`",
        f"- Geometry: R0={config['r0']} m, a={config['a']} m, q95={config['q95']}, B_pol={config['b_pol']} T",
        f"- Status: **{'pass' if evidence['passed'] else 'fail'}**",
        "",
        f"## Exact closed-form references (relative error, gate < {evidence['exact_tol']:.1e})",
        "",
        "| reference | value |",
        "| --- | --- |",
        f"| connection length L_par = pi q95 R0 | {evidence['connection_length_rel_error']:.3e} |",
        f"| parallel-flux mapping | {evidence['max_flux_mapping_rel_error']:.3e} |",
        f"| Spitzer-Härm upstream conduction integral | {evidence['max_conduction_rel_error']:.3e} |",
        f"| pressure balance n_u T_u = 2 n_t T_t | {evidence['max_pressure_balance_rel_error']:.3e} |",
        f"| Eich regression exponents | {evidence['max_scaling_rel_error']:.3e} |",
        f"| peak target heat flux | {evidence['peak_heat_flux_rel_error']:.3e} |",
        "",
        "## Eich regression scaling exponents",
        "",
        "| exponent | measured ratio | expected | rel error |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {check['name']} | {check['measured_ratio']:.6f} | {check['expected_ratio']:.6f} | {check['rel_error']:.3e} |"
        for check in evidence["scaling"]
    ]
    lines += [
        "",
        "## Detachment onset boundary",
        "",
        f"- Analytic critical density: {detach['critical_density_19']:.4f} x 10^19 m^-3",
        f"- Attached below critical (detached={detach['detached_below_critical']}), "
        f"detached above critical (detached={detach['detached_above_critical']})",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point producing schema-versioned validation evidence."""
    parser = argparse.ArgumentParser(
        description="Validate the two-point scrape-off-layer model against exact closed forms"
    )
    parser.add_argument("--target-id", type=str, default="local-sol-two-point")
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    parser.add_argument("--report", type=str, default=None, help="write sealed JSON evidence and a Markdown summary")
    args = parser.parse_args(argv)

    result = validate_sol_two_point()
    evidence = build_evidence(result, target_id=args.target_id)

    if args.report:
        _write_report(evidence, Path(args.report))

    if args.json_out:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print("Two-point scrape-off-layer model validation")
        print(
            f"  conduction + pressure: cond={result.max_conduction_rel_error:.3e} "
            f"press={result.max_pressure_balance_rel_error:.3e} "
            f"{'ok' if result.conduction_passed and result.pressure_passed else 'FAIL'}"
        )
        print(
            f"  flux mapping + L_par:  flux={result.max_flux_mapping_rel_error:.3e} "
            f"L_par={result.connection_length_rel_error:.3e} "
            f"{'ok' if result.flux_mapping_passed and result.connection_passed else 'FAIL'}"
        )
        print(
            f"  Eich + peak flux:      eich={result.max_scaling_rel_error:.3e} "
            f"peak={result.peak_heat_flux_rel_error:.3e} "
            f"{'ok' if result.scaling_passed and result.peak_flux_passed else 'FAIL'}"
        )
        print(
            f"  detachment boundary:   n_crit={result.detachment.critical_density_19:.3f}e19 "
            f"{'ok' if result.detachment_passed else 'FAIL'}"
        )
        print(f"Status: {'pass' if result.passed else 'fail'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

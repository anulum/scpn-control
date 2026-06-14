#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — DT burn-control alpha-heating analytic validation
"""Validate the DT burn-control alpha-heating algebra against exact closed forms.

The burn controller (``src/scpn_control/control/burn_controller.py``) assembles
the alpha-heating power, the fusion energy gain, the Lawson triple product, the
burn fraction, and the thermal-stability reactivity exponent from the DT
reactivity. The Bosch-Hale reactivity itself is validated separately
(``scpn_control.core.uncertainty.bosch_hale_reactivity``); this validator holds
it as the shared input and checks the alpha-power assembly, the volume
integration, the energy-gain relation, the Lawson and burn-fraction algebra, and
the reactivity-exponent finite difference against their exact closed forms, so
the validation needs no tabulated reactivity data and is self-contained.

Exact references checked against the production classes:

1. **Energy partition.** ``E_fus / E_alpha = 17.6 / 3.52 = 5`` so the fusion
   power is five times the alpha power.
2. **Alpha power density.** ``p_alpha = (n_e/2)^2 <sigma v>(T_i) E_alpha`` for the
   50:50 DT mixture, in MW/m^3.
3. **Volume integration.** With a constant power density,
   ``P_alpha = p_alpha 2 pi^2 R_0 a^2 kappa`` (the trapezoidal integral of the
   shell element ``4 pi^2 R_0 a^2 kappa rho`` is exact for the linear integrand).
4. **Energy gain.** ``Q = 5 P_alpha / P_aux``, with the ignition limits at
   ``P_aux = 0``.
5. **Lawson triple product.** ``n tau_E T`` with its linear scalings and the
   ``3e21`` ignition margin.
6. **Burn fraction.** ``f_b = a^2 n_DT <sigma v> / (4 v_th)`` with its quadratic
   ``a`` scaling.
7. **Reactivity exponent.** ``d ln <sigma v> / d ln T`` reproduced by the
   centred finite difference the production code uses, with the low-temperature
   guard.

References:
  Bosch H.-S. & Hale G.M. (1992) *Nucl. Fusion* 32, 611 (DT reactivity).
  Lawson J.D. (1957) *Proc. Phys. Soc. B* 70, 6 (ignition criterion).
  ITER Physics Basis (1999) *Nucl. Fusion* 39, 2137 (alpha-heating power).
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

import numpy as np

from scpn_control.control.burn_controller import (
    E_ALPHA_J,
    E_FUS_J,
    LAWSON_TRIPLE_PRODUCT,
    AlphaHeating,
    BurnStabilityAnalysis,
    burn_fraction,
    lawson_triple_product,
)
from scpn_control.core.uncertainty import bosch_hale_reactivity

BURN_CONTROL_SCHEMA_VERSION = "scpn-control.burn-control-validation.v1"


def _reactivity(ti_kev: float) -> float:
    return float(np.asarray(bosch_hale_reactivity(np.array([ti_kev])))[0])


@dataclass(frozen=True)
class BurnConfig:
    """Geometry and operating point for the burn-control algebra."""

    n_rho: int
    major_radius_m: float
    minor_radius_m: float
    elongation: float
    electron_density_1e20: float
    electron_temperature_kev: float
    ion_temperature_kev: float
    alpha_power_mw: float
    aux_power_mw: float
    density_m3: float
    confinement_time_s: float
    thermal_speed_ms: float

    def __post_init__(self) -> None:
        _positive_int("n_rho", self.n_rho, minimum=4)
        _positive_float("major_radius_m", self.major_radius_m)
        _positive_float("minor_radius_m", self.minor_radius_m)
        _positive_float("elongation", self.elongation)
        _positive_float("electron_density_1e20", self.electron_density_1e20)
        _positive_float("electron_temperature_kev", self.electron_temperature_kev)
        _positive_float("ion_temperature_kev", self.ion_temperature_kev)
        _positive_float("alpha_power_mw", self.alpha_power_mw)
        _positive_float("aux_power_mw", self.aux_power_mw)
        _positive_float("density_m3", self.density_m3)
        _positive_float("confinement_time_s", self.confinement_time_s)
        _positive_float("thermal_speed_ms", self.thermal_speed_ms)
        if self.minor_radius_m >= self.major_radius_m:
            raise ValueError("minor radius must be smaller than the major radius")

    def alpha_heating(self) -> AlphaHeating:
        return AlphaHeating(R0=self.major_radius_m, a=self.minor_radius_m, kappa=self.elongation)


def default_config() -> BurnConfig:
    """An ITER-like 20 keV burning-plasma operating point."""
    return BurnConfig(
        n_rho=32,
        major_radius_m=6.2,
        minor_radius_m=2.0,
        elongation=1.7,
        electron_density_1e20=1.0,
        electron_temperature_kev=20.0,
        ion_temperature_kev=20.0,
        alpha_power_mw=100.0,
        aux_power_mw=50.0,
        density_m3=1e20,
        confinement_time_s=3.0,
        thermal_speed_ms=1e6,
    )


def energy_partition_rel_error() -> float:
    """Relative error of ``E_fus / E_alpha`` against the exact factor of 5."""
    return float(abs(E_FUS_J / E_ALPHA_J - 5.0) / 5.0)


def power_density_rel_error(config: BurnConfig) -> float:
    """Relative error of the alpha power density against ``(n_e/2)^2 <sigma v> E_alpha``."""
    heating = config.alpha_heating()
    ne = np.ones(config.n_rho) * config.electron_density_1e20
    te = np.ones(config.n_rho) * config.electron_temperature_kev
    ti = np.ones(config.n_rho) * config.ion_temperature_kev
    measured = heating.power_density(ne, te, ti)[0]
    sigv = _reactivity(config.ion_temperature_kev)
    ne_m3 = config.electron_density_1e20 * 1e20
    analytic = (ne_m3 / 2.0) * (ne_m3 / 2.0) * sigv * E_ALPHA_J / 1e6
    return float(abs(measured - analytic) / analytic)


def volume_integral_rel_error(config: BurnConfig) -> float:
    """Relative error of the alpha-power volume integral for a constant power density."""
    heating = config.alpha_heating()
    rho = np.linspace(0.0, 1.0, config.n_rho)
    ne = np.ones(config.n_rho) * config.electron_density_1e20
    te = np.ones(config.n_rho) * config.electron_temperature_kev
    ti = np.ones(config.n_rho) * config.ion_temperature_kev
    p_dens = heating.power_density(ne, te, ti)[0]
    measured = heating.power(ne, te, ti, rho)
    analytic = p_dens * 2.0 * math.pi**2 * config.major_radius_m * config.minor_radius_m**2 * config.elongation
    return float(abs(measured - analytic) / analytic)


def energy_gain_rel_error(config: BurnConfig) -> float:
    """Relative error of the fusion energy gain against ``5 P_alpha / P_aux``."""
    heating = config.alpha_heating()
    measured = heating.Q(config.alpha_power_mw, config.aux_power_mw)
    analytic = 5.0 * config.alpha_power_mw / config.aux_power_mw
    return float(abs(measured - analytic) / analytic)


@dataclass(frozen=True)
class IgnitionLimitCheck:
    """Energy-gain limits at zero auxiliary power."""

    infinite_when_burning: bool
    zero_when_dark: bool


def ignition_limit_check(config: BurnConfig) -> IgnitionLimitCheck:
    """Verify the ``P_aux = 0`` ignition limits of the energy gain."""
    heating = config.alpha_heating()
    burning = heating.Q(config.alpha_power_mw, 0.0)
    dark = heating.Q(0.0, 0.0)
    return IgnitionLimitCheck(
        infinite_when_burning=bool(math.isinf(burning) and burning > 0.0),
        zero_when_dark=bool(dark == 0.0),
    )


def lawson_rel_error(config: BurnConfig) -> float:
    """Relative error of the Lawson triple product against ``n tau_E T``."""
    measured = lawson_triple_product(config.density_m3, config.confinement_time_s, config.ion_temperature_kev)
    analytic = config.density_m3 * config.confinement_time_s * config.ion_temperature_kev
    return float(abs(measured - analytic) / analytic)


def lawson_margin_rel_error(config: BurnConfig) -> float:
    """Relative error of the Lawson ignition margin against ``n tau_E T / 3e21``."""
    triple = lawson_triple_product(config.density_m3, config.confinement_time_s, config.ion_temperature_kev)
    measured = triple / LAWSON_TRIPLE_PRODUCT
    analytic = (config.density_m3 * config.confinement_time_s * config.ion_temperature_kev) / 3e21
    return float(abs(measured - analytic) / analytic)


def burn_fraction_rel_error(config: BurnConfig) -> float:
    """Relative error of the burn fraction against ``a^2 n_DT <sigma v> / (4 v_th)``."""
    sigv = _reactivity(config.ion_temperature_kev)
    measured = burn_fraction(config.density_m3, sigv, config.thermal_speed_ms, config.minor_radius_m)
    analytic = (config.minor_radius_m**2 * config.density_m3 * sigv) / (4.0 * config.thermal_speed_ms)
    return float(abs(measured - analytic) / analytic)


def reactivity_exponent_rel_error(config: BurnConfig) -> float:
    """Relative error of the reactivity exponent against the centred finite difference."""
    heating = config.alpha_heating()
    stability = BurnStabilityAnalysis(heating)
    measured = stability.reactivity_exponent(config.ion_temperature_kev)
    ti = config.ion_temperature_kev
    d_ti = 0.01 * ti
    sv_plus = _reactivity(ti + d_ti)
    sv_minus = _reactivity(ti - d_ti)
    analytic = (math.log(sv_plus) - math.log(sv_minus)) / (math.log(ti + d_ti) - math.log(ti - d_ti))
    return float(abs(measured - analytic) / abs(analytic))


def reactivity_exponent_low_temperature_guard(config: BurnConfig) -> bool:
    """Verify the reactivity exponent returns the cold-plasma guard value of 10."""
    stability = BurnStabilityAnalysis(config.alpha_heating())
    return bool(stability.reactivity_exponent(0.05) == 10.0)


@dataclass(frozen=True)
class ScalingCheck:
    """One burn-physics scaling-law observation."""

    name: str
    measured_ratio: float
    expected_ratio: float
    rel_error: float


def burn_scaling_checks(config: BurnConfig) -> tuple[ScalingCheck, ...]:
    """Verify the energy gain, Lawson product, and burn fraction scale as expected."""
    heating = config.alpha_heating()
    sigv = _reactivity(config.ion_temperature_kev)
    base_q = heating.Q(config.alpha_power_mw, config.aux_power_mw)
    base_lawson = lawson_triple_product(config.density_m3, config.confinement_time_s, config.ion_temperature_kev)
    base_burn = burn_fraction(config.density_m3, sigv, config.thermal_speed_ms, config.minor_radius_m)
    specs = (
        (
            "energy_gain_alpha_linear",
            heating.Q(2.0 * config.alpha_power_mw, config.aux_power_mw) / base_q,
            2.0,
        ),
        (
            "energy_gain_aux_inverse",
            heating.Q(config.alpha_power_mw, 2.0 * config.aux_power_mw) / base_q,
            0.5,
        ),
        (
            "lawson_density_linear",
            lawson_triple_product(2.0 * config.density_m3, config.confinement_time_s, config.ion_temperature_kev)
            / base_lawson,
            2.0,
        ),
        (
            "burn_fraction_minor_radius_square",
            burn_fraction(config.density_m3, sigv, config.thermal_speed_ms, 2.0 * config.minor_radius_m) / base_burn,
            4.0,
        ),
    )
    return tuple(
        ScalingCheck(
            name=name, measured_ratio=ratio, expected_ratio=expected, rel_error=abs(ratio - expected) / expected
        )
        for name, ratio, expected in specs
    )


@dataclass(frozen=True)
class BurnValidationResult:
    """Outcome of the burn-control alpha-heating validation."""

    config: BurnConfig
    energy_partition_rel_error: float
    power_density_rel_error: float
    volume_integral_rel_error: float
    energy_gain_rel_error: float
    ignition_limits: IgnitionLimitCheck
    lawson_rel_error: float
    lawson_margin_rel_error: float
    burn_fraction_rel_error: float
    reactivity_exponent_rel_error: float
    low_temperature_guard_ok: bool
    scaling: tuple[ScalingCheck, ...]
    max_scaling_rel_error: float
    exact_tol: float
    energetics_passed: bool
    lawson_passed: bool
    stability_passed: bool
    scaling_passed: bool
    passed: bool


def validate_burn_control(*, config: BurnConfig | None = None, exact_tol: float = 1e-9) -> BurnValidationResult:
    """Validate the production burn-control alpha-heating algebra against exact relations.

    Every alpha-power, energy-gain, Lawson, burn-fraction, and reactivity-exponent
    relation, together with the ignition limits and scaling laws, must hold to
    ``exact_tol``.
    """
    config = config or default_config()

    partition_err = energy_partition_rel_error()
    pd_err = power_density_rel_error(config)
    vol_err = volume_integral_rel_error(config)
    q_err = energy_gain_rel_error(config)
    ignition = ignition_limit_check(config)
    lawson_err = lawson_rel_error(config)
    margin_err = lawson_margin_rel_error(config)
    burn_err = burn_fraction_rel_error(config)
    react_err = reactivity_exponent_rel_error(config)
    guard_ok = reactivity_exponent_low_temperature_guard(config)
    scaling = burn_scaling_checks(config)
    max_scaling = max(check.rel_error for check in scaling)

    energetics_passed = bool(
        partition_err < exact_tol
        and pd_err < exact_tol
        and vol_err < exact_tol
        and q_err < exact_tol
        and ignition.infinite_when_burning
        and ignition.zero_when_dark
    )
    lawson_passed = bool(lawson_err < exact_tol and margin_err < exact_tol)
    stability_passed = bool(react_err < exact_tol and guard_ok)
    scaling_passed = bool(max_scaling < exact_tol)

    passed = bool(energetics_passed and lawson_passed and stability_passed and scaling_passed)
    return BurnValidationResult(
        config=config,
        energy_partition_rel_error=partition_err,
        power_density_rel_error=pd_err,
        volume_integral_rel_error=vol_err,
        energy_gain_rel_error=q_err,
        ignition_limits=ignition,
        lawson_rel_error=lawson_err,
        lawson_margin_rel_error=margin_err,
        burn_fraction_rel_error=burn_err,
        reactivity_exponent_rel_error=react_err,
        low_temperature_guard_ok=guard_ok,
        scaling=scaling,
        max_scaling_rel_error=max_scaling,
        exact_tol=exact_tol,
        energetics_passed=energetics_passed,
        lawson_passed=lawson_passed,
        stability_passed=stability_passed,
        scaling_passed=scaling_passed,
        passed=passed,
    )


def build_evidence(result: BurnValidationResult, *, target_id: str) -> dict[str, Any]:
    """Build a tamper-evident, schema-versioned validation evidence payload."""
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    payload: dict[str, Any] = {
        "schema_version": BURN_CONTROL_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "target_id": target_id,
        "config": {
            "n_rho": result.config.n_rho,
            "major_radius_m": result.config.major_radius_m,
            "minor_radius_m": result.config.minor_radius_m,
            "elongation": result.config.elongation,
            "electron_density_1e20": result.config.electron_density_1e20,
            "electron_temperature_kev": result.config.electron_temperature_kev,
            "ion_temperature_kev": result.config.ion_temperature_kev,
            "alpha_power_mw": result.config.alpha_power_mw,
            "aux_power_mw": result.config.aux_power_mw,
            "density_m3": result.config.density_m3,
            "confinement_time_s": result.config.confinement_time_s,
            "thermal_speed_ms": result.config.thermal_speed_ms,
        },
        "exact_tol": result.exact_tol,
        "energy_partition_rel_error": result.energy_partition_rel_error,
        "power_density_rel_error": result.power_density_rel_error,
        "volume_integral_rel_error": result.volume_integral_rel_error,
        "energy_gain_rel_error": result.energy_gain_rel_error,
        "ignition_limits": {
            "infinite_when_burning": result.ignition_limits.infinite_when_burning,
            "zero_when_dark": result.ignition_limits.zero_when_dark,
        },
        "lawson_rel_error": result.lawson_rel_error,
        "lawson_margin_rel_error": result.lawson_margin_rel_error,
        "burn_fraction_rel_error": result.burn_fraction_rel_error,
        "reactivity_exponent_rel_error": result.reactivity_exponent_rel_error,
        "low_temperature_guard_ok": result.low_temperature_guard_ok,
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
        "energetics_passed": result.energetics_passed,
        "lawson_passed": result.lawson_passed,
        "stability_passed": result.stability_passed,
        "scaling_passed": result.scaling_passed,
        "passed": result.passed,
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Return ``True`` when a payload is well-formed, sealed, and passing."""
    if payload.get("schema_version") != BURN_CONTROL_SCHEMA_VERSION:
        raise ValueError("unsupported burn control evidence schema_version")
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


def _positive_int(name: str, value: object, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _write_report(evidence: Mapping[str, Any], json_path: Path) -> None:
    """Persist the sealed JSON evidence and a human-readable Markdown summary."""
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = json_path.with_suffix(".md")
    ignition = evidence["ignition_limits"]
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "",
        "# DT Burn-Control Alpha-Heating Validation",
        "",
        f"- Schema: `{evidence['schema_version']}`",
        f"- Generated (UTC): {evidence['generated_utc']}",
        f"- Target: `{evidence['target_id']}`",
        f"- Status: **{'pass' if evidence['passed'] else 'fail'}**",
        "",
        f"## Exact relations (relative error, gate < {evidence['exact_tol']:.1e})",
        "",
        "| relation | value |",
        "| --- | --- |",
        f"| energy partition E_fus/E_alpha = 5 | {evidence['energy_partition_rel_error']:.3e} |",
        f"| alpha power density | {evidence['power_density_rel_error']:.3e} |",
        f"| alpha-power volume integral | {evidence['volume_integral_rel_error']:.3e} |",
        f"| energy gain Q = 5 P_alpha/P_aux | {evidence['energy_gain_rel_error']:.3e} |",
        f"| Lawson triple product n tau T | {evidence['lawson_rel_error']:.3e} |",
        f"| Lawson ignition margin | {evidence['lawson_margin_rel_error']:.3e} |",
        f"| burn fraction | {evidence['burn_fraction_rel_error']:.3e} |",
        f"| reactivity exponent d ln<sv>/d ln T | {evidence['reactivity_exponent_rel_error']:.3e} |",
        f"| scaling laws (max) | {evidence['max_scaling_rel_error']:.3e} |",
        "",
        "## Ignition and stability limits",
        "",
        f"- energy gain is infinite when burning at zero auxiliary power: {ignition['infinite_when_burning']}",
        f"- energy gain is zero with no power: {ignition['zero_when_dark']}",
        f"- reactivity-exponent cold-plasma guard returns 10: {evidence['low_temperature_guard_ok']}",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point producing schema-versioned validation evidence."""
    parser = argparse.ArgumentParser(
        description="Validate the DT burn-control alpha-heating algebra against exact closed forms"
    )
    parser.add_argument("--target-id", type=str, default="local-burn-control")
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    parser.add_argument("--report", type=str, default=None, help="write sealed JSON evidence and a Markdown summary")
    args = parser.parse_args(argv)

    result = validate_burn_control()
    evidence = build_evidence(result, target_id=args.target_id)

    if args.report:
        _write_report(evidence, Path(args.report))

    if args.json_out:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print("DT burn-control alpha-heating validation")
        print(
            f"  energetics: partition={result.energy_partition_rel_error:.3e} "
            f"density={result.power_density_rel_error:.3e} volume={result.volume_integral_rel_error:.3e} "
            f"Q={result.energy_gain_rel_error:.3e} {'ok' if result.energetics_passed else 'FAIL'}"
        )
        print(
            f"  lawson:     triple={result.lawson_rel_error:.3e} margin={result.lawson_margin_rel_error:.3e} "
            f"burn={result.burn_fraction_rel_error:.3e} {'ok' if result.lawson_passed else 'FAIL'}"
        )
        print(
            f"  stability:  reactivity_exp={result.reactivity_exponent_rel_error:.3e} "
            f"scaling={result.max_scaling_rel_error:.3e} "
            f"{'ok' if result.stability_passed and result.scaling_passed else 'FAIL'}"
        )
        print(f"Status: {'pass' if result.passed else 'fail'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

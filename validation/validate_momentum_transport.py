#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Toroidal momentum-transport analytic validation
"""Validate the toroidal momentum-transport diagnostics against exact closed forms.

The momentum-transport model (``src/scpn_control/core/momentum_transport.py``)
provides the neutral-beam torque, the neoclassical radial electric field, the
E×B shearing rate, the turbulence-suppression factor, the Rice intrinsic-rotation
scaling, and the toroidal Mach number. Every relation is an exact algebraic or
finite-difference closed form, so the model can be validated without any measured
rotation profile or external code — the validation is fully self-contained. For
linear test profiles the centred ``np.gradient`` derivative is exact, so the
force-balance and shearing terms are machine-exact.

Exact references checked against the production functions:

1. **NBI torque.** ``T_NBI = P_NBI R0 sin(theta) / v_beam`` (and zero torque for
   a non-positive beam speed).
2. **Radial electric field** (Hinton-Hazeltine force balance).
   ``E_r = (1/(e n_i)) dp_i/dr + R0 omega_phi B_theta`` — recovered exactly for a
   constant pressure (rotation term only) and for a linear pressure (full force
   balance).
3. **E×B shearing rate** (Burrell). ``omega_ExB = |R0 B_theta/B domega_phi/dr|``
   recovered exactly for a linear rotation profile.
4. **Turbulence suppression** (Biglari-Diamond-Terry).
   ``F = 1/(1 + (omega_ExB/gamma)^2)``.
5. **Rice intrinsic rotation.** ``v_phi = 3.5 W_p / I_p`` with linear ``W_p`` and
   inverse ``I_p`` scaling.
6. **Toroidal Mach number.** ``M = |omega_phi R0| / sqrt(T_i e / m_i)``.

References:
  Stacey W. M., Sigmar D. J. (1985) *Phys. Fluids* 28, 2800 (NBI torque).
  Hinton F. L., Hazeltine R. D. (1976) *Rev. Mod. Phys.* 48, 239 (force balance).
  Burrell K. H. (1997) *Phys. Plasmas* 4, 1499 (E×B shearing).
  Rice J. E. et al. (2007) *Nucl. Fusion* 47, 1618 (intrinsic rotation).
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
from numpy.typing import NDArray

from scpn_control.core.momentum_transport import (
    RotationDiagnostics,
    exb_shearing_rate,
    nbi_torque,
    radial_electric_field,
    rice_intrinsic_velocity,
    turbulence_suppression_factor,
)

FloatArray = NDArray[np.float64]

MOMENTUM_TRANSPORT_SCHEMA_VERSION = "scpn-control.momentum-transport-validation.v1"

_E_CHARGE = 1.602e-19  # C, matching momentum_transport.py
_M_ION = 2.0 * 1.67e-27  # kg, deuterium, matching momentum_transport.py
_C_RICE = 3.5  # km/s per (MJ/MA), Rice et al. 2007


@dataclass(frozen=True)
class MomentumConfig:
    """Geometry and grid for the momentum-transport diagnostics."""

    r0: float
    a: float
    b0: float
    nr: int

    def __post_init__(self) -> None:
        _positive_float("r0", self.r0)
        _positive_float("a", self.a)
        _positive_float("b0", self.b0)
        _grid_resolution("nr", self.nr)
        if self.a >= self.r0:
            raise ValueError("a must be smaller than r0 for tokamak ordering")

    def rho(self) -> FloatArray:
        return np.linspace(0.0, 1.0, self.nr, dtype=np.float64)

    def radius(self) -> FloatArray:
        return self.rho() * self.a


def default_config() -> MomentumConfig:
    """An ITER-like geometry on a 51-point radial grid."""
    return MomentumConfig(r0=1.7, a=0.5, b0=2.0, nr=51)


def nbi_torque_rel_error(config: MomentumConfig, *, v_beam: float = 2.0e6, theta_deg: float = 45.0) -> float:
    """Relative error of the NBI torque against ``P_NBI R0 sin(theta)/v_beam``."""
    p_nbi = np.full(config.nr, 1.0e5, dtype=np.float64)
    measured = nbi_torque(p_nbi, config.r0, v_beam, theta_deg)
    analytic = p_nbi * config.r0 * math.sin(math.radians(theta_deg)) / v_beam
    denom = np.maximum(np.abs(analytic), 1e-300)
    return float(np.max(np.abs(measured - analytic) / denom))


def nbi_zero_for_nonpositive_beam(config: MomentumConfig) -> bool:
    """A non-positive beam speed must give zero torque everywhere."""
    p_nbi = np.full(config.nr, 1.0e5, dtype=np.float64)
    return bool(np.all(nbi_torque(p_nbi, config.r0, 0.0, 45.0) == 0.0))


def efield_rotation_term_rel_error(config: MomentumConfig, *, omega: float = 1.0e4, b_theta: float = 0.3) -> float:
    """Relative error of ``E_r`` for a constant pressure (rotation term ``R0 omega B_theta``)."""
    n = config.nr
    ne = np.full(n, 5.0, dtype=np.float64)
    ti = np.full(n, 8.0, dtype=np.float64)
    omega_phi = np.full(n, omega, dtype=np.float64)
    b_theta_arr = np.full(n, b_theta, dtype=np.float64)
    measured = radial_electric_field(ne, ti, omega_phi, b_theta_arr, config.b0, config.r0, config.rho(), config.a)
    analytic = config.r0 * omega_phi * b_theta_arr
    return float(np.max(np.abs(measured - analytic) / np.abs(analytic)))


def efield_force_balance_rel_error(config: MomentumConfig, *, omega: float = 1.0e4, b_theta: float = 0.3) -> float:
    """Relative error of the full Hinton-Hazeltine force balance for a linear pressure."""
    rho = config.rho()
    radius = config.radius()
    ne = np.asarray(3.0 + 2.0 * rho, dtype=np.float64)  # linear -> dp/dr exact under np.gradient
    ti = np.full(config.nr, 8.0, dtype=np.float64)
    omega_phi = np.full(config.nr, omega, dtype=np.float64)
    b_theta_arr = np.full(config.nr, b_theta, dtype=np.float64)
    measured = radial_electric_field(ne, ti, omega_phi, b_theta_arr, config.b0, config.r0, rho, config.a)
    p_i = ne * 1e19 * ti * 1e3 * _E_CHARGE
    dp_dr = np.gradient(p_i, radius, edge_order=2)
    analytic = dp_dr / (_E_CHARGE * ne * 1e19) + config.r0 * omega_phi * b_theta_arr
    denom = np.maximum(np.abs(analytic), 1e-300)
    return float(np.max(np.abs(measured - analytic) / denom))


def exb_shearing_rel_error(config: MomentumConfig, *, b_theta: float = 0.3) -> float:
    """Relative error of the E×B shearing rate for a linear rotation profile."""
    rho = config.rho()
    radius = config.radius()
    omega_phi = np.asarray(1.0e4 + 5.0e4 * rho, dtype=np.float64)
    b_theta_arr = np.full(config.nr, b_theta, dtype=np.float64)
    measured = exb_shearing_rate(omega_phi, b_theta_arr, config.b0, config.r0, rho, config.a)
    domega_dr = np.gradient(omega_phi, radius, edge_order=2)
    b_tot = math.sqrt(config.b0**2 + b_theta**2)
    analytic = np.abs(config.r0 * b_theta_arr / b_tot * domega_dr)
    denom = np.maximum(np.abs(analytic), 1e-300)
    return float(np.max(np.abs(measured - analytic) / denom))


def turbulence_suppression_rel_error() -> float:
    """Relative error of the Biglari-Diamond-Terry suppression factor."""
    omega_exb = np.array([2.0, 1.0, 0.5], dtype=np.float64)
    gamma = np.array([1.0, 1.0, 2.0], dtype=np.float64)
    measured = turbulence_suppression_factor(omega_exb, gamma)
    analytic = 1.0 / (1.0 + (omega_exb / gamma) ** 2)
    return float(np.max(np.abs(measured - analytic) / analytic))


@dataclass(frozen=True)
class ScalingCheck:
    """One Rice-scaling-law observation."""

    name: str
    measured_ratio: float
    expected_ratio: float
    rel_error: float


def rice_rel_error(*, w_p: float = 0.5, i_p: float = 1.0) -> float:
    """Relative error of the Rice intrinsic velocity against ``3.5 W_p / I_p``."""
    measured = rice_intrinsic_velocity(w_p, i_p)
    analytic = _C_RICE * w_p / i_p
    return abs(measured - analytic) / analytic


def rice_scaling_checks() -> tuple[ScalingCheck, ...]:
    """Verify the Rice velocity scales linearly with ``W_p`` and inversely with ``I_p``."""
    base = rice_intrinsic_velocity(0.5, 1.0)
    specs = (
        ("stored_energy_linear", rice_intrinsic_velocity(1.0, 1.0) / base, 2.0),
        ("current_inverse", rice_intrinsic_velocity(0.5, 2.0) / base, 0.5),
    )
    return tuple(
        ScalingCheck(
            name=name, measured_ratio=ratio, expected_ratio=expected, rel_error=abs(ratio - expected) / expected
        )
        for name, ratio, expected in specs
    )


def mach_number_rel_error(config: MomentumConfig, *, omega: float = 1.0e4, ti: float = 8.0) -> float:
    """Relative error of the toroidal Mach number against ``|omega R0|/sqrt(T_i e/m_i)``."""
    omega_phi = np.full(config.nr, omega, dtype=np.float64)
    ti_arr = np.full(config.nr, ti, dtype=np.float64)
    measured = RotationDiagnostics.mach_number(omega_phi, ti_arr, config.r0)
    c_s = math.sqrt(ti * 1e3 * _E_CHARGE / _M_ION)
    analytic = abs(omega * config.r0) / c_s
    return float(np.max(np.abs(measured - analytic) / analytic))


@dataclass(frozen=True)
class MomentumValidationResult:
    """Outcome of the toroidal momentum-transport validation."""

    nbi_rel_error: float
    nbi_zero_for_nonpositive_beam: bool
    efield_rotation_rel_error: float
    efield_force_balance_rel_error: float
    exb_shearing_rel_error: float
    suppression_rel_error: float
    rice_rel_error: float
    rice_scaling: tuple[ScalingCheck, ...]
    max_rice_scaling_rel_error: float
    mach_rel_error: float
    exact_tol: float
    nbi_passed: bool
    efield_passed: bool
    exb_passed: bool
    suppression_passed: bool
    rice_passed: bool
    mach_passed: bool
    passed: bool


def validate_momentum_transport(
    *, config: MomentumConfig | None = None, exact_tol: float = 1e-9
) -> MomentumValidationResult:
    """Validate the production momentum-transport diagnostics against their exact forms.

    The NBI torque, Hinton-Hazeltine radial electric field, E×B shearing rate,
    turbulence-suppression factor, Rice intrinsic-rotation scaling, and Mach
    number must all reproduce their closed forms to ``exact_tol``.
    """
    config = config or default_config()

    nbi_err = nbi_torque_rel_error(config)
    nbi_zero = nbi_zero_for_nonpositive_beam(config)
    efield_rot = efield_rotation_term_rel_error(config)
    efield_fb = efield_force_balance_rel_error(config)
    exb_err = exb_shearing_rel_error(config)
    suppression_err = turbulence_suppression_rel_error()
    rice_err = rice_rel_error()
    rice_scaling = rice_scaling_checks()
    max_rice_scaling = max(check.rel_error for check in rice_scaling)
    mach_err = mach_number_rel_error(config)

    nbi_passed = nbi_err < exact_tol and nbi_zero
    efield_passed = efield_rot < exact_tol and efield_fb < exact_tol
    exb_passed = exb_err < exact_tol
    suppression_passed = suppression_err < exact_tol
    rice_passed = rice_err < exact_tol and max_rice_scaling < exact_tol
    mach_passed = mach_err < exact_tol

    passed = nbi_passed and efield_passed and exb_passed and suppression_passed and rice_passed and mach_passed
    return MomentumValidationResult(
        nbi_rel_error=nbi_err,
        nbi_zero_for_nonpositive_beam=nbi_zero,
        efield_rotation_rel_error=efield_rot,
        efield_force_balance_rel_error=efield_fb,
        exb_shearing_rel_error=exb_err,
        suppression_rel_error=suppression_err,
        rice_rel_error=rice_err,
        rice_scaling=rice_scaling,
        max_rice_scaling_rel_error=max_rice_scaling,
        mach_rel_error=mach_err,
        exact_tol=exact_tol,
        nbi_passed=nbi_passed,
        efield_passed=efield_passed,
        exb_passed=exb_passed,
        suppression_passed=suppression_passed,
        rice_passed=rice_passed,
        mach_passed=mach_passed,
        passed=passed,
    )


def build_evidence(result: MomentumValidationResult, *, target_id: str) -> dict[str, Any]:
    """Build a tamper-evident, schema-versioned validation evidence payload."""
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    payload: dict[str, Any] = {
        "schema_version": MOMENTUM_TRANSPORT_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "target_id": target_id,
        "exact_tol": result.exact_tol,
        "nbi_rel_error": result.nbi_rel_error,
        "nbi_zero_for_nonpositive_beam": result.nbi_zero_for_nonpositive_beam,
        "efield_rotation_rel_error": result.efield_rotation_rel_error,
        "efield_force_balance_rel_error": result.efield_force_balance_rel_error,
        "exb_shearing_rel_error": result.exb_shearing_rel_error,
        "suppression_rel_error": result.suppression_rel_error,
        "rice_rel_error": result.rice_rel_error,
        "rice_scaling": [
            {
                "name": check.name,
                "measured_ratio": check.measured_ratio,
                "expected_ratio": check.expected_ratio,
                "rel_error": check.rel_error,
            }
            for check in result.rice_scaling
        ],
        "max_rice_scaling_rel_error": result.max_rice_scaling_rel_error,
        "mach_rel_error": result.mach_rel_error,
        "nbi_passed": result.nbi_passed,
        "efield_passed": result.efield_passed,
        "exb_passed": result.exb_passed,
        "suppression_passed": result.suppression_passed,
        "rice_passed": result.rice_passed,
        "mach_passed": result.mach_passed,
        "passed": result.passed,
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Return ``True`` when a payload is well-formed, sealed, and passing."""
    if payload.get("schema_version") != MOMENTUM_TRANSPORT_SCHEMA_VERSION:
        raise ValueError("unsupported momentum transport evidence schema_version")
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


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _grid_resolution(name: str, value: object) -> int:
    result = _positive_int(name, value)
    if result < 5:
        raise ValueError(f"{name} must be at least 5 to resolve radial gradients")
    return result


def _write_report(evidence: Mapping[str, Any], json_path: Path) -> None:
    """Persist the sealed JSON evidence and a human-readable Markdown summary."""
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = json_path.with_suffix(".md")
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "",
        "# Toroidal Momentum-Transport Validation",
        "",
        f"- Schema: `{evidence['schema_version']}`",
        f"- Generated (UTC): {evidence['generated_utc']}",
        f"- Target: `{evidence['target_id']}`",
        f"- Status: **{'pass' if evidence['passed'] else 'fail'}**",
        "",
        f"## Exact closed-form references (relative error, gate < {evidence['exact_tol']:.1e})",
        "",
        "| reference | value |",
        "| --- | --- |",
        f"| NBI torque P R0 sin(theta)/v_beam | {evidence['nbi_rel_error']:.3e} |",
        f"| NBI zero for non-positive beam | {evidence['nbi_zero_for_nonpositive_beam']} |",
        f"| E_r rotation term (constant p) | {evidence['efield_rotation_rel_error']:.3e} |",
        f"| E_r force balance (linear p) | {evidence['efield_force_balance_rel_error']:.3e} |",
        f"| ExB shearing rate (linear omega) | {evidence['exb_shearing_rel_error']:.3e} |",
        f"| turbulence suppression factor | {evidence['suppression_rel_error']:.3e} |",
        f"| Rice intrinsic velocity | {evidence['rice_rel_error']:.3e} |",
        f"| Rice scaling laws (max) | {evidence['max_rice_scaling_rel_error']:.3e} |",
        f"| toroidal Mach number | {evidence['mach_rel_error']:.3e} |",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point producing schema-versioned validation evidence."""
    parser = argparse.ArgumentParser(
        description="Validate the toroidal momentum-transport diagnostics against exact closed forms"
    )
    parser.add_argument("--target-id", type=str, default="local-momentum-transport")
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    parser.add_argument("--report", type=str, default=None, help="write sealed JSON evidence and a Markdown summary")
    args = parser.parse_args(argv)

    result = validate_momentum_transport()
    evidence = build_evidence(result, target_id=args.target_id)

    if args.report:
        _write_report(evidence, Path(args.report))

    if args.json_out:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print("Toroidal momentum-transport validation")
        print(
            f"  NBI torque:    rel={result.nbi_rel_error:.3e} zero_beam={result.nbi_zero_for_nonpositive_beam} "
            f"{'ok' if result.nbi_passed else 'FAIL'}"
        )
        print(
            f"  radial E_r:    rotation={result.efield_rotation_rel_error:.3e} "
            f"force_balance={result.efield_force_balance_rel_error:.3e} "
            f"{'ok' if result.efield_passed else 'FAIL'}"
        )
        print(
            f"  ExB + suppr:   exb={result.exb_shearing_rel_error:.3e} "
            f"suppression={result.suppression_rel_error:.3e} "
            f"{'ok' if result.exb_passed and result.suppression_passed else 'FAIL'}"
        )
        print(
            f"  Rice + Mach:   rice={result.rice_rel_error:.3e} "
            f"scaling={result.max_rice_scaling_rel_error:.3e} mach={result.mach_rel_error:.3e} "
            f"{'ok' if result.rice_passed and result.mach_passed else 'FAIL'}"
        )
        print(f"Status: {'pass' if result.passed else 'fail'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

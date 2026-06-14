#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Guiding-centre orbit conservation-law validation
"""Validate the guiding-centre orbit integrator against exact conservation laws.

The fast-ion orbit follower (``src/scpn_control/core/orbit_following.py``)
advances the guiding-centre state ``[R, Z, phi, v_par]`` with an RK4 step using
the magnetic-moment, grad-B/curvature drift, and mirror-force equations of
motion (Boozer 2004). Guiding-centre motion in a static magnetic field admits
*exact* invariants that any correct integrator must preserve — no external
orbit code or measured diagnostic is required, so this validation is fully
self-contained.

Exact invariants checked along each orbit in a static analytic axisymmetric
tokamak field:

1. **Kinetic energy.** ``E = (1/2) m v_par^2 + mu B`` is conserved because the
   magnetic force does no work: the mirror term ``dv_par/dt = -(mu/m) b·∇B``
   exactly cancels ``mu dB/dt`` along the parallel motion, and the drift
   velocity is perpendicular to ``∇B``.
2. **Magnetic moment.** ``mu = m v_perp^2 / (2 B)`` is held as the adiabatic
   invariant, so the parallel speed never exceeds the total speed: the implied
   ``v_perp^2 = v_tot^2 - v_par^2`` stays non-negative.
3. **Canonical toroidal angular momentum.** In an axisymmetric field
   ``p_phi = m R v_par (B_phi/B) + q psi(R, Z)`` is conserved, where ``psi`` is
   the poloidal flux function with ``B_R = -(1/R) ∂psi/∂Z`` and
   ``B_Z = (1/R) ∂psi/∂R``. This exercises the full drift physics, not just the
   parallel streaming.

Passing and trapped (banana) orbits are both covered: a trapped orbit reverses
the sign of ``v_par`` at its bounce points, while a passing orbit does not.

References:
  Boozer A. H. (2004) "Physics of magnetically confined plasmas",
  *Rev. Mod. Phys.* 76, 1071.
  White R. B. (2014) *The Theory of Toroidally Confined Plasmas*, 3rd ed.,
  Imperial College Press, Ch. 3 (guiding-centre invariants).
  Cordey J. G. (1981) *Nucl. Fusion* 21, 1175 (orbit-width estimates).
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

from scpn_control.core.orbit_following import GuidingCenterOrbit

GUIDING_CENTRE_CONSERVATION_SCHEMA_VERSION = "scpn-control.guiding-centre-conservation-validation.v1"


@dataclass(frozen=True)
class AxisymmetricField:
    """Static analytic axisymmetric tokamak field with a closed-form flux function.

    The poloidal flux is ``psi = b_pol_coeff · ((R - r0)^2 + Z^2)``, giving
    ``B_R = -(1/R) ∂psi/∂Z = -2 c Z / R`` and ``B_Z = (1/R) ∂psi/∂R =
    2 c (R - r0) / R`` (circular poloidal field about the magnetic axis), with a
    vacuum toroidal field ``B_phi = b0 r0 / R``. The construction is divergence
    free and axisymmetric, so the canonical toroidal momentum is conserved.
    """

    r0: float
    a: float
    b0: float
    b_pol_coeff: float

    def __post_init__(self) -> None:
        _positive_float("r0", self.r0)
        _positive_float("a", self.a)
        _positive_float("b0", self.b0)
        _positive_float("b_pol_coeff", self.b_pol_coeff)

    def poloidal_flux(self, r: float, z: float) -> float:
        """Poloidal flux function ``psi(R, Z)``."""
        return self.b_pol_coeff * ((r - self.r0) ** 2 + z**2)

    def components(self, r: float, z: float) -> tuple[float, float, float]:
        """Return ``(B_R, B_Z, B_phi)`` at ``(R, Z)``."""
        if r <= 0.0:
            raise ValueError("major radius R must be positive")
        b_r = -2.0 * self.b_pol_coeff * z / r
        b_z = 2.0 * self.b_pol_coeff * (r - self.r0) / r
        b_phi = self.b0 * self.r0 / r
        return b_r, b_z, b_phi

    def magnitude(self, r: float, z: float) -> float:
        b_r, b_z, b_phi = self.components(r, z)
        return math.sqrt(b_r**2 + b_z**2 + b_phi**2)

    def __call__(self, r: float, z: float) -> tuple[float, float, float]:
        return self.components(r, z)


@dataclass(frozen=True)
class OrbitCase:
    """One fast-ion orbit to integrate for the conservation check."""

    label: str
    m_amu: float
    charge: int
    energy_kev: float
    pitch_angle: float
    r_init_offset: float
    dt_s: float
    total_time_s: float

    def __post_init__(self) -> None:
        _non_empty("label", self.label)
        _positive_float("m_amu", self.m_amu)
        if isinstance(self.charge, bool) or not isinstance(self.charge, int) or self.charge == 0:
            raise ValueError("charge must be a non-zero integer")
        _positive_float("energy_kev", self.energy_kev)
        _finite_float("pitch_angle", self.pitch_angle)
        if not 0.0 < self.pitch_angle < math.pi:
            raise ValueError("pitch_angle must lie strictly within (0, pi)")
        _finite_float("r_init_offset", self.r_init_offset)
        _positive_float("dt_s", self.dt_s)
        _positive_float("total_time_s", self.total_time_s)
        if self.total_time_s < self.dt_s:
            raise ValueError("total_time_s must be at least one dt_s")


@dataclass(frozen=True)
class OrbitConservationRecord:
    """Per-orbit conservation diagnostics."""

    label: str
    steps: int
    energy_drift_rel: float
    momentum_drift_rel: float
    max_parallel_speed_ratio: float
    trapped: bool
    energy_passed: bool
    momentum_passed: bool
    parallel_passed: bool


def run_orbit_conservation(
    field: AxisymmetricField,
    case: OrbitCase,
    *,
    energy_tol: float,
    momentum_tol: float,
    parallel_tol: float,
) -> OrbitConservationRecord:
    """Integrate one guiding-centre orbit and measure invariant drift.

    Returns the maximum relative drift of the kinetic energy and the canonical
    toroidal momentum over the trajectory, the largest parallel-to-total speed
    ratio, and whether the orbit is trapped (parallel-velocity sign reversal).
    """
    r_init = field.r0 + case.r_init_offset
    if r_init <= 0.0:
        raise ValueError("initial major radius must be positive")

    orbit = GuidingCenterOrbit(
        m_amu=case.m_amu,
        Z=case.charge,
        E_keV=case.energy_kev,
        pitch_angle=case.pitch_angle,
        R0_init=r_init,
        Z0_init=0.0,
    )
    mass_kg = orbit.m
    charge_c = orbit.Z_charge
    v_total = orbit.v_tot

    b_init = field.magnitude(r_init, 0.0)
    v_perp_0 = v_total * math.sin(case.pitch_angle)
    mu = mass_kg * v_perp_0**2 / (2.0 * b_init)

    def energy() -> float:
        b_mag = field.magnitude(orbit.R, orbit.Z)
        return 0.5 * mass_kg * orbit.v_par**2 + mu * b_mag

    def toroidal_momentum() -> float:
        b_r, b_z, b_phi = field.components(orbit.R, orbit.Z)
        b_mag = math.sqrt(b_r**2 + b_z**2 + b_phi**2)
        return mass_kg * orbit.R * orbit.v_par * (b_phi / b_mag) + charge_c * field.poloidal_flux(orbit.R, orbit.Z)

    energy_0 = energy()
    momentum_0 = toroidal_momentum()
    characteristic_momentum = mass_kg * r_init * v_total
    if abs(momentum_0) < 1e-9 * characteristic_momentum:
        raise ValueError("degenerate orbit: near-zero baseline toroidal momentum makes the relative drift ill-defined")

    max_energy_drift = 0.0
    max_momentum_drift = 0.0
    max_parallel_ratio = abs(float(orbit.v_par)) / v_total
    saw_positive = float(orbit.v_par) > 0.0
    saw_negative = float(orbit.v_par) < 0.0

    steps = int(round(case.total_time_s / case.dt_s))
    for _ in range(steps):
        orbit.step(field, case.dt_s)
        v_par = float(orbit.v_par)
        max_energy_drift = max(max_energy_drift, abs(energy() - energy_0) / abs(energy_0))
        max_momentum_drift = max(max_momentum_drift, abs(toroidal_momentum() - momentum_0) / abs(momentum_0))
        max_parallel_ratio = max(max_parallel_ratio, abs(v_par) / v_total)
        saw_positive = saw_positive or v_par > 0.0
        saw_negative = saw_negative or v_par < 0.0

    return OrbitConservationRecord(
        label=case.label,
        steps=steps,
        energy_drift_rel=float(max_energy_drift),
        momentum_drift_rel=float(max_momentum_drift),
        max_parallel_speed_ratio=float(max_parallel_ratio),
        trapped=bool(saw_positive and saw_negative),
        energy_passed=bool(max_energy_drift < energy_tol),
        momentum_passed=bool(max_momentum_drift < momentum_tol),
        parallel_passed=bool(max_parallel_ratio <= 1.0 + parallel_tol),
    )


def default_field() -> AxisymmetricField:
    """ITER-like analytic field: R0=1.7 m, a=0.5 m, B0=2 T, circular poloidal field."""
    return AxisymmetricField(r0=1.7, a=0.5, b0=2.0, b_pol_coeff=0.5)


def default_cases() -> tuple[OrbitCase, ...]:
    """A passing and a trapped orbit for deuterons and 3.5 MeV alphas."""
    return (
        OrbitCase("passing_deuteron", 2.0, 1, 100.0, 0.5, 0.2, 1e-9, 6e-6),
        OrbitCase("trapped_deuteron", 2.0, 1, 100.0, 1.45, 0.2, 1e-9, 6e-6),
        OrbitCase("passing_alpha", 4.0, 2, 3500.0, 0.6, 0.15, 5e-10, 3e-6),
        OrbitCase("trapped_alpha", 4.0, 2, 3500.0, 1.2, 0.15, 5e-10, 3e-6),
    )


@dataclass(frozen=True)
class GuidingCentreValidationResult:
    """Outcome of the guiding-centre conservation validation."""

    field: AxisymmetricField
    records: tuple[OrbitConservationRecord, ...]
    energy_tol: float
    momentum_tol: float
    parallel_tol: float
    max_energy_drift: float
    max_momentum_drift: float
    energy_passed: bool
    momentum_passed: bool
    parallel_passed: bool
    covers_passing_and_trapped: bool
    passed: bool


def validate_guiding_centre(
    *,
    field: AxisymmetricField | None = None,
    cases: Sequence[OrbitCase] | None = None,
    energy_tol: float = 1e-3,
    momentum_tol: float = 1e-3,
    parallel_tol: float = 1e-6,
) -> GuidingCentreValidationResult:
    """Validate the guiding-centre integrator against exact orbit invariants.

    Each orbit in the ensemble must conserve the kinetic energy and the canonical
    toroidal momentum within the declared relative tolerances, keep the parallel
    speed below the total speed, and the ensemble must contain at least one
    passing and one trapped orbit so both regimes are exercised.
    """
    field = field or default_field()
    case_list = tuple(cases) if cases is not None else default_cases()
    if not case_list:
        raise ValueError("at least one orbit case is required")
    energy_tol = _positive_float("energy_tol", energy_tol)
    momentum_tol = _positive_float("momentum_tol", momentum_tol)
    parallel_tol = _positive_float("parallel_tol", parallel_tol)

    records = tuple(
        run_orbit_conservation(field, case, energy_tol=energy_tol, momentum_tol=momentum_tol, parallel_tol=parallel_tol)
        for case in case_list
    )

    max_energy_drift = max(record.energy_drift_rel for record in records)
    max_momentum_drift = max(record.momentum_drift_rel for record in records)
    energy_passed = all(record.energy_passed for record in records)
    momentum_passed = all(record.momentum_passed for record in records)
    parallel_passed = all(record.parallel_passed for record in records)
    covers_passing_and_trapped = any(record.trapped for record in records) and any(
        not record.trapped for record in records
    )

    passed = energy_passed and momentum_passed and parallel_passed and covers_passing_and_trapped
    return GuidingCentreValidationResult(
        field=field,
        records=records,
        energy_tol=energy_tol,
        momentum_tol=momentum_tol,
        parallel_tol=parallel_tol,
        max_energy_drift=max_energy_drift,
        max_momentum_drift=max_momentum_drift,
        energy_passed=energy_passed,
        momentum_passed=momentum_passed,
        parallel_passed=parallel_passed,
        covers_passing_and_trapped=covers_passing_and_trapped,
        passed=passed,
    )


def build_evidence(result: GuidingCentreValidationResult, *, target_id: str) -> dict[str, Any]:
    """Build a tamper-evident, schema-versioned validation evidence payload."""
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    payload: dict[str, Any] = {
        "schema_version": GUIDING_CENTRE_CONSERVATION_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "target_id": target_id,
        "field": {
            "r0": result.field.r0,
            "a": result.field.a,
            "b0": result.field.b0,
            "b_pol_coeff": result.field.b_pol_coeff,
        },
        "energy_tol": result.energy_tol,
        "momentum_tol": result.momentum_tol,
        "parallel_tol": result.parallel_tol,
        "records": [
            {
                "label": record.label,
                "steps": record.steps,
                "energy_drift_rel": record.energy_drift_rel,
                "momentum_drift_rel": record.momentum_drift_rel,
                "max_parallel_speed_ratio": record.max_parallel_speed_ratio,
                "trapped": record.trapped,
                "energy_passed": record.energy_passed,
                "momentum_passed": record.momentum_passed,
                "parallel_passed": record.parallel_passed,
            }
            for record in result.records
        ],
        "max_energy_drift": result.max_energy_drift,
        "max_momentum_drift": result.max_momentum_drift,
        "energy_passed": result.energy_passed,
        "momentum_passed": result.momentum_passed,
        "parallel_passed": result.parallel_passed,
        "covers_passing_and_trapped": result.covers_passing_and_trapped,
        "passed": result.passed,
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Return ``True`` when a payload is well-formed, sealed, and passing."""
    if payload.get("schema_version") != GUIDING_CENTRE_CONSERVATION_SCHEMA_VERSION:
        raise ValueError("unsupported guiding-centre conservation evidence schema_version")
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


def _non_empty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _write_report(evidence: Mapping[str, Any], json_path: Path) -> None:
    """Persist the sealed JSON evidence and a human-readable Markdown summary."""
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = json_path.with_suffix(".md")
    field = evidence["field"]
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "",
        "# Guiding-Centre Orbit Conservation Validation",
        "",
        f"- Schema: `{evidence['schema_version']}`",
        f"- Generated (UTC): {evidence['generated_utc']}",
        f"- Target: `{evidence['target_id']}`",
        f"- Field: R0={field['r0']} m, a={field['a']} m, B0={field['b0']} T",
        f"- Energy tolerance: {evidence['energy_tol']:.1e}; momentum tolerance: {evidence['momentum_tol']:.1e}",
        f"- Max energy drift: {evidence['max_energy_drift']:.3e}; "
        f"max momentum drift: {evidence['max_momentum_drift']:.3e}",
        f"- Status: **{'pass' if evidence['passed'] else 'fail'}**",
        "",
        "| orbit | steps | dE/E | dp_phi/p_phi | max |v_par|/v_tot | trapped |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {record['label']} | {record['steps']} | {record['energy_drift_rel']:.3e} | "
        f"{record['momentum_drift_rel']:.3e} | {record['max_parallel_speed_ratio']:.6f} | "
        f"{record['trapped']} |"
        for record in evidence["records"]
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point producing schema-versioned validation evidence."""
    parser = argparse.ArgumentParser(
        description="Validate the guiding-centre orbit integrator against exact conservation laws"
    )
    parser.add_argument("--target-id", type=str, default="local-guiding-centre-conservation")
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    parser.add_argument("--report", type=str, default=None, help="write sealed JSON evidence and a Markdown summary")
    args = parser.parse_args(argv)

    result = validate_guiding_centre()
    evidence = build_evidence(result, target_id=args.target_id)

    if args.report:
        _write_report(evidence, Path(args.report))

    if args.json_out:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print(
            f"Guiding-centre conservation validation "
            f"(R0={result.field.r0} m, a={result.field.a} m, B0={result.field.b0} T)"
        )
        for record in result.records:
            print(
                f"  {record.label:18s} dE/E={record.energy_drift_rel:.3e} "
                f"dp_phi/p_phi={record.momentum_drift_rel:.3e} "
                f"trapped={record.trapped!s:5s} "
                f"{'ok' if record.energy_passed and record.momentum_passed and record.parallel_passed else 'FAIL'}"
            )
        print(
            f"  passing+trapped covered={result.covers_passing_and_trapped} "
            f"energy_passed={result.energy_passed} momentum_passed={result.momentum_passed}"
        )
        print(f"Status: {'pass' if result.passed else 'fail'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

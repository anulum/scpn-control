#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Runaway-electron avalanche analytic validation
"""Validate the Rosenbluth-Putvinski runaway-electron model against exact forms.

The runaway-electron model (``src/scpn_control/control/halo_re_physics.py``)
computes the Connor-Hastie critical field, the Dreicer field, the collision time,
the avalanche time constant, and the Rosenbluth-Putvinski avalanche growth rate.
Every relation is an exact algebraic closed form, so the model can be validated
without any Fokker-Planck or measured-disruption artefact — the validation is
fully self-contained.

Exact references checked against the production ``RunawayElectronModel``:

1. **Critical field.** ``E_c = n_e e^3 lnL / (4 pi eps0^2 m_e c^2)`` with linear
   scaling in the total electron density (free plus impurity-bound).
2. **Dreicer field.** ``E_D = n_e e^3 lnL / (4 pi eps0^2 T_e)`` with linear ``n_e``
   and inverse ``T_e`` scaling.
3. **Collision time.** ``tau_coll = 6 pi^2 eps0^2 m_e^2 v_th^3 / (n_e e^4 lnL)``
   with ``v_th = sqrt(2 T_e/m_e)``.
4. **Avalanche time constant.** ``tau_av = (m_e c / (e E_c)) lnL (1 + 1.5(Z_eff -
   1))`` with linear ``Z_eff`` enhancement.
5. **Avalanche growth rate** (Rosenbluth-Putvinski).
   ``gamma_av = n_RE (E/E_c - 1) / (tau_av lnL)`` for ``E > E_c`` (zero below the
   critical field), linear in ``n_RE`` and in ``(E/E_c - 1)``, with a 0.001
   deconfinement factor above 0.3 mol of injected neon.

References:
  Connor J. W., Hastie R. J. (1975) *Nucl. Fusion* 15, 415 (critical/Dreicer field).
  Rosenbluth M. N., Putvinski S. V. (1997) *Nucl. Fusion* 37, 1355 (avalanche).
  Paz-Soldan C. et al. (2019) *Nucl. Fusion* 59, 066025 (RMP deconfinement).
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

from scpn_control.control.halo_re_physics import (
    _C_LIGHT,
    _E_CHARGE,
    _EPSILON0,
    _LN_LAMBDA,
    _M_ELECTRON,
    RunawayElectronModel,
)

RUNAWAY_ELECTRON_SCHEMA_VERSION = "scpn-control.runaway-electron-validation.v1"

_NEON_ELECTRONS_PER_MOL = 5.0e21  # bound-electron contribution per mol, matching the model


def _model(
    *, n_e: float = 1e20, t_e_kev: float = 20.0, z_eff: float = 1.0, neon_mol: float = 0.0
) -> RunawayElectronModel:
    return RunawayElectronModel(n_e=n_e, T_e_keV=t_e_kev, z_eff=z_eff, neon_mol=neon_mol)


def critical_field_rel_error(*, n_e: float = 1e20) -> float:
    """Relative error of the Connor-Hastie critical field against its closed form."""
    model = _model(n_e=n_e)
    analytic = n_e * _E_CHARGE**3 * _LN_LAMBDA / (4.0 * math.pi * _EPSILON0**2 * _M_ELECTRON * _C_LIGHT**2)
    return abs(model.E_c - analytic) / analytic


def dreicer_field_rel_error(*, n_e: float = 1e20, t_e_kev: float = 20.0) -> float:
    """Relative error of the Dreicer field against its closed form."""
    model = _model(n_e=n_e, t_e_kev=t_e_kev)
    t_joules = t_e_kev * 1e3 * _E_CHARGE
    analytic = n_e * _E_CHARGE**3 * _LN_LAMBDA / (4.0 * math.pi * _EPSILON0**2 * t_joules)
    return abs(model.E_D - analytic) / analytic


def collision_time_rel_error(*, n_e: float = 1e20, t_e_kev: float = 20.0) -> float:
    """Relative error of the electron collision time against its closed form."""
    model = _model(n_e=n_e, t_e_kev=t_e_kev)
    v_th = math.sqrt(2.0 * t_e_kev * 1e3 * _E_CHARGE / _M_ELECTRON)
    analytic = 6.0 * math.pi**2 * _EPSILON0**2 * _M_ELECTRON**2 * v_th**3 / (n_e * _E_CHARGE**4 * _LN_LAMBDA)
    return float(abs(model.tau_coll - analytic) / analytic)


def avalanche_time_rel_error(*, z_eff: float = 1.0) -> float:
    """Relative error of the avalanche time constant against its closed form."""
    model = _model(z_eff=z_eff)
    analytic = (_M_ELECTRON * _C_LIGHT / (_E_CHARGE * model.E_c) * _LN_LAMBDA) * (1.0 + 1.5 * (z_eff - 1.0))
    return abs(model.tau_av - analytic) / analytic


def critical_field_includes_impurity_electrons(*, neon_mol: float = 0.5) -> float:
    """Relative error of ``E_c`` with injected neon against the total-density form."""
    model = _model(neon_mol=neon_mol)
    n_tot = 1e20 + neon_mol * _NEON_ELECTRONS_PER_MOL
    analytic = n_tot * _E_CHARGE**3 * _LN_LAMBDA / (4.0 * math.pi * _EPSILON0**2 * _M_ELECTRON * _C_LIGHT**2)
    return abs(model.E_c - analytic) / analytic


def avalanche_rate_rel_error(*, e_over_ec: float = 3.0, n_re: float = 1e15) -> float:
    """Relative error of the Rosenbluth-Putvinski avalanche rate against its closed form."""
    model = _model()
    field = e_over_ec * model.E_c
    measured = model._avalanche_rate(field, n_re)
    analytic = n_re * (field / model.E_c - 1.0) / (model.tau_av * _LN_LAMBDA)
    return abs(measured - analytic) / analytic


@dataclass(frozen=True)
class AvalancheBehaviour:
    """Threshold, linearity, and deconfinement behaviour of the avalanche rate."""

    zero_below_critical: bool
    density_linearity_rel_error: float
    drive_linearity_rel_error: float
    deconfinement_rel_error: float


def avalanche_behaviour() -> AvalancheBehaviour:
    """Check the avalanche threshold, linearity, and RMP deconfinement factor."""
    model = _model()
    e_c = model.E_c
    n_re = 1e15

    below = model._avalanche_rate(0.5 * e_c, n_re)
    base = model._avalanche_rate(3.0 * e_c, n_re)
    doubled_density = model._avalanche_rate(3.0 * e_c, 2.0 * n_re)
    density_rel = abs(doubled_density / base - 2.0) / 2.0
    # (E/E_c - 1): 2 at E=3 E_c, 4 at E=5 E_c -> rate ratio 2.
    doubled_drive = model._avalanche_rate(5.0 * e_c, n_re)
    drive_rel = abs(doubled_drive / base - 2.0) / 2.0

    neon_model = _model(neon_mol=0.5)  # > 0.3 mol -> 0.001 deconfinement
    field = 3.0 * neon_model.E_c
    deconfined = neon_model._avalanche_rate(field, n_re)
    undeconfined = n_re * (field / neon_model.E_c - 1.0) / (neon_model.tau_av * _LN_LAMBDA)
    deconfinement_rel = abs(deconfined - 0.001 * undeconfined) / (0.001 * undeconfined)

    return AvalancheBehaviour(
        zero_below_critical=bool(below == 0.0),
        density_linearity_rel_error=density_rel,
        drive_linearity_rel_error=drive_rel,
        deconfinement_rel_error=deconfinement_rel,
    )


@dataclass(frozen=True)
class ScalingCheck:
    """One scaling-law observation."""

    name: str
    measured_ratio: float
    expected_ratio: float
    rel_error: float


def scaling_checks() -> tuple[ScalingCheck, ...]:
    """Verify the density, temperature, and Z_eff scalings of the runaway fields."""
    base_ec = _model().E_c
    base_ed = _model().E_D
    base_tav = _model().tau_av
    specs = (
        ("critical_field_density_linear", _model(n_e=2e20).E_c / base_ec, 2.0),
        ("dreicer_density_linear", _model(n_e=2e20).E_D / base_ed, 2.0),
        ("dreicer_temperature_inverse", _model(t_e_kev=40.0).E_D / base_ed, 0.5),
        ("avalanche_time_z_eff", _model(z_eff=3.0).tau_av / base_tav, 4.0),
    )
    return tuple(
        ScalingCheck(
            name=name, measured_ratio=ratio, expected_ratio=expected, rel_error=abs(ratio - expected) / expected
        )
        for name, ratio, expected in specs
    )


@dataclass(frozen=True)
class RunawayValidationResult:
    """Outcome of the runaway-electron model validation."""

    critical_field_rel_error: float
    dreicer_field_rel_error: float
    collision_time_rel_error: float
    avalanche_time_rel_error: float
    impurity_critical_field_rel_error: float
    avalanche_rate_rel_error: float
    avalanche: AvalancheBehaviour
    scaling: tuple[ScalingCheck, ...]
    max_scaling_rel_error: float
    exact_tol: float
    fields_passed: bool
    avalanche_passed: bool
    scaling_passed: bool
    passed: bool


def validate_runaway_electron(*, exact_tol: float = 1e-9) -> RunawayValidationResult:
    """Validate the production runaway-electron model against its exact closed forms.

    The critical and Dreicer fields, the collision and avalanche time constants,
    the impurity-aware critical field, the avalanche growth rate, its threshold,
    linearity, and deconfinement, and the field scalings must all hold to
    ``exact_tol``.
    """
    crit = critical_field_rel_error()
    dreicer = dreicer_field_rel_error()
    collision = collision_time_rel_error()
    avalanche_time = avalanche_time_rel_error()
    impurity = critical_field_includes_impurity_electrons()
    avalanche_rate = avalanche_rate_rel_error()
    behaviour = avalanche_behaviour()
    scaling = scaling_checks()
    max_scaling = max(check.rel_error for check in scaling)

    fields_passed = (
        crit < exact_tol
        and dreicer < exact_tol
        and collision < exact_tol
        and avalanche_time < exact_tol
        and impurity < exact_tol
    )
    avalanche_passed = (
        avalanche_rate < exact_tol
        and behaviour.zero_below_critical
        and behaviour.density_linearity_rel_error < exact_tol
        and behaviour.drive_linearity_rel_error < exact_tol
        and behaviour.deconfinement_rel_error < exact_tol
    )
    scaling_passed = max_scaling < exact_tol

    passed = fields_passed and avalanche_passed and scaling_passed
    return RunawayValidationResult(
        critical_field_rel_error=crit,
        dreicer_field_rel_error=dreicer,
        collision_time_rel_error=collision,
        avalanche_time_rel_error=avalanche_time,
        impurity_critical_field_rel_error=impurity,
        avalanche_rate_rel_error=avalanche_rate,
        avalanche=behaviour,
        scaling=scaling,
        max_scaling_rel_error=max_scaling,
        exact_tol=exact_tol,
        fields_passed=fields_passed,
        avalanche_passed=avalanche_passed,
        scaling_passed=scaling_passed,
        passed=passed,
    )


def build_evidence(result: RunawayValidationResult, *, target_id: str) -> dict[str, Any]:
    """Build a tamper-evident, schema-versioned validation evidence payload."""
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    payload: dict[str, Any] = {
        "schema_version": RUNAWAY_ELECTRON_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "target_id": target_id,
        "exact_tol": result.exact_tol,
        "critical_field_rel_error": result.critical_field_rel_error,
        "dreicer_field_rel_error": result.dreicer_field_rel_error,
        "collision_time_rel_error": result.collision_time_rel_error,
        "avalanche_time_rel_error": result.avalanche_time_rel_error,
        "impurity_critical_field_rel_error": result.impurity_critical_field_rel_error,
        "avalanche_rate_rel_error": result.avalanche_rate_rel_error,
        "avalanche": {
            "zero_below_critical": result.avalanche.zero_below_critical,
            "density_linearity_rel_error": result.avalanche.density_linearity_rel_error,
            "drive_linearity_rel_error": result.avalanche.drive_linearity_rel_error,
            "deconfinement_rel_error": result.avalanche.deconfinement_rel_error,
        },
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
        "fields_passed": result.fields_passed,
        "avalanche_passed": result.avalanche_passed,
        "scaling_passed": result.scaling_passed,
        "passed": result.passed,
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Return ``True`` when a payload is well-formed, sealed, and passing."""
    if payload.get("schema_version") != RUNAWAY_ELECTRON_SCHEMA_VERSION:
        raise ValueError("unsupported runaway electron evidence schema_version")
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


def _write_report(evidence: Mapping[str, Any], json_path: Path) -> None:
    """Persist the sealed JSON evidence and a human-readable Markdown summary."""
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = json_path.with_suffix(".md")
    avalanche = evidence["avalanche"]
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "",
        "# Runaway-Electron Avalanche Validation",
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
        f"| Connor-Hastie critical field E_c | {evidence['critical_field_rel_error']:.3e} |",
        f"| Dreicer field E_D | {evidence['dreicer_field_rel_error']:.3e} |",
        f"| collision time tau_coll | {evidence['collision_time_rel_error']:.3e} |",
        f"| avalanche time tau_av | {evidence['avalanche_time_rel_error']:.3e} |",
        f"| impurity-aware E_c (neon) | {evidence['impurity_critical_field_rel_error']:.3e} |",
        f"| Rosenbluth-Putvinski avalanche rate | {evidence['avalanche_rate_rel_error']:.3e} |",
        f"| avalanche zero below E_c | {avalanche['zero_below_critical']} |",
        f"| avalanche density linearity | {avalanche['density_linearity_rel_error']:.3e} |",
        f"| avalanche drive linearity | {avalanche['drive_linearity_rel_error']:.3e} |",
        f"| RMP deconfinement factor (0.001) | {avalanche['deconfinement_rel_error']:.3e} |",
        "",
        "## Scaling laws",
        "",
        "| law | measured ratio | expected | rel error |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {check['name']} | {check['measured_ratio']:.6f} | {check['expected_ratio']:.1f} | {check['rel_error']:.3e} |"
        for check in evidence["scaling"]
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point producing schema-versioned validation evidence."""
    parser = argparse.ArgumentParser(
        description="Validate the Rosenbluth-Putvinski runaway-electron model against exact closed forms"
    )
    parser.add_argument("--target-id", type=str, default="local-runaway-electron")
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    parser.add_argument("--report", type=str, default=None, help="write sealed JSON evidence and a Markdown summary")
    args = parser.parse_args(argv)

    result = validate_runaway_electron()
    evidence = build_evidence(result, target_id=args.target_id)

    if args.report:
        _write_report(evidence, Path(args.report))

    if args.json_out:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print("Runaway-electron avalanche validation")
        print(
            f"  fields:     E_c={result.critical_field_rel_error:.3e} E_D={result.dreicer_field_rel_error:.3e} "
            f"tau_coll={result.collision_time_rel_error:.3e} tau_av={result.avalanche_time_rel_error:.3e} "
            f"{'ok' if result.fields_passed else 'FAIL'}"
        )
        print(
            f"  avalanche:  rate={result.avalanche_rate_rel_error:.3e} "
            f"threshold={result.avalanche.zero_below_critical} "
            f"deconfine={result.avalanche.deconfinement_rel_error:.3e} "
            f"{'ok' if result.avalanche_passed else 'FAIL'}"
        )
        print(f"  scaling:    max={result.max_scaling_rel_error:.3e} {'ok' if result.scaling_passed else 'FAIL'}")
        print(f"Status: {'pass' if result.passed else 'fail'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

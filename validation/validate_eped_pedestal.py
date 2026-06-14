#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — EPED pedestal model analytic validation
"""Validate the EPED1 pedestal model against its exact construction relations.

The EPED-style pedestal model (``src/scpn_control/core/eped_pedestal.py``)
predicts a self-consistent ``(p_ped, Delta_ped)`` from the peeling-ballooning and
kinetic-ballooning-mode (KBM) constraints. The model is built from exact
algebraic relations plus a fixed-point iteration; the algebraic relations hold to
machine precision in every result, so the model can be validated without any
external pedestal code — the validation is fully self-contained.

Exact references checked against the production ``eped1_predict`` and helpers:

1. **Safety factor** ``q95 = a B0 / (R0 B_pol) (1 + kappa^2)/2``.
2. **Alpha-inversion pressure** ``p_ped = alpha_crit B0^2 a Delta /
   (2 mu_0 q95^2 R0)``.
3. **Poloidal beta** ``beta_p = 2 mu_0 p_ped / B_pol^2``.
4. **Pedestal temperature** ``T_ped = p_ped / (2 n_e e)`` (T_i = T_e).
5. **Collisionality narrowing** ``Delta = Delta_KBM / (1 + 0.4 ln(1 + nu*))`` with
   the identity ``Delta = Delta_KBM`` at ``nu* = 0``.
6. **Shaping factor** normalised to unity at the ITER reference shape
   ``(kappa, delta) = (1.7, 0.33)``.
7. **KBM width constraint** ``Delta_KBM = C_KBM sqrt(beta_p)`` satisfied at the
   converged collisionless width within the fixed-point iteration tolerance.

The Rust ``control-core/src/pedestal.rs`` implements a separate simplified
ELM-trigger pedestal proxy with a different width scaling and is not exposed to
Python; it is not a parity counterpart of this EPED1 model, so no cross-language
parity is asserted.

References:
  Snyder P. B. et al. (2009) *Phys. Plasmas* 16, 056118 (EPED1, KBM width).
  Snyder P. B. et al. (2011) *Nucl. Fusion* 51, 103016 (collisionality).
  Connor J. W. et al. (1998) *Phys. Plasmas* 5, 2687 (ballooning boundary).
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

from scpn_control.core.eped_pedestal import (
    EPEDConfig,
    _approx_alpha_crit,
    _compute_q95,
    _shaping_factor,
    eped1_predict,
)

EPED_PEDESTAL_SCHEMA_VERSION = "scpn-control.eped-pedestal-validation.v1"

_MU_0 = 4.0e-7 * math.pi
_E_CHARGE = 1.602e-19  # J/eV, matching eped_pedestal.py
_KAPPA_REF = 1.7
_DELTA_REF = 0.33
_COLL_NARROW_COEFF = 0.4


def default_config() -> EPEDConfig:
    """An ITER-like pedestal operating point at finite collisionality."""
    return EPEDConfig(
        R0=1.7, a=0.5, B0=2.0, kappa=1.7, delta=0.33, Ip_MA=1.0, ne_ped_19=5.0, B_pol_ped=0.5, nu_star_e=1.0
    )


def q95_rel_error(config: EPEDConfig) -> float:
    """Relative error of ``q95`` against ``a B0/(R0 B_pol)(1+kappa^2)/2``."""
    measured = _compute_q95(config)
    analytic = (config.a * config.B0) / (config.R0 * config.B_pol_ped) * (1.0 + config.kappa**2) / 2.0
    return abs(measured - analytic) / analytic


def alpha_inversion_rel_error(config: EPEDConfig) -> float:
    """Relative error of the alpha-inversion pedestal pressure relation."""
    result = eped1_predict(config)
    q95 = _compute_q95(config)
    p_ped_pa = result.p_ped_kPa * 1000.0
    analytic = result.alpha_crit * config.B0**2 * config.a * result.delta_ped / (2.0 * _MU_0 * q95**2 * config.R0)
    return abs(p_ped_pa - analytic) / analytic


def beta_p_rel_error(config: EPEDConfig) -> float:
    """Relative error of the poloidal beta definition ``beta_p = 2 mu_0 p / B_pol^2``."""
    result = eped1_predict(config)
    p_ped_pa = result.p_ped_kPa * 1000.0
    analytic = 2.0 * _MU_0 * p_ped_pa / config.B_pol_ped**2
    return abs(result.beta_p_ped - analytic) / analytic


def temperature_rel_error(config: EPEDConfig) -> float:
    """Relative error of the ideal-gas pedestal temperature ``T_ped = p/(2 n_e e)``."""
    result = eped1_predict(config)
    p_ped_pa = result.p_ped_kPa * 1000.0
    analytic = p_ped_pa / (2.0 * config.ne_ped_19 * 1e19) / (_E_CHARGE * 1000.0)
    return abs(result.T_ped_keV - analytic) / analytic


def collisionality_rel_error(config: EPEDConfig) -> float:
    """Relative error of the collisionality width-narrowing correction."""
    result = eped1_predict(config)
    analytic = result.delta_ped_collisionless / (1.0 + _COLL_NARROW_COEFF * math.log(1.0 + config.nu_star_e))
    return abs(result.delta_ped - analytic) / analytic


def collisionless_identity_holds(config: EPEDConfig) -> bool:
    """At ``nu* = 0`` the corrected width must equal the collisionless KBM width."""
    import dataclasses

    result = eped1_predict(dataclasses.replace(config, nu_star_e=0.0))
    return result.delta_ped == result.delta_ped_collisionless


@dataclass(frozen=True)
class ShapingCheck:
    """Shaping-factor reference observation."""

    reference_value: float
    reference_rel_error: float
    monotonic_in_triangularity: bool


def shaping_checks() -> ShapingCheck:
    """The shaping factor is unity at the ITER reference and rises with triangularity."""
    reference = _shaping_factor(_KAPPA_REF, _DELTA_REF)
    higher_delta = _shaping_factor(_KAPPA_REF, 0.5)
    lower_delta = _shaping_factor(_KAPPA_REF, 0.1)
    return ShapingCheck(
        reference_value=reference,
        reference_rel_error=abs(reference - 1.0),
        monotonic_in_triangularity=bool(lower_delta < reference < higher_delta),
    )


def kbm_fixed_point_rel_error(config: EPEDConfig) -> float:
    """Relative residual of the KBM constraint at the converged collisionless width.

    Recomputes ``beta_p`` at the collisionless width and compares it to the KBM
    inversion ``Delta = C_KBM sqrt(beta_p)``. The residual reflects the
    fixed-point iteration tolerance rather than machine precision.
    """
    result = eped1_predict(config)
    q95 = _compute_q95(config)
    delta_c = result.delta_ped_collisionless
    alpha_crit = _approx_alpha_crit(delta_c, config)
    p_ped_pa = alpha_crit * config.B0**2 * config.a * delta_c / (2.0 * _MU_0 * q95**2 * config.R0)
    beta_p = 2.0 * _MU_0 * p_ped_pa / config.B_pol_ped**2
    kbm_width = config.C_KBM * math.sqrt(abs(beta_p))
    return abs(delta_c - kbm_width) / delta_c


@dataclass(frozen=True)
class EPEDValidationResult:
    """Outcome of the EPED pedestal model validation."""

    q95_rel_error: float
    alpha_inversion_rel_error: float
    beta_p_rel_error: float
    temperature_rel_error: float
    collisionality_rel_error: float
    collisionless_identity: bool
    shaping: ShapingCheck
    kbm_fixed_point_rel_error: float
    exact_tol: float
    kbm_tol: float
    construction_passed: bool
    collisionality_passed: bool
    shaping_passed: bool
    kbm_passed: bool
    passed: bool


def validate_eped_pedestal(
    *,
    config: EPEDConfig | None = None,
    exact_tol: float = 1e-9,
    kbm_tol: float = 3e-2,
) -> EPEDValidationResult:
    """Validate the production EPED pedestal model against its exact relations.

    The safety factor, alpha-inversion pressure, poloidal beta, temperature,
    collisionality correction, and shaping factor must hold to ``exact_tol``; the
    KBM width constraint must hold at the converged collisionless width within
    ``kbm_tol`` (the fixed-point iteration tolerance).
    """
    config = config or default_config()

    q95_err = q95_rel_error(config)
    alpha_err = alpha_inversion_rel_error(config)
    beta_err = beta_p_rel_error(config)
    temp_err = temperature_rel_error(config)
    coll_err = collisionality_rel_error(config)
    coll_identity = collisionless_identity_holds(config)
    shaping = shaping_checks()
    kbm_err = kbm_fixed_point_rel_error(config)

    construction_passed = (
        q95_err < exact_tol and alpha_err < exact_tol and beta_err < exact_tol and temp_err < exact_tol
    )
    collisionality_passed = coll_err < exact_tol and coll_identity
    shaping_passed = shaping.reference_rel_error < exact_tol and shaping.monotonic_in_triangularity
    kbm_passed = kbm_err < kbm_tol

    passed = construction_passed and collisionality_passed and shaping_passed and kbm_passed
    return EPEDValidationResult(
        q95_rel_error=q95_err,
        alpha_inversion_rel_error=alpha_err,
        beta_p_rel_error=beta_err,
        temperature_rel_error=temp_err,
        collisionality_rel_error=coll_err,
        collisionless_identity=coll_identity,
        shaping=shaping,
        kbm_fixed_point_rel_error=kbm_err,
        exact_tol=exact_tol,
        kbm_tol=kbm_tol,
        construction_passed=construction_passed,
        collisionality_passed=collisionality_passed,
        shaping_passed=shaping_passed,
        kbm_passed=kbm_passed,
        passed=passed,
    )


def build_evidence(result: EPEDValidationResult, *, target_id: str) -> dict[str, Any]:
    """Build a tamper-evident, schema-versioned validation evidence payload."""
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    payload: dict[str, Any] = {
        "schema_version": EPED_PEDESTAL_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "target_id": target_id,
        "exact_tol": result.exact_tol,
        "kbm_tol": result.kbm_tol,
        "q95_rel_error": result.q95_rel_error,
        "alpha_inversion_rel_error": result.alpha_inversion_rel_error,
        "beta_p_rel_error": result.beta_p_rel_error,
        "temperature_rel_error": result.temperature_rel_error,
        "collisionality_rel_error": result.collisionality_rel_error,
        "collisionless_identity": result.collisionless_identity,
        "shaping_reference_value": result.shaping.reference_value,
        "shaping_reference_rel_error": result.shaping.reference_rel_error,
        "shaping_monotonic_in_triangularity": result.shaping.monotonic_in_triangularity,
        "kbm_fixed_point_rel_error": result.kbm_fixed_point_rel_error,
        "construction_passed": result.construction_passed,
        "collisionality_passed": result.collisionality_passed,
        "shaping_passed": result.shaping_passed,
        "kbm_passed": result.kbm_passed,
        "passed": result.passed,
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Return ``True`` when a payload is well-formed, sealed, and passing."""
    if payload.get("schema_version") != EPED_PEDESTAL_SCHEMA_VERSION:
        raise ValueError("unsupported eped pedestal evidence schema_version")
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
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "",
        "# EPED Pedestal Model Validation",
        "",
        f"- Schema: `{evidence['schema_version']}`",
        f"- Generated (UTC): {evidence['generated_utc']}",
        f"- Target: `{evidence['target_id']}`",
        f"- Status: **{'pass' if evidence['passed'] else 'fail'}**",
        "",
        f"## Exact construction relations (relative error, gate < {evidence['exact_tol']:.1e})",
        "",
        "| relation | value |",
        "| --- | --- |",
        f"| q95 = a B0/(R0 B_pol)(1+kappa^2)/2 | {evidence['q95_rel_error']:.3e} |",
        f"| alpha-inversion pressure | {evidence['alpha_inversion_rel_error']:.3e} |",
        f"| beta_p = 2 mu_0 p/B_pol^2 | {evidence['beta_p_rel_error']:.3e} |",
        f"| T_ped = p/(2 n_e e) | {evidence['temperature_rel_error']:.3e} |",
        f"| collisionality narrowing | {evidence['collisionality_rel_error']:.3e} |",
        f"| nu*=0 collisionless identity | {evidence['collisionless_identity']} |",
        f"| shaping factor reference (=1) | {evidence['shaping_reference_rel_error']:.3e} |",
        f"| shaping monotonic in triangularity | {evidence['shaping_monotonic_in_triangularity']} |",
        "",
        "## KBM width constraint (fixed-point iteration tolerance)",
        "",
        f"- KBM residual at collisionless width: {evidence['kbm_fixed_point_rel_error']:.3e} "
        f"(gate < {evidence['kbm_tol']:.1e})",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point producing schema-versioned validation evidence."""
    parser = argparse.ArgumentParser(
        description="Validate the EPED pedestal model against its exact construction relations"
    )
    parser.add_argument("--target-id", type=str, default="local-eped-pedestal")
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    parser.add_argument("--report", type=str, default=None, help="write sealed JSON evidence and a Markdown summary")
    args = parser.parse_args(argv)

    result = validate_eped_pedestal()
    evidence = build_evidence(result, target_id=args.target_id)

    if args.report:
        _write_report(evidence, Path(args.report))

    if args.json_out:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print("EPED pedestal model validation")
        print(
            f"  construction: q95={result.q95_rel_error:.3e} alpha={result.alpha_inversion_rel_error:.3e} "
            f"beta_p={result.beta_p_rel_error:.3e} T_ped={result.temperature_rel_error:.3e} "
            f"{'ok' if result.construction_passed else 'FAIL'}"
        )
        print(
            f"  collisionality: rel={result.collisionality_rel_error:.3e} "
            f"identity={result.collisionless_identity} "
            f"{'ok' if result.collisionality_passed else 'FAIL'}"
        )
        print(
            f"  shaping:       ref_err={result.shaping.reference_rel_error:.3e} "
            f"monotonic={result.shaping.monotonic_in_triangularity} "
            f"{'ok' if result.shaping_passed else 'FAIL'}"
        )
        print(
            f"  KBM constraint: residual={result.kbm_fixed_point_rel_error:.3e} {'ok' if result.kbm_passed else 'FAIL'}"
        )
        print(f"Status: {'pass' if result.passed else 'fail'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

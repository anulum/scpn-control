#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — ELM peeling-ballooning and crash analytic validation
"""Validate the ELM peeling-ballooning boundary and crash against exact forms.

The ELM model (``src/scpn_control/core/elm_model.py``) evaluates the
peeling-ballooning stability boundary and applies a Type-I ELM crash to the
pedestal. Both are exact algebraic closed forms, so the model can be validated
without any external pedestal-stability code — the validation is fully
self-contained.

Exact references checked against the production methods:

1. **Ballooning limit.** ``alpha_crit = 0.5 max(s, 0.1) (1 + kappa^2(1 + 2 delta^2))``.
2. **Peeling limit.** ``j_crit ∝ (R0/a) F(kappa, delta) / (q95 sqrt(n_mode))`` with
   the exact ``1/q95``, ``1/sqrt(n_mode)``, and ``R0/a`` scaling.
3. **Elliptical PB boundary.** The stability margin is
   ``1 - sqrt((j/j_crit)^2 + (alpha/alpha_crit)^2)``; a point on the unit ellipse
   has zero margin, an interior point is stable, and an exterior point unstable,
   with ``is_unstable`` consistent with a negative margin.
4. **ELM crash energy.** A Type-I crash loses ``Delta_W = f W_ped`` and drops
   ``T`` and ``n`` by ``sqrt(1 - f)`` so the post-crash stored energy
   ``W_post = (1 - f) W_ped`` exactly; the peak heat flux is
   ``Delta_W / (A_wet tau_ELM)``.
5. **Profile crash.** Applying the crash scales the pedestal-region ``n T``
   product by ``(1 - f)`` while leaving the core region unchanged.

References:
  Snyder P. B. et al. (2002) *Phys. Plasmas* 9, 2037 (peeling-ballooning).
  Sauter O. et al. (1999) *Phys. Plasmas* 6, 2834 (shaping factor).
  Loarte A. et al. (2003) *Plasma Phys. Control. Fusion* 45, 1549 (ELM energy).
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

from scpn_control.core.elm_model import ELMCrashModel, PeelingBallooningBoundary

FloatArray = NDArray[np.float64]

ELM_PEELING_BALLOONING_SCHEMA_VERSION = "scpn-control.elm-peeling-ballooning-validation.v1"


@dataclass(frozen=True)
class ELMConfig:
    """Edge geometry for the peeling-ballooning boundary."""

    q95: float
    kappa: float
    delta: float
    a: float
    r0: float

    def __post_init__(self) -> None:
        _positive_float("q95", self.q95)
        _positive_float("kappa", self.kappa)
        _finite_float("delta", self.delta)
        _positive_float("a", self.a)
        _positive_float("r0", self.r0)
        if self.a >= self.r0:
            raise ValueError("a must be smaller than r0 for tokamak ordering")

    def boundary(self) -> PeelingBallooningBoundary:
        return PeelingBallooningBoundary(self.q95, self.kappa, self.delta, self.a, self.r0)


def default_config() -> ELMConfig:
    """An ITER-like edge with q95 = 3.5, kappa = 1.7, delta = 0.33."""
    return ELMConfig(q95=3.5, kappa=1.7, delta=0.33, a=0.5, r0=1.7)


def ballooning_limit_rel_error(config: ELMConfig, *, s_edge: float = 1.2) -> float:
    """Relative error of ``alpha_crit`` against its closed form."""
    measured = config.boundary().ballooning_limit(s_edge)
    analytic = 0.5 * max(s_edge, 0.1) * (1.0 + config.kappa**2 * (1.0 + 2.0 * config.delta**2))
    return abs(measured - analytic) / analytic


def peeling_limit_rel_error(config: ELMConfig, *, n_mode: int = 10) -> float:
    """Relative error of ``j_crit`` against its closed form."""
    measured = config.boundary().peeling_limit(1.0e6, n_mode)
    aspect = config.r0 / config.a
    shape = (1.0 + config.kappa**2) / (2.0 * config.kappa)
    triangularity = 1.0 + config.delta**2
    analytic = 1.0e6 * aspect * shape * triangularity / (config.q95 * math.sqrt(float(n_mode)))
    return abs(measured - analytic) / analytic


@dataclass(frozen=True)
class ScalingCheck:
    """One peeling-limit scaling-law observation."""

    name: str
    measured_ratio: float
    expected_ratio: float
    rel_error: float


def peeling_scaling_checks(config: ELMConfig) -> tuple[ScalingCheck, ...]:
    """Verify ``j_crit`` scales as ``1/q95``, ``1/sqrt(n_mode)``, and ``R0/a``."""
    boundary = config.boundary()
    base = boundary.peeling_limit(1.0e6, 10)
    double_q = ELMConfig(2.0 * config.q95, config.kappa, config.delta, config.a, config.r0)
    quad_n = boundary.peeling_limit(1.0e6, 40) / base
    double_aspect = ELMConfig(config.q95, config.kappa, config.delta, 0.5 * config.a, config.r0)
    specs = (
        ("q95_inverse", double_q.boundary().peeling_limit(1.0e6, 10) / base, 0.5),
        ("n_mode_inverse_sqrt", quad_n, 0.5),
        ("aspect_ratio_linear", double_aspect.boundary().peeling_limit(1.0e6, 10) / base, 2.0),
    )
    return tuple(
        ScalingCheck(
            name=name, measured_ratio=ratio, expected_ratio=expected, rel_error=abs(ratio - expected) / expected
        )
        for name, ratio, expected in specs
    )


@dataclass(frozen=True)
class BoundaryCheck:
    """Elliptical peeling-ballooning boundary observations."""

    on_boundary_margin_abs: float
    margin_formula_rel_error: float
    interior_stable: bool
    exterior_unstable: bool
    flag_margin_consistent: bool


def boundary_checks(config: ELMConfig, *, s_edge: float = 1.0) -> BoundaryCheck:
    """Check the elliptical boundary margin, on-ellipse zero, and flag consistency."""
    boundary = config.boundary()
    j_crit = boundary.peeling_limit(1.0e6)
    a_crit = boundary.ballooning_limit(s_edge)

    # On the unit ellipse: j_norm^2 + a_norm^2 = 1 -> margin = 0.
    theta = 0.7
    j_on = j_crit * math.cos(theta)
    a_on = a_crit * math.sin(theta)
    on_margin = boundary.stability_margin(a_on, j_on, s_edge)

    # Margin formula reconstruction at an exterior point.
    j_ext, a_ext = 1.5 * j_crit, 1.2 * a_crit
    measured_margin = boundary.stability_margin(a_ext, j_ext, s_edge)
    analytic_margin = 1.0 - math.sqrt((j_ext / j_crit) ** 2 + (a_ext / a_crit) ** 2)
    margin_rel = abs(measured_margin - analytic_margin) / abs(analytic_margin)

    interior_stable = (not boundary.is_unstable(0.3 * a_crit, 0.3 * j_crit, s_edge)) and boundary.stability_margin(
        0.3 * a_crit, 0.3 * j_crit, s_edge
    ) > 0.0
    exterior_unstable = boundary.is_unstable(a_ext, j_ext, s_edge) and measured_margin < 0.0
    flag_consistent = boundary.is_unstable(a_ext, j_ext, s_edge) == (measured_margin < 0.0)

    return BoundaryCheck(
        on_boundary_margin_abs=abs(on_margin),
        margin_formula_rel_error=margin_rel,
        interior_stable=interior_stable,
        exterior_unstable=exterior_unstable,
        flag_margin_consistent=flag_consistent,
    )


@dataclass(frozen=True)
class CrashCheck:
    """ELM crash energy-conservation observations."""

    energy_loss_rel_error: float
    stored_energy_ratio_rel_error: float
    peak_heat_flux_rel_error: float
    profile_pedestal_drop_rel_error: float
    profile_core_unchanged: bool


def crash_checks(*, f_elm: float = 0.08) -> CrashCheck:
    """Check the Type-I ELM crash energy loss, stored-energy ratio, and profile drop."""
    model = ELMCrashModel(f_elm)
    t_ped, n_ped, w_ped, a_wet = 2.0, 5.0, 10.0, 2.0
    result = model.crash(t_ped, n_ped, w_ped, a_wet)

    energy_loss_rel = float(abs(result.delta_W_MJ - f_elm * w_ped) / (f_elm * w_ped))
    stored_ratio = float((result.n_ped_post * result.T_ped_post) / (n_ped * t_ped))
    stored_ratio_rel = float(abs(stored_ratio - (1.0 - f_elm)) / (1.0 - f_elm))
    peak_analytic = (f_elm * w_ped / a_wet) / (result.duration_ms * 1e-3)
    peak_rel = float(abs(result.peak_heat_flux_MW_m2 - peak_analytic) / peak_analytic)

    rho = np.linspace(0.0, 1.0, 51)
    te = 1.0 + 0.0 * rho
    ne = 2.0 + 0.0 * rho
    rho_ped = 0.9
    te_new, ne_new = model.apply_to_profiles(rho, te, ne, rho_ped)
    idx = int(np.searchsorted(rho, rho_ped))
    pedestal_product = (te_new[idx:] * ne_new[idx:]) / (te[idx:] * ne[idx:])
    pedestal_drop_rel = float(np.max(np.abs(pedestal_product - (1.0 - f_elm)))) / (1.0 - f_elm)
    core_unchanged = bool(np.array_equal(te_new[:idx], te[:idx]) and np.array_equal(ne_new[:idx], ne[:idx]))

    return CrashCheck(
        energy_loss_rel_error=energy_loss_rel,
        stored_energy_ratio_rel_error=stored_ratio_rel,
        peak_heat_flux_rel_error=peak_rel,
        profile_pedestal_drop_rel_error=pedestal_drop_rel,
        profile_core_unchanged=core_unchanged,
    )


@dataclass(frozen=True)
class ELMValidationResult:
    """Outcome of the ELM peeling-ballooning and crash validation."""

    ballooning_rel_error: float
    peeling_rel_error: float
    peeling_scaling: tuple[ScalingCheck, ...]
    max_peeling_scaling_rel_error: float
    boundary: BoundaryCheck
    crash: CrashCheck
    exact_tol: float
    ballooning_passed: bool
    peeling_passed: bool
    boundary_passed: bool
    crash_passed: bool
    passed: bool


def validate_elm_peeling_ballooning(*, config: ELMConfig | None = None, exact_tol: float = 1e-9) -> ELMValidationResult:
    """Validate the production ELM peeling-ballooning boundary and crash.

    The ballooning and peeling limits and their scalings, the elliptical boundary
    margin and flag consistency, and the ELM crash energy conservation must all
    hold to ``exact_tol``.
    """
    config = config or default_config()

    ballooning_err = ballooning_limit_rel_error(config)
    peeling_err = peeling_limit_rel_error(config)
    scaling = peeling_scaling_checks(config)
    max_scaling = max(check.rel_error for check in scaling)
    boundary = boundary_checks(config)
    crash = crash_checks()

    ballooning_passed = bool(ballooning_err < exact_tol)
    peeling_passed = bool(peeling_err < exact_tol and max_scaling < exact_tol)
    boundary_passed = bool(
        boundary.on_boundary_margin_abs < exact_tol
        and boundary.margin_formula_rel_error < exact_tol
        and boundary.interior_stable
        and boundary.exterior_unstable
        and boundary.flag_margin_consistent
    )
    crash_passed = bool(
        crash.energy_loss_rel_error < exact_tol
        and crash.stored_energy_ratio_rel_error < exact_tol
        and crash.peak_heat_flux_rel_error < exact_tol
        and crash.profile_pedestal_drop_rel_error < exact_tol
        and crash.profile_core_unchanged
    )

    passed = bool(ballooning_passed and peeling_passed and boundary_passed and crash_passed)
    return ELMValidationResult(
        ballooning_rel_error=ballooning_err,
        peeling_rel_error=peeling_err,
        peeling_scaling=scaling,
        max_peeling_scaling_rel_error=max_scaling,
        boundary=boundary,
        crash=crash,
        exact_tol=exact_tol,
        ballooning_passed=ballooning_passed,
        peeling_passed=peeling_passed,
        boundary_passed=boundary_passed,
        crash_passed=crash_passed,
        passed=passed,
    )


def build_evidence(result: ELMValidationResult, *, target_id: str) -> dict[str, Any]:
    """Build a tamper-evident, schema-versioned validation evidence payload."""
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    payload: dict[str, Any] = {
        "schema_version": ELM_PEELING_BALLOONING_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "target_id": target_id,
        "exact_tol": result.exact_tol,
        "ballooning_rel_error": result.ballooning_rel_error,
        "peeling_rel_error": result.peeling_rel_error,
        "peeling_scaling": [
            {
                "name": check.name,
                "measured_ratio": check.measured_ratio,
                "expected_ratio": check.expected_ratio,
                "rel_error": check.rel_error,
            }
            for check in result.peeling_scaling
        ],
        "max_peeling_scaling_rel_error": result.max_peeling_scaling_rel_error,
        "boundary": {
            "on_boundary_margin_abs": result.boundary.on_boundary_margin_abs,
            "margin_formula_rel_error": result.boundary.margin_formula_rel_error,
            "interior_stable": result.boundary.interior_stable,
            "exterior_unstable": result.boundary.exterior_unstable,
            "flag_margin_consistent": result.boundary.flag_margin_consistent,
        },
        "crash": {
            "energy_loss_rel_error": result.crash.energy_loss_rel_error,
            "stored_energy_ratio_rel_error": result.crash.stored_energy_ratio_rel_error,
            "peak_heat_flux_rel_error": result.crash.peak_heat_flux_rel_error,
            "profile_pedestal_drop_rel_error": result.crash.profile_pedestal_drop_rel_error,
            "profile_core_unchanged": result.crash.profile_core_unchanged,
        },
        "ballooning_passed": result.ballooning_passed,
        "peeling_passed": result.peeling_passed,
        "boundary_passed": result.boundary_passed,
        "crash_passed": result.crash_passed,
        "passed": result.passed,
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Return ``True`` when a payload is well-formed, sealed, and passing."""
    if payload.get("schema_version") != ELM_PEELING_BALLOONING_SCHEMA_VERSION:
        raise ValueError("unsupported elm peeling-ballooning evidence schema_version")
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
    boundary = evidence["boundary"]
    crash = evidence["crash"]
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "",
        "# ELM Peeling-Ballooning and Crash Validation",
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
        f"| ballooning alpha_crit | {evidence['ballooning_rel_error']:.3e} |",
        f"| peeling j_crit | {evidence['peeling_rel_error']:.3e} |",
        f"| peeling scaling laws (max) | {evidence['max_peeling_scaling_rel_error']:.3e} |",
        f"| elliptical on-boundary margin | {boundary['on_boundary_margin_abs']:.3e} |",
        f"| margin formula | {boundary['margin_formula_rel_error']:.3e} |",
        f"| interior stable / exterior unstable | {boundary['interior_stable']} / {boundary['exterior_unstable']} |",
        f"| ELM energy loss | {crash['energy_loss_rel_error']:.3e} |",
        f"| stored-energy ratio (1-f) | {crash['stored_energy_ratio_rel_error']:.3e} |",
        f"| peak heat flux | {crash['peak_heat_flux_rel_error']:.3e} |",
        f"| profile pedestal drop | {crash['profile_pedestal_drop_rel_error']:.3e} |",
        f"| profile core unchanged | {crash['profile_core_unchanged']} |",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point producing schema-versioned validation evidence."""
    parser = argparse.ArgumentParser(
        description="Validate the ELM peeling-ballooning boundary and crash against exact closed forms"
    )
    parser.add_argument("--target-id", type=str, default="local-elm-peeling-ballooning")
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    parser.add_argument("--report", type=str, default=None, help="write sealed JSON evidence and a Markdown summary")
    args = parser.parse_args(argv)

    result = validate_elm_peeling_ballooning()
    evidence = build_evidence(result, target_id=args.target_id)

    if args.report:
        _write_report(evidence, Path(args.report))

    if args.json_out:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print("ELM peeling-ballooning and crash validation")
        print(
            f"  PB limits:  ballooning={result.ballooning_rel_error:.3e} "
            f"peeling={result.peeling_rel_error:.3e} scaling={result.max_peeling_scaling_rel_error:.3e} "
            f"{'ok' if result.ballooning_passed and result.peeling_passed else 'FAIL'}"
        )
        print(
            f"  boundary:   on_ellipse={result.boundary.on_boundary_margin_abs:.3e} "
            f"margin={result.boundary.margin_formula_rel_error:.3e} "
            f"{'ok' if result.boundary_passed else 'FAIL'}"
        )
        print(
            f"  crash:      W_post/(1-f)={result.crash.stored_energy_ratio_rel_error:.3e} "
            f"dW={result.crash.energy_loss_rel_error:.3e} "
            f"{'ok' if result.crash_passed else 'FAIL'}"
        )
        print(f"Status: {'pass' if result.passed else 'fail'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

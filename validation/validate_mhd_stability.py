#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Ideal-MHD stability metric analytic validation
"""Validate the ideal-MHD stability metrics against exact closed forms.

The ideal-MHD stability model (``src/scpn_control/core/stability_mhd.py``)
evaluates the Troyon normalised-beta limit, the Mercier interchange index, the
first ballooning boundary, and the Kruskal-Shafranov external-kink criterion.
Every metric is an exact algebraic closed form, so the model can be validated
without any external MHD stability code — the validation is fully self-contained.

Exact references checked against the production methods:

1. **Troyon beta limit.** ``beta_N = 100 beta_t a B0 / Ip`` with the exact
   ``beta_t``, ``a``, ``B0``, and ``1/Ip`` scaling, and the no-wall/ideal-wall
   stability boundaries ``beta_N < g``.
2. **Mercier interchange index.** ``D_M = s(s - 1) + alpha(1 - s/2)`` (Freidberg)
   with stability where ``D_M >= 0``, checked against hand-evaluated marginal
   cases.
3. **Ballooning first-stability boundary.** ``alpha_crit = s(1 - s/2)`` for
   ``s < 1`` and ``0.6 s`` for ``s >= 1`` (Connor-Hastie-Taylor), with stability
   where ``alpha <= alpha_crit``.
4. **Kruskal-Shafranov criterion.** External-kink stability iff ``q_edge > 1``,
   with margin ``q_edge - 1``.

References:
  Troyon F. et al. (1984) *Plasma Phys. Control. Fusion* 26, 209.
  Freidberg J. P. (2014) *Ideal MHD*, Cambridge University Press, Ch. 12.
  Connor J. W., Hastie R. J., Taylor J. B. (1978) *Phys. Rev. Lett.* 40, 396.
  Kruskal M. D., Schwarzschild M. (1954) *Proc. R. Soc. Lond. A* 223, 348.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from scpn_control.core.stability_mhd import (
    QProfile,
    ballooning_stability,
    kruskal_shafranov_stability,
    mercier_stability,
    troyon_beta_limit,
)

FloatArray = NDArray[np.float64]

MHD_STABILITY_SCHEMA_VERSION = "scpn-control.mhd-stability-validation.v1"


def _controlled_profile(shear: FloatArray, alpha: FloatArray) -> QProfile:
    """Build a QProfile with prescribed magnetic shear and normalised pressure gradient."""
    n = len(shear)
    rho = np.linspace(0.0, 1.0, n, dtype=np.float64)
    q = np.asarray(1.0 + rho**2, dtype=np.float64)
    return QProfile(
        rho=rho,
        q=q,
        shear=np.asarray(shear, dtype=np.float64),
        alpha_mhd=np.asarray(alpha, dtype=np.float64),
        q_min=1.0,
        q_min_rho=0.0,
        q_edge=2.0,
    )


def troyon_rel_error(*, beta_t: float = 0.025, ip_ma: float = 1.0, a: float = 0.5, b0: float = 2.0) -> float:
    """Relative error of the Troyon ``beta_N`` against ``100 beta_t a B0 / Ip``."""
    measured = troyon_beta_limit(beta_t, ip_ma, a, b0).beta_N
    analytic = 100.0 * beta_t * a * b0 / ip_ma
    return abs(measured - analytic) / analytic


@dataclass(frozen=True)
class ScalingCheck:
    """One Troyon-scaling-law observation."""

    name: str
    measured_ratio: float
    expected_ratio: float
    rel_error: float


def troyon_scaling_checks() -> tuple[ScalingCheck, ...]:
    """Verify ``beta_N`` scales linearly with ``beta_t``, ``a``, ``B0`` and as ``1/Ip``."""
    base = troyon_beta_limit(0.025, 1.0, 0.5, 2.0).beta_N
    specs = (
        ("beta_t_linear", troyon_beta_limit(0.05, 1.0, 0.5, 2.0).beta_N / base, 2.0),
        ("minor_radius_linear", troyon_beta_limit(0.025, 1.0, 1.0, 2.0).beta_N / base, 2.0),
        ("field_linear", troyon_beta_limit(0.025, 1.0, 0.5, 4.0).beta_N / base, 2.0),
        ("current_inverse", troyon_beta_limit(0.025, 2.0, 0.5, 2.0).beta_N / base, 0.5),
    )
    return tuple(
        ScalingCheck(
            name=name, measured_ratio=ratio, expected_ratio=expected, rel_error=abs(ratio - expected) / expected
        )
        for name, ratio, expected in specs
    )


def troyon_boundary_is_consistent() -> bool:
    """Stability flags must agree with ``beta_N < g`` at and around the no-wall limit."""
    # Choose beta_t so beta_N sits just below / above g_nowall = 2.8 with a = B0 = Ip = 1.
    below = troyon_beta_limit(beta_t=0.027, Ip_MA=1.0, a=1.0, B0=1.0)  # beta_N = 2.7
    above = troyon_beta_limit(beta_t=0.030, Ip_MA=1.0, a=1.0, B0=1.0)  # beta_N = 3.0
    return bool(below.stable_nowall and below.stable_wall and (not above.stable_nowall) and above.stable_wall)


def mercier_rel_error() -> float:
    """Relative error of the Mercier index against ``s(s-1) + alpha(1 - s/2)``."""
    shear = np.linspace(0.0, 2.0, 41).astype(np.float64)
    alpha = np.full_like(shear, 0.3)
    measured = mercier_stability(_controlled_profile(shear, alpha)).D_M
    analytic = shear * (shear - 1.0) + alpha * (1.0 - shear / 2.0)
    denom = np.maximum(np.abs(analytic), 1e-12)
    return float(np.max(np.abs(measured - analytic) / denom))


def mercier_marginal_cases_consistent() -> bool:
    """Hand-evaluated Mercier marginal cases must match the production stability flags."""
    cases = ((0.5, 0.0, False), (2.0, 0.0, True), (0.5, 1.0, True), (1.5, 0.0, True))
    shear = np.array([c[0] for c in cases], dtype=np.float64)
    alpha = np.array([c[1] for c in cases], dtype=np.float64)
    result = mercier_stability(_controlled_profile(shear, alpha))
    return all(bool(result.stable[i]) == expected for i, (_, _, expected) in enumerate(cases))


def ballooning_rel_error() -> float:
    """Relative error of the ballooning ``alpha_crit`` against the Connor-Hastie-Taylor form."""
    shear = np.linspace(0.0, 2.0, 41).astype(np.float64)
    alpha = np.full_like(shear, 0.3)
    measured = ballooning_stability(_controlled_profile(shear, alpha)).alpha_crit
    analytic = np.maximum(np.where(shear < 1.0, shear * (1.0 - shear / 2.0), 0.6 * shear), 0.0)
    denom = np.maximum(np.abs(analytic), 1e-12)
    return float(np.max(np.abs(measured - analytic) / denom))


def ballooning_branches_consistent() -> bool:
    """The low-shear and high-shear ballooning branches must match their closed forms."""
    shear = np.array([0.5, 0.8, 1.2, 1.8], dtype=np.float64)
    alpha = np.full_like(shear, 0.1)
    result = ballooning_stability(_controlled_profile(shear, alpha))
    expected = np.where(shear < 1.0, shear * (1.0 - shear / 2.0), 0.6 * shear)
    return bool(np.allclose(result.alpha_crit, expected, rtol=0.0, atol=1e-12))


def kruskal_shafranov_consistent() -> bool:
    """External-kink stability must hold exactly for ``q_edge > 1`` and fail below it."""
    rho = np.linspace(0.0, 1.0, 11).astype(np.float64)
    shear = np.full_like(rho, 1.0)
    alpha = np.full_like(rho, 0.1)
    stable_profile = QProfile(rho, 1.0 + rho**2, shear, alpha, 1.0, 0.0, 2.0)
    unstable_profile = QProfile(rho, 0.5 + 0.4 * rho**2, shear, alpha, 0.5, 0.0, 0.9)
    stable = kruskal_shafranov_stability(stable_profile)
    unstable = kruskal_shafranov_stability(unstable_profile)
    return bool(
        stable.stable
        and abs(stable.margin - 1.0) < 1e-12
        and (not unstable.stable)
        and abs(unstable.margin - (-0.1)) < 1e-12
    )


@dataclass(frozen=True)
class MHDStabilityValidationResult:
    """Outcome of the ideal-MHD stability metric validation."""

    troyon_rel_error: float
    troyon_scaling: tuple[ScalingCheck, ...]
    max_troyon_scaling_rel_error: float
    troyon_boundary_consistent: bool
    mercier_rel_error: float
    mercier_cases_consistent: bool
    ballooning_rel_error: float
    ballooning_branches_consistent: bool
    kruskal_shafranov_consistent: bool
    exact_tol: float
    troyon_passed: bool
    mercier_passed: bool
    ballooning_passed: bool
    kruskal_shafranov_passed: bool
    passed: bool


def validate_mhd_stability(*, exact_tol: float = 1e-9) -> MHDStabilityValidationResult:
    """Validate the production ideal-MHD stability metrics against their exact forms.

    The Troyon beta limit and its scalings, the Mercier interchange index, the
    ballooning first-stability boundary, and the Kruskal-Shafranov criterion must
    all reproduce their closed forms to ``exact_tol`` with consistent stability
    flags.
    """
    troyon_err = troyon_rel_error()
    troyon_scaling = troyon_scaling_checks()
    max_troyon_scaling = max(check.rel_error for check in troyon_scaling)
    troyon_boundary = troyon_boundary_is_consistent()
    mercier_err = mercier_rel_error()
    mercier_cases = mercier_marginal_cases_consistent()
    ballooning_err = ballooning_rel_error()
    ballooning_branches = ballooning_branches_consistent()
    ks_consistent = kruskal_shafranov_consistent()

    troyon_passed = troyon_err < exact_tol and max_troyon_scaling < exact_tol and troyon_boundary
    mercier_passed = mercier_err < exact_tol and mercier_cases
    ballooning_passed = ballooning_err < exact_tol and ballooning_branches
    kruskal_shafranov_passed = ks_consistent

    passed = troyon_passed and mercier_passed and ballooning_passed and kruskal_shafranov_passed
    return MHDStabilityValidationResult(
        troyon_rel_error=troyon_err,
        troyon_scaling=troyon_scaling,
        max_troyon_scaling_rel_error=max_troyon_scaling,
        troyon_boundary_consistent=troyon_boundary,
        mercier_rel_error=mercier_err,
        mercier_cases_consistent=mercier_cases,
        ballooning_rel_error=ballooning_err,
        ballooning_branches_consistent=ballooning_branches,
        kruskal_shafranov_consistent=ks_consistent,
        exact_tol=exact_tol,
        troyon_passed=troyon_passed,
        mercier_passed=mercier_passed,
        ballooning_passed=ballooning_passed,
        kruskal_shafranov_passed=kruskal_shafranov_passed,
        passed=passed,
    )


def build_evidence(result: MHDStabilityValidationResult, *, target_id: str) -> dict[str, Any]:
    """Build a tamper-evident, schema-versioned validation evidence payload."""
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    payload: dict[str, Any] = {
        "schema_version": MHD_STABILITY_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "target_id": target_id,
        "exact_tol": result.exact_tol,
        "troyon_rel_error": result.troyon_rel_error,
        "troyon_scaling": [
            {
                "name": check.name,
                "measured_ratio": check.measured_ratio,
                "expected_ratio": check.expected_ratio,
                "rel_error": check.rel_error,
            }
            for check in result.troyon_scaling
        ],
        "max_troyon_scaling_rel_error": result.max_troyon_scaling_rel_error,
        "troyon_boundary_consistent": result.troyon_boundary_consistent,
        "mercier_rel_error": result.mercier_rel_error,
        "mercier_cases_consistent": result.mercier_cases_consistent,
        "ballooning_rel_error": result.ballooning_rel_error,
        "ballooning_branches_consistent": result.ballooning_branches_consistent,
        "kruskal_shafranov_consistent": result.kruskal_shafranov_consistent,
        "troyon_passed": result.troyon_passed,
        "mercier_passed": result.mercier_passed,
        "ballooning_passed": result.ballooning_passed,
        "kruskal_shafranov_passed": result.kruskal_shafranov_passed,
        "passed": result.passed,
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Return ``True`` when a payload is well-formed, sealed, and passing."""
    if payload.get("schema_version") != MHD_STABILITY_SCHEMA_VERSION:
        raise ValueError("unsupported mhd stability evidence schema_version")
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
        "# Ideal-MHD Stability Metric Validation",
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
        f"| Troyon beta_N = 100 beta_t a B0 / Ip | {evidence['troyon_rel_error']:.3e} |",
        f"| Troyon scaling laws (max) | {evidence['max_troyon_scaling_rel_error']:.3e} |",
        f"| Troyon boundary flags consistent | {evidence['troyon_boundary_consistent']} |",
        f"| Mercier D_M = s(s-1) + alpha(1-s/2) | {evidence['mercier_rel_error']:.3e} |",
        f"| Mercier marginal cases consistent | {evidence['mercier_cases_consistent']} |",
        f"| ballooning alpha_crit (Connor-Hastie-Taylor) | {evidence['ballooning_rel_error']:.3e} |",
        f"| ballooning branch closed forms | {evidence['ballooning_branches_consistent']} |",
        f"| Kruskal-Shafranov q_edge > 1 | {evidence['kruskal_shafranov_consistent']} |",
        "",
        "## Troyon scaling exponents",
        "",
        "| law | measured ratio | expected | rel error |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {check['name']} | {check['measured_ratio']:.6f} | {check['expected_ratio']:.1f} | {check['rel_error']:.3e} |"
        for check in evidence["troyon_scaling"]
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point producing schema-versioned validation evidence."""
    parser = argparse.ArgumentParser(description="Validate the ideal-MHD stability metrics against exact closed forms")
    parser.add_argument("--target-id", type=str, default="local-mhd-stability")
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    parser.add_argument("--report", type=str, default=None, help="write sealed JSON evidence and a Markdown summary")
    args = parser.parse_args(argv)

    result = validate_mhd_stability()
    evidence = build_evidence(result, target_id=args.target_id)

    if args.report:
        _write_report(evidence, Path(args.report))

    if args.json_out:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print("Ideal-MHD stability metric validation")
        print(
            f"  Troyon:           beta_N={result.troyon_rel_error:.3e} "
            f"scaling={result.max_troyon_scaling_rel_error:.3e} "
            f"boundary={result.troyon_boundary_consistent} "
            f"{'ok' if result.troyon_passed else 'FAIL'}"
        )
        print(
            f"  Mercier:          D_M={result.mercier_rel_error:.3e} "
            f"cases={result.mercier_cases_consistent} "
            f"{'ok' if result.mercier_passed else 'FAIL'}"
        )
        print(
            f"  ballooning:       alpha_crit={result.ballooning_rel_error:.3e} "
            f"branches={result.ballooning_branches_consistent} "
            f"{'ok' if result.ballooning_passed else 'FAIL'}"
        )
        print(
            f"  Kruskal-Shafranov: q_edge>1 {result.kruskal_shafranov_consistent} "
            f"{'ok' if result.kruskal_shafranov_passed else 'FAIL'}"
        )
        print(f"Status: {'pass' if result.passed else 'fail'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

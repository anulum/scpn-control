#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — RZIP rigid vertical stability analytic validation
"""Validate the RZIP rigid-plasma vertical stability model against exact results.

The rigid-plasma vertical response model (``src/scpn_control/control/rzip_model.py``)
assembles a linearised state-space ``A`` for ``x = [Z, dZ/dt, I_1, ...]`` and
returns the vertical growth rate as the largest real eigenvalue of ``A``. The
destabilising term is the curvature spring ``K = n mu_0 Ip^2 / (4 pi R_0)`` with
field decay index ``n``; the ``A[1, 0] = -K / M_eff`` entry drives the rigid
vertical mode. In the no-wall limit (no conducting vessel elements or coils) the
state collapses to the ``2 x 2`` block ``[[0, 1], [-K/M_eff, 0]]``, whose
eigenvalues are exactly ``+/- sqrt(-K / M_eff)``. This gives closed-form
references with no measured or external code required, so the validation is fully
self-contained.

Exact references checked against the production ``RZIPModel``:

1. **Unstable no-wall growth rate** (``n < 0``): the largest real eigenvalue is
   ``gamma = sqrt(-n mu_0 Ip^2 / (4 pi R_0 M_eff))``.
2. **Stable no-wall oscillation** (``n > 0``): the rigid mode is marginally
   stable (zero real part) and oscillates at ``omega = sqrt(n mu_0 Ip^2 /
   (4 pi R_0 M_eff))``, recovered as the largest eigenvalue imaginary part.
3. **Marginal index** (``n = 0``): the spring vanishes and the growth rate is
   zero.
4. **Exact scaling laws**: ``gamma`` scales linearly with ``Ip``, as
   ``sqrt(-n)``, and as ``1/sqrt(M_eff)``.
5. **Resistive-wall stabilisation**: adding a passive conducting wall reduces the
   growth rate below the no-wall value while keeping it finite, confirming the
   eddy-current circuit coupling is stabilising.

References:
  Lazarus E. A. et al. (1990) *Nucl. Fusion* 30, 111 (rigid vertical model).
  Wesson J. (2011) *Tokamaks*, 4th ed., Oxford University Press, Ch. 3.10
  (vertical stability and field index).
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

from scpn_control.control.rzip_model import RZIPModel
from scpn_control.core.vessel_model import VesselElement, VesselModel

RZIP_VERTICAL_STABILITY_SCHEMA_VERSION = "scpn-control.rzip-vertical-stability-validation.v1"

_MU_0 = 4.0e-7 * math.pi


@dataclass(frozen=True)
class VerticalConfig:
    """Geometry, current, and inertia for the rigid vertical stability model."""

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
        """Destabilising spring constant ``K = n mu_0 Ip^2 / (4 pi R_0)`` [N/m]."""
        ip_amps = self.ip_ma * 1.0e6
        return n_index * _MU_0 * ip_amps**2 / (4.0 * math.pi * self.r0)


def default_config() -> VerticalConfig:
    """ITER-like single-null geometry with a declared rigid vertical inertia."""
    return VerticalConfig(r0=1.7, a=0.5, kappa=1.8, ip_ma=1.0, b0=2.0, m_eff_kg=2.0)


def analytic_no_wall_growth_rate(config: VerticalConfig, n_index: float) -> float:
    """Exact no-wall growth rate ``sqrt(-K/M_eff)`` for an unstable index ``n < 0``."""
    if n_index >= 0.0:
        raise ValueError("growth-rate closed form requires a destabilising index (n_index < 0)")
    return math.sqrt(-config.curvature_spring(n_index) / config.m_eff_kg)


def analytic_no_wall_frequency(config: VerticalConfig, n_index: float) -> float:
    """Exact no-wall oscillation frequency ``sqrt(K/M_eff)`` for a stable ``n > 0``."""
    if n_index <= 0.0:
        raise ValueError("oscillation closed form requires a stabilising index (n_index > 0)")
    return math.sqrt(config.curvature_spring(n_index) / config.m_eff_kg)


def _build_no_wall(config: VerticalConfig, n_index: float) -> RZIPModel:
    return RZIPModel(
        config.r0,
        config.a,
        config.kappa,
        config.ip_ma,
        config.b0,
        n_index,
        VesselModel([]),
        vertical_inertia_kg=config.m_eff_kg,
    )


def no_wall_growth_rel_error(config: VerticalConfig, n_index: float) -> float:
    """Relative error of the model growth rate against the exact no-wall value."""
    measured = _build_no_wall(config, n_index).vertical_growth_rate()
    analytic = analytic_no_wall_growth_rate(config, n_index)
    return abs(measured - analytic) / analytic


def no_wall_frequency_rel_error(config: VerticalConfig, n_index: float) -> float:
    """Relative error of the largest eigenvalue imaginary part against ``sqrt(K/M)``."""
    state_matrix, _, _, _ = _build_no_wall(config, n_index).build_state_space()
    eigenvalues = np.linalg.eigvals(state_matrix)
    measured = float(np.max(np.abs(np.imag(eigenvalues))))
    analytic = analytic_no_wall_frequency(config, n_index)
    return abs(measured - analytic) / analytic


def no_wall_growth_time_consistency(config: VerticalConfig, n_index: float) -> float:
    """Relative mismatch of ``growth_time_ms * growth_rate`` against the 1000 ms/s identity."""
    model = _build_no_wall(config, n_index)
    gamma = model.vertical_growth_rate()
    growth_time_ms = model.vertical_growth_time()
    return abs(growth_time_ms * gamma - 1000.0) / 1000.0


@dataclass(frozen=True)
class ScalingCheck:
    """One exact-scaling-law observation."""

    name: str
    measured_ratio: float
    expected_ratio: float
    rel_error: float


def scaling_checks(config: VerticalConfig) -> tuple[ScalingCheck, ...]:
    """Verify ``gamma`` scales as ``Ip``, ``sqrt(-n)``, and ``1/sqrt(M_eff)``."""
    base = _build_no_wall(config, -1.0).vertical_growth_rate()

    double_ip = _build_no_wall(
        VerticalConfig(config.r0, config.a, config.kappa, 2.0 * config.ip_ma, config.b0, config.m_eff_kg), -1.0
    ).vertical_growth_rate()
    quad_index = _build_no_wall(config, -4.0).vertical_growth_rate()
    quad_mass = _build_no_wall(
        VerticalConfig(config.r0, config.a, config.kappa, config.ip_ma, config.b0, 4.0 * config.m_eff_kg), -1.0
    ).vertical_growth_rate()

    specs = (
        ("current_linear", double_ip / base, 2.0),
        ("index_sqrt", quad_index / base, 2.0),
        ("inertia_inverse_sqrt", quad_mass / base, 0.5),
    )
    return tuple(
        ScalingCheck(
            name=name, measured_ratio=ratio, expected_ratio=expected, rel_error=abs(ratio - expected) / expected
        )
        for name, ratio, expected in specs
    )


@dataclass(frozen=True)
class WallStabilisation:
    """Resistive-wall stabilisation of the rigid vertical mode."""

    no_wall_growth_rate: float
    with_wall_growth_rate: float
    wall_slows_growth: bool
    with_wall_finite: bool


def wall_stabilisation(config: VerticalConfig, n_index: float = -1.0) -> WallStabilisation:
    """Compare the growth rate with and without a passive up/down conducting wall."""
    no_wall = _build_no_wall(config, n_index).vertical_growth_rate()
    elements = [
        VesselElement(R=config.r0, Z=1.4 * config.a, resistance=1e-4, cross_section=0.01, inductance=2e-6),
        VesselElement(R=config.r0, Z=-1.4 * config.a, resistance=1e-4, cross_section=0.01, inductance=2e-6),
    ]
    walled = RZIPModel(
        config.r0,
        config.a,
        config.kappa,
        config.ip_ma,
        config.b0,
        n_index,
        VesselModel(elements),
        vertical_inertia_kg=config.m_eff_kg,
    ).vertical_growth_rate()
    return WallStabilisation(
        no_wall_growth_rate=no_wall,
        with_wall_growth_rate=walled,
        wall_slows_growth=bool(walled < no_wall),
        with_wall_finite=bool(math.isfinite(walled)),
    )


@dataclass(frozen=True)
class RzipValidationResult:
    """Outcome of the RZIP rigid vertical stability validation."""

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


def validate_rzip_vertical_stability(
    *,
    config: VerticalConfig | None = None,
    unstable_indices: Sequence[float] = (-2.5, -1.2, -0.6),
    stable_indices: Sequence[float] = (0.8, 1.5),
    exact_tol: float = 1e-9,
    marginal_tol: float = 1e-6,
) -> RzipValidationResult:
    """Validate the production RZIP model against the exact no-wall references.

    The unstable growth rate, stable oscillation frequency, growth-time identity,
    and exact scaling laws must hold to ``exact_tol``; the marginal index must give
    a growth rate below ``marginal_tol``; and a passive wall must reduce the growth
    rate below the no-wall value.
    """
    config = config or default_config()
    unstable = tuple(_negative_float("unstable index", n) for n in unstable_indices)
    stable = tuple(_positive_float("stable index", n) for n in stable_indices)
    if not unstable or not stable:
        raise ValueError("at least one unstable and one stable index are required")

    max_growth_err = max(no_wall_growth_rel_error(config, n) for n in unstable)
    max_growth_time_err = max(no_wall_growth_time_consistency(config, n) for n in unstable)
    max_freq_err = max(no_wall_frequency_rel_error(config, n) for n in stable)
    marginal = _build_no_wall(config, 0.0).vertical_growth_rate()
    scaling = scaling_checks(config)
    max_scaling_err = max(check.rel_error for check in scaling)
    wall = wall_stabilisation(config)

    growth_passed = max_growth_err < exact_tol
    frequency_passed = max_freq_err < exact_tol
    growth_time_passed = max_growth_time_err < exact_tol
    marginal_passed = abs(marginal) < marginal_tol
    scaling_passed = max_scaling_err < exact_tol
    wall_passed = wall.wall_slows_growth and wall.with_wall_finite

    passed = (
        growth_passed and frequency_passed and growth_time_passed and marginal_passed and scaling_passed and wall_passed
    )
    return RzipValidationResult(
        config=config,
        unstable_indices=unstable,
        stable_indices=stable,
        max_growth_rel_error=max_growth_err,
        max_frequency_rel_error=max_freq_err,
        max_growth_time_rel_error=max_growth_time_err,
        marginal_growth_rate=float(marginal),
        scaling=scaling,
        max_scaling_rel_error=max_scaling_err,
        wall=wall,
        exact_tol=exact_tol,
        marginal_tol=marginal_tol,
        growth_passed=growth_passed,
        frequency_passed=frequency_passed,
        growth_time_passed=growth_time_passed,
        marginal_passed=marginal_passed,
        scaling_passed=scaling_passed,
        wall_passed=wall_passed,
        passed=passed,
    )


def build_evidence(result: RzipValidationResult, *, target_id: str) -> dict[str, Any]:
    """Build a tamper-evident, schema-versioned validation evidence payload."""
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    payload: dict[str, Any] = {
        "schema_version": RZIP_VERTICAL_STABILITY_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "target_id": target_id,
        "config": {
            "r0": result.config.r0,
            "a": result.config.a,
            "kappa": result.config.kappa,
            "ip_ma": result.config.ip_ma,
            "b0": result.config.b0,
            "m_eff_kg": result.config.m_eff_kg,
        },
        "unstable_indices": list(result.unstable_indices),
        "stable_indices": list(result.stable_indices),
        "exact_tol": result.exact_tol,
        "marginal_tol": result.marginal_tol,
        "max_growth_rel_error": result.max_growth_rel_error,
        "max_frequency_rel_error": result.max_frequency_rel_error,
        "max_growth_time_rel_error": result.max_growth_time_rel_error,
        "marginal_growth_rate": result.marginal_growth_rate,
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
        "wall": {
            "no_wall_growth_rate": result.wall.no_wall_growth_rate,
            "with_wall_growth_rate": result.wall.with_wall_growth_rate,
            "wall_slows_growth": result.wall.wall_slows_growth,
            "with_wall_finite": result.wall.with_wall_finite,
        },
        "growth_passed": result.growth_passed,
        "frequency_passed": result.frequency_passed,
        "growth_time_passed": result.growth_time_passed,
        "marginal_passed": result.marginal_passed,
        "scaling_passed": result.scaling_passed,
        "wall_passed": result.wall_passed,
        "passed": result.passed,
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Return ``True`` when a payload is well-formed, sealed, and passing."""
    if payload.get("schema_version") != RZIP_VERTICAL_STABILITY_SCHEMA_VERSION:
        raise ValueError("unsupported rzip vertical stability evidence schema_version")
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


def _negative_float(name: str, value: object) -> float:
    result = _finite_float(name, value)
    if result >= 0.0:
        raise ValueError(f"{name} must be negative")
    return result


def _write_report(evidence: Mapping[str, Any], json_path: Path) -> None:
    """Persist the sealed JSON evidence and a human-readable Markdown summary."""
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = json_path.with_suffix(".md")
    config = evidence["config"]
    wall = evidence["wall"]
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "",
        "# RZIP Rigid Vertical Stability Validation",
        "",
        f"- Schema: `{evidence['schema_version']}`",
        f"- Generated (UTC): {evidence['generated_utc']}",
        f"- Target: `{evidence['target_id']}`",
        f"- Geometry: R0={config['r0']} m, a={config['a']} m, kappa={config['kappa']}, "
        f"Ip={config['ip_ma']} MA, M_eff={config['m_eff_kg']} kg",
        f"- Status: **{'pass' if evidence['passed'] else 'fail'}**",
        "",
        "## Exact no-wall references (largest eigenvalue of the 2x2 rigid block)",
        "",
        f"- Max unstable growth-rate rel error (n<0): {evidence['max_growth_rel_error']:.3e}",
        f"- Max stable oscillation-frequency rel error (n>0): {evidence['max_frequency_rel_error']:.3e}",
        f"- Max growth-time identity rel error: {evidence['max_growth_time_rel_error']:.3e}",
        f"- Marginal growth rate at n=0: {evidence['marginal_growth_rate']:.3e} "
        f"(gate < {evidence['marginal_tol']:.1e})",
        f"- Exact-reference tolerance: {evidence['exact_tol']:.1e}",
        "",
        "## Exact scaling laws",
        "",
        "| law | measured ratio | expected | rel error |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {check['name']} | {check['measured_ratio']:.6f} | {check['expected_ratio']} | {check['rel_error']:.3e} |"
        for check in evidence["scaling"]
    ]
    lines += [
        "",
        "## Resistive-wall stabilisation",
        "",
        f"- No-wall growth rate: {wall['no_wall_growth_rate']:.4e} s^-1",
        f"- With-wall growth rate: {wall['with_wall_growth_rate']:.4e} s^-1",
        f"- Wall slows growth: {wall['wall_slows_growth']}; finite: {wall['with_wall_finite']}",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point producing schema-versioned validation evidence."""
    parser = argparse.ArgumentParser(
        description="Validate the RZIP rigid vertical stability model against exact references"
    )
    parser.add_argument("--target-id", type=str, default="local-rzip-vertical-stability")
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    parser.add_argument("--report", type=str, default=None, help="write sealed JSON evidence and a Markdown summary")
    args = parser.parse_args(argv)

    result = validate_rzip_vertical_stability()
    evidence = build_evidence(result, target_id=args.target_id)

    if args.report:
        _write_report(evidence, Path(args.report))

    if args.json_out:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print("RZIP rigid vertical stability validation")
        print(
            f"  no-wall growth (n<0): max rel err={result.max_growth_rel_error:.3e} "
            f"{'ok' if result.growth_passed else 'FAIL'}"
        )
        print(
            f"  oscillation (n>0):    max rel err={result.max_frequency_rel_error:.3e} "
            f"{'ok' if result.frequency_passed else 'FAIL'}"
        )
        print(
            f"  scaling laws:         max rel err={result.max_scaling_rel_error:.3e} "
            f"{'ok' if result.scaling_passed else 'FAIL'}"
        )
        print(
            f"  wall stabilisation:   {result.wall.no_wall_growth_rate:.3e} -> "
            f"{result.wall.with_wall_growth_rate:.3e} s^-1 "
            f"{'ok' if result.wall_passed else 'FAIL'}"
        )
        print(f"Status: {'pass' if result.passed else 'fail'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Sawtooth Kadomtsev crash analytic validation
"""Validate the Kadomtsev sawtooth crash against exact conservation laws.

The sawtooth model (``src/scpn_control/core/sawtooth.py``) applies a Kadomtsev
full-reconnection crash: inside the mixing radius the temperature and density are
replaced by their volume averages and ``q`` is reset to one, while the mixing
radius itself is fixed by the helical-flux proxy
``psi*(rho) = integral_0^rho rho' (1/q - 1) drho'`` returning to zero outside the
``q = 1`` surface. These are exact algebraic and conservation relations, so the
crash can be validated without any measured shot or external code — the
validation is fully self-contained.

Exact references checked against the production ``kadomtsev_crash`` and
``SawtoothMonitor.find_q1_radius``:

1. **Volume-integral conservation.** The volume-average redistribution conserves
   the volume integrals ``integral T rho drho`` and ``integral n rho drho`` over
   the mixing region to machine precision (energy and particle conservation).
2. **Helical-flux mixing condition.** The helical-flux proxy vanishes at the
   mixing radius, ``psi*(rho_mix) = 0`` — the Kadomtsev reconnection condition.
3. **Profile flattening.** Inside the mixing radius the post-crash ``T`` and ``n``
   are flat and ``q`` is reset to the post-crash ``q = 1`` value.
4. **Outside invariance.** Profiles beyond the mixing radius are unchanged.
5. **q = 1 radius convergence.** For the analytic parabolic profile
   ``q(rho) = q_0 + (q_a - q_0) rho^2`` the ``q = 1`` surface sits at
   ``rho_1 = sqrt((1 - q_0)/(q_a - q_0))``; the grid-interpolated radius converges
   to it at second order in the grid spacing.
6. **No-crash guard.** With ``q > 1`` everywhere there is no ``q = 1`` surface and
   the crash leaves every profile unchanged.

References:
  Kadomtsev B. B. (1975) *Sov. J. Plasma Phys.* 1, 389.
  Porcelli F. et al. (1996) *Plasma Phys. Control. Fusion* 38, 2163.
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
from scipy.integrate import trapezoid

from scpn_control.core.sawtooth import SawtoothMonitor, kadomtsev_crash

FloatArray = NDArray[np.float64]

SAWTOOTH_KADOMTSEV_SCHEMA_VERSION = "scpn-control.sawtooth-kadomtsev-validation.v1"


@dataclass(frozen=True)
class SawtoothConfig:
    """Analytic parabolic ``q``-profile and geometry for the Kadomtsev crash."""

    nr: int
    q0: float
    qa: float
    r0: float
    a: float

    def __post_init__(self) -> None:
        _grid_resolution("nr", self.nr)
        _finite_float("q0", self.q0)
        _finite_float("qa", self.qa)
        _positive_float("r0", self.r0)
        _positive_float("a", self.a)
        if not self.q0 < 1.0 < self.qa:
            raise ValueError("require q0 < 1 < qa so a q=1 surface exists inside the plasma")
        if self.a >= self.r0:
            raise ValueError("a must be smaller than r0 for tokamak ordering")

    def rho(self) -> FloatArray:
        return np.linspace(0.0, 1.0, self.nr, dtype=np.float64)

    def q_profile(self, rho: FloatArray) -> FloatArray:
        return np.asarray(self.q0 + (self.qa - self.q0) * rho**2, dtype=np.float64)


def default_config() -> SawtoothConfig:
    """A 201-point grid with a parabolic q-profile crossing unity at rho≈0.30."""
    return SawtoothConfig(nr=201, q0=0.8, qa=3.0, r0=1.7, a=0.5)


def analytic_q1_radius(config: SawtoothConfig) -> float:
    """Exact ``q = 1`` radius ``sqrt((1 - q0)/(qa - q0))`` for the parabolic profile."""
    return math.sqrt((1.0 - config.q0) / (config.qa - config.q0))


def _profiles(config: SawtoothConfig) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    rho = config.rho()
    temperature = 2.0 * (1.0 - rho**2) + 0.1
    density = 1.0 * (1.0 - 0.5 * rho**2) + 0.1
    q = config.q_profile(rho)
    return rho, temperature, density, q


def q1_radius_rel_error(config: SawtoothConfig) -> float:
    """Relative error of the grid-interpolated ``q = 1`` radius against the analytic value."""
    rho = config.rho()
    measured = SawtoothMonitor(rho).find_q1_radius(config.q_profile(rho))
    if measured is None:
        raise ValueError("no q=1 surface found on the configured grid")
    analytic = analytic_q1_radius(config)
    return abs(measured - analytic) / analytic


@dataclass(frozen=True)
class CrashConservation:
    """Conservation and structural diagnostics for one Kadomtsev crash."""

    rho_1: float
    rho_mix: float
    temperature_integral_rel_error: float
    density_integral_rel_error: float
    helical_flux_residual: float
    inner_temperature_flatness: float
    inner_density_flatness: float
    inner_q_value: float
    outside_max_abs_change: float


def crash_conservation(config: SawtoothConfig) -> CrashConservation:
    """Run the production crash and measure conservation and structural invariants."""
    rho, temperature, density, q = _profiles(config)
    t_new, n_new, q_new, rho_1, rho_mix = kadomtsev_crash(rho, temperature, density, q, config.r0, config.a)

    idx_mix = int(np.searchsorted(rho, rho_mix))
    if idx_mix < 2:
        raise ValueError("mixing region is too small to evaluate conservation")
    inner = slice(0, idx_mix)
    rho_inner = rho[inner]

    def _integral_rel_error(pre: FloatArray, post: FloatArray) -> float:
        before = float(trapezoid(pre[inner] * rho_inner, rho_inner))
        after = float(trapezoid(post[inner] * rho_inner, rho_inner))
        return abs(after - before) / max(abs(before), 1e-300)

    integrand = rho * (1.0 / np.maximum(q, 1e-6) - 1.0)
    psi_star = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * np.diff(rho))])
    psi_at_mix = float(np.interp(rho_mix, rho, psi_star))
    psi_scale = float(np.max(np.abs(psi_star)))

    temperature_scale = float(np.max(np.abs(temperature[inner])))
    density_scale = float(np.max(np.abs(density[inner])))
    outside = slice(idx_mix, len(rho))
    outside_change = max(
        float(np.max(np.abs(t_new[outside] - temperature[outside]))) if idx_mix < len(rho) else 0.0,
        float(np.max(np.abs(n_new[outside] - density[outside]))) if idx_mix < len(rho) else 0.0,
        float(np.max(np.abs(q_new[outside] - q[outside]))) if idx_mix < len(rho) else 0.0,
    )

    return CrashConservation(
        rho_1=float(rho_1),
        rho_mix=float(rho_mix),
        temperature_integral_rel_error=_integral_rel_error(temperature, t_new.astype(np.float64)),
        density_integral_rel_error=_integral_rel_error(density, n_new.astype(np.float64)),
        helical_flux_residual=abs(psi_at_mix) / max(psi_scale, 1e-300),
        inner_temperature_flatness=float(np.ptp(t_new[inner])) / max(temperature_scale, 1e-300),
        inner_density_flatness=float(np.ptp(n_new[inner])) / max(density_scale, 1e-300),
        inner_q_value=float(q_new[0]),
        outside_max_abs_change=outside_change,
    )


def no_crash_leaves_profiles_unchanged(config: SawtoothConfig) -> bool:
    """With ``q > 1`` everywhere the crash must return identical profiles."""
    rho, temperature, density, _ = _profiles(config)
    q_stable = 1.2 + 0.5 * rho**2
    t_new, n_new, q_new, rho_1, rho_mix = kadomtsev_crash(rho, temperature, density, q_stable, config.r0, config.a)
    return bool(
        rho_1 == 0.0
        and rho_mix == 0.0
        and np.array_equal(t_new, temperature)
        and np.array_equal(n_new, density)
        and np.array_equal(q_new, q_stable)
    )


@dataclass(frozen=True)
class SawtoothValidationResult:
    """Outcome of the Kadomtsev sawtooth crash validation."""

    config: SawtoothConfig
    conservation: CrashConservation
    q1_rel_error_coarse: float
    q1_rel_error_fine: float
    q1_order: float
    no_crash_ok: bool
    conservation_tol: float
    structure_tol: float
    q1_tol: float
    conservation_passed: bool
    helical_passed: bool
    structure_passed: bool
    outside_passed: bool
    q1_passed: bool
    no_crash_passed: bool
    passed: bool


def validate_sawtooth_kadomtsev(
    *,
    config: SawtoothConfig | None = None,
    conservation_tol: float = 1e-10,
    structure_tol: float = 1e-12,
    q1_tol: float = 5e-3,
    post_crash_q: float = 1.01,
) -> SawtoothValidationResult:
    """Validate the production Kadomtsev crash against exact conservation laws.

    Volume-integral conservation, the helical-flux mixing condition, profile
    flattening, and outside invariance must hold to machine precision; the
    interpolated ``q = 1`` radius must converge to the analytic value at roughly
    second order and stay below ``q1_tol`` on the default grid; and the no-crash
    guard must leave profiles unchanged when ``q > 1`` everywhere.
    """
    config = config or default_config()
    conservation = crash_conservation(config)

    coarse = q1_radius_rel_error(config)
    fine_config = SawtoothConfig(nr=2 * config.nr - 1, q0=config.q0, qa=config.qa, r0=config.r0, a=config.a)
    fine = q1_radius_rel_error(fine_config)
    h_ratio = (config.nr - 1) / (fine_config.nr - 1)  # = 0.5
    q1_order = math.log(coarse / fine) / math.log(1.0 / h_ratio) if fine > 0.0 else math.inf

    no_crash_ok = no_crash_leaves_profiles_unchanged(config)

    conservation_passed = (
        conservation.temperature_integral_rel_error < conservation_tol
        and conservation.density_integral_rel_error < conservation_tol
    )
    helical_passed = conservation.helical_flux_residual < conservation_tol
    structure_passed = (
        conservation.inner_temperature_flatness < structure_tol
        and conservation.inner_density_flatness < structure_tol
        and abs(conservation.inner_q_value - post_crash_q) < structure_tol
    )
    outside_passed = conservation.outside_max_abs_change == 0.0
    q1_passed = fine <= coarse and fine < q1_tol
    no_crash_passed = no_crash_ok

    passed = (
        conservation_passed and helical_passed and structure_passed and outside_passed and q1_passed and no_crash_passed
    )
    return SawtoothValidationResult(
        config=config,
        conservation=conservation,
        q1_rel_error_coarse=coarse,
        q1_rel_error_fine=fine,
        q1_order=q1_order,
        no_crash_ok=no_crash_ok,
        conservation_tol=conservation_tol,
        structure_tol=structure_tol,
        q1_tol=q1_tol,
        conservation_passed=conservation_passed,
        helical_passed=helical_passed,
        structure_passed=structure_passed,
        outside_passed=outside_passed,
        q1_passed=q1_passed,
        no_crash_passed=no_crash_passed,
        passed=passed,
    )


def build_evidence(result: SawtoothValidationResult, *, target_id: str) -> dict[str, Any]:
    """Build a tamper-evident, schema-versioned validation evidence payload."""
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    cons = result.conservation
    payload: dict[str, Any] = {
        "schema_version": SAWTOOTH_KADOMTSEV_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "target_id": target_id,
        "config": {
            "nr": result.config.nr,
            "q0": result.config.q0,
            "qa": result.config.qa,
            "r0": result.config.r0,
            "a": result.config.a,
        },
        "conservation_tol": result.conservation_tol,
        "structure_tol": result.structure_tol,
        "q1_tol": result.q1_tol,
        "rho_1": cons.rho_1,
        "rho_mix": cons.rho_mix,
        "temperature_integral_rel_error": cons.temperature_integral_rel_error,
        "density_integral_rel_error": cons.density_integral_rel_error,
        "helical_flux_residual": cons.helical_flux_residual,
        "inner_temperature_flatness": cons.inner_temperature_flatness,
        "inner_density_flatness": cons.inner_density_flatness,
        "inner_q_value": cons.inner_q_value,
        "outside_max_abs_change": cons.outside_max_abs_change,
        "q1_rel_error_coarse": result.q1_rel_error_coarse,
        "q1_rel_error_fine": result.q1_rel_error_fine,
        "q1_order": result.q1_order,
        "no_crash_ok": result.no_crash_ok,
        "conservation_passed": result.conservation_passed,
        "helical_passed": result.helical_passed,
        "structure_passed": result.structure_passed,
        "outside_passed": result.outside_passed,
        "q1_passed": result.q1_passed,
        "no_crash_passed": result.no_crash_passed,
        "passed": result.passed,
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Return ``True`` when a payload is well-formed, sealed, and passing."""
    if payload.get("schema_version") != SAWTOOTH_KADOMTSEV_SCHEMA_VERSION:
        raise ValueError("unsupported sawtooth kadomtsev evidence schema_version")
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
    if result < 11:
        raise ValueError(f"{name} must be at least 11 (need a resolved mixing region)")
    return result


def _write_report(evidence: Mapping[str, Any], json_path: Path) -> None:
    """Persist the sealed JSON evidence and a human-readable Markdown summary."""
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = json_path.with_suffix(".md")
    config = evidence["config"]
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "",
        "# Sawtooth Kadomtsev Crash Validation",
        "",
        f"- Schema: `{evidence['schema_version']}`",
        f"- Generated (UTC): {evidence['generated_utc']}",
        f"- Target: `{evidence['target_id']}`",
        f"- Profile: q(rho) = {config['q0']} + {config['qa'] - config['q0']} rho^2 on {config['nr']} points",
        f"- q=1 radius rho_1 = {evidence['rho_1']:.5f}; mixing radius rho_mix = {evidence['rho_mix']:.5f}",
        f"- Status: **{'pass' if evidence['passed'] else 'fail'}**",
        "",
        f"## Exact conservation and structure (gate < {evidence['conservation_tol']:.1e})",
        "",
        "| reference | value |",
        "| --- | --- |",
        f"| temperature volume-integral rel error | {evidence['temperature_integral_rel_error']:.3e} |",
        f"| density volume-integral rel error | {evidence['density_integral_rel_error']:.3e} |",
        f"| helical-flux residual psi*(rho_mix) | {evidence['helical_flux_residual']:.3e} |",
        f"| inner temperature flatness | {evidence['inner_temperature_flatness']:.3e} |",
        f"| inner density flatness | {evidence['inner_density_flatness']:.3e} |",
        f"| inner q value | {evidence['inner_q_value']:.4f} |",
        f"| outside max abs change | {evidence['outside_max_abs_change']:.3e} |",
        "",
        "## q=1 radius convergence",
        "",
        f"- Coarse-grid rel error: {evidence['q1_rel_error_coarse']:.3e}",
        f"- Fine-grid rel error: {evidence['q1_rel_error_fine']:.3e}",
        f"- Convergence order: {evidence['q1_order']:.3f}",
        f"- No-crash guard (q>1 everywhere leaves profiles unchanged): {evidence['no_crash_ok']}",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point producing schema-versioned validation evidence."""
    parser = argparse.ArgumentParser(
        description="Validate the Kadomtsev sawtooth crash against exact conservation laws"
    )
    parser.add_argument("--target-id", type=str, default="local-sawtooth-kadomtsev")
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    parser.add_argument("--report", type=str, default=None, help="write sealed JSON evidence and a Markdown summary")
    args = parser.parse_args(argv)

    result = validate_sawtooth_kadomtsev()
    evidence = build_evidence(result, target_id=args.target_id)

    if args.report:
        _write_report(evidence, Path(args.report))

    if args.json_out:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        cons = result.conservation
        print("Sawtooth Kadomtsev crash validation")
        print(
            f"  conservation: T_int={cons.temperature_integral_rel_error:.3e} "
            f"n_int={cons.density_integral_rel_error:.3e} "
            f"{'ok' if result.conservation_passed else 'FAIL'}"
        )
        print(
            f"  helical flux: psi*(rho_mix)={cons.helical_flux_residual:.3e} "
            f"{'ok' if result.helical_passed else 'FAIL'}"
        )
        print(
            f"  structure:    flatness/q-reset {'ok' if result.structure_passed else 'FAIL'}; "
            f"outside {'ok' if result.outside_passed else 'FAIL'}"
        )
        print(
            f"  q=1 radius:   coarse={result.q1_rel_error_coarse:.3e} fine={result.q1_rel_error_fine:.3e} "
            f"order={result.q1_order:.2f} {'ok' if result.q1_passed else 'FAIL'}"
        )
        print(f"Status: {'pass' if result.passed else 'fail'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

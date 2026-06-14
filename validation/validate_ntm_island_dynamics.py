#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — NTM island dynamics analytic validation
"""Validate the Modified Rutherford Equation solver against exact ODE references.

The neoclassical-tearing-mode model (``src/scpn_control/core/ntm_dynamics.py``)
integrates the Modified Rutherford Equation (MRE) for the island half-width
``w(t)`` with an RK4 stepper. Two limits of the MRE admit *exact* closed-form
references, so the solver can be validated without any measured or external NTM
benchmark — the validation is fully self-contained.

The classical tearing term uses the finite-island saturation
``Delta'(w) = Delta'_0 r_s / (r_s + c w)`` with ``c = 0.5`` (La Haye 2006), so
``r_s Delta'(w) = Delta'_0 r_s^2 / (r_s + 0.5 w)``.

1. **Classical-only trajectory (exact).** With the bootstrap, polarisation,
   diamagnetic, and ECCD terms switched off (``j_bs = j_cd = 0``,
   ``w_pol = w_d = 0``) the MRE reduces to the separable ODE
   ``dw/dt = K / (r_s + 0.5 w)`` with ``K = Delta'_0 r_s^2 / tau_R``, whose exact
   solution is ``w(t) = -2 r_s + 2 sqrt((r_s + 0.5 w_0)^2 + K t)``. The RK4
   ``evolve`` must reproduce this to integrator precision, and ``dw_dt`` must
   match the closed-form right-hand side algebraically.
2. **Classical + bootstrap saturated width (exact fixed point).** For a
   classically stable surface (``Delta'_0 < 0``) with a bootstrap drive and
   ``w_d = 0`` the steady state ``dw/dt = 0`` has the closed form
   ``w_sat = -a_1 (j_bs/j_phi) r_s / (Delta'_0 r_s + 0.5 a_1 (j_bs/j_phi))``.
   ``dw_dt`` must vanish there and ``evolve`` must converge monotonically to
   ``w_sat`` from both below and above, confirming it is a stable attractor.

References:
  Rutherford P. H. (1973) *Phys. Fluids* 16, 1903.
  Sauter O. et al. (1997) *Phys. Plasmas* 4, 1654.
  La Haye R. J. (2006) *Phys. Plasmas* 13, 055501.
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

from scpn_control.core.ntm_dynamics import _A1, NTMIslandDynamics

NTM_ISLAND_DYNAMICS_SCHEMA_VERSION = "scpn-control.ntm-island-dynamics-validation.v1"

_MU_0 = 4.0e-7 * math.pi
_SATURATION_COEFFICIENT = 0.5  # La Haye 2006 finite-island saturation coefficient c


@dataclass(frozen=True)
class RationalSurfaceConfig:
    """Rational-surface geometry shared by every validation case."""

    r_s: float
    m: int
    n: int
    a: float
    R0: float
    B0: float

    def __post_init__(self) -> None:
        _positive_float("r_s", self.r_s)
        _positive_float("a", self.a)
        _positive_float("R0", self.R0)
        _positive_float("B0", self.B0)
        _positive_int("m", self.m)
        _positive_int("n", self.n)
        if self.r_s > self.a:
            raise ValueError("r_s must not exceed the minor radius a")
        if self.a >= self.R0:
            raise ValueError("a must be smaller than R0 for tokamak ordering")


def default_surface() -> RationalSurfaceConfig:
    """A 2/1 rational surface in an ITER-like geometry."""
    return RationalSurfaceConfig(r_s=0.3, m=2, n=1, a=0.5, R0=1.7, B0=2.0)


def _build(surface: RationalSurfaceConfig, delta_prime_0: float) -> NTMIslandDynamics:
    return NTMIslandDynamics(
        surface.r_s, surface.m, surface.n, surface.a, surface.R0, surface.B0, Delta_prime_0=delta_prime_0
    )


def _tau_r(surface: RationalSurfaceConfig, eta: float) -> float:
    """Resistive diffusion time tau_R = mu_0 r_s^2 / eta (Rutherford 1973)."""
    return _MU_0 * surface.r_s**2 / _positive_float("eta", eta)


def classical_dw_dt_rel_error(
    surface: RationalSurfaceConfig, delta_prime_0: float, *, w: float, eta: float, j_phi: float
) -> float:
    """Relative error of ``dw_dt`` against the closed-form classical right-hand side."""
    _positive_float("w", w)
    ntm = _build(surface, delta_prime_0)
    measured = ntm.dw_dt(w, j_bs=0.0, j_phi=j_phi, j_cd=0.0, eta=eta, w_d=0.0, w_pol=0.0)
    tau_r = _tau_r(surface, eta)
    analytic = (1.0 / tau_r) * delta_prime_0 * surface.r_s**2 / (surface.r_s + _SATURATION_COEFFICIENT * w)
    return abs(measured - analytic) / max(abs(analytic), 1e-300)


def classical_trajectory_max_rel_error(
    surface: RationalSurfaceConfig,
    delta_prime_0: float,
    *,
    w0: float,
    eta: float,
    t_end: float,
    n_steps: int,
    j_phi: float,
) -> float:
    """Max relative error of the RK4 trajectory against the exact separable solution."""
    _positive_float("w0", w0)
    _positive_float("t_end", t_end)
    _positive_int("n_steps", n_steps)
    ntm = _build(surface, delta_prime_0)
    dt = t_end / n_steps
    times, widths = ntm.evolve(w0, (0.0, t_end), dt, j_bs=0.0, j_phi=j_phi, j_cd=0.0, eta=eta, w_d=0.0, w_pol=0.0)
    tau_r = _tau_r(surface, eta)
    k = delta_prime_0 * surface.r_s**2 / tau_r
    inner = (surface.r_s + _SATURATION_COEFFICIENT * w0) ** 2 + k * np.asarray(times, dtype=np.float64)
    exact = -2.0 * surface.r_s + 2.0 * np.sqrt(np.maximum(inner, 0.0))
    return float(np.max(np.abs(np.asarray(widths, dtype=np.float64) - exact)) / w0)


def analytic_saturated_width(
    surface: RationalSurfaceConfig, delta_prime_0: float, *, j_bs: float, j_phi: float
) -> float:
    """Closed-form classical+bootstrap saturated half-width (``w_d = 0``)."""
    if delta_prime_0 >= 0.0:
        raise ValueError("saturated-width closed form requires a classically stable surface (Delta_prime_0 < 0)")
    j_ratio = _positive_float("j_bs", j_bs) / _positive_float("j_phi", j_phi)
    denominator = delta_prime_0 * surface.r_s + 0.5 * _A1 * j_ratio
    width = -_A1 * j_ratio * surface.r_s / denominator
    if not (math.isfinite(width) and width > 0.0):
        raise ValueError("chosen drive does not yield a positive saturated width; increase |Delta_prime_0| or j_bs")
    return width


def saturated_width_residual(
    surface: RationalSurfaceConfig, delta_prime_0: float, *, j_bs: float, j_phi: float, eta: float
) -> float:
    """Relative ``dw_dt`` residual at the analytic saturated width.

    Normalised by ``dw_dt`` at half the saturated width to give a dimensionless
    measure of how exactly the closed-form fixed point zeroes the right-hand side.
    """
    ntm = _build(surface, delta_prime_0)
    w_sat = analytic_saturated_width(surface, delta_prime_0, j_bs=j_bs, j_phi=j_phi)
    at_sat = ntm.dw_dt(w_sat, j_bs=j_bs, j_phi=j_phi, j_cd=0.0, eta=eta, w_d=0.0, w_pol=0.0)
    reference = ntm.dw_dt(0.5 * w_sat, j_bs=j_bs, j_phi=j_phi, j_cd=0.0, eta=eta, w_d=0.0, w_pol=0.0)
    return abs(at_sat) / max(abs(reference), 1e-300)


@dataclass(frozen=True)
class SaturationApproach:
    """Convergence of the RK4 evolution to the analytic saturated width."""

    saturated_width: float
    from_below_rel_error: float
    from_above_rel_error: float
    from_below_monotonic: bool
    from_above_monotonic: bool


def saturation_convergence(
    surface: RationalSurfaceConfig,
    delta_prime_0: float,
    *,
    j_bs: float,
    j_phi: float,
    eta: float,
    t_end: float,
    n_steps: int,
) -> SaturationApproach:
    """Evolve from below and above ``w_sat`` and measure monotonic convergence."""
    _positive_float("t_end", t_end)
    _positive_int("n_steps", n_steps)
    ntm = _build(surface, delta_prime_0)
    w_sat = analytic_saturated_width(surface, delta_prime_0, j_bs=j_bs, j_phi=j_phi)
    dt = t_end / n_steps

    def _run(w0: float) -> tuple[float, bool]:
        _, widths = ntm.evolve(w0, (0.0, t_end), dt, j_bs=j_bs, j_phi=j_phi, j_cd=0.0, eta=eta, w_d=0.0, w_pol=0.0)
        widths = np.asarray(widths, dtype=np.float64)
        rel = abs(float(widths[-1]) - w_sat) / w_sat
        if w0 < w_sat:
            monotonic = bool(np.all(np.diff(widths) >= -1e-12))
        else:
            monotonic = bool(np.all(np.diff(widths) <= 1e-12))
        return rel, monotonic

    below_rel, below_mono = _run(0.5 * w_sat)
    above_rel, above_mono = _run(1.5 * w_sat)
    return SaturationApproach(
        saturated_width=w_sat,
        from_below_rel_error=below_rel,
        from_above_rel_error=above_rel,
        from_below_monotonic=below_mono,
        from_above_monotonic=above_mono,
    )


@dataclass(frozen=True)
class TrajectoryCase:
    """A classical-only trajectory case with its measured error."""

    label: str
    delta_prime_0: float
    w0: float
    dw_dt_rel_error: float
    trajectory_rel_error: float


@dataclass(frozen=True)
class NtmValidationResult:
    """Outcome of the NTM Modified Rutherford Equation validation."""

    surface: RationalSurfaceConfig
    eta: float
    trajectory_cases: tuple[TrajectoryCase, ...]
    max_dw_dt_rel_error: float
    max_trajectory_rel_error: float
    trajectory_passed: bool
    saturated_delta_prime_0: float
    saturation: SaturationApproach
    saturated_residual: float
    saturation_passed: bool
    dw_dt_tol: float
    trajectory_tol: float
    residual_tol: float
    saturation_tol: float
    passed: bool


def validate_ntm_island_dynamics(
    *,
    surface: RationalSurfaceConfig | None = None,
    eta: float = 5e-8,
    j_phi: float = 1.0e6,
    j_bs: float = 2.0e4,
    saturated_delta_prime_0: float = -6.0,
    dw_dt_tol: float = 1e-10,
    trajectory_tol: float = 1e-8,
    residual_tol: float = 1e-9,
    saturation_tol: float = 5e-3,
    t_end: float = 0.4,
    n_steps: int = 20000,
) -> NtmValidationResult:
    """Validate the MRE solver against exact classical and saturated-width references.

    The classical-only trajectory and right-hand side must match the closed-form
    separable solution to integrator precision, and the classical+bootstrap
    saturated width must zero ``dw_dt`` and act as a stable attractor for the RK4
    evolution.
    """
    surface = surface or default_surface()
    eta = _positive_float("eta", eta)

    trajectory_specs = (
        ("decaying_strong", -3.0, 0.05),
        ("decaying_weak", -1.5, 0.08),
        ("growing", 2.0, 0.01),
    )
    cases: list[TrajectoryCase] = []
    for label, delta_prime_0, w0 in trajectory_specs:
        dw_err = classical_dw_dt_rel_error(surface, delta_prime_0, w=0.04, eta=eta, j_phi=j_phi)
        traj_err = classical_trajectory_max_rel_error(
            surface, delta_prime_0, w0=w0, eta=eta, t_end=0.02, n_steps=4000, j_phi=j_phi
        )
        cases.append(
            TrajectoryCase(
                label=label,
                delta_prime_0=delta_prime_0,
                w0=w0,
                dw_dt_rel_error=dw_err,
                trajectory_rel_error=traj_err,
            )
        )

    max_dw_dt = max(case.dw_dt_rel_error for case in cases)
    max_traj = max(case.trajectory_rel_error for case in cases)
    trajectory_passed = max_dw_dt < dw_dt_tol and max_traj < trajectory_tol

    saturation = saturation_convergence(
        surface, saturated_delta_prime_0, j_bs=j_bs, j_phi=j_phi, eta=eta, t_end=t_end, n_steps=n_steps
    )
    saturated_residual = saturated_width_residual(surface, saturated_delta_prime_0, j_bs=j_bs, j_phi=j_phi, eta=eta)
    saturation_passed = (
        saturated_residual < residual_tol
        and saturation.from_below_rel_error < saturation_tol
        and saturation.from_above_rel_error < saturation_tol
        and saturation.from_below_monotonic
        and saturation.from_above_monotonic
    )

    return NtmValidationResult(
        surface=surface,
        eta=eta,
        trajectory_cases=tuple(cases),
        max_dw_dt_rel_error=max_dw_dt,
        max_trajectory_rel_error=max_traj,
        trajectory_passed=trajectory_passed,
        saturated_delta_prime_0=saturated_delta_prime_0,
        saturation=saturation,
        saturated_residual=saturated_residual,
        saturation_passed=saturation_passed,
        dw_dt_tol=dw_dt_tol,
        trajectory_tol=trajectory_tol,
        residual_tol=residual_tol,
        saturation_tol=saturation_tol,
        passed=trajectory_passed and saturation_passed,
    )


def build_evidence(result: NtmValidationResult, *, target_id: str) -> dict[str, Any]:
    """Build a tamper-evident, schema-versioned validation evidence payload."""
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    payload: dict[str, Any] = {
        "schema_version": NTM_ISLAND_DYNAMICS_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "target_id": target_id,
        "surface": {
            "r_s": result.surface.r_s,
            "m": result.surface.m,
            "n": result.surface.n,
            "a": result.surface.a,
            "R0": result.surface.R0,
            "B0": result.surface.B0,
        },
        "eta": result.eta,
        "dw_dt_tol": result.dw_dt_tol,
        "trajectory_tol": result.trajectory_tol,
        "residual_tol": result.residual_tol,
        "saturation_tol": result.saturation_tol,
        "trajectory_cases": [
            {
                "label": case.label,
                "delta_prime_0": case.delta_prime_0,
                "w0": case.w0,
                "dw_dt_rel_error": case.dw_dt_rel_error,
                "trajectory_rel_error": case.trajectory_rel_error,
            }
            for case in result.trajectory_cases
        ],
        "max_dw_dt_rel_error": result.max_dw_dt_rel_error,
        "max_trajectory_rel_error": result.max_trajectory_rel_error,
        "trajectory_passed": result.trajectory_passed,
        "saturated_delta_prime_0": result.saturated_delta_prime_0,
        "saturated_width": result.saturation.saturated_width,
        "saturated_residual": result.saturated_residual,
        "saturation_from_below_rel_error": result.saturation.from_below_rel_error,
        "saturation_from_above_rel_error": result.saturation.from_above_rel_error,
        "saturation_from_below_monotonic": result.saturation.from_below_monotonic,
        "saturation_from_above_monotonic": result.saturation.from_above_monotonic,
        "saturation_passed": result.saturation_passed,
        "passed": result.passed,
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Return ``True`` when a payload is well-formed, sealed, and passing."""
    if payload.get("schema_version") != NTM_ISLAND_DYNAMICS_SCHEMA_VERSION:
        raise ValueError("unsupported ntm island dynamics evidence schema_version")
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


def _write_report(evidence: Mapping[str, Any], json_path: Path) -> None:
    """Persist the sealed JSON evidence and a human-readable Markdown summary."""
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = json_path.with_suffix(".md")
    surface = evidence["surface"]
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "",
        "# NTM Island Dynamics Validation (Modified Rutherford Equation)",
        "",
        f"- Schema: `{evidence['schema_version']}`",
        f"- Generated (UTC): {evidence['generated_utc']}",
        f"- Target: `{evidence['target_id']}`",
        f"- Surface: {surface['m']}/{surface['n']} at r_s={surface['r_s']} m "
        f"(a={surface['a']} m, R0={surface['R0']} m, B0={surface['B0']} T)",
        f"- Status: **{'pass' if evidence['passed'] else 'fail'}**",
        "",
        "## Classical-only trajectory vs exact separable solution",
        "",
        f"- Max dw/dt relative error: {evidence['max_dw_dt_rel_error']:.3e} (gate < {evidence['dw_dt_tol']:.1e})",
        f"- Max trajectory relative error: {evidence['max_trajectory_rel_error']:.3e} "
        f"(gate < {evidence['trajectory_tol']:.1e})",
        "",
        "| case | Delta'_0 | w0 [m] | dw/dt rel err | trajectory rel err |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {case['label']} | {case['delta_prime_0']} | {case['w0']} | "
        f"{case['dw_dt_rel_error']:.3e} | {case['trajectory_rel_error']:.3e} |"
        for case in evidence["trajectory_cases"]
    ]
    lines += [
        "",
        "## Classical + bootstrap saturated width (stable attractor)",
        "",
        f"- Delta'_0 = {evidence['saturated_delta_prime_0']}; analytic w_sat = {evidence['saturated_width']:.5e} m",
        f"- Fixed-point residual |dw/dt(w_sat)| / |dw/dt(0.5 w_sat)|: "
        f"{evidence['saturated_residual']:.3e} (gate < {evidence['residual_tol']:.1e})",
        f"- Convergence from below: rel err {evidence['saturation_from_below_rel_error']:.3e}, "
        f"monotonic {evidence['saturation_from_below_monotonic']}",
        f"- Convergence from above: rel err {evidence['saturation_from_above_rel_error']:.3e}, "
        f"monotonic {evidence['saturation_from_above_monotonic']}",
        f"- Gate: rel err < {evidence['saturation_tol']:.1e} with monotonic approach",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point producing schema-versioned validation evidence."""
    parser = argparse.ArgumentParser(
        description="Validate the NTM Modified Rutherford Equation solver against exact references"
    )
    parser.add_argument("--target-id", type=str, default="local-ntm-island-dynamics")
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    parser.add_argument("--report", type=str, default=None, help="write sealed JSON evidence and a Markdown summary")
    args = parser.parse_args(argv)

    result = validate_ntm_island_dynamics()
    evidence = build_evidence(result, target_id=args.target_id)

    if args.report:
        _write_report(evidence, Path(args.report))

    if args.json_out:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print("NTM island dynamics validation (Modified Rutherford Equation)")
        print(
            f"  classical trajectory: max dw/dt err={result.max_dw_dt_rel_error:.3e} "
            f"max traj err={result.max_trajectory_rel_error:.3e} "
            f"{'ok' if result.trajectory_passed else 'FAIL'}"
        )
        print(
            f"  saturated width:      w_sat={result.saturation.saturated_width:.5e} "
            f"residual={result.saturated_residual:.3e} "
            f"below={result.saturation.from_below_rel_error:.2e} "
            f"above={result.saturation.from_above_rel_error:.2e} "
            f"{'ok' if result.saturation_passed else 'FAIL'}"
        )
        print(f"Status: {'pass' if result.passed else 'fail'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

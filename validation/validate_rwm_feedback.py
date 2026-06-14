#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Resistive-wall-mode feedback analytic validation
"""Validate the resistive-wall-mode feedback model against exact closed forms.

The resistive-wall-mode (RWM) model (``src/scpn_control/control/rwm_feedback.py``)
gives the wall-limited growth rate, rotation stabilisation, passive critical
rotation, and active PD-feedback closed-loop growth rate. Every relation is an
exact algebraic closed form, so the model can be validated without any measured
shot or external MHD code — the validation is fully self-contained.

Exact references checked against the production methods:

1. **Bondeson-Ward growth rate** (no rotation):
   ``gamma_wall = (1/tau_eff) (beta_N - beta_nw) / (beta_w - beta_N)`` inside the
   unstable window ``beta_nw < beta_N < beta_w``.
2. **Wall-gap correction**: ``tau_eff = tau_wall (b/d)^2``.
3. **Rotation stabilisation** (Fitzpatrick): ``gamma = gamma_wall - (Omega tau)^2
   / (tau (1 + (Omega tau)^2))``.
4. **Critical-rotation self-consistency**: at ``Omega = Omega_crit`` the total
   growth rate is exactly zero, linking ``critical_rotation`` and ``growth_rate``.
5. **Feedback self-consistency**: applying the ``required_feedback_gain`` drives
   the closed-loop ``effective_growth_rate`` to exactly zero, and the required
   gain equals ``(1 + gamma tau_ctrl) / M_coil``.
6. **Stability window**: ``beta_N <= beta_nw`` is stable (zero growth) and
   ``beta_N >= beta_w`` is the ideal kink (infinite growth).
7. **Wall-time scaling**: ``gamma_wall`` scales as ``1/tau_wall``.

References:
  Bondeson A., Ward D. J. (1994) *Phys. Rev. Lett.* 72, 2709.
  Fitzpatrick R. (2001) *Phys. Plasmas* 8, 4489.
  Strait E. J. et al. (2003) *Nucl. Fusion* 43, 430.
  Garofalo A. M. et al. (2002) *Phys. Plasmas* 9, 1997.
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

from scpn_control.control.rwm_feedback import RWMFeedbackController, RWMPhysics, RWMStabilityAnalysis

RWM_FEEDBACK_SCHEMA_VERSION = "scpn-control.rwm-feedback-validation.v1"


@dataclass(frozen=True)
class RWMConfig:
    """Beta limits and resistive-wall time for the RWM model."""

    beta_n_nowall: float
    beta_n_wall: float
    tau_wall_s: float

    def __post_init__(self) -> None:
        _finite_float("beta_n_nowall", self.beta_n_nowall)
        _finite_float("beta_n_wall", self.beta_n_wall)
        _positive_float("tau_wall_s", self.tau_wall_s)
        if self.beta_n_nowall >= self.beta_n_wall:
            raise ValueError("beta_n_nowall must be less than beta_n_wall")

    @property
    def window_midpoint(self) -> float:
        """``beta_N`` where the drive ratio ``A = (beta_N-beta_nw)/(beta_w-beta_N) = 1``."""
        return 0.5 * (self.beta_n_nowall + self.beta_n_wall)


def default_config() -> RWMConfig:
    """A representative unstable RWM window with a 5 ms wall time."""
    return RWMConfig(beta_n_nowall=2.0, beta_n_wall=4.0, tau_wall_s=5e-3)


def analytic_gamma_wall(config: RWMConfig, beta_n: float, *, tau_eff: float | None = None) -> float:
    """Exact Bondeson-Ward wall-limited growth rate for ``beta_nw < beta_N < beta_w``."""
    if not config.beta_n_nowall < beta_n < config.beta_n_wall:
        raise ValueError("beta_n must lie strictly inside the unstable window")
    tau = config.tau_wall_s if tau_eff is None else _positive_float("tau_eff", tau_eff)
    return (1.0 / tau) * (beta_n - config.beta_n_nowall) / (config.beta_n_wall - beta_n)


def gamma_wall_rel_error(config: RWMConfig, beta_n: float) -> float:
    """Relative error of ``growth_rate`` (no rotation) against the closed form."""
    measured = RWMPhysics(beta_n, config.beta_n_nowall, config.beta_n_wall, config.tau_wall_s).growth_rate()
    analytic = analytic_gamma_wall(config, beta_n)
    return abs(measured - analytic) / analytic


def tau_eff_rel_error(config: RWMConfig, *, wall_radius: float, plasma_radius: float) -> float:
    """Relative error of ``tau_eff`` against ``tau_wall (b/d)^2``."""
    rwm = RWMPhysics(
        config.window_midpoint,
        config.beta_n_nowall,
        config.beta_n_wall,
        config.tau_wall_s,
        wall_radius=wall_radius,
        plasma_radius=plasma_radius,
    )
    analytic = config.tau_wall_s * (wall_radius / plasma_radius) ** 2
    return abs(rwm.tau_eff() - analytic) / analytic


def rotation_rel_error(config: RWMConfig, beta_n: float, omega_phi: float) -> float:
    """Relative error of the rotation-stabilised growth rate against the closed form."""
    rwm = RWMPhysics(beta_n, config.beta_n_nowall, config.beta_n_wall, config.tau_wall_s, omega_phi=omega_phi)
    tau = config.tau_wall_s
    x = omega_phi * tau
    gamma_rot = -(x**2 / tau) / (1.0 + x**2)
    analytic = analytic_gamma_wall(config, beta_n) + gamma_rot
    return abs(rwm.growth_rate() - analytic) / abs(analytic)


def critical_rotation_residual(config: RWMConfig, beta_n: float) -> float:
    """Normalised total growth rate at ``Omega_crit`` (should be exactly zero).

    Returns ``|growth_rate(Omega_crit)| * tau_wall`` so the residual is
    dimensionless. Requires the drive ratio ``A < 1`` (``beta_N`` below the window
    midpoint), where passive rotation can fully stabilise the mode.
    """
    if not config.beta_n_nowall < beta_n < config.window_midpoint:
        raise ValueError("critical-rotation marginality requires beta_nw < beta_N < window midpoint (A < 1)")
    # Below the window midpoint the drive ratio A < 1, so critical_rotation is finite.
    base = RWMPhysics(beta_n, config.beta_n_nowall, config.beta_n_wall, config.tau_wall_s)
    omega_crit = base.critical_rotation()
    at_crit = RWMPhysics(
        beta_n, config.beta_n_nowall, config.beta_n_wall, config.tau_wall_s, omega_phi=omega_crit
    ).growth_rate()
    return abs(at_crit) * config.tau_wall_s


@dataclass(frozen=True)
class FeedbackCheck:
    """Required-gain and closed-loop marginality observation."""

    beta_n: float
    required_gain_rel_error: float
    closed_loop_residual: float


def feedback_check(config: RWMConfig, beta_n: float, *, tau_controller: float, m_coil: float) -> FeedbackCheck:
    """Check the required feedback gain matches its closed form and marginalises the mode."""
    rwm = RWMPhysics(beta_n, config.beta_n_nowall, config.beta_n_wall, config.tau_wall_s)
    gamma = rwm.growth_rate()
    required_gain = RWMStabilityAnalysis.required_feedback_gain(
        beta_n, config.beta_n_nowall, config.beta_n_wall, config.tau_wall_s, tau_controller, M_coil=m_coil
    )
    analytic_gain = (1.0 + gamma * tau_controller) / m_coil
    gain_rel = abs(required_gain - analytic_gain) / analytic_gain

    controller = RWMFeedbackController(2, 2, G_p=required_gain, G_d=0.0, tau_controller=tau_controller, M_coil=m_coil)
    closed_loop = controller.effective_growth_rate(rwm)
    return FeedbackCheck(
        beta_n=beta_n,
        required_gain_rel_error=gain_rel,
        closed_loop_residual=abs(closed_loop) * config.tau_wall_s,
    )


def tau_scaling_rel_error(config: RWMConfig, beta_n: float) -> float:
    """Relative error of the ``gamma ~ 1/tau_wall`` scaling (halving tau doubles gamma)."""
    base = RWMPhysics(beta_n, config.beta_n_nowall, config.beta_n_wall, config.tau_wall_s).growth_rate()
    halved = RWMPhysics(beta_n, config.beta_n_nowall, config.beta_n_wall, 0.5 * config.tau_wall_s).growth_rate()
    return abs(halved / base - 2.0) / 2.0


@dataclass(frozen=True)
class RwmValidationResult:
    """Outcome of the resistive-wall-mode feedback validation."""

    config: RWMConfig
    window_indices: tuple[float, ...]
    rotation_indices: tuple[float, ...]
    max_gamma_wall_rel_error: float
    tau_eff_rel_error: float
    max_rotation_rel_error: float
    max_critical_rotation_residual: float
    feedback_checks: tuple[FeedbackCheck, ...]
    max_required_gain_rel_error: float
    max_feedback_residual: float
    max_scaling_rel_error: float
    stable_growth_rate: float
    ideal_growth_rate: float
    exact_tol: float
    gamma_wall_passed: bool
    tau_eff_passed: bool
    rotation_passed: bool
    critical_rotation_passed: bool
    feedback_passed: bool
    scaling_passed: bool
    boundary_passed: bool
    passed: bool


def validate_rwm_feedback(
    *,
    config: RWMConfig | None = None,
    window_indices: Sequence[float] = (2.5, 3.0, 3.5),
    rotation_indices: Sequence[float] = (2.3, 2.5, 2.8),
    omega_phi: float = 300.0,
    tau_controller: float = 1e-4,
    m_coil: float = 1.3,
    exact_tol: float = 1e-9,
) -> RwmValidationResult:
    """Validate the production RWM model against its exact closed forms.

    The wall-limited growth rate, wall-gap correction, rotation stabilisation,
    critical-rotation marginality, feedback marginality, wall-time scaling, and
    stability-window boundaries must all hold to ``exact_tol``.
    """
    config = config or default_config()
    window = tuple(_finite_float("window index", b) for b in window_indices)
    rotation = tuple(_finite_float("rotation index", b) for b in rotation_indices)
    if not window or not rotation:
        raise ValueError("at least one window index and one rotation index are required")

    max_gamma_wall = max(gamma_wall_rel_error(config, b) for b in window)
    tau_eff_err = tau_eff_rel_error(config, wall_radius=1.2, plasma_radius=1.0)
    max_rotation = max(rotation_rel_error(config, b, omega_phi) for b in window)
    max_crit = max(critical_rotation_residual(config, b) for b in rotation)
    checks = tuple(feedback_check(config, b, tau_controller=tau_controller, m_coil=m_coil) for b in window)
    max_gain_rel = max(c.required_gain_rel_error for c in checks)
    max_fb_residual = max(c.closed_loop_residual for c in checks)
    max_scaling = max(tau_scaling_rel_error(config, b) for b in window)

    stable = RWMPhysics(config.beta_n_nowall, config.beta_n_nowall, config.beta_n_wall, config.tau_wall_s).growth_rate()
    ideal = RWMPhysics(config.beta_n_wall, config.beta_n_nowall, config.beta_n_wall, config.tau_wall_s).growth_rate()

    gamma_wall_passed = max_gamma_wall < exact_tol
    tau_eff_passed = tau_eff_err < exact_tol
    rotation_passed = max_rotation < exact_tol
    critical_rotation_passed = max_crit < exact_tol
    feedback_passed = max_gain_rel < exact_tol and max_fb_residual < exact_tol
    scaling_passed = max_scaling < exact_tol
    boundary_passed = stable == 0.0 and not math.isfinite(ideal)

    passed = (
        gamma_wall_passed
        and tau_eff_passed
        and rotation_passed
        and critical_rotation_passed
        and feedback_passed
        and scaling_passed
        and boundary_passed
    )
    return RwmValidationResult(
        config=config,
        window_indices=window,
        rotation_indices=rotation,
        max_gamma_wall_rel_error=max_gamma_wall,
        tau_eff_rel_error=tau_eff_err,
        max_rotation_rel_error=max_rotation,
        max_critical_rotation_residual=max_crit,
        feedback_checks=checks,
        max_required_gain_rel_error=max_gain_rel,
        max_feedback_residual=max_fb_residual,
        max_scaling_rel_error=max_scaling,
        stable_growth_rate=float(stable),
        ideal_growth_rate=float(ideal),
        exact_tol=exact_tol,
        gamma_wall_passed=gamma_wall_passed,
        tau_eff_passed=tau_eff_passed,
        rotation_passed=rotation_passed,
        critical_rotation_passed=critical_rotation_passed,
        feedback_passed=feedback_passed,
        scaling_passed=scaling_passed,
        boundary_passed=boundary_passed,
        passed=passed,
    )


def build_evidence(result: RwmValidationResult, *, target_id: str) -> dict[str, Any]:
    """Build a tamper-evident, schema-versioned validation evidence payload."""
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    payload: dict[str, Any] = {
        "schema_version": RWM_FEEDBACK_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "target_id": target_id,
        "config": {
            "beta_n_nowall": result.config.beta_n_nowall,
            "beta_n_wall": result.config.beta_n_wall,
            "tau_wall_s": result.config.tau_wall_s,
        },
        "window_indices": list(result.window_indices),
        "rotation_indices": list(result.rotation_indices),
        "exact_tol": result.exact_tol,
        "max_gamma_wall_rel_error": result.max_gamma_wall_rel_error,
        "tau_eff_rel_error": result.tau_eff_rel_error,
        "max_rotation_rel_error": result.max_rotation_rel_error,
        "max_critical_rotation_residual": result.max_critical_rotation_residual,
        "feedback_checks": [
            {
                "beta_n": c.beta_n,
                "required_gain_rel_error": c.required_gain_rel_error,
                "closed_loop_residual": c.closed_loop_residual,
            }
            for c in result.feedback_checks
        ],
        "max_required_gain_rel_error": result.max_required_gain_rel_error,
        "max_feedback_residual": result.max_feedback_residual,
        "max_scaling_rel_error": result.max_scaling_rel_error,
        "stable_growth_rate": result.stable_growth_rate,
        "ideal_growth_rate_is_infinite": not math.isfinite(result.ideal_growth_rate),
        "gamma_wall_passed": result.gamma_wall_passed,
        "tau_eff_passed": result.tau_eff_passed,
        "rotation_passed": result.rotation_passed,
        "critical_rotation_passed": result.critical_rotation_passed,
        "feedback_passed": result.feedback_passed,
        "scaling_passed": result.scaling_passed,
        "boundary_passed": result.boundary_passed,
        "passed": result.passed,
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Return ``True`` when a payload is well-formed, sealed, and passing."""
    if payload.get("schema_version") != RWM_FEEDBACK_SCHEMA_VERSION:
        raise ValueError("unsupported rwm feedback evidence schema_version")
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
    config = evidence["config"]
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "",
        "# Resistive-Wall-Mode Feedback Validation",
        "",
        f"- Schema: `{evidence['schema_version']}`",
        f"- Generated (UTC): {evidence['generated_utc']}",
        f"- Target: `{evidence['target_id']}`",
        f"- Window: beta_nw={config['beta_n_nowall']}, beta_w={config['beta_n_wall']}, "
        f"tau_wall={config['tau_wall_s']} s",
        f"- Status: **{'pass' if evidence['passed'] else 'fail'}**",
        "",
        f"## Exact closed-form references (relative error, gate < {evidence['exact_tol']:.1e})",
        "",
        "| reference | max rel error / residual |",
        "| --- | --- |",
        f"| Bondeson-Ward growth rate | {evidence['max_gamma_wall_rel_error']:.3e} |",
        f"| wall-gap tau_eff = tau_wall (b/d)^2 | {evidence['tau_eff_rel_error']:.3e} |",
        f"| rotation stabilisation | {evidence['max_rotation_rel_error']:.3e} |",
        f"| critical-rotation marginality | {evidence['max_critical_rotation_residual']:.3e} |",
        f"| required feedback gain | {evidence['max_required_gain_rel_error']:.3e} |",
        f"| feedback closed-loop marginality | {evidence['max_feedback_residual']:.3e} |",
        f"| 1/tau_wall scaling | {evidence['max_scaling_rel_error']:.3e} |",
        "",
        "## Stability-window boundaries",
        "",
        f"- Stable below no-wall limit: growth rate = {evidence['stable_growth_rate']:.3e}",
        f"- Ideal kink at/above wall limit: infinite growth = {evidence['ideal_growth_rate_is_infinite']}",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point producing schema-versioned validation evidence."""
    parser = argparse.ArgumentParser(
        description="Validate the resistive-wall-mode feedback model against exact closed forms"
    )
    parser.add_argument("--target-id", type=str, default="local-rwm-feedback")
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    parser.add_argument("--report", type=str, default=None, help="write sealed JSON evidence and a Markdown summary")
    args = parser.parse_args(argv)

    result = validate_rwm_feedback()
    evidence = build_evidence(result, target_id=args.target_id)

    if args.report:
        _write_report(evidence, Path(args.report))

    if args.json_out:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print("Resistive-wall-mode feedback validation")
        print(
            f"  Bondeson-Ward growth: max rel err={result.max_gamma_wall_rel_error:.3e} "
            f"{'ok' if result.gamma_wall_passed else 'FAIL'}"
        )
        print(
            f"  rotation + critical:  rot={result.max_rotation_rel_error:.3e} "
            f"crit_residual={result.max_critical_rotation_residual:.3e} "
            f"{'ok' if result.rotation_passed and result.critical_rotation_passed else 'FAIL'}"
        )
        print(
            f"  feedback marginality: gain_err={result.max_required_gain_rel_error:.3e} "
            f"residual={result.max_feedback_residual:.3e} "
            f"{'ok' if result.feedback_passed else 'FAIL'}"
        )
        print(f"  boundaries + scaling: {'ok' if result.boundary_passed and result.scaling_passed else 'FAIL'}")
        print(f"Status: {'pass' if result.passed else 'fail'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

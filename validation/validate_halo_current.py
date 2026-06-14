#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Halo-current L/R circuit analytic validation
"""Validate the Fitzpatrick halo-current L/R circuit against exact closed forms.

The halo-current model (``src/scpn_control/control/halo_re_physics.py``) advances
a Fitzpatrick-style L/R circuit ``L_h dI_h/dt + R_h I_h = M |dI_p/dt|`` during a
current quench, with closed-form circuit parameters and a wall-force estimate.
The circuit parameters are exact algebraic closed forms, and in the
fast-circuit limit the halo current quasi-statically tracks ``M |dI_p/dt| / R_h``;
neither needs a measured disruption shot, so the validation is self-contained.

Exact references checked against the production ``HaloCurrentModel``:

1. **Halo resistance.** ``R_h = eta 2 pi R0 / (d_wall a f_contact)`` with its
   ``eta``, ``R0``, ``1/a``, ``1/d_wall``, and ``1/f_contact`` scaling.
2. **Halo inductance.** ``L_h = mu_0 R0 (ln(8 R0/a) - 1.5)``.
3. **Mutual inductance.** ``M = f_contact sqrt(L_p L_h)``.
4. **Circuit time constant.** ``tau_h = L_h / R_h``.
5. **Wall force.** ``F = mu_0 I_h,peak I_p0 / (2 pi a)`` from the simulated peak.
6. **Toroidal peaking product.** ``TPF I_h / I_p0`` from the simulated peak.
7. **Quasi-static dynamics.** In the fast-circuit limit ``tau_h << tau_cq`` the
   halo current tracks ``M |dI_p/dt| / R_h`` with an error that decreases as
   ``tau_h/tau_cq``, confirming the L/R relaxation.

References:
  Fitzpatrick R. (2002) *Phys. Plasmas* 9, 3459 (halo-current/error-field model).
  Wesson J. (2011) *Tokamaks*, 4th ed., Oxford University Press, Ch. 7.
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

from scpn_control.control.halo_re_physics import _MU0, HaloCurrentModel

HALO_CURRENT_SCHEMA_VERSION = "scpn-control.halo-current-validation.v1"


@dataclass(frozen=True)
class HaloConfig:
    """Geometry and wall parameters for the halo-current circuit."""

    plasma_current_ma: float
    minor_radius_m: float
    major_radius_m: float
    wall_resistivity_ohm_m: float
    wall_thickness_m: float
    tpf: float
    contact_fraction: float

    def __post_init__(self) -> None:
        _positive_float("plasma_current_ma", self.plasma_current_ma)
        _positive_float("minor_radius_m", self.minor_radius_m)
        _positive_float("major_radius_m", self.major_radius_m)
        _positive_float("wall_resistivity_ohm_m", self.wall_resistivity_ohm_m)
        _positive_float("wall_thickness_m", self.wall_thickness_m)
        _positive_float("tpf", self.tpf)
        if not 0.0 < self.contact_fraction <= 1.0:
            raise ValueError("contact_fraction must lie in (0, 1]")
        if self.minor_radius_m >= self.major_radius_m:
            raise ValueError("minor radius must be smaller than the major radius")

    def model(self) -> HaloCurrentModel:
        return HaloCurrentModel(
            plasma_current_ma=self.plasma_current_ma,
            minor_radius_m=self.minor_radius_m,
            major_radius_m=self.major_radius_m,
            wall_resistivity_ohm_m=self.wall_resistivity_ohm_m,
            wall_thickness_m=self.wall_thickness_m,
            tpf=self.tpf,
            contact_fraction=self.contact_fraction,
        )


def default_config() -> HaloConfig:
    """An ITER-like 15 MA disruption geometry."""
    return HaloConfig(
        plasma_current_ma=15.0,
        minor_radius_m=2.0,
        major_radius_m=6.2,
        wall_resistivity_ohm_m=7e-7,
        wall_thickness_m=0.06,
        tpf=2.0,
        contact_fraction=0.3,
    )


def resistance_rel_error(config: HaloConfig) -> float:
    """Relative error of ``R_h`` against ``eta 2 pi R0 / (d_wall a f_contact)``."""
    model = config.model()
    analytic = (
        config.wall_resistivity_ohm_m
        * 2.0
        * math.pi
        * config.major_radius_m
        / (config.wall_thickness_m * config.minor_radius_m * config.contact_fraction)
    )
    return float(abs(model.R_h - analytic) / analytic)


def inductance_rel_error(config: HaloConfig) -> float:
    """Relative error of ``L_h`` against ``mu_0 R0 (ln(8 R0/a) - 1.5)``."""
    model = config.model()
    analytic = _MU0 * config.major_radius_m * (math.log(8.0 * config.major_radius_m / config.minor_radius_m) - 1.5)
    return float(abs(model.L_h - analytic) / analytic)


def mutual_inductance_rel_error(config: HaloConfig) -> float:
    """Relative error of ``M`` against ``f_contact sqrt(L_p L_h)``."""
    model = config.model()
    log_term = math.log(8.0 * config.major_radius_m / config.minor_radius_m)
    l_h = _MU0 * config.major_radius_m * (log_term - 1.5)
    l_p = _MU0 * config.major_radius_m * (log_term - 2.0 + 0.5)
    analytic = config.contact_fraction * math.sqrt(l_p * l_h)
    return float(abs(model.M - analytic) / analytic)


def time_constant_rel_error(config: HaloConfig) -> float:
    """Relative error of ``tau_h`` against ``L_h / R_h``."""
    model = config.model()
    return float(abs(model.tau_h - model.L_h / model.R_h) / (model.L_h / model.R_h))


@dataclass(frozen=True)
class ScalingCheck:
    """One halo-resistance scaling-law observation."""

    name: str
    measured_ratio: float
    expected_ratio: float
    rel_error: float


def resistance_scaling_checks(config: HaloConfig) -> tuple[ScalingCheck, ...]:
    """Verify ``R_h`` scales as ``eta``, ``1/f_contact``, ``R0``, and ``1/d_wall``."""
    import dataclasses

    base = config.model().R_h
    specs = (
        (
            "resistivity_linear",
            dataclasses.replace(config, wall_resistivity_ohm_m=2.0 * config.wall_resistivity_ohm_m).model().R_h / base,
            2.0,
        ),
        (
            "contact_inverse",
            dataclasses.replace(config, contact_fraction=2.0 * config.contact_fraction).model().R_h / base,
            0.5,
        ),
        (
            "major_radius_linear",
            dataclasses.replace(config, major_radius_m=2.0 * config.major_radius_m).model().R_h / base,
            2.0,
        ),
        (
            "thickness_inverse",
            dataclasses.replace(config, wall_thickness_m=2.0 * config.wall_thickness_m).model().R_h / base,
            0.5,
        ),
    )
    return tuple(
        ScalingCheck(
            name=name, measured_ratio=ratio, expected_ratio=expected, rel_error=abs(ratio - expected) / expected
        )
        for name, ratio, expected in specs
    )


def wall_force_rel_error(config: HaloConfig) -> float:
    """Relative error of the simulated wall force against ``mu_0 I_h,peak I_p0 / (2 pi a)``."""
    model = config.model()
    result = model.simulate(tau_cq_s=0.01, duration_s=0.05, dt_s=1e-5)
    peak_ih = result.peak_halo_ma * 1e6
    analytic = _MU0 * peak_ih * model.Ip0 / (2.0 * math.pi * config.minor_radius_m) / 1e6
    return float(abs(result.wall_force_mn_m - analytic) / analytic)


def tpf_product_rel_error(config: HaloConfig) -> float:
    """Relative error of the simulated peak TPF product against ``tpf I_h,peak / I_p0``."""
    model = config.model()
    result = model.simulate(tau_cq_s=0.01, duration_s=0.05, dt_s=1e-5)
    peak_ih = result.peak_halo_ma * 1e6
    analytic = config.tpf * peak_ih / model.Ip0
    return float(abs(result.peak_tpf_product - analytic) / max(abs(analytic), 1e-300))


@dataclass(frozen=True)
class QuasiStaticCheck:
    """Quasi-static L/R tracking in the fast-circuit limit."""

    tau_cq_values: tuple[float, ...]
    tracking_errors: tuple[float, ...]
    monotonic_decrease: bool
    finest_error: float


def quasi_static_tracking(config: HaloConfig) -> QuasiStaticCheck:
    """Check that the halo current tracks ``M |dI_p/dt| / R_h`` as ``tau_h/tau_cq -> 0``."""
    model = config.model()
    tau_cq_values = (0.5, 1.0, 2.0)
    errors: list[float] = []
    for tau_cq in tau_cq_values:
        result = model.simulate(tau_cq_s=tau_cq, duration_s=0.2, dt_s=2e-4)
        idx = len(result.halo_current_ma) // 2
        halo = result.halo_current_ma[idx] * 1e6
        plasma = result.plasma_current_ma[idx] * 1e6
        quasi_static = model.M * plasma / (model.R_h * tau_cq)
        errors.append(abs(halo - quasi_static) / quasi_static)
    monotonic = all(errors[i] < errors[i - 1] for i in range(1, len(errors)))
    return QuasiStaticCheck(
        tau_cq_values=tau_cq_values,
        tracking_errors=tuple(errors),
        monotonic_decrease=monotonic,
        finest_error=errors[-1],
    )


@dataclass(frozen=True)
class HaloValidationResult:
    """Outcome of the halo-current circuit validation."""

    config: HaloConfig
    resistance_rel_error: float
    inductance_rel_error: float
    mutual_rel_error: float
    time_constant_rel_error: float
    resistance_scaling: tuple[ScalingCheck, ...]
    max_resistance_scaling_rel_error: float
    wall_force_rel_error: float
    tpf_product_rel_error: float
    quasi_static: QuasiStaticCheck
    exact_tol: float
    quasi_static_tol: float
    circuit_passed: bool
    scaling_passed: bool
    loads_passed: bool
    quasi_static_passed: bool
    passed: bool


def validate_halo_current(
    *, config: HaloConfig | None = None, exact_tol: float = 1e-9, quasi_static_tol: float = 1e-2
) -> HaloValidationResult:
    """Validate the production halo-current circuit against its exact relations.

    The circuit parameters, resistance scalings, wall force, and TPF product must
    hold to ``exact_tol``; the quasi-static dynamics must track the fast-circuit
    limit with a monotonically decreasing error below ``quasi_static_tol``.
    """
    config = config or default_config()

    r_err = resistance_rel_error(config)
    l_err = inductance_rel_error(config)
    m_err = mutual_inductance_rel_error(config)
    tau_err = time_constant_rel_error(config)
    scaling = resistance_scaling_checks(config)
    max_scaling = max(check.rel_error for check in scaling)
    wall_err = wall_force_rel_error(config)
    tpf_err = tpf_product_rel_error(config)
    quasi = quasi_static_tracking(config)

    circuit_passed = bool(r_err < exact_tol and l_err < exact_tol and m_err < exact_tol and tau_err < exact_tol)
    scaling_passed = bool(max_scaling < exact_tol)
    loads_passed = bool(wall_err < exact_tol and tpf_err < exact_tol)
    quasi_static_passed = bool(quasi.monotonic_decrease and quasi.finest_error < quasi_static_tol)

    passed = bool(circuit_passed and scaling_passed and loads_passed and quasi_static_passed)
    return HaloValidationResult(
        config=config,
        resistance_rel_error=r_err,
        inductance_rel_error=l_err,
        mutual_rel_error=m_err,
        time_constant_rel_error=tau_err,
        resistance_scaling=scaling,
        max_resistance_scaling_rel_error=max_scaling,
        wall_force_rel_error=wall_err,
        tpf_product_rel_error=tpf_err,
        quasi_static=quasi,
        exact_tol=exact_tol,
        quasi_static_tol=quasi_static_tol,
        circuit_passed=circuit_passed,
        scaling_passed=scaling_passed,
        loads_passed=loads_passed,
        quasi_static_passed=quasi_static_passed,
        passed=passed,
    )


def build_evidence(result: HaloValidationResult, *, target_id: str) -> dict[str, Any]:
    """Build a tamper-evident, schema-versioned validation evidence payload."""
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    payload: dict[str, Any] = {
        "schema_version": HALO_CURRENT_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "target_id": target_id,
        "config": {
            "plasma_current_ma": result.config.plasma_current_ma,
            "minor_radius_m": result.config.minor_radius_m,
            "major_radius_m": result.config.major_radius_m,
            "wall_resistivity_ohm_m": result.config.wall_resistivity_ohm_m,
            "wall_thickness_m": result.config.wall_thickness_m,
            "tpf": result.config.tpf,
            "contact_fraction": result.config.contact_fraction,
        },
        "exact_tol": result.exact_tol,
        "quasi_static_tol": result.quasi_static_tol,
        "resistance_rel_error": result.resistance_rel_error,
        "inductance_rel_error": result.inductance_rel_error,
        "mutual_rel_error": result.mutual_rel_error,
        "time_constant_rel_error": result.time_constant_rel_error,
        "resistance_scaling": [
            {
                "name": check.name,
                "measured_ratio": check.measured_ratio,
                "expected_ratio": check.expected_ratio,
                "rel_error": check.rel_error,
            }
            for check in result.resistance_scaling
        ],
        "max_resistance_scaling_rel_error": result.max_resistance_scaling_rel_error,
        "wall_force_rel_error": result.wall_force_rel_error,
        "tpf_product_rel_error": result.tpf_product_rel_error,
        "quasi_static": {
            "tau_cq_values": list(result.quasi_static.tau_cq_values),
            "tracking_errors": list(result.quasi_static.tracking_errors),
            "monotonic_decrease": result.quasi_static.monotonic_decrease,
            "finest_error": result.quasi_static.finest_error,
        },
        "circuit_passed": result.circuit_passed,
        "scaling_passed": result.scaling_passed,
        "loads_passed": result.loads_passed,
        "quasi_static_passed": result.quasi_static_passed,
        "passed": result.passed,
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Return ``True`` when a payload is well-formed, sealed, and passing."""
    if payload.get("schema_version") != HALO_CURRENT_SCHEMA_VERSION:
        raise ValueError("unsupported halo current evidence schema_version")
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
    quasi = evidence["quasi_static"]
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "",
        "# Halo-Current L/R Circuit Validation",
        "",
        f"- Schema: `{evidence['schema_version']}`",
        f"- Generated (UTC): {evidence['generated_utc']}",
        f"- Target: `{evidence['target_id']}`",
        f"- Status: **{'pass' if evidence['passed'] else 'fail'}**",
        "",
        f"## Exact circuit relations (relative error, gate < {evidence['exact_tol']:.1e})",
        "",
        "| relation | value |",
        "| --- | --- |",
        f"| halo resistance R_h | {evidence['resistance_rel_error']:.3e} |",
        f"| halo inductance L_h | {evidence['inductance_rel_error']:.3e} |",
        f"| mutual inductance M | {evidence['mutual_rel_error']:.3e} |",
        f"| time constant tau_h = L_h/R_h | {evidence['time_constant_rel_error']:.3e} |",
        f"| R_h scaling laws (max) | {evidence['max_resistance_scaling_rel_error']:.3e} |",
        f"| wall force F = mu0 I_h I_p/(2 pi a) | {evidence['wall_force_rel_error']:.3e} |",
        f"| TPF product | {evidence['tpf_product_rel_error']:.3e} |",
        "",
        "## Quasi-static L/R tracking (fast-circuit limit)",
        "",
        f"- tau_cq values: {quasi['tau_cq_values']}",
        f"- tracking errors: {[f'{e:.3e}' for e in quasi['tracking_errors']]}",
        f"- monotonic decrease: {quasi['monotonic_decrease']}; finest error: {quasi['finest_error']:.3e} "
        f"(gate < {evidence['quasi_static_tol']:.1e})",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point producing schema-versioned validation evidence."""
    parser = argparse.ArgumentParser(
        description="Validate the Fitzpatrick halo-current L/R circuit against exact closed forms"
    )
    parser.add_argument("--target-id", type=str, default="local-halo-current")
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    parser.add_argument("--report", type=str, default=None, help="write sealed JSON evidence and a Markdown summary")
    args = parser.parse_args(argv)

    result = validate_halo_current()
    evidence = build_evidence(result, target_id=args.target_id)

    if args.report:
        _write_report(evidence, Path(args.report))

    if args.json_out:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print("Halo-current L/R circuit validation")
        print(
            f"  circuit:    R_h={result.resistance_rel_error:.3e} L_h={result.inductance_rel_error:.3e} "
            f"M={result.mutual_rel_error:.3e} tau_h={result.time_constant_rel_error:.3e} "
            f"{'ok' if result.circuit_passed else 'FAIL'}"
        )
        print(
            f"  loads:      wall_force={result.wall_force_rel_error:.3e} "
            f"tpf={result.tpf_product_rel_error:.3e} "
            f"{'ok' if result.loads_passed else 'FAIL'}"
        )
        print(
            f"  quasi-static: finest_err={result.quasi_static.finest_error:.3e} "
            f"monotonic={result.quasi_static.monotonic_decrease} "
            f"{'ok' if result.quasi_static_passed else 'FAIL'}"
        )
        print(f"Status: {'pass' if result.passed else 'fail'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

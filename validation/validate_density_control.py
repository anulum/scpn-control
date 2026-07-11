#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Density-control particle-balance analytic validation
"""Validate the density-control particle balance against exact closed forms.

The density controller (``src/scpn_control/control/density_controller.py``)
budgets the line-averaged density against the Greenwald limit and advances a
cylindrical particle-transport equation with normalised gas-puff, neutral-beam,
and recycling sources and a cryopump sink. The Greenwald limit, the
volume-averaged Greenwald fraction, the circular flux-surface volume elements,
the source normalisation, the cryopump sink, and the diffusion operator are all
exact algebraic or finite-volume closed forms, so the validation needs no
measured density trace and is fully self-contained.

Exact references checked against the production classes:

1. **Greenwald limit.** ``compute_greenwald_limit(I_p, a) = I_p/(pi a^2) 1e20``
   with its linear ``I_p`` and inverse-square ``a`` scaling.
2. **Greenwald fraction.** ``greenwald_fraction = <n>/n_GW`` with the
   volume-averaged density ``<n> = int n V' drho / int V' drho``.
3. **Flux-surface volumes.** ``V' = 4 pi^2 R_0 a^2 rho`` and
   ``V = 2 pi^2 R_0 (a rho)^2``.
4. **Source normalisation.** The gas-puff, neutral-beam, and recycling source
   profiles integrate to their requested particle rate (recycling scaled by the
   recycling coefficient), confirming particle conservation of the source.
5. **Neutral-beam rate.** The neutral-beam source carries
   ``P/(E_beam) / e`` particles per second.
6. **Cryopump sink.** ``cryopump_sink[-1] = S_pump n_edge / (V'[-1] drho)``.
7. **Diffusion operator.** With a spatially uniform density, zero pinch, and no
   sources, the finite-volume diffusion operator leaves the interior unchanged
   (it vanishes on constants), so only the open edge cell evolves.

References:
  Greenwald M. (2002) *Plasma Phys. Control. Fusion* 44, R27 (density limit).
  ITER Physics Basis (1999) *Nucl. Fusion* 39, 2175, §4.2 (recycling).
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
import numpy.typing as npt

from scpn_control.control.density_controller import DensityController, ParticleTransportModel

DENSITY_CONTROL_SCHEMA_VERSION = "scpn-control.density-control-validation.v1"


@dataclass(frozen=True)
class DensityConfig:
    """Geometry, plasma, and actuator parameters for the particle balance."""

    n_rho: int
    major_radius_m: float
    minor_radius_m: float
    plasma_current_ma: float
    gas_puff_rate_per_s: float
    nbi_energy_kev: float
    nbi_power_mw: float
    recycling_outflux_per_s: float
    recycling_coeff: float
    pump_speed_m3_s: float
    edge_density_m3: float
    uniform_density_m3: float

    def __post_init__(self) -> None:
        _positive_int("n_rho", self.n_rho, minimum=4)
        _positive_float("major_radius_m", self.major_radius_m)
        _positive_float("minor_radius_m", self.minor_radius_m)
        _positive_float("plasma_current_ma", self.plasma_current_ma)
        _positive_float("gas_puff_rate_per_s", self.gas_puff_rate_per_s)
        _positive_float("nbi_energy_kev", self.nbi_energy_kev)
        _positive_float("nbi_power_mw", self.nbi_power_mw)
        _positive_float("recycling_outflux_per_s", self.recycling_outflux_per_s)
        _positive_float("pump_speed_m3_s", self.pump_speed_m3_s)
        _positive_float("edge_density_m3", self.edge_density_m3)
        _positive_float("uniform_density_m3", self.uniform_density_m3)
        if not 0.0 < self.recycling_coeff <= 1.0:
            raise ValueError("recycling_coeff must lie in (0, 1]")
        if self.minor_radius_m >= self.major_radius_m:
            raise ValueError("minor radius must be smaller than the major radius")

    def model(self) -> ParticleTransportModel:
        return ParticleTransportModel(n_rho=self.n_rho, R0=self.major_radius_m, a=self.minor_radius_m)

    def controller(self) -> DensityController:
        return DensityController(self.model())

    def density_profile(self) -> npt.NDArray[np.floating[Any]]:
        """A monotone ITER-like density profile spanning the grid."""
        return np.linspace(1.1e20, 0.7e20, self.n_rho)


def default_config() -> DensityConfig:
    """An ITER-like 15 MA particle-balance geometry."""
    return DensityConfig(
        n_rho=64,
        major_radius_m=6.2,
        minor_radius_m=2.0,
        plasma_current_ma=15.0,
        gas_puff_rate_per_s=1e21,
        nbi_energy_kev=100.0,
        nbi_power_mw=33.0,
        recycling_outflux_per_s=1e21,
        recycling_coeff=0.97,
        pump_speed_m3_s=50.0,
        edge_density_m3=1e19,
        uniform_density_m3=1e20,
    )


def greenwald_limit_rel_error(config: DensityConfig) -> float:
    """Relative error of ``compute_greenwald_limit`` against ``I_p/(pi a^2) 1e20``."""
    controller = config.controller()
    analytic = config.plasma_current_ma / (math.pi * config.minor_radius_m**2) * 1e20
    measured = controller.compute_greenwald_limit(config.plasma_current_ma, config.minor_radius_m)
    return float(abs(measured - analytic) / analytic)


def greenwald_fraction_rel_error(config: DensityConfig) -> float:
    """Relative error of ``greenwald_fraction`` against ``<n>/n_GW``."""
    model = config.model()
    controller = DensityController(model)
    profile = config.density_profile()
    volume = np.sum(model.V_prime * model.drho)
    n_avg = np.sum(profile * model.V_prime * model.drho) / volume
    n_gw = controller.compute_greenwald_limit(config.plasma_current_ma, config.minor_radius_m)
    analytic = n_avg / n_gw
    measured = controller.greenwald_fraction(profile, config.plasma_current_ma, config.minor_radius_m)
    return float(abs(measured - analytic) / analytic)


def volume_element_rel_error(config: DensityConfig) -> float:
    """Relative error of the circular flux-surface volume elements ``V'`` and ``V``."""
    model = config.model()
    rho = model.rho
    v_prime = 4.0 * math.pi**2 * config.major_radius_m * config.minor_radius_m**2 * rho
    volume = 2.0 * math.pi**2 * config.major_radius_m * (config.minor_radius_m * rho) ** 2
    scale_vp = float(np.max(np.abs(v_prime)))
    scale_v = float(np.max(np.abs(volume)))
    err_vp = float(np.max(np.abs(model.V_prime - v_prime)) / scale_vp)
    err_v = float(np.max(np.abs(model.V - volume)) / scale_v)
    return max(err_vp, err_v)


def _source_integral(model: ParticleTransportModel, profile: npt.NDArray[np.float64]) -> float:
    return float(np.sum(profile * model.V_prime * model.drho))


def gas_puff_conservation_rel_error(config: DensityConfig) -> float:
    """Relative error of the gas-puff source integral against the requested rate."""
    model = config.model()
    source = model.gas_puff_source(config.gas_puff_rate_per_s)
    return float(abs(_source_integral(model, source) - config.gas_puff_rate_per_s) / config.gas_puff_rate_per_s)


def nbi_conservation_rel_error(config: DensityConfig) -> float:
    """Relative error of the neutral-beam source integral against ``P/(E) / e``."""
    model = config.model()
    source = model.nbi_source(config.nbi_energy_kev, config.nbi_power_mw)
    i_beam = config.nbi_power_mw * 1e6 / (config.nbi_energy_kev * 1e3)
    rate = i_beam / 1.6e-19
    return float(abs(_source_integral(model, source) - rate) / rate)


def recycling_conservation_rel_error(config: DensityConfig) -> float:
    """Relative error of the recycling source integral against the recycled outflux."""
    model = config.model()
    source = model.recycling_source(config.recycling_outflux_per_s, config.recycling_coeff)
    expected = config.recycling_outflux_per_s * config.recycling_coeff
    return float(abs(_source_integral(model, source) - expected) / expected)


def cryopump_sink_rel_error(config: DensityConfig) -> float:
    """Relative error of the cryopump edge sink against its closed form."""
    model = config.model()
    sink = model.cryopump_sink(config.pump_speed_m3_s, config.edge_density_m3)
    analytic = config.pump_speed_m3_s * config.edge_density_m3 / (model.V_prime[-1] * model.drho + 1e-10)
    return float(abs(sink[-1] - analytic) / analytic)


def diffusion_uniform_invariance_abs_error(config: DensityConfig) -> float:
    """Maximum interior change when the diffusion operator acts on a uniform profile.

    With a spatially uniform density, zero pinch, and no sources the finite-volume
    diffusion operator vanishes on the interior (it has no gradient to act on), so
    only the open edge cell evolves. The interior must stay bit-identical.
    """
    model = config.model()
    model.set_transport(np.ones(config.n_rho), np.zeros(config.n_rho))
    uniform = np.ones(config.n_rho) * config.uniform_density_m3
    evolved = model.step(uniform, np.zeros(config.n_rho), 1e-4)
    interior_change = np.abs(evolved[:-1] - uniform[:-1])
    return float(np.max(interior_change) / config.uniform_density_m3)


@dataclass(frozen=True)
class ScalingCheck:
    """One Greenwald-limit scaling-law observation."""

    name: str
    measured_ratio: float
    expected_ratio: float
    rel_error: float


def greenwald_scaling_checks(config: DensityConfig) -> tuple[ScalingCheck, ...]:
    """Verify the Greenwald limit scales linearly in ``I_p`` and inverse-square in ``a``."""
    controller = config.controller()
    base = controller.compute_greenwald_limit(config.plasma_current_ma, config.minor_radius_m)
    specs = (
        (
            "current_linear",
            controller.compute_greenwald_limit(2.0 * config.plasma_current_ma, config.minor_radius_m) / base,
            2.0,
        ),
        (
            "minor_radius_inverse_square",
            controller.compute_greenwald_limit(config.plasma_current_ma, 2.0 * config.minor_radius_m) / base,
            0.25,
        ),
    )
    return tuple(
        ScalingCheck(
            name=name, measured_ratio=ratio, expected_ratio=expected, rel_error=abs(ratio - expected) / expected
        )
        for name, ratio, expected in specs
    )


@dataclass(frozen=True)
class DensityValidationResult:
    """Outcome of the density-control particle-balance validation."""

    config: DensityConfig
    greenwald_limit_rel_error: float
    greenwald_fraction_rel_error: float
    volume_element_rel_error: float
    gas_puff_conservation_rel_error: float
    nbi_conservation_rel_error: float
    recycling_conservation_rel_error: float
    cryopump_sink_rel_error: float
    diffusion_uniform_invariance_abs_error: float
    scaling: tuple[ScalingCheck, ...]
    max_scaling_rel_error: float
    exact_tol: float
    invariance_tol: float
    greenwald_passed: bool
    sources_passed: bool
    diffusion_passed: bool
    scaling_passed: bool
    passed: bool


def validate_density_control(
    *, config: DensityConfig | None = None, exact_tol: float = 1e-9, invariance_tol: float = 1e-12
) -> DensityValidationResult:
    """Validate the production density-control particle balance against exact relations.

    The Greenwald limit and fraction, the flux-surface volumes, the source
    normalisation, the cryopump sink, and the scaling laws must hold to
    ``exact_tol``; the diffusion operator must leave a uniform interior unchanged
    to ``invariance_tol``.
    """
    config = config or default_config()

    gw_limit = greenwald_limit_rel_error(config)
    gw_fraction = greenwald_fraction_rel_error(config)
    vol_err = volume_element_rel_error(config)
    gas_err = gas_puff_conservation_rel_error(config)
    nbi_err = nbi_conservation_rel_error(config)
    rec_err = recycling_conservation_rel_error(config)
    cryo_err = cryopump_sink_rel_error(config)
    diffusion_err = diffusion_uniform_invariance_abs_error(config)
    scaling = greenwald_scaling_checks(config)
    max_scaling = max(check.rel_error for check in scaling)

    greenwald_passed = bool(gw_limit < exact_tol and gw_fraction < exact_tol and vol_err < exact_tol)
    sources_passed = bool(gas_err < exact_tol and nbi_err < exact_tol and rec_err < exact_tol and cryo_err < exact_tol)
    diffusion_passed = bool(diffusion_err < invariance_tol)
    scaling_passed = bool(max_scaling < exact_tol)

    passed = bool(greenwald_passed and sources_passed and diffusion_passed and scaling_passed)
    return DensityValidationResult(
        config=config,
        greenwald_limit_rel_error=gw_limit,
        greenwald_fraction_rel_error=gw_fraction,
        volume_element_rel_error=vol_err,
        gas_puff_conservation_rel_error=gas_err,
        nbi_conservation_rel_error=nbi_err,
        recycling_conservation_rel_error=rec_err,
        cryopump_sink_rel_error=cryo_err,
        diffusion_uniform_invariance_abs_error=diffusion_err,
        scaling=scaling,
        max_scaling_rel_error=max_scaling,
        exact_tol=exact_tol,
        invariance_tol=invariance_tol,
        greenwald_passed=greenwald_passed,
        sources_passed=sources_passed,
        diffusion_passed=diffusion_passed,
        scaling_passed=scaling_passed,
        passed=passed,
    )


def build_evidence(result: DensityValidationResult, *, target_id: str) -> dict[str, Any]:
    """Build a tamper-evident, schema-versioned validation evidence payload."""
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    payload: dict[str, Any] = {
        "schema_version": DENSITY_CONTROL_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "target_id": target_id,
        "config": {
            "n_rho": result.config.n_rho,
            "major_radius_m": result.config.major_radius_m,
            "minor_radius_m": result.config.minor_radius_m,
            "plasma_current_ma": result.config.plasma_current_ma,
            "gas_puff_rate_per_s": result.config.gas_puff_rate_per_s,
            "nbi_energy_kev": result.config.nbi_energy_kev,
            "nbi_power_mw": result.config.nbi_power_mw,
            "recycling_outflux_per_s": result.config.recycling_outflux_per_s,
            "recycling_coeff": result.config.recycling_coeff,
            "pump_speed_m3_s": result.config.pump_speed_m3_s,
            "edge_density_m3": result.config.edge_density_m3,
            "uniform_density_m3": result.config.uniform_density_m3,
        },
        "exact_tol": result.exact_tol,
        "invariance_tol": result.invariance_tol,
        "greenwald_limit_rel_error": result.greenwald_limit_rel_error,
        "greenwald_fraction_rel_error": result.greenwald_fraction_rel_error,
        "volume_element_rel_error": result.volume_element_rel_error,
        "gas_puff_conservation_rel_error": result.gas_puff_conservation_rel_error,
        "nbi_conservation_rel_error": result.nbi_conservation_rel_error,
        "recycling_conservation_rel_error": result.recycling_conservation_rel_error,
        "cryopump_sink_rel_error": result.cryopump_sink_rel_error,
        "diffusion_uniform_invariance_abs_error": result.diffusion_uniform_invariance_abs_error,
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
        "greenwald_passed": result.greenwald_passed,
        "sources_passed": result.sources_passed,
        "diffusion_passed": result.diffusion_passed,
        "scaling_passed": result.scaling_passed,
        "passed": result.passed,
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Return ``True`` when a payload is well-formed, sealed, and passing."""
    if payload.get("schema_version") != DENSITY_CONTROL_SCHEMA_VERSION:
        raise ValueError("unsupported density control evidence schema_version")
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


def _positive_int(name: str, value: object, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _write_report(evidence: Mapping[str, Any], json_path: Path) -> None:
    """Persist the sealed JSON evidence and a human-readable Markdown summary."""
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = json_path.with_suffix(".md")
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "",
        "# Density-Control Particle-Balance Validation",
        "",
        f"- Schema: `{evidence['schema_version']}`",
        f"- Generated (UTC): {evidence['generated_utc']}",
        f"- Target: `{evidence['target_id']}`",
        f"- Status: **{'pass' if evidence['passed'] else 'fail'}**",
        "",
        f"## Exact relations (relative error, gate < {evidence['exact_tol']:.1e})",
        "",
        "| relation | value |",
        "| --- | --- |",
        f"| Greenwald limit I_p/(pi a^2) | {evidence['greenwald_limit_rel_error']:.3e} |",
        f"| Greenwald fraction <n>/n_GW | {evidence['greenwald_fraction_rel_error']:.3e} |",
        f"| flux-surface volumes V', V | {evidence['volume_element_rel_error']:.3e} |",
        f"| gas-puff source conservation | {evidence['gas_puff_conservation_rel_error']:.3e} |",
        f"| neutral-beam source conservation | {evidence['nbi_conservation_rel_error']:.3e} |",
        f"| recycling source conservation | {evidence['recycling_conservation_rel_error']:.3e} |",
        f"| cryopump edge sink | {evidence['cryopump_sink_rel_error']:.3e} |",
        f"| Greenwald scaling laws (max) | {evidence['max_scaling_rel_error']:.3e} |",
        "",
        "## Diffusion operator on a uniform profile",
        "",
        f"- maximum interior relative change: {evidence['diffusion_uniform_invariance_abs_error']:.3e} "
        f"(gate < {evidence['invariance_tol']:.1e})",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point producing schema-versioned validation evidence."""
    parser = argparse.ArgumentParser(
        description="Validate the density-control particle balance against exact closed forms"
    )
    parser.add_argument("--target-id", type=str, default="local-density-control")
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    parser.add_argument("--report", type=str, default=None, help="write sealed JSON evidence and a Markdown summary")
    args = parser.parse_args(argv)

    result = validate_density_control()
    evidence = build_evidence(result, target_id=args.target_id)

    if args.report:
        _write_report(evidence, Path(args.report))

    if args.json_out:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print("Density-control particle-balance validation")
        print(
            f"  greenwald:  limit={result.greenwald_limit_rel_error:.3e} "
            f"fraction={result.greenwald_fraction_rel_error:.3e} volumes={result.volume_element_rel_error:.3e} "
            f"{'ok' if result.greenwald_passed else 'FAIL'}"
        )
        print(
            f"  sources:    gas={result.gas_puff_conservation_rel_error:.3e} "
            f"nbi={result.nbi_conservation_rel_error:.3e} recycling={result.recycling_conservation_rel_error:.3e} "
            f"cryo={result.cryopump_sink_rel_error:.3e} {'ok' if result.sources_passed else 'FAIL'}"
        )
        print(
            f"  diffusion:  uniform_interior={result.diffusion_uniform_invariance_abs_error:.3e} "
            f"scaling={result.max_scaling_rel_error:.3e} "
            f"{'ok' if result.diffusion_passed and result.scaling_passed else 'FAIL'}"
        )
        print(f"Status: {'pass' if result.passed else 'fail'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MARFE radiation-condensation analytic validation
"""Validate the MARFE model against repository-owned closed forms.

The MARFE model (``src/scpn_control/core/marfe.py``) exposes a bounded
radiation-condensation contract: Greenwald comparison, impurity-fraction
scaling, parallel connection-length scaling, a cooling-slope onset scan, and a
front-temperature detector. These checks validate the algebra implemented in
the repository; measured MARFE campaign or documented public-reference artefacts
remain required before facility density-limit claims can be promoted.

Exact relations checked against the production methods:

1. ``n_GW = I_p/(pi a^2)`` with ``I_p`` in MA and ``a`` in metres.
2. ``n_MARFE = n_GW sqrt(P_SOL)/(10 sqrt(f_imp))`` for the bounded heuristic
   density limit.
3. ``L_parallel = pi q95 R0`` for the edge connection length.
4. Radiation-condensation neutral stability
   ``n_e^2 f_imp |dL_Z/dT| = kappa_parallel k_parallel^2``.
5. The onset temperature returned by the scan lies on a negative cooling-slope
   point from the same scan.
6. The stability diagram changes sign across its density-limit boundary.
7. The front detector admits only ``T_min < 20 eV`` and ``T_max > 50 eV``.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scpn_control.core.marfe import DensityLimitPredictor, MARFEFrontModel, MARFEStabilityDiagram, RadiationCondensation

MARFE_ONSET_SCHEMA_VERSION = "scpn-control.marfe-onset-validation.v1"


@dataclass(frozen=True)
class MARFEConfig:
    """Geometry, impurity, and radiation-condensation inputs."""

    major_radius_m: float
    minor_radius_m: float
    q95: float
    plasma_current_ma: float
    p_sol_mw: float
    impurity: str
    impurity_fraction: float
    electron_density_20: float
    temperature_eV: float
    k_parallel_m_inv: float
    kappa_parallel: float
    temperature_scan_eV: tuple[float, ...]

    def __post_init__(self) -> None:
        _positive_float("major_radius_m", self.major_radius_m)
        _positive_float("minor_radius_m", self.minor_radius_m)
        if self.minor_radius_m >= self.major_radius_m:
            raise ValueError("minor_radius_m must be smaller than major_radius_m")
        _positive_float("q95", self.q95)
        _positive_float("plasma_current_ma", self.plasma_current_ma)
        _positive_float("p_sol_mw", self.p_sol_mw)
        _impurity_fraction(self.impurity_fraction)
        _positive_float("electron_density_20", self.electron_density_20)
        _positive_float("temperature_eV", self.temperature_eV)
        _positive_float("k_parallel_m_inv", self.k_parallel_m_inv)
        _positive_float("kappa_parallel", self.kappa_parallel)
        if not isinstance(self.impurity, str) or not self.impurity.strip():
            raise ValueError("impurity must be a non-empty string")
        _ordered_positive_scan("temperature_scan_eV", self.temperature_scan_eV)

    def radiation(self, *, electron_density_20: float | None = None) -> RadiationCondensation:
        """Build the production radiation-condensation model for this config."""
        return RadiationCondensation(
            self.impurity,
            ne_20=self.electron_density_20 if electron_density_20 is None else electron_density_20,
            f_imp=self.impurity_fraction,
        )

    def diagram(self) -> MARFEStabilityDiagram:
        """Build the production density-power stability diagram."""
        return MARFEStabilityDiagram(self.major_radius_m, self.minor_radius_m, self.q95, self.impurity)


@dataclass(frozen=True)
class ScalingCheck:
    """One exact scaling-law observation."""

    name: str
    measured_ratio: float
    expected_ratio: float
    rel_error: float


@dataclass(frozen=True)
class CriticalDensityBoundary:
    """Radiation-condensation neutral-stability bracket."""

    critical_density_20: float
    low_density_growth_rate_s: float
    high_density_growth_rate_s: float


@dataclass(frozen=True)
class OnsetTemperatureMembership:
    """Cooling-slope state for the scan-selected onset point."""

    onset_temperature_eV: float
    slope_at_onset: float
    is_scan_member: bool


@dataclass(frozen=True)
class ScanBoundaryClassification:
    """Stability-diagram states immediately below and above the density limit."""

    critical_density_20: float
    below_limit_state: int
    above_limit_state: int


@dataclass(frozen=True)
class FrontDetectionThresholds:
    """Front-detector states around the declared temperature thresholds."""

    below_threshold_is_marfe: bool
    at_temperature_threshold_is_marfe: bool
    hot_profile_is_marfe: bool


@dataclass(frozen=True)
class MARFEValidationResult:
    """Outcome of the MARFE bounded-model validation."""

    config: MARFEConfig
    greenwald_limit_rel_error: float
    greenwald_scaling: tuple[ScalingCheck, ...]
    max_greenwald_scaling_rel_error: float
    marfe_limit_rel_error: float
    marfe_limit_scaling: tuple[ScalingCheck, ...]
    max_marfe_limit_scaling_rel_error: float
    connection_length_rel_error: float
    critical_density: CriticalDensityBoundary
    onset_temperature: OnsetTemperatureMembership
    scan_boundary: ScanBoundaryClassification
    front_detection: FrontDetectionThresholds
    exact_tol: float
    greenwald_passed: bool
    marfe_limit_passed: bool
    condensation_passed: bool
    scan_passed: bool
    front_detection_passed: bool
    passed: bool


def default_config() -> MARFEConfig:
    """ITER-like edge geometry with a tungsten radiation-condensation scan."""
    return MARFEConfig(
        major_radius_m=6.2,
        minor_radius_m=2.0,
        q95=3.0,
        plasma_current_ma=15.0,
        p_sol_mw=40.0,
        impurity="W",
        impurity_fraction=1e-4,
        electron_density_20=2.0,
        temperature_eV=2000.0,
        k_parallel_m_inv=0.01,
        kappa_parallel=2000.0,
        temperature_scan_eV=tuple(float(value) for value in np.linspace(50.0, 5000.0, 500)),
    )


def greenwald_limit_rel_error(config: MARFEConfig) -> float:
    """Relative error of ``greenwald_limit`` against ``I_p/(pi a^2)``."""
    measured = DensityLimitPredictor.greenwald_limit(config.plasma_current_ma, config.minor_radius_m)
    analytic = config.plasma_current_ma / (math.pi * config.minor_radius_m**2)
    return abs(measured - analytic) / analytic


def greenwald_scaling_checks(config: MARFEConfig) -> tuple[ScalingCheck, ...]:
    """Verify the exact current and minor-radius scaling of the Greenwald limit."""
    base = DensityLimitPredictor.greenwald_limit(config.plasma_current_ma, config.minor_radius_m)
    specs = (
        (
            "current_linear",
            DensityLimitPredictor.greenwald_limit(2.0 * config.plasma_current_ma, config.minor_radius_m),
            2.0,
        ),
        (
            "minor_radius_inverse_square",
            DensityLimitPredictor.greenwald_limit(config.plasma_current_ma, 2.0 * config.minor_radius_m),
            0.25,
        ),
    )
    return tuple(_scaling_check(name, measured / base, expected) for name, measured, expected in specs)


def marfe_limit_rel_error(config: MARFEConfig) -> float:
    """Relative error of ``marfe_limit`` against its declared closed form."""
    measured = DensityLimitPredictor.marfe_limit(
        config.plasma_current_ma,
        config.minor_radius_m,
        config.p_sol_mw,
        config.impurity,
        config.impurity_fraction,
    )
    n_gw = config.plasma_current_ma / (math.pi * config.minor_radius_m**2)
    analytic = n_gw * math.sqrt(config.p_sol_mw) / (10.0 * math.sqrt(config.impurity_fraction))
    return abs(measured - analytic) / analytic


def marfe_limit_scaling_checks(config: MARFEConfig) -> tuple[ScalingCheck, ...]:
    """Verify power and impurity-fraction scaling for the bounded MARFE limit."""
    base = DensityLimitPredictor.marfe_limit(
        config.plasma_current_ma,
        config.minor_radius_m,
        config.p_sol_mw,
        config.impurity,
        config.impurity_fraction,
    )
    specs = (
        (
            "power_sqrt",
            DensityLimitPredictor.marfe_limit(
                config.plasma_current_ma,
                config.minor_radius_m,
                2.0 * config.p_sol_mw,
                config.impurity,
                config.impurity_fraction,
            ),
            2.0**0.5,
        ),
        (
            "impurity_inverse_sqrt",
            DensityLimitPredictor.marfe_limit(
                config.plasma_current_ma,
                config.minor_radius_m,
                config.p_sol_mw,
                config.impurity,
                2.0 * config.impurity_fraction,
            ),
            2.0**-0.5,
        ),
    )
    return tuple(_scaling_check(name, measured / base, expected) for name, measured, expected in specs)


def connection_length_rel_error(config: MARFEConfig) -> float:
    """Relative error of ``connection_length_m`` against ``pi q95 R0``."""
    measured = config.diagram().connection_length_m
    analytic = math.pi * config.q95 * config.major_radius_m
    return abs(measured - analytic) / analytic


def critical_density_boundary(config: MARFEConfig) -> CriticalDensityBoundary:
    """Bracket the neutral-stability density from the production growth rate."""
    density = config.radiation().critical_density(config.temperature_eV, config.k_parallel_m_inv, config.kappa_parallel)
    low = config.radiation(electron_density_20=0.99 * density).growth_rate(
        config.temperature_eV,
        config.k_parallel_m_inv,
        config.kappa_parallel,
    )
    high = config.radiation(electron_density_20=1.01 * density).growth_rate(
        config.temperature_eV,
        config.k_parallel_m_inv,
        config.kappa_parallel,
    )
    return CriticalDensityBoundary(
        critical_density_20=density,
        low_density_growth_rate_s=low,
        high_density_growth_rate_s=high,
    )


def onset_temperature_membership(config: MARFEConfig) -> OnsetTemperatureMembership:
    """Confirm the scan-selected onset belongs to the scan and has negative slope."""
    model = config.radiation()
    scan = np.asarray(config.temperature_scan_eV, dtype=float)
    onset = model.onset_temperature(scan)
    slope = model._dL_dT(onset)
    is_member = bool(np.any(np.isclose(scan, onset, rtol=0.0, atol=1e-12)))
    return OnsetTemperatureMembership(onset_temperature_eV=onset, slope_at_onset=slope, is_scan_member=is_member)


def scan_boundary_classification(config: MARFEConfig) -> ScanBoundaryClassification:
    """Confirm the density-power diagram changes sign across its own boundary."""
    diagram = config.diagram()
    connection_factor = diagram._REFERENCE_CONNECTION_LENGTH_M / diagram.connection_length_m
    density_limit = (
        DensityLimitPredictor.marfe_limit(
            config.plasma_current_ma,
            config.minor_radius_m,
            config.p_sol_mw,
            config.impurity,
            config.impurity_fraction,
        )
        * connection_factor
    )
    states = diagram.scan_density_power(
        np.array([0.99 * density_limit, 1.01 * density_limit]),
        np.array([config.p_sol_mw]),
    )
    return ScanBoundaryClassification(
        critical_density_20=density_limit,
        below_limit_state=int(states[0, 0]),
        above_limit_state=int(states[1, 0]),
    )


def front_detection_thresholds() -> FrontDetectionThresholds:
    """Confirm the detached-front detector's two strict temperature thresholds."""
    below = MARFEFrontModel(L_par=100.0, kappa_par=20.0, q_perp=10.0, impurity="W", f_imp=1e-2)
    below.T = np.linspace(19.0, 60.0, below.n_s)
    at_threshold = MARFEFrontModel(L_par=100.0, kappa_par=20.0, q_perp=10.0, impurity="W", f_imp=1e-2)
    at_threshold.T = np.linspace(20.0, 60.0, at_threshold.n_s)
    hot = MARFEFrontModel(L_par=100.0, kappa_par=20.0, q_perp=10.0, impurity="W", f_imp=1e-2)
    hot.T = np.linspace(30.0, 80.0, hot.n_s)
    return FrontDetectionThresholds(
        below_threshold_is_marfe=below.is_marfe(),
        at_temperature_threshold_is_marfe=at_threshold.is_marfe(),
        hot_profile_is_marfe=hot.is_marfe(),
    )


def validate_marfe_onset(
    *,
    config: MARFEConfig | None = None,
    exact_tol: float = 1e-9,
) -> MARFEValidationResult:
    """Validate bounded MARFE algebra and threshold contracts."""
    config = config or default_config()
    _positive_float("exact_tol", exact_tol)
    greenwald_err = greenwald_limit_rel_error(config)
    greenwald_scaling = greenwald_scaling_checks(config)
    max_greenwald_scaling = max(check.rel_error for check in greenwald_scaling)
    marfe_err = marfe_limit_rel_error(config)
    marfe_scaling = marfe_limit_scaling_checks(config)
    max_marfe_scaling = max(check.rel_error for check in marfe_scaling)
    connection_err = connection_length_rel_error(config)
    critical = critical_density_boundary(config)
    onset = onset_temperature_membership(config)
    scan = scan_boundary_classification(config)
    front = front_detection_thresholds()

    greenwald_passed = greenwald_err < exact_tol and max_greenwald_scaling < exact_tol
    marfe_limit_passed = marfe_err < exact_tol and max_marfe_scaling < exact_tol and connection_err < exact_tol
    condensation_passed = (
        critical.critical_density_20 > 0.0
        and critical.low_density_growth_rate_s < 0.0
        and critical.high_density_growth_rate_s > 0.0
        and onset.slope_at_onset < 0.0
        and onset.is_scan_member
    )
    scan_passed = scan.critical_density_20 > 0.0 and scan.below_limit_state == 1 and scan.above_limit_state == -1
    front_detection_passed = (
        front.below_threshold_is_marfe
        and not front.at_temperature_threshold_is_marfe
        and not front.hot_profile_is_marfe
    )
    passed = greenwald_passed and marfe_limit_passed and condensation_passed and scan_passed and front_detection_passed

    return MARFEValidationResult(
        config=config,
        greenwald_limit_rel_error=greenwald_err,
        greenwald_scaling=greenwald_scaling,
        max_greenwald_scaling_rel_error=max_greenwald_scaling,
        marfe_limit_rel_error=marfe_err,
        marfe_limit_scaling=marfe_scaling,
        max_marfe_limit_scaling_rel_error=max_marfe_scaling,
        connection_length_rel_error=connection_err,
        critical_density=critical,
        onset_temperature=onset,
        scan_boundary=scan,
        front_detection=front,
        exact_tol=exact_tol,
        greenwald_passed=greenwald_passed,
        marfe_limit_passed=marfe_limit_passed,
        condensation_passed=condensation_passed,
        scan_passed=scan_passed,
        front_detection_passed=front_detection_passed,
        passed=passed,
    )


def build_evidence(result: MARFEValidationResult, *, target_id: str) -> dict[str, Any]:
    """Build a tamper-evident, schema-versioned validation evidence payload."""
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    payload: dict[str, Any] = {
        "schema_version": MARFE_ONSET_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "target_id": target_id,
        "claim_status": "bounded_model",
        "public_claim_allowed": False,
        "config": {
            "major_radius_m": result.config.major_radius_m,
            "minor_radius_m": result.config.minor_radius_m,
            "q95": result.config.q95,
            "plasma_current_ma": result.config.plasma_current_ma,
            "p_sol_mw": result.config.p_sol_mw,
            "impurity": result.config.impurity,
            "impurity_fraction": result.config.impurity_fraction,
            "electron_density_20": result.config.electron_density_20,
            "temperature_eV": result.config.temperature_eV,
            "k_parallel_m_inv": result.config.k_parallel_m_inv,
            "kappa_parallel": result.config.kappa_parallel,
        },
        "exact_tol": result.exact_tol,
        "greenwald_limit_rel_error": result.greenwald_limit_rel_error,
        "greenwald_scaling": [_scaling_payload(check) for check in result.greenwald_scaling],
        "max_greenwald_scaling_rel_error": result.max_greenwald_scaling_rel_error,
        "marfe_limit_rel_error": result.marfe_limit_rel_error,
        "marfe_limit_scaling": [_scaling_payload(check) for check in result.marfe_limit_scaling],
        "max_marfe_limit_scaling_rel_error": result.max_marfe_limit_scaling_rel_error,
        "connection_length_rel_error": result.connection_length_rel_error,
        "critical_density": {
            "critical_density_20": result.critical_density.critical_density_20,
            "low_density_growth_rate_s": result.critical_density.low_density_growth_rate_s,
            "high_density_growth_rate_s": result.critical_density.high_density_growth_rate_s,
        },
        "onset_temperature": {
            "onset_temperature_eV": result.onset_temperature.onset_temperature_eV,
            "slope_at_onset": result.onset_temperature.slope_at_onset,
            "is_scan_member": result.onset_temperature.is_scan_member,
        },
        "scan_boundary": {
            "critical_density_20": result.scan_boundary.critical_density_20,
            "below_limit_state": result.scan_boundary.below_limit_state,
            "above_limit_state": result.scan_boundary.above_limit_state,
        },
        "front_detection": {
            "below_threshold_is_marfe": result.front_detection.below_threshold_is_marfe,
            "at_temperature_threshold_is_marfe": result.front_detection.at_temperature_threshold_is_marfe,
            "hot_profile_is_marfe": result.front_detection.hot_profile_is_marfe,
        },
        "greenwald_passed": result.greenwald_passed,
        "marfe_limit_passed": result.marfe_limit_passed,
        "condensation_passed": result.condensation_passed,
        "scan_passed": result.scan_passed,
        "front_detection_passed": result.front_detection_passed,
        "passed": result.passed,
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Return ``True`` when a payload is well-formed, sealed, and passing."""
    if payload.get("schema_version") != MARFE_ONSET_SCHEMA_VERSION:
        raise ValueError("unsupported MARFE onset evidence schema_version")
    declared = payload.get("payload_sha256")
    if not _is_sha256(declared):
        raise ValueError("payload_sha256 must be a SHA-256 hex digest")
    if declared != _payload_sha256(payload):
        raise ValueError("payload_sha256 does not match payload")
    return bool(payload.get("passed"))


def _scaling_check(name: str, measured_ratio: float, expected_ratio: float) -> ScalingCheck:
    return ScalingCheck(
        name=name,
        measured_ratio=measured_ratio,
        expected_ratio=expected_ratio,
        rel_error=abs(measured_ratio - expected_ratio) / expected_ratio,
    )


def _scaling_payload(check: ScalingCheck) -> dict[str, float | str]:
    return {
        "name": check.name,
        "measured_ratio": check.measured_ratio,
        "expected_ratio": check.expected_ratio,
        "rel_error": check.rel_error,
    }


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


def _impurity_fraction(value: object) -> float:
    result = _positive_float("impurity_fraction", value)
    if result > 1.0:
        raise ValueError("impurity_fraction must lie in (0, 1]")
    return result


def _ordered_positive_scan(name: str, values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        raise ValueError(f"{name} must be non-empty")
    scan = tuple(_positive_float(name, value) for value in values)
    if any(next_value <= value for value, next_value in zip(scan, scan[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return scan


def _write_report(evidence: Mapping[str, Any], json_path: Path) -> None:
    """Persist sealed JSON evidence and a Markdown summary."""
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config = evidence["config"]
    critical = evidence["critical_density"]
    onset = evidence["onset_temperature"]
    scan = evidence["scan_boundary"]
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "",
        "# MARFE Radiation-Condensation Validation",
        "",
        f"- Schema: `{evidence['schema_version']}`",
        f"- Generated (UTC): {evidence['generated_utc']}",
        f"- Target: `{evidence['target_id']}`",
        f"- Claim status: `{evidence['claim_status']}`",
        f"- Public claim allowed: `{evidence['public_claim_allowed']}`",
        f"- Geometry: R0={config['major_radius_m']} m, a={config['minor_radius_m']} m, q95={config['q95']}",
        f"- Impurity: {config['impurity']} at fraction {config['impurity_fraction']}",
        f"- Status: **{'pass' if evidence['passed'] else 'fail'}**",
        "",
        f"## Exact closed-form references (relative error, gate < {evidence['exact_tol']:.1e})",
        "",
        "| reference | value |",
        "| --- | --- |",
        f"| Greenwald limit | {evidence['greenwald_limit_rel_error']:.3e} |",
        f"| Greenwald scaling | {evidence['max_greenwald_scaling_rel_error']:.3e} |",
        f"| MARFE density-limit scaling | {evidence['marfe_limit_rel_error']:.3e} |",
        f"| MARFE scaling exponents | {evidence['max_marfe_limit_scaling_rel_error']:.3e} |",
        f"| connection length L_parallel = pi q95 R0 | {evidence['connection_length_rel_error']:.3e} |",
        "",
        "## Radiation-condensation boundary",
        "",
        f"- Critical density: {critical['critical_density_20']:.6g} x 10^20 m^-3",
        f"- Growth below critical: {critical['low_density_growth_rate_s']:.6g} s^-1",
        f"- Growth above critical: {critical['high_density_growth_rate_s']:.6g} s^-1",
        f"- Onset temperature: {onset['onset_temperature_eV']:.6g} eV",
        f"- Cooling slope at onset: {onset['slope_at_onset']:.6e}",
        "",
        "## Stability diagram and front detector",
        "",
        f"- Scan critical density: {scan['critical_density_20']:.6g} x 10^20 m^-3",
        f"- Below/above limit states: {scan['below_limit_state']} / {scan['above_limit_state']}",
        f"- Front detection passed: `{evidence['front_detection_passed']}`",
        "",
        "This is bounded local-regression evidence. Measured MARFE campaign or documented public-reference",
        "artefacts remain required for facility density-limit claims.",
    ]
    json_path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point producing schema-versioned validation evidence."""
    parser = argparse.ArgumentParser(description="Validate MARFE bounded-model algebra against exact closed forms")
    parser.add_argument("--target-id", type=str, default="local-marfe-onset")
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    parser.add_argument("--report", type=str, default=None, help="write sealed JSON evidence and a Markdown summary")
    args = parser.parse_args(argv)

    result = validate_marfe_onset()
    evidence = build_evidence(result, target_id=args.target_id)

    if args.report:
        _write_report(evidence, Path(args.report))

    if args.json_out:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print("MARFE radiation-condensation validation")
        print(
            f"  Greenwald + MARFE limit: gw={result.greenwald_limit_rel_error:.3e} "
            f"marfe={result.marfe_limit_rel_error:.3e} "
            f"{'ok' if result.greenwald_passed and result.marfe_limit_passed else 'FAIL'}"
        )
        print(
            f"  critical density:       ncrit={result.critical_density.critical_density_20:.6g}e20 "
            f"{'ok' if result.condensation_passed else 'FAIL'}"
        )
        print(
            f"  scan + front detector:  scan={result.scan_boundary.below_limit_state}/"
            f"{result.scan_boundary.above_limit_state} "
            f"{'ok' if result.scan_passed and result.front_detection_passed else 'FAIL'}"
        )
        print(f"Status: {'pass' if result.passed else 'fail'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

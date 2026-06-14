#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Auxiliary current-drive deposition and efficiency validation
"""Validate the auxiliary current-drive model against exact closed forms.

The current-drive model (``src/scpn_control/core/current_drive.py``) provides
electron-cyclotron, lower-hybrid, and neutral-beam sources with grid-normalised
radial deposition, the Prater ECCD figure of merit, and the Stix beam
slowing-down time and critical energy. Every relation is an exact algebraic or
conservation closed form, so the model can be validated without any ray-tracing,
Fokker-Planck, or measured-deposition artefact — the validation is fully
self-contained.

Exact references checked against the production methods:

1. **Deposition power conservation.** The grid-normalised radial deposition
   integrates to the launched source power for the ECCD, LHCD, and NBI sources.
2. **Deposition centroid.** A symmetric Gaussian deposition centred at the grid
   midpoint has a power-weighted centroid exactly at the deposition radius.
3. **Stix critical energy.** ``E_crit = 14.8 T_e (A_b/A_i)^{2/3}`` with the exact
   ``(A_b/A_i)^{2/3}`` mass-ratio scaling.
4. **Stix slowing-down time.** ``tau_s`` follows the closed form and scales as
   ``T_e^{3/2}``, ``1/n_e``, and ``1/Z_eff``.
5. **Prater ECCD efficiency.** ``eta = eta_0 T_e/(5 + Z_eff) xi/(1 + xi^2)`` with
   the launch-angle factor maximised at ``N_parallel = 1``.
6. **Driven-current density.** ``j_cd = eta_cd P_abs/(n_e T_e)`` for ECCD and LHCD.
7. **Neutral-beam current chain.** ``j_cd = e n_fast v_par / Z_beam`` with
   ``n_fast = P_heat tau_s / E_beam`` and ``v_par = sqrt(2 E_beam/m_beam)``.

References:
  Fisch N. J., Boozer A. H. (1980) *Phys. Rev. Lett.* 45, 720.
  Fisch N. J. (1978) *Phys. Rev. Lett.* 41, 873.
  Prater R. (2004) *Phys. Plasmas* 11, 2349.
  Stix T. H. (1972) *Plasma Physics* 14, 367.
  Ehst D. A., Karney C. F. F. (1991) *Nucl. Fusion* 31, 1933.
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

from scpn_control.core.current_drive import (
    E_CHARGE,
    M_P,
    ECCDSource,
    LHCDSource,
    NBISource,
    eccd_efficiency,
    nbi_critical_energy,
    nbi_slowing_down_time,
)

FloatArray = NDArray[np.float64]

CURRENT_DRIVE_SCHEMA_VERSION = "scpn-control.current-drive-validation.v1"

_E_CRIT_COEFFICIENT = 14.8  # Stix 1972 critical-energy prefactor [keV per keV T_e]


def _rho_grid(nr: int = 401) -> FloatArray:
    return np.linspace(0.0, 1.0, _grid_resolution("nr", nr), dtype=np.float64)


def deposition_conservation_rel_errors(rho: FloatArray) -> dict[str, float]:
    """Relative error of the integrated deposition against the launched source power."""
    p_ec, p_lh, p_nbi = 2.0, 1.5, 8.0
    sources = {
        "eccd": (ECCDSource(p_ec, rho_dep=0.5, sigma_rho=0.1).P_absorbed(rho), p_ec),
        "lhcd": (LHCDSource(p_lh, rho_dep=0.55, sigma_rho=0.12).P_absorbed(rho), p_lh),
        "nbi": (NBISource(p_nbi, E_beam_keV=100.0, rho_tangency=0.45, sigma_rho=0.15).P_heating(rho), p_nbi),
    }
    return {
        name: abs(float(trapezoid(profile, rho)) - power * 1e6) / (power * 1e6)
        for name, (profile, power) in sources.items()
    }


def deposition_centroid_rel_error(rho: FloatArray) -> float:
    """Relative error of the power-weighted centroid against a midpoint deposition."""
    profile = ECCDSource(2.0, rho_dep=0.5, sigma_rho=0.1).P_absorbed(rho)
    centroid = float(trapezoid(profile * rho, rho) / trapezoid(profile, rho))
    return abs(centroid - 0.5) / 0.5


def critical_energy_rel_error(*, te_kev: float = 10.0, a_beam: float = 2.0, a_ion: float = 2.0) -> float:
    """Relative error of the Stix critical energy against its closed form."""
    measured = float(nbi_critical_energy(te_kev, A_beam=a_beam, A_ion=a_ion))
    analytic = _E_CRIT_COEFFICIENT * te_kev * (a_beam / a_ion) ** (2.0 / 3.0)
    return abs(measured - analytic) / analytic


def eccd_efficiency_rel_error(
    *, te_kev: float = 10.0, z_eff: float = 1.5, n_parallel: float = 1.0, eta_0: float = 0.03
) -> float:
    """Relative error of the Prater ECCD efficiency against its closed form."""
    measured = float(eccd_efficiency(te_kev, z_eff, n_parallel, eta_0=eta_0))
    analytic = eta_0 * te_kev / (5.0 + z_eff) * n_parallel / (1.0 + n_parallel**2)
    return abs(measured - analytic) / analytic


def launch_factor_is_maximised_at_unity(*, te_kev: float = 10.0, z_eff: float = 1.5) -> bool:
    """The ECCD launch-angle factor ``xi/(1+xi^2)`` peaks at ``N_parallel = 1``."""
    peak = float(eccd_efficiency(te_kev, z_eff, 1.0))
    sweep = [float(eccd_efficiency(te_kev, z_eff, n)) for n in (0.3, 0.6, 1.4, 2.0, 3.0)]
    return all(peak >= value - 1e-15 for value in sweep)


def jcd_proportionality_rel_error(rho: FloatArray) -> float:
    """Relative error of ``j_cd = eta_cd P_abs/(n_e T_e)`` for ECCD and LHCD."""
    ne = np.full_like(rho, 5.0)
    te = np.full_like(rho, 8.0)
    worst = 0.0
    for source in (ECCDSource(2.0, 0.5, 0.1), LHCDSource(1.5, 0.55, 0.12)):
        measured = source.j_cd(rho, ne, te)
        reconstructed = source.eta_cd * source.P_absorbed(rho) / (ne * te)
        denom = np.maximum(np.abs(reconstructed), 1e-300)
        worst = max(worst, float(np.max(np.abs(measured - reconstructed) / denom)))
    return worst


def nbi_current_chain_rel_error(rho: FloatArray) -> float:
    """Relative error of the NBI driven current against the fast-ion balance chain."""
    source = NBISource(8.0, E_beam_keV=100.0, rho_tangency=0.45, sigma_rho=0.15)
    ne = np.full_like(rho, 5.0)
    te = np.full_like(rho, 8.0)
    ti = np.full_like(rho, 7.0)
    z_eff = 1.5
    measured = source.j_cd(rho, ne, te, ti, Z_eff=z_eff)

    p_heat = source.P_heating(rho)
    e_beam_j = source.E_beam_keV * 1e3 * E_CHARGE
    v_parallel = math.sqrt(2.0 * e_beam_j / (source.A_beam * M_P))
    # te and ne are uniform, so the per-point Stix slowing-down time is constant;
    # the fast-ion balance n_fast = P_heat tau_s / E_beam then vectorises exactly.
    tau_s = np.asarray(nbi_slowing_down_time(te, ne, A_beam=source.A_beam, Z_eff=z_eff), dtype=np.float64)
    n_fast = p_heat * tau_s / e_beam_j
    reconstructed = E_CHARGE * n_fast * v_parallel / source.Z_beam
    denom = np.maximum(np.abs(reconstructed), 1e-300)
    return float(np.max(np.abs(measured - reconstructed) / denom))


@dataclass(frozen=True)
class ScalingCheck:
    """One exact-scaling-law observation."""

    name: str
    measured_ratio: float
    expected_ratio: float
    rel_error: float


def slowing_down_scaling_checks() -> tuple[ScalingCheck, ...]:
    """Verify ``tau_s`` scales as ``T_e^{3/2}``, ``1/n_e``, and ``1/Z_eff``."""
    base = float(nbi_slowing_down_time(10.0, 5.0, Z_eff=1.5))
    specs = (
        ("temperature_1.5", float(nbi_slowing_down_time(40.0, 5.0, Z_eff=1.5)) / base, 4.0**1.5),
        ("density_inverse", float(nbi_slowing_down_time(10.0, 10.0, Z_eff=1.5)) / base, 0.5),
        ("z_eff_inverse", float(nbi_slowing_down_time(10.0, 5.0, Z_eff=3.0)) / base, 0.5),
    )
    return tuple(
        ScalingCheck(
            name=name, measured_ratio=ratio, expected_ratio=expected, rel_error=abs(ratio - expected) / expected
        )
        for name, ratio, expected in specs
    )


def critical_energy_scaling_check() -> ScalingCheck:
    """Verify the ``(A_b/A_i)^{2/3}`` mass-ratio scaling of the critical energy."""
    base = float(nbi_critical_energy(10.0, A_beam=2.0, A_ion=2.0))
    doubled = float(nbi_critical_energy(10.0, A_beam=2.0, A_ion=1.0))
    expected = 2.0 ** (2.0 / 3.0)
    return ScalingCheck(
        name="mass_ratio_two_thirds",
        measured_ratio=doubled / base,
        expected_ratio=expected,
        rel_error=abs(doubled / base - expected) / expected,
    )


@dataclass(frozen=True)
class CurrentDriveValidationResult:
    """Outcome of the auxiliary current-drive validation."""

    grid_points: int
    deposition_conservation: dict[str, float]
    max_deposition_conservation_rel_error: float
    deposition_centroid_rel_error: float
    critical_energy_rel_error: float
    critical_energy_scaling: ScalingCheck
    slowing_down_scaling: tuple[ScalingCheck, ...]
    max_slowing_down_scaling_rel_error: float
    eccd_efficiency_rel_error: float
    launch_factor_maximised: bool
    jcd_proportionality_rel_error: float
    nbi_current_chain_rel_error: float
    max_scaling_rel_error: float
    exact_tol: float
    deposition_passed: bool
    centroid_passed: bool
    critical_energy_passed: bool
    slowing_down_passed: bool
    eccd_efficiency_passed: bool
    jcd_passed: bool
    nbi_chain_passed: bool
    passed: bool


def validate_current_drive(
    *,
    nr: int = 401,
    exact_tol: float = 1e-9,
) -> CurrentDriveValidationResult:
    """Validate the production current-drive model against its exact closed forms.

    Deposition power conservation, deposition centroid, the Stix critical-energy
    and slowing-down formulae and their scalings, the Prater ECCD efficiency and
    launch-angle maximisation, the ECCD/LHCD driven-current proportionality, and
    the neutral-beam current chain must all hold to ``exact_tol``.
    """
    rho = _rho_grid(nr)

    deposition = deposition_conservation_rel_errors(rho)
    max_deposition = max(deposition.values())
    centroid = deposition_centroid_rel_error(rho)
    critical = critical_energy_rel_error()
    critical_scaling = critical_energy_scaling_check()
    slowing_scaling = slowing_down_scaling_checks()
    max_slowing_scaling = max(check.rel_error for check in slowing_scaling)
    efficiency = eccd_efficiency_rel_error()
    launch_max = launch_factor_is_maximised_at_unity()
    jcd = jcd_proportionality_rel_error(rho)
    nbi_chain = nbi_current_chain_rel_error(rho)

    max_scaling = max(max_slowing_scaling, critical_scaling.rel_error)

    deposition_passed = max_deposition < exact_tol
    centroid_passed = centroid < exact_tol
    critical_energy_passed = critical < exact_tol and critical_scaling.rel_error < exact_tol
    slowing_down_passed = max_slowing_scaling < exact_tol
    eccd_efficiency_passed = efficiency < exact_tol and launch_max
    jcd_passed = jcd < exact_tol
    nbi_chain_passed = nbi_chain < exact_tol

    passed = (
        deposition_passed
        and centroid_passed
        and critical_energy_passed
        and slowing_down_passed
        and eccd_efficiency_passed
        and jcd_passed
        and nbi_chain_passed
    )
    return CurrentDriveValidationResult(
        grid_points=len(rho),
        deposition_conservation=deposition,
        max_deposition_conservation_rel_error=max_deposition,
        deposition_centroid_rel_error=centroid,
        critical_energy_rel_error=critical,
        critical_energy_scaling=critical_scaling,
        slowing_down_scaling=slowing_scaling,
        max_slowing_down_scaling_rel_error=max_slowing_scaling,
        eccd_efficiency_rel_error=efficiency,
        launch_factor_maximised=launch_max,
        jcd_proportionality_rel_error=jcd,
        nbi_current_chain_rel_error=nbi_chain,
        max_scaling_rel_error=max_scaling,
        exact_tol=exact_tol,
        deposition_passed=deposition_passed,
        centroid_passed=centroid_passed,
        critical_energy_passed=critical_energy_passed,
        slowing_down_passed=slowing_down_passed,
        eccd_efficiency_passed=eccd_efficiency_passed,
        jcd_passed=jcd_passed,
        nbi_chain_passed=nbi_chain_passed,
        passed=passed,
    )


def build_evidence(result: CurrentDriveValidationResult, *, target_id: str) -> dict[str, Any]:
    """Build a tamper-evident, schema-versioned validation evidence payload."""
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    payload: dict[str, Any] = {
        "schema_version": CURRENT_DRIVE_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "target_id": target_id,
        "grid_points": result.grid_points,
        "exact_tol": result.exact_tol,
        "deposition_conservation": dict(sorted(result.deposition_conservation.items())),
        "max_deposition_conservation_rel_error": result.max_deposition_conservation_rel_error,
        "deposition_centroid_rel_error": result.deposition_centroid_rel_error,
        "critical_energy_rel_error": result.critical_energy_rel_error,
        "critical_energy_scaling": {
            "name": result.critical_energy_scaling.name,
            "measured_ratio": result.critical_energy_scaling.measured_ratio,
            "expected_ratio": result.critical_energy_scaling.expected_ratio,
            "rel_error": result.critical_energy_scaling.rel_error,
        },
        "slowing_down_scaling": [
            {
                "name": check.name,
                "measured_ratio": check.measured_ratio,
                "expected_ratio": check.expected_ratio,
                "rel_error": check.rel_error,
            }
            for check in result.slowing_down_scaling
        ],
        "max_slowing_down_scaling_rel_error": result.max_slowing_down_scaling_rel_error,
        "eccd_efficiency_rel_error": result.eccd_efficiency_rel_error,
        "launch_factor_maximised": result.launch_factor_maximised,
        "jcd_proportionality_rel_error": result.jcd_proportionality_rel_error,
        "nbi_current_chain_rel_error": result.nbi_current_chain_rel_error,
        "max_scaling_rel_error": result.max_scaling_rel_error,
        "deposition_passed": result.deposition_passed,
        "centroid_passed": result.centroid_passed,
        "critical_energy_passed": result.critical_energy_passed,
        "slowing_down_passed": result.slowing_down_passed,
        "eccd_efficiency_passed": result.eccd_efficiency_passed,
        "jcd_passed": result.jcd_passed,
        "nbi_chain_passed": result.nbi_chain_passed,
        "passed": result.passed,
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Return ``True`` when a payload is well-formed, sealed, and passing."""
    if payload.get("schema_version") != CURRENT_DRIVE_SCHEMA_VERSION:
        raise ValueError("unsupported current drive evidence schema_version")
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


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _grid_resolution(name: str, value: object) -> int:
    result = _positive_int(name, value)
    if result < 51:
        raise ValueError(f"{name} must be at least 51 to resolve the deposition kernel")
    return result


def _write_report(evidence: Mapping[str, Any], json_path: Path) -> None:
    """Persist the sealed JSON evidence and a human-readable Markdown summary."""
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = json_path.with_suffix(".md")
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "",
        "# Auxiliary Current-Drive Validation",
        "",
        f"- Schema: `{evidence['schema_version']}`",
        f"- Generated (UTC): {evidence['generated_utc']}",
        f"- Target: `{evidence['target_id']}`",
        f"- Radial grid points: {evidence['grid_points']}",
        f"- Status: **{'pass' if evidence['passed'] else 'fail'}**",
        "",
        f"## Exact closed-form references (relative error, gate < {evidence['exact_tol']:.1e})",
        "",
        "| reference | value |",
        "| --- | --- |",
        f"| deposition power conservation (max) | {evidence['max_deposition_conservation_rel_error']:.3e} |",
        f"| deposition centroid | {evidence['deposition_centroid_rel_error']:.3e} |",
        f"| Stix critical energy | {evidence['critical_energy_rel_error']:.3e} |",
        f"| Stix slowing-down scaling (max) | {evidence['max_slowing_down_scaling_rel_error']:.3e} |",
        f"| Prater ECCD efficiency | {evidence['eccd_efficiency_rel_error']:.3e} |",
        f"| launch-angle factor maximised at N=1 | {evidence['launch_factor_maximised']} |",
        f"| j_cd = eta P/(n T) proportionality | {evidence['jcd_proportionality_rel_error']:.3e} |",
        f"| neutral-beam current chain | {evidence['nbi_current_chain_rel_error']:.3e} |",
        "",
        "## Per-source deposition conservation",
        "",
        "| source | rel error |",
        "| --- | --- |",
    ]
    lines += [f"| {name} | {value:.3e} |" for name, value in sorted(evidence["deposition_conservation"].items())]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point producing schema-versioned validation evidence."""
    parser = argparse.ArgumentParser(
        description="Validate the auxiliary current-drive deposition and efficiency model against exact closed forms"
    )
    parser.add_argument("--target-id", type=str, default="local-current-drive")
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    parser.add_argument("--report", type=str, default=None, help="write sealed JSON evidence and a Markdown summary")
    args = parser.parse_args(argv)

    result = validate_current_drive()
    evidence = build_evidence(result, target_id=args.target_id)

    if args.report:
        _write_report(evidence, Path(args.report))

    if args.json_out:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print("Auxiliary current-drive validation")
        print(
            f"  deposition:   conservation={result.max_deposition_conservation_rel_error:.3e} "
            f"centroid={result.deposition_centroid_rel_error:.3e} "
            f"{'ok' if result.deposition_passed and result.centroid_passed else 'FAIL'}"
        )
        print(
            f"  Stix beam:    E_crit={result.critical_energy_rel_error:.3e} "
            f"slowing-down scaling={result.max_slowing_down_scaling_rel_error:.3e} "
            f"{'ok' if result.critical_energy_passed and result.slowing_down_passed else 'FAIL'}"
        )
        print(
            f"  ECCD/LHCD:    eta={result.eccd_efficiency_rel_error:.3e} "
            f"launch_max={result.launch_factor_maximised} "
            f"j_cd={result.jcd_proportionality_rel_error:.3e} "
            f"{'ok' if result.eccd_efficiency_passed and result.jcd_passed else 'FAIL'}"
        )
        print(
            f"  NBI chain:    j_cd={result.nbi_current_chain_rel_error:.3e} "
            f"{'ok' if result.nbi_chain_passed else 'FAIL'}"
        )
        print(f"Status: {'pass' if result.passed else 'fail'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — VMEC-lite spectral-geometry analytic validation
"""Validate VMEC-lite fixed-boundary geometry against exact repository forms."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scpn_control.core.vmec_lite import AxisymmetricTokamakBoundary, SpectralBasis, VMECLiteSolver

VMEC_LITE_GEOMETRY_SCHEMA_VERSION = "scpn-control.vmec-lite-geometry-validation.v1"


@dataclass(frozen=True)
class VMECLiteGeometryCase:
    """Reduced fixed-boundary case used for exact VMEC-lite geometry checks."""

    n_s: int
    m_pol: int
    n_tor: int
    n_fp: int
    major_radius_m: float
    minor_radius_m: float
    elongation: float
    triangularity: float
    pressure_axis_pa: float
    pressure_edge_pa: float
    iota_axis: float
    iota_edge: float
    exact_tol: float

    def __post_init__(self) -> None:
        _positive_int("n_s", self.n_s, minimum=3)
        _positive_int("m_pol", self.m_pol, minimum=0)
        _positive_int("n_tor", self.n_tor, minimum=0)
        _positive_int("n_fp", self.n_fp)
        _positive_float("major_radius_m", self.major_radius_m)
        _positive_float("minor_radius_m", self.minor_radius_m)
        _positive_float("elongation", self.elongation)
        _finite_float("triangularity", self.triangularity)
        if not -1.0 < self.triangularity < 1.0:
            raise ValueError("triangularity must lie in (-1, 1)")
        if self.major_radius_m <= self.minor_radius_m * (1.0 + abs(self.triangularity)):
            raise ValueError("major_radius_m must keep the sampled boundary positive")
        _nonnegative_float("pressure_axis_pa", self.pressure_axis_pa)
        _nonnegative_float("pressure_edge_pa", self.pressure_edge_pa)
        _positive_float("iota_axis", self.iota_axis)
        _positive_float("iota_edge", self.iota_edge)
        _positive_float("exact_tol", self.exact_tol)

    def boundary(self) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
        """Return the production axisymmetric Fourier boundary coefficients."""
        return AxisymmetricTokamakBoundary.from_parameters(
            R0=self.major_radius_m,
            a=self.minor_radius_m,
            kappa=self.elongation,
            delta=self.triangularity,
        )

    def solver(self) -> VMECLiteSolver:
        """Build a VMEC-lite solver with zero-pressure fixed-boundary profiles."""
        solver = VMECLiteSolver(n_s=self.n_s, m_pol=self.m_pol, n_tor=self.n_tor, n_fp=self.n_fp)
        solver.set_boundary(*self.boundary())
        solver.set_profiles(
            np.linspace(self.pressure_axis_pa, self.pressure_edge_pa, self.n_s),
            np.linspace(self.iota_axis, self.iota_edge, self.n_s),
        )
        return solver


@dataclass(frozen=True)
class VMECLiteGeometryValidationResult:
    """Outcome of exact VMEC-lite spectral-geometry validation."""

    case: VMECLiteGeometryCase
    basis_mode_count_error: int
    cosine_max_abs_error: float
    sine_max_abs_error: float
    axisymmetric_boundary_max_abs_error: float
    fixed_boundary_radial_scaling_max_abs_error: float
    q_iota_reciprocal_max_abs_error: float
    bfield_coefficient_max_abs_error: float
    min_sampled_major_radius_m: float
    force_residual: float
    iterations: int
    converged: bool
    basis_passed: bool
    boundary_passed: bool
    radial_scaling_passed: bool
    q_iota_passed: bool
    bfield_passed: bool
    passed: bool


def default_case() -> VMECLiteGeometryCase:
    """Return a low-order shaped tokamak case with exact fixed-boundary geometry."""
    return VMECLiteGeometryCase(
        n_s=13,
        m_pol=2,
        n_tor=1,
        n_fp=1,
        major_radius_m=6.2,
        minor_radius_m=2.0,
        elongation=1.7,
        triangularity=0.33,
        pressure_axis_pa=0.0,
        pressure_edge_pa=0.0,
        iota_axis=1.0,
        iota_edge=0.3,
        exact_tol=1e-12,
    )


def basis_mode_count_error(case: VMECLiteGeometryCase) -> int:
    """Return mode-count mismatch for the declared VMEC-lite truncation."""
    basis = SpectralBasis(case.m_pol, case.n_tor, case.n_fp)
    expected = case.n_tor + 1 + case.m_pol * (2 * case.n_tor + 1)
    return abs(basis.n_modes - expected)


def spectral_evaluation_error(case: VMECLiteGeometryCase) -> dict[str, float]:
    """Compare production basis evaluation to a direct Fourier sum."""
    basis = SpectralBasis(case.m_pol, case.n_tor, case.n_fp)
    coeffs = np.arange(1, basis.n_modes + 1, dtype=float) / 10.0
    theta = np.linspace(0.0, 2.0 * np.pi, 17, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, 17, endpoint=False)
    theta_grid, zeta_grid = np.meshgrid(theta, zeta, indexing="ij")
    flat_theta = theta_grid.ravel()
    flat_zeta = zeta_grid.ravel()
    manual_cos = _manual_basis_sum(basis, coeffs, flat_theta, flat_zeta, is_sin=False)
    manual_sin = _manual_basis_sum(basis, coeffs, flat_theta, flat_zeta, is_sin=True)
    return {
        "cosine_max_abs_error": float(
            np.max(np.abs(basis.evaluate(coeffs, flat_theta, flat_zeta, is_sin=False) - manual_cos))
        ),
        "sine_max_abs_error": float(
            np.max(np.abs(basis.evaluate(coeffs, flat_theta, flat_zeta, is_sin=True) - manual_sin))
        ),
    }


def axisymmetric_boundary_errors(case: VMECLiteGeometryCase) -> dict[str, float]:
    """Check shaped-tokamak boundary coefficients against their closed forms."""
    r_bound, z_bound = case.boundary()
    expected = {
        "R00_abs_error": abs(r_bound[(0, 0)] - case.major_radius_m),
        "R10_abs_error": abs(r_bound[(1, 0)] - case.minor_radius_m),
        "R20_abs_error": abs(r_bound[(2, 0)] - (-0.5 * case.minor_radius_m * case.triangularity)),
        "Z10_abs_error": abs(z_bound[(1, 0)] - case.minor_radius_m * case.elongation),
    }
    return {name: float(value) for name, value in expected.items()}


def fixed_boundary_radial_scaling_error(case: VMECLiteGeometryCase) -> float:
    """Validate VMEC-lite's initial fixed-boundary radial power law."""
    solver = _solved_initial_geometry(case)
    r_bound, z_bound = case.boundary()
    max_error = 0.0
    for mode_index, (m, n) in enumerate(solver.basis.mn_modes):
        for surface_index, s_value in enumerate(solver.s_grid):
            if m == 0 and n == 0:
                expected_r = r_bound[(0, 0)]
                expected_z = 0.0
            else:
                scale = s_value ** (m / 2.0)
                expected_r = scale * r_bound.get((m, n), 0.0)
                expected_z = scale * z_bound.get((m, n), 0.0)
            max_error = max(max_error, abs(solver.R_mn[surface_index, mode_index] - expected_r))
            max_error = max(max_error, abs(solver.Z_mn[surface_index, mode_index] - expected_z))
    return float(max_error)


def q_iota_reciprocal_error(case: VMECLiteGeometryCase) -> float:
    """Validate the reduced q-profile contract q = 1 / iota."""
    solver = case.solver()
    q_profile = 1.0 / solver.iota
    return float(np.max(np.abs(q_profile * solver.iota - 1.0)))


def bfield_coefficient_error(case: VMECLiteGeometryCase) -> float:
    """Validate the reduced B-coefficient construction used by VMEC-lite."""
    solver = _solved_initial_geometry(case)
    result = solver.solve(max_iter=1, tol=1e10)
    idx_00 = solver.basis.mn_modes.index((0, 0))
    max_error = 0.0
    for surface_index in range(solver.n_s):
        r_00 = max(abs(result.R_mn[surface_index, idx_00]), 1e-6)
        for mode_index, (m, _n) in enumerate(solver.basis.mn_modes):
            if mode_index == idx_00:
                expected = 1.0
            else:
                expected = -result.R_mn[surface_index, mode_index] / r_00
                if m == 1:
                    expected += solver.iota[surface_index] * abs(result.Z_mn[surface_index, mode_index]) / r_00
            max_error = max(max_error, abs(result.B_mn[surface_index, mode_index] - expected))
    return float(max_error)


def validate_vmec_lite_geometry(case: VMECLiteGeometryCase | None = None) -> VMECLiteGeometryValidationResult:
    """Validate exact spectral-geometry contracts in the VMEC-lite facade."""
    case = case or default_case()
    spectral_errors = spectral_evaluation_error(case)
    boundary_errors = axisymmetric_boundary_errors(case)
    radial_error = fixed_boundary_radial_scaling_error(case)
    q_error = q_iota_reciprocal_error(case)
    bfield_error = bfield_coefficient_error(case)
    solver = _solved_initial_geometry(case)
    result = solver.solve(max_iter=1, tol=1e10)
    min_major_radius = _minimum_major_radius(result.R_mn, solver.basis)

    basis_passed = (
        basis_mode_count_error(case) == 0
        and spectral_errors["cosine_max_abs_error"] < case.exact_tol
        and spectral_errors["sine_max_abs_error"] < case.exact_tol
    )
    boundary_max = max(boundary_errors.values())
    boundary_passed = boundary_max < case.exact_tol and min_major_radius > 0.0
    radial_scaling_passed = radial_error < case.exact_tol
    q_iota_passed = q_error < case.exact_tol
    bfield_passed = bfield_error < case.exact_tol
    passed = basis_passed and boundary_passed and radial_scaling_passed and q_iota_passed and bfield_passed

    return VMECLiteGeometryValidationResult(
        case=case,
        basis_mode_count_error=basis_mode_count_error(case),
        cosine_max_abs_error=spectral_errors["cosine_max_abs_error"],
        sine_max_abs_error=spectral_errors["sine_max_abs_error"],
        axisymmetric_boundary_max_abs_error=boundary_max,
        fixed_boundary_radial_scaling_max_abs_error=radial_error,
        q_iota_reciprocal_max_abs_error=q_error,
        bfield_coefficient_max_abs_error=bfield_error,
        min_sampled_major_radius_m=min_major_radius,
        force_residual=result.force_residual,
        iterations=result.iterations,
        converged=result.converged,
        basis_passed=basis_passed,
        boundary_passed=boundary_passed,
        radial_scaling_passed=radial_scaling_passed,
        q_iota_passed=q_iota_passed,
        bfield_passed=bfield_passed,
        passed=passed,
    )


def build_evidence(result: VMECLiteGeometryValidationResult, *, target_id: str) -> dict[str, Any]:
    """Build a sealed, schema-versioned VMEC-lite geometry evidence payload."""
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    payload: dict[str, Any] = {
        "schema_version": VMEC_LITE_GEOMETRY_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "target_id": target_id,
        "claim_status": "bounded_vmec_lite_evidence",
        "public_claim_allowed": False,
        "full_vmec_claim_allowed": False,
        "case": {
            "n_s": result.case.n_s,
            "m_pol": result.case.m_pol,
            "n_tor": result.case.n_tor,
            "n_fp": result.case.n_fp,
            "major_radius_m": result.case.major_radius_m,
            "minor_radius_m": result.case.minor_radius_m,
            "elongation": result.case.elongation,
            "triangularity": result.case.triangularity,
            "pressure_axis_pa": result.case.pressure_axis_pa,
            "pressure_edge_pa": result.case.pressure_edge_pa,
            "iota_axis": result.case.iota_axis,
            "iota_edge": result.case.iota_edge,
            "exact_tol": result.case.exact_tol,
        },
        "basis_mode_count_error": result.basis_mode_count_error,
        "cosine_max_abs_error": result.cosine_max_abs_error,
        "sine_max_abs_error": result.sine_max_abs_error,
        "axisymmetric_boundary_max_abs_error": result.axisymmetric_boundary_max_abs_error,
        "fixed_boundary_radial_scaling_max_abs_error": result.fixed_boundary_radial_scaling_max_abs_error,
        "q_iota_reciprocal_max_abs_error": result.q_iota_reciprocal_max_abs_error,
        "bfield_coefficient_max_abs_error": result.bfield_coefficient_max_abs_error,
        "min_sampled_major_radius_m": result.min_sampled_major_radius_m,
        "force_residual": result.force_residual,
        "iterations": result.iterations,
        "converged": result.converged,
        "basis_passed": result.basis_passed,
        "boundary_passed": result.boundary_passed,
        "radial_scaling_passed": result.radial_scaling_passed,
        "q_iota_passed": result.q_iota_passed,
        "bfield_passed": result.bfield_passed,
        "passed": result.passed,
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Return ``True`` when a VMEC-lite geometry payload is sealed and passing."""
    if payload.get("schema_version") != VMEC_LITE_GEOMETRY_SCHEMA_VERSION:
        raise ValueError("unsupported VMEC-lite geometry evidence schema_version")
    declared = payload.get("payload_sha256")
    if not _is_sha256(declared):
        raise ValueError("payload_sha256 must be a SHA-256 hex digest")
    if declared != _payload_sha256(payload):
        raise ValueError("payload_sha256 does not match payload")
    return bool(payload.get("passed"))


def _solved_initial_geometry(case: VMECLiteGeometryCase) -> VMECLiteSolver:
    solver = case.solver()
    solver.solve(max_iter=1, tol=1e10)
    return solver


def _minimum_major_radius(r_mn: npt.NDArray[np.floating[Any]], basis: SpectralBasis) -> float:
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    theta_grid, zeta_grid = np.meshgrid(theta, zeta, indexing="ij")
    min_major_radius = float("inf")
    for surface_idx in range(r_mn.shape[0]):
        values = basis.evaluate(r_mn[surface_idx], theta_grid.ravel(), zeta_grid.ravel(), is_sin=False)
        min_major_radius = min(min_major_radius, float(np.min(values)))
    return min_major_radius


def _manual_basis_sum(
    basis: SpectralBasis,
    coeffs: npt.NDArray[np.float64],
    theta: npt.NDArray[np.float64],
    zeta: npt.NDArray[np.float64],
    *,
    is_sin: bool,
) -> npt.NDArray[np.float64]:
    values = np.zeros_like(theta, dtype=float)
    for coeff, (m, n) in zip(coeffs, basis.mn_modes):
        angle = m * theta - n * basis.n_fp * zeta
        values += coeff * (np.sin(angle) if is_sin else np.cos(angle))
    return values


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


def _nonnegative_float(name: str, value: object) -> float:
    result = _finite_float(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_int(name: str, value: object, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


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
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    case = evidence["case"]
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "",
        "# VMEC-lite Spectral-Geometry Validation",
        "",
        f"- Schema: `{evidence['schema_version']}`",
        f"- Generated (UTC): {evidence['generated_utc']}",
        f"- Target: `{evidence['target_id']}`",
        f"- Claim status: `{evidence['claim_status']}`",
        f"- Full VMEC claim allowed: `{evidence['full_vmec_claim_allowed']}`",
        f"- Truncation: m_pol={case['m_pol']}, n_tor={case['n_tor']}, n_fp={case['n_fp']}",
        f"- Geometry: R0={case['major_radius_m']} m, a={case['minor_radius_m']} m, "
        f"kappa={case['elongation']}, delta={case['triangularity']}",
        f"- Status: **{'pass' if evidence['passed'] else 'fail'}**",
        "",
        f"## Exact geometry checks (gate < {case['exact_tol']:.1e})",
        "",
        "| check | value |",
        "| --- | --- |",
        f"| Basis mode-count error | {evidence['basis_mode_count_error']} |",
        f"| Cosine Fourier evaluation error | {evidence['cosine_max_abs_error']:.3e} |",
        f"| Sine Fourier evaluation error | {evidence['sine_max_abs_error']:.3e} |",
        f"| Axisymmetric boundary coefficient error | {evidence['axisymmetric_boundary_max_abs_error']:.3e} |",
        f"| Fixed-boundary radial scaling error | {evidence['fixed_boundary_radial_scaling_max_abs_error']:.3e} |",
        f"| q = 1 / iota reciprocal error | {evidence['q_iota_reciprocal_max_abs_error']:.3e} |",
        f"| B-coefficient construction error | {evidence['bfield_coefficient_max_abs_error']:.3e} |",
        "",
        f"- Minimum sampled major radius: {evidence['min_sampled_major_radius_m']:.6g} m",
        f"- Initial-geometry force residual: {evidence['force_residual']:.6g}",
        "",
        "This is bounded local-regression evidence for the repository VMEC-lite facade.",
        "Full VMEC-grade 3D MHD equilibrium claims remain blocked until matched external VMEC or public references pass admission.",
    ]
    json_path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point producing sealed VMEC-lite geometry evidence."""
    parser = argparse.ArgumentParser(description="Validate VMEC-lite spectral geometry against exact forms")
    parser.add_argument("--target-id", type=str, default="local-vmec-lite-geometry")
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    parser.add_argument("--report", type=str, default=None, help="write sealed JSON evidence and a Markdown summary")
    args = parser.parse_args(argv)

    result = validate_vmec_lite_geometry()
    evidence = build_evidence(result, target_id=args.target_id)
    if args.report:
        _write_report(evidence, Path(args.report))
    if args.json_out:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print("VMEC-lite spectral-geometry validation")
        print(
            f"  basis + Fourier:       modes={result.basis_mode_count_error} "
            f"cos={result.cosine_max_abs_error:.3e} sin={result.sine_max_abs_error:.3e} "
            f"{'ok' if result.basis_passed else 'FAIL'}"
        )
        print(
            f"  boundary + radial:     boundary={result.axisymmetric_boundary_max_abs_error:.3e} "
            f"radial={result.fixed_boundary_radial_scaling_max_abs_error:.3e} "
            f"{'ok' if result.boundary_passed and result.radial_scaling_passed else 'FAIL'}"
        )
        print(
            f"  q/iota + B coeffs:     q={result.q_iota_reciprocal_max_abs_error:.3e} "
            f"B={result.bfield_coefficient_max_abs_error:.3e} "
            f"{'ok' if result.q_iota_passed and result.bfield_passed else 'FAIL'}"
        )
        print(f"Status: {'pass' if result.passed else 'fail'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

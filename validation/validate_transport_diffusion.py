#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Radial heat-diffusion solver analytic validation
"""Validate the transport heat-diffusion operator and solver against exact results.

The integrated transport solver (``src/scpn_control/core/integrated_transport_solver.py``)
evolves the radial heat equation on normalised flux-surface radius ``rho ∈ [0, 1]``
with the cylindrical diffusion operator

    L[T] = (1/a^2) (1/rho) d/drho ( rho chi dT/drho )

discretised with conservative half-grid diffusivities, advanced implicitly with a
Crank-Nicolson tridiagonal step solved by the Thomas algorithm. Both the operator
and the implicit solve have *exact* analytic references — no measured discharge or
external integrated-modelling artefact is required, so this validation is fully
self-contained.

Three checks, all exercising the production methods:

1. **Operator eigenvalue (Bessel).** The cylindrical Laplacian eigenfunction
   ``J_0(lambda rho)`` satisfies ``L[J_0(lambda rho)] = -(chi lambda^2 / a^2)
   J_0(lambda rho)``. ``TransportSolver._explicit_diffusion_rhs`` applied to the
   sampled eigenfunction must reproduce this at second order in ``Δrho``.
2. **Manufactured steady state.** For the manufactured profile ``T*(rho) =
   1 - rho^3`` the cylindrical operator gives ``L[T*] = -9 chi rho / a^2``, so a
   fixed source ``S = 9 chi rho / a^2`` makes ``T*`` the steady state. Crank-
   Nicolson stepping (``_build_cn_tridiag`` + ``_thomas_solve``) to convergence,
   with Dirichlet data taken from ``T*``, must recover ``T*`` at second order.
3. **Polyglot Thomas parity.** The Rust tridiagonal solver
   ``scpn_control_rs.py_thomas_solve`` — the compute primitive used by the Rust
   ``transport_step`` — must reproduce the Python ``_thomas_solve`` solution of
   the identical Crank-Nicolson system to machine precision.

References:
  Wesson J. (2011) *Tokamaks*, 4th ed., Oxford University Press, Ch. 4
  (radial transport equations).
  Crank J., Nicolson P. (1947) *Proc. Camb. Phil. Soc.* 43, 50.
  Abramowitz M., Stegun I. (1965) *Handbook of Mathematical Functions*,
  Ch. 9 (Bessel functions).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import j0, jn_zeros

from scpn_control.core.integrated_transport_solver import TransportSolver

FloatArray = NDArray[np.float64]

TRANSPORT_DIFFUSION_SCHEMA_VERSION = "scpn-control.transport-diffusion-validation.v1"

DEFAULT_RESOLUTIONS: tuple[int, ...] = (33, 65, 129)
_CHI_VALUE = 0.7
_MINOR_RADIUS = 0.5
_FIRST_BESSEL_ZERO = float(jn_zeros(0, 1)[0])


def _build_solver(nr: int) -> TransportSolver:
    """Instantiate the production ``TransportSolver`` on an ``nr``-point rho grid."""
    _grid_resolution("nr", nr)
    config: dict[str, Any] = {
        "reactor_name": "transport-diffusion-validation",
        "grid_resolution": [16, 16],
        "dimensions": {
            "R_min": 1.7 - _MINOR_RADIUS,
            "R_max": 1.7 + _MINOR_RADIUS,
            "Z_min": -1.0,
            "Z_max": 1.0,
        },
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0, "R0": 1.7, "a": _MINOR_RADIUS, "B0": 2.0},
        "coils": [{"name": "PF1", "r": 1.2, "z": 0.5, "current": 1.0}],
        "solver": {
            "max_iterations": 1,
            "convergence_threshold": 1e-8,
            "relaxation_factor": 0.1,
            "solver_method": "sor",
            "sor_omega": 1.6,
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "transport_config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return TransportSolver(str(config_path), nr=nr)


def bessel_operator_error(nr: int, *, chi_value: float = _CHI_VALUE, lam: float = _FIRST_BESSEL_ZERO) -> float:
    """Max interior relative error of the diffusion operator on a Bessel eigenmode.

    Applies ``_explicit_diffusion_rhs`` to ``J_0(lam rho)`` and compares to the
    analytic eigenvalue result ``-(chi lam^2 / a^2) J_0(lam rho)``.
    """
    _positive_float("chi_value", chi_value)
    _positive_float("lam", lam)
    solver = _build_solver(nr)
    chi = np.full(solver.nr, chi_value, dtype=np.float64)
    eigenmode = np.asarray(j0(lam * solver.rho), dtype=np.float64)
    applied = np.asarray(solver._explicit_diffusion_rhs(eigenmode, chi), dtype=np.float64)
    eigenvalue = -(chi_value * lam**2 / solver.a**2)
    exact = eigenvalue * eigenmode
    interior_error = np.abs(applied[1:-1] - exact[1:-1])
    scale = float(np.max(np.abs(exact[1:-1])))
    return float(interior_error.max() / scale)


def manufactured_steady_state(nr: int, *, chi_value: float = _CHI_VALUE) -> float:
    """Steady-state NRMSE of the Crank-Nicolson operator against ``T* = 1 - rho^3``.

    The fixed source ``S = 9 chi rho / a^2`` makes ``T*`` the exact steady state of
    the cylindrical diffusion operator. The production ``_build_cn_tridiag``
    coefficients (the implicit LHS ``I - 0.5 dt L``) are converted to the steady
    operator ``-L`` and solved once with the production ``_thomas_solve`` under
    Dirichlet data from ``T*``; the reconstructed profile converges to ``T*`` at
    second order in ``Δrho``. Solving the steady system directly avoids the slow
    Crank-Nicolson relaxation while exercising the same production methods.
    """
    _positive_float("chi_value", chi_value)
    solver = _build_solver(nr)
    chi = np.full(solver.nr, chi_value, dtype=np.float64)
    target = 1.0 - solver.rho**3
    source = 9.0 * chi_value * solver.rho / solver.a**2

    dt = 1.0
    half = 0.5 * dt
    sub, main, sup = solver._build_cn_tridiag(chi, dt)

    n = solver.nr
    steady_diag = np.ones(n, dtype=np.float64)
    steady_sub = np.zeros(n - 1, dtype=np.float64)
    steady_sup = np.zeros(n - 1, dtype=np.float64)
    rhs = np.zeros(n, dtype=np.float64)
    # Convert the CN LHS (I - 0.5 dt L) into the steady operator (-L): dividing the
    # interior coefficients by 0.5 dt recovers -L, whose system (-L) T = S is solved
    # once. Boundary rows stay Dirichlet (diag = 1, rhs = T* at the endpoints).
    for i in range(1, n - 1):
        steady_diag[i] = (main[i] - 1.0) / half
        steady_sup[i] = sup[i] / half
        steady_sub[i - 1] = sub[i - 1] / half
        rhs[i] = float(source[i])
    rhs[0] = float(target[0])
    rhs[-1] = float(target[-1])

    temperature = np.asarray(solver._thomas_solve(steady_sub, steady_diag, steady_sup, rhs), dtype=np.float64)
    span = float(target.max() - target.min())
    return float(np.sqrt(np.mean((temperature[1:-1] - target[1:-1]) ** 2)) / span)


@dataclass(frozen=True)
class ThomasParityRecord:
    """Python/Rust Thomas-solver parity on a Crank-Nicolson system."""

    max_abs_diff: float
    max_abs_diff_vs_dense: float
    matches: bool


def thomas_parity(
    nr: int = 65, *, chi_value: float = _CHI_VALUE, dt: float = 0.1, tolerance: float = 1e-9
) -> ThomasParityRecord | None:
    """Compare the Rust ``py_thomas_solve`` to the Python ``_thomas_solve``.

    Returns ``None`` when the compiled extension is unavailable. The Rust solver
    uses the convention ``a, b, c`` all of length ``n`` (``a[0]`` and ``c[n-1]``
    unused), so the Python ``n-1``-length off-diagonals are zero-padded.
    """
    try:
        import scpn_control_rs as rust
    except ImportError:
        return None
    if not hasattr(rust, "py_thomas_solve"):
        return None

    solver = _build_solver(nr)
    chi = np.full(solver.nr, chi_value, dtype=np.float64)
    sub, main, sup = solver._build_cn_tridiag(chi, dt)
    rhs = np.linspace(1.0, 0.0, solver.nr) + 0.1 * np.sin(np.linspace(0.0, 3.0, solver.nr))

    python_solution = np.asarray(solver._thomas_solve(sub, main, sup, rhs), dtype=np.float64)
    sub_padded = np.concatenate([[0.0], sub])
    sup_padded = np.concatenate([sup, [0.0]])
    rust_solution = np.asarray(rust.py_thomas_solve(sub_padded, main, sup_padded, rhs), dtype=np.float64)

    dense = np.diag(main) + np.diag(sub, -1) + np.diag(sup, 1)
    dense_solution = np.linalg.solve(dense, rhs)

    max_abs_diff = float(np.max(np.abs(python_solution - rust_solution)))
    max_abs_diff_vs_dense = float(np.max(np.abs(rust_solution - dense_solution)))
    return ThomasParityRecord(
        max_abs_diff=max_abs_diff,
        max_abs_diff_vs_dense=max_abs_diff_vs_dense,
        matches=max_abs_diff < tolerance,
    )


@dataclass(frozen=True)
class ConvergenceRecord:
    """Per-resolution error sample for a convergence study."""

    resolution: int
    mesh_spacing: float
    error: float


def _log_log_slope(records: Sequence[ConvergenceRecord]) -> float:
    if len(records) < 2:
        raise ValueError("at least two resolutions are required to estimate an order")
    log_h = np.log(np.array([record.mesh_spacing for record in records], dtype=np.float64))
    log_e = np.log(np.array([record.error for record in records], dtype=np.float64))
    slope, _ = np.polyfit(log_h, log_e, 1)
    return float(slope)


@dataclass(frozen=True)
class TransportDiffusionValidationResult:
    """Outcome of the transport heat-diffusion validation."""

    resolutions: tuple[int, ...]
    operator_records: tuple[ConvergenceRecord, ...]
    operator_order: float
    operator_error_finest: float
    operator_passed: bool
    steady_records: tuple[ConvergenceRecord, ...]
    steady_order: float
    steady_nrmse_finest: float
    steady_passed: bool
    thomas_available: bool
    thomas_record: ThomasParityRecord | None
    thomas_passed: bool
    min_order: float
    operator_error_gate: float
    steady_nrmse_gate: float
    passed: bool


def validate_transport_diffusion(
    *,
    resolutions: Sequence[int] = DEFAULT_RESOLUTIONS,
    chi_value: float = _CHI_VALUE,
    min_order: float = 1.8,
    operator_error_gate: float = 5e-3,
    steady_nrmse_gate: float = 1e-3,
    thomas_tolerance: float = 1e-9,
    include_rust: bool = True,
) -> TransportDiffusionValidationResult:
    """Validate the production heat-diffusion operator and Crank-Nicolson solver.

    The diffusion operator must reproduce the Bessel eigenvalue at order
    ``>= min_order`` with finest-grid relative error below
    ``operator_error_gate``; the Crank-Nicolson solver must recover the
    manufactured steady state at order ``>= min_order`` with finest-grid NRMSE
    below ``steady_nrmse_gate``; and, when the Rust extension is present, the
    Rust Thomas solver must match the Python solver to within ``thomas_tolerance``.
    """
    ordered = tuple(sorted({_grid_resolution("resolution", n) for n in resolutions}))
    if len(ordered) < 2:
        raise ValueError("at least two distinct resolutions are required")

    operator_records: list[ConvergenceRecord] = []
    steady_records: list[ConvergenceRecord] = []
    for nr in ordered:
        h = 1.0 / (nr - 1)
        operator_records.append(
            ConvergenceRecord(resolution=nr, mesh_spacing=h, error=bessel_operator_error(nr, chi_value=chi_value))
        )
        steady_records.append(
            ConvergenceRecord(resolution=nr, mesh_spacing=h, error=manufactured_steady_state(nr, chi_value=chi_value))
        )

    operator_order = _log_log_slope(operator_records)
    operator_error_finest = operator_records[-1].error
    operator_passed = operator_order >= min_order and operator_error_finest < operator_error_gate

    steady_order = _log_log_slope(steady_records)
    steady_nrmse_finest = steady_records[-1].error
    steady_passed = steady_order >= min_order and steady_nrmse_finest < steady_nrmse_gate

    thomas_record = thomas_parity(tolerance=thomas_tolerance) if include_rust else None
    thomas_available = thomas_record is not None
    thomas_passed = thomas_record is None or thomas_record.matches

    passed = operator_passed and steady_passed and thomas_passed
    return TransportDiffusionValidationResult(
        resolutions=ordered,
        operator_records=tuple(operator_records),
        operator_order=operator_order,
        operator_error_finest=operator_error_finest,
        operator_passed=operator_passed,
        steady_records=tuple(steady_records),
        steady_order=steady_order,
        steady_nrmse_finest=steady_nrmse_finest,
        steady_passed=steady_passed,
        thomas_available=thomas_available,
        thomas_record=thomas_record,
        thomas_passed=thomas_passed,
        min_order=min_order,
        operator_error_gate=operator_error_gate,
        steady_nrmse_gate=steady_nrmse_gate,
        passed=passed,
    )


def build_evidence(result: TransportDiffusionValidationResult, *, target_id: str) -> dict[str, Any]:
    """Build a tamper-evident, schema-versioned validation evidence payload."""
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    thomas = result.thomas_record
    payload: dict[str, Any] = {
        "schema_version": TRANSPORT_DIFFUSION_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "target_id": target_id,
        "minor_radius_m": _MINOR_RADIUS,
        "chi_m2_s": _CHI_VALUE,
        "bessel_lambda": _FIRST_BESSEL_ZERO,
        "resolutions": list(result.resolutions),
        "min_order": result.min_order,
        "operator_error_gate": result.operator_error_gate,
        "steady_nrmse_gate": result.steady_nrmse_gate,
        "operator_records": [
            {"resolution": rec.resolution, "mesh_spacing": rec.mesh_spacing, "error": rec.error}
            for rec in result.operator_records
        ],
        "operator_order": result.operator_order,
        "operator_error_finest": result.operator_error_finest,
        "operator_passed": result.operator_passed,
        "steady_records": [
            {"resolution": rec.resolution, "mesh_spacing": rec.mesh_spacing, "nrmse": rec.error}
            for rec in result.steady_records
        ],
        "steady_order": result.steady_order,
        "steady_nrmse_finest": result.steady_nrmse_finest,
        "steady_passed": result.steady_passed,
        "thomas_available": result.thomas_available,
        "thomas_record": (
            None
            if thomas is None
            else {
                "max_abs_diff": thomas.max_abs_diff,
                "max_abs_diff_vs_dense": thomas.max_abs_diff_vs_dense,
                "matches": thomas.matches,
            }
        ),
        "thomas_passed": result.thomas_passed,
        "passed": result.passed,
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Return ``True`` when a payload is well-formed, sealed, and passing."""
    if payload.get("schema_version") != TRANSPORT_DIFFUSION_SCHEMA_VERSION:
        raise ValueError("unsupported transport diffusion evidence schema_version")
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
    if result < 5:
        raise ValueError(f"{name} must be at least 5 (need interior diffusion nodes)")
    return result


def _write_report(evidence: Mapping[str, Any], json_path: Path) -> None:
    """Persist the sealed JSON evidence and a human-readable Markdown summary."""
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = json_path.with_suffix(".md")
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "",
        "# Transport Heat-Diffusion Validation",
        "",
        f"- Schema: `{evidence['schema_version']}`",
        f"- Generated (UTC): {evidence['generated_utc']}",
        f"- Target: `{evidence['target_id']}`",
        f"- chi = {evidence['chi_m2_s']} m^2/s; a = {evidence['minor_radius_m']} m; "
        f"lambda = {evidence['bessel_lambda']:.6f} (first J0 zero)",
        f"- Status: **{'pass' if evidence['passed'] else 'fail'}**",
        "",
        "## Diffusion operator vs Bessel eigenvalue (`_explicit_diffusion_rhs`)",
        "",
        f"- Order of accuracy: {evidence['operator_order']:.3f} (gate >= {evidence['min_order']})",
        f"- Finest-grid relative error: {evidence['operator_error_finest']:.3e} "
        f"(gate < {evidence['operator_error_gate']:.1e})",
        "",
        "| resolution | h | relative error |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {rec['resolution']} | {rec['mesh_spacing']:.4e} | {rec['error']:.4e} |"
        for rec in evidence["operator_records"]
    ]
    lines += [
        "",
        "## Manufactured steady state (`_build_cn_tridiag` + `_thomas_solve`)",
        "",
        f"- Order of accuracy: {evidence['steady_order']:.3f} (gate >= {evidence['min_order']})",
        f"- Finest-grid NRMSE: {evidence['steady_nrmse_finest']:.3e} (gate < {evidence['steady_nrmse_gate']:.1e})",
        "",
        "| resolution | h | NRMSE |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {rec['resolution']} | {rec['mesh_spacing']:.4e} | {rec['nrmse']:.4e} |"
        for rec in evidence["steady_records"]
    ]
    thomas = evidence["thomas_record"]
    lines += ["", "## Polyglot Thomas parity (`scpn_control_rs.py_thomas_solve`)", ""]
    if thomas is None:
        lines.append("- Compiled extension not present; Rust solver not exercised.")
    else:
        lines += [
            f"- max |Python - Rust|: {thomas['max_abs_diff']:.3e}",
            f"- max |Rust - dense numpy|: {thomas['max_abs_diff_vs_dense']:.3e}",
            f"- Matches within tolerance: {thomas['matches']}",
        ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point producing schema-versioned validation evidence."""
    parser = argparse.ArgumentParser(
        description="Validate the transport heat-diffusion operator and Crank-Nicolson solver"
    )
    parser.add_argument("--resolutions", type=int, nargs="+", default=list(DEFAULT_RESOLUTIONS))
    parser.add_argument("--target-id", type=str, default="local-transport-diffusion")
    parser.add_argument("--no-rust", action="store_true", help="skip the Rust Thomas parity check")
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    parser.add_argument("--report", type=str, default=None, help="write sealed JSON evidence and a Markdown summary")
    args = parser.parse_args(argv)

    result = validate_transport_diffusion(resolutions=args.resolutions, include_rust=not args.no_rust)
    evidence = build_evidence(result, target_id=args.target_id)

    if args.report:
        _write_report(evidence, Path(args.report))

    if args.json_out:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print("Transport heat-diffusion validation")
        print(
            f"  operator (Bessel):  order={result.operator_order:.3f} "
            f"finest_err={result.operator_error_finest:.3e} "
            f"{'ok' if result.operator_passed else 'FAIL'}"
        )
        print(
            f"  CN steady state:    order={result.steady_order:.3f} "
            f"finest_nrmse={result.steady_nrmse_finest:.3e} "
            f"{'ok' if result.steady_passed else 'FAIL'}"
        )
        if result.thomas_record is not None:
            print(
                f"  Rust Thomas parity: max|py-rs|={result.thomas_record.max_abs_diff:.3e} "
                f"{'ok' if result.thomas_passed else 'FAIL'}"
            )
        print(f"Status: {'pass' if result.passed else 'fail'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

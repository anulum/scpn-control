#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Grad-Shafranov Solov'ev analytic-equilibrium validation
"""Validate the production fusion-kernel Grad-Shafranov solver against Solov'ev.

The fusion kernel (``src/scpn_control/core/fusion_kernel.py``) discretises the
toroidal Grad-Shafranov elliptic operator

    Δ*ψ = ∂²ψ/∂R² − (1/R) ∂ψ/∂R + ∂²ψ/∂Z²

with a second-order centred five-point stencil and solves the fixed-boundary
problem with a Red-Black successive-over-relaxation (SOR) sweep that carries the
toroidal ``1/R`` term. The Solov'ev family admits an exact polynomial solution
of the homogeneous-source Grad-Shafranov equation, which makes it the canonical
analytic benchmark for any tokamak equilibrium discretisation.

Manufactured exact solution (constant ``p'`` and ``FF'`` Solov'ev branch):

    ψ(R, Z) = c₁ R⁴/8 + c₂ Z²        ⇒        Δ*ψ = c₁ R² + 2 c₂.

Two **production** code paths are validated against this exact field:

1. **Discrete operator.** ``FusionKernel._apply_gs_operator`` (the Newton/GMRES
   matvec, sharing the stencil used by ``_mg_residual``) is applied to the exact
   ψ sampled on grids of increasing resolution. Its truncation error against the
   analytic ``Δ*ψ`` must vanish at second order in the mesh spacing ``h``.
2. **SOR equilibrium solver.** The production ``_sor_step`` smoother is iterated
   to a fixed residual with Dirichlet data taken from the exact ψ. The
   reconstructed flux must converge to the exact Solov'ev field at second order.

A third, **polyglot**, path is recorded for transparency but not gated: the Rust
``scpn_control_rs.py_multigrid_solve`` binding is run on the same problem when
the compiled extension is present. The binding exposes a fixed 100-cycle / 1e-8
budget that does not converge on this forcing: it preserves the injected
Dirichlet boundary but leaves a large interior Grad-Shafranov residual, so it
does not reproduce the analytic equilibrium. This is recorded honestly and
cross-references the Rust/Python SOR parity gap tracked in
``tests/test_rust_python_parity.py``. The Python ``_multigrid_vcycle`` V-cycle is
likewise not the validated reconstruction path here because it is not a stand-
alone contraction on this forcing; the SOR smoother is the validated solver.

References:
  Solov'ev L. S. (1968) "The theory of hydromagnetic stability of toroidal
  plasma configurations", *Sov. Phys. JETP* 26, 400.
  Cerfon A. J., Freidberg J. P. (2010) "One size fits all analytic solutions to
  the Grad-Shafranov equation", *Phys. Plasmas* 17, 032502.
  Jardin S. (2010) *Computational Methods in Plasma Physics*, CRC Press, Ch. 4.
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

from scpn_control.core.fusion_kernel import FusionKernel

FloatArray = NDArray[np.float64]

GRAD_SHAFRANOV_SOLOVEV_SCHEMA_VERSION = "scpn-control.grad-shafranov-solovev-validation.v1"

DEFAULT_RESOLUTIONS: tuple[int, ...] = (33, 49, 65, 97)


@dataclass(frozen=True)
class SolovevGeometry:
    """Rectangular ``(R, Z)`` domain and Solov'ev coefficients for the benchmark.

    The exact field is ``ψ = c1 · R⁴/8 + c2 · Z²`` with analytic source
    ``Δ*ψ = c1 · R² + 2 c2``. ``r0`` and ``a`` are the major and minor radii of
    the embedded tokamak; the box is widened by ``half_width_factor`` so the
    plasma sits strictly inside the Dirichlet boundary.
    """

    r0: float
    a: float
    r_min: float
    r_max: float
    z_min: float
    z_max: float
    c1: float
    c2: float

    def __post_init__(self) -> None:
        _positive_float("r0", self.r0)
        _positive_float("a", self.a)
        _positive_float("c1", self.c1)
        _positive_float("c2", self.c2)
        if not self.r_min > 0.0:
            raise ValueError("r_min must be positive (toroidal 1/R term is singular at R=0)")
        if not self.r_max > self.r_min:
            raise ValueError("r_max must exceed r_min")
        if not self.z_max > self.z_min:
            raise ValueError("z_max must exceed z_min")

    @classmethod
    def from_aspect(
        cls,
        *,
        r0: float = 1.7,
        a: float = 0.5,
        half_width_factor: float = 1.5,
        c1: float = 1.0,
        c2: float = 0.5,
    ) -> "SolovevGeometry":
        """Build a centred box of half-width ``half_width_factor · a`` about ``r0``."""
        _positive_float("half_width_factor", half_width_factor)
        half = half_width_factor * _positive_float("a", a)
        return cls(
            r0=r0,
            a=a,
            r_min=r0 - half,
            r_max=r0 + half,
            z_min=-half,
            z_max=half,
            c1=c1,
            c2=c2,
        )


def solovev_psi(rr: FloatArray, zz: FloatArray, geometry: SolovevGeometry) -> FloatArray:
    """Exact Solov'ev flux ``ψ = c1 R⁴/8 + c2 Z²`` on the supplied mesh."""
    return np.asarray(geometry.c1 * rr**4 / 8.0 + geometry.c2 * zz**2, dtype=np.float64)


def solovev_source(rr: FloatArray, geometry: SolovevGeometry) -> FloatArray:
    """Analytic Grad-Shafranov source ``Δ*ψ = c1 R² + 2 c2`` for the exact field."""
    return np.asarray(geometry.c1 * rr**2 + 2.0 * geometry.c2, dtype=np.float64)


def _build_kernel(geometry: SolovevGeometry, n: int) -> FusionKernel:
    """Instantiate the production ``FusionKernel`` on an ``n × n`` Solov'ev mesh."""
    _grid_resolution("n", n)
    config: dict[str, Any] = {
        "reactor_name": "solovev-grad-shafranov-validation",
        "grid_resolution": [n, n],
        "dimensions": {
            "R_min": geometry.r_min,
            "R_max": geometry.r_max,
            "Z_min": geometry.z_min,
            "Z_max": geometry.z_max,
        },
        "physics": {
            "plasma_current_target": 1.0,
            "vacuum_permeability": 1.0,
            "R0": geometry.r0,
            "a": geometry.a,
            "B0": 2.0,
        },
        "coils": [{"name": "PF1", "r": geometry.r_min, "z": geometry.z_max, "current": 1.0}],
        "solver": {
            "max_iterations": 1,
            "convergence_threshold": 1e-10,
            "relaxation_factor": 0.1,
            "solver_method": "sor",
            "sor_omega": 1.6,
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "solovev_config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return FusionKernel(str(config_path))


def _enforce_dirichlet(psi: FloatArray, psi_exact: FloatArray) -> None:
    """Overwrite the four boundary edges of ``psi`` with the exact field in place."""
    psi[0, :] = psi_exact[0, :]
    psi[-1, :] = psi_exact[-1, :]
    psi[:, 0] = psi_exact[:, 0]
    psi[:, -1] = psi_exact[:, -1]


def _interior_nrmse(numerical: FloatArray, exact: FloatArray) -> float:
    """Interior-node NRMSE normalised by the exact field range."""
    num_int = numerical[1:-1, 1:-1]
    exact_int = exact[1:-1, 1:-1]
    span = float(exact.max() - exact.min())
    rmse = float(np.sqrt(np.mean((num_int - exact_int) ** 2)))
    return rmse / max(span, 1e-15)


def operator_truncation_error(geometry: SolovevGeometry, n: int) -> float:
    """Max interior truncation error of the production discrete ``Δ*`` operator.

    The exact ψ is fed to ``FusionKernel._apply_gs_operator`` and compared to the
    analytic ``Δ*ψ``; the returned value is the maximum absolute residual over
    interior nodes, which must decay as ``O(h²)``.
    """
    kernel = _build_kernel(geometry, n)
    psi_exact = solovev_psi(kernel.RR, kernel.ZZ, geometry)
    source = solovev_source(kernel.RR, geometry)
    kernel.Psi = psi_exact.copy()
    applied = np.asarray(kernel._apply_gs_operator(psi_exact), dtype=np.float64)
    error = np.abs(applied[1:-1, 1:-1] - source[1:-1, 1:-1])
    return float(error.max())


@dataclass(frozen=True)
class SorReconstruction:
    """Outcome of an SOR reconstruction of the Solov'ev equilibrium."""

    nrmse: float
    iterations: int
    converged: bool
    residual_inf: float


def sor_reconstruction(
    geometry: SolovevGeometry,
    n: int,
    *,
    omega: float = 1.6,
    residual_tol: float = 1e-9,
    max_sweeps: int = 40000,
    check_every: int = 50,
) -> SorReconstruction:
    """Iterate the production ``_sor_step`` smoother to a fixed residual.

    Dirichlet data are taken from the exact Solov'ev field. Iteration stops when
    the infinity-norm of the discrete Grad-Shafranov residual falls below
    ``residual_tol`` (decoupling iteration error from discretisation error) or
    when ``max_sweeps`` is reached.
    """
    _positive_float("omega", omega)
    _positive_float("residual_tol", residual_tol)
    _positive_int("max_sweeps", max_sweeps)
    _positive_int("check_every", check_every)

    kernel = _build_kernel(geometry, n)
    psi_exact = solovev_psi(kernel.RR, kernel.ZZ, geometry)
    source = solovev_source(kernel.RR, geometry)

    psi = np.zeros_like(psi_exact)
    _enforce_dirichlet(psi, psi_exact)
    kernel.Psi = psi.copy()

    iterations = 0
    residual_inf = math.inf
    converged = False
    for sweep in range(max_sweeps):
        updated = np.asarray(kernel._sor_step(kernel.Psi, source, omega=omega), dtype=np.float64)
        _enforce_dirichlet(updated, psi_exact)
        kernel.Psi = updated
        iterations = sweep + 1
        if sweep % check_every == 0:
            residual = np.asarray(kernel._apply_gs_operator(kernel.Psi), dtype=np.float64) - source
            residual_inf = float(np.max(np.abs(residual[1:-1, 1:-1])))
            if residual_inf < residual_tol:
                converged = True
                break

    nrmse = _interior_nrmse(kernel.Psi, psi_exact)
    return SorReconstruction(nrmse=nrmse, iterations=iterations, converged=converged, residual_inf=residual_inf)


@dataclass(frozen=True)
class RustBackendRecord:
    """Recorded behaviour of the Rust multigrid binding on the Solov'ev problem."""

    resolution: int
    nrmse: float
    residual_inf: float
    boundary_preserved: bool
    meets_analytic_tolerance: bool


def rust_multigrid_reconstruction(
    geometry: SolovevGeometry,
    n: int,
    *,
    analytic_tolerance: float,
) -> RustBackendRecord | None:
    """Run ``scpn_control_rs.py_multigrid_solve`` and record its analytic error.

    Returns ``None`` when the compiled extension is unavailable. The record is
    informational only: the binding's fixed-cycle multigrid does not converge on
    this forcing (it preserves the injected boundary but leaves a large interior
    residual), so ``meets_analytic_tolerance`` is expected to be ``False`` and
    this surface is not part of the pass/fail gate.
    """
    try:
        import scpn_control_rs as rust
    except ImportError:
        return None
    if not hasattr(rust, "py_multigrid_solve"):
        return None

    kernel = _build_kernel(geometry, n)
    psi_exact = solovev_psi(kernel.RR, kernel.ZZ, geometry)
    source = solovev_source(kernel.RR, geometry)
    psi = np.zeros_like(psi_exact)
    _enforce_dirichlet(psi, psi_exact)

    result = np.asarray(rust.py_multigrid_solve(psi, source, kernel.R, kernel.Z), dtype=np.float64)
    residual = np.asarray(kernel._apply_gs_operator(result), dtype=np.float64) - source
    residual_inf = float(np.max(np.abs(residual[1:-1, 1:-1])))
    nrmse = _interior_nrmse(result, psi_exact)
    boundary_preserved = bool(
        np.allclose(result[0, :], psi_exact[0, :])
        and np.allclose(result[-1, :], psi_exact[-1, :])
        and np.allclose(result[:, 0], psi_exact[:, 0])
        and np.allclose(result[:, -1], psi_exact[:, -1])
    )
    return RustBackendRecord(
        resolution=n,
        nrmse=nrmse,
        residual_inf=residual_inf,
        boundary_preserved=boundary_preserved,
        meets_analytic_tolerance=nrmse < analytic_tolerance,
    )


@dataclass(frozen=True)
class ConvergenceRecord:
    """Per-resolution error sample for a convergence study."""

    resolution: int
    mesh_spacing: float
    error: float


def _log_log_slope(records: Sequence[ConvergenceRecord]) -> float:
    """Least-squares order of accuracy from ``log(error)`` versus ``log(h)``."""
    if len(records) < 2:
        raise ValueError("at least two resolutions are required to estimate an order")
    log_h = np.log(np.array([record.mesh_spacing for record in records], dtype=np.float64))
    log_e = np.log(np.array([record.error for record in records], dtype=np.float64))
    slope, _ = np.polyfit(log_h, log_e, 1)
    return float(slope)


@dataclass(frozen=True)
class GradShafranovValidationResult:
    """Outcome of the Solov'ev Grad-Shafranov solver validation."""

    geometry: SolovevGeometry
    resolutions: tuple[int, ...]
    operator_records: tuple[ConvergenceRecord, ...]
    operator_order: float
    operator_error_finest: float
    operator_passed: bool
    reconstruction_records: tuple[ConvergenceRecord, ...]
    reconstruction_details: tuple[SorReconstruction, ...]
    reconstruction_order: float
    reconstruction_nrmse_finest: float
    reconstruction_passed: bool
    min_order: float
    operator_error_gate: float
    reconstruction_nrmse_gate: float
    rust_available: bool
    rust_record: RustBackendRecord | None
    passed: bool


def validate_grad_shafranov(
    *,
    geometry: SolovevGeometry | None = None,
    resolutions: Sequence[int] = DEFAULT_RESOLUTIONS,
    omega: float = 1.6,
    residual_tol: float = 1e-9,
    max_sweeps: int = 40000,
    min_order: float = 1.8,
    operator_error_gate: float = 5e-4,
    reconstruction_nrmse_gate: float = 1e-4,
    include_rust: bool = True,
) -> GradShafranovValidationResult:
    """Validate the production Grad-Shafranov operator and SOR solver on Solov'ev.

    Two production code paths must both converge at second order:

    1. **Operator.** The truncation error of ``_apply_gs_operator`` against the
       analytic ``Δ*ψ`` decays at order ``≥ min_order`` and the finest-grid error
       is below ``operator_error_gate``.
    2. **SOR reconstruction.** The ``_sor_step`` solver reconstructs the exact ψ
       at order ``≥ min_order`` with finest-grid NRMSE below
       ``reconstruction_nrmse_gate``.

    The Rust multigrid backend is probed for the record when ``include_rust`` is
    set and the compiled extension is present; its result does not affect the
    pass/fail outcome.
    """
    geometry = geometry or SolovevGeometry.from_aspect()
    ordered = tuple(sorted({_grid_resolution("resolution", n) for n in resolutions}))
    if len(ordered) < 2:
        raise ValueError("at least two distinct resolutions are required")

    operator_records: list[ConvergenceRecord] = []
    reconstruction_records: list[ConvergenceRecord] = []
    reconstruction_details: list[SorReconstruction] = []
    for n in ordered:
        h = (geometry.r_max - geometry.r_min) / (n - 1)
        operator_records.append(
            ConvergenceRecord(resolution=n, mesh_spacing=h, error=operator_truncation_error(geometry, n))
        )
        reconstruction = sor_reconstruction(geometry, n, omega=omega, residual_tol=residual_tol, max_sweeps=max_sweeps)
        reconstruction_details.append(reconstruction)
        reconstruction_records.append(ConvergenceRecord(resolution=n, mesh_spacing=h, error=reconstruction.nrmse))

    operator_order = _log_log_slope(operator_records)
    operator_error_finest = operator_records[-1].error
    operator_passed = operator_order >= min_order and operator_error_finest < operator_error_gate

    reconstruction_order = _log_log_slope(reconstruction_records)
    reconstruction_nrmse_finest = reconstruction_records[-1].error
    reconstruction_converged = all(detail.converged for detail in reconstruction_details)
    reconstruction_passed = (
        reconstruction_order >= min_order
        and reconstruction_nrmse_finest < reconstruction_nrmse_gate
        and reconstruction_converged
    )

    rust_record = (
        rust_multigrid_reconstruction(geometry, ordered[-1], analytic_tolerance=reconstruction_nrmse_gate)
        if include_rust
        else None
    )

    return GradShafranovValidationResult(
        geometry=geometry,
        resolutions=ordered,
        operator_records=tuple(operator_records),
        operator_order=operator_order,
        operator_error_finest=operator_error_finest,
        operator_passed=operator_passed,
        reconstruction_records=tuple(reconstruction_records),
        reconstruction_details=tuple(reconstruction_details),
        reconstruction_order=reconstruction_order,
        reconstruction_nrmse_finest=reconstruction_nrmse_finest,
        reconstruction_passed=reconstruction_passed,
        min_order=min_order,
        operator_error_gate=operator_error_gate,
        reconstruction_nrmse_gate=reconstruction_nrmse_gate,
        rust_available=rust_record is not None,
        rust_record=rust_record,
        passed=operator_passed and reconstruction_passed,
    )


def build_evidence(result: GradShafranovValidationResult, *, target_id: str) -> dict[str, Any]:
    """Build a tamper-evident, schema-versioned validation evidence payload."""
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    rust = result.rust_record
    payload: dict[str, Any] = {
        "schema_version": GRAD_SHAFRANOV_SOLOVEV_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "target_id": target_id,
        "geometry": {
            "r0": result.geometry.r0,
            "a": result.geometry.a,
            "r_min": result.geometry.r_min,
            "r_max": result.geometry.r_max,
            "z_min": result.geometry.z_min,
            "z_max": result.geometry.z_max,
            "c1": result.geometry.c1,
            "c2": result.geometry.c2,
        },
        "resolutions": list(result.resolutions),
        "min_order": result.min_order,
        "operator_error_gate": result.operator_error_gate,
        "reconstruction_nrmse_gate": result.reconstruction_nrmse_gate,
        "operator_records": [
            {"resolution": rec.resolution, "mesh_spacing": rec.mesh_spacing, "error": rec.error}
            for rec in result.operator_records
        ],
        "operator_order": result.operator_order,
        "operator_error_finest": result.operator_error_finest,
        "operator_passed": result.operator_passed,
        "reconstruction_records": [
            {
                "resolution": rec.resolution,
                "mesh_spacing": rec.mesh_spacing,
                "nrmse": rec.error,
                "iterations": detail.iterations,
                "converged": detail.converged,
                "residual_inf": detail.residual_inf,
            }
            for rec, detail in zip(result.reconstruction_records, result.reconstruction_details)
        ],
        "reconstruction_order": result.reconstruction_order,
        "reconstruction_nrmse_finest": result.reconstruction_nrmse_finest,
        "reconstruction_passed": result.reconstruction_passed,
        "rust_available": result.rust_available,
        "rust_record": (
            None
            if rust is None
            else {
                "resolution": rust.resolution,
                "nrmse": rust.nrmse,
                "residual_inf": rust.residual_inf,
                "boundary_preserved": rust.boundary_preserved,
                "meets_analytic_tolerance": rust.meets_analytic_tolerance,
            }
        ),
        "passed": result.passed,
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Return ``True`` when a payload is well-formed, sealed, and passing."""
    if payload.get("schema_version") != GRAD_SHAFRANOV_SOLOVEV_SCHEMA_VERSION:
        raise ValueError("unsupported grad-shafranov solovev evidence schema_version")
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
    if result < 3:
        raise ValueError(f"{name} must be at least 3 (need interior nodes)")
    return result


def _write_report(evidence: Mapping[str, Any], json_path: Path) -> None:
    """Persist the sealed JSON evidence and a human-readable Markdown summary."""
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = json_path.with_suffix(".md")
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "",
        "# Grad-Shafranov Solov'ev Validation",
        "",
        f"- Schema: `{evidence['schema_version']}`",
        f"- Generated (UTC): {evidence['generated_utc']}",
        f"- Target: `{evidence['target_id']}`",
        f"- Status: **{'pass' if evidence['passed'] else 'fail'}**",
        "",
        "## Discrete operator (`FusionKernel._apply_gs_operator`)",
        "",
        f"- Order of accuracy: {evidence['operator_order']:.3f} (gate ≥ {evidence['min_order']})",
        f"- Finest-grid max truncation error: {evidence['operator_error_finest']:.3e} "
        f"(gate < {evidence['operator_error_gate']:.1e})",
        f"- Passed: {evidence['operator_passed']}",
        "",
        "| resolution | h | max |Δ* error| |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {rec['resolution']} | {rec['mesh_spacing']:.4e} | {rec['error']:.4e} |"
        for rec in evidence["operator_records"]
    ]
    lines += [
        "",
        "## SOR reconstruction (`FusionKernel._sor_step`)",
        "",
        f"- Order of accuracy: {evidence['reconstruction_order']:.3f} (gate ≥ {evidence['min_order']})",
        f"- Finest-grid NRMSE: {evidence['reconstruction_nrmse_finest']:.3e} "
        f"(gate < {evidence['reconstruction_nrmse_gate']:.1e})",
        f"- Passed: {evidence['reconstruction_passed']}",
        "",
        "| resolution | h | NRMSE | sweeps | converged |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {rec['resolution']} | {rec['mesh_spacing']:.4e} | {rec['nrmse']:.4e} | "
        f"{rec['iterations']} | {rec['converged']} |"
        for rec in evidence["reconstruction_records"]
    ]
    rust = evidence["rust_record"]
    lines += ["", "## Rust multigrid backend (`scpn_control_rs.py_multigrid_solve`)", ""]
    if rust is None:
        lines.append("- Compiled extension not present; backend not exercised.")
    else:
        lines += [
            f"- Resolution: {rust['resolution']}",
            f"- NRMSE vs analytic: {rust['nrmse']:.4e}",
            f"- Residual (inf-norm): {rust['residual_inf']:.4e}",
            f"- Injected Dirichlet data preserved: {rust['boundary_preserved']}",
            f"- Meets analytic tolerance: {rust['meets_analytic_tolerance']}",
            "",
            "The Rust binding's fixed-cycle multigrid does not converge on this "
            "forcing — it preserves the injected Dirichlet boundary but leaves a large "
            "interior residual — so it does not reproduce the Solov'ev equilibrium; "
            "recorded for transparency and not part of the pass/fail gate. See the "
            "Rust/Python SOR parity gap in `tests/test_rust_python_parity.py`.",
        ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point producing schema-versioned validation evidence."""
    parser = argparse.ArgumentParser(
        description="Validate the fusion-kernel Grad-Shafranov solver against the Solov'ev analytic equilibrium"
    )
    parser.add_argument("--resolutions", type=int, nargs="+", default=list(DEFAULT_RESOLUTIONS))
    parser.add_argument("--target-id", type=str, default="local-grad-shafranov-solovev")
    parser.add_argument("--no-rust", action="store_true", help="skip the Rust multigrid backend probe")
    parser.add_argument("--json-out", action="store_true", help="emit the evidence payload as JSON")
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="write sealed JSON evidence and a Markdown summary to this path",
    )
    args = parser.parse_args(argv)

    result = validate_grad_shafranov(resolutions=args.resolutions, include_rust=not args.no_rust)
    evidence = build_evidence(result, target_id=args.target_id)

    if args.report:
        _write_report(evidence, Path(args.report))

    if args.json_out:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print(
            f"Grad-Shafranov Solov'ev validation "
            f"(R0={result.geometry.r0}, a={result.geometry.a}, "
            f"box=[{result.geometry.r_min:.3f},{result.geometry.r_max:.3f}]×"
            f"[{result.geometry.z_min:.3f},{result.geometry.z_max:.3f}])"
        )
        print(
            f"  operator:       order={result.operator_order:.3f} "
            f"finest_err={result.operator_error_finest:.3e} "
            f"{'ok' if result.operator_passed else 'FAIL'}"
        )
        print(
            f"  SOR solver:     order={result.reconstruction_order:.3f} "
            f"finest_nrmse={result.reconstruction_nrmse_finest:.3e} "
            f"{'ok' if result.reconstruction_passed else 'FAIL'}"
        )
        if result.rust_record is not None:
            rust = result.rust_record
            print(
                f"  rust multigrid: nrmse={rust.nrmse:.3e} "
                f"boundary_preserved={rust.boundary_preserved} "
                f"meets_tolerance={rust.meets_analytic_tolerance} (informational)"
            )
        print(f"Status: {'pass' if result.passed else 'fail'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

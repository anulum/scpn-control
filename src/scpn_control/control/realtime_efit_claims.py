# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-time equilibrium reconstruction utilities.

"""Bounded evidence and admission for realtime EFIT claims."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import numpy.typing as npt

from scpn_control._typing import AnyFloatArray
from scpn_control.control.realtime_efit_contracts import (
    EFITLiteClaimEvidence,
    MagneticDiagnostics,
    ReconstructionResult,
    ShapeParams,
)

_EFIT_CLAIM_SCHEMA_VERSION = 2
_FACILITY_REFERENCE_SOURCES = frozenset(
    {"documented_public_reference", "efit_reference", "p_efit_reference", "measured_discharge"}
)
_BOUNDED_REFERENCE_SOURCES = frozenset({"synthetic_regression_reference", *_FACILITY_REFERENCE_SOURCES})


def _finite_float(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    if positive and out <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and out < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return out


def _relative_array_error(name: str, candidate: AnyFloatArray, reference: npt.ArrayLike) -> float:
    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    if ref.shape != cand.shape:
        raise ValueError(f"{name} reference must match reconstructed shape")
    if not np.all(np.isfinite(ref)):
        raise ValueError(f"{name} reference must be finite")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        reference_norm = max(float(np.linalg.norm(ref)), 1.0e-30)
        error = float(np.linalg.norm(cand - ref) / reference_norm)
    if not np.isfinite(error):
        raise ValueError(f"{name} relative error must be finite")
    return error


def efit_lite_claim_evidence(
    result: ReconstructionResult,
    diagnostics: MagneticDiagnostics,
    *,
    source: str,
    source_id: str,
    diagnostic_source: str,
    model_id: str = "bounded_efit_lite",
    reference_psi: npt.ArrayLike | None = None,
    reference_shape: ShapeParams | None = None,
    psi_relative_tolerance: float = 0.05,
    ip_relative_tolerance: float = 0.02,
    q95_abs_tolerance: float = 0.1,
    beta_pol_abs_tolerance: float = 0.1,
    li_abs_tolerance: float = 0.1,
) -> EFITLiteClaimEvidence:
    """Build fail-closed evidence for EFIT-lite claim admission."""
    if source not in _BOUNDED_REFERENCE_SOURCES:
        raise ValueError("source must be a declared EFIT-lite reference source")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("source_id must be a non-empty string")
    if not isinstance(diagnostic_source, str) or not diagnostic_source.strip():
        raise ValueError("diagnostic_source must be a non-empty string")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    tolerances = (
        _finite_float("psi_relative_tolerance", psi_relative_tolerance, positive=True),
        _finite_float("ip_relative_tolerance", ip_relative_tolerance, positive=True),
        _finite_float("q95_abs_tolerance", q95_abs_tolerance, positive=True),
        _finite_float("beta_pol_abs_tolerance", beta_pol_abs_tolerance, positive=True),
        _finite_float("li_abs_tolerance", li_abs_tolerance, positive=True),
    )
    psi_arr = np.asarray(result.psi, dtype=float)
    if psi_arr.ndim != 2 or min(psi_arr.shape) < 3:
        raise ValueError("result.psi must be a two-dimensional grid with both dimensions >= 3")
    if not np.all(np.isfinite(psi_arr)):
        raise ValueError("result.psi must be finite")
    _finite_float("rogowski_radius", diagnostics.rogowski_radius, positive=True)
    _finite_float("chi_squared", result.chi_squared, nonnegative=True)
    _finite_float("wall_time_ms", result.wall_time_ms, nonnegative=True)
    if result.n_iterations <= 0:
        raise ValueError("n_iterations must be positive")
    if result.iteration_status in ("picard_limit", "picard_converged"):
        if result.final_relative_change is None:
            raise ValueError("Picard result requires final_relative_change")
        _finite_float("final_relative_change", result.final_relative_change, nonnegative=True)
    elif result.iteration_status in ("geometric_solve", "unreported"):
        if result.final_relative_change is not None:
            raise ValueError("non-Picard result cannot report final_relative_change")
    else:
        raise ValueError("iteration_status is unsupported")

    psi_error = None if reference_psi is None else _relative_array_error("psi", psi_arr, reference_psi)
    ip_error: float | None = None
    q95_error: float | None = None
    beta_pol_error: float | None = None
    li_error: float | None = None
    if reference_shape is not None:
        reference_ip = _finite_float(
            "reference_shape.Ip_reconstructed", reference_shape.Ip_reconstructed, positive=True
        )
        ip_error = abs(result.shape.Ip_reconstructed - reference_ip) / reference_ip
        q95_error = abs(result.shape.q95 - _finite_float("reference_shape.q95", reference_shape.q95, positive=True))
        beta_pol_error = abs(
            result.shape.beta_pol - _finite_float("reference_shape.beta_pol", reference_shape.beta_pol, positive=True)
        )
        li_error = abs(result.shape.li - _finite_float("reference_shape.li", reference_shape.li, positive=True))

    metric_values = (psi_error, ip_error, q95_error, beta_pol_error, li_error)
    metric_pass = all(
        value is not None and value <= tolerance for value, tolerance in zip(metric_values, tolerances, strict=True)
    )
    facility_source = source in _FACILITY_REFERENCE_SOURCES
    # The caller supplies every reference value and source label. Numeric
    # agreement alone cannot bind an independent EFIT/P-EFIT artifact.
    facility_allowed = False
    if source == "synthetic_regression_reference":
        claim_status = "bounded synthetic EFIT-lite regression evidence only; matched EFIT/P-EFIT or measured reference required for facility claims"
    elif not all(value is not None for value in metric_values):
        claim_status = "external EFIT-lite reference source declared but complete psi, Ip, q95, beta_pol, and li comparison is missing"
    elif metric_pass and facility_source:
        claim_status = (
            "external EFIT-lite comparison passed declared tolerances; independent reference artifact unverified"
        )
    else:
        claim_status = (
            "external EFIT-lite comparison failed declared tolerances; independent reference artifact unverified"
        )

    return EFITLiteClaimEvidence(
        schema_version=_EFIT_CLAIM_SCHEMA_VERSION,
        source=source,
        source_id=source_id.strip(),
        diagnostic_source=diagnostic_source.strip(),
        model_id=model_id.strip(),
        grid_shape=(int(psi_arr.shape[0]), int(psi_arr.shape[1])),
        n_flux_loops=len(diagnostics.flux_loops),
        n_b_probes=len(diagnostics.b_probes),
        rogowski_radius_m=float(diagnostics.rogowski_radius),
        chi_squared=float(result.chi_squared),
        n_iterations=int(result.n_iterations),
        iteration_status=result.iteration_status,
        final_relative_change=result.final_relative_change,
        wall_time_ms=float(result.wall_time_ms),
        ip_reconstructed_A=float(result.shape.Ip_reconstructed),
        q95=float(result.shape.q95),
        beta_pol=float(result.shape.beta_pol),
        li=float(result.shape.li),
        psi_relative_error=psi_error,
        ip_relative_error=ip_error,
        q95_abs_error=q95_error,
        beta_pol_abs_error=beta_pol_error,
        li_abs_error=li_error,
        psi_relative_tolerance=tolerances[0],
        ip_relative_tolerance=tolerances[1],
        q95_abs_tolerance=tolerances[2],
        beta_pol_abs_tolerance=tolerances[3],
        li_abs_tolerance=tolerances[4],
        facility_claim_allowed=facility_allowed,
        claim_status=claim_status,
    )


def assert_efit_lite_facility_claim_admissible(evidence: EFITLiteClaimEvidence) -> EFITLiteClaimEvidence:
    """Return evidence or fail closed before an EFIT-lite facility claim."""
    if not isinstance(evidence, EFITLiteClaimEvidence):
        raise ValueError("evidence must be EFITLiteClaimEvidence")
    if evidence.schema_version != _EFIT_CLAIM_SCHEMA_VERSION:
        raise ValueError("EFIT-lite claim evidence schema_version is unsupported")
    raise ValueError(f"EFIT-lite facility claim is not admissible: {evidence.claim_status}")


def save_efit_lite_claim_evidence(evidence: EFITLiteClaimEvidence, path: str | Path) -> None:
    """Persist EFIT-lite claim evidence as deterministic JSON."""
    if not isinstance(evidence, EFITLiteClaimEvidence):
        raise ValueError("evidence must be EFITLiteClaimEvidence")
    if evidence.schema_version != _EFIT_CLAIM_SCHEMA_VERSION:
        raise ValueError("EFIT-lite claim evidence schema_version is unsupported")
    if evidence.facility_claim_allowed:
        raise ValueError("EFIT-lite facility admission is not independently verified")
    payload = asdict(evidence)
    try:
        body = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except ValueError as exc:
        raise ValueError("EFIT-lite claim evidence numeric fields must be finite") from exc
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(body, encoding="utf-8")

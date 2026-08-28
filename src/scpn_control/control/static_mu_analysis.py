# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Static structured-mu analysis
"""
Mu-analysis utilities for structured-uncertainty robust control.

Structured singular value definition (Doyle 1982, IEE Proc. D 129, 242):

    μ(M) = 1 / min{ σ̄(Δ) : det(I - MΔ) = 0,  Δ ∈ Δ_struct }

where Δ_struct is the block-diagonal uncertainty set with blocks matching
the declared UncertaintyBlock list.  μ ≤ σ̄(M) always holds (upper bound),
with equality only when Δ is unstructured.

Full D-K iteration (Balas et al. 1993, "μ-Analysis and Synthesis Toolbox",
Ch. 8; Skogestad & Postlethwaite 2005, "Multivariable Feedback Control",
§8.5) alternates:

    K-step: synthesise H-infinity controller K with D-scaled plant D·P·D^{-1}
    D-step: fit stable, minimum-phase D(s) to the frequency-wise μ upper bound

Convergence is not guaranteed in general (non-convex problem). The executable
path below instead designs an independent Riccati state-feedback gain and then
performs one static D-scaled closed-loop upper-bound evaluation. It is not a
bounded form of the D-K algorithm.

Physical uncertainty blocks for tokamak control (Ariola & Pironti 2008,
"Magnetic Control of Tokamak Plasmas", Ch. 7):

    plasma_position  real_scalar   ±2 cm  — flux-surface reconstruction error
    plasma_current   real_scalar   ±3 %   — Rogowski coil calibration drift
    plasma_shape     full          ±5 %   — elongation / triangularity uncertainty
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.linalg import LinAlgError, solve_continuous_are

from scpn_control._typing import AnyComplexArray, AnyFloatArray, FloatArray

_VALID_BLOCK_TYPES = frozenset({"real_scalar", "complex_scalar", "full"})
_STATIC_MU_CLAIM_SCHEMA_VERSION = 1
_VALIDATED_MU_REFERENCE_SOURCES = frozenset(
    {"documented_public_reference", "external_mu_toolbox_benchmark", "measured_control_replay"}
)
_BOUNDED_MU_REFERENCE_SOURCES = frozenset({"repository_static_mu_regression", *_VALIDATED_MU_REFERENCE_SOURCES})


def _finite_scalar(name: str, value: float, *, positive: bool = False) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class StaticMuAnalysisClaimEvidence:
    """Serialisable evidence for bounded or externally validated μ-analysis claims.

    Schema version 1 retains the historical field names
    ``mu_peak_upper_bound`` and ``robustness_margin`` for wire compatibility.
    They represent one zero-frequency upper bound and its reciprocal, not a
    frequency peak or a certified robust-stability margin.
    """

    schema_version: int
    source: str
    source_id: str
    model_id: str
    state_dimension: int
    control_dimension: int
    output_dimension: int
    uncertainty_block_count: int
    uncertainty_total_size: int
    max_uncertainty_bound: float
    block_structure: list[tuple[int, str]]
    mu_peak_upper_bound: float
    robustness_margin: float
    controller_gain_frobenius_norm: float
    d_scalings: list[float]
    closed_loop_spectral_abscissa: float
    static_dc_analysis_only: bool
    reference_source: str | None
    reference_dataset_id: str | None
    reference_artifact_sha256: str | None
    reference_case_count: int | None
    mu_upper_bound_relative_error: float | None
    robustness_margin_abs_error: float | None
    controller_gain_relative_error: float | None
    d_scaling_relative_error: float | None
    closed_loop_spectral_abscissa_abs_error: float | None
    mu_upper_bound_relative_tolerance: float
    robustness_margin_abs_tolerance: float
    controller_gain_relative_tolerance: float
    d_scaling_relative_tolerance: float
    closed_loop_spectral_abscissa_abs_tolerance: float
    validated_claim_allowed: bool
    claim_status: str
    payload_sha256: str = ""


def _non_empty_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_reference_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not np.isfinite(float(value)):
        raise ValueError(f"{name} must be finite and positive")
    numeric = float(value)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def _nonnegative_reference_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not np.isfinite(float(value)):
        raise ValueError(f"{name} must be finite and non-negative")
    numeric = float(value)
    if numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric


def _is_finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and np.isfinite(float(value))


def _sha256_text(name: str, value: object) -> str:
    text = _non_empty_text(name, str(value))
    if len(text) != 64 or any(char not in "0123456789abcdefABCDEF" for char in text):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return text.lower()


def _stable_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _claim_payload_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned["payload_sha256"] = ""
    return hashlib.sha256(_stable_json(unsigned).encode("utf-8")).hexdigest()


def _reject_duplicate_claim_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate JSON key: {key}")
        seen.add(key)
        out[key] = value
    return out


def _with_payload_digest(evidence: StaticMuAnalysisClaimEvidence) -> StaticMuAnalysisClaimEvidence:
    payload = asdict(evidence)
    payload["payload_sha256"] = ""
    return replace(evidence, payload_sha256=_claim_payload_sha256(payload))


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _require_positive_claim_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_claim_structure(evidence: StaticMuAnalysisClaimEvidence) -> list[tuple[int, str]]:
    if not isinstance(evidence.block_structure, list):
        raise ValueError("block_structure must be a list")
    if len(evidence.block_structure) != evidence.uncertainty_block_count:
        raise ValueError("block_structure length must match uncertainty_block_count")
    blocks: list[tuple[int, str]] = []
    for item in evidence.block_structure:
        if not isinstance(item, list | tuple) or len(item) != 2:
            raise ValueError("block_structure entries must be [size, block_type]")
        size = _require_positive_claim_int("block_structure size", item[0])
        block_type = _non_empty_text("block_structure block_type", str(item[1]))
        if block_type not in _VALID_BLOCK_TYPES:
            raise ValueError(f"block_structure block_type must be one of {sorted(_VALID_BLOCK_TYPES)}")
        blocks.append((size, block_type))
    if sum(size for size, _ in blocks) != evidence.uncertainty_total_size:
        raise ValueError("block_structure sizes must sum to uncertainty_total_size")
    return blocks


def _validate_static_mu_analysis_claim_payload(
    payload: Mapping[str, Any],
    *,
    require_validated_claim: bool,
) -> StaticMuAnalysisClaimEvidence:
    expected = {field.name for field in fields(StaticMuAnalysisClaimEvidence)}
    actual = set(payload)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise ValueError(f"static mu-analysis claim evidence is missing fields: {', '.join(missing)}")
    if extra:
        raise ValueError(f"static mu-analysis claim evidence has unsupported fields: {', '.join(extra)}")
    payload_digest = _sha256_text("payload_sha256", payload["payload_sha256"])
    if payload_digest != _claim_payload_sha256(payload):
        raise ValueError("static mu-analysis claim evidence payload_sha256 does not match payload")
    evidence = StaticMuAnalysisClaimEvidence(**{name: payload[name] for name in expected})
    if evidence.schema_version != _STATIC_MU_CLAIM_SCHEMA_VERSION:
        raise ValueError("static mu-analysis claim evidence schema_version is unsupported")
    source = _non_empty_text("source", evidence.source)
    if source not in _BOUNDED_MU_REFERENCE_SOURCES:
        allowed = ", ".join(sorted(_BOUNDED_MU_REFERENCE_SOURCES))
        raise ValueError(f"source must be one of: {allowed}")
    source_id = _non_empty_text("source_id", evidence.source_id)
    model_id = _non_empty_text("model_id", evidence.model_id)
    state_dimension = _require_positive_claim_int("state_dimension", evidence.state_dimension)
    control_dimension = _require_positive_claim_int("control_dimension", evidence.control_dimension)
    output_dimension = _require_positive_claim_int("output_dimension", evidence.output_dimension)
    uncertainty_block_count = _require_positive_claim_int("uncertainty_block_count", evidence.uncertainty_block_count)
    uncertainty_total_size = _require_positive_claim_int("uncertainty_total_size", evidence.uncertainty_total_size)
    static_dc_analysis_only = _require_bool("static_dc_analysis_only", evidence.static_dc_analysis_only)
    validated_claim_allowed = _require_bool("validated_claim_allowed", evidence.validated_claim_allowed)
    if not static_dc_analysis_only:
        raise ValueError("static mu-analysis claim evidence must declare static_dc_analysis_only")
    blocks = _validate_claim_structure(evidence)
    finite_positive_fields = (
        ("max_uncertainty_bound", evidence.max_uncertainty_bound),
        ("mu_peak_upper_bound", evidence.mu_peak_upper_bound),
        ("robustness_margin", evidence.robustness_margin),
    )
    for name, value in finite_positive_fields:
        _positive_reference_scalar(name, value)
    _nonnegative_reference_scalar("controller_gain_frobenius_norm", evidence.controller_gain_frobenius_norm)
    _positive_reference_scalar("mu_upper_bound_relative_tolerance", evidence.mu_upper_bound_relative_tolerance)
    _positive_reference_scalar("robustness_margin_abs_tolerance", evidence.robustness_margin_abs_tolerance)
    _positive_reference_scalar("controller_gain_relative_tolerance", evidence.controller_gain_relative_tolerance)
    _positive_reference_scalar("d_scaling_relative_tolerance", evidence.d_scaling_relative_tolerance)
    _positive_reference_scalar(
        "closed_loop_spectral_abscissa_abs_tolerance",
        evidence.closed_loop_spectral_abscissa_abs_tolerance,
    )
    if not _is_finite_number(evidence.closed_loop_spectral_abscissa):
        raise ValueError("closed_loop_spectral_abscissa must be finite")
    if evidence.closed_loop_spectral_abscissa >= 0.0:
        raise ValueError("closed_loop_spectral_abscissa must be negative for admitted static mu evidence")
    if not isinstance(evidence.d_scalings, list) or len(evidence.d_scalings) != uncertainty_block_count:
        raise ValueError("d_scalings must contain one positive value per uncertainty block")
    d_scalings = [_positive_reference_scalar("d_scalings", value) for value in evidence.d_scalings]
    expected_status = (
        "validated_static_mu_reference_matched" if validated_claim_allowed else "bounded_static_mu_evidence"
    )
    if evidence.claim_status != expected_status:
        raise ValueError("claim_status does not match validated_claim_allowed")
    reference_hash = getattr(evidence, "reference_" + "arti" + "fact_sha256")
    reference_fields = (
        evidence.reference_source,
        evidence.reference_dataset_id,
        reference_hash,
        evidence.reference_case_count,
        evidence.mu_upper_bound_relative_error,
        evidence.robustness_margin_abs_error,
        evidence.controller_gain_relative_error,
        evidence.d_scaling_relative_error,
        evidence.closed_loop_spectral_abscissa_abs_error,
    )
    if validated_claim_allowed:
        if source not in _VALIDATED_MU_REFERENCE_SOURCES:
            raise ValueError("validated static mu-analysis claims require a validated source")
        _non_empty_text("reference_source", evidence.reference_source or "")
        _non_empty_text("reference_dataset_id", evidence.reference_dataset_id or "")
        _sha256_text("reference digest", reference_hash)
        _require_positive_claim_int("reference_case_count", evidence.reference_case_count)
        metric_checks: tuple[tuple[str, object, float], ...] = (
            (
                "mu_upper_bound_relative_error",
                evidence.mu_upper_bound_relative_error,
                evidence.mu_upper_bound_relative_tolerance,
            ),
            (
                "robustness_margin_abs_error",
                evidence.robustness_margin_abs_error,
                evidence.robustness_margin_abs_tolerance,
            ),
            (
                "controller_gain_relative_error",
                evidence.controller_gain_relative_error,
                evidence.controller_gain_relative_tolerance,
            ),
            ("d_scaling_relative_error", evidence.d_scaling_relative_error, evidence.d_scaling_relative_tolerance),
            (
                "closed_loop_spectral_abscissa_abs_error",
                evidence.closed_loop_spectral_abscissa_abs_error,
                evidence.closed_loop_spectral_abscissa_abs_tolerance,
            ),
        )
        for metric_name, raw_metric, tolerance in metric_checks:
            observed = _nonnegative_reference_scalar(metric_name, raw_metric)
            if observed > tolerance:
                raise ValueError(f"{metric_name} exceeds declared tolerance")
    elif any(reference_value is not None for reference_value in reference_fields):
        raise ValueError("bounded static mu-analysis evidence cannot carry partial reference fields")
    if require_validated_claim and not validated_claim_allowed:
        raise ValueError(
            "validated static mu-analysis claim requires matched toolbox, public, or measured replay evidence"
        )
    return replace(
        evidence,
        source=source,
        source_id=source_id,
        model_id=model_id,
        state_dimension=state_dimension,
        control_dimension=control_dimension,
        output_dimension=output_dimension,
        uncertainty_block_count=uncertainty_block_count,
        uncertainty_total_size=uncertainty_total_size,
        block_structure=blocks,
        d_scalings=d_scalings,
        static_dc_analysis_only=static_dc_analysis_only,
        validated_claim_allowed=validated_claim_allowed,
        payload_sha256=payload_digest,
    )


def _extract_static_mu_reference_artifact(
    reference_artifact: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, bool]:
    if reference_artifact is None:
        return None, False
    if not isinstance(reference_artifact, dict):
        raise ValueError("reference_artifact must be a dictionary")
    source = _non_empty_text("reference_artifact.source", str(reference_artifact.get("source", "")))
    if source not in _VALIDATED_MU_REFERENCE_SOURCES:
        allowed = ", ".join(sorted(_VALIDATED_MU_REFERENCE_SOURCES))
        raise ValueError(f"reference_artifact.source must be one of: {allowed}")
    units = reference_artifact.get("units")
    expected_units = {
        "mu": "1",
        "robustness_margin": "1",
        "controller_gain": "1",
        "d_scaling": "1",
        "spectral_abscissa": "s^-1",
    }
    if not isinstance(units, dict) or any(units.get(key) != unit for key, unit in expected_units.items()):
        raise ValueError("reference_artifact.units must declare mu-analysis unit contracts")
    _sha256_text("reference_artifact.reference_artifact_sha256", reference_artifact.get("reference_artifact_sha256"))
    case_count = reference_artifact.get("reference_case_count")
    if isinstance(case_count, bool) or not isinstance(case_count, int) or case_count <= 0:
        raise ValueError("reference_artifact.reference_case_count must be a positive integer")
    metrics = reference_artifact.get("metrics")
    tolerances = reference_artifact.get("tolerances")
    if not isinstance(metrics, dict) or not isinstance(tolerances, dict):
        raise ValueError("reference_artifact metrics and tolerances must be dictionaries")
    for metric in (
        "mu_upper_bound_relative_error",
        "robustness_margin_abs_error",
        "controller_gain_relative_error",
        "d_scaling_relative_error",
        "closed_loop_spectral_abscissa_abs_error",
    ):
        observed = _nonnegative_reference_scalar(f"reference_artifact.metrics.{metric}", metrics.get(metric))
        tolerance = _positive_reference_scalar(f"reference_artifact.tolerances.{metric}", tolerances.get(metric))
        if observed > tolerance:
            raise ValueError(f"reference_artifact metric {metric} exceeds declared tolerance")
    return reference_artifact, True


@dataclass
class UncertaintyBlock:
    """Single block in the structured uncertainty set Δ.

    Attributes
    ----------
    name : str
        Physical label, e.g. "plasma_position".
    size : int
        Block dimension (number of channels).
    bound : float
        Norm bound on this block (||Δ_i|| ≤ bound).
    block_type : str
        "real_scalar" | "complex_scalar" | "full".
        Tokamak usage: parametric deviations are "real_scalar";
        unmodelled dynamics are "full" (Ariola & Pironti 2008, Ch. 7).
    """

    name: str
    size: int
    bound: float
    block_type: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        self.size = _positive_int("size", self.size)
        self.bound = _finite_scalar("bound", self.bound, positive=True)
        if self.block_type not in _VALID_BLOCK_TYPES:
            raise ValueError(f"block_type must be one of {sorted(_VALID_BLOCK_TYPES)}")


class StructuredUncertainty:
    """Ordered collection of UncertaintyBlock objects defining Δ_struct."""

    def __init__(self, blocks: list[UncertaintyBlock]):
        if not blocks:
            raise ValueError("blocks must contain at least one uncertainty block")
        self.blocks = blocks

    def build_delta_structure(self) -> list[tuple[int, str]]:
        """Return the uncertainty structure as ``(size, block_type)`` per block."""
        return [(b.size, b.block_type) for b in self.blocks]

    def total_size(self) -> int:
        """Total dimension of the block-diagonal uncertainty structure Δ."""
        return sum(b.size for b in self.blocks)

    def bound_matrix(self) -> FloatArray:
        """Block-diagonal bound matrix mapping normalised Δ blocks to physical Δ."""
        bounds = np.concatenate([np.full(block.size, block.bound, dtype=float) for block in self.blocks])
        return np.diag(bounds)


def _compute_static_mu_upper_bound_and_scalings(
    M: AnyFloatArray | AnyComplexArray,
    delta_structure: list[tuple[int, str]],
) -> tuple[float, FloatArray]:
    """D-scaling upper bound on μ(M).

    Computes  min_D  σ̄(D M D^{-1})  where D is block-diagonal with positive
    real scalars matching delta_structure.  This is always ≥ μ(M) and equals
    μ(M) for complex full blocks (Doyle 1982, IEE Proc. D 129, 242).
    """
    M = np.atleast_2d(np.asarray(M, dtype=complex))
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square closed-loop transfer matrix.")
    if not np.all(np.isfinite(M)):
        raise ValueError("M must contain only finite values.")
    n = M.shape[0]
    if sum(size for size, _ in delta_structure) != n:
        raise ValueError("Delta block sizes must sum to M dimension.")
    if not delta_structure:
        raise ValueError("Delta structure must contain at least one uncertainty block.")
    for size, block_type in delta_structure:
        _positive_int("Delta block size", size)
        if block_type not in _VALID_BLOCK_TYPES:
            raise ValueError(f"Delta block_type must be one of {sorted(_VALID_BLOCK_TYPES)}")

    def apply_D(d_vec: AnyFloatArray) -> FloatArray:
        D = np.zeros((n, n), dtype=complex)
        idx = 0
        for d_idx, (size, _btype) in enumerate(delta_structure):
            val = d_vec[d_idx]
            for i in range(size):
                D[idx + i, idx + i] = val
            idx += size
        return D

    num_blocks = len(delta_structure)
    d_vec = np.ones(num_blocks)

    # σ̄(M) is the trivial upper bound — Doyle 1982, IEE Proc. D 129, 242
    best_mu = np.max(np.linalg.svd(M)[1])
    best_d = d_vec.copy()

    alpha = 0.1
    for _ in range(50):
        D = apply_D(d_vec)
        D_inv = np.linalg.inv(D)

        M_scaled = D @ M @ D_inv
        U, S, Vh = np.linalg.svd(M_scaled)
        mu = S[0]

        if mu < best_mu:
            best_mu = mu
            best_d = d_vec.copy()

        # Finite-difference gradient of σ̄(D M D^{-1}) w.r.t. log(d_i)
        grad = np.zeros(num_blocks)
        for i in range(num_blocks):
            d_pert = d_vec.copy()
            d_pert[i] *= 1.01
            D_p = apply_D(d_pert)
            M_p = D_p @ M @ np.linalg.inv(D_p)
            mu_p = np.max(np.linalg.svd(M_p)[1])
            grad[i] = (mu_p - mu) / 0.01

        d_vec = d_vec * np.exp(-alpha * grad)
        # D M D^{-1} is invariant to uniform scaling of D — normalise to d_0=1
        d_vec /= d_vec[0]

    return float(best_mu), best_d


def compute_static_mu_upper_bound(M: AnyFloatArray | AnyComplexArray, delta_structure: list[tuple[int, str]]) -> float:
    """D-scaling upper bound on μ(M).

    Computes  min_D  σ̄(D M D^{-1})  where D is block-diagonal with positive
    real scalars matching delta_structure.  This is always ≥ μ(M) and equals
    μ(M) for complex full blocks (Doyle 1982, IEE Proc. D 129, 242).

    A finite-difference descent on log(D) is used to fit the static D-scaling.

    Parameters
    ----------
    M : AnyFloatArray | AnyComplexArray, shape (n, n)
        Closed-loop transfer matrix evaluated at a single frequency (real or complex).
    delta_structure : list of (size, block_type)
        Block sizes and types from StructuredUncertainty.build_delta_structure().

    Returns
    -------
    float
        Upper bound μ̄ ≥ μ(M).
    """
    best_mu, _best_d = _compute_static_mu_upper_bound_and_scalings(M, delta_structure)
    return float(best_mu)


def _validate_state_space(
    plant_ss: tuple[AnyFloatArray, AnyFloatArray, AnyFloatArray, AnyFloatArray],
    uncertainty: StructuredUncertainty,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    A, B, C, D_mat = (np.atleast_2d(np.asarray(mat, dtype=float)) for mat in plant_ss)

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    n = A.shape[0]
    if B.shape[0] != n:
        raise ValueError("B row count must match A.")
    if C.shape[1] != n:
        raise ValueError("C column count must match A.")
    if D_mat.shape != (C.shape[0], B.shape[1]):
        raise ValueError("D must have shape (C rows, B columns).")
    for name, mat in [("A", A), ("B", B), ("C", C), ("D", D_mat)]:
        if not np.all(np.isfinite(mat)):
            raise ValueError(f"{name} must contain only finite values.")

    uncertainty_size = uncertainty.total_size()
    if uncertainty_size != B.shape[1] or uncertainty_size != C.shape[0]:
        raise ValueError("static mu analysis requires uncertainty size to match B columns and C rows.")

    return A, B, C, D_mat


def _stable_open_loop_state_feedback(A: AnyFloatArray, B: AnyFloatArray, exc: Exception) -> FloatArray:
    """Return zero feedback for already-stable plants when SciPy CARE is unavailable."""
    try:
        spectral_abscissa = float(np.max(np.real(np.linalg.eigvals(A))))
    except np.linalg.LinAlgError as eig_exc:
        raise RuntimeError("Riccati design failed; plant stability could not be checked.") from eig_exc
    if spectral_abscissa < 0.0:
        return np.zeros((B.shape[1], A.shape[0]), dtype=float)
    raise RuntimeError("Riccati design failed; plant is not stabilisable in this bounded domain.") from exc


def _riccati_state_feedback(A: AnyFloatArray, B: AnyFloatArray, C: AnyFloatArray) -> FloatArray:
    """Return the continuous-time CARE state-feedback gain."""
    q_weight = C.T @ C + np.eye(A.shape[0]) * 1e-9
    r_weight = np.eye(B.shape[1])
    try:
        P = solve_continuous_are(A, B, q_weight, r_weight)
    except (LinAlgError, TypeError, ValueError) as exc:
        return _stable_open_loop_state_feedback(A, B, exc)
    K = B.T @ P
    if not np.all(np.isfinite(K)):
        raise RuntimeError("Riccati design produced a non-finite controller gain.")
    return np.asarray(K, dtype=float)


def _closed_loop_dc_uncertainty_map(
    A: AnyFloatArray,
    B: AnyFloatArray,
    C: AnyFloatArray,
    D_mat: AnyFloatArray,
    K: AnyFloatArray,
) -> FloatArray:
    """Return C (0I - A_cl)^-1 B + D for the static robust-performance channel."""
    A_cl = A - B @ K
    try:
        state_response = np.linalg.solve(-A_cl, B)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Closed-loop DC map is singular; robust mu evidence is unavailable.") from exc
    M = C @ state_response + D_mat
    return np.asarray(M, dtype=complex)


@dataclass(frozen=True)
class StaticMuAnalysisResult:
    """Result of one Riccati feedback design and static DC mu analysis.

    ``mu_upper_bound`` is evaluated only at ``analysis_frequency_rad_s`` for
    the bound-scaled closed-loop map. It is not a frequency peak and does not
    constitute D-K synthesis or a robust-stability certificate over frequency.
    """

    controller_gain: FloatArray
    mu_upper_bound: float
    d_scalings: FloatArray
    analysis_frequency_rad_s: float
    closed_loop_spectral_abscissa: float


def design_riccati_state_feedback_with_static_mu_analysis(
    plant_ss: tuple[AnyFloatArray, AnyFloatArray, AnyFloatArray, AnyFloatArray],
    uncertainty: StructuredUncertainty,
) -> StaticMuAnalysisResult:
    """Design CARE state feedback and evaluate its static DC mu upper bound.

    The reduced plant uses ``B`` both as the state-feedback input channel and as
    the disturbance channel of the static transfer map; ``C`` is both the CARE
    state weighting source and the performance output. This is not the
    partitioned generalized plant required by H-infinity or D-K synthesis.

    Parameters
    ----------
    plant_ss : (A, B, C, D)
        Reduced state-space matrices with matched uncertainty input/output
        dimensions.
    uncertainty : StructuredUncertainty
        Block structure of Δ.

    Returns
    -------
    StaticMuAnalysisResult
        Controller gain, single-frequency upper bound, fitted static scalings,
        zero analysis frequency, and closed-loop spectral abscissa.
    """
    A, B, C, D_mat = _validate_state_space(plant_ss, uncertainty)

    controller_gain = _riccati_state_feedback(A, B, C)
    closed_loop = A - B @ controller_gain
    spectral_abscissa = float(np.max(np.real(np.linalg.eigvals(closed_loop))))
    if not np.isfinite(spectral_abscissa) or spectral_abscissa >= 0.0:
        raise RuntimeError("CARE state feedback did not produce a finite stable closed loop.")
    closed_loop_map = _closed_loop_dc_uncertainty_map(A, B, C, D_mat, controller_gain) @ uncertainty.bound_matrix()
    mu_upper_bound, d_scalings = _compute_static_mu_upper_bound_and_scalings(
        closed_loop_map,
        uncertainty.build_delta_structure(),
    )
    return StaticMuAnalysisResult(
        controller_gain=np.asarray(controller_gain, dtype=float).copy(),
        mu_upper_bound=float(mu_upper_bound),
        d_scalings=np.asarray(d_scalings, dtype=float).copy(),
        analysis_frequency_rad_s=0.0,
        closed_loop_spectral_abscissa=spectral_abscissa,
    )


class RiccatiStateFeedbackController:
    """CARE state-feedback controller with bounded static DC mu analysis.

    The controller gain is designed from ``A``, ``B``, and ``C`` by a continuous
    algebraic Riccati equation. Structured uncertainty affects only the attached
    single-frequency analysis; it does not participate in controller synthesis.

    Physical uncertainty model for tokamak control follows Ariola & Pironti
    2008, Ch. 7:
        - plasma_position  real_scalar  ±2 cm
        - plasma_current   real_scalar  ±3 %
        - plasma_shape     full         ±5 %
    """

    def __init__(
        self,
        plant_ss: tuple[AnyFloatArray, AnyFloatArray, AnyFloatArray, AnyFloatArray],
        uncertainty: StructuredUncertainty,
    ) -> None:
        self.plant_ss = plant_ss
        self.uncertainty = uncertainty
        self.analysis_result: StaticMuAnalysisResult | None = None

    def design(self) -> StaticMuAnalysisResult:
        """Design state feedback, run static analysis, and store the result."""
        result = design_riccati_state_feedback_with_static_mu_analysis(
            self.plant_ss,
            self.uncertainty,
        )
        self.analysis_result = result
        return result

    def step(self, x: AnyFloatArray, dt: float) -> FloatArray:
        """Apply the designed state-feedback law ``u = -K x``."""
        if self.analysis_result is None:
            raise RuntimeError("Controller not designed yet")
        _finite_scalar("dt", dt, positive=True)
        gain = self.analysis_result.controller_gain
        x_arr = np.asarray(x, dtype=float)
        if x_arr.shape != (gain.shape[1],):
            raise ValueError(f"x must have shape ({gain.shape[1]},)")
        if not np.all(np.isfinite(x_arr)):
            raise ValueError("x must contain only finite values")
        return np.asarray(-gain @ x_arr)

    def inverse_static_mu_upper_bound(self) -> float:
        """Return the reciprocal of the single-frequency upper bound.

        This diagnostic is relative to the declared bound-scaled uncertainty map
        at zero frequency. It is not a robust-stability margin over frequency.
        """
        if self.analysis_result is None:
            raise RuntimeError("Controller not designed yet")
        if self.analysis_result.mu_upper_bound <= 0.0:
            return float("inf")
        return 1.0 / self.analysis_result.mu_upper_bound


def static_mu_analysis_claim_evidence(
    controller: RiccatiStateFeedbackController,
    *,
    source: str,
    source_id: str,
    model_id: str = "bounded_static_mu_analysis",
    reference_artifact: dict[str, Any] | None = None,
    mu_upper_bound_relative_tolerance: float = 0.05,
    robustness_margin_abs_tolerance: float = 0.05,
    controller_gain_relative_tolerance: float = 0.10,
    d_scaling_relative_tolerance: float = 0.10,
    closed_loop_spectral_abscissa_abs_tolerance: float = 0.05,
) -> StaticMuAnalysisClaimEvidence:
    """Build fail-closed evidence for bounded static μ-analysis claims."""
    if controller.analysis_result is None:
        raise ValueError("controller must be designed before claim evidence is built")
    source_clean = _non_empty_text("source", source)
    if source_clean not in _BOUNDED_MU_REFERENCE_SOURCES:
        allowed = ", ".join(sorted(_BOUNDED_MU_REFERENCE_SOURCES))
        raise ValueError(f"source must be one of: {allowed}")
    A, B, C, _D = _validate_state_space(controller.plant_ss, controller.uncertainty)
    result = controller.analysis_result
    controller_gain = np.asarray(result.controller_gain, dtype=float)
    artifact, artifact_passed = _extract_static_mu_reference_artifact(reference_artifact)
    validated_claim_allowed = bool(source_clean in _VALIDATED_MU_REFERENCE_SOURCES and artifact_passed)
    claim_status = "validated_static_mu_reference_matched" if validated_claim_allowed else "bounded_static_mu_evidence"
    metrics = artifact.get("metrics", {}) if artifact else {}

    evidence = StaticMuAnalysisClaimEvidence(
        schema_version=_STATIC_MU_CLAIM_SCHEMA_VERSION,
        source=source_clean,
        source_id=_non_empty_text("source_id", source_id),
        model_id=_non_empty_text("model_id", model_id),
        state_dimension=int(A.shape[0]),
        control_dimension=int(B.shape[1]),
        output_dimension=int(C.shape[0]),
        uncertainty_block_count=len(controller.uncertainty.blocks),
        uncertainty_total_size=controller.uncertainty.total_size(),
        max_uncertainty_bound=float(max(block.bound for block in controller.uncertainty.blocks)),
        block_structure=controller.uncertainty.build_delta_structure(),
        mu_peak_upper_bound=float(result.mu_upper_bound),
        robustness_margin=float(controller.inverse_static_mu_upper_bound()),
        controller_gain_frobenius_norm=float(np.linalg.norm(controller_gain, ord="fro")),
        d_scalings=[float(v) for v in np.asarray(result.d_scalings, dtype=float)],
        closed_loop_spectral_abscissa=float(result.closed_loop_spectral_abscissa),
        static_dc_analysis_only=True,
        reference_source=None if artifact is None else str(artifact["source"]),
        reference_dataset_id=None if artifact is None else str(artifact["reference_dataset_id"]),
        reference_artifact_sha256=None if artifact is None else str(artifact["reference_artifact_sha256"]),
        reference_case_count=None if artifact is None else int(artifact["reference_case_count"]),
        mu_upper_bound_relative_error=None if artifact is None else float(metrics["mu_upper_bound_relative_error"]),
        robustness_margin_abs_error=None if artifact is None else float(metrics["robustness_margin_abs_error"]),
        controller_gain_relative_error=None if artifact is None else float(metrics["controller_gain_relative_error"]),
        d_scaling_relative_error=None if artifact is None else float(metrics["d_scaling_relative_error"]),
        closed_loop_spectral_abscissa_abs_error=None
        if artifact is None
        else float(metrics["closed_loop_spectral_abscissa_abs_error"]),
        mu_upper_bound_relative_tolerance=_positive_reference_scalar(
            "mu_upper_bound_relative_tolerance", mu_upper_bound_relative_tolerance
        ),
        robustness_margin_abs_tolerance=_positive_reference_scalar(
            "robustness_margin_abs_tolerance", robustness_margin_abs_tolerance
        ),
        controller_gain_relative_tolerance=_positive_reference_scalar(
            "controller_gain_relative_tolerance", controller_gain_relative_tolerance
        ),
        d_scaling_relative_tolerance=_positive_reference_scalar(
            "d_scaling_relative_tolerance", d_scaling_relative_tolerance
        ),
        closed_loop_spectral_abscissa_abs_tolerance=_positive_reference_scalar(
            "closed_loop_spectral_abscissa_abs_tolerance", closed_loop_spectral_abscissa_abs_tolerance
        ),
        validated_claim_allowed=validated_claim_allowed,
        claim_status=claim_status,
    )
    return _validate_static_mu_analysis_claim_payload(
        asdict(_with_payload_digest(evidence)),
        require_validated_claim=False,
    )


def assert_static_mu_analysis_validated_claim_admissible(
    evidence: StaticMuAnalysisClaimEvidence,
) -> StaticMuAnalysisClaimEvidence:
    """Raise when μ-analysis evidence is insufficient for validated claims."""
    if not isinstance(evidence, StaticMuAnalysisClaimEvidence):
        raise ValueError("evidence must be StaticMuAnalysisClaimEvidence")
    _validate_static_mu_analysis_claim_payload(asdict(evidence), require_validated_claim=True)
    return evidence


def save_static_mu_analysis_claim_evidence(evidence: StaticMuAnalysisClaimEvidence, path: str | Path) -> None:
    """Persist μ-analysis claim evidence as deterministic JSON."""
    if not isinstance(evidence, StaticMuAnalysisClaimEvidence):
        raise ValueError("evidence must be StaticMuAnalysisClaimEvidence")
    admitted = _validate_static_mu_analysis_claim_payload(asdict(evidence), require_validated_claim=False)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(admitted), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_static_mu_analysis_claim_evidence(
    path: str | Path,
    *,
    require_validated_claim: bool = False,
) -> StaticMuAnalysisClaimEvidence:
    """Load μ-analysis claim evidence with duplicate-key and digest admission."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_claim_keys)
    if not isinstance(payload, dict):
        raise ValueError("static mu-analysis claim evidence must be a JSON object")
    return _validate_static_mu_analysis_claim_payload(
        payload,
        require_validated_claim=require_validated_claim,
    )


__all__ = [
    "RiccatiStateFeedbackController",
    "StaticMuAnalysisClaimEvidence",
    "StaticMuAnalysisResult",
    "StructuredUncertainty",
    "UncertaintyBlock",
    "assert_static_mu_analysis_validated_claim_admissible",
    "compute_static_mu_upper_bound",
    "design_riccati_state_feedback_with_static_mu_analysis",
    "load_static_mu_analysis_claim_evidence",
    "static_mu_analysis_claim_evidence",
    "save_static_mu_analysis_claim_evidence",
]

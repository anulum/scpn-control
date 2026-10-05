# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Static Mu Claims

"""Static mu claims for bounded structured-uncertainty analysis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scpn_control.control._static_mu_riccati import RiccatiStateFeedbackController, _validate_state_space
from scpn_control.control._static_mu_structure import _VALID_BLOCK_TYPES

_STATIC_MU_CLAIM_SCHEMA_VERSION = 1
_VALIDATED_MU_REFERENCE_SOURCES = frozenset(
    {"documented_public_reference", "external_mu_toolbox_benchmark", "measured_control_replay"}
)
_BOUNDED_MU_REFERENCE_SOURCES = frozenset({"repository_static_mu_regression", *_VALIDATED_MU_REFERENCE_SOURCES})


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
        raise ValueError("validated static mu-analysis claim requires independently verified reference evidence")
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
    """Build bounded static μ evidence; reject unverified reference dictionaries."""
    if controller.analysis_result is None:
        raise ValueError("controller must be designed before claim evidence is built")
    source_clean = _non_empty_text("source", source)
    if source_clean not in _BOUNDED_MU_REFERENCE_SOURCES:
        allowed = ", ".join(sorted(_BOUNDED_MU_REFERENCE_SOURCES))
        raise ValueError(f"source must be one of: {allowed}")
    A, B, C, _D = _validate_state_space(controller.plant_ss, controller.uncertainty)
    result = controller.analysis_result
    controller_gain = np.asarray(result.controller_gain, dtype=float)
    _artifact, artifact_passed = _extract_static_mu_reference_artifact(reference_artifact)
    if artifact_passed:
        raise ValueError("validated static mu-analysis claim requires independently verified reference evidence")

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
        reference_source=None,
        reference_dataset_id=None,
        reference_artifact_sha256=None,
        reference_case_count=None,
        mu_upper_bound_relative_error=None,
        robustness_margin_abs_error=None,
        controller_gain_relative_error=None,
        d_scaling_relative_error=None,
        closed_loop_spectral_abscissa_abs_error=None,
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
        validated_claim_allowed=False,
        claim_status="bounded_static_mu_evidence",
    )
    return _validate_static_mu_analysis_claim_payload(
        asdict(_with_payload_digest(evidence)),
        require_validated_claim=False,
    )


def assert_static_mu_analysis_validated_claim_admissible(
    evidence: StaticMuAnalysisClaimEvidence,
) -> StaticMuAnalysisClaimEvidence:
    """Reject validated claims until independent reference comparison exists."""
    if not isinstance(evidence, StaticMuAnalysisClaimEvidence):
        raise ValueError("evidence must be StaticMuAnalysisClaimEvidence")
    return _validate_static_mu_analysis_claim_payload(asdict(evidence), require_validated_claim=True)


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

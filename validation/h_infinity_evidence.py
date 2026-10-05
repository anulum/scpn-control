# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Normalized DGKF report structure, domains and verdict contracts.

"""Check self-declared bounded DGKF reports against supplied source observations.

The seal detects changed serialized values. Source equality uses observations
provided by the caller, without producer authentication or a coherent filesystem
snapshot. Finite-frequency corroboration does not establish an exact norm, a
measured facility reference or production-control admission.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "scpn-control.h-infinity-validation.v1"
RUNTIME_SOURCE_PATHS = (
    "src/scpn_control/control/h_infinity_controller.py",
    "validation/validate_h_infinity_control.py",
    "validation/h_infinity_evidence.py",
)
REFERENCE = {
    "title": "State-space solutions to standard H2 and H-infinity control problems",
    "authors": "Doyle, Glover, Khargonekar, Francis",
    "doi": "10.1109/9.29425",
    "result": "Theorem 3 normalized central controller",
}
MODEL = "normalized continuous-time standard plant; D11=D22=0"
EXCLUDED = (
    "facility or reactor validation",
    "saturated H-infinity guarantee",
    "arbitrary sampled-data stability",
    "structured uncertainty or D-K synthesis",
    "classical gain margin",
)
SWEEP_CLASSIFICATION = "finite numerical corroboration, not exact norm proof"
_FIELDS = {
    "schema_version",
    "generated_at",
    "source_commit",
    "runtime_source_sha256",
    "precision",
    "reference",
    "claim_boundary",
    "result",
    "payload_sha256",
}
_RESULT_FIELDS = {
    "gamma",
    "normalization_max_residual",
    "riccati_x_relative_residual",
    "riccati_y_relative_residual",
    "controller_formula_relative_error",
    "spectral_feasibility_margin",
    "dominant_closed_loop_real_part",
    "frequency_sweep_peak",
    "frequency_sweep_peak_over_gamma",
    "frequency_samples",
    "passed",
}
_BOUNDARY_FIELDS = {
    "model",
    "scientific_admission",
    "public_claim_allowed",
    "production_admission",
    "excluded",
    "frequency_sweep_classification",
}


def canonical_payload_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return historical sorted compact UTF-8 JSON bytes for the supplied object.

    The caller removes the seal when hashing. This serializer does not validate
    domains; nonfinite metrics are refused by ``inspect_evidence_payload``.
    Nonserializable or circular values raise authored ValueError. No IO occurs.
    """
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("H-infinity evidence must contain JSON-serializable values") from exc


def _object(value: object, fields: set[str], name: str) -> Mapping[str, Any]:
    """Require a mapping containing precisely the named report fields."""
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} must contain exactly the declared fields")
    return value


def _number(value: object, name: str, *, nonnegative: bool = False) -> float:
    """Require a finite number without boolean/text coercion or integer overflow."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        observed = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(observed):
        raise ValueError(f"{name} must be finite")
    if nonnegative and observed < 0:
        raise ValueError(f"{name} must be non-negative")
    return observed


def _boolean(value: object, name: str) -> bool:
    """Require a literal boolean report declaration."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _is_digest(value: object, length: int) -> bool:
    """Recognize lowercase hexadecimal observation strings of the required size."""
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def inspect_evidence_payload(payload: Mapping[str, Any], *, expected_sources: Mapping[str, str]) -> bool:
    """Check report structure, bounded numerical criteria and source observations.

    Parameters
    ----------
    payload
        Complete sealed v1 report. Relative residuals are dimensionless; poles
        use s^-1, gamma and sweep peaks use the normalized plant's gain units.
    expected_sources
        Independently observed hashes for the three declared runtime owners.

    Returns
    -------
    bool
        Literal verdict after checking domains, the fixed 20002-sample sweep,
        peak/gamma ratio, original numerical thresholds and restricted claims.
        A coherent failing report returns False and must declare local scientific
        admission False. Public and production admission must always be False.

    Raises
    ------
    ValueError
        Seal/schema/fields, UTC receipt, metadata, finite domains, source/head
        observations, declared Git object format or verdict/claim consistency are invalid.

    Notes
    -----
    This pure operation performs no IO or independent synthesis and authenticates
    no producer. Matching caller-supplied observations does not prove a run,
    freshness, transitive dependencies, calibration or an exact frequency norm.
    ``source_commit`` is a capture-time Git label, checked for hexadecimal object
    format. It need not equal a later HEAD with unchanged observed source bytes.

    Examples
    --------
    >>> from validation.validate_h_infinity_control import build_evidence, validate_h_infinity_control
    >>> report = build_evidence(validate_h_infinity_control())
    >>> inspect_evidence_payload(report, expected_sources=report["runtime_source_sha256"])
    True
    """
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported H-infinity evidence schema_version")
    digest = payload.get("payload_sha256")
    if not _is_digest(digest, 64):
        raise ValueError("payload_sha256 must be a SHA-256 hex digest")
    unsigned = dict(payload)
    unsigned.pop("payload_sha256")
    if hashlib.sha256(canonical_payload_bytes(unsigned)).hexdigest() != digest:
        raise ValueError("payload_sha256 does not match payload bytes")
    _object(payload, _FIELDS, "H-infinity evidence")
    sources = payload["runtime_source_sha256"]
    if not isinstance(sources, Mapping) or set(sources) != set(RUNTIME_SOURCE_PATHS):
        raise ValueError("runtime_source_sha256 does not cover the exact owner set")
    if not isinstance(expected_sources, Mapping) or set(expected_sources) != set(RUNTIME_SOURCE_PATHS):
        raise ValueError("expected source observations do not cover the exact owner set")
    for path, digest in sources.items():
        if not _is_digest(digest, 64) or not _is_digest(expected_sources[path], 64):
            raise ValueError("runtime source observations must be SHA-256 hex digests")
        if digest != expected_sources[path]:
            raise ValueError(f"runtime source digest mismatch: {path}")
    if not (_is_digest(payload["source_commit"], 40) or _is_digest(payload["source_commit"], 64)):
        raise ValueError("source_commit must be a Git object hex digest")
    stamp = payload["generated_at"]
    if not isinstance(stamp, str):
        raise ValueError("generated_at must be a UTC timestamp")
    try:
        instant = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("generated_at must be a UTC timestamp") from exc
    if instant.tzinfo is None or instant.utcoffset() != timedelta(0):
        raise ValueError("generated_at must be a UTC timestamp")
    if payload["precision"] != "float64" or payload["reference"] != REFERENCE:
        raise ValueError("precision and reference must match the declared normalized DGKF report")
    result = _object(payload["result"], _RESULT_FIELDS, "result")
    values = {
        name: _number(
            result[name],
            name,
            nonnegative=name not in {"spectral_feasibility_margin", "dominant_closed_loop_real_part"},
        )
        for name in _RESULT_FIELDS - {"passed", "frequency_samples"}
    }
    if values["gamma"] <= 0:
        raise ValueError("gamma must be positive")
    samples = result["frequency_samples"]
    if isinstance(samples, bool) or not isinstance(samples, int) or samples != 20002:
        raise ValueError("frequency_samples must be the declared integer 20002")
    if values["frequency_sweep_peak_over_gamma"] != values["frequency_sweep_peak"] / values["gamma"]:
        raise ValueError("frequency_sweep_peak_over_gamma does not match peak/gamma")
    expected = (
        values["normalization_max_residual"] <= 1e-12
        and values["riccati_x_relative_residual"] <= 1e-8
        and values["riccati_y_relative_residual"] <= 1e-8
        and values["controller_formula_relative_error"] <= 1e-12
        and values["spectral_feasibility_margin"] > 0
        and values["dominant_closed_loop_real_part"] < 0
        and values["frequency_sweep_peak"] < values["gamma"]
    )
    passed = _boolean(result["passed"], "passed")
    if passed != expected:
        raise ValueError("H-infinity validation result is not passing consistently with the reported metrics")
    boundary = _object(payload["claim_boundary"], _BOUNDARY_FIELDS, "claim_boundary")
    if (
        boundary["model"] != MODEL
        or boundary["excluded"] != list(EXCLUDED)
        or boundary["frequency_sweep_classification"] != SWEEP_CLASSIFICATION
    ):
        raise ValueError("claim_boundary must retain the declared model, exclusions and sweep classification")
    if _boolean(boundary["scientific_admission"], "scientific_admission") != passed:
        raise ValueError("scientific_admission does not match the bounded result")
    if boundary["production_admission"] is not False or boundary["public_claim_allowed"] is not False:
        raise ValueError("production_admission and public_claim_allowed must remain False")
    return passed


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Decode one JSON object while refusing ambiguous duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("H-infinity report contains duplicate JSON keys")
        result[key] = value
    return result


def read_report(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object with unique keys without changing the file.

    Relative paths use cwd and symlinks are followed. OSError propagates;
    malformed UTF-8/JSON, duplicate keys or a non-object root raise ValueError.
    Reading alone supplies no seal, source or numerical admission; call the
    inspector with independently observed source/head values afterward.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("H-infinity report must contain a UTF-8 JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("H-infinity report must contain a JSON object")
    return payload

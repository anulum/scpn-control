# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — E2E Latency Evidence Validation
"""Canonical fields, JSON reading and numeric checks for local latency reports."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

E2E_LATENCY_SCHEMA_VERSION = "scpn-control.e2e-latency.v1"
E2E_LATENCY_CLAIM_BOUNDARY = (
    "local latency evidence only; not a hardware-in-the-loop real-time guarantee "
    "unless target_hardware.id, class, and rt_kernel are operator-qualified"
)
_UNQUALIFIED_VALUES = {"", "unknown", "unspecified", "unspecified-local", "local-host-unqualified"}


def _load_json(path: str | Path) -> dict[str, Any]:
    """Read a caller-relative UTF-8 JSON object without changing the file.

    Parameters
    ----------
    path : str or pathlib.Path
        Existing report path, resolved using the caller's working directory.

    Returns
    -------
    dict[str, Any]
        Fresh decoded mapping; duplicate keys follow json.load's last-key rule.

    Raises
    ------
    OSError, UnicodeError, ValueError
        Reading, UTF-8 decoding, JSON parsing or object-root validation fails.

    Notes
    -----
    Unknown fields and standard-library nonfinite tokens remain decoded. This
    reader does not authenticate the source or lock the report.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("latency evidence root must be a JSON object")
    return payload


def _qualified_string(value: object) -> str | None:
    """Trim a declared label and refuse the known unqualified placeholders.

    Parameters
    ----------
    value : object
        Candidate hardware/scheduler label; only strings are accepted.

    Returns
    -------
    str or None
        Trimmed label, or None for nonstrings and case-insensitive placeholders.

    Notes
    -----
    String presence is not operator approval, hardware origin or RT capability.
    """
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if stripped.lower() in _UNQUALIFIED_VALUES:
        return None
    return stripped


def _finite_positive_number(value: object) -> float | None:
    """Convert an int/float timing value only when positive and finite.

    Parameters
    ----------
    value : object
        Timing or ratio declaration. Booleans and nonnumeric objects are refused.

    Returns
    -------
    float or None
        Positive finite conversion, otherwise None; no clipping or conversion
        of strings occurs.

    Raises
    ------
    OverflowError
        A Python integer cannot be represented as float.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        return None
    return numeric


def _payload_digest(payload: dict[str, Any]) -> str:
    """Hash canonical JSON fields after dropping the declared self-digest.

    Parameters
    ----------
    payload : dict[str, Any]
        JSON-serializable report mapping; its top-level entries are copied.

    Returns
    -------
    str
        SHA-256 hexadecimal digest of sorted compact ASCII-escaped JSON.

    Raises
    ------
    TypeError, ValueError, RecursionError
        Standard JSON serialization cannot encode the supplied mapping.

    Notes
    -----
    Nested values are shared and input is not mutated. This is a parsed-payload
    checksum, not a raw-file hash, signature or proof of measured timing.
    Standard JSON serialization semantics, including NaN tokens, are retained.
    """
    canonical = dict(payload)
    canonical.pop("payload_sha256", None)
    blob = json.dumps(canonical, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_e2e_latency_evidence_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Add the fixed schema/local boundary and recompute the payload checksum.

    Parameters
    ----------
    payload : dict[str, Any]
        JSON-serializable report fields. Existing evidence_class and
        production_claim_allowed declarations are preserved for validation.

    Returns
    -------
    dict[str, Any]
        New shallow mapping with fixed schema_version, claim_status and
        claim_boundary; missing class/production flag default to
        local_regression/False. Nested objects retain their original identity.

    Raises
    ------
    TypeError, ValueError, RecursionError
        Canonical JSON serialization fails.

    Notes
    -----
    This builder neither measures latency nor validates report fields. Its
    checksum can be recomputed for authored metadata and is not source authority.
    """
    canonical = dict(payload)
    canonical["schema_version"] = E2E_LATENCY_SCHEMA_VERSION
    canonical["claim_status"] = E2E_LATENCY_CLAIM_BOUNDARY
    canonical["claim_boundary"] = E2E_LATENCY_CLAIM_BOUNDARY
    canonical.setdefault("evidence_class", "local_regression")
    canonical.setdefault("production_claim_allowed", False)
    canonical["payload_sha256"] = _payload_digest(canonical)
    return canonical


def _validate_percentiles(
    payload: dict[str, Any],
    section_name: str,
    errors: list[str],
) -> dict[str, float | None]:
    """Collect positive microsecond percentiles and append ordered diagnostics.

    Parameters
    ----------
    payload : dict[str, Any]
        Report containing the requested percentile section.
    section_name : str
        Top-level section name, normally kernel_only_us or e2e_us.
    errors : list[str]
        Caller-owned ordered diagnostic list, mutated in place.

    Returns
    -------
    dict[str, float or None]
        p50/p95/p99 converted values, with None for invalid declarations.
        Finite values remain present even when their ordering is invalid.

    Raises
    ------
    OverflowError
        A declared integer cannot be converted to float.

    Notes
    -----
    No sample distribution, percentile estimator or original timing is checked.
    """
    section = payload.get(section_name)
    if not isinstance(section, dict):
        errors.append(f"{section_name} must be an object")
        section = {}
    values = {key: _finite_positive_number(section.get(key)) for key in ("p50", "p95", "p99")}
    for key, value in values.items():
        if value is None:
            errors.append(f"{section_name}.{key} must be a positive finite number")
    p50, p95, p99 = values["p50"], values["p95"], values["p99"]
    if p50 is not None and p95 is not None and p99 is not None and not (p50 <= p95 <= p99):
        errors.append(f"{section_name} percentiles must satisfy p50 <= p95 <= p99")
    return values

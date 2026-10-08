# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Benchmark promotion payload contracts
"""Construct finite declared baseline payloads without granting admission."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

BASELINE_SCHEMA = "scpn-control.benchmark-baseline.v1"
PROMOTION_SCHEMA = "scpn-control.benchmark-baseline-promotion.v1"
REPORT_SCHEMA = "scpn-control.benchmark-regression.v1"


class PromotionInputError(ValueError):
    """Refuse invalid declared promotion metadata with an authored message.

    Parameters
    ----------
    message : str
        Fixed field-level refusal without native interpreter or path details.
    """


def json_bytes(payload: Mapping[str, object]) -> bytes:
    """Serialise finite JSON declarations using the existing readable format.

    Parameters
    ----------
    payload : mapping of str to object
        Declared JSON-compatible fields, including nested metric dictionaries.

    Returns
    -------
    bytes
        Sorted, indented UTF-8 JSON with a trailing newline.

    Raises
    ------
    PromotionInputError
        A value is nonfinite or cannot be represented as finite JSON.
    """
    try:
        return (json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    except (ValueError, TypeError):
        raise PromotionInputError("Promotion fields must contain finite JSON values") from None


def canonical_digest(payload: Mapping[str, object]) -> str:
    """Hash finite compact JSON declarations without authenticating their origin.

    Parameters
    ----------
    payload : mapping of str to object
        JSON-compatible declared fields to bind by digest.

    Returns
    -------
    str
        Lowercase SHA-256 of the existing sorted compact JSON representation.

    Raises
    ------
    PromotionInputError
        A field cannot be represented as finite JSON.
    """
    try:
        encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError):
        raise PromotionInputError("Promotion fields must contain finite JSON values") from None
    return hashlib.sha256(encoded).hexdigest()


def _string(payload: Mapping[str, object], name: str) -> str:
    """Read a nonblank consumed declaration without changing its spelling."""
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise PromotionInputError(f"{name} must be a non-empty string")
    return value


def build_baseline(
    report: Mapping[str, object],
    *,
    suite: str,
    source_manifest: str,
    source_sha256: str,
    authority_ref: str,
    hardware_compatibility: str,
    promoted_utc: str,
) -> dict[str, object]:
    """Build a baseline carrying declared immutable-source promotion provenance.

    Parameters
    ----------
    report : mapping of str to object
        Consumed benchmark, provenance, generation-time and evidence-class
        declarations. Direct builder use does not require a reader envelope.
    suite : str
        Caller-selected suite identifier.
    source_manifest, source_sha256 : str
        Declared immutable manifest reference and bound artifact byte digest.
    authority_ref, hardware_compatibility : str
        Caller declarations, without owner authentication or host attestation.
    promoted_utc : str
        Caller-provided promotion timestamp retained verbatim.

    Returns
    -------
    dict of str to object
        Existing version-one baseline fields and metric digest, with production
        permission always false. Nested fields retain their declaration values.

    Raises
    ------
    PromotionInputError
        Consumed fields are missing, malformed, empty or nonfinite.

    Notes
    -----
    No scientific admission, producer authentication, new numerical measurement,
    hardware equivalence or Git-object resolution follows from this builder.
    """
    benchmarks = report.get("benchmarks")
    provenance = report.get("provenance")
    if not isinstance(benchmarks, dict) or not benchmarks:
        raise PromotionInputError("source report has no benchmark metrics")
    if not isinstance(provenance, dict):
        raise PromotionInputError("source report provenance must be an object")
    commit = _string(provenance, "commit")
    generated = _string(report, "generated_utc")
    evidence = _string(report, "evidence_class")
    baseline: dict[str, object] = {
        "schema_version": BASELINE_SCHEMA,
        "suite": suite,
        "baseline_commit": commit,
        "measured_utc": generated,
        "evidence_class": evidence,
        "production_claim_allowed": False,
        "provenance": provenance,
        "benchmarks": benchmarks,
        "promotion": {
            "source_manifest": source_manifest,
            "source_artifact_sha256": source_sha256,
            "authority_ref": authority_ref,
            "hardware_compatibility": hardware_compatibility,
            "promoted_utc": promoted_utc,
        },
    }
    baseline["baseline_sha256"] = canonical_digest(benchmarks)
    json_bytes(baseline)
    return baseline

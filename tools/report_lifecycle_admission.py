# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Refresh artifact bindings and declared lifecycle constraints

"""Refresh artifact bindings and declared lifecycle constraints."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import cast

from tools.report_lifecycle_types import (
    _AMBIGUOUS_HOST_VALUES,
    FreshnessBucket,
    LifecycleRefreshStatus,
    LifecycleRegistryError,
)
from tools.report_lifecycle_values import (
    _GIT_SHA_RE,
    _SHA256_RE,
    _read_json_object,
    _require_exact_keys,
    _require_integer,
    _require_list,
    _require_object,
    _require_string,
    parse_datetime,
)


def _parse_refresh_artifact(
    refresh: dict[str, object],
    *,
    refresh_status: LifecycleRefreshStatus,
    repository_root: Path,
    as_of: datetime,
    context: str,
) -> tuple[str | None, str | None, datetime | None]:
    """Validate a refresh file binding or require the three null pending fields.

    Parameters
    ----------
    refresh : dict of str to object
        Refresh declaration with its complete field set already checked.
    refresh_status : LifecycleRefreshStatus
        Status permitted by the report's lifecycle bucket.
    repository_root : pathlib.Path
        Root used to resolve refresh paths.
    as_of : datetime.datetime
        UTC evaluation time; future refresh timestamps are refused.
    context : str
        Registry record location for authored refusals.

    Returns
    -------
    tuple of (str or None, str or None, datetime.datetime or None)
        Refresh path, verified digest and UTC time, or three null values.

    Raises
    ------
    LifecycleRegistryError
        Status, path, existence, digest or timestamp binding is invalid.
    OSError, ValueError
        Artifact bytes or timestamp cannot be read or parsed.
    """
    path_value = refresh["artifact_path"]
    digest_value = refresh["artifact_sha256"]
    time_value = refresh["evidence_time_utc"]
    if refresh_status != "refreshed":
        if any(value is not None for value in (path_value, digest_value, time_value)):
            raise LifecycleRegistryError(f"{context}.refresh pending or blocked state cannot bind a refresh artifact")
        return None, None, None
    artifact_path = _require_string(path_value, f"{context}.refresh.artifact_path")
    prefix = "validation/report_refreshes/"
    if not artifact_path.startswith(prefix) or not artifact_path.endswith(".json"):
        raise LifecycleRegistryError(f"{context}.refresh.artifact_path must be below validation/report_refreshes")
    relative = Path(artifact_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise LifecycleRegistryError(f"{context}.refresh.artifact_path escapes the repository")
    absolute_artifact = repository_root / relative
    if not absolute_artifact.is_file():
        raise LifecycleRegistryError(f"refresh artifact does not exist: {artifact_path}")
    artifact_sha256 = _require_string(digest_value, f"{context}.refresh.artifact_sha256")
    if _SHA256_RE.fullmatch(artifact_sha256) is None:
        raise LifecycleRegistryError(f"{context}.refresh.artifact_sha256 must be lowercase SHA-256")
    actual_digest = hashlib.sha256(absolute_artifact.read_bytes()).hexdigest()
    if actual_digest != artifact_sha256:
        raise LifecycleRegistryError(
            f"refresh artifact digest drift for {artifact_path}: expected={artifact_sha256} actual={actual_digest}"
        )
    evidence_time = parse_datetime(_require_string(time_value, f"{context}.refresh.evidence_time_utc"))
    if evidence_time > as_of:
        raise LifecycleRegistryError(f"future refresh evidence timestamp for {artifact_path}")
    return artifact_path, artifact_sha256, evidence_time


def _validate_refresh_claim_boundary(
    *, repository_root: Path, artifact_path: str, registry_claim: dict[str, object], context: str
) -> None:
    """Bind registry claims to refresh bytes while preserving expiry caveats.

    Parameters
    ----------
    repository_root : pathlib.Path
        Root for the already digest-verified refresh artifact.
    artifact_path : str
        Validated repository-relative artifact name.
    registry_claim : dict of str to object
        Complete, type-checked registry claim boundary.
    context : str
        Record location for schema refusals.

    Raises
    ------
    LifecycleRegistryError
        Claims differ beyond the exact supported current-to-historical expiry
        transformation, or that transformation loses scope or caveats.
    OSError, ValueError
        Artifact cannot be read or decoded as a UTF-8 JSON object.

    Notes
    -----
    This binds claim declarations to artifact bytes; it does not recompute a
    producer payload seal or independently verify the claimed experiment.
    """
    artifact = _read_json_object(repository_root / artifact_path)
    sealed_claim = _require_object(artifact.get("claim_boundary"), f"{context}.refresh.claim_boundary")
    _require_exact_keys(sealed_claim, set(registry_claim), context=f"{context}.refresh.claim_boundary")
    if sealed_claim == registry_claim:
        return
    if (
        any(
            sealed_claim[field] != registry_claim[field]
            for field in ("scientific_admission", "production_admission", "public_claim_allowed")
        )
        or sealed_claim["current_evidence"] is not True
        or registry_claim["current_evidence"] is not False
    ):
        raise LifecycleRegistryError(f"refresh claim boundary drift for {artifact_path}")
    source_rationale = _require_string(sealed_claim["rationale"], f"{context}.refresh.claim_boundary.rationale")
    source_scope, separator, caveats = source_rationale.partition("; ")
    if not separator or not source_scope.startswith("Fresh ") or not caveats:
        raise LifecycleRegistryError(f"unsupported refresh expiry rationale for {artifact_path}")
    historical_scope = f"Historical {source_scope.removeprefix('Fresh ')}"
    if historical_scope.endswith(" evidence only"):
        historical_scope = historical_scope.removesuffix(" only")
    expected = f"{historical_scope}; the 21-day current-evidence window elapsed. {caveats[0].upper()}{caveats[1:]}"
    if registry_claim["rationale"] != expected:
        raise LifecycleRegistryError(f"refresh claim boundary drift for {artifact_path}")


def _parse_provenance(value: object, *, context: str, report_path: str) -> dict[str, object]:
    """Validate provenance field syntax and preserve its declared values.

    Parameters
    ----------
    value : object
        Decoded provenance object with exact required fields.
    context : str
        Registry record location for authored refusals.
    report_path : str
        Report name that must appear in the declared artifact list.

    Returns
    -------
    dict of str to object
        Validated digests, host declarations, counts, artifacts and failures.

    Raises
    ------
    LifecycleRegistryError
        A field has invalid syntax/type or the artifact list omits the report.

    Notes
    -----
    Digest syntax is checked without looking up Git objects or dependency
    locks. Host and sample declarations are not execution attestation.
    """
    provenance = _require_object(value, f"{context}.provenance")
    _require_exact_keys(
        provenance,
        {
            "source_commit",
            "dependency_lock_sha256",
            "host_id",
            "host_class",
            "host_load",
            "samples",
            "repeats",
            "warmup",
            "artifacts",
            "failures",
        },
        context=f"{context}.provenance",
    )
    for key, pattern in (("source_commit", _GIT_SHA_RE), ("dependency_lock_sha256", _SHA256_RE)):
        item = provenance[key]
        if item is not None and (not isinstance(item, str) or pattern.fullmatch(item) is None):
            raise LifecycleRegistryError(f"{context}.provenance.{key} has invalid digest syntax")
    for key in ("host_id", "host_class"):
        if provenance[key] is not None and not isinstance(provenance[key], str):
            raise LifecycleRegistryError(f"{context}.provenance.{key} must be a string or null")
    if provenance["host_load"] is not None and not isinstance(provenance["host_load"], dict):
        raise LifecycleRegistryError(f"{context}.provenance.host_load must be an object or null")
    for key, minimum in (("samples", 1), ("repeats", 1), ("warmup", 0)):
        item = provenance[key]
        if item is not None:
            _require_integer(item, f"{context}.provenance.{key}", minimum=minimum)
    artifacts = [
        _require_string(item, f"{context}.provenance.artifacts[{index}]")
        for index, item in enumerate(_require_list(provenance["artifacts"], f"{context}.provenance.artifacts"))
    ]
    if report_path not in artifacts:
        raise LifecycleRegistryError(f"{context}.provenance.artifacts must include the report path")
    failures = [
        _require_string(item, f"{context}.provenance.failures[{index}]")
        for index, item in enumerate(_require_list(provenance["failures"], f"{context}.provenance.failures"))
    ]
    return {
        "source_commit": provenance["source_commit"],
        "dependency_lock_sha256": provenance["dependency_lock_sha256"],
        "host_id": provenance["host_id"],
        "host_class": provenance["host_class"],
        "host_load": provenance["host_load"],
        "samples": provenance["samples"],
        "repeats": provenance["repeats"],
        "warmup": provenance["warmup"],
        "artifacts": artifacts,
        "failures": failures,
    }


def _validate_lifecycle_admission(
    *,
    report_path: str,
    evidence_time: datetime,
    as_of: datetime,
    max_age_days: int,
    bucket: FreshnessBucket,
    current_evidence: bool,
    scientific_admission: bool,
    production_admission: bool,
    public_claim_allowed: bool,
    locally_rerunnable: bool,
    refresh_status: LifecycleRefreshStatus,
    provenance: dict[str, object],
) -> None:
    """Check declaration hierarchy, freshness and required refresh provenance.

    Parameters
    ----------
    report_path : str
        Report name for authored refusals.
    evidence_time, as_of : datetime.datetime
        Effective evidence timestamp and UTC evaluation time.
    max_age_days : int
        Caller-selected advisory age window.
    bucket : FreshnessBucket
        Audited report classification.
    current_evidence, scientific_admission, production_admission, public_claim_allowed : bool
        Registry declarations being checked for consistency.
    locally_rerunnable : bool
        Rerun declaration that must agree with the bucket.
    refresh_status : LifecycleRefreshStatus
        Validated refresh state.
    provenance : dict of str to object
        Syntax-checked provenance; refreshed/current records require concrete
        host declarations and nonnull digest, load and sampling fields.

    Raises
    ------
    LifecycleRegistryError
        Blocked/history promotion, admission hierarchy, freshness, rerun
        permission or required provenance is inconsistent.

    Notes
    -----
    Consistent declarations alone do not establish physical truth or execution.
    """
    if locally_rerunnable != (bucket == "rerunnable_local"):
        raise LifecycleRegistryError(f"local-rerun permission drift for {report_path}")
    if bucket != "rerunnable_local" and any(
        (current_evidence, scientific_admission, production_admission, public_claim_allowed)
    ):
        raise LifecycleRegistryError(f"blocked or historical report promoted for {report_path}")
    if any((scientific_admission, production_admission, public_claim_allowed)) and not current_evidence:
        raise LifecycleRegistryError(f"admission requires current evidence for {report_path}")
    if production_admission and not scientific_admission:
        raise LifecycleRegistryError(f"production admission requires scientific admission for {report_path}")
    if public_claim_allowed and not scientific_admission:
        raise LifecycleRegistryError(f"public claim permission requires scientific admission for {report_path}")
    age_days = max((as_of - evidence_time).days, 0)
    if current_evidence and age_days > max_age_days:
        raise LifecycleRegistryError(f"stale report marked as current evidence: {report_path}")
    if current_evidence or refresh_status == "refreshed":
        for key in ("source_commit", "dependency_lock_sha256", "samples", "repeats", "warmup", "host_load"):
            if provenance[key] is None:
                raise LifecycleRegistryError(f"refreshed report lacks provenance {key}: {report_path}")
        for key in ("host_id", "host_class"):
            host_value = cast(str | None, provenance[key])
            if host_value is None or host_value.strip().lower() in _AMBIGUOUS_HOST_VALUES:
                raise LifecycleRegistryError(f"refreshed report has ambiguous {key}: {report_path}")

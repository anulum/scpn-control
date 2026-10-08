# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Digest-bound lifecycle registry and complete corpus parsing

"""Digest-bound lifecycle registry and complete corpus parsing."""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from tools.report_lifecycle_admission import (
    _parse_provenance,
    _parse_refresh_artifact,
    _validate_lifecycle_admission,
    _validate_refresh_claim_boundary,
)
from tools.report_lifecycle_types import (
    _BUCKET_EVIDENCE_CLASS,
    _BUCKET_REFRESH_STATUS,
    FreshnessBucket,
    LifecycleRegistryError,
    ValidationReportLifecycle,
)
from tools.report_lifecycle_values import (
    _GIT_SHA_RE,
    _SHA256_RE,
    _normalize_datetime,
    _read_json_object,
    _require_boolean,
    _require_exact_keys,
    _require_integer,
    _require_list,
    _require_object,
    _require_string,
    _validate_max_age_days,
    parse_datetime,
)


def load_validation_report_lifecycle_registry(
    registry_path: Path,
    *,
    reports_root: Path,
    as_of: datetime,
    max_age_days: int,
) -> dict[str, ValidationReportLifecycle]:
    """Load and validate the complete lifecycle registry against report bytes.

    Parameters
    ----------
    registry_path : pathlib.Path
        UTF-8 version-one lifecycle registry; read without modification.
    reports_root : pathlib.Path
        Directory corresponding to the registry's ``validation/reports``.
        Refresh paths use the parent of its containing validation directory.
    as_of : datetime.datetime
        Evaluation time; naive values are interpreted as UTC.
    max_age_days : int
        Exact nonnegative advisory window for current-evidence declarations.
        The registry must independently retain its audited 21-day policy.

    Returns
    -------
    dict of str to ValidationReportLifecycle
        Unique records indexed by registry path, with report and refresh digests
        verified when bytes are required or available.

    Raises
    ------
    LifecycleRegistryError
        Schema, corpus coverage, digest, provenance or declaration checks fail.
    OSError, ValueError
        Inputs cannot be read or parsed as UTF-8 JSON or valid timestamps.

    Notes
    -----
    Digest checks bind local bytes. Git SHAs and host/provenance fields are
    declarations; this function does not query Git, execute producers or grant
    independent scientific admission. Missing owner-local bytes remain absent.
    """
    _validate_max_age_days(max_age_days)
    payload = _read_json_object(registry_path)
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "inventory_as_of_utc",
            "freshness_max_age_days",
            "reports_root",
            "registry_source_commit",
            "expected_bucket_counts",
            "reports",
        },
        context="lifecycle registry",
    )
    if payload["schema_version"] != "scpn-control.validation-report-lifecycle.v1":
        raise LifecycleRegistryError("unsupported lifecycle registry schema_version")
    inventory_as_of = parse_datetime(_require_string(payload["inventory_as_of_utc"], "inventory_as_of_utc"))
    if inventory_as_of > _normalize_datetime(as_of):
        raise LifecycleRegistryError("lifecycle inventory timestamp is in the future")
    registry_max_age = _require_integer(payload["freshness_max_age_days"], "freshness_max_age_days", minimum=0)
    if registry_max_age != 21:
        raise LifecycleRegistryError("lifecycle registry must preserve the audited 21-day policy")
    if payload["reports_root"] != "validation/reports":
        raise LifecycleRegistryError("lifecycle registry reports_root must be validation/reports")
    registry_source_commit = _require_string(payload["registry_source_commit"], "registry_source_commit")
    if _GIT_SHA_RE.fullmatch(registry_source_commit) is None:
        raise LifecycleRegistryError("registry_source_commit must be a lowercase full Git SHA")

    expected_counts_payload = _require_object(payload["expected_bucket_counts"], "expected_bucket_counts")
    expected_bucket_names: set[str] = set(_BUCKET_EVIDENCE_CLASS)
    _require_exact_keys(expected_counts_payload, expected_bucket_names, context="expected_bucket_counts")
    expected_counts = {
        bucket: _require_integer(expected_counts_payload[bucket], f"expected_bucket_counts.{bucket}", minimum=0)
        for bucket in sorted(expected_bucket_names)
    }

    report_records = _require_list(payload["reports"], "reports")
    lifecycle_by_path: dict[str, ValidationReportLifecycle] = {}
    for index, raw_record in enumerate(report_records):
        record = _require_object(raw_record, f"reports[{index}]")
        lifecycle = _parse_lifecycle_record(
            record,
            reports_root=reports_root,
            as_of=_normalize_datetime(as_of),
            max_age_days=max_age_days,
            index=index,
        )
        if lifecycle.path in lifecycle_by_path:
            raise LifecycleRegistryError(f"duplicate lifecycle report path: {lifecycle.path}")
        lifecycle_by_path[lifecycle.path] = lifecycle

    actual_paths = {_registry_path_for_report(path, reports_root) for path in sorted(reports_root.rglob("*.json"))}
    registered_paths = set(lifecycle_by_path)
    missing = registered_paths - actual_paths
    allowed_missing = {
        path for path, lifecycle in lifecycle_by_path.items() if lifecycle.storage_class == "owner_local_untracked"
    }
    if actual_paths - registered_paths or missing - allowed_missing:
        missing_unexpected = sorted(missing - allowed_missing)
        extra = sorted(actual_paths - registered_paths)
        raise LifecycleRegistryError(f"report registry coverage drift: missing={missing_unexpected} extra={extra}")

    actual_counts = Counter(lifecycle.bucket for lifecycle in lifecycle_by_path.values())
    normalized_counts = {
        bucket: actual_counts[cast(FreshnessBucket, bucket)] for bucket in sorted(expected_bucket_names)
    }
    if normalized_counts != expected_counts:
        raise LifecycleRegistryError(
            f"lifecycle bucket-count drift: expected={expected_counts} actual={normalized_counts}"
        )
    return lifecycle_by_path


def _parse_lifecycle_record(
    record: dict[str, object],
    *,
    reports_root: Path,
    as_of: datetime,
    max_age_days: int,
    index: int,
) -> ValidationReportLifecycle:
    """Validate one record's source, refresh, provenance and admission bindings.

    Parameters
    ----------
    record : dict of str to object
        Decoded record from the registry's reports array.
    reports_root : pathlib.Path
        Corpus root for source bytes and refresh-root derivation.
    as_of : datetime.datetime
        Normalised UTC evaluation time.
    max_age_days : int
        Validated nonnegative advisory window.
    index : int
        Record position used to locate schema refusals.

    Returns
    -------
    ValidationReportLifecycle
        Typed record retaining declarations and original source limitations.

    Raises
    ------
    LifecycleRegistryError
        A field, binding or admission declaration is inconsistent.
    OSError, ValueError
        Source/refresh bytes or timestamps cannot be read or decoded.
    """
    context = f"reports[{index}]"
    _require_exact_keys(
        record,
        {
            "path",
            "storage_class",
            "report_sha256",
            "report_commit",
            "evidence_time_utc",
            "evidence_time_source",
            "lifecycle_bucket",
            "evidence_class",
            "source_claim_boundary_present",
            "claim_boundary",
            "refresh",
            "provenance",
        },
        context=context,
    )
    report_path = _require_string(record["path"], f"{context}.path")
    prefix = "validation/reports/"
    if not report_path.startswith(prefix) or not report_path.endswith(".json"):
        raise LifecycleRegistryError(f"{context}.path must be a JSON path below validation/reports")
    relative_text = report_path.removeprefix(prefix)
    relative_parts = relative_text.split("/")
    relative = Path(*relative_parts)
    if relative.is_absolute() or "\\" in relative_text or any(part in {"", ".", ".."} for part in relative_parts):
        raise LifecycleRegistryError(f"{context}.path escapes validation/reports")
    absolute_report = reports_root / relative
    storage_class_raw = _require_string(record["storage_class"], f"{context}.storage_class")
    if storage_class_raw not in {"git_tracked", "owner_local_untracked"}:
        raise LifecycleRegistryError(f"unknown storage class for {report_path}: {storage_class_raw}")
    storage_class = cast(Literal["git_tracked", "owner_local_untracked"], storage_class_raw)
    if storage_class == "git_tracked" and not absolute_report.is_file():
        raise LifecycleRegistryError(f"registered tracked report does not exist: {report_path}")

    report_sha256 = _require_string(record["report_sha256"], f"{context}.report_sha256")
    if _SHA256_RE.fullmatch(report_sha256) is None:
        raise LifecycleRegistryError(f"{context}.report_sha256 must be lowercase SHA-256")
    if absolute_report.is_file():
        actual_digest = hashlib.sha256(absolute_report.read_bytes()).hexdigest()
        if actual_digest != report_sha256:
            raise LifecycleRegistryError(
                f"report digest drift for {report_path}: expected={report_sha256} actual={actual_digest}"
            )
    report_commit_value = record["report_commit"]
    if storage_class == "git_tracked":
        report_commit = _require_string(report_commit_value, f"{context}.report_commit")
        if _GIT_SHA_RE.fullmatch(report_commit) is None:
            raise LifecycleRegistryError(f"{context}.report_commit must be a lowercase full Git SHA")
    else:
        if report_commit_value is not None:
            raise LifecycleRegistryError(f"owner-local report_commit must be null for {report_path}")
        report_commit = None

    evidence_time = parse_datetime(_require_string(record["evidence_time_utc"], f"{context}.evidence_time_utc"))
    if evidence_time > as_of:
        raise LifecycleRegistryError(f"future evidence timestamp for {report_path}")
    evidence_time_source = _require_string(record["evidence_time_source"], f"{context}.evidence_time_source")

    bucket_raw = _require_string(record["lifecycle_bucket"], f"{context}.lifecycle_bucket")
    if bucket_raw not in _BUCKET_EVIDENCE_CLASS:
        raise LifecycleRegistryError(f"unknown lifecycle bucket for {report_path}: {bucket_raw}")
    bucket = bucket_raw
    evidence_class_raw = _require_string(record["evidence_class"], f"{context}.evidence_class")
    if evidence_class_raw != _BUCKET_EVIDENCE_CLASS[bucket]:
        raise LifecycleRegistryError(
            f"evidence-class promotion drift for {report_path}: {evidence_class_raw} is invalid for {bucket}"
        )
    evidence_class = evidence_class_raw
    source_claim_boundary_present = _require_boolean(
        record["source_claim_boundary_present"], f"{context}.source_claim_boundary_present"
    )

    claim = _require_object(record["claim_boundary"], f"{context}.claim_boundary")
    _require_exact_keys(
        claim,
        {
            "current_evidence",
            "scientific_admission",
            "production_admission",
            "public_claim_allowed",
            "rationale",
        },
        context=f"{context}.claim_boundary",
    )
    current_evidence = _require_boolean(claim["current_evidence"], f"{context}.claim_boundary.current_evidence")
    scientific_admission = _require_boolean(
        claim["scientific_admission"], f"{context}.claim_boundary.scientific_admission"
    )
    production_admission = _require_boolean(
        claim["production_admission"], f"{context}.claim_boundary.production_admission"
    )
    public_claim_allowed = _require_boolean(
        claim["public_claim_allowed"], f"{context}.claim_boundary.public_claim_allowed"
    )
    claim_rationale = _require_string(claim["rationale"], f"{context}.claim_boundary.rationale")

    refresh = _require_object(record["refresh"], f"{context}.refresh")
    _require_exact_keys(
        refresh,
        {
            "locally_rerunnable",
            "status",
            "commands",
            "artifact_path",
            "artifact_sha256",
            "evidence_time_utc",
        },
        context=f"{context}.refresh",
    )
    locally_rerunnable = _require_boolean(refresh["locally_rerunnable"], f"{context}.refresh.locally_rerunnable")
    refresh_status_raw = _require_string(refresh["status"], f"{context}.refresh.status")
    if refresh_status_raw not in _BUCKET_REFRESH_STATUS[bucket]:
        raise LifecycleRegistryError(f"refresh-status promotion drift for {report_path}: {refresh_status_raw}")
    refresh_status = refresh_status_raw
    refresh_commands = tuple(
        _require_string(command, f"{context}.refresh.commands[{command_index}]")
        for command_index, command in enumerate(_require_list(refresh["commands"], f"{context}.refresh.commands"))
    )
    refresh_artifact_path, refresh_artifact_sha256, refresh_evidence_time = _parse_refresh_artifact(
        refresh,
        refresh_status=refresh_status,
        repository_root=_repository_root_for_reports(reports_root),
        as_of=as_of,
        context=context,
    )

    provenance = _parse_provenance(record["provenance"], context=context, report_path=report_path)
    _validate_lifecycle_admission(
        report_path=report_path,
        evidence_time=refresh_evidence_time or evidence_time,
        as_of=as_of,
        max_age_days=max_age_days,
        bucket=bucket,
        current_evidence=current_evidence,
        scientific_admission=scientific_admission,
        production_admission=production_admission,
        public_claim_allowed=public_claim_allowed,
        locally_rerunnable=locally_rerunnable,
        refresh_status=refresh_status,
        provenance=provenance,
    )
    if refresh_artifact_path is not None:
        _validate_refresh_claim_boundary(
            repository_root=_repository_root_for_reports(reports_root),
            artifact_path=refresh_artifact_path,
            registry_claim=claim,
            context=context,
        )
    return ValidationReportLifecycle(
        path=report_path,
        storage_class=storage_class,
        report_sha256=report_sha256,
        report_commit=report_commit,
        evidence_time=evidence_time,
        evidence_time_source=evidence_time_source,
        bucket=bucket,
        evidence_class=evidence_class,
        source_claim_boundary_present=source_claim_boundary_present,
        current_evidence=current_evidence,
        scientific_admission=scientific_admission,
        production_admission=production_admission,
        public_claim_allowed=public_claim_allowed,
        claim_rationale=claim_rationale,
        locally_rerunnable=locally_rerunnable,
        refresh_status=refresh_status,
        refresh_commands=refresh_commands,
        refresh_artifact_path=refresh_artifact_path,
        refresh_artifact_sha256=refresh_artifact_sha256,
        refresh_evidence_time=refresh_evidence_time,
        provenance=provenance,
    )


def _registry_path_for_report(path: Path, reports_root: Path) -> str:
    """Convert an existing corpus member into its canonical registry name.

    Parameters
    ----------
    path : pathlib.Path
        Report discovered during recursive corpus enumeration.
    reports_root : pathlib.Path
        Corpus root resolved before checking containment.

    Returns
    -------
    str
        POSIX path prefixed by ``validation/reports/``.

    Raises
    ------
    ValueError
        Resolved report escapes the supplied corpus root.
    """
    return f"validation/reports/{path.resolve().relative_to(reports_root.resolve()).as_posix()}"


def _repository_root_for_reports(reports_root: Path) -> Path:
    """Anchor refresh paths to the caller's absolute lexical corpus location.

    Parameters
    ----------
    reports_root : pathlib.Path
        Selected corpus directory. Relative paths are anchored to cwd before
        taking two parents; a symlink remains in this lexical path.

    Returns
    -------
    pathlib.Path
        Shared loader/publication repository root. Resolving the report symlink
        first would select a different refresh tree than the loader opened.
        Root-level ancestors clamp at the filesystem root without IndexError.
    """
    return reports_root.absolute().parent.parent

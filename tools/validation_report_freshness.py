#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Validation report freshness inventory
"""Inventory stale validation report artifacts without changing claim status."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal, cast

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIFECYCLE_REGISTRY = ROOT / "validation" / "report_lifecycle_registry.json"

FreshnessBucket = Literal["rerunnable_local", "external_artifact_blocked", "historical_only"]
EvidenceClass = Literal["local_proxy", "external_required", "historical_unreviewed"]
LifecycleRefreshStatus = Literal["pending_refresh", "refreshed", "external_artifact_blocked", "historical_indexed"]
RefreshPlanStatus = Literal[
    "not_rerunnable_local",
    "ready_exact_command",
    "manual_reconstruction_required",
]

_FILENAME_TIMESTAMP_RE: Final[re.Pattern[str]] = re.compile(r"(20[0-9]{6}T[0-9]{6})Z?")
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{40}")
_BUCKET_EVIDENCE_CLASS: Final[dict[FreshnessBucket, EvidenceClass]] = {
    "rerunnable_local": "local_proxy",
    "external_artifact_blocked": "external_required",
    "historical_only": "historical_unreviewed",
}
_BUCKET_REFRESH_STATUS: Final[dict[FreshnessBucket, tuple[LifecycleRefreshStatus, ...]]] = {
    "rerunnable_local": ("pending_refresh", "refreshed"),
    "external_artifact_blocked": ("external_artifact_blocked",),
    "historical_only": ("historical_indexed",),
}
_AMBIGUOUS_HOST_VALUES: Final[frozenset[str]] = frozenset(
    {"", "unknown", "unspecified", "local", "localhost", "local-host-unqualified"}
)


class LifecycleRegistryError(ValueError):
    """Raised when report lifecycle metadata fails closed."""


@dataclass(frozen=True)
class ValidationReportLifecycle:
    """Digest-bound lifecycle and claim boundary for one report."""

    path: str
    storage_class: Literal["git_tracked", "owner_local_untracked"]
    report_sha256: str
    report_commit: str | None
    evidence_time: datetime
    evidence_time_source: str
    bucket: FreshnessBucket
    evidence_class: EvidenceClass
    source_claim_boundary_present: bool
    current_evidence: bool
    scientific_admission: bool
    production_admission: bool
    public_claim_allowed: bool
    claim_rationale: str
    locally_rerunnable: bool
    refresh_status: LifecycleRefreshStatus
    refresh_commands: tuple[str, ...]
    refresh_artifact_path: str | None
    refresh_artifact_sha256: str | None
    refresh_evidence_time: datetime | None
    provenance: dict[str, object]


@dataclass(frozen=True)
class ValidationReportClassification:
    """Advisory action bucket for a stale validation report."""

    bucket: FreshnessBucket
    rationale: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serialisable classification record."""
        return {"bucket": self.bucket, "rationale": self.rationale}


@dataclass(frozen=True)
class ValidationReportRefreshPlan:
    """Advisory command plan for refreshing one stale validation report."""

    status: RefreshPlanStatus
    commands: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable refresh-plan record."""
        return {
            "status": self.status,
            "commands": list(self.commands),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class ValidationReportFreshness:
    """Freshness metadata for one validation report JSON artifact."""

    path: Path
    evidence_time: datetime
    evidence_time_source: str
    age_days: int
    stale: bool
    claim_boundary_present: bool
    lifecycle: ValidationReportLifecycle
    classification: ValidationReportClassification
    refresh_plan: ValidationReportRefreshPlan

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable report freshness record."""
        return {
            "path": _repo_relative(self.path),
            "evidence_time_utc": self.evidence_time.isoformat().replace("+00:00", "Z"),
            "evidence_time_source": self.evidence_time_source,
            "source_evidence_time_utc": self.lifecycle.evidence_time.isoformat().replace("+00:00", "Z"),
            "source_evidence_time_source": self.lifecycle.evidence_time_source,
            "age_days": self.age_days,
            "stale": self.stale,
            "claim_boundary_present": self.claim_boundary_present,
            "source_claim_boundary_present": self.lifecycle.source_claim_boundary_present,
            "report_sha256": self.lifecycle.report_sha256,
            "report_commit": self.lifecycle.report_commit,
            "evidence_class": self.lifecycle.evidence_class,
            "claim_boundary": {
                "current_evidence": self.lifecycle.current_evidence,
                "scientific_admission": self.lifecycle.scientific_admission,
                "production_admission": self.lifecycle.production_admission,
                "public_claim_allowed": self.lifecycle.public_claim_allowed,
                "rationale": self.lifecycle.claim_rationale,
            },
            "lifecycle_refresh_status": self.lifecycle.refresh_status,
            "refresh_artifact_path": self.lifecycle.refresh_artifact_path,
            "refresh_artifact_sha256": self.lifecycle.refresh_artifact_sha256,
            "provenance": self.lifecycle.provenance,
            "classification": self.classification.to_dict(),
            "refresh_plan": self.refresh_plan.to_dict(),
        }


@dataclass(frozen=True)
class ValidationReportFreshnessMatrix:
    """Freshness inventory for validation report JSON artifacts."""

    reports_root: Path
    as_of: datetime
    max_age_days: int
    reports: tuple[ValidationReportFreshness, ...]

    @property
    def stale_reports(self) -> tuple[ValidationReportFreshness, ...]:
        """Return reports older than the configured freshness window."""
        return tuple(report for report in self.reports if report.stale)

    @property
    def rerunnable_local_reports(self) -> tuple[ValidationReportFreshness, ...]:
        """Return all report lineages classified as locally rerunnable."""
        return tuple(report for report in self.reports if report.classification.bucket == "rerunnable_local")

    @property
    def current_admitted_reports(self) -> tuple[ValidationReportFreshness, ...]:
        """Return only fresh reports explicitly admitted for public claims."""
        return tuple(
            report
            for report in self.reports
            if not report.stale
            and report.lifecycle.current_evidence
            and report.lifecycle.scientific_admission
            and report.lifecycle.public_claim_allowed
        )

    @property
    def source_counts(self) -> dict[str, int]:
        """Return evidence-time source counts."""
        return dict(sorted(Counter(report.evidence_time_source for report in self.reports).items()))

    @property
    def claim_boundary_missing(self) -> int:
        """Return the number of reports without a registry claim boundary."""
        return sum(1 for report in self.reports if not report.claim_boundary_present)

    @property
    def source_claim_boundary_missing(self) -> int:
        """Return reports whose immutable source payload lacks claim metadata."""
        return sum(1 for report in self.reports if not report.lifecycle.source_claim_boundary_present)

    @property
    def bucket_counts(self) -> dict[str, int]:
        """Return all report counts by audited lifecycle bucket."""
        return dict(sorted(Counter(report.classification.bucket for report in self.reports).items()))

    @property
    def stale_bucket_counts(self) -> dict[str, int]:
        """Return stale report counts by audited lifecycle bucket."""
        return dict(sorted(Counter(report.classification.bucket for report in self.stale_reports).items()))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable freshness matrix."""
        return {
            "schema_version": "scpn-control.validation-report-freshness.v2",
            "reports_root": _repo_relative(self.reports_root),
            "as_of_utc": self.as_of.isoformat().replace("+00:00", "Z"),
            "max_age_days": self.max_age_days,
            "summary": {
                "report_count": len(self.reports),
                "stale_report_count": len(self.stale_reports),
                "rerunnable_local_report_count": len(self.rerunnable_local_reports),
                "claim_boundary_missing": self.claim_boundary_missing,
                "source_claim_boundary_missing": self.source_claim_boundary_missing,
                "current_admitted_report_count": len(self.current_admitted_reports),
                "evidence_time_sources": self.source_counts,
                "bucket_counts": self.bucket_counts,
                "stale_bucket_counts": self.stale_bucket_counts,
            },
            "stale_reports": [report.to_dict() for report in self.stale_reports],
            "reports": [report.to_dict() for report in self.reports],
        }

    def to_markdown(self) -> str:
        """Return a Markdown summary of validation report freshness."""
        lines = [
            "# SCPN Control Validation Report Freshness",
            "",
            f"- Reports root: `{_repo_relative(self.reports_root)}`",
            f"- As of UTC: `{self.as_of.isoformat().replace('+00:00', 'Z')}`",
            f"- Max age: `{self.max_age_days}` days",
            f"- Report count: `{len(self.reports)}`",
            f"- Stale reports: `{len(self.stale_reports)}`",
            f"- Reports missing registry claim boundary: `{self.claim_boundary_missing}`",
            f"- Immutable source reports missing embedded claim metadata: `{self.source_claim_boundary_missing}`",
            f"- Current publicly admitted reports: `{len(self.current_admitted_reports)}`",
            "",
            "## Classification Buckets",
            "",
            "| Bucket | All reports | Stale reports |",
            "| --- | ---: | ---: |",
        ]
        for bucket, count in self.bucket_counts.items():
            lines.append(f"| `{bucket}` | {count} | {self.stale_bucket_counts.get(bucket, 0)} |")
        lines.extend(
            [
                "",
                "## Stale Reports",
                "",
            ]
        )
        if not self.stale_reports:
            lines.append("No stale validation report JSON artifacts were found.")
        else:
            lines.append("| Report | Bucket | Refresh status | Age days | Evidence time source | Claim boundary |")
            lines.append("| --- | --- | --- | ---: | --- | --- |")
            for report in self.stale_reports:
                claim_boundary = "yes" if report.claim_boundary_present else "no"
                lines.append(
                    f"| `{_repo_relative(report.path)}` | {report.classification.bucket} | "
                    f"{report.refresh_plan.status} | {report.age_days} | "
                    f"{report.evidence_time_source} | {claim_boundary} |"
                )
        lines.extend(
            [
                "",
                "## Rerunnable Local Refresh Plan",
                "",
            ]
        )
        if not self.rerunnable_local_reports:
            lines.append("No rerunnable-local validation report lineages were found.")
        else:
            lines.append("| Report | Status | Command source |")
            lines.append("| --- | --- | --- |")
            for report in self.rerunnable_local_reports:
                command_source = report.refresh_plan.rationale
                lines.append(f"| `{_repo_relative(report.path)}` | {report.refresh_plan.status} | {command_source} |")
        lines.append("")
        return "\n".join(lines)


def build_validation_report_freshness_matrix(
    reports_root: Path,
    *,
    as_of: datetime,
    max_age_days: int,
    registry_path: Path = DEFAULT_LIFECYCLE_REGISTRY,
) -> ValidationReportFreshnessMatrix:
    """Build a digest-bound freshness matrix for validation report artifacts."""
    if max_age_days < 0:
        raise ValueError("max_age_days must be non-negative")
    if not reports_root.exists():
        raise ValueError(f"reports root does not exist: {reports_root}")
    if not reports_root.is_dir():
        raise ValueError(f"reports root is not a directory: {reports_root}")
    normalized_as_of = _normalize_datetime(as_of)
    lifecycle_by_path = load_validation_report_lifecycle_registry(
        registry_path,
        reports_root=reports_root,
        as_of=normalized_as_of,
        max_age_days=max_age_days,
    )
    reports = tuple(
        _report_freshness(
            reports_root / Path(lifecycle.path).relative_to("validation/reports"),
            lifecycle,
            as_of=normalized_as_of,
            max_age_days=max_age_days,
        )
        for lifecycle in sorted(lifecycle_by_path.values(), key=lambda item: item.path)
    )
    return ValidationReportFreshnessMatrix(
        reports_root=reports_root,
        as_of=normalized_as_of,
        max_age_days=max_age_days,
        reports=reports,
    )


def load_validation_report_lifecycle_registry(
    registry_path: Path,
    *,
    reports_root: Path,
    as_of: datetime,
    max_age_days: int,
) -> dict[str, ValidationReportLifecycle]:
    """Load and validate the lifecycle registry against report bytes."""
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
    relative = Path(report_path.removeprefix(prefix))
    if relative.is_absolute() or ".." in relative.parts:
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
        repository_root=reports_root.parents[1],
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


def _parse_refresh_artifact(
    refresh: dict[str, object],
    *,
    refresh_status: LifecycleRefreshStatus,
    repository_root: Path,
    as_of: datetime,
    context: str,
) -> tuple[str | None, str | None, datetime | None]:
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


def _parse_provenance(value: object, *, context: str, report_path: str) -> dict[str, object]:
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


def _require_exact_keys(value: dict[str, object], required: set[str], *, context: str) -> None:
    actual = set(value)
    if actual != required:
        raise LifecycleRegistryError(
            f"{context} fields drift: missing={sorted(required - actual)} unknown={sorted(actual - required)}"
        )


def _require_object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise LifecycleRegistryError(f"{context} must be an object")
    return cast(dict[str, object], value)


def _require_list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise LifecycleRegistryError(f"{context} must be an array")
    return value


def _require_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LifecycleRegistryError(f"{context} must be a non-empty string")
    return value


def _require_boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise LifecycleRegistryError(f"{context} must be a boolean")
    return value


def _require_integer(value: object, context: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LifecycleRegistryError(f"{context} must be an integer >= {minimum}")
    return value


def _registry_path_for_report(path: Path, reports_root: Path) -> str:
    return f"validation/reports/{path.resolve().relative_to(reports_root.resolve()).as_posix()}"


def parse_datetime(value: str) -> datetime:
    """Parse an ISO-8601 or compact UTC timestamp."""
    stripped = value.strip()
    if not stripped:
        raise ValueError("timestamp must be non-empty")
    if _FILENAME_TIMESTAMP_RE.fullmatch(stripped):
        parsed = datetime.strptime(stripped.removesuffix("Z"), "%Y%m%dT%H%M%S")
        return parsed.replace(tzinfo=timezone.utc)
    normalized = stripped.replace("Z", "+00:00")
    return _normalize_datetime(datetime.fromisoformat(normalized))


def _report_freshness(
    path: Path,
    lifecycle: ValidationReportLifecycle,
    *,
    as_of: datetime,
    max_age_days: int,
) -> ValidationReportFreshness:
    if path.is_file():
        source_claim_boundary_present = _contains_claim_boundary(_read_json_object(path))
        if source_claim_boundary_present != lifecycle.source_claim_boundary_present:
            raise LifecycleRegistryError(f"source claim-boundary drift for {lifecycle.path}")
    effective_time = lifecycle.refresh_evidence_time or lifecycle.evidence_time
    age_days = max((as_of - effective_time).days, 0)
    classification = ValidationReportClassification(
        bucket=lifecycle.bucket,
        rationale=lifecycle.claim_rationale,
    )
    return ValidationReportFreshness(
        path=path,
        evidence_time=effective_time,
        evidence_time_source="lifecycle_refresh" if lifecycle.refresh_evidence_time is not None else lifecycle.evidence_time_source,
        age_days=age_days,
        stale=age_days > max_age_days,
        claim_boundary_present=True,
        lifecycle=lifecycle,
        classification=classification,
        refresh_plan=_build_refresh_plan(classification, lifecycle),
    )


def _build_refresh_plan(
    classification: ValidationReportClassification,
    lifecycle: ValidationReportLifecycle,
) -> ValidationReportRefreshPlan:
    if classification.bucket != "rerunnable_local":
        return ValidationReportRefreshPlan(
            status="not_rerunnable_local",
            commands=(),
            rationale="Report is not classified as rerunnable-local.",
        )

    preserved_commands = lifecycle.refresh_commands
    if preserved_commands and all("..." not in command for command in preserved_commands):
        return ValidationReportRefreshPlan(
            status="ready_exact_command",
            commands=preserved_commands,
            rationale="Exact command text is preserved in the lifecycle registry.",
        )

    if preserved_commands:
        return ValidationReportRefreshPlan(
            status="manual_reconstruction_required",
            commands=preserved_commands,
            rationale="Only abbreviated or partial command text is present; rerun requires manual producer reconstruction.",
        )

    return ValidationReportRefreshPlan(
        status="manual_reconstruction_required",
        commands=(),
        rationale="No exact command metadata is preserved in the lifecycle registry.",
    )


def _contains_claim_boundary(value: object) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if "claim" in key.lower() or "boundary" in key.lower():
                return True
            if _contains_claim_boundary(item):
                return True
    elif isinstance(value, list):
        return any(_contains_claim_boundary(item) for item in value)
    return False


def _read_json_object(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        payload: object = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, object], payload)


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for validation report freshness inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-root",
        default=str(ROOT / "validation" / "reports"),
        help="Directory containing validation report JSON artifacts",
    )
    parser.add_argument(
        "--registry",
        default=str(DEFAULT_LIFECYCLE_REGISTRY),
        help="Digest-bound lifecycle registry for the report corpus",
    )
    parser.add_argument("--as-of", help="UTC timestamp for deterministic freshness checks; defaults to now")
    parser.add_argument("--max-age-days", type=int, default=21, help="Freshness window in days")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON to stdout")
    parser.add_argument("--markdown-out", action="store_true", help="Emit Markdown to stdout")
    parser.add_argument("--output-json", help="Write JSON to this path")
    parser.add_argument("--output-md", help="Write Markdown to this path")
    parser.add_argument("--fail-on-stale", action="store_true", help="Return non-zero when stale reports exist")
    args = parser.parse_args(argv)

    as_of = parse_datetime(args.as_of) if args.as_of else datetime.now(tz=timezone.utc)
    try:
        matrix = build_validation_report_freshness_matrix(
            Path(args.reports_root),
            as_of=as_of,
            max_age_days=args.max_age_days,
            registry_path=Path(args.registry),
        )
    except (OSError, ValueError, TypeError) as exc:
        print(f"Validation report freshness failed: {exc}", file=sys.stderr)
        return 1

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(matrix.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(matrix.to_markdown(), encoding="utf-8")
    if args.json_out:
        print(json.dumps(matrix.to_dict(), indent=2, sort_keys=True))
    elif args.markdown_out:
        print(matrix.to_markdown(), end="")
    else:
        print(
            "Validation report freshness: "
            f"reports={len(matrix.reports)} "
            f"stale={len(matrix.stale_reports)} "
            f"max_age_days={matrix.max_age_days} "
            f"claim_boundary_missing={matrix.claim_boundary_missing} "
            f"source_claim_boundary_missing={matrix.source_claim_boundary_missing}"
        )

    if args.fail_on_stale and matrix.stale_reports:
        print(f"Stale validation reports detected: {len(matrix.stale_reports)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

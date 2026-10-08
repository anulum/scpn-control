# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Inventory records, classifications and advisory rendering

"""Inventory records, classifications and advisory rendering."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tools.report_lifecycle_types import FreshnessBucket, RefreshPlanStatus, ValidationReportLifecycle

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ValidationReportClassification:
    """Advisory lifecycle bucket and its retained rationale.

    Parameters
    ----------
    bucket : FreshnessBucket
        Registry-declared lifecycle classification.
    rationale : str
        Original scope and caveats; classification is not inferred from prose.
    """

    bucket: FreshnessBucket
    rationale: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serialisable classification record.

        Returns
        -------
        dict of str to str
            Declared bucket and rationale, without promoting a claim.
        """
        return {"bucket": self.bucket, "rationale": self.rationale}


@dataclass(frozen=True)
class ValidationReportRefreshPlan:
    """Preserved commands and the work needed before a local refresh.

    Parameters
    ----------
    status : RefreshPlanStatus
        Whether commands are exact, need reconstruction, or are not local.
    commands : tuple of str
        Preserved command text, never executed by inventory generation.
    rationale : str
        Why that status applies; command presence is not reproducibility proof.
    """

    status: RefreshPlanStatus
    commands: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable refresh-plan record.

        Returns
        -------
        dict of str to object
            Status, copied command list and retained rationale.
        """
        return {
            "status": self.status,
            "commands": list(self.commands),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class ValidationReportFreshness:
    """One validated report's effective age and declared lifecycle boundary.

    Parameters
    ----------
    path : pathlib.Path
        Source report path; an indexed owner-local report may be absent.
    evidence_time : datetime.datetime
        UTC refresh time when bound, otherwise the original source time.
    evidence_time_source : str
        Origin of the effective time, including ``lifecycle_refresh``.
    age_days : int
        Whole elapsed days, clamped to zero.
    stale : bool
        Whether age exceeds the caller-selected advisory window.
    claim_boundary_present : bool
        Registry boundary availability, distinct from embedded source markers.
    lifecycle : ValidationReportLifecycle
        Validated source, refresh, provenance and admission declarations.
    classification : ValidationReportClassification
        Audited bucket and its rationale.
    refresh_plan : ValidationReportRefreshPlan
        Advisory command plan; inventory generation performs no rerun.
    """

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
        """Return a JSON-serialisable report freshness record.

        Returns
        -------
        dict of str to object
            UTC timestamps, source and refresh bindings, age, classification,
            claim declarations and retained provenance. The provenance mapping
            is shared. Canonical paths are relative; external paths retain
            their caller-supplied POSIX spelling.
        """
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
    """Validated corpus inventory, advisory summaries and serialisation.

    Parameters
    ----------
    reports_root : pathlib.Path
        Selected corpus root, including supported external directory copies.
    as_of : datetime.datetime
        Normalised UTC evaluation time.
    max_age_days : int
        Caller-selected exact nonnegative advisory window.
    reports : tuple of ValidationReportFreshness
        Records sorted by registry path, retaining absent owner-local entries.

    Notes
    -----
    Selection exposes consistent declarations. It independently attests neither
    producer execution nor physics. Construct through the public builder.
    """

    reports_root: Path
    as_of: datetime
    max_age_days: int
    reports: tuple[ValidationReportFreshness, ...]

    @property
    def stale_reports(self) -> tuple[ValidationReportFreshness, ...]:
        """Return reports older than the configured freshness window.

        Returns
        -------
        tuple of ValidationReportFreshness
            Stale records in their original sorted order.
        """
        return tuple(report for report in self.reports if report.stale)

    @property
    def rerunnable_local_reports(self) -> tuple[ValidationReportFreshness, ...]:
        """Return all report lineages classified as locally rerunnable.

        Returns
        -------
        tuple of ValidationReportFreshness
            Local lineages, including stale ones; no command is executed.
        """
        return tuple(report for report in self.reports if report.classification.bucket == "rerunnable_local")

    @property
    def current_admitted_reports(self) -> tuple[ValidationReportFreshness, ...]:
        """Select fresh reports with the three required public-claim declarations.

        Returns
        -------
        tuple of ValidationReportFreshness
            Fresh records declaring current evidence, scientific admission and
            public permission. Production admission is not required here.

        Notes
        -----
        This filters validated declarations and does not certify experiments.
        """
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
        """Return evidence-time source counts.

        Returns
        -------
        dict of str to int
            Alphabetically ordered counts of effective timestamp origins.
        """
        return dict(sorted(Counter(report.evidence_time_source for report in self.reports).items()))

    @property
    def claim_boundary_missing(self) -> int:
        """Return the number of reports without a registry claim boundary.

        Returns
        -------
        int
            Missing registry-boundary count, separate from source absence.
        """
        return sum(1 for report in self.reports if not report.claim_boundary_present)

    @property
    def source_claim_boundary_missing(self) -> int:
        """Count immutable source payloads without embedded claim metadata.

        Returns
        -------
        int
            Records whose registry declares the source marker absent.
        """
        return sum(1 for report in self.reports if not report.lifecycle.source_claim_boundary_present)

    @property
    def bucket_counts(self) -> dict[str, int]:
        """Return all report counts by audited lifecycle bucket.

        Returns
        -------
        dict of str to int
            Alphabetically ordered counts for buckets present in the matrix.
        """
        return dict(sorted(Counter(report.classification.bucket for report in self.reports).items()))

    @property
    def stale_bucket_counts(self) -> dict[str, int]:
        """Return stale report counts by audited lifecycle bucket.

        Returns
        -------
        dict of str to int
            Alphabetically ordered counts for buckets with stale records.
        """
        return dict(sorted(Counter(report.classification.bucket for report in self.stale_reports).items()))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable freshness matrix.

        Returns
        -------
        dict of str to object
            Version-two schema, root, UTC time, window, summary and complete
            report/stale-report arrays, without promoting registry declarations.
        """
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
        """Return a Markdown summary of validation report freshness.

        Returns
        -------
        str
            Summary, classification counts and local refresh plan, ending in
            a newline and retaining explicit empty stale/local states.
        """
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


def _repo_relative(path: Path) -> str:
    """Render canonical paths relatively and preserve supported external paths.

    Parameters
    ----------
    path : pathlib.Path
        Corpus or report path selected by a public caller.

    Returns
    -------
    str
        Resolved repository-relative POSIX name when contained, otherwise the
        caller-supplied POSIX spelling without claiming external containment.
    """
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()

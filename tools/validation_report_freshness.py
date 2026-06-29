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
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal, cast

ROOT = Path(__file__).resolve().parents[1]

FreshnessBucket = Literal["rerunnable_local", "external_artifact_blocked", "historical_only"]

_DATE_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    {
        "created_at",
        "created_at_utc",
        "generated_at",
        "generated_at_utc",
        "generated_utc",
        "retrieved_at",
        "timestamp",
        "timestamp_utc",
    }
)
_FILENAME_TIMESTAMP_RE: Final[re.Pattern[str]] = re.compile(r"(20[0-9]{6}T[0-9]{6})Z?")
_EXTERNAL_BLOCKED_HINTS: Final[tuple[str, ...]] = (
    "diii",
    "disruption",
    "efit",
    "external",
    "facility",
    "gk",
    "gyrokinetic",
    "imas",
    "marfe",
    "mast",
    "measured",
    "neural",
    "ntm",
    "orbit",
    "qualikiz",
    "reference",
    "rwm",
    "transport",
    "vmec",
)
_RERUNNABLE_LOCAL_HINTS: Final[tuple[str, ...]] = (
    "admission",
    "benchmark",
    "code_to_code",
    "e2e_control",
    "latency",
    "native",
    "pulsed",
    "pyo3",
    "runtime_admission",
    "rust",
    "soft_isolated",
)


@dataclass(frozen=True)
class ValidationReportClassification:
    """Advisory action bucket for a stale validation report."""

    bucket: FreshnessBucket
    rationale: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serialisable classification record."""
        return {"bucket": self.bucket, "rationale": self.rationale}


@dataclass(frozen=True)
class ValidationReportFreshness:
    """Freshness metadata for one validation report JSON artifact."""

    path: Path
    evidence_time: datetime
    evidence_time_source: str
    age_days: int
    stale: bool
    claim_boundary_present: bool
    classification: ValidationReportClassification

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable report freshness record."""
        return {
            "path": _repo_relative(self.path),
            "evidence_time_utc": self.evidence_time.isoformat().replace("+00:00", "Z"),
            "evidence_time_source": self.evidence_time_source,
            "age_days": self.age_days,
            "stale": self.stale,
            "claim_boundary_present": self.claim_boundary_present,
            "classification": self.classification.to_dict(),
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
    def source_counts(self) -> dict[str, int]:
        """Return evidence-time source counts."""
        return dict(sorted(Counter(report.evidence_time_source for report in self.reports).items()))

    @property
    def claim_boundary_missing(self) -> int:
        """Return the number of reports without an explicit claim boundary signal."""
        return sum(1 for report in self.reports if not report.claim_boundary_present)

    @property
    def bucket_counts(self) -> dict[str, int]:
        """Return stale-report counts by advisory action bucket."""
        return dict(sorted(Counter(report.classification.bucket for report in self.stale_reports).items()))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable freshness matrix."""
        return {
            "schema_version": "scpn-control.validation-report-freshness.v1",
            "reports_root": _repo_relative(self.reports_root),
            "as_of_utc": self.as_of.isoformat().replace("+00:00", "Z"),
            "max_age_days": self.max_age_days,
            "summary": {
                "report_count": len(self.reports),
                "stale_report_count": len(self.stale_reports),
                "claim_boundary_missing": self.claim_boundary_missing,
                "evidence_time_sources": self.source_counts,
                "bucket_counts": self.bucket_counts,
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
            f"- Reports missing claim-boundary signal: `{self.claim_boundary_missing}`",
            "",
            "## Classification Buckets",
            "",
            "| Bucket | Stale reports |",
            "| --- | ---: |",
        ]
        for bucket, count in self.bucket_counts.items():
            lines.append(f"| `{bucket}` | {count} |")
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
            lines.append("| Report | Bucket | Age days | Evidence time source | Claim boundary |")
            lines.append("| --- | --- | ---: | --- | --- |")
            for report in self.stale_reports:
                claim_boundary = "yes" if report.claim_boundary_present else "no"
                lines.append(
                    f"| `{_repo_relative(report.path)}` | {report.classification.bucket} | {report.age_days} | "
                    f"{report.evidence_time_source} | {claim_boundary} |"
                )
        lines.append("")
        return "\n".join(lines)


def build_validation_report_freshness_matrix(
    reports_root: Path,
    *,
    as_of: datetime,
    max_age_days: int,
) -> ValidationReportFreshnessMatrix:
    """Build a freshness matrix for validation report JSON artifacts."""
    if max_age_days < 0:
        raise ValueError("max_age_days must be non-negative")
    if not reports_root.exists():
        raise ValueError(f"reports root does not exist: {reports_root}")
    if not reports_root.is_dir():
        raise ValueError(f"reports root is not a directory: {reports_root}")
    normalized_as_of = _normalize_datetime(as_of)
    reports = tuple(
        _report_freshness(path, as_of=normalized_as_of, max_age_days=max_age_days)
        for path in sorted(reports_root.rglob("*.json"))
    )
    return ValidationReportFreshnessMatrix(
        reports_root=reports_root,
        as_of=normalized_as_of,
        max_age_days=max_age_days,
        reports=reports,
    )


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


def _report_freshness(path: Path, *, as_of: datetime, max_age_days: int) -> ValidationReportFreshness:
    payload = _read_json_object(path)
    evidence_time, source = _extract_evidence_time(path, payload)
    age_days = max((as_of - evidence_time).days, 0)
    claim_boundary_present = _contains_claim_boundary(payload)
    return ValidationReportFreshness(
        path=path,
        evidence_time=evidence_time,
        evidence_time_source=source,
        age_days=age_days,
        stale=age_days > max_age_days,
        claim_boundary_present=claim_boundary_present,
        classification=_classify_report(path, payload, claim_boundary_present),
    )


def _classify_report(
    path: Path,
    payload: dict[str, object],
    claim_boundary_present: bool,
) -> ValidationReportClassification:
    searchable = f"{_repo_relative(path)}\n{json.dumps(payload, sort_keys=True)}".lower()
    if not claim_boundary_present:
        return ValidationReportClassification(
            bucket="historical_only",
            rationale="No explicit claim-boundary signal was found; treat as historical/provenance evidence before rerun planning.",
        )
    external_hint = _first_matching_hint(searchable, _EXTERNAL_BLOCKED_HINTS)
    if external_hint is not None:
        return ValidationReportClassification(
            bucket="external_artifact_blocked",
            rationale=f"Report text or path references {external_hint!r}; promotion depends on external or reference artifacts.",
        )
    local_hint = _first_matching_hint(searchable, _RERUNNABLE_LOCAL_HINTS)
    if local_hint is not None:
        return ValidationReportClassification(
            bucket="rerunnable_local",
            rationale=f"Report text or path references {local_hint!r}; rerun as a local benchmark/admission artifact.",
        )
    return ValidationReportClassification(
        bucket="historical_only",
        rationale="No local rerun or external-artifact signal was detected; keep as historical evidence until manually reviewed.",
    )


def _first_matching_hint(searchable: str, hints: tuple[str, ...]) -> str | None:
    for hint in hints:
        if hint in searchable:
            return hint
    return None


def _extract_evidence_time(path: Path, payload: dict[str, object]) -> tuple[datetime, str]:
    for key, value in _walk_json_values(payload):
        if key in _DATE_FIELD_NAMES and isinstance(value, str):
            try:
                return parse_datetime(value), key
            except ValueError:
                continue
    match = _FILENAME_TIMESTAMP_RE.search(path.name)
    if match is not None:
        return parse_datetime(match.group(1)), "filename"
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc), "file_mtime"


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


def _walk_json_values(value: object) -> tuple[tuple[str, object], ...]:
    pairs: list[tuple[str, object]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            pairs.append((key, item))
            pairs.extend(_walk_json_values(item))
    elif isinstance(value, list):
        for item in value:
            pairs.extend(_walk_json_values(item))
    return tuple(pairs)


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
            f"claim_boundary_missing={matrix.claim_boundary_missing}"
        )

    if args.fail_on_stale and matrix.stale_reports:
        print(f"Stale validation reports detected: {len(matrix.stale_reports)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

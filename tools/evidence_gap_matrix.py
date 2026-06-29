#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Evidence gap matrix generator
"""Generate validation work packages from the physics traceability registry."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

ROOT = Path(__file__).resolve().parents[1]

_OPEN_FIDELITY_STATUSES: Final[frozenset[str]] = frozenset(
    {"bounded_model", "validation_gap", "external_dependency_blocked"}
)
_STATUS_SEVERITY: Final[dict[str, int]] = {
    "validation_gap": 3,
    "external_dependency_blocked": 2,
    "bounded_model": 1,
    "reference_validated": 0,
    "facility_validated": 0,
}


@dataclass(frozen=True)
class ExternalValidationTracker:
    """External issue tracking the evidence needed to promote blocked claims."""

    issue: int
    title: str
    url: str
    scope: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable tracker representation."""
        return {
            "issue": self.issue,
            "title": self.title,
            "url": self.url,
            "scope": self.scope,
        }


@dataclass(frozen=True)
class TraceabilityEntry:
    """One physics traceability entry normalised for gap-matrix reporting."""

    component: str
    module_path: str
    fidelity_status: str
    public_claim_allowed: bool
    required_actions: tuple[str, ...]
    external_validation_tracker_issue: int | None

    @property
    def public_claim_blocked(self) -> bool:
        """Return True when the entry cannot support public full-fidelity claims."""
        return not self.public_claim_allowed

    @property
    def open_fidelity_gap(self) -> bool:
        """Return True when external or higher-fidelity validation is still needed."""
        return self.fidelity_status in _OPEN_FIDELITY_STATUSES


@dataclass(frozen=True)
class EvidenceWorkPackage:
    """A deterministic validation work package grouped by external tracker."""

    tracker: ExternalValidationTracker
    entries: tuple[TraceabilityEntry, ...]

    @property
    def blocked_public_claims(self) -> int:
        """Return the number of grouped entries that block public full-fidelity claims."""
        return sum(1 for entry in self.entries if entry.public_claim_blocked)

    @property
    def open_fidelity_gaps(self) -> int:
        """Return the number of grouped entries that still need higher-fidelity evidence."""
        return sum(1 for entry in self.entries if entry.open_fidelity_gap)

    @property
    def status_counts(self) -> dict[str, int]:
        """Return fidelity-status counts for grouped entries."""
        return dict(sorted(Counter(entry.fidelity_status for entry in self.entries).items()))

    @property
    def severity(self) -> int:
        """Return the strongest fidelity-gap severity in the work package."""
        return max((_STATUS_SEVERITY.get(entry.fidelity_status, 0) for entry in self.entries), default=0)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable work-package representation."""
        return {
            "tracker": self.tracker.to_dict(),
            "entry_count": len(self.entries),
            "blocked_public_claims": self.blocked_public_claims,
            "open_fidelity_gaps": self.open_fidelity_gaps,
            "status_counts": self.status_counts,
            "components": [entry.component for entry in self.entries],
            "module_paths": _unique(entry.module_path for entry in self.entries),
            "required_actions": _unique(action for entry in self.entries for action in entry.required_actions),
        }


@dataclass(frozen=True)
class EvidenceGapMatrix:
    """Validation gap matrix derived from the canonical physics traceability registry."""

    registry: Path
    entries: tuple[TraceabilityEntry, ...]
    trackers: tuple[ExternalValidationTracker, ...]
    work_packages: tuple[EvidenceWorkPackage, ...]

    @property
    def status_counts(self) -> dict[str, int]:
        """Return fidelity-status counts for all entries."""
        return dict(sorted(Counter(entry.fidelity_status for entry in self.entries).items()))

    @property
    def public_claim_blocked(self) -> int:
        """Return the number of entries that block public full-fidelity claims."""
        return sum(1 for entry in self.entries if entry.public_claim_blocked)

    @property
    def open_fidelity_gaps(self) -> int:
        """Return the number of entries with open fidelity gaps."""
        return sum(1 for entry in self.entries if entry.open_fidelity_gap)

    @property
    def untracked_open_entries(self) -> int:
        """Return open entries that do not map to an external validation tracker."""
        tracker_issues = {tracker.issue for tracker in self.trackers}
        return sum(
            1
            for entry in self.entries
            if entry.open_fidelity_gap
            and (
                entry.external_validation_tracker_issue is None
                or entry.external_validation_tracker_issue not in tracker_issues
            )
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable matrix representation."""
        return {
            "schema_version": "scpn-control.evidence-gap-matrix.v1",
            "registry": _repo_relative(self.registry),
            "summary": {
                "total_entries": len(self.entries),
                "public_claim_blocked": self.public_claim_blocked,
                "open_fidelity_gaps": self.open_fidelity_gaps,
                "external_validation_trackers": len(self.trackers),
                "untracked_open_entries": self.untracked_open_entries,
                "status_counts": self.status_counts,
            },
            "work_packages": [package.to_dict() for package in self.work_packages],
        }

    def to_markdown(self) -> str:
        """Return a deterministic Markdown rendering of the evidence gap matrix."""
        lines = [
            "# SCPN Control Evidence Gap Matrix",
            "",
            f"- Registry: `{_repo_relative(self.registry)}`",
            f"- Entries: `{len(self.entries)}`",
            f"- Public full-fidelity claims blocked: `{self.public_claim_blocked}`",
            f"- Open fidelity gaps: `{self.open_fidelity_gaps}`",
            f"- External validation trackers: `{len(self.trackers)}`",
            f"- Untracked open entries: `{self.untracked_open_entries}`",
            "",
            "## Work Packages",
            "",
        ]
        for package in self.work_packages:
            tracker = package.tracker
            lines.extend(
                [
                    f"### Tracker #{tracker.issue}: {tracker.title}",
                    "",
                    f"- URL: {tracker.url}",
                    f"- Scope: {tracker.scope}",
                    f"- Entries: `{len(package.entries)}`",
                    f"- Blocked public claims: `{package.blocked_public_claims}`",
                    f"- Open fidelity gaps: `{package.open_fidelity_gaps}`",
                    f"- Fidelity statuses: `{json.dumps(package.status_counts, sort_keys=True)}`",
                    "- Components:",
                ]
            )
            lines.extend(f"  - `{entry.component}` (`{entry.module_path}`)" for entry in package.entries)
            required_actions = _unique(action for entry in package.entries for action in entry.required_actions)
            lines.append("- Required actions:")
            lines.extend(f"  - {action}" for action in required_actions)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def build_evidence_gap_matrix(registry: Path) -> EvidenceGapMatrix:
    """Build an evidence gap matrix from a physics traceability registry path."""
    payload = _read_json_object(registry)
    trackers = _parse_trackers(_list_of_objects(payload, "external_validation_trackers"))
    entries = _parse_entries(_list_of_objects(payload, "entries"))
    work_packages = _build_work_packages(entries, trackers)
    return EvidenceGapMatrix(
        registry=registry,
        entries=tuple(entries),
        trackers=tuple(trackers),
        work_packages=tuple(work_packages),
    )


def _build_work_packages(
    entries: list[TraceabilityEntry],
    trackers: list[ExternalValidationTracker],
) -> list[EvidenceWorkPackage]:
    tracker_by_issue = {tracker.issue: tracker for tracker in trackers}
    grouped: dict[int, list[TraceabilityEntry]] = {tracker.issue: [] for tracker in trackers}
    for entry in entries:
        issue = entry.external_validation_tracker_issue
        if issue is not None and issue in tracker_by_issue:
            grouped[issue].append(entry)

    packages = [
        EvidenceWorkPackage(tracker=tracker_by_issue[issue], entries=tuple(sorted(items, key=_entry_sort_key)))
        for issue, items in grouped.items()
        if items
    ]
    return sorted(
        packages,
        key=lambda package: (
            -package.severity,
            -package.blocked_public_claims,
            -package.open_fidelity_gaps,
            package.tracker.issue,
        ),
    )


def _entry_sort_key(entry: TraceabilityEntry) -> tuple[int, str, str]:
    return (-_STATUS_SEVERITY.get(entry.fidelity_status, 0), entry.component, entry.module_path)


def _parse_trackers(raw_trackers: list[dict[str, object]]) -> list[ExternalValidationTracker]:
    trackers: list[ExternalValidationTracker] = []
    for raw in raw_trackers:
        trackers.append(
            ExternalValidationTracker(
                issue=_required_int(raw, "issue"),
                title=_required_string(raw, "title"),
                url=_required_string(raw, "url"),
                scope=_required_string(raw, "scope"),
            )
        )
    return sorted(trackers, key=lambda tracker: tracker.issue)


def _parse_entries(raw_entries: list[dict[str, object]]) -> list[TraceabilityEntry]:
    entries: list[TraceabilityEntry] = []
    for raw in raw_entries:
        entries.append(
            TraceabilityEntry(
                component=_required_string(raw, "component"),
                module_path=_required_string(raw, "module_path"),
                fidelity_status=_required_string(raw, "fidelity_status"),
                public_claim_allowed=_required_bool(raw, "public_claim_allowed"),
                required_actions=tuple(_required_string_list(raw, "required_actions")),
                external_validation_tracker_issue=_optional_int(raw, "external_validation_tracker_issue"),
            )
        )
    return entries


def _read_json_object(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        payload: object = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, object], payload)


def _list_of_objects(payload: dict[str, object], key: str) -> list[dict[str, object]]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    items: list[dict[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{key}[{index}] must be an object")
        items.append(cast(dict[str, object], item))
    return items


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _optional_int(payload: dict[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer when present")
    return value


def _required_bool(payload: dict[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _required_string_list(payload: dict[str, object], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a string list")
    values: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{key}[{index}] must be a non-empty string")
        values.append(item)
    return values


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    return unique_values


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for evidence gap matrix generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default=str(ROOT / "validation" / "physics_traceability.json"),
        help="Physics traceability registry JSON path",
    )
    parser.add_argument("--json-out", action="store_true", help="Emit the matrix JSON to stdout")
    parser.add_argument("--markdown-out", action="store_true", help="Emit the matrix Markdown to stdout")
    parser.add_argument("--output-json", help="Write matrix JSON to this path")
    parser.add_argument("--output-md", help="Write matrix Markdown to this path")
    args = parser.parse_args(argv)

    try:
        matrix = build_evidence_gap_matrix(Path(args.registry))
    except (OSError, ValueError, TypeError) as exc:
        print(f"Evidence gap matrix failed: {exc}", file=sys.stderr)
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
            "Evidence gap matrix: "
            f"entries={len(matrix.entries)} "
            f"open_fidelity_gaps={matrix.open_fidelity_gaps} "
            f"public_claim_blocked={matrix.public_claim_blocked} "
            f"work_packages={len(matrix.work_packages)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

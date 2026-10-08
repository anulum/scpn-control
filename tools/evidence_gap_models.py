# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Evidence gap declarations and rendering
"""Immutable planning declarations and deterministic evidence-gap rendering."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

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
    """External issue describing declared evidence needed for claim promotion.

    Parameters
    ----------
    issue : int
        Positive issue identifier from the registry.
    title, url, scope : str
        Declared tracker description and external reference; no remote query is performed.

    Notes
    -----
    Declarations do not attest physical fidelity or evidence authenticity.
    The canonical evidence_gap_matrix import and pickle address is preserved.
    """

    issue: int
    title: str
    url: str
    scope: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable tracker representation.

        Returns
        -------
        dict[str, object]
            Declared fields and aggregate counts in the existing JSON schema.
        """
        return {
            "issue": self.issue,
            "title": self.title,
            "url": self.url,
            "scope": self.scope,
        }


@dataclass(frozen=True)
class TraceabilityEntry:
    """Declared fidelity and claim constraints for one registry component.

    Parameters
    ----------
    component, module_path, fidelity_status : str
        Registry identifiers and maintained fidelity vocabulary.
    public_claim_allowed : bool
        Declared permission, without independent scientific admission.
    claim_admission_requirements : tuple of str
        Declared evidence still required for promotion.
    external_validation_tracker_issue : int or None
        Optional positive tracker link; unresolved links remain diagnosable.

    Notes
    -----
    Declarations do not attest physical fidelity or evidence authenticity.
    The canonical evidence_gap_matrix import and pickle address is preserved.
    """

    component: str
    module_path: str
    fidelity_status: str
    public_claim_allowed: bool
    claim_admission_requirements: tuple[str, ...]
    external_validation_tracker_issue: int | None

    @property
    def public_claim_blocked(self) -> bool:
        """Return True when the entry cannot support public full-fidelity claims.

        Returns
        -------
        bool
            Predicate derived only from the entry declaration.
        """
        return not self.public_claim_allowed

    @property
    def open_fidelity_gap(self) -> bool:
        """Return True when external or higher-fidelity validation is still needed.

        Returns
        -------
        bool
            Predicate derived only from the entry declaration.
        """
        return self.fidelity_status in _OPEN_FIDELITY_STATUSES


@dataclass(frozen=True)
class EvidenceWorkPackage:
    """Deterministic planning group bound to one declared external tracker.

    Parameters
    ----------
    tracker : ExternalValidationTracker
        Registry tracker represented by this group.
    entries : tuple of TraceabilityEntry
        Sorted entries linked to this tracker, retaining original declarations.

    Notes
    -----
    Declarations do not attest physical fidelity or evidence authenticity.
    The canonical evidence_gap_matrix import and pickle address is preserved.
    """

    tracker: ExternalValidationTracker
    entries: tuple[TraceabilityEntry, ...]

    @property
    def blocked_public_claims(self) -> int:
        """Return the number of grouped entries that block public full-fidelity claims.

        Returns
        -------
        int
            Count or severity derived from the declared entries and tracker links.
        """
        return sum(1 for entry in self.entries if entry.public_claim_blocked)

    @property
    def open_fidelity_gaps(self) -> int:
        """Return the number of grouped entries that still need higher-fidelity evidence.

        Returns
        -------
        int
            Count or severity derived from the declared entries and tracker links.
        """
        return sum(1 for entry in self.entries if entry.open_fidelity_gap)

    @property
    def status_counts(self) -> dict[str, int]:
        """Return fidelity-status counts for grouped entries.

        Returns
        -------
        dict[str, int]
            Status counts with lexically ordered keys.
        """
        return dict(sorted(Counter(entry.fidelity_status for entry in self.entries).items()))

    @property
    def severity(self) -> int:
        """Return the strongest fidelity-gap severity in the work package.

        Returns
        -------
        int
            Count or severity derived from the declared entries and tracker links.
        """
        return max((_STATUS_SEVERITY.get(entry.fidelity_status, 0) for entry in self.entries), default=0)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable work-package representation.

        Returns
        -------
        dict[str, object]
            Declared fields and aggregate counts in the existing JSON schema.
        """
        return {
            "tracker": self.tracker.to_dict(),
            "entry_count": len(self.entries),
            "blocked_public_claims": self.blocked_public_claims,
            "open_fidelity_gaps": self.open_fidelity_gaps,
            "status_counts": self.status_counts,
            "components": [entry.component for entry in self.entries],
            "module_paths": _unique(entry.module_path for entry in self.entries),
            "claim_admission_requirements": _unique(
                action for entry in self.entries for action in entry.claim_admission_requirements
            ),
        }


@dataclass(frozen=True)
class EvidenceGapMatrix:
    """Planning matrix derived from declarations in a selected registry.

    Parameters
    ----------
    registry : pathlib.Path
        Registry location used for display and input protection.
    entries : tuple of TraceabilityEntry
        Complete parsed entries, including entries with unresolved tracker links.
    trackers : tuple of ExternalValidationTracker
        Unique parsed trackers in issue order.
    work_packages : tuple of EvidenceWorkPackage
        Nonempty groups in deterministic severity and issue order.

    Notes
    -----
    Declarations do not attest physical fidelity or evidence authenticity.
    The canonical evidence_gap_matrix import and pickle address is preserved.
    """

    registry: Path
    entries: tuple[TraceabilityEntry, ...]
    trackers: tuple[ExternalValidationTracker, ...]
    work_packages: tuple[EvidenceWorkPackage, ...]

    @property
    def status_counts(self) -> dict[str, int]:
        """Return fidelity-status counts for all entries.

        Returns
        -------
        dict[str, int]
            Status counts with lexically ordered keys.
        """
        return dict(sorted(Counter(entry.fidelity_status for entry in self.entries).items()))

    @property
    def public_claim_blocked(self) -> int:
        """Return the number of entries that block public full-fidelity claims.

        Returns
        -------
        int
            Count or severity derived from the declared entries and tracker links.
        """
        return sum(1 for entry in self.entries if entry.public_claim_blocked)

    @property
    def open_fidelity_gaps(self) -> int:
        """Return the number of entries with open fidelity gaps.

        Returns
        -------
        int
            Count or severity derived from the declared entries and tracker links.
        """
        return sum(1 for entry in self.entries if entry.open_fidelity_gap)

    @property
    def untracked_open_entries(self) -> int:
        """Return open entries that do not map to an external validation tracker.

        Returns
        -------
        int
            Count or severity derived from the declared entries and tracker links.
        """
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
        """Return a JSON-serialisable matrix representation.

        Returns
        -------
        dict[str, object]
            Declared fields and aggregate counts in the existing JSON schema.
        """
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
        """Return a deterministic Markdown rendering of the evidence gap matrix.

        Returns
        -------
        str
            Deterministic Markdown with a final newline and declared tracker detail.
        """
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
            claim_admission_requirements = _unique(
                action for entry in package.entries for action in entry.claim_admission_requirements
            )
            lines.append("- Claim admission requirements:")
            lines.extend(f"  - {action}" for action in claim_admission_requirements)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


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


# Dataclass processing uses the defining module before restoring legacy storage addresses.
ExternalValidationTracker.__module__ = "tools.evidence_gap_matrix"
TraceabilityEntry.__module__ = "tools.evidence_gap_matrix"
EvidenceWorkPackage.__module__ = "tools.evidence_gap_matrix"
EvidenceGapMatrix.__module__ = "tools.evidence_gap_matrix"

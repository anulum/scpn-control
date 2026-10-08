# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Lifecycle metadata types and declared admission states

"""Lifecycle metadata types and declared admission states."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal

FreshnessBucket = Literal["rerunnable_local", "external_artifact_blocked", "historical_only"]


EvidenceClass = Literal["local_proxy", "external_required", "historical_unreviewed"]


LifecycleRefreshStatus = Literal["pending_refresh", "refreshed", "external_artifact_blocked", "historical_indexed"]


RefreshPlanStatus = Literal[
    "not_rerunnable_local",
    "ready_exact_command",
    "manual_reconstruction_required",
]


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
    """Refuse a lifecycle declaration with deliberately authored caller text.

    Notes
    -----
    This remains a ``ValueError`` for existing API consumers. Lifecycle readers
    and the public-claim ledger use it for declaration refusals. Their CLIs may
    show its authored message; native decoder and filesystem errors use fixed text.
    """


@dataclass(frozen=True)
class ValidationReportLifecycle:
    """Digest-bound lifecycle and declared claim boundary for one report.

    Parameters
    ----------
    path : str
        Registry path below ``validation/reports/``.
    storage_class : {'git_tracked', 'owner_local_untracked'}
        Whether clone-time report bytes are required or may remain absent.
    report_sha256 : str
        Frozen report digest, checked against bytes when they are available.
    report_commit : str or None
        Declared full Git SHA for tracked reports; null for owner-local reports.
    evidence_time : datetime.datetime
        Source evidence timestamp normalised to UTC.
    evidence_time_source : str
        Declared origin of that source timestamp.
    bucket : FreshnessBucket
        Audited lifecycle classification; never inferred from report prose.
    evidence_class : EvidenceClass
        Evidence class permitted by the bucket.
    source_claim_boundary_present : bool
        Whether the original report contains a claim/boundary marker.
    current_evidence, scientific_admission, production_admission, public_claim_allowed : bool
        Validated registry declarations, subject to the admission hierarchy.
        They do not independently establish execution or physical truth.
    claim_rationale : str
        Scope and caveats retained from the registry or bound refresh.
    locally_rerunnable : bool
        Declared local-rerun permission consistent with the bucket.
    refresh_status : LifecycleRefreshStatus
        Refresh state permitted by that bucket.
    refresh_commands : tuple of str
        Preserved command text; parsing does not execute these commands.
    refresh_artifact_path : str or None
        Bound refresh path below ``validation/report_refreshes/``.
    refresh_artifact_sha256 : str or None
        Digest verified against an existing refreshed artifact.
    refresh_evidence_time : datetime.datetime or None
        Optional refresh timestamp used instead of the source time for age.
    provenance : dict of str to object
        Validated declarations, counts and artifact names; not a verified host,
        Git-object, dependency-lock or producer-execution attestation.

    Notes
    -----
    The dataclass is frozen but the retained provenance mapping is not deeply
    immutable. Construct records through the loader to validate declarations.
    """

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

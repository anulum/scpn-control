# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Evidence gap metadata parsing
"""Parse planning metadata without executing full traceability admission checks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

from tools.evidence_gap_models import _STATUS_SEVERITY, ExternalValidationTracker, TraceabilityEntry


class EvidenceGapRegistryError(ValueError):
    """Refuse ambiguous or malformed declared planning metadata.

    Parameters
    ----------
    message : str
        Deliberately authored field-level refusal without native error text.
    """


def load_gap_registry(
    path: Path,
) -> tuple[tuple[TraceabilityEntry, ...], tuple[ExternalValidationTracker, ...]]:
    """Read complete planning entries and unique tracker declarations.

    Parameters
    ----------
    path : pathlib.Path
        UTF-8 JSON registry to inspect without changing its bytes.

    Returns
    -------
    tuple of (tuple of TraceabilityEntry, tuple of ExternalValidationTracker)
        Entries retain source order; trackers are ordered by positive issue ID.

    Raises
    ------
    EvidenceGapRegistryError
        Repeated JSON keys, nonfinite values, unknown statuses, invalid fields,
        nonpositive/noninteger issue IDs, or duplicate tracker declarations.
    OSError, UnicodeError, json.JSONDecodeError
        Registry cannot be read or decoded. CLI callers map native errors to a
        fixed sentence, without interpolating path or interpreter details.

    Notes
    -----
    Only fields consumed by planning are validated. Header, source existence,
    evidence authenticity and physical admission belong to the full validator.
    Missing or unresolved positive tracker links remain useful diagnostics.
    """
    with path.open(encoding="utf-8") as handle:
        payload: object = json.load(
            handle, object_pairs_hook=_unique_object, parse_constant=_nonfinite, parse_float=_finite_float
        )
    if not isinstance(payload, dict):
        raise EvidenceGapRegistryError("Evidence gap registry must contain a JSON object")
    data = cast(dict[str, object], payload)
    trackers: list[ExternalValidationTracker] = []
    issues: set[int] = set()
    for raw in _objects(data, "external_validation_trackers"):
        issue = _positive_issue(raw.get("issue"), "issue")
        if issue in issues:
            raise EvidenceGapRegistryError("Tracker issue numbers must be unique")
        issues.add(issue)
        trackers.append(
            ExternalValidationTracker(
                issue=issue, title=_string(raw, "title"), url=_string(raw, "url"), scope=_string(raw, "scope")
            )
        )
    entries: list[TraceabilityEntry] = []
    for raw in _objects(data, "entries"):
        status = _string(raw, "fidelity_status")
        if status not in _STATUS_SEVERITY:
            raise EvidenceGapRegistryError("fidelity_status must use the maintained vocabulary")
        allowed = raw.get("public_claim_allowed")
        if not isinstance(allowed, bool):
            raise EvidenceGapRegistryError("public_claim_allowed must be a boolean")
        requirements = raw.get("claim_admission_requirements")
        if not isinstance(requirements, list):
            raise EvidenceGapRegistryError("claim_admission_requirements must be a string list")
        actions: list[str] = []
        for item in requirements:
            if not isinstance(item, str) or not item.strip():
                raise EvidenceGapRegistryError("claim_admission_requirements must contain non-empty strings")
            actions.append(item)
        issue_value = raw.get("external_validation_tracker_issue")
        entries.append(
            TraceabilityEntry(
                component=_string(raw, "component"),
                module_path=_string(raw, "module_path"),
                fidelity_status=status,
                public_claim_allowed=allowed,
                claim_admission_requirements=tuple(actions),
                external_validation_tracker_issue=None
                if issue_value is None
                else _positive_issue(issue_value, "external_validation_tracker_issue"),
            )
        )
    return tuple(entries), tuple(sorted(trackers, key=lambda tracker: tracker.issue))


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Construct one JSON object only when each member has a distinct name."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceGapRegistryError("Evidence gap registry JSON member names must be unique")
        result[key] = value
    return result


def _nonfinite(value: str) -> object:
    """Refuse JSON decoder extensions for NaN and signed infinity."""
    raise EvidenceGapRegistryError("Evidence gap registry must not contain nonfinite JSON numbers")


def _finite_float(value: str) -> float:
    """Read a JSON decimal only when its decoded value remains finite."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise EvidenceGapRegistryError("Evidence gap registry must not contain nonfinite JSON numbers")
    return parsed


def _objects(data: dict[str, object], field: str) -> list[dict[str, object]]:
    """Read an array containing only declared JSON objects."""
    value = data.get(field)
    if not isinstance(value, list):
        raise EvidenceGapRegistryError(f"{field} must be a list")
    result: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            raise EvidenceGapRegistryError(f"{field} must contain objects")
        result.append(cast(dict[str, object], item))
    return result


def _string(data: dict[str, object], field: str) -> str:
    """Read a required nonblank string without changing declaration bytes."""
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise EvidenceGapRegistryError(f"{field} must be a non-empty string")
    return value


def _positive_issue(value: object, field: str) -> int:
    """Read an exact positive built-in integer, excluding boolean issue IDs."""
    if type(value) is not int or value <= 0:
        raise EvidenceGapRegistryError(f"{field} must be a positive integer")
    return value

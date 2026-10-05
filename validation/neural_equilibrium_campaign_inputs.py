# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Campaign metadata and local storage custody
"""Validate campaign declarations and inspect selected local dataset bytes.

These checks admit planning metadata, not predictive or facility evidence.
Absent storage can carry an explicit operator attestation; this does not hash
remote bytes. Present local files are streamed and must match the declared SHA.
Unknown dataset metadata is retained after finite, duplicate-free JSON decoding.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PureWindowsPath
from typing import Any

from validation.neural_equilibrium_dataset_contracts import DATASET_SCHEMA
from validation.validate_public_data_acquisition import validate_public_data_acquisition_directory


class CampaignPlanError(ValueError):
    """An invalid declaration, selected byte binding or report cannot prepare a plan."""


def canonical_campaign_digest(payload: dict[str, Any]) -> str:
    """Hash finite canonical JSON, binding ``payload_sha256`` to literal null.

    This consistency digest is neither a signature nor dataset verification.

    >>> canonical_campaign_digest({}) == canonical_campaign_digest({'payload_sha256': 'ignored'})
    True
    """
    try:
        encoded = json.dumps(
            {**payload, "payload_sha256": None},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise CampaignPlanError(f"campaign payload must be finite JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Refuse duplicate keys at every decoded object depth."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CampaignPlanError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_float(text: str) -> float:
    """Refuse exponent overflow before it becomes a non-finite float."""
    value = float(text)
    if not math.isfinite(value):
        raise CampaignPlanError("dataset report must contain finite JSON numbers")
    return value


def _reject_constant(text: str) -> Any:
    """Refuse JSON decoder extensions such as NaN and Infinity."""
    raise CampaignPlanError(f"dataset report must contain finite JSON numbers: {text}")


def _text(value: Any, field: str) -> str:
    """Require a trimmed nonempty string without ASCII controls."""
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise CampaignPlanError(f"{field} must be a nonempty trimmed string without controls")
    return value


def _integer(value: Any, field: str, *, minimum: int = 1) -> int:
    """Require a genuine integer count, excluding booleans and coercion."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CampaignPlanError(f"{field} must be an integer >= {minimum}")
    return value


def _relative_path(value: Any, field: str) -> str:
    """Require a safe storage-relative POSIX spelling across host platforms."""
    text = _text(value, field)
    path = Path(text)
    win = PureWindowsPath(text)
    if (
        path.is_absolute()
        or win.drive
        or win.root
        or "\\" in text
        or any(p in {"", ".", ".."} for p in text.split("/"))
    ):
        raise CampaignPlanError(f"{field} must be a safe storage-relative path")
    return text


def _strings(value: Any, field: str, *, allow_empty: bool = False) -> list[str]:
    """Require distinct declared string members, permitting an explicit empty fallback list."""
    if not isinstance(value, list) or (not value and not allow_empty):
        raise CampaignPlanError(f"{field} must be an array of strings")
    items = [_text(v, field) for v in value]
    if len(set(items)) != len(items):
        raise CampaignPlanError(f"{field} must contain distinct strings")
    return items


def read_campaign_dataset_report(path: Path) -> dict[str, Any]:
    """Read finite UTF-8 metadata and validate every dataset field consumed by the planner.

    Counts, grid and split totals are checked as declarations. Optional producer
    self-digests must match using the producer's null-field convention. Paths
    remain storage-relative; dataset tensors, split membership and target units
    are not inspected here. Supported read/decode failures become CampaignPlanError.
    """
    try:
        report = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_float=_finite_float,
            parse_constant=_reject_constant,
        )
    except (OSError, ValueError, RecursionError, RuntimeError) as exc:
        raise CampaignPlanError(f"cannot read dataset report {path}: {exc}") from exc
    return validate_campaign_dataset_metadata(report)


def validate_campaign_dataset_metadata(report: dict[str, Any]) -> dict[str, Any]:
    """Validate finite in-memory dataset declarations using the same contract as the public file reader.

    Returns the supplied mapping on success. This checks metadata and optional
    null-field self-digest, not local tensors or authenticated source authority.
    """
    if not isinstance(report, dict):
        raise CampaignPlanError("dataset report must contain a JSON object")
    canonical_campaign_digest(report)
    if report.get("schema_version") != DATASET_SCHEMA:
        raise CampaignPlanError("MAST EFM dataset report has unsupported schema_version")
    if report.get("status") != "blocked":
        raise CampaignPlanError("MAST EFM dataset report must preserve blocked predictive-admission state")
    if "payload_sha256" in report and report["payload_sha256"] != canonical_campaign_digest(report):
        raise CampaignPlanError("dataset report payload_sha256 does not match its contents")
    count = _integer(report.get("equilibria_count"), "equilibria_count")
    sha = report.get("dataset_sha256")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise CampaignPlanError("dataset_sha256 must be a lowercase SHA-256 hex digest")
    _text(report.get("reference_dataset_id"), "reference_dataset_id")
    for field in ("dataset_path", "candidate_report"):
        _relative_path(report.get(field), field)
    grid = report.get("grid_shape")
    if not isinstance(grid, list) or len(grid) != 2:
        raise CampaignPlanError("grid_shape must contain two positive integer dimensions")
    for dim in grid:
        _integer(dim, "grid_shape")
    splits = report.get("split_counts")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise CampaignPlanError("split_counts must declare train, validation and test")
    if sum(_integer(v, "split_counts", minimum=0) for v in splits.values()) != count:
        raise CampaignPlanError("split_counts must sum to equilibria_count")
    _strings(report.get("fallback_features"), "fallback_features", allow_empty=True)
    ragged = report.get("ragged_target_policy")
    if not isinstance(ragged, dict):
        raise CampaignPlanError("ragged_target_policy must be an object")
    _strings(ragged.get("keys"), "ragged_target_policy.keys")
    _text(ragged.get("padding"), "ragged_target_policy.padding")
    _text(ragged.get("point_count_key"), "ragged_target_policy.point_count_key")
    _integer(ragged.get("max_lcfs_points"), "ragged_target_policy.max_lcfs_points")
    return report


def inspect_storage_payload(
    report: dict[str, Any], storage_root: Path, require_payload: bool, verified_payload: bool
) -> dict[str, Any]:
    """Contain and hash a selected local file or record an explicit remote attestation.

    ``report`` is the validated output of read_campaign_dataset_report. Present
    non-files, escaped symlinks and wrong SHA refuse even with an attestation.
    The declared candidate-report command path must also remain within storage.
    Missing required storage without an attestation raises FileNotFoundError.
    Reads are streamed; filesystem lookup/read is not an atomic snapshot.
    """
    relative = _relative_path(report.get("dataset_path"), "dataset_path")
    try:
        root = storage_root.resolve()
        absolute = root / relative
        for field in ("dataset_path", "candidate_report"):
            (root / report[field]).resolve().relative_to(root)
        exists = absolute.exists() or absolute.is_symlink()
        if exists:
            if not absolute.is_file():
                raise CampaignPlanError("selected storage payload must be a regular file")
            digest = hashlib.sha256()
            with absolute.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != report["dataset_sha256"]:
                raise CampaignPlanError("selected storage payload SHA-256 does not match dataset_sha256")
    except (OSError, ValueError, RuntimeError) as exc:
        raise CampaignPlanError(f"cannot verify storage payload: {exc}") from exc
    available = exists or verified_payload
    if require_payload and not available:
        raise FileNotFoundError(f"storage-host dataset payload is missing: {absolute}")
    return {
        "relative_path": relative,
        "absolute_path": str(absolute),
        "exists_on_this_host": exists,
        "verified_available": available,
        "sha256": report["dataset_sha256"],
        "sha256_verified_on_this_host": exists,
        "remote_operator_attestation": verified_payload,
        "availability_basis": "local_sha256"
        if exists
        else "remote_operator_attestation"
        if verified_payload
        else "unobserved",
    }


def summarise_campaign_public_data(root: Path, repository: Path) -> dict[str, Any]:
    """Require acquisition PASS and retain its counters and manifests for planning.

    Aggregate FAIL counters are partial diagnostics and cannot prepare a plan.
    Canonical in-repository manifest paths become repository-relative strings.
    Deferred byte counts remain declarations; remote data are not fetched.
    """
    report = validate_public_data_acquisition_directory(root)
    if report["status"] != "pass":
        raise CampaignPlanError("public-data acquisition failed: " + json.dumps(report["errors"], sort_keys=True))
    manifests = []
    for manifest in report["manifests"]:
        item = dict(manifest)
        try:
            item["path"] = str(Path(item["path"]).relative_to(repository))
        except ValueError:
            pass
        manifests.append(item)
    return {**report, "manifests": manifests}

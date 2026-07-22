#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — FAIR-MAST campaign lineage and licence reconciliation
"""Reconcile a legacy MAST campaign without promoting its proxy labels.

The gate pins every campaign input by file SHA-256, verifies the legacy source
and labelled-dataset artefacts, and reconstructs the exact source/replay/dataset
shot-set relationships. Legacy licence declarations are replaced only in an
in-memory source-manifest migration and in this report; source files remain
immutable. Missing producer-time parent and transform digests keep reuse,
training, cohort, scientific, facility, and control claims blocked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast

import numpy as np

from scpn_control.core.real_data_manifest import RealDataManifestError, validate_real_data_manifest
from validation.build_mast_disruption_dataset import MEASURED_CHANNELS
from validation.evaluate_mast_disruption import REPORT_SCHEMA as EVALUATION_SCHEMA
from validation.fair_mast_source_policy import FAIR_MAST_LICENCE, fair_mast_provenance
from validation.mast_source_object_manifest import (
    LEGACY_MATERIAL_MANIFEST_SCHEMA,
    SOURCE_OBJECT_MANIFEST_SCHEMA,
    SourceObjectManifestError,
    canonical_json_sha256,
)
from validation.migrate_mast_source_object_manifest import migrate_material_manifest_v1

RECONCILIATION_SPEC_SCHEMA = "scpn-control.mast-campaign-reconciliation-spec.v1.0.0"
RECONCILIATION_REPORT_SCHEMA = "scpn-control.mast-campaign-lineage-reconciliation.v1.0.0"
REPLAY_SCHEMA = "scpn-control.mast-disruption-replay-channels.v1"
LEGACY_DATASET_SCHEMA = "scpn-control.mast-disruption-supervised-dataset.v1"

INPUT_NAMES = (
    "channels_archive",
    "dataset_manifest",
    "dataset_report",
    "evaluation_report",
    "material_manifest",
    "replay_report",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHOT_FILE_RE = re.compile(r"^shot_([1-9][0-9]*)\.npz$")
_CLAIM_FIELDS = (
    "cohort_admission",
    "control_admission",
    "facility_prediction",
    "reuse_admissible",
    "scientific_validation",
    "training_admission",
)


class CampaignReconciliationError(ValueError):
    """Raised when campaign evidence is malformed or not the pinned object."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise CampaignReconciliationError(f"duplicate JSON key {key!r}")
        payload[key] = value
    return payload


def _read_bytes(path: Path) -> tuple[bytes, str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CampaignReconciliationError(f"cannot read pinned input {path}: {exc}") from exc
    return raw, hashlib.sha256(raw).hexdigest()


def _parse_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError, CampaignReconciliationError) as exc:
        raise CampaignReconciliationError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CampaignReconciliationError(f"{label} root must be an object")
    return cast(dict[str, Any], payload)


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    raw, digest = _read_bytes(path)
    return _parse_json(raw, label=label), digest


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CampaignReconciliationError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _non_empty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CampaignReconciliationError(f"{field} must be a non-empty string")
    return value


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CampaignReconciliationError(f"{field} must be an integer >= {minimum}")
    return value


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignReconciliationError(f"{field} must be an object")
    return cast(Mapping[str, Any], value)


def _list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise CampaignReconciliationError(f"{field} must be an array")
    return value


def _verify_self_digest(payload: Mapping[str, Any], *, field: str) -> str:
    claimed = _sha256(payload.get("payload_sha256"), field=f"{field}.payload_sha256")
    unsigned = dict(payload)
    unsigned["payload_sha256"] = None
    if canonical_json_sha256(unsigned) != claimed:
        raise CampaignReconciliationError(f"{field}.payload_sha256 does not match the payload")
    return claimed


def _self_digest_valid(payload: Mapping[str, Any]) -> bool:
    claimed = payload.get("payload_sha256")
    if not isinstance(claimed, str) or _SHA256_RE.fullmatch(claimed) is None:
        return False
    unsigned = dict(payload)
    unsigned["payload_sha256"] = None
    return canonical_json_sha256(unsigned) == claimed


def _resolve_beneath(root: Path, relative: str, *, field: str) -> Path:
    candidate = Path(relative)
    posix = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if (
        not relative
        or candidate.is_absolute()
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in posix.parts)
        or "\\" in relative
    ):
        raise CampaignReconciliationError(f"{field} must be a confined POSIX relative path")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise CampaignReconciliationError(f"{field} escapes the campaign root") from exc
    if not resolved.is_file():
        raise CampaignReconciliationError(f"{field} does not resolve to a file")
    return resolved


def _require_output_outside_campaign(campaign_root: Path, output_path: Path) -> None:
    """Reject a report destination that could mutate campaign evidence."""
    resolved_root = campaign_root.resolve()
    resolved_output = output_path.resolve()
    try:
        resolved_output.relative_to(resolved_root)
    except ValueError:
        return
    raise CampaignReconciliationError("json_out must be outside campaign_root")


def _load_spec(path: Path) -> tuple[str, dict[str, tuple[str, str]], str, str]:
    payload, file_digest = _read_json(path, label="reconciliation spec")
    if set(payload) != {"schema_version", "dataset_id", "inputs", "payload_sha256"}:
        raise CampaignReconciliationError("reconciliation spec fields do not match the schema")
    if payload.get("schema_version") != RECONCILIATION_SPEC_SCHEMA:
        raise CampaignReconciliationError("unsupported reconciliation spec schema")
    dataset_id = _non_empty_string(payload.get("dataset_id"), field="dataset_id")
    digest = _verify_self_digest(payload, field="reconciliation spec")
    inputs = _mapping(payload.get("inputs"), field="inputs")
    if set(inputs) != set(INPUT_NAMES):
        raise CampaignReconciliationError("spec inputs must name the exact reconciliation surfaces")
    parsed: dict[str, tuple[str, str]] = {}
    seen_paths: set[str] = set()
    for name in INPUT_NAMES:
        binding = _mapping(inputs[name], field=f"inputs.{name}")
        if set(binding) != {"path", "file_sha256"}:
            raise CampaignReconciliationError(f"inputs.{name} fields do not match the schema")
        relative = _non_empty_string(binding.get("path"), field=f"inputs.{name}.path")
        if relative in seen_paths:
            raise CampaignReconciliationError("spec input paths must be unique")
        seen_paths.add(relative)
        parsed[name] = (relative, _sha256(binding.get("file_sha256"), field=f"inputs.{name}.file_sha256"))
    return dataset_id, parsed, digest, file_digest


def _load_pinned_inputs(
    root: Path,
    bindings: Mapping[str, tuple[str, str]],
) -> tuple[dict[str, Path], dict[str, bytes], dict[str, str]]:
    paths: dict[str, Path] = {}
    snapshots: dict[str, bytes] = {}
    digests: dict[str, str] = {}
    for name in INPUT_NAMES:
        relative, expected = bindings[name]
        path = _resolve_beneath(root, relative, field=f"inputs.{name}.path")
        raw, observed = _read_bytes(path)
        if observed != expected:
            raise CampaignReconciliationError(f"inputs.{name}.file_sha256 does not match the pinned input")
        paths[name] = path
        snapshots[name] = raw
        digests[name] = observed
    return paths, snapshots, digests


def _material_inventory(payload: Mapping[str, Any]) -> tuple[set[int], set[int], dict[int, str], int]:
    if payload.get("schema_version") != LEGACY_MATERIAL_MANIFEST_SCHEMA:
        raise CampaignReconciliationError("unsupported legacy material schema")
    _verify_self_digest(payload, field="material manifest")
    shots = _list(payload.get("shots"), field="material manifest.shots")
    acquired: set[int] = set()
    failed: set[int] = set()
    failure_reasons: dict[int, str] = {}
    total_bytes = 0
    for index, value in enumerate(shots):
        shot = _mapping(value, field=f"material manifest.shots[{index}]")
        shot_id = _integer(shot.get("shot_id"), field=f"material manifest.shots[{index}].shot_id", minimum=1)
        if shot_id in acquired or shot_id in failed:
            raise CampaignReconciliationError(f"duplicate material shot_id {shot_id}")
        status = shot.get("status")
        if status == "failed":
            failed.add(shot_id)
            failure_reasons[shot_id] = _non_empty_string(
                shot.get("error"), field=f"material manifest.shots[{index}].error"
            )
        elif status == "acquired":
            acquired.add(shot_id)
            _sha256(shot.get("checksum_sha256"), field=f"material manifest.shots[{index}].checksum_sha256")
            total_bytes += _integer(shot.get("bytes"), field=f"material manifest.shots[{index}].bytes", minimum=0)
            _non_empty_string(shot.get("npz"), field=f"material manifest.shots[{index}].npz")
        else:
            raise CampaignReconciliationError(f"material manifest.shots[{index}].status is unsupported")
    if _integer(payload.get("n_requested"), field="material manifest.n_requested") != len(shots):
        raise CampaignReconciliationError("material manifest.n_requested does not match shots")
    if _integer(payload.get("n_acquired"), field="material manifest.n_acquired") != len(acquired):
        raise CampaignReconciliationError("material manifest.n_acquired does not match acquired shots")
    if _integer(payload.get("total_bytes"), field="material manifest.total_bytes") != total_bytes:
        raise CampaignReconciliationError("material manifest.total_bytes does not match acquired shots")
    return acquired, failed, failure_reasons, total_bytes


def _replay_inventory(payload: Mapping[str, Any]) -> tuple[set[int], set[int], dict[int, str], str]:
    if payload.get("schema_version") != REPLAY_SCHEMA:
        raise CampaignReconciliationError("unsupported replay report schema")
    digest = _verify_self_digest(payload, field="replay report")
    if payload.get("synthetic") is not False:
        raise CampaignReconciliationError("replay report must declare synthetic:false")
    if tuple(payload.get("channel_schema", ())) != MEASURED_CHANNELS:
        raise CampaignReconciliationError("replay report channel schema does not match the eleven-channel contract")
    records = _list(payload.get("shots"), field="replay report.shots")
    derived: set[int] = set()
    failed: set[int] = set()
    reasons: dict[int, str] = {}
    for index, value in enumerate(records):
        record = _mapping(value, field=f"replay report.shots[{index}]")
        shot_id = _integer(record.get("shot_id"), field=f"replay report.shots[{index}].shot_id", minimum=1)
        if shot_id in derived or shot_id in failed:
            raise CampaignReconciliationError(f"duplicate replay shot_id {shot_id}")
        if record.get("status") == "derived":
            derived.add(shot_id)
            _integer(record.get("n_samples"), field=f"replay report.shots[{index}].n_samples", minimum=1)
        elif record.get("status") == "failed":
            failed.add(shot_id)
            reasons[shot_id] = _non_empty_string(record.get("error"), field=f"replay report.shots[{index}].error")
        else:
            raise CampaignReconciliationError(f"replay report.shots[{index}].status is unsupported")
    if _integer(payload.get("n_derived"), field="replay report.n_derived") != len(derived):
        raise CampaignReconciliationError("replay report.n_derived does not match derived shots")
    return derived, failed, reasons, digest


def _channels_inventory(raw: bytes) -> tuple[set[int], int, int]:
    try:
        with np.load(BytesIO(raw), allow_pickle=False) as archive:
            members = archive.files
            if len(members) != len(set(members)) or "shot_ids" not in members:
                raise CampaignReconciliationError("channels archive must contain unique members and shot_ids")
            identifiers = np.asarray(archive["shot_ids"])
            if identifiers.ndim != 1 or identifiers.size == 0 or not np.issubdtype(identifiers.dtype, np.integer):
                raise CampaignReconciliationError("channels archive shot_ids must be a non-empty integer vector")
            shot_ids = {int(value) for value in identifiers.tolist()}
            if len(shot_ids) != identifiers.size or any(shot_id <= 0 for shot_id in shot_ids):
                raise CampaignReconciliationError("channels archive shot_ids must be unique positive integers")
            expected = {"shot_ids"} | {f"{shot_id}:{channel}" for shot_id in shot_ids for channel in MEASURED_CHANNELS}
            if set(members) != expected:
                raise CampaignReconciliationError("channels archive members do not match its shot/channel inventory")
            total_samples = 0
            for shot_id in sorted(shot_ids):
                sample_count: int | None = None
                for channel in MEASURED_CHANNELS:
                    values = np.asarray(archive[f"{shot_id}:{channel}"])
                    if values.ndim != 1 or values.size == 0 or not np.issubdtype(values.dtype, np.floating):
                        raise CampaignReconciliationError(
                            f"channels archive {shot_id}:{channel} must be a non-empty floating-point vector"
                        )
                    if not np.isfinite(values).all():
                        raise CampaignReconciliationError(
                            f"channels archive {shot_id}:{channel} contains non-finite values"
                        )
                    if sample_count is None:
                        sample_count = values.size
                    elif values.size != sample_count:
                        raise CampaignReconciliationError(f"channels archive shot {shot_id} channel lengths differ")
                total_samples += cast(int, sample_count)
    except CampaignReconciliationError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise CampaignReconciliationError(f"cannot validate channels archive: {exc}") from exc
    return shot_ids, len(members) - 1, total_samples


def _dataset_records(payload: Mapping[str, Any]) -> tuple[dict[int, tuple[str, str]], str, str, int]:
    if payload.get("schema_version") != LEGACY_DATASET_SCHEMA:
        raise CampaignReconciliationError("unsupported legacy dataset report schema")
    digest = _verify_self_digest(payload, field="dataset report")
    if payload.get("status") != "blocked" or payload.get("synthetic") is not False:
        raise CampaignReconciliationError("dataset report must remain blocked and non-synthetic")
    records = _list(payload.get("shots"), field="dataset report.shots")
    parsed: dict[int, tuple[str, str]] = {}
    disruptive = 0
    checksums: list[str] = []
    for index, value in enumerate(records):
        record = _mapping(value, field=f"dataset report.shots[{index}]")
        shot_id = _integer(record.get("shot_id"), field=f"dataset report.shots[{index}].shot_id", minimum=1)
        if shot_id in parsed:
            raise CampaignReconciliationError(f"duplicate dataset shot_id {shot_id}")
        relative = _non_empty_string(record.get("npz"), field=f"dataset report.shots[{index}].npz")
        match = _SHOT_FILE_RE.fullmatch(relative)
        if match is None or int(match.group(1)) != shot_id:
            raise CampaignReconciliationError(f"dataset report shot {shot_id} has a mismatched NPZ filename")
        checksum = _sha256(record.get("checksum_sha256"), field=f"dataset report.shots[{index}].checksum_sha256")
        label = _integer(record.get("label"), field=f"dataset report.shots[{index}].label")
        if label not in {0, 1}:
            raise CampaignReconciliationError(f"dataset report.shots[{index}].label must be zero or one")
        disruptive += label
        checksums.append(checksum)
        parsed[shot_id] = (relative, checksum)
    if _integer(payload.get("n_shots"), field="dataset report.n_shots") != len(parsed):
        raise CampaignReconciliationError("dataset report.n_shots does not match shots")
    if _integer(payload.get("n_disruptive"), field="dataset report.n_disruptive") != disruptive:
        raise CampaignReconciliationError("dataset report.n_disruptive does not match labels")
    fingerprint = hashlib.sha256("".join(sorted(checksums)).encode("utf-8")).hexdigest()
    if payload.get("dataset_sha256") != fingerprint:
        raise CampaignReconciliationError("dataset report.dataset_sha256 does not match shot checksums")
    return parsed, digest, fingerprint, disruptive


def _manifest_records(payload: dict[str, Any], *, dataset_id: str) -> tuple[dict[int, tuple[str, str]], str | None]:
    try:
        manifest = validate_real_data_manifest(payload)
    except RealDataManifestError as exc:
        raise CampaignReconciliationError(f"dataset manifest is invalid: {exc}") from exc
    if manifest.dataset_id != dataset_id or manifest.machine != "MAST" or manifest.synthetic:
        raise CampaignReconciliationError("dataset manifest identity must be the non-synthetic MAST campaign")
    parsed: dict[int, tuple[str, str]] = {}
    for artifact in manifest.artifacts:
        match = _SHOT_FILE_RE.fullmatch(artifact.uri)
        if match is None:
            raise CampaignReconciliationError("dataset manifest artifacts must use shot_<id>.npz filenames")
        shot_id = int(match.group(1))
        if shot_id in parsed:
            raise CampaignReconciliationError(f"duplicate dataset-manifest shot_id {shot_id}")
        parsed[shot_id] = (artifact.uri, artifact.checksum_sha256)
    if not parsed:
        raise CampaignReconciliationError("dataset manifest must enumerate dataset artefacts")
    return parsed, manifest.licence


def _verify_dataset_files(manifest_path: Path, records: Mapping[int, tuple[str, str]]) -> int:
    total_bytes = 0
    for shot_id in sorted(records):
        relative, expected = records[shot_id]
        path = _resolve_beneath(manifest_path.parent, relative, field=f"dataset artifact {shot_id}")
        raw, observed = _read_bytes(path)
        if observed != expected:
            raise CampaignReconciliationError(f"dataset artifact {shot_id} checksum mismatch")
        total_bytes += len(raw)
    return total_bytes


def _evaluation_inventory(
    payload: Mapping[str, Any],
    *,
    dataset_id: str,
) -> tuple[set[int], bool, str | None]:
    if payload.get("schema_version") != EVALUATION_SCHEMA:
        raise CampaignReconciliationError("unsupported evaluation report schema")
    if payload.get("dataset_id") != dataset_id or payload.get("status") != "blocked":
        raise CampaignReconciliationError("evaluation report identity/status mismatch")
    if payload.get("admission_ready") is not False:
        raise CampaignReconciliationError("evaluation report must remain admission_ready:false")
    boundary = _mapping(payload.get("claim_boundary"), field="evaluation report.claim_boundary")
    if boundary.get("public_claim_allowed") is not False or boundary.get("facility_roc_validated") is not False:
        raise CampaignReconciliationError("evaluation report must keep public and facility claims false")
    shots: set[int] = set()
    for index, value in enumerate(_list(payload.get("shots"), field="evaluation report.shots")):
        record = _mapping(value, field=f"evaluation report.shots[{index}]")
        identifier = _non_empty_string(record.get("shot_id"), field=f"evaluation report.shots[{index}].shot_id")
        match = _SHOT_FILE_RE.fullmatch(f"{identifier}.npz")
        if match is None:
            raise CampaignReconciliationError(f"evaluation report.shots[{index}].shot_id is malformed")
        shot_id = int(match.group(1))
        if shot_id in shots:
            raise CampaignReconciliationError(f"duplicate evaluation shot_id {shot_id}")
        shots.add(shot_id)
    provenance = _mapping(payload.get("data_provenance"), field="evaluation report.data_provenance")
    licence = provenance.get("licence")
    return shots, _self_digest_valid(payload), licence if isinstance(licence, str) else None


def reconcile_campaign(
    *,
    campaign_root: Path,
    spec_path: Path,
    generated_at: str,
) -> dict[str, Any]:
    """Build a deterministic, fail-closed campaign reconciliation report."""
    if not generated_at:
        raise CampaignReconciliationError("generated_at must be non-empty")
    if not campaign_root.is_dir():
        raise CampaignReconciliationError("campaign_root must be an existing directory")
    dataset_id, bindings, spec_payload_digest, spec_file_digest = _load_spec(spec_path)
    paths, snapshots, input_digests = _load_pinned_inputs(campaign_root, bindings)
    json_inputs = {name: _parse_json(snapshots[name], label=name) for name in INPUT_NAMES if name != "channels_archive"}

    material = json_inputs["material_manifest"]
    acquired, acquisition_failed, acquisition_failure_reasons, material_bytes = _material_inventory(material)
    try:
        migrated = migrate_material_manifest_v1(material, artifact_root=campaign_root)
    except (OSError, SourceObjectManifestError) as exc:
        raise CampaignReconciliationError(f"legacy material migration failed: {exc}") from exc
    if migrated.get("schema_version") != SOURCE_OBJECT_MANIFEST_SCHEMA:
        raise CampaignReconciliationError("legacy migration did not produce a source-object manifest v2")

    replay = json_inputs["replay_report"]
    replay_derived, replay_failed, replay_failure_reasons, replay_digest = _replay_inventory(replay)
    if replay.get("channels_npz") != Path(bindings["channels_archive"][0]).name:
        raise CampaignReconciliationError("replay report does not name the pinned channels archive")
    channel_shots, channel_members, channel_samples = _channels_inventory(snapshots["channels_archive"])
    if channel_shots != replay_derived:
        raise CampaignReconciliationError("channels archive shot inventory differs from the replay report")

    dataset_report = json_inputs["dataset_report"]
    if dataset_report.get("dataset_id") != dataset_id:
        raise CampaignReconciliationError("dataset report dataset_id mismatch")
    dataset_records, dataset_digest, dataset_fingerprint, disruptive = _dataset_records(dataset_report)
    manifest_records, dataset_licence = _manifest_records(json_inputs["dataset_manifest"], dataset_id=dataset_id)
    if dataset_records != manifest_records:
        raise CampaignReconciliationError("dataset report and manifest artefact inventories differ")
    dataset_bytes = _verify_dataset_files(paths["dataset_manifest"], manifest_records)

    evaluation_shots, evaluation_digest_valid, evaluation_licence = _evaluation_inventory(
        json_inputs["evaluation_report"], dataset_id=dataset_id
    )
    material_licence = material.get("licence") if isinstance(material.get("licence"), str) else None
    authoritative_policy = {"licence_spdx": FAIR_MAST_LICENCE, **fair_mast_provenance()}

    dataset_ids = set(dataset_records)
    blockers = {
        "dataset_artifacts_lack_source_parent_and_transform_digests",
        "legacy_material_native_zarr_bytes_and_source_generation_not_preserved",
        "replay_archive_not_digest_bound_by_its_producer_report",
    }
    if material_licence != FAIR_MAST_LICENCE:
        blockers.add("legacy_material_manifest_licence_requires_in_memory_replacement")
    if dataset_licence != FAIR_MAST_LICENCE:
        blockers.add("persisted_dataset_manifest_licence_requires_regeneration")
    if evaluation_licence != FAIR_MAST_LICENCE:
        blockers.add("evaluation_report_uses_legacy_licence")
    if not evaluation_digest_valid:
        blockers.add("evaluation_report_self_digest_invalid")
    if not replay_derived.issubset(acquired):
        blockers.add("replay_contains_shots_absent_from_acquired_material")
    if not replay_failed.issubset(acquired):
        blockers.add("replay_failure_records_absent_from_acquired_material")
    if dataset_ids != replay_derived:
        blockers.add("dataset_shot_set_differs_from_replay_derived_set")
    if evaluation_shots != dataset_ids:
        blockers.add("evaluation_shot_set_differs_from_dataset")
    unexplained_exclusions = sorted((acquired - dataset_ids) - replay_failed)
    if unexplained_exclusions:
        blockers.add("acquired_shots_missing_without_digest_bound_replay_failure")

    exclusions = [
        {
            "shot_id": shot_id,
            "reason": replay_failure_reasons.get(shot_id),
            "reason_evidence": (
                "verified_self_digested_replay_report" if shot_id in replay_failure_reasons else "unverified"
            ),
        }
        for shot_id in sorted(acquired - dataset_ids)
    ]
    report: dict[str, Any] = {
        "schema_version": RECONCILIATION_REPORT_SCHEMA,
        "status": "blocked",
        "dataset_id": dataset_id,
        "blockers": sorted(blockers),
        "spec_binding": {
            "schema_version": RECONCILIATION_SPEC_SCHEMA,
            "file_sha256": spec_file_digest,
            "payload_sha256": spec_payload_digest,
        },
        "input_bindings": {
            name: {
                "path": bindings[name][0],
                "file_sha256": input_digests[name],
            }
            for name in INPUT_NAMES
        },
        "licence_reconciliation": {
            "authoritative_policy": authoritative_policy,
            "legacy_material_declared_licence": material_licence,
            "legacy_dataset_declared_licence": dataset_licence,
            "legacy_evaluation_declared_licence": evaluation_licence,
            "source_policy_action": "replaced_in_memory",
            "dataset_manifest_action": ("none" if dataset_licence == FAIR_MAST_LICENCE else "regeneration_required"),
            "source_files_mutated": False,
            "licence_reconciled_for_reuse": False,
        },
        "source_inventory": {
            "requested_count": len(acquired | acquisition_failed),
            "acquired_count": len(acquired),
            "failed_count": len(acquisition_failed),
            "material_bytes": material_bytes,
            "acquired_shot_ids": sorted(acquired),
            "failed_shot_ids": sorted(acquisition_failed),
            "failure_reasons": [
                {"shot_id": shot_id, "reason": acquisition_failure_reasons[shot_id]}
                for shot_id in sorted(acquisition_failure_reasons)
            ],
            "migrated_schema_version": SOURCE_OBJECT_MANIFEST_SCHEMA,
            "migrated_payload_sha256": migrated["payload_sha256"],
            "native_zarr_bytes_preserved": False,
            "source_generation_pinned": False,
        },
        "replay_inventory": {
            "payload_sha256": replay_digest,
            "channels_archive_file_sha256": input_digests["channels_archive"],
            "channels_archive_producer_digest_bound": False,
            "channels_archive_shot_inventory_verified": True,
            "channels_archive_member_count": channel_members,
            "channels_archive_total_samples": channel_samples,
            "derived_count": len(replay_derived),
            "failed_count": len(replay_failed),
            "derived_shot_ids": sorted(replay_derived),
            "failed_shot_ids": sorted(replay_failed),
        },
        "dataset_inventory": {
            "payload_sha256": dataset_digest,
            "artifact_count": len(dataset_ids),
            "artifact_bytes": dataset_bytes,
            "dataset_sha256": dataset_fingerprint,
            "proxy_positive_count": disruptive,
            "proxy_negative_count": len(dataset_ids) - disruptive,
            "shot_ids": sorted(dataset_ids),
            "per_shot_source_parent_digest_present": False,
            "per_shot_transform_digest_present": False,
        },
        "evaluation_inventory": {
            "payload_sha256": json_inputs["evaluation_report"].get("payload_sha256"),
            "payload_sha256_valid": evaluation_digest_valid,
            "shot_count": len(evaluation_shots),
            "shot_ids": sorted(evaluation_shots),
            "historical_only": True,
        },
        "cross_inventory": {
            "replay_subset_of_acquired": replay_derived.issubset(acquired),
            "dataset_equals_replay": dataset_ids == replay_derived,
            "evaluation_equals_dataset": evaluation_shots == dataset_ids,
            "replay_without_acquired": sorted(replay_derived - acquired),
            "replay_failures_without_acquired": sorted(replay_failed - acquired),
            "acquired_without_dataset": sorted(acquired - dataset_ids),
            "dataset_without_replay": sorted(dataset_ids - replay_derived),
            "evaluation_without_dataset": sorted(evaluation_shots - dataset_ids),
            "dataset_without_evaluation": sorted(dataset_ids - evaluation_shots),
            "exclusions": exclusions,
            "derivation_link_status": "unproven_missing_source_parent_and_transform_digests",
        },
        "claim_boundary": {field: False for field in _CLAIM_FIELDS},
        "generated_at": generated_at,
        "payload_sha256": None,
    }
    report["payload_sha256"] = canonical_json_sha256(report)
    return report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run reconciliation and write one deterministic JSON report."""
    args = _parse_args(argv)
    report = reconcile_campaign(
        campaign_root=args.campaign_root,
        spec_path=args.spec,
        generated_at=args.generated_at,
    )
    _require_output_outside_campaign(args.campaign_root, args.json_out)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source = cast(Mapping[str, Any], report["source_inventory"])
    dataset = cast(Mapping[str, Any], report["dataset_inventory"])
    print(
        "campaign reconciliation blocked: "
        f"requested={source['requested_count']} acquired={source['acquired_count']} "
        f"dataset={dataset['artifact_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

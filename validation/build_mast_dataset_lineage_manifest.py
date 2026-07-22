#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — immutable post-hoc FAIR-MAST dataset lineage manifest
"""Bind legacy MAST dataset bytes without inventing producer-time lineage.

The legacy disruption campaign preserved source, replay, and labelled-dataset
bytes, but its producers did not persist a complete derivation graph.  This
module emits a new immutable reconciliation manifest that binds every retained
shot to those verified bytes and to an explicit transform specification.  It
also records every acquired-but-excluded shot.

The result remains post-hoc evidence.  It cannot attest that the historical
producer consumed the reconstructed parent or transform, so all scientific,
cohort, training, prediction, reuse, and control admission fields remain false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast

import numpy as np

from validation.build_mast_disruption_dataset import MEASURED_CHANNELS
from validation.fair_mast_source_policy import FAIR_MAST_LICENCE, fair_mast_provenance
from validation.mast_source_object_manifest import (
    array_value_sha256,
    canonical_json_sha256,
)
from validation.migrate_mast_source_object_manifest import migrate_material_manifest_v1
from validation.reconcile_mast_campaign_lineage import (
    LEGACY_DATASET_SCHEMA,
    RECONCILIATION_REPORT_SCHEMA,
    REPLAY_SCHEMA,
    reconcile_campaign,
)

DATASET_LINEAGE_TRANSFORM_SCHEMA = "scpn-control.mast-dataset-transform-spec.v1.0.0"
DATASET_LINEAGE_MANIFEST_SCHEMA = "scpn-control.mast-dataset-lineage-manifest.v1.0.0"
LINEAGE_MODE = "post_hoc_reconciliation"
TRANSFORM_OPERATIONS = (
    "select_verified_replay_channels",
    "derive_ip_quench_proxy_label",
    "write_shot_npz",
    "sha256_bind_output",
)
CLAIM_FIELDS = (
    "cohort_admission",
    "control_admission",
    "facility_prediction",
    "reuse_admissible",
    "scientific_validation",
    "training_admission",
)
BLOCKERS = (
    "historical_dataset_not_emitted_by_lineage_aware_producer",
    "independent_outcome_authority_absent",
    "native_zarr_bytes_and_source_generation_unavailable",
    "producer_time_parent_and_transform_attestation_absent",
    "replay_archive_not_digest_bound_by_its_producer",
    "sealed_evaluation_regeneration_required",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "status",
    "lineage_mode",
    "producer_time_lineage",
    "dataset_id",
    "synthetic",
    "licence_spdx",
    "licence",
    "licence_url",
    "citation",
    "citations",
    "source_policy_url",
    "reconciliation_binding",
    "transform_spec_binding",
    "replay_archive_binding",
    "lineage_records",
    "exclusion_ledger",
    "counts",
    "blockers",
    "claim_boundary",
    "generated_at",
    "payload_sha256",
}


class DatasetLineageError(ValueError):
    """Raised when post-hoc lineage evidence is malformed or inconsistent."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise DatasetLineageError(f"duplicate JSON key {key!r}")
        payload[key] = value
    return payload


def _read_bytes(path: Path, *, label: str) -> tuple[bytes, str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DatasetLineageError(f"cannot read {label} {path}: {exc}") from exc
    return raw, hashlib.sha256(raw).hexdigest()


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    raw, digest = _read_bytes(path, label=label)
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError, DatasetLineageError) as exc:
        raise DatasetLineageError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DatasetLineageError(f"{label} root must be an object")
    return cast(dict[str, Any], payload), digest


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetLineageError(f"{field} must be an object")
    return cast(Mapping[str, Any], value)


def _list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise DatasetLineageError(f"{field} must be an array")
    return value


def _non_empty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetLineageError(f"{field} must be a non-empty string")
    return value


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DatasetLineageError(f"{field} must be a positive integer")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DatasetLineageError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _verify_self_digest(payload: Mapping[str, Any], *, field: str) -> str:
    digest = _sha256(payload.get("payload_sha256"), field=f"{field}.payload_sha256")
    unsigned = dict(payload)
    unsigned["payload_sha256"] = None
    if canonical_json_sha256(unsigned) != digest:
        raise DatasetLineageError(f"{field}.payload_sha256 does not match the payload")
    return digest


def _portable_relative_path(value: Any, *, field: str) -> str:
    relative = _non_empty_string(value, field=field)
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
        raise DatasetLineageError(f"{field} must be a confined POSIX relative path")
    return relative


def _resolve_beneath(root: Path, relative: str, *, field: str) -> Path:
    relative = _portable_relative_path(relative, field=field)
    candidate = Path(relative)
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise DatasetLineageError(f"{field} escapes the campaign root") from exc
    if not resolved.is_file():
        raise DatasetLineageError(f"{field} does not resolve to a file")
    return resolved


def _require_output_outside_campaign(campaign_root: Path, output_path: Path) -> None:
    resolved_root = campaign_root.resolve()
    resolved_output = output_path.resolve()
    try:
        resolved_output.relative_to(resolved_root)
    except ValueError:
        return
    raise DatasetLineageError("json_out must be outside campaign_root")


def _bound_path(
    report: Mapping[str, Any],
    campaign_root: Path,
    name: str,
) -> tuple[Path, str, str]:
    bindings = _mapping(report.get("input_bindings"), field="reconciliation.input_bindings")
    binding = _mapping(bindings.get(name), field=f"reconciliation.input_bindings.{name}")
    relative = _non_empty_string(binding.get("path"), field=f"reconciliation.input_bindings.{name}.path")
    expected = _sha256(
        binding.get("file_sha256"),
        field=f"reconciliation.input_bindings.{name}.file_sha256",
    )
    path = _resolve_beneath(campaign_root, relative, field=f"reconciliation input {name}")
    _, observed = _read_bytes(path, label=name)
    if observed != expected:
        raise DatasetLineageError(f"reconciliation input {name} checksum mismatch")
    return path, relative, expected


def validate_transform_spec(payload: Mapping[str, Any]) -> str:
    """Validate a bounded post-hoc transform spec and return its payload digest."""
    expected_fields = {
        "schema_version",
        "transform_id",
        "lineage_mode",
        "input_schema",
        "output_schema",
        "operation_chain",
        "label_authority",
        "producer_time_attested",
        "independent_label_authority",
        "training_admission",
        "payload_sha256",
    }
    if set(payload) != expected_fields:
        raise DatasetLineageError("transform spec fields do not match the v1 contract")
    if payload.get("schema_version") != DATASET_LINEAGE_TRANSFORM_SCHEMA:
        raise DatasetLineageError("unsupported transform spec schema")
    _non_empty_string(payload.get("transform_id"), field="transform_spec.transform_id")
    if payload.get("lineage_mode") != LINEAGE_MODE:
        raise DatasetLineageError(f"transform_spec.lineage_mode must equal {LINEAGE_MODE!r}")
    if payload.get("input_schema") != REPLAY_SCHEMA:
        raise DatasetLineageError("transform_spec.input_schema must bind the legacy replay schema")
    if payload.get("output_schema") != LEGACY_DATASET_SCHEMA:
        raise DatasetLineageError("transform_spec.output_schema must bind the legacy dataset schema")
    if tuple(_list(payload.get("operation_chain"), field="transform_spec.operation_chain")) != TRANSFORM_OPERATIONS:
        raise DatasetLineageError("transform_spec.operation_chain does not match the bounded transform")
    if payload.get("label_authority") != "ip_proxy":
        raise DatasetLineageError("transform_spec.label_authority must remain 'ip_proxy'")
    for field in ("producer_time_attested", "independent_label_authority", "training_admission"):
        if payload.get(field) is not False:
            raise DatasetLineageError(f"transform_spec.{field} must remain false")
    return _verify_self_digest(payload, field="transform_spec")


def _source_records(migrated: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    records: dict[int, Mapping[str, Any]] = {}
    for index, value in enumerate(_list(migrated.get("shots"), field="migrated_source.shots")):
        shot = _mapping(value, field=f"migrated_source.shots[{index}]")
        if shot.get("status") != "acquired":
            continue
        shot_id = _positive_integer(shot.get("shot_id"), field=f"migrated_source.shots[{index}].shot_id")
        artifacts = _list(shot.get("artifacts"), field=f"migrated_source.shots[{index}].artifacts")
        if len(artifacts) != 1:
            raise DatasetLineageError(f"migrated source shot {shot_id} must have exactly one artifact")
        if shot_id in records:
            raise DatasetLineageError(f"duplicate migrated source shot_id {shot_id}")
        records[shot_id] = _mapping(artifacts[0], field=f"migrated source shot {shot_id} artifact")
    if not records:
        raise DatasetLineageError("migrated source must contain acquired shots")
    return records


def _dataset_records(payload: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    records: dict[int, Mapping[str, Any]] = {}
    for index, value in enumerate(_list(payload.get("shots"), field="dataset_report.shots")):
        shot = _mapping(value, field=f"dataset_report.shots[{index}]")
        shot_id = _positive_integer(shot.get("shot_id"), field=f"dataset_report.shots[{index}].shot_id")
        if shot_id in records:
            raise DatasetLineageError(f"duplicate dataset shot_id {shot_id}")
        records[shot_id] = shot
    if not records:
        raise DatasetLineageError("dataset report must contain shots")
    return records


def _replay_member_digests(path: Path) -> dict[int, str]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            identifiers = np.asarray(archive["shot_ids"])
            digests: dict[int, str] = {}
            for value in identifiers:
                shot_id = int(value)
                channels = [
                    {
                        "name": channel,
                        "value_sha256": array_value_sha256(np.asarray(archive[f"{shot_id}:{channel}"])),
                    }
                    for channel in MEASURED_CHANNELS
                ]
                if shot_id in digests:
                    raise DatasetLineageError(f"duplicate replay shot_id {shot_id}")
                digests[shot_id] = canonical_json_sha256({"shot_id": shot_id, "channels": channels})
    except (OSError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, DatasetLineageError):
            raise
        raise DatasetLineageError(f"cannot derive replay member digests: {exc}") from exc
    return digests


def _source_binding(artifact: Mapping[str, Any], *, shot_id: int) -> dict[str, str]:
    return {
        "artifact_sha256": _sha256(artifact.get("sha256"), field=f"source shot {shot_id}.sha256"),
        "selected_array_parent_sha256": _sha256(
            artifact.get("parent_digest"), field=f"source shot {shot_id}.parent_digest"
        ),
        "acquisition_transform_sha256": _sha256(
            artifact.get("transform_digest"), field=f"source shot {shot_id}.transform_digest"
        ),
    }


def _validate_source_binding(value: Any, *, field: str) -> None:
    source = _mapping(value, field=field)
    expected = {
        "artifact_sha256",
        "selected_array_parent_sha256",
        "acquisition_transform_sha256",
    }
    if set(source) != expected:
        raise DatasetLineageError(f"{field} fields do not match")
    for key in expected:
        _sha256(source.get(key), field=f"{field}.{key}")


def build_dataset_lineage_manifest(
    *,
    campaign_root: Path,
    spec_path: Path,
    reconciliation_report_path: Path,
    transform_spec_path: Path,
    generated_at: str,
) -> dict[str, Any]:
    """Build a deterministic post-hoc dataset lineage manifest."""
    if not generated_at:
        raise DatasetLineageError("generated_at must be non-empty")
    if not campaign_root.is_dir():
        raise DatasetLineageError("campaign_root must be an existing directory")

    reconciliation, reconciliation_file_digest = _read_json(reconciliation_report_path, label="reconciliation report")
    if reconciliation.get("schema_version") != RECONCILIATION_REPORT_SCHEMA:
        raise DatasetLineageError("unsupported reconciliation report schema")
    reconciliation_payload_digest = _verify_self_digest(reconciliation, field="reconciliation report")
    reconciliation_generated_at = _non_empty_string(
        reconciliation.get("generated_at"), field="reconciliation.generated_at"
    )
    try:
        recomputed = reconcile_campaign(
            campaign_root=campaign_root,
            spec_path=spec_path,
            generated_at=reconciliation_generated_at,
        )
    except ValueError as exc:
        raise DatasetLineageError(f"fresh reconciliation failed: {exc}") from exc
    if recomputed != reconciliation:
        raise DatasetLineageError("reconciliation report does not equal a fresh reconstruction")

    transform_spec, transform_file_digest = _read_json(transform_spec_path, label="transform spec")
    transform_payload_digest = validate_transform_spec(transform_spec)
    dataset_id = _non_empty_string(reconciliation.get("dataset_id"), field="reconciliation.dataset_id")

    material_path, _, _ = _bound_path(reconciliation, campaign_root, "material_manifest")
    material, _ = _read_json(material_path, label="material manifest")
    try:
        migrated = migrate_material_manifest_v1(material, artifact_root=campaign_root)
    except (OSError, ValueError) as exc:
        raise DatasetLineageError(f"material migration failed: {exc}") from exc
    source_records = _source_records(migrated)

    dataset_report_path, _, _ = _bound_path(reconciliation, campaign_root, "dataset_report")
    dataset_report, _ = _read_json(dataset_report_path, label="dataset report")
    dataset_records = _dataset_records(dataset_report)
    _, dataset_manifest_relative, _ = _bound_path(reconciliation, campaign_root, "dataset_manifest")
    dataset_dir = PurePosixPath(dataset_manifest_relative).parent

    replay_path, replay_relative, replay_archive_digest = _bound_path(reconciliation, campaign_root, "channels_archive")
    replay_member_digests = _replay_member_digests(replay_path)
    replay_inventory = _mapping(reconciliation.get("replay_inventory"), field="reconciliation.replay_inventory")
    replay_report_digest = _sha256(
        replay_inventory.get("payload_sha256"), field="reconciliation.replay_inventory.payload_sha256"
    )
    if replay_inventory.get("channels_archive_producer_digest_bound") is not False:
        raise DatasetLineageError("legacy replay producer binding must remain false")

    cross = _mapping(reconciliation.get("cross_inventory"), field="reconciliation.cross_inventory")
    if (
        cross.get("replay_subset_of_acquired") is not True
        or cross.get("dataset_equals_replay") is not True
        or cross.get("evaluation_equals_dataset") is not True
    ):
        raise DatasetLineageError("lineage manifest requires exact source/replay/dataset/evaluation shot sets")
    raw_exclusions = _list(cross.get("exclusions"), field="reconciliation.cross_inventory.exclusions")
    exclusions_by_shot: dict[int, Mapping[str, Any]] = {}
    for index, value in enumerate(raw_exclusions):
        exclusion = _mapping(value, field=f"reconciliation.cross_inventory.exclusions[{index}]")
        shot_id = _positive_integer(exclusion.get("shot_id"), field=f"exclusions[{index}].shot_id")
        if exclusion.get("reason_evidence") != "verified_self_digested_replay_report":
            raise DatasetLineageError(f"exclusion {shot_id} lacks verified replay reason evidence")
        _non_empty_string(exclusion.get("reason"), field=f"exclusions[{index}].reason")
        if shot_id in exclusions_by_shot:
            raise DatasetLineageError(f"duplicate exclusion shot_id {shot_id}")
        exclusions_by_shot[shot_id] = exclusion

    dataset_ids = set(dataset_records)
    replay_ids = set(replay_member_digests)
    exclusion_ids = set(exclusions_by_shot)
    acquired_ids = set(source_records)
    if dataset_ids != replay_ids:
        raise DatasetLineageError("dataset shot set differs from replay archive")
    if dataset_ids & exclusion_ids:
        raise DatasetLineageError("included and excluded shot sets overlap")
    if dataset_ids | exclusion_ids != acquired_ids:
        raise DatasetLineageError("included and excluded shots do not partition acquired source shots")

    lineage_records: list[dict[str, Any]] = []
    for shot_id in sorted(dataset_ids):
        dataset = dataset_records[shot_id]
        npz_name = _non_empty_string(dataset.get("npz"), field=f"dataset shot {shot_id}.npz")
        expected_name = f"shot_{shot_id}.npz"
        if npz_name != expected_name:
            raise DatasetLineageError(f"dataset shot {shot_id}.npz must equal {expected_name!r}")
        lineage_records.append(
            {
                "shot_id": shot_id,
                "source_parent": _source_binding(source_records[shot_id], shot_id=shot_id),
                "replay_member": {
                    "archive_path": replay_relative,
                    "archive_sha256": replay_archive_digest,
                    "member_sha256": replay_member_digests[shot_id],
                    "producer_digest_bound": False,
                },
                "dataset_artifact": {
                    "path": (dataset_dir / npz_name).as_posix(),
                    "sha256": _sha256(
                        dataset.get("checksum_sha256"),
                        field=f"dataset shot {shot_id}.checksum_sha256",
                    ),
                },
                "transform_spec_sha256": transform_payload_digest,
                "producer_time_attested": False,
            }
        )

    exclusion_ledger = [
        {
            "shot_id": shot_id,
            "source_parent": _source_binding(source_records[shot_id], shot_id=shot_id),
            "reason": exclusions_by_shot[shot_id]["reason"],
            "reason_evidence": exclusions_by_shot[shot_id]["reason_evidence"],
            "replay_report_payload_sha256": replay_report_digest,
            "replay_member_present": False,
            "dataset_artifact_present": False,
        }
        for shot_id in sorted(exclusion_ids)
    ]

    provenance = fair_mast_provenance()
    manifest: dict[str, Any] = {
        "schema_version": DATASET_LINEAGE_MANIFEST_SCHEMA,
        "status": "blocked",
        "lineage_mode": LINEAGE_MODE,
        "producer_time_lineage": False,
        "dataset_id": dataset_id,
        "synthetic": False,
        "licence_spdx": FAIR_MAST_LICENCE,
        **provenance,
        "reconciliation_binding": {
            "file_sha256": reconciliation_file_digest,
            "payload_sha256": reconciliation_payload_digest,
        },
        "transform_spec_binding": {
            "file_sha256": transform_file_digest,
            "payload_sha256": transform_payload_digest,
        },
        "replay_archive_binding": {
            "path": replay_relative,
            "file_sha256": replay_archive_digest,
            "producer_digest_bound": False,
        },
        "lineage_records": lineage_records,
        "exclusion_ledger": exclusion_ledger,
        "counts": {
            "acquired_source_shots": len(acquired_ids),
            "lineage_records": len(lineage_records),
            "excluded_shots": len(exclusion_ledger),
        },
        "blockers": list(BLOCKERS),
        "claim_boundary": {field: False for field in CLAIM_FIELDS},
        "generated_at": generated_at,
        "payload_sha256": None,
    }
    manifest["payload_sha256"] = canonical_json_sha256(manifest)
    validate_dataset_lineage_manifest(manifest)
    return manifest


def validate_dataset_lineage_manifest(payload: Mapping[str, Any]) -> str:
    """Validate a sealed post-hoc lineage manifest and return its digest."""
    if set(payload) != _TOP_LEVEL_FIELDS:
        raise DatasetLineageError("dataset lineage manifest fields do not match the v1 contract")
    if payload.get("schema_version") != DATASET_LINEAGE_MANIFEST_SCHEMA:
        raise DatasetLineageError("unsupported dataset lineage manifest schema")
    if payload.get("status") != "blocked" or payload.get("lineage_mode") != LINEAGE_MODE:
        raise DatasetLineageError("dataset lineage manifest must remain blocked post-hoc evidence")
    if payload.get("producer_time_lineage") is not False or payload.get("synthetic") is not False:
        raise DatasetLineageError("producer_time_lineage and synthetic must remain false")
    _non_empty_string(payload.get("dataset_id"), field="dataset_id")
    _non_empty_string(payload.get("generated_at"), field="generated_at")
    provenance = fair_mast_provenance()
    if payload.get("licence_spdx") != FAIR_MAST_LICENCE:
        raise DatasetLineageError("licence_spdx does not match FAIR-MAST policy")
    for key, value in provenance.items():
        if payload.get(key) != value:
            raise DatasetLineageError(f"{key} does not match FAIR-MAST policy")

    reconciliation = _mapping(payload.get("reconciliation_binding"), field="reconciliation_binding")
    transform = _mapping(payload.get("transform_spec_binding"), field="transform_spec_binding")
    replay = _mapping(payload.get("replay_archive_binding"), field="replay_archive_binding")
    for name, binding in (("reconciliation", reconciliation), ("transform_spec", transform)):
        if set(binding) != {"file_sha256", "payload_sha256"}:
            raise DatasetLineageError(f"{name}_binding fields do not match")
        _sha256(binding.get("file_sha256"), field=f"{name}_binding.file_sha256")
        _sha256(binding.get("payload_sha256"), field=f"{name}_binding.payload_sha256")
    if set(replay) != {"path", "file_sha256", "producer_digest_bound"}:
        raise DatasetLineageError("replay_archive_binding fields do not match")
    _portable_relative_path(replay.get("path"), field="replay_archive_binding.path")
    replay_digest = _sha256(replay.get("file_sha256"), field="replay_archive_binding.file_sha256")
    if replay.get("producer_digest_bound") is not False:
        raise DatasetLineageError("replay_archive_binding.producer_digest_bound must remain false")

    records = _list(payload.get("lineage_records"), field="lineage_records")
    exclusions = _list(payload.get("exclusion_ledger"), field="exclusion_ledger")
    if not records:
        raise DatasetLineageError("lineage_records must contain at least one included shot")
    record_ids: list[int] = []
    transform_digest = _sha256(transform.get("payload_sha256"), field="transform_spec_binding.payload_sha256")
    for index, value in enumerate(records):
        record = _mapping(value, field=f"lineage_records[{index}]")
        if set(record) != {
            "shot_id",
            "source_parent",
            "replay_member",
            "dataset_artifact",
            "transform_spec_sha256",
            "producer_time_attested",
        }:
            raise DatasetLineageError(f"lineage_records[{index}] fields do not match")
        shot_id = _positive_integer(record.get("shot_id"), field=f"lineage_records[{index}].shot_id")
        record_ids.append(shot_id)
        _validate_source_binding(record.get("source_parent"), field=f"lineage_records[{index}].source_parent")
        member = _mapping(record.get("replay_member"), field=f"lineage_records[{index}].replay_member")
        if set(member) != {"archive_path", "archive_sha256", "member_sha256", "producer_digest_bound"}:
            raise DatasetLineageError(f"lineage_records[{index}].replay_member fields do not match")
        if (
            member.get("archive_path") != replay.get("path")
            or member.get("archive_sha256") != replay_digest
            or member.get("producer_digest_bound") is not False
        ):
            raise DatasetLineageError(f"lineage_records[{index}].replay_member binding mismatch")
        _sha256(member.get("member_sha256"), field=f"lineage_records[{index}].replay_member.member_sha256")
        artifact = _mapping(record.get("dataset_artifact"), field=f"lineage_records[{index}].dataset_artifact")
        if set(artifact) != {"path", "sha256"}:
            raise DatasetLineageError(f"lineage_records[{index}].dataset_artifact fields do not match")
        _portable_relative_path(artifact.get("path"), field=f"lineage_records[{index}].dataset_artifact.path")
        _sha256(artifact.get("sha256"), field=f"lineage_records[{index}].dataset_artifact.sha256")
        if record.get("transform_spec_sha256") != transform_digest:
            raise DatasetLineageError(f"lineage_records[{index}].transform_spec_sha256 mismatch")
        if record.get("producer_time_attested") is not False:
            raise DatasetLineageError(f"lineage_records[{index}].producer_time_attested must remain false")

    exclusion_ids: list[int] = []
    for index, value in enumerate(exclusions):
        exclusion = _mapping(value, field=f"exclusion_ledger[{index}]")
        if set(exclusion) != {
            "shot_id",
            "source_parent",
            "reason",
            "reason_evidence",
            "replay_report_payload_sha256",
            "replay_member_present",
            "dataset_artifact_present",
        }:
            raise DatasetLineageError(f"exclusion_ledger[{index}] fields do not match")
        shot_id = _positive_integer(exclusion.get("shot_id"), field=f"exclusion_ledger[{index}].shot_id")
        exclusion_ids.append(shot_id)
        _validate_source_binding(exclusion.get("source_parent"), field=f"exclusion_ledger[{index}].source_parent")
        _non_empty_string(exclusion.get("reason"), field=f"exclusion_ledger[{index}].reason")
        if exclusion.get("reason_evidence") != "verified_self_digested_replay_report":
            raise DatasetLineageError(f"exclusion_ledger[{index}].reason_evidence is not verified")
        _sha256(
            exclusion.get("replay_report_payload_sha256"),
            field=f"exclusion_ledger[{index}].replay_report_payload_sha256",
        )
        if (
            exclusion.get("replay_member_present") is not False
            or exclusion.get("dataset_artifact_present") is not False
        ):
            raise DatasetLineageError(f"exclusion_ledger[{index}] presence flags must remain false")
    if record_ids != sorted(set(record_ids)) or exclusion_ids != sorted(set(exclusion_ids)):
        raise DatasetLineageError("included and excluded shot identifiers must be unique and sorted")
    if set(record_ids) & set(exclusion_ids):
        raise DatasetLineageError("included and excluded shot identifiers overlap")

    counts = _mapping(payload.get("counts"), field="counts")
    expected_counts = {
        "acquired_source_shots": len(records) + len(exclusions),
        "lineage_records": len(records),
        "excluded_shots": len(exclusions),
    }
    if dict(counts) != expected_counts:
        raise DatasetLineageError("counts do not match lineage and exclusion records")
    if payload.get("blockers") != list(BLOCKERS):
        raise DatasetLineageError("blockers do not match the bounded post-hoc contract")
    claims = _mapping(payload.get("claim_boundary"), field="claim_boundary")
    if set(claims) != set(CLAIM_FIELDS) or any(value is not False for value in claims.values()):
        raise DatasetLineageError("all claim_boundary fields must be present and false")
    return _verify_self_digest(payload, field="dataset lineage manifest")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--reconciliation-report", type=Path, required=True)
    parser.add_argument("--transform-spec", type=Path, required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build and write one immutable lineage manifest outside the campaign."""
    args = _parse_args(argv)
    manifest = build_dataset_lineage_manifest(
        campaign_root=args.campaign_root,
        spec_path=args.spec,
        reconciliation_report_path=args.reconciliation_report,
        transform_spec_path=args.transform_spec,
        generated_at=args.generated_at,
    )
    _require_output_outside_campaign(args.campaign_root, args.json_out)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    counts = cast(Mapping[str, int], manifest["counts"])
    print(
        f"post-hoc dataset lineage blocked: records={counts['lineage_records']} exclusions={counts['excluded_shots']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Producer-time FAIR-MAST dataset lineage
"""Build a fresh proxy-labelled dataset with producer-time lineage bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast

import numpy as np

import validation.build_disruption_replay_channels as replay_channels
from scpn_control.core.real_data_manifest import RealDataManifestError, load_real_data_manifest
from validation.build_mast_disruption_dataset import DATASET_SCHEMA, MEASURED_CHANNELS, build_dataset
from validation.fair_mast_source_policy import FAIR_MAST_LICENCE, fair_mast_provenance
from validation.mast_source_object_manifest import (
    SOURCE_OBJECT_MANIFEST_SCHEMA,
    SourceObjectManifestError,
    canonical_json_sha256,
    file_sha256,
    require_source_object_manifest_v2,
)

PRODUCER_LINEAGE_SCHEMA = "scpn-control.mast-dataset-producer-lineage.v1.0.0"
TRANSFORM_SPEC_SCHEMA = "scpn-control.mast-dataset-producer-transform-spec.v1.0.0"
LINEAGE_MODE = "producer_time_attested"
TRANSFORM_OPERATIONS = (
    "select_verified_replay_v2_members",
    "bind_source_object_parent",
    "derive_ip_quench_proxy_label",
    "write_shot_npz",
    "sha256_bind_output",
)
CLAIM_FIELDS = (
    "cohort_admission",
    "control_admission",
    "facility_prediction",
    "independent_label_authority",
    "reuse_admissible",
    "scientific_validation",
    "training_admission",
)
BASE_BLOCKERS = (
    "independent_outcome_authority_absent",
    "ip_proxy_labels_are_input_derived_and_uncalibrated",
    "sealed_admitted_cohort_absent",
)
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
    "source_manifest_binding",
    "replay_report_binding",
    "replay_archive_binding",
    "transform_spec_binding",
    "dataset_report_binding",
    "dataset_manifest_binding",
    "lineage_records",
    "exclusion_ledger",
    "counts",
    "source_generation_pinned_for_all_records",
    "blockers",
    "claim_boundary",
    "generated_at",
    "payload_sha256",
}


class DatasetProducerLineageError(ValueError):
    """Raised when fresh dataset production cannot prove its declared lineage."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise DatasetProducerLineageError(f"duplicate JSON key {key!r}")
        payload[key] = value
    return payload


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError, DatasetProducerLineageError) as exc:
        raise DatasetProducerLineageError(f"cannot read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DatasetProducerLineageError(f"{label} root must be an object")
    return cast(dict[str, Any], payload), hashlib.sha256(raw).hexdigest()


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetProducerLineageError(f"{field} must be an object")
    return cast(Mapping[str, Any], value)


def _list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise DatasetProducerLineageError(f"{field} must be an array")
    return value


def _non_empty(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetProducerLineageError(f"{field} must be a non-empty string")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DatasetProducerLineageError(f"{field} must be a positive integer")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise DatasetProducerLineageError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _verify_self_digest(payload: Mapping[str, Any], *, field: str) -> str:
    claimed = _sha256(payload.get("payload_sha256"), field=f"{field}.payload_sha256")
    unsigned = dict(payload)
    unsigned["payload_sha256"] = None
    if canonical_json_sha256(unsigned) != claimed:
        raise DatasetProducerLineageError(f"{field}.payload_sha256 does not match the payload")
    return claimed


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def validate_transform_spec(payload: Mapping[str, Any]) -> str:
    """Validate the bounded producer-time transformation.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Parsed transform-specification object.

    Returns
    -------
    str
        Verified canonical payload SHA-256.

    Raises
    ------
    DatasetProducerLineageError
        If the schema, operation chain, self-digest, or claim boundary drifts.
    """
    expected = {
        "schema_version",
        "transform_id",
        "lineage_mode",
        "source_schema",
        "input_schema",
        "output_schema",
        "operation_chain",
        "label_authority",
        "producer_time_attested",
        "independent_label_authority",
        "training_admission",
        "payload_sha256",
    }
    if set(payload) != expected:
        raise DatasetProducerLineageError("transform spec fields do not match the contract")
    if payload.get("schema_version") != TRANSFORM_SPEC_SCHEMA:
        raise DatasetProducerLineageError("unsupported transform spec schema")
    if payload.get("lineage_mode") != LINEAGE_MODE:
        raise DatasetProducerLineageError("transform spec lineage_mode mismatch")
    if payload.get("source_schema") != SOURCE_OBJECT_MANIFEST_SCHEMA:
        raise DatasetProducerLineageError("transform spec source_schema mismatch")
    if payload.get("input_schema") != replay_channels.REPORT_SCHEMA:
        raise DatasetProducerLineageError("transform spec input_schema mismatch")
    if payload.get("output_schema") != DATASET_SCHEMA:
        raise DatasetProducerLineageError("transform spec output_schema mismatch")
    if tuple(_list(payload.get("operation_chain"), field="transform spec.operation_chain")) != TRANSFORM_OPERATIONS:
        raise DatasetProducerLineageError("transform spec operation_chain mismatch")
    if payload.get("label_authority") != "ip_proxy" or payload.get("producer_time_attested") is not True:
        raise DatasetProducerLineageError("transform spec must attest producer-time ip_proxy processing")
    if payload.get("independent_label_authority") is not False or payload.get("training_admission") is not False:
        raise DatasetProducerLineageError("transform spec claim fields must remain false")
    _non_empty(payload.get("transform_id"), field="transform spec.transform_id")
    return _verify_self_digest(payload, field="transform spec")


def _source_inventory(payload: Mapping[str, Any]) -> tuple[dict[int, Mapping[str, Any]], dict[int, str]]:
    acquired: dict[int, Mapping[str, Any]] = {}
    programme_classes: dict[int, str] = {}
    for index, value in enumerate(_list(payload.get("shots"), field="source manifest.shots")):
        shot = _mapping(value, field=f"source manifest.shots[{index}]")
        if shot.get("status") != "acquired":
            continue
        shot_id = _positive_int(shot.get("shot_id"), field=f"source manifest.shots[{index}].shot_id")
        artifacts = _list(shot.get("artifacts"), field=f"source manifest.shots[{index}].artifacts")
        if len(artifacts) != 1:
            raise DatasetProducerLineageError(f"source shot {shot_id} must have exactly one artifact")
        acquired[shot_id] = _mapping(artifacts[0], field=f"source shot {shot_id}.artifact")
        programme_classes[shot_id] = _non_empty(
            shot.get("programme_class"), field=f"source shot {shot_id}.programme_class"
        )
    if not acquired:
        raise DatasetProducerLineageError("source manifest must contain acquired shots")
    return acquired, programme_classes


def _source_binding(artifact: Mapping[str, Any], *, shot_id: int) -> dict[str, Any]:
    generation = artifact.get("source_generation")
    generation_digest = None
    if generation is not None:
        generation_digest = canonical_json_sha256(
            _mapping(generation, field=f"source shot {shot_id}.source_generation")
        )
    return {
        "artifact_sha256": _sha256(artifact.get("sha256"), field=f"source shot {shot_id}.sha256"),
        "selected_array_parent_sha256": _sha256(
            artifact.get("parent_digest"), field=f"source shot {shot_id}.parent_digest"
        ),
        "acquisition_transform_sha256": _sha256(
            artifact.get("transform_digest"), field=f"source shot {shot_id}.transform_digest"
        ),
        "source_generation_sha256": generation_digest,
    }


def _replay_inventory(
    report: Mapping[str, Any], archive_binding: Mapping[str, Any]
) -> tuple[list[int], dict[int, str], dict[int, str], str]:
    if report.get("schema_version") != replay_channels.REPORT_SCHEMA:
        raise DatasetProducerLineageError("replay report must use the producer-bound v2 schema")
    digest = _verify_self_digest(report, field="replay report")
    if report.get("synthetic") is not False or tuple(report.get("channel_schema", ())) != MEASURED_CHANNELS:
        raise DatasetProducerLineageError("replay report identity or channel schema mismatch")
    declared_binding = _mapping(report.get("channels_archive"), field="replay report.channels_archive")
    if dict(declared_binding) != dict(archive_binding):
        raise DatasetProducerLineageError("replay report archive binding does not match pinned bytes")
    derived: list[int] = []
    failed: dict[int, str] = {}
    seen: set[int] = set()
    for index, value in enumerate(_list(report.get("shots"), field="replay report.shots")):
        record = _mapping(value, field=f"replay report.shots[{index}]")
        shot_id = _positive_int(record.get("shot_id"), field=f"replay report.shots[{index}].shot_id")
        if shot_id in seen:
            raise DatasetProducerLineageError(f"duplicate replay shot_id {shot_id}")
        seen.add(shot_id)
        if record.get("status") == "derived":
            _positive_int(record.get("n_samples"), field=f"replay report.shots[{index}].n_samples")
            derived.append(shot_id)
        elif record.get("status") == "failed":
            failed[shot_id] = _non_empty(record.get("error"), field=f"replay report.shots[{index}].error")
        else:
            raise DatasetProducerLineageError(f"replay report.shots[{index}].status is unsupported")
    if report.get("n_derived") != len(derived):
        raise DatasetProducerLineageError("replay report.n_derived mismatch")
    members = {
        _positive_int(member.get("shot_id"), field="replay member.shot_id"): _sha256(
            member.get("sha256"), field="replay member.sha256"
        )
        for member in (
            _mapping(value, field="replay archive.shot_members")
            for value in _list(archive_binding.get("shot_members"), field="replay archive.shot_members")
        )
    }
    if derived != sorted(members) or derived != sorted(derived):
        raise DatasetProducerLineageError("replay report derived inventory differs from the archive")
    return derived, failed, members, digest


def _load_shots(raw: bytes, shot_ids: list[int], programme_classes: Mapping[int, str]) -> list[dict[str, Any]]:
    with np.load(BytesIO(raw), allow_pickle=False) as archive:
        return [
            {
                "shot_id": shot_id,
                "programme_class": programme_classes[shot_id],
                "channels": {
                    channel: np.asarray(archive[f"{shot_id}:{channel}"], dtype=np.float64)
                    for channel in MEASURED_CHANNELS
                },
            }
            for shot_id in shot_ids
        ]


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _validate_source_parent(value: Any, *, field: str) -> bool:
    parent = _mapping(value, field=field)
    expected = {
        "artifact_sha256",
        "selected_array_parent_sha256",
        "acquisition_transform_sha256",
        "source_generation_sha256",
    }
    if set(parent) != expected:
        raise DatasetProducerLineageError(f"{field} fields do not match")
    for key in expected - {"source_generation_sha256"}:
        _sha256(parent.get(key), field=f"{field}.{key}")
    generation = parent.get("source_generation_sha256")
    if generation is not None:
        _sha256(generation, field=f"{field}.source_generation_sha256")
    return generation is not None


def _validate_file_payload_binding(value: Any, *, field: str) -> None:
    binding = _mapping(value, field=field)
    if set(binding) != {"file_sha256", "payload_sha256"}:
        raise DatasetProducerLineageError(f"{field} fields do not match")
    _sha256(binding.get("file_sha256"), field=f"{field}.file_sha256")
    _sha256(binding.get("payload_sha256"), field=f"{field}.payload_sha256")


def _resolve_output(root: Path, relative: str, *, field: str) -> Path:
    posix = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if (
        not relative
        or Path(relative).is_absolute()
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in posix.parts)
        or "\\" in relative
    ):
        raise DatasetProducerLineageError(f"{field} must be a confined POSIX relative path")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise DatasetProducerLineageError(f"{field} escapes artifact_root") from exc
    if not resolved.is_file():
        raise DatasetProducerLineageError(f"{field} does not resolve to a file")
    return resolved


def validate_producer_lineage_manifest(payload: Mapping[str, Any], *, artifact_root: Path | None = None) -> str:
    """Validate a sealed producer-lineage manifest.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Parsed producer-lineage manifest.
    artifact_root : Path or None
        Optional output root used to reopen and checksum all bound dataset files.

    Returns
    -------
    str
        Verified canonical payload SHA-256.

    Raises
    ------
    DatasetProducerLineageError
        If structure, lineage, digests, files, blockers, or claims do not match.
    """
    if set(payload) != _TOP_LEVEL_FIELDS:
        raise DatasetProducerLineageError("producer lineage fields do not match the contract")
    if payload.get("schema_version") != PRODUCER_LINEAGE_SCHEMA:
        raise DatasetProducerLineageError("unsupported producer lineage schema")
    if (
        payload.get("status") != "blocked"
        or payload.get("lineage_mode") != LINEAGE_MODE
        or payload.get("producer_time_lineage") is not True
        or payload.get("synthetic") is not False
    ):
        raise DatasetProducerLineageError("producer lineage identity/status fields do not match")
    _non_empty(payload.get("dataset_id"), field="dataset_id")
    _non_empty(payload.get("generated_at"), field="generated_at")
    provenance = fair_mast_provenance()
    if payload.get("licence_spdx") != FAIR_MAST_LICENCE or any(
        payload.get(key) != value for key, value in provenance.items()
    ):
        raise DatasetProducerLineageError("producer lineage FAIR-MAST provenance mismatch")
    digest = _verify_self_digest(payload, field="producer lineage")
    for field in ("source_manifest_binding", "replay_report_binding", "transform_spec_binding"):
        _validate_file_payload_binding(payload.get(field), field=field)

    archive = _mapping(payload.get("replay_archive_binding"), field="replay_archive_binding")
    if set(archive) != {"path", "file_sha256", "bytes", "shot_count", "member_digest_kind", "shot_members"}:
        raise DatasetProducerLineageError("replay_archive_binding fields do not match")
    _non_empty(archive.get("path"), field="replay_archive_binding.path")
    _sha256(archive.get("file_sha256"), field="replay_archive_binding.file_sha256")
    if not isinstance(archive.get("bytes"), int) or isinstance(archive.get("bytes"), bool) or archive["bytes"] <= 0:
        raise DatasetProducerLineageError("replay_archive_binding.bytes must be a positive integer")
    _non_empty(archive.get("member_digest_kind"), field="replay_archive_binding.member_digest_kind")
    archive_members = _list(archive.get("shot_members"), field="replay_archive_binding.shot_members")
    if archive.get("shot_count") != len(archive_members):
        raise DatasetProducerLineageError("replay_archive_binding.shot_count mismatch")

    lineage = _list(payload.get("lineage_records"), field="lineage_records")
    exclusions = _list(payload.get("exclusion_ledger"), field="exclusion_ledger")
    lineage_ids: set[int] = set()
    generation_flags: list[bool] = []
    for index, value in enumerate(lineage):
        record = _mapping(value, field=f"lineage_records[{index}]")
        expected = {
            "shot_id",
            "source_parent",
            "replay_member",
            "transform",
            "label_record_sha256",
            "dataset_artifact",
            "producer_time_attested",
        }
        if set(record) != expected or record.get("producer_time_attested") is not True:
            raise DatasetProducerLineageError(f"lineage_records[{index}] fields or attestation do not match")
        shot_id = _positive_int(record.get("shot_id"), field=f"lineage_records[{index}].shot_id")
        if shot_id in lineage_ids:
            raise DatasetProducerLineageError(f"duplicate lineage shot_id {shot_id}")
        lineage_ids.add(shot_id)
        generation_flags.append(
            _validate_source_parent(record.get("source_parent"), field=f"lineage_records[{index}].source_parent")
        )
        replay = _mapping(record.get("replay_member"), field=f"lineage_records[{index}].replay_member")
        if (
            set(replay) != {"archive_sha256", "member_sha256", "producer_digest_bound"}
            or replay.get("producer_digest_bound") is not True
        ):
            raise DatasetProducerLineageError(f"lineage_records[{index}].replay_member is not producer-bound")
        _sha256(replay.get("archive_sha256"), field=f"lineage_records[{index}].replay_member.archive_sha256")
        _sha256(replay.get("member_sha256"), field=f"lineage_records[{index}].replay_member.member_sha256")
        _sha256(record.get("label_record_sha256"), field=f"lineage_records[{index}].label_record_sha256")
        transform = _mapping(record.get("transform"), field=f"lineage_records[{index}].transform")
        if set(transform) != {"transform_id", "payload_sha256"}:
            raise DatasetProducerLineageError(f"lineage_records[{index}].transform fields do not match")
        _non_empty(transform.get("transform_id"), field=f"lineage_records[{index}].transform.transform_id")
        _sha256(transform.get("payload_sha256"), field=f"lineage_records[{index}].transform.payload_sha256")
        artifact = _mapping(record.get("dataset_artifact"), field=f"lineage_records[{index}].dataset_artifact")
        if set(artifact) != {"path", "sha256"}:
            raise DatasetProducerLineageError(f"lineage_records[{index}].dataset_artifact fields do not match")
        _non_empty(artifact.get("path"), field=f"lineage_records[{index}].dataset_artifact.path")
        _sha256(artifact.get("sha256"), field=f"lineage_records[{index}].dataset_artifact.sha256")

    exclusion_ids: set[int] = set()
    for index, value in enumerate(exclusions):
        exclusion = _mapping(value, field=f"exclusion_ledger[{index}]")
        expected = {"shot_id", "source_parent", "reason", "replay_member_present", "dataset_artifact_present"}
        if (
            set(exclusion) != expected
            or exclusion.get("replay_member_present") is not False
            or exclusion.get("dataset_artifact_present") is not False
        ):
            raise DatasetProducerLineageError(f"exclusion_ledger[{index}] fields or presence flags do not match")
        shot_id = _positive_int(exclusion.get("shot_id"), field=f"exclusion_ledger[{index}].shot_id")
        if shot_id in exclusion_ids or shot_id in lineage_ids:
            raise DatasetProducerLineageError(f"duplicate or overlapping exclusion shot_id {shot_id}")
        exclusion_ids.add(shot_id)
        generation_flags.append(
            _validate_source_parent(exclusion.get("source_parent"), field=f"exclusion_ledger[{index}].source_parent")
        )
        _non_empty(exclusion.get("reason"), field=f"exclusion_ledger[{index}].reason")

    counts = _mapping(payload.get("counts"), field="counts")
    expected_counts = {
        "acquired_source_shots": len(lineage_ids | exclusion_ids),
        "lineage_records": len(lineage_ids),
        "excluded_shots": len(exclusion_ids),
    }
    if dict(counts) != expected_counts:
        raise DatasetProducerLineageError("producer lineage counts mismatch")
    generation_pinned = bool(generation_flags) and all(generation_flags)
    if payload.get("source_generation_pinned_for_all_records") is not generation_pinned:
        raise DatasetProducerLineageError("source-generation aggregate mismatch")
    expected_blockers = set(BASE_BLOCKERS)
    if not generation_pinned:
        expected_blockers.add("source_generation_not_pinned_for_every_shot")
    if set(_list(payload.get("blockers"), field="blockers")) != expected_blockers:
        raise DatasetProducerLineageError("producer lineage blockers mismatch")
    claims = _mapping(payload.get("claim_boundary"), field="claim_boundary")
    if set(claims) != set(CLAIM_FIELDS) or any(value is not False for value in claims.values()):
        raise DatasetProducerLineageError("producer lineage claim boundary must remain false")

    dataset_report = _mapping(payload.get("dataset_report_binding"), field="dataset_report_binding")
    if set(dataset_report) != {"path", "file_sha256", "payload_sha256"}:
        raise DatasetProducerLineageError("dataset_report_binding fields do not match")
    _non_empty(dataset_report.get("path"), field="dataset_report_binding.path")
    _sha256(dataset_report.get("file_sha256"), field="dataset_report_binding.file_sha256")
    _sha256(dataset_report.get("payload_sha256"), field="dataset_report_binding.payload_sha256")
    dataset_manifest = _mapping(payload.get("dataset_manifest_binding"), field="dataset_manifest_binding")
    if set(dataset_manifest) != {"path", "file_sha256"}:
        raise DatasetProducerLineageError("dataset_manifest_binding fields do not match")
    _non_empty(dataset_manifest.get("path"), field="dataset_manifest_binding.path")
    _sha256(dataset_manifest.get("file_sha256"), field="dataset_manifest_binding.file_sha256")
    if artifact_root is not None:
        report_path = _resolve_output(
            artifact_root,
            cast(str, dataset_report["path"]),
            field="dataset_report_binding.path",
        )
        report_payload, observed_report_digest = _read_json(report_path, label="bound dataset report")
        if observed_report_digest != dataset_report["file_sha256"]:
            raise DatasetProducerLineageError("bound dataset report file_sha256 mismatch")
        if _verify_self_digest(report_payload, field="bound dataset report") != dataset_report["payload_sha256"]:
            raise DatasetProducerLineageError("bound dataset report payload_sha256 mismatch")
        manifest_path = _resolve_output(
            artifact_root,
            cast(str, dataset_manifest["path"]),
            field="dataset_manifest_binding.path",
        )
        if file_sha256(manifest_path) != dataset_manifest["file_sha256"]:
            raise DatasetProducerLineageError("bound dataset manifest file_sha256 mismatch")
        try:
            load_real_data_manifest(manifest_path, verify_artifact=True)
        except RealDataManifestError as exc:
            raise DatasetProducerLineageError(f"bound dataset manifest is invalid: {exc}") from exc
        for index, value in enumerate(lineage):
            artifact = _mapping(
                _mapping(value, field=f"lineage_records[{index}]").get("dataset_artifact"),
                field=f"lineage_records[{index}].dataset_artifact",
            )
            artifact_path = _resolve_output(
                artifact_root,
                cast(str, artifact["path"]),
                field=f"lineage_records[{index}].dataset_artifact.path",
            )
            if file_sha256(artifact_path) != artifact["sha256"]:
                raise DatasetProducerLineageError(f"lineage_records[{index}].dataset_artifact.sha256 mismatch")
    return digest


def build_lineage_bound_dataset(
    *,
    source_manifest_path: Path,
    replay_report_path: Path,
    replay_archive_path: Path,
    transform_spec_path: Path,
    dataset_id: str,
    out_dir: Path,
    retrieved_at: str,
    generated_at: str,
    drop_fraction: float = 0.8,
    quench_window_ms: float = 5.0,
) -> dict[str, Any]:
    """Build fresh dataset bytes and seal their producer-time lineage.

    Parameters
    ----------
    source_manifest_path : Path
        Source-object manifest v2 whose referenced cache files are verified.
    replay_report_path : Path
        Self-digested replay-v2 producer report.
    replay_archive_path : Path
        Exact replay archive named and bound by ``replay_report_path``.
    transform_spec_path : Path
        Self-digested bounded transformation specification.
    dataset_id : str
        Stable identifier for the generated dataset and RealDataManifest.
    out_dir : Path
        Previously absent output directory outside every immutable input root.
    retrieved_at : str
        Acquisition timestamp propagated to the dataset manifest.
    generated_at : str
        Fixed producer timestamp used for deterministic output.
    drop_fraction : float
        Fractional plasma-current drop used by the proxy-label detector.
    quench_window_ms : float
        Maximum proxy current-quench interval in milliseconds.

    Returns
    -------
    dict[str, Any]
        Validated, self-digested producer-lineage manifest.

    Raises
    ------
    DatasetProducerLineageError
        If an input, inventory, destination, or binding fails closed.
    """
    for value, field in ((dataset_id, "dataset_id"), (retrieved_at, "retrieved_at"), (generated_at, "generated_at")):
        _non_empty(value, field=field)
    for protected in (source_manifest_path.parent, replay_report_path.parent, replay_archive_path.parent):
        if _is_within(out_dir, protected):
            raise DatasetProducerLineageError("out_dir must be outside every immutable input directory")
    if out_dir.exists():
        raise DatasetProducerLineageError("refusing to overwrite an existing output directory")
    if not out_dir.parent.is_dir():
        raise DatasetProducerLineageError("out_dir parent must be an existing directory")

    source_manifest, source_file_digest = _read_json(source_manifest_path, label="source manifest")
    try:
        require_source_object_manifest_v2(source_manifest, artifact_root=source_manifest_path.parent)
    except SourceObjectManifestError as exc:
        raise DatasetProducerLineageError(f"source manifest is invalid: {exc}") from exc
    source_payload_digest = _verify_self_digest(source_manifest, field="source manifest")
    source_records, programme_classes = _source_inventory(source_manifest)

    replay_report, replay_report_file_digest = _read_json(replay_report_path, label="replay report")
    try:
        archive_raw = replay_archive_path.read_bytes()
    except OSError as exc:
        raise DatasetProducerLineageError(f"cannot read replay archive: {exc}") from exc
    try:
        archive_binding = replay_channels.inspect_replay_archive_bytes(archive_raw, path_name=replay_archive_path.name)
    except ValueError as exc:
        raise DatasetProducerLineageError(f"replay archive is invalid: {exc}") from exc
    derived_ids, failures, member_digests, replay_payload_digest = _replay_inventory(replay_report, archive_binding)
    if set(source_records) != set(derived_ids) | set(failures):
        raise DatasetProducerLineageError("source acquired shots must equal replay derived and failed inventories")

    transform_spec, transform_file_digest = _read_json(transform_spec_path, label="transform spec")
    transform_payload_digest = validate_transform_spec(transform_spec)
    transform_id = _non_empty(transform_spec.get("transform_id"), field="transform spec.transform_id")

    created = False
    try:
        out_dir.mkdir()
        created = True
        shots = _load_shots(archive_raw, derived_ids, programme_classes)
        dataset_report = build_dataset(
            shots,
            dataset_id=dataset_id,
            out_dir=out_dir,
            retrieved_at=retrieved_at,
            generated_at=generated_at,
            drop_fraction=drop_fraction,
            quench_window_ms=quench_window_ms,
        )
        report_path = out_dir / "dataset-report.json"
        report_bytes = _json_bytes(dataset_report)
        report_path.write_bytes(report_bytes)
        dataset_manifest_path = out_dir / f"{dataset_id}.manifest.json"
        records_by_id = {cast(int, record["shot_id"]): record for record in dataset_report["shots"]}
        lineage_records: list[dict[str, Any]] = [
            {
                "shot_id": shot_id,
                "source_parent": _source_binding(source_records[shot_id], shot_id=shot_id),
                "replay_member": {
                    "archive_sha256": archive_binding["file_sha256"],
                    "member_sha256": member_digests[shot_id],
                    "producer_digest_bound": True,
                },
                "transform": {"transform_id": transform_id, "payload_sha256": transform_payload_digest},
                "label_record_sha256": _sha256(
                    _mapping(records_by_id[shot_id]["label_record"], field=f"dataset shot {shot_id}.label_record").get(
                        "payload_sha256"
                    ),
                    field=f"dataset shot {shot_id}.label_record.payload_sha256",
                ),
                "dataset_artifact": {
                    "path": records_by_id[shot_id]["npz"],
                    "sha256": records_by_id[shot_id]["checksum_sha256"],
                },
                "producer_time_attested": True,
            }
            for shot_id in derived_ids
        ]
        generation_pinned = all(
            _source_binding(artifact, shot_id=shot_id)["source_generation_sha256"] is not None
            for shot_id, artifact in source_records.items()
        )
        blockers = list(BASE_BLOCKERS)
        if not generation_pinned:
            blockers.append("source_generation_not_pinned_for_every_shot")
        manifest: dict[str, Any] = {
            "schema_version": PRODUCER_LINEAGE_SCHEMA,
            "status": "blocked",
            "lineage_mode": LINEAGE_MODE,
            "producer_time_lineage": True,
            "dataset_id": dataset_id,
            "synthetic": False,
            "licence_spdx": FAIR_MAST_LICENCE,
            **fair_mast_provenance(),
            "source_manifest_binding": {
                "file_sha256": source_file_digest,
                "payload_sha256": source_payload_digest,
            },
            "replay_report_binding": {
                "file_sha256": replay_report_file_digest,
                "payload_sha256": replay_payload_digest,
            },
            "replay_archive_binding": archive_binding,
            "transform_spec_binding": {
                "file_sha256": transform_file_digest,
                "payload_sha256": transform_payload_digest,
            },
            "dataset_report_binding": {
                "path": report_path.name,
                "file_sha256": hashlib.sha256(report_bytes).hexdigest(),
                "payload_sha256": dataset_report["payload_sha256"],
            },
            "dataset_manifest_binding": {
                "path": dataset_manifest_path.name,
                "file_sha256": file_sha256(dataset_manifest_path),
            },
            "lineage_records": lineage_records,
            "exclusion_ledger": [
                {
                    "shot_id": shot_id,
                    "source_parent": _source_binding(source_records[shot_id], shot_id=shot_id),
                    "reason": failures[shot_id],
                    "replay_member_present": False,
                    "dataset_artifact_present": False,
                }
                for shot_id in sorted(failures)
            ],
            "counts": {
                "acquired_source_shots": len(source_records),
                "lineage_records": len(lineage_records),
                "excluded_shots": len(failures),
            },
            "source_generation_pinned_for_all_records": generation_pinned,
            "blockers": sorted(blockers),
            "claim_boundary": {field: False for field in CLAIM_FIELDS},
            "generated_at": generated_at,
            "payload_sha256": None,
        }
        manifest["payload_sha256"] = canonical_json_sha256(manifest)
        validate_producer_lineage_manifest(manifest, artifact_root=out_dir)
        lineage_path = out_dir / "producer-lineage.json"
        lineage_path.write_bytes(_json_bytes(manifest))
        return manifest
    except Exception:
        if created:
            shutil.rmtree(out_dir)
        raise


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--replay-archive", type=Path, required=True)
    parser.add_argument("--transform-spec", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--retrieved-at", required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--drop-fraction", type=float, default=0.8)
    parser.add_argument("--quench-window-ms", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build one fresh producer-lineage dataset from command-line arguments.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector; ``None`` reads process arguments.

    Returns
    -------
    int
        Zero after successful production and validation.
    """
    args = _parse_args(argv)
    manifest = build_lineage_bound_dataset(
        source_manifest_path=args.source_manifest,
        replay_report_path=args.replay_report,
        replay_archive_path=args.replay_archive,
        transform_spec_path=args.transform_spec,
        dataset_id=args.dataset_id,
        out_dir=args.out_dir,
        retrieved_at=args.retrieved_at,
        generated_at=args.generated_at,
        drop_fraction=args.drop_fraction,
        quench_window_ms=args.quench_window_ms,
    )
    print(f"producer lineage: {manifest['counts']['lineage_records']} records (status=blocked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

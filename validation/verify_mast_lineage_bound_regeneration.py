#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Verify reproducible lineage-bound FAIR-MAST regeneration
"""Reopen and reconcile two fresh lineage-bound FAIR-MAST dataset runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from validation.build_disruption_replay_channels import REPORT_SCHEMA, inspect_replay_archive_bytes
from validation.build_mast_lineage_bound_dataset import CLAIM_FIELDS, validate_producer_lineage_manifest
from validation.mast_source_object_manifest import canonical_json_sha256, require_source_object_manifest_v2

REGENERATION_SCHEMA = "scpn-control.mast-lineage-bound-regeneration-verification.v1.0.0"
REGENERATION_STATUS = "reproducible_blocked"
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "status",
    "synthetic",
    "dataset_id",
    "source_manifest_binding",
    "replay_report_binding",
    "replay_archive_binding",
    "run_a",
    "run_b",
    "comparison",
    "source_generation_pinned_for_all_records",
    "blockers",
    "claim_boundary",
    "generated_at",
    "payload_sha256",
}


class RegenerationVerificationError(ValueError):
    """Raised when two claimed regeneration runs do not form one exact proof."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise RegenerationVerificationError(f"duplicate JSON key {key!r}")
        payload[key] = value
    return payload


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError, RegenerationVerificationError) as exc:
        raise RegenerationVerificationError(f"cannot read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RegenerationVerificationError(f"{label} root must be an object")
    return cast(dict[str, Any], payload), raw


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegenerationVerificationError(f"{field} must be an object")
    return cast(Mapping[str, Any], value)


def _list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise RegenerationVerificationError(f"{field} must be an array")
    return value


def _non_empty(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegenerationVerificationError(f"{field} must be a non-empty string")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise RegenerationVerificationError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RegenerationVerificationError(f"{field} must be a positive integer")
    return value


def _verify_self_digest(payload: Mapping[str, Any], *, field: str) -> str:
    claimed = _sha256(payload.get("payload_sha256"), field=f"{field}.payload_sha256")
    unsigned = dict(payload)
    unsigned["payload_sha256"] = None
    if canonical_json_sha256(unsigned) != claimed:
        raise RegenerationVerificationError(f"{field}.payload_sha256 does not match the payload")
    return claimed


def _file_binding(raw: bytes, payload: Mapping[str, Any]) -> dict[str, str]:
    return {
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "payload_sha256": _verify_self_digest(payload, field="bound JSON"),
    }


def _tree_inventory(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise RegenerationVerificationError(f"run root is not a directory: {root}")
    inventory: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RegenerationVerificationError(f"run tree contains a symlink: {path.name}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RegenerationVerificationError(f"run tree contains a non-regular entry: {path.name}")
        raw = path.read_bytes()
        inventory.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    if not inventory:
        raise RegenerationVerificationError("run tree must contain files")
    return inventory


def _source_shots(payload: Mapping[str, Any]) -> tuple[list[int], bool]:
    shot_ids: list[int] = []
    generation_flags: list[bool] = []
    for index, value in enumerate(_list(payload.get("shots"), field="source manifest.shots")):
        shot = _mapping(value, field=f"source manifest.shots[{index}]")
        if shot.get("status") != "acquired":
            continue
        shot_id = _positive_int(shot.get("shot_id"), field=f"source manifest.shots[{index}].shot_id")
        artifacts = _list(shot.get("artifacts"), field=f"source manifest.shots[{index}].artifacts")
        if len(artifacts) != 1:
            raise RegenerationVerificationError(f"source shot {shot_id} must have one artifact")
        artifact = _mapping(artifacts[0], field=f"source shot {shot_id}.artifact")
        shot_ids.append(shot_id)
        generation_flags.append(artifact.get("source_generation") is not None)
    if shot_ids != sorted(set(shot_ids)) or not shot_ids:
        raise RegenerationVerificationError("source acquired-shot inventory must be non-empty, unique, and sorted")
    return shot_ids, all(generation_flags)


def _replay_shots(payload: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    derived: list[int] = []
    failed: list[int] = []
    for index, value in enumerate(_list(payload.get("shots"), field="replay report.shots")):
        record = _mapping(value, field=f"replay report.shots[{index}]")
        shot_id = _positive_int(record.get("shot_id"), field=f"replay report.shots[{index}].shot_id")
        status = record.get("status")
        if status == "derived":
            derived.append(shot_id)
        elif status == "failed":
            failed.append(shot_id)
        else:
            raise RegenerationVerificationError(f"replay shot {shot_id} has an unsupported status")
    if sorted(derived + failed) != sorted(set(derived + failed)):
        raise RegenerationVerificationError("replay shot inventory contains duplicates")
    return derived, failed


def _run_binding(
    root: Path,
    *,
    label: str,
    source_binding: Mapping[str, str],
    replay_binding: Mapping[str, str],
    archive_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = _tree_inventory(root)
    lineage_path = root / "producer-lineage.json"
    lineage, raw = _read_json(lineage_path, label=f"{label} producer lineage")
    try:
        validate_producer_lineage_manifest(lineage, artifact_root=root)
    except ValueError as exc:
        raise RegenerationVerificationError(f"{label} producer lineage is invalid: {exc}") from exc
    after = _tree_inventory(root)
    if before != after:
        raise RegenerationVerificationError(f"{label} changed while it was being verified")
    if lineage.get("source_manifest_binding") != source_binding:
        raise RegenerationVerificationError(f"{label} source-manifest binding mismatch")
    if lineage.get("replay_report_binding") != replay_binding:
        raise RegenerationVerificationError(f"{label} replay-report binding mismatch")
    if lineage.get("replay_archive_binding") != archive_binding:
        raise RegenerationVerificationError(f"{label} replay-archive binding mismatch")
    if lineage.get("source_generation_pinned_for_all_records") is not True:
        raise RegenerationVerificationError(f"{label} does not pin every source generation")
    return (
        {
            "label": label,
            "producer_lineage": _file_binding(raw, lineage),
            "file_inventory": before,
        },
        lineage,
    )


def validate_regeneration_verification(payload: Mapping[str, Any]) -> str:
    """Validate a sealed two-run regeneration verification report.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Parsed verification report.

    Returns
    -------
    str
        Verified canonical payload SHA-256.

    Raises
    ------
    RegenerationVerificationError
        If structure, digests, inventories, blockers, or claims drift.
    """
    if set(payload) != _TOP_LEVEL_FIELDS:
        raise RegenerationVerificationError("verification report fields do not match the contract")
    if (
        payload.get("schema_version") != REGENERATION_SCHEMA
        or payload.get("status") != REGENERATION_STATUS
        or payload.get("synthetic") is not False
        or payload.get("source_generation_pinned_for_all_records") is not True
    ):
        raise RegenerationVerificationError("verification identity/status fields do not match")
    _non_empty(payload.get("dataset_id"), field="dataset_id")
    _non_empty(payload.get("generated_at"), field="generated_at")
    digest = _verify_self_digest(payload, field="regeneration verification")
    comparison = _mapping(payload.get("comparison"), field="comparison")
    if set(comparison) != {"byte_identical", "file_count", "inventory_sha256"}:
        raise RegenerationVerificationError("comparison fields do not match")
    if comparison.get("byte_identical") is not True:
        raise RegenerationVerificationError("comparison must remain byte-identical")
    file_count = _positive_int(comparison.get("file_count"), field="comparison.file_count")
    _sha256(comparison.get("inventory_sha256"), field="comparison.inventory_sha256")
    run_a = _mapping(payload.get("run_a"), field="run_a")
    run_b = _mapping(payload.get("run_b"), field="run_b")
    for field, run in (("run_a", run_a), ("run_b", run_b)):
        if set(run) != {"label", "producer_lineage", "file_inventory"}:
            raise RegenerationVerificationError(f"{field} fields do not match")
        _non_empty(run.get("label"), field=f"{field}.label")
        lineage = _mapping(run.get("producer_lineage"), field=f"{field}.producer_lineage")
        if set(lineage) != {"file_sha256", "payload_sha256"}:
            raise RegenerationVerificationError(f"{field}.producer_lineage fields do not match")
        _sha256(lineage.get("file_sha256"), field=f"{field}.producer_lineage.file_sha256")
        _sha256(lineage.get("payload_sha256"), field=f"{field}.producer_lineage.payload_sha256")
    inventory_a = _list(run_a.get("file_inventory"), field="run_a.file_inventory")
    inventory_b = _list(run_b.get("file_inventory"), field="run_b.file_inventory")
    if inventory_a != inventory_b or len(inventory_a) != file_count:
        raise RegenerationVerificationError("run inventories are not exactly equal")
    if canonical_json_sha256({"files": inventory_a}) != comparison["inventory_sha256"]:
        raise RegenerationVerificationError("comparison.inventory_sha256 mismatch")
    for index, value in enumerate(inventory_a):
        record = _mapping(value, field=f"file_inventory[{index}]")
        if set(record) != {"path", "bytes", "sha256"}:
            raise RegenerationVerificationError(f"file_inventory[{index}] fields do not match")
        _non_empty(record.get("path"), field=f"file_inventory[{index}].path")
        _positive_int(record.get("bytes"), field=f"file_inventory[{index}].bytes")
        _sha256(record.get("sha256"), field=f"file_inventory[{index}].sha256")
    if run_a["producer_lineage"] != run_b["producer_lineage"]:
        raise RegenerationVerificationError("producer-lineage bindings differ between runs")
    blockers = _list(payload.get("blockers"), field="blockers")
    if not blockers or blockers != sorted(set(blockers)) or any(not isinstance(value, str) or not value for value in blockers):
        raise RegenerationVerificationError("blockers must be a non-empty sorted unique string array")
    claims = _mapping(payload.get("claim_boundary"), field="claim_boundary")
    if set(claims) != set(CLAIM_FIELDS) or any(value is not False for value in claims.values()):
        raise RegenerationVerificationError("claim boundary must remain exactly false")
    for field in ("source_manifest_binding", "replay_report_binding"):
        binding = _mapping(payload.get(field), field=field)
        for key in ("file_sha256", "payload_sha256"):
            _sha256(binding.get(key), field=f"{field}.{key}")
    archive = _mapping(payload.get("replay_archive_binding"), field="replay_archive_binding")
    _sha256(archive.get("file_sha256"), field="replay_archive_binding.file_sha256")
    return digest


def build_regeneration_verification(
    *,
    source_manifest_path: Path,
    replay_report_path: Path,
    replay_archive_path: Path,
    run_a_root: Path,
    run_b_root: Path,
    generated_at: str,
) -> dict[str, Any]:
    """Build a self-digested proof that two fresh dataset runs are byte-identical.

    Parameters
    ----------
    source_manifest_path : Path
        SourceObjectManifest-v2 input beside its referenced artifacts.
    replay_report_path : Path
        Producer-bound replay-v2 report.
    replay_archive_path : Path
        Exact replay archive bound by ``replay_report_path``.
    run_a_root, run_b_root : Path
        Distinct fresh lineage-bound dataset roots.
    generated_at : str
        Explicit evidence-generation timestamp.

    Returns
    -------
    dict[str, Any]
        Sealed reproducibility report with every scientific claim false.

    Raises
    ------
    RegenerationVerificationError
        If any input, output, binding, inventory, or claim is inconsistent.
    """
    if not generated_at:
        raise RegenerationVerificationError("generated_at must be non-empty")
    if run_a_root.resolve() == run_b_root.resolve():
        raise RegenerationVerificationError("run roots must be distinct")
    source, source_raw = _read_json(source_manifest_path, label="source manifest")
    try:
        require_source_object_manifest_v2(source, artifact_root=source_manifest_path.parent)
    except ValueError as exc:
        raise RegenerationVerificationError(f"source manifest is invalid: {exc}") from exc
    source_shots, source_generation_pinned = _source_shots(source)
    if not source_generation_pinned:
        raise RegenerationVerificationError("source generation is not pinned for every acquired shot")
    source_binding = _file_binding(source_raw, source)

    replay, replay_raw = _read_json(replay_report_path, label="replay report")
    if replay.get("schema_version") != REPORT_SCHEMA or replay.get("synthetic") is not False:
        raise RegenerationVerificationError("replay report must be a non-synthetic producer-bound v2 report")
    replay_binding = _file_binding(replay_raw, replay)
    try:
        archive_raw = replay_archive_path.read_bytes()
        archive_binding = inspect_replay_archive_bytes(archive_raw, path_name=replay_archive_path.name)
    except (OSError, ValueError) as exc:
        raise RegenerationVerificationError(f"replay archive is invalid: {exc}") from exc
    if replay.get("channels_archive") != archive_binding:
        raise RegenerationVerificationError("replay report does not bind the exact replay archive")
    derived, failed = _replay_shots(replay)
    if source_shots != sorted(derived + failed):
        raise RegenerationVerificationError("source shots do not equal the replay derived/failed partition")
    if replay.get("n_derived") != len(derived):
        raise RegenerationVerificationError("replay report.n_derived mismatch")

    run_a, lineage_a = _run_binding(
        run_a_root,
        label="run-a",
        source_binding=source_binding,
        replay_binding=replay_binding,
        archive_binding=archive_binding,
    )
    run_b, lineage_b = _run_binding(
        run_b_root,
        label="run-b",
        source_binding=source_binding,
        replay_binding=replay_binding,
        archive_binding=archive_binding,
    )
    if lineage_a.get("dataset_id") != lineage_b.get("dataset_id"):
        raise RegenerationVerificationError("dataset identifiers differ between runs")
    if lineage_a.get("generated_at") != lineage_b.get("generated_at"):
        raise RegenerationVerificationError("dataset generation timestamps differ between runs")
    if run_a["file_inventory"] != run_b["file_inventory"]:
        raise RegenerationVerificationError("fresh dataset runs are not byte-identical")
    if lineage_a.get("blockers") != lineage_b.get("blockers"):
        raise RegenerationVerificationError("dataset blockers differ between runs")
    if lineage_a.get("claim_boundary") != lineage_b.get("claim_boundary"):
        raise RegenerationVerificationError("dataset claim boundaries differ between runs")

    inventory = cast(list[dict[str, Any]], run_a["file_inventory"])
    report: dict[str, Any] = {
        "schema_version": REGENERATION_SCHEMA,
        "status": REGENERATION_STATUS,
        "synthetic": False,
        "dataset_id": lineage_a["dataset_id"],
        "source_manifest_binding": source_binding,
        "replay_report_binding": replay_binding,
        "replay_archive_binding": archive_binding,
        "run_a": run_a,
        "run_b": run_b,
        "comparison": {
            "byte_identical": True,
            "file_count": len(inventory),
            "inventory_sha256": canonical_json_sha256({"files": inventory}),
        },
        "source_generation_pinned_for_all_records": True,
        "blockers": sorted(cast(list[str], lineage_a["blockers"])),
        "claim_boundary": lineage_a["claim_boundary"],
        "generated_at": generated_at,
        "payload_sha256": None,
    }
    report["payload_sha256"] = canonical_json_sha256(report)
    validate_regeneration_verification(report)
    return report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--replay-archive", type=Path, required=True)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write an exclusive L2F-90d regeneration verification report."""
    args = _parse_args(argv)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.json_out.open("x", encoding="utf-8") as handle:
            report = build_regeneration_verification(
                source_manifest_path=args.source_manifest,
                replay_report_path=args.replay_report,
                replay_archive_path=args.replay_archive,
                run_a_root=args.run_a,
                run_b_root=args.run_b,
                generated_at=args.generated_at,
            )
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise RegenerationVerificationError("refusing to overwrite an existing verification report") from exc
    except Exception:
        args.json_out.unlink(missing_ok=True)
        raise
    print(f"verified {report['comparison']['file_count']} byte-identical files (status={report['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

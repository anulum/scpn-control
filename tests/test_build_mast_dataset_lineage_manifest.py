# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — tests for immutable post-hoc FAIR-MAST dataset lineage
"""Adversarial tests for the L2F-90b dataset-lineage contract."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import validation.build_mast_dataset_lineage_manifest as lineage
from validation.build_mast_disruption_dataset import MEASURED_CHANNELS
from validation.fair_mast_source_policy import FAIR_MAST_LICENCE, fair_mast_provenance
from validation.mast_source_object_manifest import canonical_json_sha256
from validation.reconcile_mast_campaign_lineage import (
    LEGACY_DATASET_SCHEMA,
    RECONCILIATION_REPORT_SCHEMA,
    REPLAY_SCHEMA,
)

_FIXED_TS = "2026-07-22T20:00:00Z"
_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_DIGEST_C = "c" * 64


def _seal(payload: dict[str, Any]) -> dict[str, Any]:
    payload["payload_sha256"] = None
    payload["payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _file_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _transform_spec() -> dict[str, Any]:
    return _seal(
        {
            "schema_version": lineage.DATASET_LINEAGE_TRANSFORM_SCHEMA,
            "transform_id": "mast-disruption-legacy-replay-to-supervised-npz-v1",
            "lineage_mode": lineage.LINEAGE_MODE,
            "input_schema": REPLAY_SCHEMA,
            "output_schema": LEGACY_DATASET_SCHEMA,
            "operation_chain": list(lineage.TRANSFORM_OPERATIONS),
            "label_authority": "ip_proxy",
            "producer_time_attested": False,
            "independent_label_authority": False,
            "training_admission": False,
            "payload_sha256": None,
        }
    )


def _source_binding() -> dict[str, str]:
    return {
        "artifact_sha256": _DIGEST_A,
        "selected_array_parent_sha256": _DIGEST_B,
        "acquisition_transform_sha256": _DIGEST_C,
    }


def _valid_manifest() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": lineage.DATASET_LINEAGE_MANIFEST_SCHEMA,
        "status": "blocked",
        "lineage_mode": lineage.LINEAGE_MODE,
        "producer_time_lineage": False,
        "dataset_id": "mast-disruption-campaign01",
        "synthetic": False,
        "licence_spdx": FAIR_MAST_LICENCE,
        **fair_mast_provenance(),
        "reconciliation_binding": {"file_sha256": _DIGEST_A, "payload_sha256": _DIGEST_B},
        "transform_spec_binding": {"file_sha256": _DIGEST_B, "payload_sha256": _DIGEST_C},
        "replay_archive_binding": {
            "path": "derived/channels.npz",
            "file_sha256": _DIGEST_A,
            "producer_digest_bound": False,
        },
        "lineage_records": [
            {
                "shot_id": 101,
                "source_parent": _source_binding(),
                "replay_member": {
                    "archive_path": "derived/channels.npz",
                    "archive_sha256": _DIGEST_A,
                    "member_sha256": _DIGEST_B,
                    "producer_digest_bound": False,
                },
                "dataset_artifact": {"path": "dataset/shot_101.npz", "sha256": _DIGEST_C},
                "transform_spec_sha256": _DIGEST_C,
                "producer_time_attested": False,
            }
        ],
        "exclusion_ledger": [
            {
                "shot_id": 102,
                "source_parent": _source_binding(),
                "reason": "summary.line_average_n_e absent",
                "reason_evidence": "verified_self_digested_replay_report",
                "replay_report_payload_sha256": _DIGEST_B,
                "replay_member_present": False,
                "dataset_artifact_present": False,
            }
        ],
        "counts": {"acquired_source_shots": 2, "lineage_records": 1, "excluded_shots": 1},
        "blockers": list(lineage.BLOCKERS),
        "claim_boundary": {field: False for field in lineage.CLAIM_FIELDS},
        "generated_at": _FIXED_TS,
        "payload_sha256": None,
    }
    return _seal(payload)


def _rebind(payload: dict[str, Any]) -> None:
    payload["payload_sha256"] = None
    payload["payload_sha256"] = canonical_json_sha256(payload)


def _write_replay(path: Path, shot_ids: tuple[int, ...] = (101,)) -> None:
    arrays: dict[str, np.ndarray[Any, Any]] = {"shot_ids": np.asarray(shot_ids, dtype=np.int64)}
    for shot_id in shot_ids:
        for channel in MEASURED_CHANNELS:
            arrays[f"{shot_id}:{channel}"] = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)  # type: ignore[arg-type]


def _campaign_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict[str, Any], dict[str, Any]]:
    root = tmp_path / "campaign"
    root.mkdir(parents=True)
    material = _write_json(root / "material.json", {"legacy": True})
    dataset_manifest = _write_json(root / "dataset/manifest.json", {"legacy": True})
    dataset_npz = root / "dataset/shot_101.npz"
    np.savez(dataset_npz, Ip_MA=np.asarray([1.0, 0.0]))
    dataset_report = _write_json(
        root / "dataset/report.json",
        {
            "shots": [
                {
                    "shot_id": 101,
                    "npz": "shot_101.npz",
                    "checksum_sha256": _file_digest(dataset_npz),
                }
            ]
        },
    )
    replay_archive = root / "derived/channels.npz"
    _write_replay(replay_archive)
    spec = _write_json(tmp_path / "spec.json", {"fixture": True})
    transform = _write_json(tmp_path / "transform.json", _transform_spec())

    input_paths = {
        "material_manifest": material,
        "dataset_manifest": dataset_manifest,
        "dataset_report": dataset_report,
        "channels_archive": replay_archive,
    }
    reconciliation = _seal(
        {
            "schema_version": RECONCILIATION_REPORT_SCHEMA,
            "dataset_id": "mast-disruption-campaign01",
            "generated_at": _FIXED_TS,
            "input_bindings": {
                name: {
                    "path": path.relative_to(root).as_posix(),
                    "file_sha256": _file_digest(path),
                }
                for name, path in input_paths.items()
            },
            "replay_inventory": {
                "payload_sha256": _DIGEST_B,
                "channels_archive_producer_digest_bound": False,
            },
            "cross_inventory": {
                "replay_subset_of_acquired": True,
                "dataset_equals_replay": True,
                "evaluation_equals_dataset": True,
                "exclusions": [
                    {
                        "shot_id": 102,
                        "reason": "summary.line_average_n_e absent",
                        "reason_evidence": "verified_self_digested_replay_report",
                    }
                ],
            },
            "payload_sha256": None,
        }
    )
    report = _write_json(tmp_path / "reconciliation.json", reconciliation)
    migrated = {
        "shots": [
            {
                "shot_id": shot_id,
                "status": "acquired",
                "artifacts": [
                    {
                        "sha256": digest,
                        "parent_digest": _DIGEST_B,
                        "transform_digest": _DIGEST_C,
                    }
                ],
            }
            for shot_id, digest in ((101, _DIGEST_A), (102, _DIGEST_B))
        ]
    }
    return root, spec, report, transform, reconciliation, migrated


def test_transform_spec_and_manifest_validators_accept_sealed_contracts() -> None:
    """Accept exact schemas whose self-digests and bounded claims match."""
    transform = _transform_spec()
    assert lineage.validate_transform_spec(transform) == transform["payload_sha256"]
    manifest = _valid_manifest()
    assert lineage.validate_dataset_lineage_manifest(manifest) == manifest["payload_sha256"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.update({"training_admission": True}), "training_admission"),
        (lambda p: p.update({"producer_time_attested": True}), "producer_time_attested"),
        (lambda p: p.update({"label_authority": "programme_metadata"}), "label_authority"),
        (lambda p: p.update({"operation_chain": []}), "operation_chain"),
        (lambda p: p.update({"extra": True}), "fields do not match"),
    ],
)
def test_transform_spec_rejects_claim_drift(
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    """Reject transform specs that overstate authority or change shape."""
    payload = _transform_spec()
    mutate(payload)
    _rebind(payload)
    with pytest.raises(lineage.DatasetLineageError, match=message):
        lineage.validate_transform_spec(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.update({"producer_time_lineage": True}), "producer_time_lineage"),
        (lambda p: p["claim_boundary"].update({"training_admission": True}), "claim_boundary"),
        (lambda p: p["lineage_records"][0].update({"producer_time_attested": True}), "producer_time_attested"),
        (lambda p: p["lineage_records"][0].update({"transform_spec_sha256": _DIGEST_A}), "transform_spec"),
        (lambda p: p["replay_archive_binding"].update({"producer_digest_bound": True}), "producer_digest"),
        (lambda p: p["exclusion_ledger"][0].update({"reason_evidence": "unverified"}), "reason_evidence"),
        (lambda p: p["counts"].update({"lineage_records": 2}), "counts"),
        (lambda p: p["lineage_records"][0].update({"shot_id": 102}), "overlap"),
    ],
)
def test_manifest_validator_rejects_tampering_and_claim_promotion(
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    """Reject altered bindings, counters, identifiers, and claim boundaries."""
    payload = deepcopy(_valid_manifest())
    mutate(payload)
    _rebind(payload)
    with pytest.raises(lineage.DatasetLineageError, match=message):
        lineage.validate_dataset_lineage_manifest(payload)


def test_build_manifest_binds_every_stage_and_exclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build one deterministic included record and one verified exclusion."""
    root, spec, report, transform, reconciliation, migrated = _campaign_fixture(tmp_path)
    monkeypatch.setattr(lineage, "reconcile_campaign", lambda **_: deepcopy(reconciliation))
    monkeypatch.setattr(lineage, "migrate_material_manifest_v1", lambda *_args, **_kwargs: deepcopy(migrated))

    manifest = lineage.build_dataset_lineage_manifest(
        campaign_root=root,
        spec_path=spec,
        reconciliation_report_path=report,
        transform_spec_path=transform,
        generated_at=_FIXED_TS,
    )

    assert manifest["counts"] == {"acquired_source_shots": 2, "lineage_records": 1, "excluded_shots": 1}
    record = manifest["lineage_records"][0]
    assert record["shot_id"] == 101
    assert record["source_parent"] == _source_binding()
    assert record["dataset_artifact"]["path"] == "dataset/shot_101.npz"
    assert record["replay_member"]["archive_sha256"] == _file_digest(root / "derived/channels.npz")
    assert record["transform_spec_sha256"] == _transform_spec()["payload_sha256"]
    assert manifest["exclusion_ledger"][0]["shot_id"] == 102
    assert set(manifest["claim_boundary"].values()) == {False}
    assert lineage.validate_dataset_lineage_manifest(manifest) == manifest["payload_sha256"]


def test_build_is_deterministic_and_rejects_stale_or_unverified_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require an exact fresh reconciliation and verified exclusion reasons."""
    root, spec, report, transform, reconciliation, migrated = _campaign_fixture(tmp_path)
    monkeypatch.setattr(lineage, "migrate_material_manifest_v1", lambda *_args, **_kwargs: deepcopy(migrated))
    monkeypatch.setattr(lineage, "reconcile_campaign", lambda **_: deepcopy(reconciliation))
    first = lineage.build_dataset_lineage_manifest(
        campaign_root=root,
        spec_path=spec,
        reconciliation_report_path=report,
        transform_spec_path=transform,
        generated_at=_FIXED_TS,
    )
    second = lineage.build_dataset_lineage_manifest(
        campaign_root=root,
        spec_path=spec,
        reconciliation_report_path=report,
        transform_spec_path=transform,
        generated_at=_FIXED_TS,
    )
    assert first == second

    stale = deepcopy(reconciliation)
    stale["dataset_id"] = "other"
    monkeypatch.setattr(lineage, "reconcile_campaign", lambda **_: stale)
    with pytest.raises(lineage.DatasetLineageError, match="fresh reconstruction"):
        lineage.build_dataset_lineage_manifest(
            campaign_root=root,
            spec_path=spec,
            reconciliation_report_path=report,
            transform_spec_path=transform,
            generated_at=_FIXED_TS,
        )

    unverified = deepcopy(reconciliation)
    unverified["cross_inventory"]["exclusions"][0]["reason_evidence"] = "unverified"
    _rebind(unverified)
    _write_json(report, unverified)
    monkeypatch.setattr(lineage, "reconcile_campaign", lambda **_: deepcopy(unverified))
    with pytest.raises(lineage.DatasetLineageError, match="lacks verified replay reason"):
        lineage.build_dataset_lineage_manifest(
            campaign_root=root,
            spec_path=spec,
            reconciliation_report_path=report,
            transform_spec_path=transform,
            generated_at=_FIXED_TS,
        )


def test_build_rejects_input_drift_and_output_inside_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed on post-reconciliation byte drift and campaign writes."""
    root, spec, report, transform, reconciliation, migrated = _campaign_fixture(tmp_path)
    monkeypatch.setattr(lineage, "reconcile_campaign", lambda **_: deepcopy(reconciliation))
    monkeypatch.setattr(lineage, "migrate_material_manifest_v1", lambda *_args, **_kwargs: deepcopy(migrated))
    (root / "dataset/report.json").write_text("drift\n", encoding="utf-8")
    with pytest.raises(lineage.DatasetLineageError, match="checksum mismatch"):
        lineage.build_dataset_lineage_manifest(
            campaign_root=root,
            spec_path=spec,
            reconciliation_report_path=report,
            transform_spec_path=transform,
            generated_at=_FIXED_TS,
        )
    with pytest.raises(lineage.DatasetLineageError, match="outside campaign_root"):
        lineage._require_output_outside_campaign(root, root / "lineage.json")


def test_json_and_scalar_helpers_fail_closed(tmp_path: Path) -> None:
    """Normalise malformed JSON, missing files, scalars, and self-digests."""
    with pytest.raises(lineage.DatasetLineageError, match="cannot read"):
        lineage._read_json(tmp_path / "missing.json", label="missing")
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(lineage.DatasetLineageError, match="cannot parse"):
        lineage._read_json(malformed, label="malformed")
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(lineage.DatasetLineageError, match="duplicate JSON key"):
        lineage._read_json(duplicate, label="duplicate")
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    with pytest.raises(lineage.DatasetLineageError, match="root must be an object"):
        lineage._read_json(scalar, label="scalar")

    with pytest.raises(lineage.DatasetLineageError, match="must be an object"):
        lineage._mapping([], field="mapping")
    with pytest.raises(lineage.DatasetLineageError, match="must be an array"):
        lineage._list({}, field="list")
    for value in (None, ""):
        with pytest.raises(lineage.DatasetLineageError, match="non-empty string"):
            lineage._non_empty_string(value, field="text")
    invalid_integers: list[Any] = [True, "1", 0]
    for invalid_integer in invalid_integers:
        with pytest.raises(lineage.DatasetLineageError, match="positive integer"):
            lineage._positive_integer(invalid_integer, field="integer")
    for value in (None, "A" * 64):
        with pytest.raises(lineage.DatasetLineageError, match="lowercase SHA-256"):
            lineage._sha256(value, field="digest")
    with pytest.raises(lineage.DatasetLineageError, match="does not match"):
        lineage._verify_self_digest({"payload_sha256": _DIGEST_A}, field="sealed")


@pytest.mark.parametrize("relative", ["", "/absolute", "../escape", "a/../b", "C:\\escape", "a\\b"])
def test_path_confinement_rejects_nonportable_paths(tmp_path: Path, relative: str) -> None:
    """Reject empty, absolute, parent, drive, and backslash paths."""
    with pytest.raises(lineage.DatasetLineageError, match="non-empty string|confined POSIX"):
        lineage._resolve_beneath(tmp_path, relative, field="path")


def test_path_confinement_rejects_missing_and_symlink_escape(tmp_path: Path) -> None:
    """Require an existing file whose resolved target remains below root."""
    with pytest.raises(lineage.DatasetLineageError, match="does not resolve"):
        lineage._resolve_beneath(tmp_path, "missing.json", field="path")
    outside = tmp_path.parent / "outside-lineage.json"
    outside.write_text("{}", encoding="utf-8")
    (tmp_path / "link.json").symlink_to(outside)
    with pytest.raises(lineage.DatasetLineageError, match="escapes"):
        lineage._resolve_beneath(tmp_path, "link.json", field="path")
    inside = tmp_path / "inside.json"
    inside.write_text("{}", encoding="utf-8")
    assert lineage._resolve_beneath(tmp_path, "inside.json", field="path") == inside
    lineage._require_output_outside_campaign(tmp_path, tmp_path.parent / "lineage-out.json")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.update({"schema_version": "wrong"}), "unsupported transform"),
        (lambda p: p.update({"transform_id": ""}), "non-empty"),
        (lambda p: p.update({"lineage_mode": "producer_time"}), "lineage_mode"),
        (lambda p: p.update({"input_schema": "wrong"}), "input_schema"),
        (lambda p: p.update({"output_schema": "wrong"}), "output_schema"),
        (lambda p: p.update({"operation_chain": {}}), "must be an array"),
        (lambda p: p.update({"independent_label_authority": True}), "independent_label_authority"),
    ],
)
def test_transform_spec_rejects_every_schema_boundary(
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    """Exercise each transform schema and authority boundary."""
    payload = _transform_spec()
    mutate(payload)
    _rebind(payload)
    with pytest.raises(lineage.DatasetLineageError, match=message):
        lineage.validate_transform_spec(payload)


def test_source_and_dataset_inventory_helpers_reject_malformed_records() -> None:
    """Require unique acquired source artifacts and unique dataset shots."""
    assert lineage._source_records(
        {
            "shots": [
                {"shot_id": 1, "status": "failed"},
                {"shot_id": 2, "status": "acquired", "artifacts": [_source_binding()]},
            ]
        }
    ) == {2: _source_binding()}
    malformed_sources: list[tuple[dict[str, Any], str]] = [
        ({"shots": {}}, "must be an array"),
        ({"shots": [[]]}, "must be an object"),
        ({"shots": [{"shot_id": 0, "status": "acquired", "artifacts": [{}]}]}, "positive integer"),
        ({"shots": [{"shot_id": 1, "status": "acquired", "artifacts": []}]}, "exactly one"),
        (
            {
                "shots": [
                    {"shot_id": 1, "status": "acquired", "artifacts": [{}]},
                    {"shot_id": 1, "status": "acquired", "artifacts": [{}]},
                ]
            },
            "duplicate",
        ),
        ({"shots": [{"shot_id": 1, "status": "failed"}]}, "must contain acquired"),
    ]
    for payload, message in malformed_sources:
        with pytest.raises(lineage.DatasetLineageError, match=message):
            lineage._source_records(payload)

    malformed_datasets: list[tuple[dict[str, Any], str]] = [
        ({"shots": {}}, "must be an array"),
        ({"shots": [[]]}, "must be an object"),
        ({"shots": [{"shot_id": False}]}, "positive integer"),
        ({"shots": [{"shot_id": 1}, {"shot_id": 1}]}, "duplicate"),
        ({"shots": []}, "must contain shots"),
    ]
    for payload, message in malformed_datasets:
        with pytest.raises(lineage.DatasetLineageError, match=message):
            lineage._dataset_records(payload)


def test_replay_member_digest_rejects_duplicate_and_missing_members(tmp_path: Path) -> None:
    """Reject duplicate shot identities and incomplete replay archives."""
    duplicate = tmp_path / "duplicate.npz"
    _write_replay(duplicate, (101, 101))
    with pytest.raises(lineage.DatasetLineageError, match="duplicate replay"):
        lineage._replay_member_digests(duplicate)
    missing = tmp_path / "missing.npz"
    np.savez(
        missing,
        shot_ids=np.asarray([101]),
        **{"101:time_s": np.asarray([1.0])},  # type: ignore[arg-type]  # numpy savez **kwds stub limitation
    )
    with pytest.raises(lineage.DatasetLineageError, match="cannot derive replay"):
        lineage._replay_member_digests(missing)


def test_source_binding_validator_rejects_shape_and_digest() -> None:
    """Require exactly three lowercase digest fields in source bindings."""
    with pytest.raises(lineage.DatasetLineageError, match="fields do not match"):
        lineage._validate_source_binding({}, field="source")
    malformed = _source_binding()
    malformed["artifact_sha256"] = "bad"
    with pytest.raises(lineage.DatasetLineageError, match="lowercase SHA-256"):
        lineage._validate_source_binding(malformed, field="source")


def _run_fixture_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, Any], dict[str, Any]], None],
) -> None:
    root, spec, report, transform, reconciliation, migrated = _campaign_fixture(tmp_path)
    mutate(reconciliation, migrated)
    _rebind(reconciliation)
    _write_json(report, reconciliation)
    monkeypatch.setattr(lineage, "reconcile_campaign", lambda **_: deepcopy(reconciliation))
    monkeypatch.setattr(lineage, "migrate_material_manifest_v1", lambda *_args, **_kwargs: deepcopy(migrated))
    lineage.build_dataset_lineage_manifest(
        campaign_root=root,
        spec_path=spec,
        reconciliation_report_path=report,
        transform_spec_path=transform,
        generated_at=_FIXED_TS,
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda r, _m: r["replay_inventory"].update({"channels_archive_producer_digest_bound": True}),
            "producer binding",
        ),
        (lambda r, _m: r["cross_inventory"].update({"dataset_equals_replay": False}), "exact source"),
        (
            lambda r, _m: r["cross_inventory"]["exclusions"].append(deepcopy(r["cross_inventory"]["exclusions"][0])),
            "duplicate exclusion",
        ),
        (
            lambda r, _m: r["cross_inventory"]["exclusions"][0].update({"shot_id": 101}),
            "overlap",
        ),
        (lambda r, m: m["shots"].append({"shot_id": 103, "status": "acquired", "artifacts": [{}]}), "partition"),
    ],
)
def test_builder_rejects_cross_stage_inconsistency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, Any], dict[str, Any]], None],
    message: str,
) -> None:
    """Reject replay, exclusion, and acquired-set inconsistencies."""
    with pytest.raises(lineage.DatasetLineageError, match=message):
        _run_fixture_build(tmp_path, monkeypatch, mutate)


def test_builder_rejects_replay_set_and_dataset_filename_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject replay membership and legacy shot filename drift."""
    root, spec, report, transform, reconciliation, migrated = _campaign_fixture(tmp_path)
    _write_replay(root / "derived/channels.npz", (101, 103))
    reconciliation["input_bindings"]["channels_archive"]["file_sha256"] = _file_digest(root / "derived/channels.npz")
    _rebind(reconciliation)
    _write_json(report, reconciliation)
    monkeypatch.setattr(lineage, "reconcile_campaign", lambda **_: deepcopy(reconciliation))
    monkeypatch.setattr(lineage, "migrate_material_manifest_v1", lambda *_args, **_kwargs: deepcopy(migrated))
    with pytest.raises(lineage.DatasetLineageError, match="dataset shot set differs"):
        lineage.build_dataset_lineage_manifest(
            campaign_root=root,
            spec_path=spec,
            reconciliation_report_path=report,
            transform_spec_path=transform,
            generated_at=_FIXED_TS,
        )

    root, spec, report, transform, reconciliation, migrated = _campaign_fixture(tmp_path / "second")
    dataset = json.loads((root / "dataset/report.json").read_text(encoding="utf-8"))
    dataset["shots"][0]["npz"] = "wrong.npz"
    _write_json(root / "dataset/report.json", dataset)
    reconciliation["input_bindings"]["dataset_report"]["file_sha256"] = _file_digest(root / "dataset/report.json")
    _rebind(reconciliation)
    _write_json(report, reconciliation)
    monkeypatch.setattr(lineage, "reconcile_campaign", lambda **_: deepcopy(reconciliation))
    monkeypatch.setattr(lineage, "migrate_material_manifest_v1", lambda *_args, **_kwargs: deepcopy(migrated))
    with pytest.raises(lineage.DatasetLineageError, match="must equal"):
        lineage.build_dataset_lineage_manifest(
            campaign_root=root,
            spec_path=spec,
            reconciliation_report_path=report,
            transform_spec_path=transform,
            generated_at=_FIXED_TS,
        )


def test_builder_normalises_activation_and_migration_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalise missing inputs, bad reports, reconciliation, and migration failures."""
    root, spec, report, transform, reconciliation, _ = _campaign_fixture(tmp_path)
    with pytest.raises(lineage.DatasetLineageError, match="generated_at"):
        lineage.build_dataset_lineage_manifest(
            campaign_root=root,
            spec_path=spec,
            reconciliation_report_path=report,
            transform_spec_path=transform,
            generated_at="",
        )
    with pytest.raises(lineage.DatasetLineageError, match="campaign_root"):
        lineage.build_dataset_lineage_manifest(
            campaign_root=tmp_path / "absent",
            spec_path=spec,
            reconciliation_report_path=report,
            transform_spec_path=transform,
            generated_at=_FIXED_TS,
        )
    wrong = deepcopy(reconciliation)
    wrong["schema_version"] = "wrong"
    _rebind(wrong)
    _write_json(report, wrong)
    with pytest.raises(lineage.DatasetLineageError, match="unsupported reconciliation"):
        lineage.build_dataset_lineage_manifest(
            campaign_root=root,
            spec_path=spec,
            reconciliation_report_path=report,
            transform_spec_path=transform,
            generated_at=_FIXED_TS,
        )

    _write_json(report, reconciliation)
    monkeypatch.setattr(lineage, "reconcile_campaign", lambda **_: (_ for _ in ()).throw(ValueError("boom")))
    with pytest.raises(lineage.DatasetLineageError, match="fresh reconciliation failed"):
        lineage.build_dataset_lineage_manifest(
            campaign_root=root,
            spec_path=spec,
            reconciliation_report_path=report,
            transform_spec_path=transform,
            generated_at=_FIXED_TS,
        )
    monkeypatch.setattr(lineage, "reconcile_campaign", lambda **_: deepcopy(reconciliation))
    monkeypatch.setattr(
        lineage,
        "migrate_material_manifest_v1",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("boom")),
    )
    with pytest.raises(lineage.DatasetLineageError, match="material migration failed"):
        lineage.build_dataset_lineage_manifest(
            campaign_root=root,
            spec_path=spec,
            reconciliation_report_path=report,
            transform_spec_path=transform,
            generated_at=_FIXED_TS,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.update({"extra": True}), "fields do not match"),
        (lambda p: p.update({"schema_version": "wrong"}), "unsupported dataset"),
        (lambda p: p.update({"status": "ready"}), "blocked post-hoc"),
        (lambda p: p.update({"synthetic": True}), "synthetic"),
        (lambda p: p.update({"licence_spdx": "MIT"}), "licence_spdx"),
        (lambda p: p.update({"source_policy_url": "https://invalid"}), "source_policy_url"),
        (lambda p: p["reconciliation_binding"].update({"extra": True}), "binding fields"),
        (lambda p: p["replay_archive_binding"].update({"extra": True}), "replay_archive_binding fields"),
        (lambda p: p["replay_archive_binding"].update({"path": "../escape"}), "confined POSIX"),
        (lambda p: p.update({"lineage_records": []}), "at least one included"),
        (lambda p: p["lineage_records"][0].update({"extra": True}), r"lineage_records\[0\] fields"),
        (lambda p: p["lineage_records"][0]["source_parent"].update({"extra": True}), "source_parent fields"),
        (lambda p: p["lineage_records"][0]["replay_member"].update({"extra": True}), "replay_member fields"),
        (
            lambda p: p["lineage_records"][0]["replay_member"].update({"archive_path": "other"}),
            "replay_member binding",
        ),
        (lambda p: p["lineage_records"][0]["dataset_artifact"].update({"extra": True}), "artifact fields"),
        (
            lambda p: p["lineage_records"][0]["dataset_artifact"].update({"path": "/absolute"}),
            "confined POSIX",
        ),
        (lambda p: p["exclusion_ledger"][0].update({"extra": True}), r"exclusion_ledger\[0\] fields"),
        (lambda p: p["exclusion_ledger"][0].update({"replay_member_present": True}), "presence flags"),
        (lambda p: p["blockers"].append("invented"), "blockers"),
        (lambda p: p["claim_boundary"].pop("training_admission"), "claim_boundary"),
    ],
)
def test_manifest_validator_rejects_every_structural_boundary(
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    """Exercise schema, provenance, binding, record, and blocker branches."""
    payload = deepcopy(_valid_manifest())
    mutate(payload)
    _rebind(payload)
    with pytest.raises(lineage.DatasetLineageError, match=message):
        lineage.validate_dataset_lineage_manifest(payload)


def test_manifest_validator_rejects_unsorted_identifiers() -> None:
    """Require deterministic sorted record and exclusion order."""
    payload = deepcopy(_valid_manifest())
    second = deepcopy(payload["lineage_records"][0])
    second["shot_id"] = 100
    payload["lineage_records"].append(second)
    payload["counts"] = {"acquired_source_shots": 3, "lineage_records": 2, "excluded_shots": 1}
    _rebind(payload)
    with pytest.raises(lineage.DatasetLineageError, match="unique and sorted"):
        lineage.validate_dataset_lineage_manifest(payload)


def test_cli_writes_valid_manifest_outside_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Write the sealed result and report bounded record counts."""
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    output = tmp_path / "output/lineage.json"
    manifest = _valid_manifest()
    monkeypatch.setattr(lineage, "build_dataset_lineage_manifest", lambda **_: deepcopy(manifest))
    assert (
        lineage.main(
            [
                "--campaign-root",
                str(campaign),
                "--spec",
                str(tmp_path / "spec.json"),
                "--reconciliation-report",
                str(tmp_path / "report.json"),
                "--transform-spec",
                str(tmp_path / "transform.json"),
                "--generated-at",
                _FIXED_TS,
                "--json-out",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    assert "records=1 exclusions=1" in capsys.readouterr().out

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — FAIR-MAST source-object manifest tests
"""Real-NPZ tests for :mod:`validation.mast_source_object_manifest`."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from validation import mast_source_object_manifest as manifest
from validation import migrate_mast_source_object_manifest as migration


def _arrays() -> dict[str, np.ndarray]:
    return {
        "magnetics.b_field_tor_probe_saddle_field": np.arange(24, dtype="<f8").reshape(2, 12),
        "magnetics.time_saddle": np.linspace(0.0, 2.2e-4, 12, dtype="<f8"),
        "summary.ip": np.asarray([5.0e5, 4.8e5], dtype="<f4"),
    }


def _metadata() -> dict[str, dict[str, Any]]:
    return {
        "magnetics.b_field_tor_probe_saddle_field": {
            "dimensions": ["channel", "time_saddle"],
            "units": "T",
            "timebase": {"kind": "source_dimension", "dimensions": ["time_saddle"]},
            "source_attributes": {"long_name": "toroidal saddle field"},
            "source_chunks": [[2], [12]],
            "metadata_status": "source_xarray",
        },
        "magnetics.time_saddle": {
            "dimensions": ["time_saddle"],
            "units": "s",
            "timebase": {"kind": "source_dimension", "dimensions": ["time_saddle"]},
            "source_attributes": {},
            "source_chunks": None,
            "metadata_status": "source_xarray",
        },
        "summary.ip": {
            "dimensions": ["time"],
            "units": "A",
            "timebase": {"kind": "source_dimension", "dimensions": ["time"]},
            "source_attributes": {},
            "source_chunks": None,
            "metadata_status": "source_xarray",
        },
    }


def _write_npz(root: Path, name: str = "shot_30421.npz") -> tuple[Path, dict[str, np.ndarray]]:
    arrays = _arrays()
    path = root / name
    np.savez_compressed(path, **arrays)
    return path, arrays


def _artifact(root: Path) -> dict[str, Any]:
    path, arrays = _write_npz(root)
    return manifest.build_derived_npz_artifact(
        local_path=path.name,
        artifact_path=path,
        source_uri="s3://mast/level2/shots/30421.zarr",
        arrays=arrays,
        source_metadata=_metadata(),
    )


def _root_fields(*, status: str) -> dict[str, Any]:
    return {
        "machine": "MAST",
        "campaign": "FAIR-MAST level2 disruption material",
        "retrieved_at": "2026-07-10T00:00:00Z",
        "synthetic": False,
        "licence_spdx": "CC-BY-SA-4.0",
        "licence": "CC-BY-SA-4.0",
        "citation": "FAIR-MAST citation",
        "citations": ["FAIR-MAST citation"],
        "source_policy_url": "https://mastapp.site/",
        "status": status,
    }


def _v2(artifact: dict[str, Any]) -> dict[str, Any]:
    return manifest.finalise_source_object_manifest(
        {
            "schema_version": manifest.SOURCE_OBJECT_MANIFEST_SCHEMA,
            **_root_fields(status="complete"),
            "n_requested": 1,
            "n_acquired": 1,
            "total_bytes": artifact["bytes"],
            "shots": [
                {
                    "shot_id": 30421,
                    "status": "acquired",
                    "programme_class": "unknown",
                    "artifacts": [artifact],
                }
            ],
        }
    )


def _rebind_array_lineage(artifact: dict[str, Any]) -> None:
    artifact["source_hierarchy"] = manifest.source_hierarchy(artifact["arrays"])
    artifact["parent"]["descriptor"]["arrays"] = artifact["arrays"]
    parent_digest = manifest.canonical_json_sha256(artifact["parent"]["descriptor"])
    artifact["parent"]["sha256"] = parent_digest
    artifact["parent_digest"] = parent_digest


def _legacy(root: Path) -> dict[str, Any]:
    path, _arrays_by_key = _write_npz(root)
    return {
        "schema_version": manifest.LEGACY_MATERIAL_MANIFEST_SCHEMA,
        "status": "partial",
        "synthetic": False,
        "consumers": ["SCPN-CONTROL", "SCPN-FUSION-CORE"],
        "source": {"path_template": "s3://mast/level2/shots/{shot_id}.zarr"},
        "licence": "MIT",
        "licence_url": "https://opensource.org/license/mit",
        "citation": "FAIR-MAST citation",
        "citations": ["FAIR-MAST citation"],
        "source_policy_url": "https://mastapp.site/",
        "retrieved_at": "2026-07-10T00:00:00Z",
        "generated_at": "2026-07-10T00:00:00Z",
        "shots": [
            {
                "shot_id": 30421,
                "status": "acquired",
                "npz": path.name,
                "checksum_sha256": manifest.file_sha256(path),
                "bytes": path.stat().st_size,
                "saddle_channels": 2,
                "saddle_samples": 12,
                "variables": sorted(_arrays()),
            },
            {"shot_id": 30422, "status": "failed", "error": "KeyError: density"},
        ],
    }


def test_build_and_validate_derived_npz_binds_values_metadata_and_lineage(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    payload = _v2(artifact)

    manifest.validate_source_object_manifest(payload, artifact_root=tmp_path)
    manifest.validate_source_object_manifest(payload)
    assert manifest.require_source_object_manifest_v2(payload, artifact_root=tmp_path) == payload
    assert artifact["source_hierarchy"] == {
        "magnetics": ["b_field_tor_probe_saddle_field", "time_saddle"],
        "summary": ["ip"],
    }
    saddle = next(item for item in artifact["arrays"] if item["array_name"] == "b_field_tor_probe_saddle_field")
    assert saddle["dtype"] == "<f8"
    assert saddle["shape"] == [2, 12]
    assert saddle["dimensions"] == ["channel", "time_saddle"]
    assert saddle["units"] == "T"
    assert saddle["unrepresented_metadata"] == {}
    assert artifact["parent_digest"] == artifact["parent"]["sha256"]
    assert artifact["transform_digest"] == artifact["transform"]["sha256"]


def test_value_digest_binds_dtype_shape_and_values() -> None:
    base = np.asarray([[1, 2]], dtype="<i4")
    assert manifest.array_value_sha256(base) != manifest.array_value_sha256(base.astype("<i8"))
    assert manifest.array_value_sha256(base) != manifest.array_value_sha256(base.reshape(2, 1))
    changed = base.copy()
    changed[0, 0] = 9
    assert manifest.array_value_sha256(base) != manifest.array_value_sha256(changed)


def test_legacy_v1_is_rejected_until_real_npz_migration(tmp_path: Path) -> None:
    legacy = _legacy(tmp_path)
    with pytest.raises(manifest.SourceObjectManifestError, match="migrate_material_manifest_v1"):
        manifest.require_source_object_manifest_v2(legacy, artifact_root=tmp_path)

    migrated = migration.migrate_material_manifest_v1(legacy, artifact_root=tmp_path)
    manifest.validate_source_object_manifest(migrated, artifact_root=tmp_path)
    assert migrated["schema_version"] == manifest.SOURCE_OBJECT_MANIFEST_SCHEMA
    assert migrated["n_requested"] == 2
    assert migrated["n_acquired"] == 1
    assert migrated["shots"][1]["error"] == "KeyError: density"
    assert migrated["migration"]["method"] == "verified_npz_reconstruction"
    assert migrated["licence_spdx"] == "CC-BY-SA-4.0"
    assert migrated["migration"]["legacy_declared_licence"] == "MIT"
    acquired = migrated["shots"][0]
    assert acquired["summary"]["saddle_channels"] == 2
    assert all(array["metadata_status"] == "values_only" for array in acquired["artifacts"][0]["arrays"])
    assert all(array["dimensions"] is None for array in acquired["artifacts"][0]["arrays"])


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload.update(schema_version="wrong"), "schema_version"),
        (lambda payload: payload.update(payload_sha256="bad"), "lowercase SHA-256"),
        (lambda payload: payload.update(payload_sha256="0" * 64), "payload_sha256 mismatch"),
        (lambda payload: payload.update(shots={}), "shots must be a list"),
        (lambda payload: payload.update(n_requested=2), "n_requested"),
        (lambda payload: payload.update(n_acquired=0), "n_acquired"),
        (lambda payload: payload.update(total_bytes=0), "total_bytes"),
    ],
)
def test_manifest_rejects_invalid_root_contract(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    payload = _v2(_artifact(tmp_path))
    mutation(payload)
    if match not in {"lowercase SHA-256", "payload_sha256 mismatch", "schema_version"}:
        payload = manifest.finalise_source_object_manifest(payload)
    with pytest.raises(manifest.SourceObjectManifestError, match=match):
        manifest.validate_source_object_manifest(payload, artifact_root=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload.update(machine=""), "machine"),
        (lambda payload: payload.update(campaign=None), "campaign"),
        (lambda payload: payload.update(retrieved_at=" "), "retrieved_at"),
        (lambda payload: payload.update(synthetic=True), "synthetic"),
        (lambda payload: payload.update(licence_spdx="MIT"), "licence"),
        (lambda payload: payload.update(licence="MIT"), "licence"),
        (lambda payload: payload.update(source_policy_url="https://invalid.example"), "source_policy_url"),
        (lambda payload: payload.update(citation=None), "citation"),
        (lambda payload: payload.update(citations=None), "citations"),
        (lambda payload: payload.update(citations=[]), "citations"),
        (lambda payload: payload.update(status="partial"), "status does not match"),
    ],
)
def test_manifest_rejects_missing_or_false_source_identity(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    payload = _v2(_artifact(tmp_path))
    mutation(payload)
    payload = manifest.finalise_source_object_manifest(payload)
    with pytest.raises(manifest.SourceObjectManifestError, match=match):
        manifest.validate_source_object_manifest(payload, artifact_root=tmp_path)


@pytest.mark.parametrize(
    ("mutate_artifact", "match"),
    [
        (lambda artifact: artifact.update(artifact_kind="raw_npz"), "artifact_kind"),
        (lambda artifact: artifact.update(local_path=None), "local_path must be a string"),
        (lambda artifact: artifact.update(local_path="../escape.npz"), "beneath"),
        (lambda artifact: artifact.update(source_uri=None), "source_uri"),
        (lambda artifact: artifact.update(source_uri="https://example.invalid/shot"), "source_uri"),
        (lambda artifact: artifact.update(sha256="bad"), "sha256"),
        (lambda artifact: artifact.update(bytes=-1), "bytes"),
        (lambda artifact: artifact.update(bytes=True), "bytes"),
        (lambda artifact: artifact.update(arrays=[]), "arrays"),
        (lambda artifact: artifact.update(source_hierarchy={}), "source_hierarchy"),
        (lambda artifact: artifact.update(parent=None), "parent must be an object"),
        (lambda artifact: artifact.update(parent={}), "parent descriptor"),
        (lambda artifact: artifact["parent"]["descriptor"].update(source_uri="s3://mast/wrong"), "source_uri mismatch"),
        (lambda artifact: artifact["parent"].update(digest_kind="raw-byte-sha256"), "digest_kind"),
        (lambda artifact: artifact.update(parent_digest="0" * 64), "parent_digest"),
        (lambda artifact: artifact.update(transform=None), "transform must be an object"),
        (lambda artifact: artifact.update(transform={}), "preserves_sample_values"),
        (lambda artifact: artifact.update(transform_digest="0" * 64), "transform_digest"),
        (lambda artifact: artifact["transform"].update(preserves_sample_values=False), "preserves_sample_values"),
        (
            lambda artifact: artifact["transform"].update(preserves_source_hierarchy_in_npz=True),
            "preserves_source_hierarchy",
        ),
        (
            lambda artifact: artifact["transform"].update(preserves_source_metadata_in_npz=True),
            "preserves_source_metadata",
        ),
        (
            lambda artifact: artifact["transform"].update(preserves_source_chunking_in_npz=True),
            "preserves_source_chunking",
        ),
    ],
)
def test_manifest_rejects_broken_artifact_contract(
    tmp_path: Path,
    mutate_artifact: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    artifact = _artifact(tmp_path)
    mutate_artifact(artifact)
    payload = _v2(artifact)
    with pytest.raises(manifest.SourceObjectManifestError, match=match):
        manifest.validate_source_object_manifest(payload, artifact_root=tmp_path)


def test_manifest_rejects_local_file_tampering(tmp_path: Path) -> None:
    payload = _v2(_artifact(tmp_path))
    path = tmp_path / "shot_30421.npz"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(manifest.SourceObjectManifestError, match="bytes does not match"):
        manifest.validate_source_object_manifest(payload, artifact_root=tmp_path)


def test_manifest_rejects_npz_member_value_drift_even_with_rebound_file_digest(tmp_path: Path) -> None:
    payload = _v2(_artifact(tmp_path))
    path = tmp_path / "shot_30421.npz"
    arrays = _arrays()
    arrays["summary.ip"][0] = 1.0
    np.savez_compressed(path, **arrays)
    artifact = payload["shots"][0]["artifacts"][0]
    artifact["sha256"] = manifest.file_sha256(path)
    artifact["bytes"] = path.stat().st_size
    payload["total_bytes"] = path.stat().st_size
    payload = manifest.finalise_source_object_manifest(payload)
    with pytest.raises(manifest.SourceObjectManifestError, match="value digest mismatch"):
        manifest.validate_source_object_manifest(payload, artifact_root=tmp_path)


@pytest.mark.parametrize(
    ("mutate_arrays", "match"),
    [
        (lambda arrays: arrays.__setitem__(0, None), "must be an object"),
        (lambda arrays: arrays.append(copy.deepcopy(arrays[0])), "unique string"),
        (lambda arrays: arrays[0].update(group="wrong"), "hierarchy"),
        (lambda arrays: arrays[0].update(dtype="not-a-dtype"), "dtype is invalid"),
        (lambda arrays: arrays[0].update(shape="bad"), "shape"),
        (lambda arrays: arrays[0].update(shape=[-1]), "shape"),
        (lambda arrays: arrays[0].update(nbytes="bad"), "nbytes"),
        (lambda arrays: arrays[0].update(nbytes=-1), "nbytes"),
        (lambda arrays: arrays[0].update(value_sha256="bad"), "value_sha256"),
    ],
)
def test_manifest_rejects_invalid_array_inventory(
    tmp_path: Path,
    mutate_arrays: Callable[[list[Any]], None],
    match: str,
) -> None:
    artifact = _artifact(tmp_path)
    mutate_arrays(artifact["arrays"])
    if all(isinstance(array, dict) for array in artifact["arrays"]):
        _rebind_array_lineage(artifact)
    payload = _v2(artifact)
    with pytest.raises(manifest.SourceObjectManifestError, match=match):
        manifest.validate_source_object_manifest(payload, artifact_root=tmp_path)


@pytest.mark.parametrize(
    ("mutate_array", "match"),
    [
        (lambda array: array.update(dtype="|O"), "must not contain Python objects"),
        (lambda array: array.update(metadata_status="invented"), "metadata_status"),
        (lambda array: array.update(dimensions=None), "dimensions must preserve"),
        (lambda array: array.update(dimensions=[1, "time_saddle"]), "dimensions must preserve"),
        (lambda array: array.update(dimensions=["flat"]), "dimensions must match"),
        (lambda array: array.update(source_attributes=None), "source_attributes"),
        (lambda array: array.update(units=1), "units"),
        (lambda array: array.update(timebase="time_saddle"), "timebase"),
        (lambda array: array.update(timebase={"kind": "guessed", "dimensions": []}), "timebase"),
        (lambda array: array.update(timebase={"kind": "source_dimension", "dimensions": None}), "timebase dimensions"),
        (
            lambda array: array.update(timebase={"kind": "source_dimension", "dimensions": ["missing"]}),
            "timebase dimensions",
        ),
        (lambda array: array.update(source_chunks="unchunked"), "source_chunks must match"),
        (lambda array: array.update(source_chunks=[[2]]), "source_chunks must match"),
        (lambda array: array.update(source_chunks=[[], [12]]), "does not tile"),
        (lambda array: array.update(source_chunks=[[True], [12]]), "does not tile"),
        (lambda array: array.update(source_chunks=[[1], [12]]), "does not tile"),
        (lambda array: array.update(metadata_status="values_only"), "values_only metadata fields"),
        (lambda array: array.update(unrepresented_metadata=None), "unrepresented_metadata"),
    ],
)
def test_manifest_rejects_false_or_incomplete_array_metadata(
    tmp_path: Path,
    mutate_array: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    artifact = _artifact(tmp_path)
    mutate_array(artifact["arrays"][0])
    _rebind_array_lineage(artifact)
    payload = _v2(artifact)
    with pytest.raises(manifest.SourceObjectManifestError, match=match):
        manifest.validate_source_object_manifest(payload, artifact_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("dtype", "<f4", "dtype mismatch"),
        ("shape", [24], "shape mismatch"),
        ("nbytes", 1, "nbytes does not match"),
    ],
)
def test_manifest_rejects_npz_structure_drift(
    tmp_path: Path,
    field: str,
    replacement: Any,
    match: str,
) -> None:
    artifact = _artifact(tmp_path)
    artifact["arrays"][0][field] = replacement
    if field == "dtype":
        artifact["arrays"][0]["nbytes"] = 96
    if field == "shape":
        artifact["arrays"][0]["dimensions"] = ["flat"]
        artifact["arrays"][0]["timebase"] = None
        artifact["arrays"][0]["source_chunks"] = [[24]]
    _rebind_array_lineage(artifact)
    payload = _v2(artifact)
    with pytest.raises(manifest.SourceObjectManifestError, match=match):
        manifest.validate_source_object_manifest(payload, artifact_root=tmp_path)


def test_manifest_rejects_npz_member_drift_and_rebound_byte_digest(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    path = tmp_path / "shot_30421.npz"
    np.savez_compressed(path, **{"summary.ip": _arrays()["summary.ip"]})
    artifact["sha256"] = manifest.file_sha256(path)
    artifact["bytes"] = path.stat().st_size
    payload = _v2(artifact)
    with pytest.raises(manifest.SourceObjectManifestError, match="NPZ members"):
        manifest.validate_source_object_manifest(payload, artifact_root=tmp_path)


def test_manifest_rejects_rebound_file_checksum_and_missing_file(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    artifact["sha256"] = "0" * 64
    payload = _v2(artifact)
    with pytest.raises(manifest.SourceObjectManifestError, match="sha256 does not match"):
        manifest.validate_source_object_manifest(payload, artifact_root=tmp_path)

    artifact = _artifact(tmp_path)
    artifact["local_path"] = "missing.npz"
    payload = _v2(artifact)
    with pytest.raises(manifest.SourceObjectManifestError, match="does not resolve to a file"):
        manifest.validate_source_object_manifest(payload, artifact_root=tmp_path)


def test_manifest_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    outside_path, arrays = _write_npz(outside)
    (root / "link").symlink_to(outside, target_is_directory=True)
    artifact = manifest.build_derived_npz_artifact(
        local_path="link/shot_30421.npz",
        artifact_path=outside_path,
        source_uri="s3://mast/level2/shots/30421.zarr",
        arrays=arrays,
    )
    payload = _v2(artifact)
    with pytest.raises(manifest.SourceObjectManifestError, match="escapes the artefact root"):
        manifest.validate_source_object_manifest(payload, artifact_root=root)


@pytest.mark.parametrize("local_path", ["", "/absolute.npz", "../escape.npz", "nested\\windows.npz"])
def test_builder_rejects_nonportable_or_escaping_paths(tmp_path: Path, local_path: str) -> None:
    path, arrays = _write_npz(tmp_path)
    with pytest.raises(manifest.SourceObjectManifestError, match="portable|beneath"):
        manifest.build_derived_npz_artifact(
            local_path=local_path,
            artifact_path=path,
            source_uri="s3://mast/level2/shots/30421.zarr",
            arrays=arrays,
        )


def test_builder_rejects_missing_empty_and_unsafe_arrays(tmp_path: Path) -> None:
    path, arrays = _write_npz(tmp_path)
    with pytest.raises(manifest.SourceObjectManifestError, match="does not exist"):
        manifest.build_derived_npz_artifact(
            local_path=path.name,
            artifact_path=tmp_path / "missing.npz",
            source_uri="s3://mast/level2/shots/30421.zarr",
            arrays=arrays,
        )
    with pytest.raises(manifest.SourceObjectManifestError, match="at least one"):
        manifest.build_array_inventory({})
    with pytest.raises(manifest.SourceObjectManifestError, match="group"):
        manifest.build_array_inventory({"flat": np.asarray([1])})
    with pytest.raises(manifest.SourceObjectManifestError, match="object dtype"):
        manifest.build_array_inventory({"summary.bad": np.asarray([object()], dtype=object)})
    with pytest.raises(manifest.SourceObjectManifestError, match="object-dtype"):
        manifest.array_value_sha256(np.asarray([object()], dtype=object))
    with pytest.raises(manifest.SourceObjectManifestError, match="metadata_status"):
        manifest.build_array_inventory(
            {"summary.ip": np.asarray([1.0])},
            source_metadata={"summary.ip": {"metadata_status": "invented"}},
        )
    values_only = manifest.build_array_inventory(
        {"summary.ip": np.asarray([1.0])},
        source_metadata={},
    )
    assert values_only[0]["metadata_status"] == "values_only"


def test_legacy_migration_fails_closed_on_untrusted_inputs(tmp_path: Path) -> None:
    legacy = _legacy(tmp_path)
    wrong_schema = {**legacy, "schema_version": "wrong"}
    with pytest.raises(manifest.SourceObjectManifestError, match="expected legacy schema"):
        migration.migrate_material_manifest_v1(wrong_schema, artifact_root=tmp_path)

    corrupted = copy.deepcopy(legacy)
    corrupted["shots"][0]["checksum_sha256"] = "0" * 64
    with pytest.raises(manifest.SourceObjectManifestError, match="checksum mismatch"):
        migration.migrate_material_manifest_v1(corrupted, artifact_root=tmp_path)

    wrong_bytes = copy.deepcopy(legacy)
    wrong_bytes["shots"][0]["bytes"] = 1
    with pytest.raises(manifest.SourceObjectManifestError, match="byte count mismatch"):
        migration.migrate_material_manifest_v1(wrong_bytes, artifact_root=tmp_path)


@pytest.mark.parametrize(
    ("mutate_legacy", "match"),
    [
        (lambda legacy: legacy.update(shots={}), "legacy shots must be a list"),
        (lambda legacy: legacy.update(source=None), "legacy source must be an object"),
        (lambda legacy: legacy.update(source={"path_template": 1}), "path_template"),
        (lambda legacy: legacy.update(source={"path_template": "s3://mast/static.zarr"}), "path_template"),
        (lambda legacy: legacy.update(shots=[None]), "must be an object"),
        (lambda legacy: legacy["shots"][0].update(shot_id=0), "positive integer"),
        (lambda legacy: legacy["shots"][0].update(shot_id=True), "positive integer"),
        (lambda legacy: legacy["shots"][0].update(npz=None), "npz must be a string"),
    ],
)
def test_legacy_migration_rejects_invalid_structure(
    tmp_path: Path,
    mutate_legacy: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    legacy = _legacy(tmp_path)
    mutate_legacy(legacy)
    with pytest.raises(manifest.SourceObjectManifestError, match=match):
        migration.migrate_material_manifest_v1(legacy, artifact_root=tmp_path)


def test_legacy_migration_preserves_explicit_unknowns_when_optional_fields_are_absent(tmp_path: Path) -> None:
    legacy = _legacy(tmp_path)
    for field in ("licence", "licence_url", "citation", "citations", "source_policy_url"):
        legacy.pop(field)
    legacy["shots"] = [{"shot_id": 30422, "status": "failed"}]
    migrated = migration.migrate_material_manifest_v1(legacy, artifact_root=tmp_path)
    assert migrated["shots"][0]["error"] == "legacy acquisition failed without an error message"
    assert migrated["licence"] == "CC-BY-SA-4.0"
    assert migrated["migration"]["legacy_declared_licence"] is None


@pytest.mark.parametrize(
    ("shots", "match"),
    [
        ([None], "must be an object"),
        ([{"shot_id": 0, "status": "failed", "error": "x"}], "positive integer"),
        ([{"shot_id": True, "status": "failed", "error": "x"}], "positive integer"),
        (
            [
                {"shot_id": 1, "programme_class": "unknown", "status": "failed", "error": "x"},
                {"shot_id": 1, "programme_class": "unknown", "status": "failed", "error": "x"},
            ],
            "duplicate shot_id",
        ),
        ([{"shot_id": 1, "programme_class": "unknown", "status": "planned"}], "status must be"),
        ([{"shot_id": 1, "programme_class": "unknown", "status": "acquired", "artifacts": []}], "non-empty list"),
    ],
)
def test_manifest_rejects_invalid_shot_records(shots: list[Any], match: str) -> None:
    payload = manifest.finalise_source_object_manifest(
        {
            "schema_version": manifest.SOURCE_OBJECT_MANIFEST_SCHEMA,
            **_root_fields(status="empty"),
            "n_requested": len(shots),
            "n_acquired": 0,
            "total_bytes": 0,
            "shots": shots,
        }
    )
    with pytest.raises(manifest.SourceObjectManifestError, match=match):
        manifest.validate_source_object_manifest(payload)


def test_manifest_rejects_nonobject_artifact(tmp_path: Path) -> None:
    payload = manifest.finalise_source_object_manifest(
        {
            "schema_version": manifest.SOURCE_OBJECT_MANIFEST_SCHEMA,
            **_root_fields(status="complete"),
            "n_requested": 1,
            "n_acquired": 1,
            "total_bytes": 0,
            "shots": [{"shot_id": 1, "programme_class": "unknown", "status": "acquired", "artifacts": [None]}],
        }
    )
    with pytest.raises(manifest.SourceObjectManifestError, match="must be an object"):
        manifest.validate_source_object_manifest(payload, artifact_root=tmp_path)


def test_failed_shot_requires_an_explanation(tmp_path: Path) -> None:
    payload = manifest.finalise_source_object_manifest(
        {
            "schema_version": manifest.SOURCE_OBJECT_MANIFEST_SCHEMA,
            **_root_fields(status="empty"),
            "n_requested": 1,
            "n_acquired": 0,
            "total_bytes": 0,
            "shots": [{"shot_id": 30421, "programme_class": "unknown", "status": "failed", "error": ""}],
        }
    )
    with pytest.raises(manifest.SourceObjectManifestError, match="must explain"):
        manifest.validate_source_object_manifest(payload, artifact_root=tmp_path)


def test_shot_programme_class_is_closed_vocabulary() -> None:
    payload = manifest.finalise_source_object_manifest(
        {
            "schema_version": manifest.SOURCE_OBJECT_MANIFEST_SCHEMA,
            **_root_fields(status="empty"),
            "n_requested": 1,
            "n_acquired": 0,
            "total_bytes": 0,
            "shots": [{"shot_id": 30421, "programme_class": "guessed", "status": "failed", "error": "x"}],
        }
    )
    with pytest.raises(manifest.SourceObjectManifestError, match="programme_class"):
        manifest.validate_source_object_manifest(payload)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tests for the verified FAIR-MAST source artefact reader
"""Contract tests for :mod:`validation.mast_source_artifact_reader`."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from validation import mast_source_artifact_reader as reader
from validation.fair_mast_source_policy import FAIR_MAST_LICENCE, fair_mast_provenance
from validation.mast_source_artifact_reader import (
    SourceArtifactReaderError,
    load_pinned_source_manifest,
    load_verified_source_manifest,
    read_verified_npz_artifact,
)
from validation.mast_source_object_manifest import (
    SOURCE_OBJECT_MANIFEST_SCHEMA,
    build_derived_npz_artifact,
    file_sha256,
    finalise_source_object_manifest,
    require_source_object_manifest_v2,
)


def _save_named_arrays(path: Path, arrays: Mapping[str, object]) -> None:
    """Write dynamically named members through NumPy's real NPZ interface."""
    writer = cast(Callable[..., None], np.savez_compressed)
    writer(path, **arrays)


def _write_manifest(root: Path, *, failed: bool = False) -> tuple[Path, dict[str, Any]]:
    arrays = {
        "magnetics.b_field_tor_probe_saddle_field": np.arange(24, dtype=np.float64).reshape(2, 12),
        "summary.ip": np.linspace(4.0e5, 3.0e5, 12, dtype=np.float64),
        "summary.time": np.linspace(0.0, 0.011, 12, dtype=np.float64),
    }
    artifact_path = root / "shot_30421.npz"
    _save_named_arrays(artifact_path, arrays)
    metadata = {
        key: {
            "dimensions": ["channel", "time"] if value.ndim == 2 else ["time"],
            "units": "T" if "field" in key else ("A" if key.endswith("ip") else "s"),
            "timebase": {"kind": "source_dimension", "dimensions": ["time"]},
            "source_attributes": {"source": "FAIR-MAST"},
            "source_chunks": None,
            "metadata_status": "source_xarray",
        }
        for key, value in arrays.items()
    }
    artifact = build_derived_npz_artifact(
        local_path=artifact_path.name,
        artifact_path=artifact_path,
        source_uri="s3://mast/level2/shots/30421.zarr",
        arrays=arrays,
        source_metadata=metadata,
    )
    if failed:
        shots = [
            {
                "shot_id": 30421,
                "status": "failed",
                "programme_class": "unknown",
                "error": "source group unavailable",
            }
        ]
        n_acquired = 0
        total_bytes = 0
        status = "empty"
    else:
        shots = [
            {
                "shot_id": 30421,
                "status": "acquired",
                "programme_class": "unknown",
                "artifacts": [artifact],
            }
        ]
        n_acquired = 1
        total_bytes = artifact["bytes"]
        status = "complete"
    payload = finalise_source_object_manifest(
        {
            "schema_version": SOURCE_OBJECT_MANIFEST_SCHEMA,
            "manifest_kind": "source_object_inventory",
            "machine": "MAST",
            "campaign": "reader contract",
            "status": status,
            "synthetic": False,
            "licence_spdx": FAIR_MAST_LICENCE,
            **fair_mast_provenance(),
            "retrieved_at": "2026-07-22T00:00:00Z",
            "n_acquired": n_acquired,
            "n_requested": 1,
            "total_bytes": total_bytes,
            "shots": shots,
        }
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path, payload


def test_reader_exposes_verified_group_aware_immutable_arrays(tmp_path: Path) -> None:
    """The public reader exposes only exact keys and immutable verified values."""
    manifest_path, _payload = _write_manifest(tmp_path)
    manifest = load_verified_source_manifest(manifest_path, artifact_root=tmp_path)
    artifact = read_verified_npz_artifact(manifest, artifact_root=tmp_path, shot_id=30421)

    assert artifact.archive_keys == (
        "magnetics.b_field_tor_probe_saddle_field",
        "summary.ip",
        "summary.time",
    )
    assert artifact.metadata["summary.ip"]["units"] == "A"
    assert artifact.metadata["summary.ip"]["dimensions"] == ("time",)
    assert artifact.arrays["summary.ip"].flags.writeable is False
    assert artifact.source_uri == "s3://mast/level2/shots/30421.zarr"
    assert len(artifact.manifest_sha256) == len(artifact.artifact_sha256) == 64
    with pytest.raises(ValueError, match="read-only"):
        artifact.arrays["summary.ip"][0] = 0.0
    with pytest.raises(TypeError):
        cast(Any, artifact.arrays)["new"] = np.asarray([1.0])
    with pytest.raises(TypeError):
        cast(Any, artifact.metadata["summary.ip"])["units"] = "kA"
    with pytest.raises(TypeError):
        artifact.metadata["summary.ip"]["source_attributes"]["source"] = "changed"


def test_manifest_loader_rejects_non_object_and_invalid_json(tmp_path: Path) -> None:
    """Malformed JSON roots fail before any artefact can be selected."""
    with pytest.raises(SourceArtifactReaderError, match="cannot read"):
        load_verified_source_manifest(tmp_path / "missing.json", artifact_root=tmp_path)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("[]", encoding="utf-8")
    with pytest.raises(SourceArtifactReaderError, match="root must be an object"):
        load_verified_source_manifest(manifest_path, artifact_root=tmp_path)
    manifest_path.write_text("{", encoding="utf-8")
    with pytest.raises(SourceArtifactReaderError, match="cannot read"):
        load_verified_source_manifest(manifest_path, artifact_root=tmp_path)
    manifest_path.write_text('{"schema_version":"a","schema_version":"b"}', encoding="utf-8")
    with pytest.raises(SourceArtifactReaderError, match="duplicate JSON key"):
        load_verified_source_manifest(manifest_path, artifact_root=tmp_path)


def test_manifest_loader_wraps_schema_and_local_file_failures(tmp_path: Path) -> None:
    """Schema drift and local-byte drift share the bounded reader error surface."""
    manifest_path, payload = _write_manifest(tmp_path)
    payload["schema_version"] = "legacy"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SourceArtifactReaderError, match="verification failed"):
        load_verified_source_manifest(manifest_path, artifact_root=tmp_path)

    manifest_path, _payload = _write_manifest(tmp_path)
    (tmp_path / "shot_30421.npz").write_bytes(b"changed")
    with pytest.raises(SourceArtifactReaderError, match="verification failed"):
        load_verified_source_manifest(manifest_path, artifact_root=tmp_path)


def test_pinned_manifest_loader_binds_exact_bytes_and_v2_payload(tmp_path: Path) -> None:
    """The pinned loader verifies both file bytes and the manifest self-digest."""
    manifest_path, payload = _write_manifest(tmp_path)

    verified, byte_sha256 = load_pinned_source_manifest(
        manifest_path,
        artifact_root=tmp_path,
        expected_sha256=file_sha256(manifest_path),
    )

    assert byte_sha256 == file_sha256(manifest_path)
    assert verified["payload_sha256"] == payload["payload_sha256"]

    with pytest.raises(SourceArtifactReaderError, match="pinned digest"):
        load_pinned_source_manifest(
            manifest_path,
            artifact_root=tmp_path,
            expected_sha256="0" * 64,
        )


def test_reader_rejects_missing_failed_and_wrong_kind_shots(tmp_path: Path) -> None:
    """Shot state and exact artefact kind are mandatory selection boundaries."""
    _manifest_path, payload = _write_manifest(tmp_path)
    with pytest.raises(SourceArtifactReaderError, match="exactly one manifest record"):
        read_verified_npz_artifact(payload, artifact_root=tmp_path, shot_id=99999)
    with pytest.raises(SourceArtifactReaderError, match="exactly one artefact"):
        read_verified_npz_artifact(payload, artifact_root=tmp_path, shot_id=30421, artifact_kind="native_zarr")

    _manifest_path, failed_payload = _write_manifest(tmp_path, failed=True)
    with pytest.raises(SourceArtifactReaderError, match="is not acquired"):
        read_verified_npz_artifact(failed_payload, artifact_root=tmp_path, shot_id=30421)


def test_reader_rejects_duplicate_kind_and_invalid_manifest(tmp_path: Path) -> None:
    """Ambiguous artefact cardinality and invalid self-digests fail closed."""
    _manifest_path, payload = _write_manifest(tmp_path)
    duplicate = copy.deepcopy(payload)
    artifact = duplicate["shots"][0]["artifacts"][0]
    duplicate["shots"][0]["artifacts"].append(copy.deepcopy(artifact))
    duplicate["total_bytes"] *= 2
    duplicate = finalise_source_object_manifest(duplicate)
    with pytest.raises(SourceArtifactReaderError, match="found 2"):
        read_verified_npz_artifact(duplicate, artifact_root=tmp_path, shot_id=30421)

    payload["payload_sha256"] = "0" * 64
    with pytest.raises(SourceArtifactReaderError, match="verification failed"):
        read_verified_npz_artifact(payload, artifact_root=tmp_path, shot_id=30421)


def _replace_after_manifest_verification(
    monkeypatch: pytest.MonkeyPatch,
    replacement: dict[str, npt.NDArray[np.float64]] | None,
    artifact_path: Path,
) -> None:
    original = require_source_object_manifest_v2

    def validate_then_replace(manifest: Any, *, artifact_root: Path | None = None) -> dict[str, Any]:
        verified = original(manifest, artifact_root=artifact_root)
        if replacement is None:
            artifact_path.unlink()
        else:
            _save_named_arrays(artifact_path, replacement)
        return verified

    monkeypatch.setattr(reader, "require_source_object_manifest_v2", validate_then_replace)


def test_reader_rechecks_member_set_after_manifest_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A member-set race is rejected after the initial whole-manifest check."""
    _manifest_path, payload = _write_manifest(tmp_path)
    _replace_after_manifest_verification(
        monkeypatch,
        {"summary.time": np.linspace(0.0, 0.011, 12, dtype=np.float64)},
        tmp_path / "shot_30421.npz",
    )
    with pytest.raises(SourceArtifactReaderError, match="member set changed"):
        read_verified_npz_artifact(payload, artifact_root=tmp_path, shot_id=30421)


@pytest.mark.parametrize("mutation", ["dtype", "shape", "value"])
def test_reader_rechecks_each_member_property_after_manifest_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Dtype, shape, and value drift each fail the second member check."""
    _manifest_path, payload = _write_manifest(tmp_path)
    with np.load(tmp_path / "shot_30421.npz", allow_pickle=False) as archive:
        replacement = {key: np.asarray(archive[key]).copy() for key in archive.files}
    if mutation == "dtype":
        replacement["summary.ip"] = replacement["summary.ip"].astype(np.float32)
    elif mutation == "shape":
        replacement["summary.ip"] = replacement["summary.ip"].reshape(3, 4)
    else:
        replacement["summary.ip"][0] = 0.0
    _replace_after_manifest_verification(monkeypatch, replacement, tmp_path / "shot_30421.npz")
    with pytest.raises(SourceArtifactReaderError, match="member 'summary.ip' changed"):
        read_verified_npz_artifact(payload, artifact_root=tmp_path, shot_id=30421)


def test_reader_wraps_open_race_and_rechecks_final_file_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Open-time disappearance and post-open byte drift use bounded errors."""
    _manifest_path, payload = _write_manifest(tmp_path)
    artifact_path = tmp_path / "shot_30421.npz"
    _replace_after_manifest_verification(monkeypatch, None, artifact_path)
    with pytest.raises(SourceArtifactReaderError, match="cannot open verified NPZ"):
        read_verified_npz_artifact(payload, artifact_root=tmp_path, shot_id=30421)

    _manifest_path, payload = _write_manifest(tmp_path)
    monkeypatch.undo()
    monkeypatch.setattr(reader, "file_sha256", lambda _path: "0" * 64)
    with pytest.raises(SourceArtifactReaderError, match="changed while it was opened"):
        read_verified_npz_artifact(payload, artifact_root=tmp_path, shot_id=30421)

    def fail_digest(_path: Path) -> str:
        raise OSError("gone")

    monkeypatch.setattr(reader, "file_sha256", fail_digest)
    with pytest.raises(SourceArtifactReaderError, match="cannot recheck"):
        read_verified_npz_artifact(payload, artifact_root=tmp_path, shot_id=30421)


def test_reader_rechecks_root_confinement_after_manifest_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A symlink race cannot redirect a verified artefact outside its root."""
    _manifest_path, payload = _write_manifest(tmp_path)
    artifact_path = tmp_path / "shot_30421.npz"
    outside = tmp_path.parent / f"{tmp_path.name}_outside.npz"
    outside.write_bytes(artifact_path.read_bytes())
    original = require_source_object_manifest_v2

    def validate_then_redirect(manifest: Any, *, artifact_root: Path | None = None) -> dict[str, Any]:
        verified = original(manifest, artifact_root=artifact_root)
        artifact_path.unlink()
        artifact_path.symlink_to(outside)
        return verified

    monkeypatch.setattr(reader, "require_source_object_manifest_v2", validate_then_redirect)
    with pytest.raises(SourceArtifactReaderError, match="escape its root"):
        read_verified_npz_artifact(payload, artifact_root=tmp_path, shot_id=30421)

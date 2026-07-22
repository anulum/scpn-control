# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — FAIR-MAST material-manifest migration tests
"""Direct real-NPZ tests for the legacy source-manifest migration surface."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from validation import mast_source_object_manifest as manifest
from validation import migrate_mast_source_object_manifest as migration


def _save_named_arrays(path: Path, arrays: Mapping[str, object]) -> None:
    """Write dynamically named NPZ members through NumPy's runtime interface."""
    writer = cast(Callable[..., None], np.savez_compressed)
    writer(path, **arrays)


def _legacy_path_record(local_path: str) -> dict[str, object]:
    return {
        "schema_version": manifest.LEGACY_MATERIAL_MANIFEST_SCHEMA,
        "source": {"path_template": "s3://mast/level2/shots/{shot_id}.zarr"},
        "retrieved_at": "2026-07-10T00:00:00Z",
        "shots": [
            {
                "shot_id": 30421,
                "status": "acquired",
                "npz": local_path,
                "checksum_sha256": "0" * 64,
                "bytes": 0,
            }
        ],
    }


def test_migrate_material_manifest_v1_verifies_npz_and_replaces_false_policy(tmp_path: Path) -> None:
    """Migration verifies real NPZ bytes and replaces false legacy policy."""
    arrays = {
        "magnetics.time_saddle": np.asarray([0.0, 2.0e-5], dtype="<f8"),
        "magnetics.b_field_tor_probe_saddle_field": np.asarray([[1.0, 2.0]], dtype="<f4"),
    }
    path = tmp_path / "shot_30421.npz"
    _save_named_arrays(path, arrays)
    legacy = {
        "schema_version": manifest.LEGACY_MATERIAL_MANIFEST_SCHEMA,
        "status": "complete",
        "synthetic": False,
        "source": {"path_template": "s3://mast/level2/shots/{shot_id}.zarr"},
        "licence": "MIT",
        "retrieved_at": "2026-07-10T00:00:00Z",
        "shots": [
            {
                "shot_id": 30421,
                "status": "acquired",
                "npz": path.name,
                "checksum_sha256": manifest.file_sha256(path),
                "bytes": path.stat().st_size,
            }
        ],
    }

    migrated = migration.migrate_material_manifest_v1(legacy, artifact_root=tmp_path)

    manifest.validate_source_object_manifest(migrated, artifact_root=tmp_path)
    assert migrated["licence_spdx"] == "CC-BY-SA-4.0"
    assert migrated["migration"]["legacy_declared_licence"] == "MIT"
    assert migrated["migration"]["source_policy_action"] == "replaced with authoritative FAIR-MAST policy"
    migrated_arrays = migrated["shots"][0]["artifacts"][0]["arrays"]
    assert all(item["metadata_status"] == "values_only" for item in migrated_arrays)


@pytest.mark.parametrize("local_path", ["../escape.npz", "/absolute.npz"])
def test_migration_rejects_legacy_paths_outside_the_artifact_root(tmp_path: Path, local_path: str) -> None:
    """Legacy absolute and parent-relative paths fail closed on every OS."""
    with pytest.raises(
        manifest.SourceObjectManifestError,
        match=r"(?:beneath|escapes) the artefact root",
    ):
        migration.migrate_material_manifest_v1(_legacy_path_record(local_path), artifact_root=tmp_path)


def test_migration_rejects_symlink_escape_and_missing_artifact(tmp_path: Path) -> None:
    """Migration rejects symlink escapes and absent source artifacts."""
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    outside_file = outside / "shot_30421.npz"
    _save_named_arrays(outside_file, {"summary.ip": np.asarray([1.0])})
    (root / "link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(manifest.SourceObjectManifestError, match="escapes the artefact root"):
        migration.migrate_material_manifest_v1(
            _legacy_path_record("link/shot_30421.npz"),
            artifact_root=root,
        )
    with pytest.raises(manifest.SourceObjectManifestError, match="does not resolve to a file"):
        migration.migrate_material_manifest_v1(
            _legacy_path_record("missing.npz"),
            artifact_root=root,
        )

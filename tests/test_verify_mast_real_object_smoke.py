# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Provenance-bound FAIR-MAST real-object smoke tests
"""Direct file-boundary tests for the digest-pinned MAST smoke gate."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from validation.mast_source_object_manifest import file_sha256
from validation.verify_mast_real_object_smoke import RealObjectSmokeError, main, verify_real_object_smoke


def _save_named_arrays(path: Path, arrays: Mapping[str, object]) -> None:
    """Write dynamically named members through NumPy's real NPZ interface."""
    writer = cast(Callable[..., None], np.savez_compressed)
    writer(path, **arrays)


def _material_fixture(tmp_path: Path, *, synthetic: bool = False) -> tuple[Path, Path, str, str]:
    """Create a legacy manifest and real NPZ file with group-aware members."""
    artifact = tmp_path / "shot_30421.npz"
    _save_named_arrays(
        artifact,
        {
            "summary.time": np.asarray([0.0, 2.0e-5, 4.0e-5], dtype="<f8"),
            "summary.ip": np.asarray([0.0, 1.2e6, 1.1e6], dtype="<f8"),
            "magnetics.b_field_tor_probe_saddle_field": np.asarray([[0.1, 0.2, 0.3]], dtype="<f4"),
        },
    )
    manifest = tmp_path / "manifest.json"
    payload = {
        "schema_version": "scpn-control.mast-disruption-material.v1",
        "synthetic": synthetic,
        "source": {"path_template": "s3://mast/level2/shots/{shot_id}.zarr"},
        "licence": "MIT",
        "retrieved_at": "2026-07-10T00:00:00Z",
        "shots": [
            {
                "shot_id": 30421,
                "status": "acquired",
                "npz": artifact.name,
                "checksum_sha256": file_sha256(artifact),
                "bytes": artifact.stat().st_size,
            }
        ],
    }
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return manifest, artifact, file_sha256(manifest), file_sha256(artifact)


def test_real_object_smoke_verifies_pinned_transport_and_blocks_claims(tmp_path: Path) -> None:
    """The production gate opens the pinned NPZ but promotes no scientific claim."""
    manifest, artifact, manifest_sha256, artifact_sha256 = _material_fixture(tmp_path)

    report = verify_real_object_smoke(
        manifest,
        artifact_root=tmp_path,
        shot_id=30421,
        expected_manifest_sha256=manifest_sha256,
        expected_artifact_sha256=artifact_sha256,
    )

    assert artifact.is_file()
    assert report["status"] == "verified_transport_only"
    assert report["transport_interoperability_verified"] is True
    assert report["channel_extraction_admissible"] is False
    assert report["scientific_validity_claim_admissible"] is False
    assert report["facility_claim_admissible"] is False
    assert report["control_admission_admissible"] is False
    assert report["legacy_manifest_sha256"] == manifest_sha256
    assert report["artifact_sha256"] == artifact_sha256
    assert report["legacy_declared_licence"] == "MIT"
    assert report["licence_spdx"] == "CC-BY-SA-4.0"
    assert report["licence_corrected_in_memory"] is True
    assert report["archive_key_count"] == 3
    assert report["payload_sha256"]


@pytest.mark.parametrize("digest_field", ["manifest", "artifact"])
def test_real_object_smoke_rejects_unpinned_external_bytes(tmp_path: Path, digest_field: str) -> None:
    """Neither a substituted manifest nor substituted artifact can satisfy the gate."""
    manifest, _, manifest_sha256, artifact_sha256 = _material_fixture(tmp_path)
    expected_manifest = "0" * 64 if digest_field == "manifest" else manifest_sha256
    expected_artifact = "0" * 64 if digest_field == "artifact" else artifact_sha256

    with pytest.raises(RealObjectSmokeError, match="pinned digest"):
        verify_real_object_smoke(
            manifest,
            artifact_root=tmp_path,
            shot_id=30421,
            expected_manifest_sha256=expected_manifest,
            expected_artifact_sha256=expected_artifact,
        )


def test_real_object_smoke_rejects_synthetic_declaration_and_duplicate_keys(tmp_path: Path) -> None:
    """A synthetic or structurally ambiguous manifest cannot become real-object proof."""
    manifest, _, manifest_sha256, artifact_sha256 = _material_fixture(tmp_path, synthetic=True)
    with pytest.raises(RealObjectSmokeError, match="synthetic=false"):
        verify_real_object_smoke(
            manifest,
            artifact_root=tmp_path,
            shot_id=30421,
            expected_manifest_sha256=manifest_sha256,
            expected_artifact_sha256=artifact_sha256,
        )

    manifest.write_text('{"schema_version":"one","schema_version":"two"}\n', encoding="utf-8")
    with pytest.raises(RealObjectSmokeError, match="duplicate JSON key"):
        verify_real_object_smoke(
            manifest,
            artifact_root=tmp_path,
            shot_id=30421,
            expected_manifest_sha256=file_sha256(manifest),
            expected_artifact_sha256=artifact_sha256,
        )


def test_real_object_smoke_cli_writes_digest_bound_report(tmp_path: Path) -> None:
    """The CLI exercises the real manifest-to-report file boundary."""
    manifest, _, manifest_sha256, artifact_sha256 = _material_fixture(tmp_path)
    report_path = tmp_path / "evidence" / "smoke.json"

    result = main(
        [
            "--manifest",
            str(manifest),
            "--artifact-root",
            str(tmp_path),
            "--shot-id",
            "30421",
            "--expected-manifest-sha256",
            manifest_sha256,
            "--expected-artifact-sha256",
            artifact_sha256,
            "--json-out",
            str(report_path),
        ]
    )

    assert result == 0
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["artifact_sha256"] == artifact_sha256
    assert written["facility_claim_admissible"] is False


@pytest.mark.parametrize("shot_id", [0, -1, True])
def test_real_object_smoke_rejects_invalid_shot_identity(tmp_path: Path, shot_id: int) -> None:
    """Only positive non-boolean shot identifiers enter external verification."""
    with pytest.raises(RealObjectSmokeError, match="positive integer"):
        verify_real_object_smoke(
            tmp_path / "unused.json",
            artifact_root=tmp_path,
            shot_id=shot_id,
            expected_manifest_sha256="0" * 64,
            expected_artifact_sha256="0" * 64,
        )


def test_real_object_smoke_wraps_manifest_io_and_json_failures(tmp_path: Path) -> None:
    """Absent, malformed, and non-object manifests fail through the bounded API."""
    missing = tmp_path / "missing.json"
    with pytest.raises(RealObjectSmokeError, match="cannot read legacy material manifest"):
        verify_real_object_smoke(
            missing,
            artifact_root=tmp_path,
            shot_id=30421,
            expected_manifest_sha256="0" * 64,
            expected_artifact_sha256="0" * 64,
        )

    malformed = tmp_path / "malformed.json"
    malformed.write_bytes(b"\xff")
    with pytest.raises(RealObjectSmokeError, match="cannot decode legacy material manifest"):
        verify_real_object_smoke(
            malformed,
            artifact_root=tmp_path,
            shot_id=30421,
            expected_manifest_sha256=file_sha256(malformed),
            expected_artifact_sha256="0" * 64,
        )

    non_object = tmp_path / "list.json"
    non_object.write_text("[]\n", encoding="utf-8")
    with pytest.raises(RealObjectSmokeError, match="root must be an object"):
        verify_real_object_smoke(
            non_object,
            artifact_root=tmp_path,
            shot_id=30421,
            expected_manifest_sha256=file_sha256(non_object),
            expected_artifact_sha256="0" * 64,
        )


def test_real_object_smoke_wraps_missing_selected_shot(tmp_path: Path) -> None:
    """A digest-valid manifest cannot prove a shot it does not contain."""
    manifest, _, manifest_sha256, artifact_sha256 = _material_fixture(tmp_path)
    with pytest.raises(RealObjectSmokeError, match="external source-object verification failed"):
        verify_real_object_smoke(
            manifest,
            artifact_root=tmp_path,
            shot_id=30422,
            expected_manifest_sha256=manifest_sha256,
            expected_artifact_sha256=artifact_sha256,
        )

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Digest-bound FAIR-MAST real-object alignment tests
"""Real file-boundary tests for the L2F-11b external alignment gate."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from validation.fair_mast_source_policy import FAIR_MAST_LICENCE, fair_mast_provenance
from validation.mast_source_object_manifest import (
    SOURCE_OBJECT_MANIFEST_SCHEMA,
    build_derived_npz_artifact,
    file_sha256,
    finalise_source_object_manifest,
)
from validation.verify_mast_real_object_alignment import (
    RealObjectAlignmentError,
    main,
    verify_real_object_alignment,
)

_SHOT_ID = 30421


def _save_named_arrays(path: Path, arrays: Mapping[str, object]) -> None:
    """Write dynamically named members through NumPy's real NPZ interface."""
    writer = cast(Callable[..., None], np.savez_compressed)
    writer(path, **arrays)


def _legacy_fixture(tmp_path: Path) -> tuple[Path, Path, str, str]:
    """Create the recovered values-only manifest boundary used by L2F-11b."""
    artifact = tmp_path / f"shot_{_SHOT_ID}.npz"
    _save_named_arrays(
        artifact,
        {
            "summary.time": np.linspace(0.0, 0.02, 21, dtype=np.float64),
            "summary.ip": np.linspace(1.0e6, 0.8e6, 21, dtype=np.float64),
            "summary.line_average_n_e": np.linspace(2.0e19, 1.5e19, 21, dtype=np.float64),
            "equilibrium.time": np.linspace(0.0, 0.02, 11, dtype=np.float64),
            "equilibrium.q95": np.linspace(3.0, 4.0, 11, dtype=np.float64),
        },
    )
    manifest = tmp_path / "legacy.json"
    payload = {
        "schema_version": "scpn-control.mast-disruption-material.v1",
        "synthetic": False,
        "source": {"path_template": "s3://mast/level2/shots/{shot_id}.zarr"},
        "licence": "MIT",
        "retrieved_at": "2026-07-10T00:00:00Z",
        "shots": [
            {
                "shot_id": _SHOT_ID,
                "status": "acquired",
                "npz": artifact.name,
                "checksum_sha256": file_sha256(artifact),
                "bytes": artifact.stat().st_size,
            }
        ],
    }
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return manifest, artifact, file_sha256(manifest), file_sha256(artifact)


def _source_metadata(units: str) -> dict[str, object]:
    return {
        "dimensions": ["time"],
        "units": units,
        "timebase": {"kind": "source_dimension", "dimensions": ["time"]},
        "source_attributes": {"source": "FAIR-MAST"},
        "source_chunks": None,
        "metadata_status": "source_xarray",
    }


def _v2_fixture(
    tmp_path: Path,
    *,
    synthetic: bool = False,
    nonuniform_reference: bool = False,
) -> tuple[Path, Path, str, str]:
    """Create a metadata-attested v2 object spanning all five bound scalars."""
    summary_time = (
        np.asarray([0.0, 0.001, 0.003, 0.004], dtype=np.float64)
        if nonuniform_reference
        else np.linspace(0.0, 0.02, 21, dtype=np.float64)
    )
    equilibrium_time = np.linspace(float(summary_time[0]), float(summary_time[-1]), 11, dtype=np.float64)
    arrays = {
        "summary.time": summary_time,
        "summary.ip": np.linspace(1.0e6, 0.8e6, summary_time.size, dtype=np.float64),
        "summary.line_average_n_e": np.linspace(2.0e19, 1.5e19, summary_time.size, dtype=np.float64),
        "equilibrium.time": equilibrium_time,
        "equilibrium.q95": np.linspace(3.0, 4.0, equilibrium_time.size, dtype=np.float64),
        "equilibrium.magnetic_axis_z": np.linspace(-0.02, 0.02, equilibrium_time.size, dtype=np.float64),
    }
    metadata = {
        "summary.time": _source_metadata("s"),
        "summary.ip": _source_metadata("A"),
        "summary.line_average_n_e": _source_metadata("1 / m ** 3"),
        "equilibrium.time": _source_metadata("s"),
        "equilibrium.q95": _source_metadata(""),
        "equilibrium.magnetic_axis_z": _source_metadata("m"),
    }
    artifact_path = tmp_path / f"shot_{_SHOT_ID}.npz"
    _save_named_arrays(artifact_path, arrays)
    artifact = build_derived_npz_artifact(
        local_path=artifact_path.name,
        artifact_path=artifact_path,
        source_uri=f"s3://mast/level2/shots/{_SHOT_ID}.zarr",
        arrays=arrays,
        source_metadata=metadata,
    )
    payload = finalise_source_object_manifest(
        {
            "schema_version": SOURCE_OBJECT_MANIFEST_SCHEMA,
            "manifest_kind": "source_object_inventory",
            "machine": "MAST",
            "campaign": "L2F-11b alignment contract",
            "status": "complete",
            "synthetic": synthetic,
            "licence_spdx": FAIR_MAST_LICENCE,
            **fair_mast_provenance(),
            "retrieved_at": "2026-07-22T00:00:00Z",
            "n_acquired": 1,
            "n_requested": 1,
            "total_bytes": artifact["bytes"],
            "shots": [
                {
                    "shot_id": _SHOT_ID,
                    "status": "acquired",
                    "programme_class": "unknown",
                    "artifacts": [artifact],
                }
            ],
        }
    )
    manifest_path = tmp_path / "source-object-v2.json"
    manifest_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path, artifact_path, file_sha256(manifest_path), file_sha256(artifact_path)


def test_legacy_alignment_proves_transport_and_metadata_blockers(tmp_path: Path) -> None:
    """Recovered values-only snapshots remain explicit fail-closed evidence."""
    manifest, _artifact, manifest_sha256, artifact_sha256 = _legacy_fixture(tmp_path)

    report = verify_real_object_alignment(
        manifest,
        artifact_root=tmp_path,
        shot_id=_SHOT_ID,
        expected_manifest_sha256=manifest_sha256,
        expected_artifact_sha256=artifact_sha256,
    )

    assert report["status"] == "alignment_blocked"
    assert report["transport_interoperability_verified"] is True
    assert report["bound_scalar_alignment_complete"] is False
    assert report["n_bound_channels_aligned"] == 0
    assert report["n_channels_not_aligned"] == 11
    assert report["scientific_validity_claim_admissible"] is False
    assert report["facility_claim_admissible"] is False
    assert report["control_admission_admissible"] is False
    assert report["input_manifest_sha256"] == manifest_sha256
    assert len(report["source_manifest_sha256"]) == 64
    channel = next(item for item in report["channels"] if item["channel"] == "Ip_MA")
    assert channel == {
        "channel": "Ip_MA",
        "status": "not_aligned",
        "reason_code": "manifest_does_not_attest_source_xarray_metadata",
        "binding_assessment_status": "source_metadata_unverified",
    }
    assert all("values" not in item and "valid_mask" not in item for item in report["channels"])


def test_v2_alignment_proves_five_scalar_digests_without_raw_arrays(tmp_path: Path) -> None:
    """A metadata-attested v2 object aligns five scalars and retains six blockers."""
    manifest, _artifact, manifest_sha256, artifact_sha256 = _v2_fixture(tmp_path)

    report = verify_real_object_alignment(
        manifest,
        manifest_format="source-object-v2",
        artifact_root=tmp_path,
        shot_id=_SHOT_ID,
        expected_manifest_sha256=manifest_sha256,
        expected_artifact_sha256=artifact_sha256,
    )

    assert report["status"] == "bound_scalar_alignment_verified"
    assert report["bound_scalar_alignment_complete"] is True
    assert report["full_canonical_extraction_admissible"] is False
    assert report["n_bound_channels_aligned"] == 5
    assert report["n_channels_not_aligned"] == 6
    aligned = [item for item in report["channels"] if item["status"] == "aligned_with_validity_mask"]
    assert {item["channel"] for item in aligned} == {
        "time_s",
        "Ip_MA",
        "q95",
        "ne_1e19",
        "vertical_position_m",
    }
    assert all(len(item["values_sha256"]) == 64 for item in aligned)
    assert all(len(item["valid_mask_sha256"]) == 64 for item in aligned)
    assert all("values" not in item and "valid_mask" not in item for item in aligned)


def test_v2_alignment_turns_invalid_reference_into_digest_bound_blocker(tmp_path: Path) -> None:
    """An inadmissible real timebase produces a bounded report, not interpolation."""
    manifest, _artifact, manifest_sha256, artifact_sha256 = _v2_fixture(tmp_path, nonuniform_reference=True)

    report = verify_real_object_alignment(
        manifest,
        manifest_format="source-object-v2",
        artifact_root=tmp_path,
        shot_id=_SHOT_ID,
        expected_manifest_sha256=manifest_sha256,
        expected_artifact_sha256=artifact_sha256,
    )

    assert report["status"] == "alignment_blocked"
    assert report["reason_code"] == "reference_or_source_timebase_inadmissible"
    assert report["channels"] == []
    assert len(report["alignment_report_sha256"]) == 64


def test_alignment_rejects_wrong_format_synthetic_v2_and_substituted_bytes(tmp_path: Path) -> None:
    """Format, source truth, and both expected digests are hard boundaries."""
    manifest, _artifact, manifest_sha256, artifact_sha256 = _v2_fixture(tmp_path, synthetic=True)
    with pytest.raises(RealObjectAlignmentError, match="synthetic must be false"):
        verify_real_object_alignment(
            manifest,
            manifest_format="source-object-v2",
            artifact_root=tmp_path,
            shot_id=_SHOT_ID,
            expected_manifest_sha256=manifest_sha256,
            expected_artifact_sha256=artifact_sha256,
        )

    with pytest.raises(RealObjectAlignmentError, match="pinned digest"):
        verify_real_object_alignment(
            manifest,
            manifest_format="source-object-v2",
            artifact_root=tmp_path,
            shot_id=_SHOT_ID,
            expected_manifest_sha256="0" * 64,
            expected_artifact_sha256=artifact_sha256,
        )

    manifest, _artifact, manifest_sha256, artifact_sha256 = _v2_fixture(tmp_path)
    with pytest.raises(RealObjectAlignmentError, match="selected NPZ"):
        verify_real_object_alignment(
            manifest,
            manifest_format="source-object-v2",
            artifact_root=tmp_path,
            shot_id=_SHOT_ID,
            expected_manifest_sha256=manifest_sha256,
            expected_artifact_sha256="0" * 64,
        )

    with pytest.raises(RealObjectAlignmentError, match="unsupported manifest format"):
        verify_real_object_alignment(
            manifest,
            manifest_format=cast(Any, "guessed"),
            artifact_root=tmp_path,
            shot_id=_SHOT_ID,
            expected_manifest_sha256=manifest_sha256,
            expected_artifact_sha256=artifact_sha256,
        )

    legacy, _artifact, legacy_sha256, artifact_sha256 = _legacy_fixture(tmp_path)
    with pytest.raises(RealObjectAlignmentError, match="pinned legacy"):
        verify_real_object_alignment(
            legacy,
            artifact_root=tmp_path,
            shot_id=_SHOT_ID,
            expected_manifest_sha256="0" * 64,
            expected_artifact_sha256=artifact_sha256,
        )


def test_alignment_cli_writes_self_digested_raw_data_free_report(tmp_path: Path) -> None:
    """The CLI traverses manifest, artefact, alignment, and JSON file boundaries."""
    manifest, _artifact, manifest_sha256, artifact_sha256 = _legacy_fixture(tmp_path)
    report_path = tmp_path / "evidence" / "alignment.json"

    result = main(
        [
            "--manifest",
            str(manifest),
            "--artifact-root",
            str(tmp_path),
            "--shot-id",
            str(_SHOT_ID),
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
    assert written["status"] == "alignment_blocked"
    assert written["payload_sha256"]
    assert "raw_arrays" not in written

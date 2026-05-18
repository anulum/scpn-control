# SPDX-License-Identifier: AGPL-3.0-or-later
# ----------------------------------------------------------------------
# SCPN Control - Data Manifest Validation Runner Tests
# Copyright (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ----------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path

from validation.validate_data_manifests import main, validate_manifest_directory


ROOT = Path(__file__).resolve().parents[1]


def test_validate_manifest_directory_reports_repository_manifests() -> None:
    report = validate_manifest_directory(
        ROOT / "validation" / "reference_data",
        verify_artifacts=True,
    )

    assert report["status"] == "pass"
    assert report["total"] >= 3
    assert report["real"] >= 2
    assert report["synthetic"] >= 1
    assert report["artifact_coverage"]["expected"] == 21
    assert report["artifact_coverage"]["covered"] == 21
    assert report["artifact_coverage"]["missing"] == []
    assert report["acquisition_specs"]["total"] >= 1
    assert report["acquisition_specs"]["mdsplus"] >= 1
    assert not report["errors"]


def test_validate_manifest_directory_rejects_bad_checksum(tmp_path) -> None:
    artefact = tmp_path / "bad_shot.npz"
    artefact.write_bytes(b"changed data")
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifest = {
        "schema_version": "1.0",
        "dataset_id": "bad-checksum",
        "machine": "DIII-D",
        "shot": "163303",
        "synthetic": False,
        "source": {
            "kind": "local_archive",
            "uri": str(artefact),
            "access": "temporary test archive",
        },
        "retrieved_at": "2026-05-18T01:30:00Z",
        "checksum_sha256": "0" * 64,
        "licence": "temporary test policy",
        "signals": [
            {
                "name": "plasma_current",
                "path": "Ip_MA",
                "units": "MA",
                "timebase": "time_s",
            }
        ],
    }
    (manifest_dir / "bad.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_manifest_directory(tmp_path, verify_artifacts=True)

    assert report["status"] == "fail"
    assert report["total"] == 1
    assert len(report["errors"]) == 1
    assert "checksum mismatch" in report["errors"][0]["error"]


def test_validate_manifest_directory_rejects_bad_acquisition_spec(tmp_path) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifest = {
        "schema_version": "1.0",
        "dataset_id": "synthetic-ci",
        "machine": "DIII-D",
        "shot": "synthetic",
        "synthetic": True,
        "source": {
            "kind": "synthetic",
            "uri": "generated://unit-test",
            "access": "test fixture",
        },
        "synthetic_generator": "tests.test_validate_data_manifests",
        "synthetic_seed": 7,
        "signals": [
            {
                "name": "plasma_current",
                "path": "Ip_MA",
                "units": "MA",
                "timebase": "time_s",
            }
        ],
    }
    (manifest_dir / "synthetic.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    spec_dir = tmp_path / "acquisition_specs"
    spec_dir.mkdir()
    (spec_dir / "bad_mdsplus.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "tree": "DIII-D",
                "shot": 163303,
                "source_uri": "mdsplus://DIII-D/163303",
                "access_policy": "facility-approved",
                "licence": "facility data policy",
                "signals": [],
            }
        ),
        encoding="utf-8",
    )

    report = validate_manifest_directory(tmp_path, verify_artifacts=True)

    assert report["status"] == "fail"
    assert report["acquisition_specs"]["total"] == 1
    assert len(report["errors"]) == 1
    assert "MDSplus acquisition requires at least one signal" in report["errors"][0]["error"]


def test_main_writes_json_report(tmp_path, capsys) -> None:
    output = tmp_path / "data_manifest_report.json"

    exit_code = main(
        [
            "--root",
            str(ROOT / "validation" / "reference_data"),
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["total"] >= 3
    assert report["acquisition_specs"]["total"] >= 1
    stdout = capsys.readouterr().out
    assert "acquisition_specs=" in stdout

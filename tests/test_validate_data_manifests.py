# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Data Manifest Validation Runner Tests

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from validation.validate_data_manifests import load_acquisition_spec, main, validate_manifest_directory


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
    assert report["acquisition_specs"]["realised"] == 0
    assert report["acquisition_specs"]["pending"] >= 1
    spec = report["acquisition_specs"]["specs"][0]
    assert spec["expected_dataset_id"] == "diii-d-163303-mdsplus"
    assert spec["manifest_path"] is None
    assert not report["errors"]


def test_validate_manifest_directory_can_require_real_acquisitions() -> None:
    report = validate_manifest_directory(
        ROOT / "validation" / "reference_data",
        require_real_acquisition=True,
    )

    assert report["status"] == "fail"
    assert report["acquisition_specs"]["pending"] >= 1
    assert any(error["error"] == "missing acquired MDSplus manifest" for error in report["errors"])


def test_validate_manifest_directory_does_not_import_numpy_for_spec_gate() -> None:
    code = f"""
import builtins
import json

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "numpy" or name.startswith("numpy."):
        raise ModuleNotFoundError("numpy intentionally blocked for manifest gate")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from validation.validate_data_manifests import validate_manifest_directory

report = validate_manifest_directory({str(ROOT / "validation" / "reference_data")!r}, verify_artifacts=False)
print(json.dumps({{"status": report["status"], "mdsplus": report["acquisition_specs"]["mdsplus"]}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"status": "pass", "mdsplus": 1}


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
            "uri": artefact.name,
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


def test_load_acquisition_spec_rejects_duplicate_keys(tmp_path) -> None:
    spec = tmp_path / "duplicate_spec.json"
    spec.write_text(
        '{"schema_version":"1.0","tree":"DIII-D","tree":"NSTX-U"}',
        encoding="utf-8",
    )

    try:
        load_acquisition_spec(spec)
    except ValueError as exc:
        assert str(exc) == "duplicate JSON key: tree"
    else:
        raise AssertionError("duplicate acquisition spec key was accepted")


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


def test_main_can_require_real_acquisition(tmp_path) -> None:
    output = tmp_path / "strict_data_manifest_report.json"

    exit_code = main(
        [
            "--root",
            str(ROOT / "validation" / "reference_data"),
            "--require-real-acquisition",
            "--output-json",
            str(output),
            "--json-out",
        ]
    )

    assert exit_code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["acquisition_specs"]["pending"] >= 1

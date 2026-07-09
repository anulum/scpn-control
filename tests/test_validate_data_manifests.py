# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Data Manifest Validation Runner Tests

from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import validation.validate_data_manifests as validator
from validation.validate_data_manifests import load_acquisition_spec, main, validate_manifest_directory


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_manifest_payload(*, dataset_id: str = "synthetic-ci") -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "dataset_id": dataset_id,
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


def _real_mdsplus_manifest_payload(*, dataset_id: str = "diii-d-163303-mdsplus") -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "dataset_id": dataset_id,
        "machine": "DIII-D",
        "shot": "163303",
        "synthetic": False,
        "source": {
            "kind": "mdsplus",
            "uri": "mdsplus://DIII-D/163303",
            "access": "facility-approved",
        },
        "retrieved_at": "2026-05-18T01:20:00Z",
        "checksum_sha256": "a" * 64,
        "licence": "facility data policy",
        "signals": [
            {
                "name": "plasma_current",
                "path": "\\IP",
                "units": "A",
                "timebase": "time_s",
            }
        ],
    }


def _acquisition_spec_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "tree": "DIII-D",
        "shot": 163303,
        "source_uri": "mdsplus://DIII-D/163303",
        "access_policy": "facility-approved",
        "licence": "facility data policy",
        "signals": [
            {
                "name": "plasma_current",
                "node": "\\IP",
                "units": "A",
                "timebase": "\\TIME",
            }
        ],
    }


def test_validate_manifest_directory_reports_repository_manifests() -> None:
    report = validate_manifest_directory(
        ROOT / "validation" / "reference_data",
        verify_artifacts=True,
    )

    assert report["status"] == "pass"
    assert report["total"] >= 3
    assert report["real"] == 0
    assert report["synthetic"] >= 5
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


def test_manifest_api_loader_fails_closed_when_spec_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *_args, **_kwargs: None)

    with pytest.raises(ImportError, match="cannot load real-data manifest contract"):
        validator._load_real_data_manifest_api()


def test_validate_manifest_directory_reports_empty_manifest_root(tmp_path: Path) -> None:
    report = validate_manifest_directory(tmp_path)

    assert report["status"] == "fail"
    assert report["errors"] == [{"path": str(tmp_path), "error": "no data manifests found"}]


def test_validate_manifest_directory_links_realised_mdsplus_spec(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifest_path = manifest_dir / "realised.manifest.json"
    manifest_path.write_text(json.dumps(_real_mdsplus_manifest_payload()), encoding="utf-8")
    spec_dir = tmp_path / "acquisition_specs"
    spec_dir.mkdir()
    spec_dir.joinpath("shot_163303_mdsplus.json").write_text(json.dumps(_acquisition_spec_payload()), encoding="utf-8")

    report = validate_manifest_directory(tmp_path, verify_artifacts=True, require_real_acquisition=True)

    assert report["status"] == "pass"
    assert report["real"] == 1
    assert report["acquisition_specs"]["realised"] == 1
    assert report["acquisition_specs"]["pending"] == 0
    assert report["acquisition_specs"]["specs"][0]["manifest_path"] == str(manifest_path)


def test_validate_manifest_directory_reports_missing_artifact_coverage(tmp_path: Path) -> None:
    diiid = tmp_path / "diiid"
    diiid.mkdir()
    missing = diiid / "unmanifested.geqdsk"
    missing.write_text("fixture", encoding="utf-8")
    manifest_dir = diiid / "manifests"
    manifest_dir.mkdir()
    manifest_dir.joinpath("synthetic.manifest.json").write_text(
        json.dumps(_synthetic_manifest_payload()),
        encoding="utf-8",
    )

    report = validate_manifest_directory(tmp_path, verify_artifacts=True)

    assert report["status"] == "fail"
    assert report["artifact_coverage"]["expected"] == 1
    assert report["artifact_coverage"]["covered"] == 0
    assert report["artifact_coverage"]["missing"] == [str(missing.resolve())]
    assert report["errors"][-1] == {"path": str(missing.resolve()), "error": "missing data manifest coverage"}


def test_validate_manifest_directory_rejects_bad_checksum(tmp_path: Path) -> None:
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


def test_validate_manifest_directory_rejects_bad_synthetic_artifact_checksum(tmp_path: Path) -> None:
    artefact = tmp_path / "synthetic_shot.npz"
    artefact.write_bytes(b"changed synthetic data")
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifest = {
        "schema_version": "1.0",
        "dataset_id": "bad-synthetic-checksum",
        "machine": "DIII-D",
        "shot": "synthetic",
        "synthetic": True,
        "source": {
            "kind": "synthetic",
            "uri": "synthetic://unit-test",
            "access": "temporary synthetic fixture",
        },
        "synthetic_generator": "tests.test_validate_data_manifests",
        "synthetic_seed": 11,
        "artifacts": [
            {
                "uri": artefact.name,
                "checksum_sha256": "0" * 64,
            }
        ],
        "signals": [
            {
                "name": "plasma_current",
                "path": "Ip_MA",
                "units": "MA",
                "timebase": "time_s",
            }
        ],
    }
    (manifest_dir / "bad_synthetic.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_manifest_directory(tmp_path, verify_artifacts=True)

    assert report["status"] == "fail"
    assert report["total"] == 1
    assert len(report["errors"]) == 1
    assert "checksum mismatch" in report["errors"][0]["error"]


def test_validate_manifest_directory_rejects_bad_acquisition_spec(tmp_path: Path) -> None:
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


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (["not", "an", "object"], "MDSplus acquisition request root must be a JSON object"),
        (
            {
                "schema_version": "2.0",
                "tree": "DIII-D",
                "shot": 163303,
                "source_uri": "mdsplus://DIII-D/163303",
                "access_policy": "facility-approved",
                "licence": "facility data policy",
                "signals": [{"name": "ip", "node": "\\IP", "units": "A", "timebase": "\\TIME"}],
            },
            "MDSplus acquisition request schema_version must be '1.0'",
        ),
        (
            {
                "schema_version": "1.0",
                "tree": "DIII-D",
                "shot": True,
                "source_uri": "mdsplus://DIII-D/163303",
                "access_policy": "facility-approved",
                "licence": "facility data policy",
                "signals": [{"name": "ip", "node": "\\IP", "units": "A", "timebase": "\\TIME"}],
            },
            "MDSplus acquisition request shot must be an integer",
        ),
        (
            {
                "schema_version": "1.0",
                "tree": "DIII-D",
                "shot": 163303,
                "source_uri": "mdsplus://DIII-D/163303",
                "access_policy": "facility-approved",
                "licence": "facility data policy",
                "signals": "not-list",
            },
            "MDSplus acquisition request requires a signals array",
        ),
        (
            {
                "schema_version": "1.0",
                "tree": "",
                "shot": 163303,
                "source_uri": "mdsplus://DIII-D/163303",
                "access_policy": "facility-approved",
                "licence": "facility data policy",
                "signals": [{"name": "ip", "node": "\\IP", "units": "A", "timebase": "\\TIME"}],
            },
            "MDSplus acquisition request requires non-empty tree",
        ),
    ],
)
def test_load_acquisition_spec_rejects_invalid_request_shapes(tmp_path: Path, payload: object, message: str) -> None:
    spec = tmp_path / "bad_spec.json"
    spec.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_acquisition_spec(spec)


@pytest.mark.parametrize(
    ("signal", "message"),
    [
        ("not-object", "MDSplus signal specification must be a JSON object"),
        ({"name": "ip", "node": "\\IP", "units": "A", "timebase": "\\TIME"}, "duplicate MDSplus signal name: ip"),
        ({"name": "ip", "node": "", "units": "A", "timebase": "\\TIME"}, "requires non-empty node"),
        ({"name": "ip", "node": "\\IP", "units": "", "timebase": "\\TIME"}, "requires non-empty units"),
        ({"name": "ip", "node": "\\IP", "units": "A", "timebase": ""}, "requires non-empty timebase"),
    ],
)
def test_load_acquisition_spec_rejects_invalid_signal_specs(tmp_path: Path, signal: object, message: str) -> None:
    payload = _acquisition_spec_payload()
    payload["signals"] = [signal, signal] if isinstance(signal, dict) and signal.get("node") else [signal]
    spec = tmp_path / "bad_signal_spec.json"
    spec.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_acquisition_spec(spec)


def test_covered_artifact_uris_accepts_real_local_archive_without_artifacts() -> None:
    manifest = SimpleNamespace(
        artifacts=(),
        synthetic=False,
        checksum_sha256="a" * 64,
        source=SimpleNamespace(kind="local_archive", uri="shot.npz"),
    )

    assert validator._covered_artifact_uris(manifest) == ["shot.npz"]


def test_resolve_manifest_uri_handles_absolute_and_missing_paths(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifests" / "m.json"
    manifest_path.parent.mkdir()
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"artifact")

    assert validator._resolve_manifest_uri(str(artifact), manifest_path, tmp_path) == artifact.resolve()
    assert validator._resolve_manifest_uri(str(tmp_path / "missing.bin"), manifest_path, tmp_path) is None
    assert validator._resolve_manifest_uri("missing-relative.bin", manifest_path, tmp_path) is None


def test_load_acquisition_spec_rejects_duplicate_keys(tmp_path: Path) -> None:
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


def test_main_writes_json_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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


def test_main_can_require_real_acquisition(tmp_path: Path) -> None:
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


def test_main_reports_errors_to_stderr(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--root", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Data manifests: fail total=0 real=0 synthetic=0 acquisition_specs=0" in captured.out
    assert "ERROR" in captured.err
    assert "no data manifests found" in captured.err

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Data Manifest Validation Runner Tests

"""Exercise real manifest declarations, local custody and public refusal behavior."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import validation.validate_data_manifests as validator
from validation.validate_data_manifests import load_acquisition_spec, main, validate_manifest_directory

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_manifest_payload(*, dataset_id: str = "synthetic-ci") -> dict[str, Any]:
    """Build explicit synthetic manifest payload declarations for this test contract."""
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
    """Build explicit real mdsplus manifest payload declarations for this test contract."""
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
    """Build explicit acquisition spec payload declarations for this test contract."""
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
    """Validate manifest directory reports repository manifests."""
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
    """Validate manifest directory can require real acquisitions."""
    report = validate_manifest_directory(
        ROOT / "validation" / "reference_data",
        require_real_acquisition=True,
    )

    assert report["status"] == "fail"
    assert report["acquisition_specs"]["pending"] >= 1
    assert any(error["error"] == "missing acquired MDSplus manifest" for error in report["errors"])


def test_validate_manifest_directory_does_not_import_numpy_for_spec_gate() -> None:
    """Validate manifest directory does not import numpy for spec gate."""
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
    """Manifest api loader fails closed when spec missing."""
    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *_args, **_kwargs: None)

    with pytest.raises(ImportError, match="cannot load real-data manifest contract"):
        validator._load_real_data_manifest_api()


def test_validate_manifest_directory_reports_empty_manifest_root(tmp_path: Path) -> None:
    """Validate manifest directory reports empty manifest root."""
    report = validate_manifest_directory(tmp_path)

    assert report["status"] == "fail"
    assert report["errors"] == [{"path": str(tmp_path), "error": "no data manifests found"}]


def test_validate_manifest_directory_links_realised_mdsplus_spec(tmp_path: Path) -> None:
    """Validate manifest directory links realised mdsplus spec."""
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifest_path = manifest_dir / "realised.manifest.json"
    payload = _real_mdsplus_manifest_payload()
    payload["signals"][0]["timebase"] = "\\TIME"
    content = b"test custody bytes; not measured facility evidence"
    (tmp_path / "shot.bin").write_bytes(content)
    payload["artifacts"] = [{"uri": "shot.bin", "checksum_sha256": sha256(content).hexdigest()}]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
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
    """Validate manifest directory reports missing artifact coverage."""
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
    """Validate manifest directory rejects bad checksum."""
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
    """Validate manifest directory rejects bad synthetic artifact checksum."""
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
    """Validate manifest directory rejects bad acquisition spec."""
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
    """Load acquisition spec rejects invalid request shapes."""
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
    """Load acquisition spec rejects invalid signal specs."""
    payload = _acquisition_spec_payload()
    payload["signals"] = [signal, signal] if isinstance(signal, dict) and signal.get("node") else [signal]
    spec = tmp_path / "bad_signal_spec.json"
    spec.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_acquisition_spec(spec)


def test_covered_artifact_uris_accepts_real_local_archive_without_artifacts() -> None:
    """Covered artifact uris accepts real local archive without artifacts."""
    manifest = SimpleNamespace(
        artifacts=(),
        synthetic=False,
        checksum_sha256="a" * 64,
        source=SimpleNamespace(kind="local_archive", uri="shot.npz"),
    )

    assert validator._covered_artifact_uris(manifest) == ["shot.npz"]


def test_resolve_manifest_uri_handles_absolute_and_missing_paths(tmp_path: Path) -> None:
    """Resolve manifest uri handles absolute and missing paths."""
    manifest_path = tmp_path / "manifests" / "m.json"
    manifest_path.parent.mkdir()
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"artifact")

    assert validator._resolve_manifest_uri(str(artifact), manifest_path, tmp_path) is None
    assert validator._resolve_manifest_uri(artifact.name, manifest_path, tmp_path) == artifact.resolve()
    assert validator._resolve_manifest_uri(str(tmp_path / "missing.bin"), manifest_path, tmp_path) is None
    assert validator._resolve_manifest_uri("missing-relative.bin", manifest_path, tmp_path) is None


def test_load_acquisition_spec_rejects_duplicate_keys(tmp_path: Path) -> None:
    """Load acquisition spec rejects duplicate keys."""
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
    """Main writes json report."""
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
    """Main can require real acquisition."""
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
    """Main reports errors to stderr."""
    exit_code = main(["--root", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Data manifests: fail total=0 real=0 synthetic=0 acquisition_specs=0" in captured.out
    assert "ERROR" in captured.err
    assert "no data manifests found" in captured.err


def _prepared_acquisition(root: Path) -> tuple[Path, Path, dict[str, Any]]:
    """Write explicit test metadata and local custody bytes, with no facility evidence claim."""
    manifests = root / "manifests"
    specs = root / "acquisition_specs"
    manifests.mkdir(parents=True)
    specs.mkdir()
    content = b"test custody only, no measured signal arrays"
    (root / "archive.bin").write_bytes(content)
    payload = _real_mdsplus_manifest_payload()
    payload["signals"][0]["timebase"] = "\\TIME"
    payload["artifacts"] = [{"uri": "archive.bin", "checksum_sha256": sha256(content).hexdigest()}]
    manifest_path = manifests / "m.manifest.json"
    manifest_path.write_text(json.dumps(payload))
    spec_path = specs / "s.json"
    spec_path.write_text(json.dumps(_acquisition_spec_payload()))
    return manifest_path, spec_path, payload


@pytest.mark.parametrize(
    "field", ["machine", "shot", "uri", "access", "licence", "name", "path", "units", "timebase", "artifacts"]
)
def test_public_directory_refuses_identity_only_acquisition(tmp_path: Path, field: str) -> None:
    """A stable dataset ID cannot hide mismatched request declarations or absent custody."""
    path, _, payload = _prepared_acquisition(tmp_path)
    if field in {"uri", "access"}:
        payload["source"][field] = "unrelated"
    elif field in {"name", "path", "units", "timebase"}:
        payload["signals"][0][field] = "unrelated"
    elif field == "artifacts":
        payload[field] = []
    else:
        payload[field] = "unrelated"
    path.write_text(json.dumps(payload))
    report = validate_manifest_directory(tmp_path, require_real_acquisition=True)
    assert report["status"] == "fail" and report["acquisition_specs"]["realised"] == 0
    assert report["acquisition_specs"]["pending"] == 1
    assert any("does not match" in e["error"] for e in report["errors"])


def test_public_directory_allows_requested_subset_but_marks_unverified_bytes(tmp_path: Path) -> None:
    """Extra signals are allowed; disabling hashes is recorded as metadata-only inspection."""
    path, _, payload = _prepared_acquisition(tmp_path)
    payload["signals"].append({"name": "extra", "path": "\\EXTRA", "units": "A", "timebase": "\\TIME"})
    path.write_text(json.dumps(payload))
    (tmp_path / "archive.bin").write_bytes(b"changed bytes")
    unchecked = validate_manifest_directory(tmp_path, verify_artifacts=False, require_real_acquisition=True)
    assert unchecked["status"] == "pass" and unchecked["acquisition_specs"]["realised"] == 1
    assert unchecked["artifact_verification"] is False
    assert validate_manifest_directory(tmp_path, verify_artifacts=True)["status"] == "fail"


@pytest.mark.parametrize("kind", ["manifest", "spec"])
def test_public_directory_refuses_duplicate_dataset_identities(tmp_path: Path, kind: str) -> None:
    """An additional file with an identical dataset/spec identity cannot replace the first."""
    path, spec, _ = _prepared_acquisition(tmp_path)
    selected = path if kind == "manifest" else spec
    copy = selected.parent / ("z.manifest.json" if kind == "manifest" else "z.json")
    copy.write_bytes(selected.read_bytes())
    report = validate_manifest_directory(tmp_path)
    assert report["status"] == "fail"
    assert any("duplicate" in e["error"] for e in report["errors"])


@pytest.mark.parametrize("kind", ["manifest", "spec"])
@pytest.mark.parametrize("contents", [b"\xff", b"[" * 2000 + b"]" * 2000])
def test_public_directory_reports_utf8_and_depth_findings(tmp_path: Path, kind: str, contents: bytes) -> None:
    """Actual decoder failures are report findings rather than uncaught directory crashes."""
    path, spec, _ = _prepared_acquisition(tmp_path)
    selected = path if kind == "manifest" else spec
    selected.write_bytes(contents)
    report = validate_manifest_directory(tmp_path)
    assert report["status"] == "fail" and report["errors"][0]["path"] == str(selected)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e9999"])
def test_public_spec_refuses_nonfinite_unknown_metadata(tmp_path: Path, token: str) -> None:
    """All-depth finite decoding applies even to uninterpreted acquisition metadata."""
    _, spec, _ = _prepared_acquisition(tmp_path)
    text = spec.read_text()
    spec.write_text(text[:-1] + ', "extra": {"value": ' + token + "}}")
    report = validate_manifest_directory(tmp_path)
    assert report["status"] == "fail" and "nonfinite" in report["errors"][0]["error"]


def test_public_spec_accepts_finite_unknown_metadata(tmp_path: Path) -> None:
    """An ordinary finite decimal is accepted without assigning acquisition meaning."""
    _, spec, _ = _prepared_acquisition(tmp_path)
    payload = json.loads(spec.read_text())
    payload["extra"] = 1.25
    spec.write_text(json.dumps(payload))
    report = validate_manifest_directory(tmp_path, require_real_acquisition=True)
    assert report["status"] == "pass" and report["acquisition_specs"]["realised"] == 1


@pytest.mark.parametrize("policy", ["verify_artifacts", "require_real_acquisition"])
def test_public_directory_requires_boolean_policies(tmp_path: Path, policy: str) -> None:
    """Truthiness cannot silently select a directory verification policy."""
    report = validate_manifest_directory(tmp_path, **{policy: cast(Any, 1)})
    assert report["status"] == "fail" and "booleans" in report["errors"][0]["error"]


@pytest.mark.parametrize(
    "kind", ["missing", "file", "null", "loop", "escaping-manifest", "escaping-spec", "escaping-artifact"]
)
def test_public_directory_refuses_unresolvable_or_escaping_discovery(tmp_path: Path, kind: str) -> None:
    """Actual path/containment failures prevent incomplete discovered graphs from passing."""
    root = tmp_path / "root"
    if kind == "missing":
        pass
    elif kind == "file":
        root.write_bytes(b"not a directory")
    elif kind == "null":
        root = Path("bad\x00root")
    elif kind == "loop":
        root.symlink_to(root)
    else:
        path, spec, _ = _prepared_acquisition(root)
        outside = tmp_path / "outside"
        outside.write_bytes(b"outside")
        if kind == "escaping-artifact":
            selected = root / "outside.geqdsk"
        else:
            selected = path if kind == "escaping-manifest" else spec
            selected.unlink()
        selected.symlink_to(outside)
    report = validate_manifest_directory(root)
    assert report["status"] == "fail" and "cannot scan" in report["errors"][0]["error"]


def test_public_coverage_uses_the_same_actual_file_as_checksum_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unrelated cwd filename cannot shadow the verified evidence-root artifact."""
    root = tmp_path / "root"
    path, _, payload = _prepared_acquisition(root)
    content = b"coverage metadata fixture"
    local = root / "shot.geqdsk"
    local.write_bytes(content)
    payload["artifacts"] = [{"uri": local.name, "checksum_sha256": sha256(content).hexdigest()}]
    path.write_text(json.dumps(payload))
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / local.name).write_bytes(b"unrelated")
    monkeypatch.chdir(foreign)
    report = validate_manifest_directory(root, require_real_acquisition=True)
    assert report["status"] == "pass" and report["artifact_coverage"]["covered"] == 1


@pytest.mark.parametrize("kind", ["directory", "parent-file", "null"])
def test_actual_standalone_cli_reports_output_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], kind: str
) -> None:
    """Actual output filesystem refusals return structured FAIL without a traceback."""
    root = tmp_path / "root"
    _prepared_acquisition(root)
    output = tmp_path / "output"
    if kind == "directory":
        output.mkdir()
    elif kind == "parent-file":
        output.write_bytes(b"parent")
        output = output / "report.json"
    else:
        output = Path("bad\x00output")
    assert main(["--root", str(root), "--json-out", "--output-json", str(output)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "fail" and "cannot write report" in report["errors"][-1]["error"]


def test_actual_standalone_stdlib_cli_requires_real_acquisition(tmp_path: Path) -> None:
    """The defining standalone entry point runs without site packages or numerical imports."""
    root = tmp_path / "root"
    _prepared_acquisition(root)
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "validation/validate_data_manifests.py"),
            "--root",
            str(root),
            "--json-out",
            "--require-real-acquisition",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["acquisition_specs"]["realised"] == 1


def test_public_metadata_only_unresolved_uri_cannot_cover_an_expected_file(tmp_path: Path) -> None:
    """A nonexistent artifact spelling supplies no coverage when hashing is disabled."""
    path, _, payload = _prepared_acquisition(tmp_path)
    (tmp_path / "required.geqdsk").write_bytes(b"coverage test bytes")
    payload["artifacts"][0]["uri"] = "absent.bin"
    path.write_text(json.dumps(payload))
    report = validate_manifest_directory(tmp_path, verify_artifacts=False)
    assert report["status"] == "fail" and report["artifact_coverage"]["covered"] == 0
    assert report["artifact_coverage"]["missing"] == [str(tmp_path / "required.geqdsk")]

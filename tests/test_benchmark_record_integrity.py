# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Benchmark latest integrity contracts.
"""Verify persisted public campaign reports and their actual immutable payloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scpn_control.benchmark_record_integrity import load_verified_latest
from scpn_control.benchmark_records import BenchmarkOutput, BenchmarkRun


def _begin(records: Path, output: Path, campaign_id: str) -> BenchmarkRun:
    """Reserve one actual output through the public campaign API."""
    return BenchmarkRun.begin(
        repository_root=records.parent,
        records_root=records,
        family="controller-latency",
        outputs=[BenchmarkOutput("report", output)],
        command=["public-test-producer"],
        campaign_id=campaign_id,
    )


def _successful_run(root: Path) -> BenchmarkRun:
    """Create a persisted nonempty result using the real record owner."""
    output = root / "report.json"
    run = _begin(root / "records", output, "validated")
    output.write_bytes(b"original")
    run.finish(exit_code=0)
    return run


def _persist_changed_manifest(run: BenchmarkRun, payload: object, *, refresh_payload: bool = True) -> None:
    """Give a changed carrier a valid outer byte binding to exercise deeper refusal."""
    if isinstance(payload, dict) and refresh_payload:
        unsigned = dict(payload)
        unsigned.pop("payload_sha256", None)
        payload["payload_sha256"] = hashlib.sha256(
            (json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")
        ).hexdigest()
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (run.run_directory / "manifest.json").write_bytes(data)
    index = run.records_root / "latest/controller-latency.json"
    latest = json.loads(index.read_text(encoding="utf-8"))
    latest["manifest_sha256"] = hashlib.sha256(data).hexdigest()
    index.write_text(json.dumps(latest), encoding="utf-8")


def test_latest_uses_successful_publication_order_for_disjoint_runs(tmp_path: Path) -> None:
    """Later completion of an earlier-started run is the documented latest result."""
    records = tmp_path / "records"
    first_output, second_output = tmp_path / "first.json", tmp_path / "second.json"
    first = _begin(records, first_output, "started-first")
    second = _begin(records, second_output, "started-second")
    second_output.write_bytes(b"second")
    second.finish(exit_code=0)
    first_output.write_bytes(b"first")
    first.finish(exit_code=0)
    latest, manifest = load_verified_latest(records, "controller-latency")
    assert latest["campaign_id"] == manifest["campaign_id"] == "started-first"


@pytest.mark.parametrize("change", ["content", "size", "missing", "kind"])
def test_latest_refuses_changed_immutable_artifact(tmp_path: Path, change: str) -> None:
    """Manifest-only integrity cannot admit changed, removed or replaced payloads."""
    output = tmp_path / "report.json"
    run = _begin(tmp_path / "records", output, "immutable-check")
    output.write_bytes(b"original")
    manifest = json.loads(run.finish(exit_code=0).read_text(encoding="utf-8"))
    artifact = run.run_directory / manifest["artifacts"][0]["immutable_path_in_run"]
    if change == "content":
        artifact.write_bytes(b"changed!")
    elif change == "size":
        artifact.write_bytes(b"short")
    else:
        artifact.unlink()
        if change == "kind":
            artifact.mkdir()
    with pytest.raises(ValueError, match="artifact"):
        load_verified_latest(run.records_root, run.family)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "unsupported"),
        ("benchmark_family", "another-family"),
        ("status", "failed"),
        ("exit_code", 1),
        ("exit_code", False),
        ("campaign_id", "another-run"),
        ("missing_output_roles", ["report"]),
        ("empty_output_roles", ["report"]),
        ("artifacts", []),
        ("artifacts", [None]),
    ],
)
def test_latest_refuses_inadmissible_rebound_manifest(tmp_path: Path, field: str, value: object) -> None:
    """Consistent outer checksums cannot admit another family, failure or incomplete carrier."""
    run = _successful_run(tmp_path)
    manifest = json.loads((run.run_directory / "manifest.json").read_text(encoding="utf-8"))
    manifest[field] = value
    _persist_changed_manifest(run, manifest)
    with pytest.raises(ValueError):
        load_verified_latest(run.records_root, run.family)


@pytest.mark.parametrize(
    "field,value",
    [
        ("role", ""),
        ("role", 1),
        ("immutable_path_in_run", None),
        ("immutable_path_in_run", "artifacts\\report.json"),
        ("immutable_path_in_run", "artifacts/../report.json"),
        ("immutable_path_in_run", "report.json"),
        ("kind", "unsupported"),
        ("size_bytes", True),
        ("size_bytes", 0),
        ("size_bytes", 9),
        ("sha256", "0" * 64),
        ("digest_algorithm", "unsupported"),
    ],
)
def test_latest_refuses_inconsistent_artifact_declarations(tmp_path: Path, field: str, value: object) -> None:
    """Actual persisted declarations must agree with the immutable bytes and path contract."""
    run = _successful_run(tmp_path)
    manifest = json.loads((run.run_directory / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"][0][field] = value
    _persist_changed_manifest(run, manifest)
    with pytest.raises(ValueError):
        load_verified_latest(run.records_root, run.family)


def test_latest_refuses_manifest_payload_digest_mismatch(tmp_path: Path) -> None:
    """The outer index checksum cannot substitute for the manifest's declared payload binding."""
    run = _successful_run(tmp_path)
    manifest = json.loads((run.run_directory / "manifest.json").read_text(encoding="utf-8"))
    manifest["payload_sha256"] = "0" * 64
    _persist_changed_manifest(run, manifest, refresh_payload=False)
    with pytest.raises(ValueError, match="payload digest"):
        load_verified_latest(run.records_root, run.family)


def test_latest_refuses_index_artifact_digest_mismatch(tmp_path: Path) -> None:
    """Every role digest in the selecting index must match the verified artifact set."""
    run = _successful_run(tmp_path)
    index = run.records_root / "latest/controller-latency.json"
    latest = json.loads(index.read_text(encoding="utf-8"))
    latest["artifact_sha256"] = {"report": "0" * 64}
    index.write_text(json.dumps(latest), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact digests"):
        load_verified_latest(run.records_root, run.family)


@pytest.mark.parametrize("root", ["latest", "manifest"])
def test_latest_refuses_nonobject_carriers(tmp_path: Path, root: str) -> None:
    """A physically replaced root array produces a declared refusal rather than AttributeError."""
    run = _successful_run(tmp_path)
    if root == "latest":
        (run.records_root / "latest/controller-latency.json").write_text("[]", encoding="utf-8")
    else:
        _persist_changed_manifest(run, [])
    with pytest.raises(ValueError, match="must be an object"):
        load_verified_latest(run.records_root, run.family)


def test_latest_refuses_duplicate_roles_and_unbound_legacy_path(tmp_path: Path) -> None:
    """Legacy and current carriers cannot hide duplicate roles or absent path observations."""
    run = _successful_run(tmp_path)
    original = (run.run_directory / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(original)
    manifest["artifacts"].append(dict(manifest["artifacts"][0]))
    _persist_changed_manifest(run, manifest)
    with pytest.raises(ValueError, match="duplicate"):
        load_verified_latest(run.records_root, run.family)
    manifest = json.loads(original)
    del manifest["artifacts"][0]["immutable_path_in_run"]
    del manifest["artifacts"][0]["immutable_path"]
    _persist_changed_manifest(run, manifest)
    with pytest.raises(ValueError, match="path is missing"):
        load_verified_latest(run.records_root, run.family)


@pytest.mark.parametrize("windows_path", [False, True])
def test_latest_accepts_legacy_file_path_observation_without_rewriting_it(tmp_path: Path, windows_path: bool) -> None:
    """An older wire spelling still verifies its actual artifact under the immutable run."""
    run = _successful_run(tmp_path)
    manifest = json.loads((run.run_directory / "manifest.json").read_text(encoding="utf-8"))
    current = run.run_directory / manifest["artifacts"][0]["immutable_path_in_run"]
    legacy = run.run_directory / "artifacts/report.json"
    current.rename(legacy)
    manifest["artifacts"][0]["immutable_path"] = str(legacy)
    del manifest["artifacts"][0]["immutable_path_in_run"]
    del manifest["artifacts"][0]["digest_algorithm"]
    if windows_path:
        manifest["artifacts"][0]["immutable_path"] = r"C:\historical-external-custody\artifacts\report.json"
    _persist_changed_manifest(run, manifest)
    before = (run.run_directory / "manifest.json").read_bytes()
    assert load_verified_latest(run.records_root, run.family)[1]["campaign_id"] == run.campaign_id
    assert (run.run_directory / "manifest.json").read_bytes() == before


def test_latest_verifies_directory_structure_and_legacy_algorithm(tmp_path: Path) -> None:
    """Directory receipts name their digest algorithm, retaining the old file-only wire explicitly."""
    from scpn_control.benchmark_artifacts import LEGACY_DIRECTORY_DIGEST_ALGORITHM, sha256_path

    output = tmp_path / "result"
    run = _begin(tmp_path / "records", output, "directory-result")
    output.mkdir()
    (output / "data").write_bytes(b"result")
    (output / "empty").mkdir()
    run.finish(exit_code=0)
    assert load_verified_latest(run.records_root, run.family)[1]["artifacts"][0]["kind"] == "directory"
    manifest = json.loads((run.run_directory / "manifest.json").read_text(encoding="utf-8"))
    artifact = run.run_directory / manifest["artifacts"][0]["immutable_path_in_run"]
    (artifact / "empty").rmdir()
    with pytest.raises(ValueError, match="artifact digest"):
        load_verified_latest(run.records_root, run.family)
    manifest = json.loads((run.run_directory / "manifest.json").read_text(encoding="utf-8"))
    legacy = run.run_directory / "artifacts/report"
    artifact.rename(legacy)
    artifact = legacy
    manifest["artifacts"][0]["immutable_path"] = str(legacy)
    del manifest["artifacts"][0]["immutable_path_in_run"]
    del manifest["artifacts"][0]["digest_algorithm"]
    manifest["artifacts"][0]["sha256"] = sha256_path(artifact, directory_algorithm=LEGACY_DIRECTORY_DIGEST_ALGORITHM)
    _persist_changed_manifest(run, manifest)
    index = run.records_root / "latest/controller-latency.json"
    latest = json.loads(index.read_text(encoding="utf-8"))
    latest["artifact_sha256"] = {"report": manifest["artifacts"][0]["sha256"]}
    index.write_text(json.dumps(latest), encoding="utf-8")
    assert (
        load_verified_latest(run.records_root, run.family)[1]["artifacts"][0]["sha256"]
        == manifest["artifacts"][0]["sha256"]
    )


def test_latest_refuses_actual_symlink_substitution(tmp_path: Path) -> None:
    """A byte-identical external replacement cannot stand in for an immutable stored node."""
    run = _successful_run(tmp_path)
    manifest = json.loads((run.run_directory / "manifest.json").read_text(encoding="utf-8"))
    artifact = run.run_directory / manifest["artifacts"][0]["immutable_path_in_run"]
    outside = tmp_path / "outside"
    artifact.rename(outside)
    artifact.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        load_verified_latest(run.records_root, run.family)
    assert outside.read_bytes() == b"original"


def test_latest_refuses_invalid_family_before_filesystem_access(tmp_path: Path) -> None:
    """Invalid family spelling cannot select a path outside the named registry."""
    with pytest.raises(ValueError, match="family identifier"):
        load_verified_latest(tmp_path, "../other")
    assert list(tmp_path.iterdir()) == []


def test_latest_refuses_nonstring_directory_algorithm(tmp_path: Path) -> None:
    """A persisted numeric protocol tag cannot acquire directory integrity semantics."""
    output = tmp_path / "result"
    run = _begin(tmp_path / "records", output, "directory-protocol")
    output.mkdir()
    (output / "data").write_bytes(b"result")
    run.finish(exit_code=0)
    manifest = json.loads((run.run_directory / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"][0]["digest_algorithm"] = 1
    _persist_changed_manifest(run, manifest)
    with pytest.raises(ValueError, match="digest algorithm"):
        load_verified_latest(run.records_root, run.family)


@pytest.mark.parametrize("directory_archive", [False, True])
def test_latest_preserves_explicit_historical_predecessor_archive_paths(
    tmp_path: Path, directory_archive: bool
) -> None:
    """Read an old role-named archive declaration without moving its original bytes."""
    import shutil

    from scpn_control.benchmark_artifacts import LEGACY_DIRECTORY_DIGEST_ALGORITHM, sha256_path

    run = _successful_run(tmp_path)
    manifest_path = run.run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    source = root / (
        "validation/reference_data/gk_species" if directory_archive else "validation/reports/kinetic_efit_claims.json"
    )
    digest = sha256_path(source, directory_algorithm=LEGACY_DIRECTORY_DIGEST_ALGORITHM)
    archive = run.records_root / "legacy" / digest / "report.json"
    archive.parent.mkdir(parents=True)
    if directory_archive:
        shutil.copytree(source, archive)
    else:
        shutil.copyfile(source, archive)
    manifest["legacy_inputs"] = [{"role": "report", "archived_path": str(archive), "sha256": digest}]
    _persist_changed_manifest(run, manifest)
    before = manifest_path.read_bytes()
    _, returned = load_verified_latest(run.records_root, run.family)
    assert returned["legacy_inputs"] == manifest["legacy_inputs"]
    observed = Path(returned["legacy_inputs"][0]["archived_path"])
    assert observed == archive and observed.is_dir() == directory_archive
    assert sha256_path(observed, directory_algorithm=LEGACY_DIRECTORY_DIGEST_ALGORITHM) == digest
    assert sha256_path(source, directory_algorithm=LEGACY_DIRECTORY_DIGEST_ALGORITHM) == digest
    assert manifest_path.read_bytes() == before

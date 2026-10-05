# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Public Data Acquisition Manifest Tests

"""Exercise public acquisition metadata, local byte custody and real CLI refusals."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from copy import deepcopy
from hashlib import md5, sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import pytest

from validation.validate_public_data_acquisition import (
    SCHEMA_VERSION,
    PublicDataAcquisitionError,
    load_public_data_acquisition_manifest,
    main,
    validate_public_data_acquisition_directory,
    validate_public_data_acquisition_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
QLKNN_ROOT = ROOT / "validation" / "reference_data" / "qlknn"
QLKNN10D_MANIFEST = QLKNN_ROOT / "zenodo_3497066" / "files_manifest.json"


def _artifact_root() -> Path:
    """Return the ignored in-repository scratch directory, creating it in a clean checkout."""
    root = ROOT / "artifacts"
    root.mkdir(exist_ok=True)
    return root


def _payload() -> dict[str, Any]:
    """Build explicit valid test metadata bound to the current owned test source bytes."""
    local = ROOT / "tests" / "test_public_data_acquisition.py"
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "zenodo",
        "doi": "10.5281/zenodo.3497066",
        "title": "QLKNN10D training set",
        "license": "cc-by-4.0",
        "record_sha256": "a" * 64,
        "large_numeric_files_downloaded": False,
        "large_numeric_files_policy": "deferred: pull multi-GB arrays on the storage target",
        "files": [
            {
                "key": "README.md",
                "size_bytes": local.stat().st_size,
                "checksum": "md5:" + md5(local.read_bytes(), usedforsecurity=False).hexdigest(),
                "download_url": "https://zenodo.org/api/records/3497066/files/README.md/content",
                "local_path": "tests/test_public_data_acquisition.py",
                "local_sha256": sha256(local.read_bytes()).hexdigest(),
            },
            {
                "key": "Zeffcombo_prepared.nc.1",
                "size_bytes": 12584525386,
                "checksum": "md5:ad5b69e2e670f33c48b5e8242e4c0196",
                "download_url": "https://zenodo.org/api/records/3497066/files/Zeffcombo_prepared.nc.1/content",
            },
        ],
    }


def test_public_qlknn_acquisition_manifests_validate() -> None:
    """Public qlknn acquisition manifests validate."""
    report = validate_public_data_acquisition_directory(QLKNN_ROOT)

    assert report["status"] == "pass"
    assert report["records"] == 3
    assert report["local_files"] == 0
    assert report["deferred_files"] == 52
    assert report["deferred_bytes"] > 300_000_000_000


def test_load_public_data_acquisition_manifest_binds_record_sha() -> None:
    """Load public data acquisition manifest binds record sha."""
    manifest = load_public_data_acquisition_manifest(QLKNN10D_MANIFEST)

    assert manifest.doi == "10.5281/zenodo.3497066"
    assert manifest.record_sha256 == "77d168ed8a7fb84f7d60308a051943b9a00ebbd486812c28e9063e8e41b8b9ae"
    assert manifest.local_files == ()


def test_manifest_rejects_tampered_record_sha(tmp_path: Path) -> None:
    """Manifest rejects tampered record sha."""
    payload = _payload()
    path = tmp_path / "files_manifest.json"
    path.with_name("record.json").write_bytes(b"zenodo record bytes")
    payload["record_sha256"] = "b" * 64

    with pytest.raises(PublicDataAcquisitionError, match="record_sha256 does not match"):
        validate_public_data_acquisition_manifest(payload, manifest_path=path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update({"schema_version": "legacy"}), "unsupported public-data acquisition"),
        (lambda payload: payload.update({"source": "mirror"}), "source must be zenodo"),
        (lambda payload: payload.update({"doi": "10.0000/not-zenodo"}), "DOI must identify a Zenodo"),
        (lambda payload: payload.update({"large_numeric_files_downloaded": "false"}), "must be a boolean"),
        (lambda payload: payload.update({"files": []}), "files must be a non-empty array"),
    ],
)
def test_manifest_rejects_malformed_top_level_metadata(
    mutation: Callable[[dict[str, Any]], object], message: str
) -> None:
    """Manifest rejects malformed top level metadata."""
    payload = _payload()
    mutation(payload)

    with pytest.raises(PublicDataAcquisitionError, match=message):
        validate_public_data_acquisition_manifest(payload)


def test_manifest_rejects_deferred_files_without_policy() -> None:
    """Manifest rejects deferred files without policy."""
    payload = _payload()
    payload["large_numeric_files_policy"] = "pull later"

    with pytest.raises(PublicDataAcquisitionError, match="deferred large numeric files"):
        validate_public_data_acquisition_manifest(payload)


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"key": "../escape.nc"}, "safe relative file name"),
        ({"size_bytes": 0}, "positive integer"),
        ({"checksum": "sha256:" + "a" * 64}, "md5:<32 lowercase hex>"),
        ({"download_url": "http://zenodo.org/api/records/3497066/files/README.md/content"}, "https://zenodo.org"),
        ({"download_url": "https://example.invalid/README.md"}, "https://zenodo.org"),
    ],
)
def test_manifest_rejects_unsafe_remote_file_entries(patch: dict[str, Any], message: str) -> None:
    """Manifest rejects unsafe remote file entries."""
    payload = _payload()
    file_payload = deepcopy(payload["files"])[0]
    assert isinstance(file_payload, dict)
    file_payload.update(patch)
    payload["files"] = [file_payload]

    with pytest.raises(PublicDataAcquisitionError, match=message):
        validate_public_data_acquisition_manifest(payload)


def test_manifest_rejects_local_sha_mismatch() -> None:
    """Manifest rejects local sha mismatch."""
    payload = _payload()
    file_payload = deepcopy(payload["files"])[0]
    assert isinstance(file_payload, dict)
    file_payload["local_sha256"] = "c" * 64
    payload["files"] = [file_payload]

    with pytest.raises(PublicDataAcquisitionError, match="local_sha256 does not match"):
        validate_public_data_acquisition_manifest(payload)


def test_manifest_rejects_local_path_traversal() -> None:
    """Manifest rejects local path traversal."""
    payload = _payload()
    file_payload = deepcopy(payload["files"])[0]
    assert isinstance(file_payload, dict)
    file_payload["local_path"] = "../README.md"
    payload["files"] = [file_payload]

    with pytest.raises(PublicDataAcquisitionError, match="local_path must stay"):
        validate_public_data_acquisition_manifest(payload)


def test_directory_report_fails_without_public_manifests(tmp_path: Path) -> None:
    """Directory report fails without public manifests."""
    report = validate_public_data_acquisition_directory(tmp_path)

    assert report["status"] == "fail"
    assert report["errors"] == [{"path": str(tmp_path), "error": "no public acquisition manifests found"}]


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    """Loader rejects duplicate json keys."""
    path = tmp_path / "files_manifest.json"
    path.write_text('{"schema_version":"x","schema_version":"y"}', encoding="utf-8")

    with pytest.raises(PublicDataAcquisitionError, match="duplicate JSON key: schema_version"):
        load_public_data_acquisition_manifest(path)


def _write_manifest(root: Path, payload: dict[str, Any] | None = None) -> Path:
    """Write explicit deferred metadata or selected caller metadata to a real file."""
    data = _payload() if payload is None else payload
    if payload is None:
        data["files"] = [data["files"][1]]
    path = root / "files_manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "patch,message",
    [
        ({"doi": "10.5281/zenodo.foo"}, "DOI"),
        ({"doi": "10.5281/zenodo.0"}, "DOI"),
        ({"record_sha256": "bad"}, "SHA-256"),
        ({"title": ""}, "non-empty"),
        ({"large_numeric_files_downloaded": True}, "cannot be true"),
        ({"files": [None]}, "must be an object"),
    ],
)
def test_public_mapping_domain_refusals(patch: dict[str, Any], message: str) -> None:
    """Invalid declaration domains fail through the actual public mapping API."""
    payload = _payload()
    payload.update(patch)
    with pytest.raises(PublicDataAcquisitionError, match=message):
        validate_public_data_acquisition_manifest(payload)


@pytest.mark.parametrize(
    "patch,message",
    [
        ({"key": "C:\\escape"}, "safe relative"),
        ({"key": "\\rooted"}, "safe relative"),
        ({"key": "..\\escape"}, "safe relative"),
        ({"key": "."}, "safe relative"),
        ({"key": "bad\x00key"}, "safe relative"),
        ({"size_bytes": True}, "positive integer"),
        ({"download_url": "https://zenodo.org/api/records/1/files/README.md/content"}, "match"),
        ({"download_url": "https://zenodo.org/api/records/3497066/files/wrong/content"}, "match"),
        ({"download_url": "https://zenodo.org/not-an-api"}, "record file URL"),
        ({"download_url": "https://zenodo.org/api/records/3497066/files/bad%ZZ/content"}, "malformed escapes"),
        ({"download_url": "https://zenodo.org/api/records/3497066/files/x?query/content"}, "query"),
        ({"local_path": None}, "local_path"),
        ({"local_sha256": None}, "local_sha256"),
        ({"local_path": "C:\\escape"}, "must stay"),
        ({"local_path": "bad\x00path"}, "must stay"),
        ({"local_path": "definitely-missing-file"}, "does not exist"),
        ({"size_bytes": 1}, "size_bytes does not match"),
        ({"checksum": "md5:" + "0" * 32}, "local file MD5"),
    ],
)
def test_public_file_custody_refusals(patch: dict[str, Any], message: str) -> None:
    """Real path/digest/identity and advertised byte inconsistencies refuse admission."""
    payload = _payload()
    payload["files"] = [dict(payload["files"][0], **patch)]
    with pytest.raises(PublicDataAcquisitionError, match=message):
        validate_public_data_acquisition_manifest(payload)


def test_public_duplicate_file_keys_are_not_double_counted() -> None:
    """Duplicate advertised identities fail before an accepted file count is produced."""
    payload = _payload()
    payload["files"] = [payload["files"][1], payload["files"][1]]
    with pytest.raises(PublicDataAcquisitionError, match="duplicate public file key"):
        validate_public_data_acquisition_manifest(payload)


def test_public_percent_encoded_key_linkage_accepts_a_matching_identity() -> None:
    """Valid single URL decoding preserves an advertised filename containing a space."""
    payload = _payload()
    entry = payload["files"][1]
    entry.update(key="file name.nc", download_url="https://zenodo.org/api/records/3497066/files/file%20name.nc/content")
    payload["files"] = [entry]
    manifest = validate_public_data_acquisition_manifest(payload)
    assert manifest.deferred_files[0].key == "file name.nc"


def test_public_optional_record_digest_is_checked_without_decoding(tmp_path: Path) -> None:
    """Optional record custody hashes exact bytes without inventing remote authentication."""
    content = b"record test bytes; not a fetched Zenodo record"
    (tmp_path / "record.json").write_bytes(content)
    payload = _payload()
    payload["record_sha256"] = sha256(content).hexdigest()
    payload["files"] = [payload["files"][1]]
    path = _write_manifest(tmp_path, payload)
    manifest = load_public_data_acquisition_manifest(path)
    assert manifest.record_sha256 == sha256(content).hexdigest()


def test_public_local_stream_checks_full_relative_adjacent_fallback() -> None:
    """Actual contained mirrors bind a multi-chunk stream and full adjacent relative path."""
    with TemporaryDirectory(prefix="public-acquisition-", dir=_artifact_root()) as directory:
        root = Path(directory)
        (root / "nested").mkdir()
        content = b"test byte custody\n" * 70_000
        local = root / "nested/owned.bin"
        local.write_bytes(content)
        payload = _payload()
        payload["large_numeric_files_downloaded"] = True
        payload["files"] = [
            dict(
                payload["files"][0],
                local_path="nested/owned.bin",
                local_sha256=sha256(content).hexdigest(),
                size_bytes=len(content),
                checksum="md5:" + md5(content, usedforsecurity=False).hexdigest(),
            )
        ]
        path = _write_manifest(root, payload)
        assert len(load_public_data_acquisition_manifest(path).local_files) == 1
        # The old basename fallback would incorrectly admit this obstruction.
        local.rename(root / "owned.bin")
        with pytest.raises(PublicDataAcquisitionError, match="does not exist"):
            load_public_data_acquisition_manifest(path)


@pytest.mark.parametrize("kind", ["root-escape", "fallback-escape", "record-escape", "loop"])
def test_public_containment_refusals_use_real_symlinks(tmp_path: Path, kind: str) -> None:
    """Resolver and optional record refusals exercise actual escaping/looping paths."""
    with TemporaryDirectory(prefix="public-acquisition-", dir=_artifact_root()) as directory:
        root = Path(directory)
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"outside bytes")
        payload = _payload()
        if kind == "record-escape":
            (root / "record.json").symlink_to(outside)
        else:
            local = root / "owned.bin"
            local.symlink_to(local if kind == "loop" else outside)
            spelling = str(local.relative_to(ROOT)) if kind in {"root-escape", "loop"} else local.name
            payload["files"][0]["local_path"] = spelling
        path = _write_manifest(root, payload)
        with pytest.raises(PublicDataAcquisitionError):
            load_public_data_acquisition_manifest(path)


@pytest.mark.parametrize("kind", ["utf8", "json", "depth", "array", "nonfinite", "overflow"])
def test_public_json_decoder_refusals(tmp_path: Path, kind: str) -> None:
    """Malformed and deep/nonfinite real JSON files raise the authored loader error."""
    path = _write_manifest(tmp_path)
    content = {"utf8": b"\xff", "json": b"{", "depth": b"[" * 10_000 + b"]" * 10_000, "array": b"[]"}.get(kind)
    if content is None:
        token = "NaN" if kind == "nonfinite" else "1e9999"
        content = (path.read_text()[:-1] + ', "extra": {"value": ' + token + "}}").encode()
    path.write_bytes(content)
    with pytest.raises(PublicDataAcquisitionError):
        load_public_data_acquisition_manifest(path)
    assert validate_public_data_acquisition_directory(tmp_path)["status"] == "fail"


def test_public_finite_extra_metadata_is_uninterpreted(tmp_path: Path) -> None:
    """Finite extra numbers pass decoding without becoming scientific evidence."""
    payload = _payload()
    payload["files"] = [payload["files"][1]]
    payload["extra"] = 1.25
    assert load_public_data_acquisition_manifest(_write_manifest(tmp_path, payload)).doi == payload["doi"]


@pytest.mark.parametrize("kind", ["missing", "file", "null", "loop", "escape"])
def test_public_scan_path_refusals(tmp_path: Path, kind: str) -> None:
    """Directory inspection refuses unsupported roots and escaping discovered manifests."""
    root = tmp_path / "root"
    if kind == "file":
        root.write_bytes(b"file")
    elif kind == "null":
        root = Path("bad\x00root")
    elif kind == "loop":
        root.symlink_to(root)
    elif kind == "escape":
        root.mkdir()
        path = _write_manifest(tmp_path)
        (root / path.name).symlink_to(path)
    report = validate_public_data_acquisition_directory(root)
    assert report["status"] == "fail" and "cannot scan" in report["errors"][0]["error"]


@pytest.mark.parametrize("output_kind", ["good", "directory", "parent-file", "null"])
def test_actual_cli_json_output_custody(tmp_path: Path, capsys: pytest.CaptureFixture[str], output_kind: str) -> None:
    """Standalone JSON reports preserve PASS or actual supported output refusal findings."""
    root = tmp_path / "root"
    root.mkdir()
    _write_manifest(root)
    output = tmp_path / "output"
    if output_kind == "directory":
        output.mkdir()
    elif output_kind == "parent-file":
        output.write_bytes(b"parent")
        output = output / "report.json"
    elif output_kind == "null":
        output = Path("bad\x00out")
    code = main(["--root", str(root), "--output-json", str(output), "--json-out"])
    report = json.loads(capsys.readouterr().out)
    assert code == (0 if output_kind == "good" else 1)
    assert report["status"] == ("pass" if output_kind == "good" else "fail")
    if output_kind == "good":
        assert json.loads(output.read_text()) == report
    else:
        assert "cannot write" in report["errors"][-1]["error"]


def test_actual_cli_text_and_stdlib_entrypoints(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Normal and no-site standalone paths inspect real metadata without runtime dependencies."""
    assert main([]) == 0
    assert "records=3" in capsys.readouterr().out
    assert main(["--root", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "no public acquisition" in captured.err and "fail" in captured.out
    for prefix in [[], ["-S"]]:
        result = subprocess.run(
            [sys.executable, *prefix, str(ROOT / "validation/validate_public_data_acquisition.py"), "--json-out"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["deferred_files"] == 52


def test_public_mapping_api_refuses_nonobject_root() -> None:
    """Runtime-invalid mapping roots raise the authored domain error before field access."""
    with pytest.raises(PublicDataAcquisitionError, match="root must be an object"):
        validate_public_data_acquisition_manifest(cast(Any, []))


@pytest.mark.parametrize("kind", ["directory", "dangling", "loop"])
def test_public_optional_record_must_be_a_regular_file_when_present(tmp_path: Path, kind: str) -> None:
    """Present invalid record paths cannot silently be treated as absent declarations."""
    record = tmp_path / "record.json"
    if kind == "directory":
        record.mkdir()
    else:
        record.symlink_to(record if kind == "loop" else tmp_path / "missing")
    path = _write_manifest(tmp_path)
    with pytest.raises(PublicDataAcquisitionError, match="regular file"):
        load_public_data_acquisition_manifest(path)

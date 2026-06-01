# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Public Data Acquisition Manifest Tests

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from validation.validate_public_data_acquisition import (
    PublicDataAcquisitionError,
    SCHEMA_VERSION,
    load_public_data_acquisition_manifest,
    validate_public_data_acquisition_directory,
    validate_public_data_acquisition_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
QLKNN_ROOT = ROOT / "validation" / "reference_data" / "qlknn"
QLKNN10D_MANIFEST = QLKNN_ROOT / "zenodo_3497066" / "files_manifest.json"


def _payload() -> dict[str, object]:
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
                "checksum": "md5:2e0cd6344e3269a6aca196f9913d6fff",
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
    report = validate_public_data_acquisition_directory(QLKNN_ROOT)

    assert report["status"] == "pass"
    assert report["records"] == 3
    assert report["local_files"] == 0
    assert report["deferred_files"] == 52
    assert report["deferred_bytes"] > 300_000_000_000


def test_load_public_data_acquisition_manifest_binds_record_sha() -> None:
    manifest = load_public_data_acquisition_manifest(QLKNN10D_MANIFEST)

    assert manifest.doi == "10.5281/zenodo.3497066"
    assert manifest.record_sha256 == "77d168ed8a7fb84f7d60308a051943b9a00ebbd486812c28e9063e8e41b8b9ae"
    assert manifest.local_files == ()


def test_manifest_rejects_tampered_record_sha(tmp_path) -> None:
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
def test_manifest_rejects_malformed_top_level_metadata(mutation, message) -> None:
    payload = _payload()
    mutation(payload)

    with pytest.raises(PublicDataAcquisitionError, match=message):
        validate_public_data_acquisition_manifest(payload)


def test_manifest_rejects_deferred_files_without_policy() -> None:
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
def test_manifest_rejects_unsafe_remote_file_entries(patch, message) -> None:
    payload = _payload()
    file_payload = deepcopy(payload["files"])[0]
    assert isinstance(file_payload, dict)
    file_payload.update(patch)
    payload["files"] = [file_payload]

    with pytest.raises(PublicDataAcquisitionError, match=message):
        validate_public_data_acquisition_manifest(payload)


def test_manifest_rejects_local_sha_mismatch() -> None:
    payload = _payload()
    file_payload = deepcopy(payload["files"])[0]
    assert isinstance(file_payload, dict)
    file_payload["local_sha256"] = "c" * 64
    payload["files"] = [file_payload]

    with pytest.raises(PublicDataAcquisitionError, match="local_sha256 does not match"):
        validate_public_data_acquisition_manifest(payload)


def test_manifest_rejects_local_path_traversal() -> None:
    payload = _payload()
    file_payload = deepcopy(payload["files"])[0]
    assert isinstance(file_payload, dict)
    file_payload["local_path"] = "../README.md"
    payload["files"] = [file_payload]

    with pytest.raises(PublicDataAcquisitionError, match="local_path must stay"):
        validate_public_data_acquisition_manifest(payload)


def test_directory_report_fails_without_public_manifests(tmp_path) -> None:
    report = validate_public_data_acquisition_directory(tmp_path)

    assert report["status"] == "fail"
    assert report["errors"] == [{"path": str(tmp_path), "error": "no public acquisition manifests found"}]


def test_loader_rejects_duplicate_json_keys(tmp_path) -> None:
    path = tmp_path / "files_manifest.json"
    path.write_text('{"schema_version":"x","schema_version":"y"}', encoding="utf-8")

    with pytest.raises(PublicDataAcquisitionError, match="duplicate JSON key: schema_version"):
        load_public_data_acquisition_manifest(path)

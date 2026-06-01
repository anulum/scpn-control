#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Public Data Acquisition Manifest Validation

"""Validate public-data acquisition manifests and locally mirrored files."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "scpn-control.public-data-acquisition.v1"
_ZENODO_RECORD_RE = re.compile(r"^https://zenodo\.org/api/records/[0-9]+/files/.+/content$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_MD5_RE = re.compile(r"^md5:[0-9a-f]{32}$")


class PublicDataAcquisitionError(ValueError):
    """Raised when public acquisition metadata is unsafe or inconsistent."""


@dataclass(frozen=True)
class PublicDataFile:
    """Single public file advertised by an acquisition manifest."""

    key: str
    size_bytes: int
    checksum: str
    download_url: str
    local_path: str | None = None
    local_sha256: str | None = None


@dataclass(frozen=True)
class PublicDataAcquisitionManifest:
    """Validated public-data acquisition manifest."""

    path: Path
    doi: str
    title: str
    licence: str
    record_sha256: str
    large_numeric_files_downloaded: bool
    large_numeric_files_policy: str
    files: tuple[PublicDataFile, ...]

    @property
    def local_files(self) -> tuple[PublicDataFile, ...]:
        """Files mirrored locally with SHA-256 evidence."""
        return tuple(file for file in self.files if file.local_path is not None)

    @property
    def deferred_files(self) -> tuple[PublicDataFile, ...]:
        """Files advertised by Zenodo but not mirrored into this checkout."""
        return tuple(file for file in self.files if file.local_path is None)


def iter_public_data_manifest_paths(root: str | Path) -> list[Path]:
    """Return public-data acquisition manifests under ``root`` in stable order."""
    return sorted(Path(root).glob("**/files_manifest.json"))


def load_public_data_acquisition_manifest(path: str | Path) -> PublicDataAcquisitionManifest:
    """Load and validate one public-data acquisition manifest."""
    manifest_path = Path(path)
    payload = _load_json_object(manifest_path)
    return validate_public_data_acquisition_manifest(payload, manifest_path=manifest_path)


def validate_public_data_acquisition_manifest(
    payload: dict[str, Any],
    *,
    manifest_path: Path | None = None,
) -> PublicDataAcquisitionManifest:
    """Validate public acquisition metadata and local SHA-256 bindings."""
    path = Path("<memory>") if manifest_path is None else manifest_path
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise PublicDataAcquisitionError("unsupported public-data acquisition schema_version")
    if _required_str(payload, "source") != "zenodo":
        raise PublicDataAcquisitionError("public-data acquisition source must be zenodo")

    doi = _required_str(payload, "doi")
    if not doi.startswith("10.5281/zenodo."):
        raise PublicDataAcquisitionError("public-data DOI must identify a Zenodo record")
    title = _required_str(payload, "title")
    licence = _required_str(payload, "license")
    record_sha256 = _required_sha256(payload, "record_sha256")
    large_numeric_files_downloaded = _required_bool(payload, "large_numeric_files_downloaded")
    large_numeric_files_policy = _required_str(payload, "large_numeric_files_policy")
    if not large_numeric_files_downloaded and "deferred" not in large_numeric_files_policy.lower():
        raise PublicDataAcquisitionError("deferred large numeric files require an explicit policy")

    record_path = path.with_name("record.json")
    if record_path.is_file():
        observed = _sha256_file(record_path)
        if not _constant_time_equal(observed, record_sha256):
            raise PublicDataAcquisitionError("record_sha256 does not match record.json bytes")

    files_payload = payload.get("files")
    if not isinstance(files_payload, list) or not files_payload:
        raise PublicDataAcquisitionError("files must be a non-empty array")
    files = tuple(_validate_file_entry(entry, index, path) for index, entry in enumerate(files_payload))
    if large_numeric_files_downloaded and any(file.local_path is None for file in files):
        raise PublicDataAcquisitionError("large_numeric_files_downloaded cannot be true while files are deferred")
    return PublicDataAcquisitionManifest(
        path=path,
        doi=doi,
        title=title,
        licence=licence,
        record_sha256=record_sha256,
        large_numeric_files_downloaded=large_numeric_files_downloaded,
        large_numeric_files_policy=large_numeric_files_policy,
        files=files,
    )


def validate_public_data_acquisition_directory(root: str | Path) -> dict[str, Any]:
    """Validate all public acquisition manifests below ``root``."""
    root_path = Path(root)
    manifest_paths = iter_public_data_manifest_paths(root_path)
    report: dict[str, Any] = {
        "status": "pass",
        "schema_version": SCHEMA_VERSION,
        "root": str(root_path),
        "records": 0,
        "files": 0,
        "local_files": 0,
        "deferred_files": 0,
        "deferred_bytes": 0,
        "manifests": [],
        "errors": [],
    }
    if not manifest_paths:
        report["status"] = "fail"
        report["errors"].append({"path": str(root_path), "error": "no public acquisition manifests found"})
        return report

    manifest_reports: list[dict[str, Any]] = report["manifests"]
    errors: list[dict[str, str]] = report["errors"]
    for manifest_path in manifest_paths:
        try:
            manifest = load_public_data_acquisition_manifest(manifest_path)
        except (OSError, json.JSONDecodeError, PublicDataAcquisitionError) as exc:
            errors.append({"path": str(manifest_path), "error": str(exc)})
            continue
        local_files = manifest.local_files
        deferred_files = manifest.deferred_files
        report["records"] += 1
        report["files"] += len(manifest.files)
        report["local_files"] += len(local_files)
        report["deferred_files"] += len(deferred_files)
        report["deferred_bytes"] += sum(file.size_bytes for file in deferred_files)
        manifest_reports.append(
            {
                "path": str(manifest.path),
                "doi": manifest.doi,
                "title": manifest.title,
                "licence": manifest.licence,
                "record_sha256": manifest.record_sha256,
                "files": len(manifest.files),
                "local_files": len(local_files),
                "deferred_files": len(deferred_files),
                "large_numeric_files_downloaded": manifest.large_numeric_files_downloaded,
                "large_numeric_files_policy": manifest.large_numeric_files_policy,
            }
        )
    if errors:
        report["status"] = "fail"
    return report


def _validate_file_entry(payload: object, index: int, manifest_path: Path) -> PublicDataFile:
    if not isinstance(payload, dict):
        raise PublicDataAcquisitionError(f"files[{index}] must be an object")
    key = _required_str(payload, "key")
    if Path(key).is_absolute() or ".." in Path(key).parts:
        raise PublicDataAcquisitionError(f"files[{index}].key must be a safe relative file name")
    size_bytes = payload.get("size_bytes")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
        raise PublicDataAcquisitionError(f"files[{index}].size_bytes must be a positive integer")
    checksum = _required_str(payload, "checksum")
    if _MD5_RE.fullmatch(checksum) is None:
        raise PublicDataAcquisitionError(f"files[{index}].checksum must be md5:<32 lowercase hex>")
    download_url = _required_str(payload, "download_url")
    _validate_zenodo_download_url(download_url, index)

    local_path = payload.get("local_path")
    local_sha256 = payload.get("local_sha256")
    if local_path is None and local_sha256 is None:
        return PublicDataFile(key=key, size_bytes=size_bytes, checksum=checksum, download_url=download_url)
    if not isinstance(local_path, str) or not local_path.strip():
        raise PublicDataAcquisitionError(f"files[{index}].local_path must be a non-empty string")
    if not isinstance(local_sha256, str) or _HEX64_RE.fullmatch(local_sha256) is None:
        raise PublicDataAcquisitionError(f"files[{index}].local_sha256 must be lowercase SHA-256 hex")
    resolved = _resolve_local_path(local_path, manifest_path, index)
    observed = _sha256_file(resolved)
    if not _constant_time_equal(observed, local_sha256):
        raise PublicDataAcquisitionError(f"files[{index}].local_sha256 does not match local file bytes")
    return PublicDataFile(
        key=key,
        size_bytes=size_bytes,
        checksum=checksum,
        download_url=download_url,
        local_path=local_path,
        local_sha256=local_sha256,
    )


def _validate_zenodo_download_url(url: str, index: int) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "zenodo.org":
        raise PublicDataAcquisitionError(f"files[{index}].download_url must use https://zenodo.org")
    if _ZENODO_RECORD_RE.fullmatch(url) is None:
        raise PublicDataAcquisitionError(f"files[{index}].download_url must be a Zenodo record file URL")


def _resolve_local_path(local_path: str, manifest_path: Path, index: int) -> Path:
    candidate = Path(local_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PublicDataAcquisitionError(f"files[{index}].local_path must stay under the repository root")
    resolved = (ROOT / candidate).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise PublicDataAcquisitionError(f"files[{index}].local_path escapes the repository root") from exc
    if not resolved.is_file():
        fallback = (manifest_path.parent / candidate.name).resolve()
        try:
            fallback.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise PublicDataAcquisitionError(f"files[{index}].local_path escapes the repository root") from exc
        if fallback.is_file():
            return fallback
        raise PublicDataAcquisitionError(f"files[{index}].local_path does not exist")
    return resolved


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
    if not isinstance(payload, dict):
        raise PublicDataAcquisitionError("public-data acquisition manifest root must be an object")
    return payload


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise PublicDataAcquisitionError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PublicDataAcquisitionError(f"{key} must be a non-empty string")
    return value.strip()


def _required_bool(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise PublicDataAcquisitionError(f"{key} must be a boolean")
    return value


def _required_sha256(payload: dict[str, Any], key: str) -> str:
    value = _required_str(payload, key)
    if _HEX64_RE.fullmatch(value) is None:
        raise PublicDataAcquisitionError(f"{key} must be lowercase SHA-256 hex")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _constant_time_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left, right)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for public-data acquisition manifest validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(ROOT / "validation" / "reference_data" / "qlknn"),
        help="Root directory containing public-data files_manifest.json files",
    )
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    parser.add_argument("--output-json", help="Write JSON report to this path")
    args = parser.parse_args(argv)

    report = validate_public_data_acquisition_directory(args.root)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "Public data acquisition manifests: "
            f"{report['status']} "
            f"records={report['records']} "
            f"files={report['files']} "
            f"local_files={report['local_files']} "
            f"deferred_files={report['deferred_files']}"
        )
        for error in report["errors"]:
            print(f"ERROR {error['path']}: {error['error']}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Public Data Acquisition Manifest Validation

"""Inspect offline Zenodo acquisition declarations and selected local byte custody.

Metadata PASS neither downloads nor authenticates remote records, licences or
numeric tensors. Optional raw records bind bytes only when present; canonical
raw records remain unvendored. Standalone CLI inspection needs only stdlib,
while direct frozen constructors bypass validation. See public API examples
for actual deferred corpus observations and failure/lookup contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "scpn-control.public-data-acquisition.v1"
_ZENODO_RECORD_RE = re.compile(r"^https://zenodo\.org/api/records/[0-9]+/files/.+/content$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_MD5_RE = re.compile(r"^md5:[0-9a-f]{32}$")
_DOI_RE = re.compile(r"^10\.5281/zenodo\.([1-9][0-9]*)$")


class PublicDataAcquisitionError(ValueError):
    """Raised when public acquisition metadata is unsafe or inconsistent."""


@dataclass(frozen=True)
class PublicDataFile:
    """Frozen advertised file declaration with optional validated local custody.

    Fields key/positive size_bytes/md5 checksum/download_url identify advertised
    metadata. Optional local_path/local_sha256 default to None and must be supplied
    together by the validator. Direct construction bypasses parsing and hashing.
    No network download or numeric tensor is held by this record.
    """

    key: str
    size_bytes: int
    checksum: str
    download_url: str
    local_path: str | None = None
    local_sha256: str | None = None


@dataclass(frozen=True)
class PublicDataAcquisitionManifest:
    """Frozen validated acquisition declaration with immutable file tuple.

    Fields path/doi/title/licence/record_sha256 retain normalized declaration
    provenance. large_numeric_files_downloaded is a literal policy boolean;
    large_numeric_files_policy records deferred storage policy. Files enumerate
    advertised identities, not authenticated remote availability. A missing raw
    record is allowed; record_sha256 then remains unverified metadata. Direct
    construction bypasses validation and does not prove acquisition.
    """

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
    """List sorted ``**/files_manifest.json`` paths beneath a root.

    Parameters
    ----------
    root : str or Path
        Glob search directory; a missing root yields an empty list.

    Returns
    -------
    list of Path
        Discovered paths, without JSON/schema/containment validation.

    Raises
    ------
    OSError, ValueError, RuntimeError
        Supported filesystem/path failures, handled by the directory inspector.
    """
    return sorted(Path(root).glob("**/files_manifest.json"))


def load_public_data_acquisition_manifest(path: str | Path) -> PublicDataAcquisitionManifest:
    """Read finite unique-key UTF-8 JSON and validate one acquisition declaration.

    Parameters
    ----------
    path : str or Path
        Manifest file. Optional adjacent record.json is hash-bound if present;
        local mirrors remain resolved within the defining repository root.

    Returns
    -------
    PublicDataAcquisitionManifest
        Validated metadata and selected local SHA-256/MD5/size observations.

    Raises
    ------
    PublicDataAcquisitionError
        Supported JSON/UTF-8/read/path/depth, schema or selected custody failure.

    Examples
    --------
    >>> manifest = load_public_data_acquisition_manifest(ROOT / "validation/reference_data/qlknn/zenodo_3497066/files_manifest.json")
    >>> (manifest.doi, len(manifest.deferred_files), len(manifest.local_files))
    ('10.5281/zenodo.3497066', 5, 0)
    """
    manifest_path = Path(path)
    payload = _load_json_object(manifest_path)
    return validate_public_data_acquisition_manifest(payload, manifest_path=manifest_path)


def validate_public_data_acquisition_manifest(
    payload: dict[str, Any],
    *,
    manifest_path: Path | None = None,
) -> PublicDataAcquisitionManifest:
    """Validate Zenodo acquisition declarations and selected existing local bytes.

    Parameters
    ----------
    payload : dict
        Versioned schema object. Unknown mapping fields are not interpreted.
    manifest_path : Path or None
        Optional declaration location; None uses a cwd-relative <memory> spelling.

    Returns
    -------
    PublicDataAcquisitionManifest
        Numeric positive-record DOI and unique safe file keys, with exact DOI-record
        and decoded-key URL linkage. Local SHA-256, advertised MD5 and positive byte
        size bind one observed stream. Missing raw record remains a declaration.

    Raises
    ------
    PublicDataAcquisitionError
        Unsupported schema/shape/policy/URL/path, duplicate file key, optional
        record mismatch/escape, local byte inconsistency or supported read failure.

    Notes
    -----
    If record.json exists, its bytes are SHA-256 checked inside the manifest's
    directory. Its JSON contents are not decoded/authenticated or compared with
    declarations. ROOT-relative mirrors take precedence over a full-relative-path
    adjacent fallback, both canonically contained in ROOT; no basename substitution
    or cwd mirror lookup. Deferred files have no local verification. MD5 is an
    advertised byte check, not a security/authenticity claim. No network request,
    numeric replay, coherent filesystem snapshot or scientific admission occurs.
    """
    try:
        path = Path("<memory>") if manifest_path is None else Path(manifest_path)
        return _validate_manifest(payload, path)
    except PublicDataAcquisitionError:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        raise PublicDataAcquisitionError(f"cannot inspect public acquisition: {exc}") from exc


def _validate_manifest(payload: dict[str, Any], path: Path) -> PublicDataAcquisitionManifest:
    """Validate declaration shapes and optional locally available custody bytes."""
    if not isinstance(payload, dict):
        raise PublicDataAcquisitionError("public-data acquisition manifest root must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise PublicDataAcquisitionError("unsupported public-data acquisition schema_version")
    if _required_str(payload, "source") != "zenodo":
        raise PublicDataAcquisitionError("public-data acquisition source must be zenodo")

    doi = _required_str(payload, "doi")
    match = _DOI_RE.fullmatch(doi)
    if match is None:
        raise PublicDataAcquisitionError("public-data DOI must identify a Zenodo record")
    title = _required_str(payload, "title")
    licence = _required_str(payload, "license")
    record_sha256 = _required_sha256(payload, "record_sha256")
    large_numeric_files_downloaded = _required_bool(payload, "large_numeric_files_downloaded")
    large_numeric_files_policy = _required_str(payload, "large_numeric_files_policy")
    if not large_numeric_files_downloaded and "deferred" not in large_numeric_files_policy.lower():
        raise PublicDataAcquisitionError("deferred large numeric files require an explicit policy")

    record_path = path.with_name("record.json")
    if record_path.exists() or record_path.is_symlink():
        if not record_path.is_file():
            raise PublicDataAcquisitionError("record.json must be a readable regular file when present")
        record_path.resolve().relative_to(path.parent.resolve())
        observed = _sha256_file(record_path)
        if not _constant_time_equal(observed, record_sha256):
            raise PublicDataAcquisitionError("record_sha256 does not match record.json bytes")

    files_payload = payload.get("files")
    if not isinstance(files_payload, list) or not files_payload:
        raise PublicDataAcquisitionError("files must be a non-empty array")
    files = tuple(_validate_file_entry(entry, index, path) for index, entry in enumerate(files_payload))
    if len({file.key for file in files}) != len(files):
        raise PublicDataAcquisitionError("duplicate public file key")
    for file in files:
        parsed = urlparse(file.download_url)
        parts = parsed.path.split("/")
        if parts[3] != match.group(1) or unquote("/".join(parts[5:-1]), errors="strict") != file.key:
            raise PublicDataAcquisitionError("download_url must match the manifest DOI record and file key")
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
    """Inspect discovered declarations and report admitted counts and failures.

    Parameters
    ----------
    root : str or Path
        Existing directory; discovered symlink targets must stay in this scan root.

    Returns
    -------
    dict
        pass/fail status, schema/root, accepted record/file/local/deferred counts,
        deferred bytes, summaries and ordered errors. Invalid declarations supply
        findings and no accepted counters; empty discovery fails. Counters retained
        on aggregate FAIL are diagnostic, not complete acquisition readiness.

    Notes
    -----
    Local record/custody validation follows the defining APIs. No raw record is
    required, duplicate DOI records across separate manifests are not deduplicated,
    and no download or training is initiated.

    Examples
    --------
    >>> report = validate_public_data_acquisition_directory(ROOT / "validation/reference_data/qlknn")
    >>> (report["status"], report["records"], report["local_files"], report["deferred_files"])
    ('pass', 3, 0, 52)
    """
    manifest_paths: list[Path] = []
    report: dict[str, Any] = {
        "status": "pass",
        "schema_version": SCHEMA_VERSION,
        "root": str(root),
        "records": 0,
        "files": 0,
        "local_files": 0,
        "deferred_files": 0,
        "deferred_bytes": 0,
        "manifests": [],
        "errors": [],
    }
    try:
        root_path = Path(root).resolve()
        if not root_path.is_dir():
            raise ValueError("public acquisition root must be a directory")
        manifest_paths = iter_public_data_manifest_paths(root_path)
        for path in manifest_paths:
            path.resolve().relative_to(root_path)
    except (OSError, ValueError, RuntimeError) as exc:
        report["status"] = "fail"
        report["errors"].append({"path": str(root), "error": f"cannot scan public acquisition root: {exc}"})
        return report
    if not manifest_paths:
        report["status"] = "fail"
        report["errors"].append({"path": str(root_path), "error": "no public acquisition manifests found"})
        return report

    manifest_reports: list[dict[str, Any]] = report["manifests"]
    errors: list[dict[str, str]] = report["errors"]
    for manifest_path in manifest_paths:
        try:
            manifest = load_public_data_acquisition_manifest(manifest_path)
        except (OSError, ValueError, RuntimeError) as exc:
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
    """Parse advertised metadata and verify selected local SHA-256/MD5/size in one stream."""
    if not isinstance(payload, dict):
        raise PublicDataAcquisitionError(f"files[{index}] must be an object")
    key = _required_str(payload, "key")
    if not _safe_relative_path(key):
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
    observed, observed_md5, observed_size = _local_fingerprints(resolved)
    if not _constant_time_equal(observed, local_sha256):
        raise PublicDataAcquisitionError(f"files[{index}].local_sha256 does not match local file bytes")
    if observed_size != size_bytes:
        raise PublicDataAcquisitionError(f"files[{index}].size_bytes does not match local file bytes")
    if not _constant_time_equal("md5:" + observed_md5, checksum):
        raise PublicDataAcquisitionError(f"files[{index}].checksum does not match local file MD5")
    return PublicDataFile(
        key=key,
        size_bytes=size_bytes,
        checksum=checksum,
        download_url=download_url,
        local_path=local_path,
        local_sha256=local_sha256,
    )


def _validate_zenodo_download_url(url: str, index: int) -> None:
    """Require canonical HTTPS Zenodo record-file URLs without query/fragment or malformed escapes."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "zenodo.org":
        raise PublicDataAcquisitionError(f"files[{index}].download_url must use https://zenodo.org")
    if _ZENODO_RECORD_RE.fullmatch(url) is None:
        raise PublicDataAcquisitionError(f"files[{index}].download_url must be a Zenodo record file URL")
    if parsed.query or parsed.fragment or re.search(r"%(?![0-9a-fA-F]{2})", parsed.path):
        raise PublicDataAcquisitionError(
            f"files[{index}].download_url must not contain query/fragment or malformed escapes"
        )


def _resolve_local_path(local_path: str, manifest_path: Path, index: int) -> Path:
    """Resolve the ROOT-relative mirror then full adjacent fallback, with canonical ROOT containment."""
    candidate = Path(local_path)
    if not _safe_relative_path(local_path):
        raise PublicDataAcquisitionError(f"files[{index}].local_path must stay under the repository root")
    resolved = (ROOT / candidate).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise PublicDataAcquisitionError(f"files[{index}].local_path escapes the repository root") from exc
    if not resolved.is_file():
        fallback = (manifest_path.parent / candidate).resolve()
        try:
            fallback.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise PublicDataAcquisitionError(f"files[{index}].local_path escapes the repository root") from exc
        if fallback.is_file():
            return fallback
        raise PublicDataAcquisitionError(f"files[{index}].local_path does not exist")
    return resolved


def _load_json_object(path: Path) -> dict[str, Any]:
    """Decode finite unique-key UTF-8 JSON and translate supported read/depth errors."""
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json,
                parse_float=_finite_json_float,
            )
    except PublicDataAcquisitionError:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        raise PublicDataAcquisitionError(f"cannot load public acquisition JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise PublicDataAcquisitionError("public-data acquisition manifest root must be an object")
    return payload


def _reject_nonfinite_json(token: str) -> Any:
    """Refuse nonstandard nonfinite constants at every JSON depth."""
    raise PublicDataAcquisitionError(f"nonfinite JSON value: {token}")


def _finite_json_float(token: str) -> float:
    """Parse finite float metadata, refusing exponent overflow to infinity."""
    value = float(token)
    if not math.isfinite(value):
        raise PublicDataAcquisitionError(f"nonfinite JSON value: {token}")
    return value


def _safe_relative_path(value: str) -> bool:
    """Require nonempty local POSIX/Windows spellings without roots/drives/traversal."""
    if "\x00" in value:
        return False
    paths = (PurePosixPath(value), PureWindowsPath(value))
    return all(p.parts and not p.root and not p.drive and ".." not in p.parts for p in paths)


def _local_fingerprints(path: Path) -> tuple[str, str, int]:
    """Hash/count one observed byte stream for SHA-256, advertised MD5 and size."""
    sha = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    return sha.hexdigest(), md5.hexdigest(), size


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build JSON objects while refusing duplicate keys at every nested object depth."""
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise PublicDataAcquisitionError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _required_str(payload: dict[str, Any], key: str) -> str:
    """Return a required nonempty trimmed declaration string or raise a schema finding."""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PublicDataAcquisitionError(f"{key} must be a non-empty string")
    return value.strip()


def _required_bool(payload: dict[str, Any], key: str) -> bool:
    """Require a literal policy boolean without truthiness conversion."""
    value = payload.get(key)
    if not isinstance(value, bool):
        raise PublicDataAcquisitionError(f"{key} must be a boolean")
    return value


def _required_sha256(payload: dict[str, Any], key: str) -> str:
    """Require a nonempty lowercase 64-hex declaration digest."""
    value = _required_str(payload, key)
    if _HEX64_RE.fullmatch(value) is None:
        raise PublicDataAcquisitionError(f"{key} must be lowercase SHA-256 hex")
    return value


def _sha256_file(path: Path) -> str:
    """Stream optional raw record bytes in bounded chunks into a lowercase SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _constant_time_equal(left: str, right: str) -> bool:
    """Compare string digest declarations using the standard constant-time primitive."""
    return hmac.compare_digest(left, right)


def main(argv: list[str] | None = None) -> int:
    """Execute the standalone acquisition metadata report CLI.

    Parameters
    ----------
    argv : list of str or None
        argparse tokens; None reads process arguments. Default root is the actual
        repository QLKNN metadata directory, independent of cwd.

    Returns
    -------
    int
        0 for PASS, 1 for inspection or supported output IO/path failure. JSON mode
        emits the report; text mode prints a summary and ordered stderr findings.
        Output parents are created; partial/failed writes do not establish custody.

    Raises
    ------
    SystemExit
        argparse help or argument refusal. No data transfer or training runs.
    """
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
        try:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except (OSError, ValueError) as exc:
            report["status"] = "fail"
            report["errors"].append(
                {"path": args.output_json, "error": f"cannot write public acquisition report: {exc}"}
            )
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

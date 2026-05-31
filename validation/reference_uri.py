#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Reference Artifact URI Validation

"""Shared URI admission checks for external physics-reference artifacts."""

from __future__ import annotations

from pathlib import PurePosixPath
from urllib.parse import urlparse

_REMOTE_SCHEMES = {"https", "s3", "gs"}
_LOCAL_FILE_PREFIXES = ("/validation/reports/", "/validation/reference_data/")
_EXECUTABLE_PATH_PREFIXES = (
    "/opt/",
    "/usr/local/",
    "/usr/bin/",
    "/bin/",
    "/nix/store/",
    "/validation/external_bins/",
    "/facility/",
    "/mnt/facility/",
    "/gpfs/",
    "/lustre/",
)
_BLOCKED_EXECUTABLE_PATH_PREFIXES = (
    "/dev/",
    "/etc/",
    "/proc/",
    "/run/",
    "/sys/",
    "/tmp/",
    "/var/tmp/",
)


def reference_artifact_uri_error(value: object, field: str) -> str | None:
    """Return an admission error for an artifact URI, or ``None`` when valid."""
    if not isinstance(value, str) or not value.strip():
        return f"{field} must be a non-empty URI"
    uri = value.strip()
    parsed = urlparse(uri)
    if not parsed.scheme:
        return f"{field} must include an explicit URI scheme"
    if parsed.scheme == "file":
        return _file_uri_error(parsed.netloc, parsed.path, field)
    if parsed.scheme in _REMOTE_SCHEMES:
        if not parsed.netloc or not parsed.path or _has_parent_traversal(parsed.path):
            return f"{field} must identify a stable remote artifact path"
        return None
    return f"{field} scheme must be file, https, s3, or gs"


def external_executable_path_error(value: object, field: str = "binary_path") -> str | None:
    """Return an admission error for real external-code executable provenance."""
    if not isinstance(value, str) or not value.strip():
        return f"{field} must be a non-empty absolute executable path"
    path = value.strip()
    parsed = urlparse(path)
    if parsed.scheme or parsed.netloc:
        return f"{field} must be an absolute filesystem path, not a URI"
    if "\x00" in path or any(ord(char) < 32 for char in path):
        return f"{field} must not contain control characters"
    posix = PurePosixPath(path)
    if not posix.is_absolute():
        return f"{field} must be an absolute filesystem path"
    if _has_parent_traversal(path):
        return f"{field} must not contain parent traversal"
    if path.endswith("/") or posix.name in {"", ".", ".."}:
        return f"{field} must identify an executable file path"
    if path.startswith(_BLOCKED_EXECUTABLE_PATH_PREFIXES):
        return f"{field} must not point to mutable or system-control paths"
    if not path.startswith(_EXECUTABLE_PATH_PREFIXES):
        return f"{field} must be under an admitted deployment or facility executable root"
    return None


def _file_uri_error(netloc: str, path: str, field: str) -> str | None:
    if netloc:
        return f"{field} file URI must not include a host"
    if not path.startswith(_LOCAL_FILE_PREFIXES):
        return f"{field} file URI must be under /validation/reports or /validation/reference_data"
    if _has_parent_traversal(path):
        return f"{field} must not contain parent traversal"
    return None


def _has_parent_traversal(path: str) -> bool:
    return any(part == ".." for part in PurePosixPath(path).parts)

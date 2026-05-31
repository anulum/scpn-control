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

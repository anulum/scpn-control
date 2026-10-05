#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Reference Artifact URI Validation

"""Apply shared lexical policies to declared reference URIs and executable paths.

These pure string checks read no files, follow no symlinks, fetch no URLs and
authenticate no binary or reference. ``None`` means only that the declaration
meets the policy below. Callers own all byte custody and scientific admission.
Surrounding spaces are ignored; ASCII control characters are refused before
parsing can discard them. URI parsing failures become authored field findings.

Artifact URIs allow hostless file paths under the two validation prefixes or
https/s3/gs addresses with an authority and path. Traversal checks are literal
POSIX components: percent escapes, queries, fragments and credentials are not
decoded or authenticated. Local artifact prefixes alone do not require a file
leaf. Executable declarations instead require an absolute POSIX path in an
admitted deployment/facility prefix, with no URI authority, parent component,
trailing slash or final dot component. Existence and executability are untested.
There is no mutable state, cache, lock or platform filesystem interpretation.
"""

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
    """Return a field finding unless an artifact declaration meets the URI policy.

    Parameters
    ----------
    value
        Candidate string; other types and empty/space-only strings are refused.
    field
        Caller-selected field name prepended to each authored finding.

    Returns
    -------
    str or None
        A finding for malformed syntax, controls, an unsupported scheme or an
        out-of-policy path. None supplies no download, digest or provenance proof.

    Examples
    --------
    >>> reference_artifact_uri_error("file:///etc/passwd", "reference")
    'reference file URI must be under /validation/reports or /validation/reference_data'
    >>> reference_artifact_uri_error("https://[", "reference")
    'reference must be a syntactically valid URI'
    """
    if not isinstance(value, str) or not value.strip():
        return f"{field} must be a non-empty URI"
    if _has_control_characters(value):
        return f"{field} must not contain control characters"
    uri = value.strip()
    try:
        parsed = urlparse(uri)
    except ValueError:
        return f"{field} must be a syntactically valid URI"
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
    """Check a declared POSIX executable location without inspecting the binary.

    Parameters
    ----------
    value
        Candidate absolute path string in an admitted deployment/facility root.
        Spaces around the declaration are ignored; ASCII controls are refused.
    field
        Field label in findings; defaults to binary_path.

    Returns
    -------
    str or None
        Authored syntax/location refusal, or None for a policy-conforming
        declaration. No executable bit, existence, version or digest is checked.

    Examples
    --------
    >>> external_executable_path_error("/tmp/solver")
    'binary_path must not point to mutable or system-control paths'
    """
    if not isinstance(value, str) or not value.strip():
        return f"{field} must be a non-empty absolute executable path"
    if _has_control_characters(value):
        return f"{field} must not contain control characters"
    path = value.strip()
    try:
        parsed = urlparse(path)
    except ValueError:
        return f"{field} must be a syntactically valid absolute executable path"
    if parsed.scheme or parsed.netloc:
        return f"{field} must be an absolute filesystem path, not a URI"
    posix = PurePosixPath(path)
    if not posix.is_absolute():
        return f"{field} must be an absolute filesystem path"
    if _has_parent_traversal(path):
        return f"{field} must not contain parent traversal"
    if path.endswith("/") or path.rsplit("/", 1)[-1] == ".":
        return f"{field} must identify an executable file path"
    if path.startswith(_BLOCKED_EXECUTABLE_PATH_PREFIXES):
        return f"{field} must not point to mutable or system-control paths"
    if not path.startswith(_EXECUTABLE_PATH_PREFIXES):
        return f"{field} must be under an admitted deployment or facility executable root"
    return None


def _file_uri_error(netloc: str, path: str, field: str) -> str | None:
    """Apply hostless validation-prefix and literal-parent checks to parsed file components."""
    if netloc:
        return f"{field} file URI must not include a host"
    if not path.startswith(_LOCAL_FILE_PREFIXES):
        return f"{field} file URI must be under /validation/reports or /validation/reference_data"
    if _has_parent_traversal(path):
        return f"{field} must not contain parent traversal"
    return None


def _has_parent_traversal(path: str) -> bool:
    """Detect a literal POSIX parent component without decoding escapes or resolving a filesystem."""
    return any(part == ".." for part in PurePosixPath(path).parts)


def _has_control_characters(value: str) -> bool:
    """Detect ASCII C0 or DEL in the original declaration before whitespace/parser normalization."""
    return any(ord(char) < 32 or ord(char) == 127 for char in value)

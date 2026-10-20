# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Bounded Python release metadata inspection.
"""Read a named release's declared Python constraint without installing it.

HTTPS uses the standard certificate verification. Explicit HTTP origins are
limited to loopback for local metadata services. A successful response binds
the declared name/version, not an artifact digest or runtime qualification.
"""

from __future__ import annotations

import json
import math
from http.client import HTTPConnection, HTTPException, HTTPSConnection
from urllib.parse import quote, urlsplit

from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

PYPI_METADATA_URL = "https://pypi.org/pypi"
MAX_METADATA_BYTES = 1 << 20


class DependencyMetadataError(ValueError):
    """Refuse unavailable, malformed or mismatched release declarations.

    Only authored diagnostics are exposed; network, parser and OS exception
    text is not included. No refused read produces a compatibility verdict.
    """


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Refuse duplicate JSON members before interpreting release fields."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DependencyMetadataError("Release metadata contains duplicate JSON fields.")
        result[key] = value
    return result


def _invalid_constant(value: str) -> object:
    """Refuse non-JSON numeric constants anywhere in a metadata response."""
    raise DependencyMetadataError("Release metadata contains a non-JSON numeric constant.")


def _connection(base_url: str, timeout_s: float) -> tuple[HTTPConnection, str]:
    """Validate an explicit metadata origin and return its unopened connection."""
    try:
        url = urlsplit(base_url)
        host, port = url.hostname, url.port
    except ValueError:
        raise DependencyMetadataError("The metadata origin is invalid.") from None
    if (
        not host
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
        or url.scheme not in {"http", "https"}
        or (url.scheme == "http" and host not in {"localhost", "127.0.0.1", "::1"})
    ):
        raise DependencyMetadataError("The metadata origin requires HTTPS or explicit loopback HTTP.")
    connection = (
        HTTPSConnection(host, port, timeout=timeout_s)
        if url.scheme == "https"
        else HTTPConnection(host, port, timeout=timeout_s)
    )
    return connection, url.path.rstrip("/")


def read_python_constraint(
    name: str,
    version: str,
    *,
    base_url: str = PYPI_METADATA_URL,
    timeout_s: float = 10.0,
) -> str | None:
    """Read and bind ``info.requires_python`` for one exact release.

    Parameters
    ----------
    name, version : str
        Canonicalisable distribution name and valid PEP 440 version. Release
        metadata must declare the same identities after normalisation.
    base_url : str
        Explicit JSON metadata origin and prefix, normally PyPI's ``/pypi``.
        A custom origin supplies operator-selected declarations, not PyPI proof.
        Credentials, query strings, fragments and remote plain HTTP are refused.
    timeout_s : float
        Finite positive per-socket timeout in seconds. There are no retries or
        redirects. The body is bounded to one MiB, not a total elapsed deadline.

    Returns
    -------
    str or None
        Published constraint, including an explicit empty string or null when
        the release declares no restriction. An absent field is refused.

    Raises
    ------
    DependencyMetadataError
        Inputs, transport, status, content type, size, JSON or release fields
        cannot be checked. This is declaration inspection, not installation,
        artifact integrity, advisory scanning or executable compatibility proof.
    """
    if not isinstance(name, str) or not name or not isinstance(version, str):
        raise DependencyMetadataError("A distribution name and exact version are required.")
    try:
        canonical_name = canonicalize_name(name, validate=True)
        canonical_version = Version(version)
    except ValueError:
        raise DependencyMetadataError("The release identity is invalid.") from None
    if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise DependencyMetadataError("The metadata timeout must be finite and positive.")
    connection, prefix = _connection(base_url, timeout_s)
    path = f"{prefix}/{quote(canonical_name, safe='')}/{quote(str(canonical_version), safe='')}/json"
    try:
        connection.request("GET", path, headers={"Accept": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise DependencyMetadataError("The metadata service did not return HTTP 200.")
        if (response.getheader("Content-Type") or "").partition(";")[0].strip().lower() != "application/json":
            raise DependencyMetadataError("The metadata response is not application/json.")
        data = response.read(MAX_METADATA_BYTES + 1)
    except (OSError, HTTPException, UnicodeError):
        raise DependencyMetadataError("Could not read the requested release metadata.") from None
    finally:
        connection.close()
    if not data or len(data) > MAX_METADATA_BYTES:
        raise DependencyMetadataError("The metadata body is empty or exceeds one MiB.")
    try:
        payload: object = json.loads(
            data.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_invalid_constant
        )
    except DependencyMetadataError:
        raise
    except (ValueError, RecursionError):
        raise DependencyMetadataError("Release metadata is not valid UTF-8 JSON.") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("info"), dict):
        raise DependencyMetadataError("Release metadata requires an info object.")
    info = payload["info"]
    declared_name, declared_version = info.get("name"), info.get("version")
    if not isinstance(declared_name, str) or not isinstance(declared_version, str):
        raise DependencyMetadataError("Release metadata requires a name and version.")
    try:
        matches = (
            canonicalize_name(declared_name, validate=True) == canonical_name
            and Version(declared_version) == canonical_version
        )
    except (ValueError, InvalidVersion):
        raise DependencyMetadataError("Release metadata declares an invalid identity.") from None
    if not matches:
        raise DependencyMetadataError("Release metadata does not match the requested identity.")
    constraint = info.get("requires_python")
    if "requires_python" not in info or (constraint is not None and not isinstance(constraint, str)):
        raise DependencyMetadataError("Release metadata requires a string or null Python constraint.")
    return constraint

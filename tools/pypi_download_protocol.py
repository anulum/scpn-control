# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — PyPI HTTP protocol and snapshot validation.
"""Fetch and validate declared PyPI overall download snapshots."""

from __future__ import annotations

import http.client
import json
import time
from collections.abc import Callable
from datetime import date
from typing import cast
from urllib.parse import quote

PYPISTATS_HOST = "pypistats.org"
PYPISTATS_PATH = "/api/packages/{package}/overall"
PYPISTATS_TYPE = "overall_downloads"
CATEGORIES = ("without_mirrors", "with_mirrors")
CSV_HEADER = ("date", *CATEGORIES)
PAYLOAD_KEYS = frozenset(("data", "package", "type"))
ROW_KEYS = frozenset(("category", "date", "downloads"))
PROJECT_PACKAGE = "scpn-control"

REQUEST_TIMEOUT_SECONDS = 30
MAX_RESPONSE_BYTES = 5_000_000
RETRYABLE_STATUSES = (429, 502, 503, 504)
RETRY_DELAYS = (15.0, 30.0, 60.0)
MAX_TRANSIENT_WINDOW_SECONDS = REQUEST_TIMEOUT_SECONDS * (len(RETRY_DELAYS) + 1) + int(sum(RETRY_DELAYS))

Fetch = Callable[[str], bytes]
Sleep = Callable[[float], None]
DownloadRows = dict[str, dict[str, int]]


class DownloadConfigurationError(ValueError):
    """Refuse absent project metadata or an alternate project-only target.

    Parameters
    ----------
    message : str
        Authored configuration refusal. Native TOML and I/O errors stay separate.
    """

    __module__ = "tools.pypi_downloads"


class DownloadSnapshotError(RuntimeError):
    """Refuse a malformed snapshot or a permanent service response.

    Parameters
    ----------
    message : str
        Deliberately authored failure description, without native exception text.

    Notes
    -----
    The CLI echoes this explicit refusal type. Service reason phrases and parser
    or transport exception strings must never be embedded in its message.
    """

    __module__ = "tools.pypi_downloads"


class RetryableSnapshotError(DownloadSnapshotError):
    """Signal a transient service or transport failure eligible for retry.

    Parameters
    ----------
    message : str
        Deliberately authored transient-failure description.

    Notes
    -----
    Four failed attempts produce the existing soft skip rather than a CSV update.
    This subtype preserves the public DownloadSnapshotError hierarchy.
    """

    __module__ = "tools.pypi_downloads"


class DuplicateJSONObjectKeyError(ValueError):
    """Refuse repeated JSON member names before snapshot validation.

    Parameters
    ----------
    message : str
        Authored duplicate-member refusal, excluding the incoming member name.
    """

    __module__ = "tools.pypi_downloads"


def _reject_duplicate_object_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build one JSON object while rejecting duplicate names at every depth."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONObjectKeyError("duplicate JSON object name")
        result[key] = value
    return result


def package_endpoint_path(package: str) -> str:
    """Encode a package name inside the fixed pypistats endpoint path.

    Parameters
    ----------
    package : str
        Package identity; outer whitespace is stripped for the request path.

    Returns
    -------
    str
        Relative overall-series path with all package path characters escaped.

    Raises
    ------
    DownloadConfigurationError
        Package is blank. No request or file write occurs here.
    """
    if not package.strip():
        raise DownloadConfigurationError("package name must not be empty")
    return PYPISTATS_PATH.format(package=quote(package.strip(), safe=""))


def _http_get(package: str) -> bytes:
    """Read one size-limited JSON response from the fixed HTTPS host.

    Parameters
    ----------
    package : str
        Package identity encoded into the overall-series request path.

    Returns
    -------
    bytes
        Complete response body of at most MAX_RESPONSE_BYTES bytes.

    Raises
    ------
    RetryableSnapshotError
        A retryable status, native socket error or HTTP protocol fault occurs.
    DownloadSnapshotError
        Status, exact JSON media type or response size is refused.

    Notes
    -----
    The connection closes after every attempt. The30-second socket timeout is
    per socket operation, not a total request deadline. Parameters such as charset
    are permitted after the exact application/json media type.
    """
    connection = http.client.HTTPSConnection(PYPISTATS_HOST, timeout=REQUEST_TIMEOUT_SECONDS)
    try:
        connection.request(
            "GET",
            package_endpoint_path(package),
            headers={
                "Accept": "application/json",
                "User-Agent": "scpn-control-metrics/1",
            },
        )
        response = connection.getresponse()
        body = response.read(MAX_RESPONSE_BYTES + 1)
        if response.status in RETRYABLE_STATUSES:
            raise RetryableSnapshotError(f"pypistats returned transient HTTP {response.status}")
        if response.status != 200:
            raise DownloadSnapshotError(f"pypistats returned HTTP {response.status}")
        content_type = response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise DownloadSnapshotError("pypistats returned unexpected Content-Type")
        if len(body) > MAX_RESPONSE_BYTES:
            raise DownloadSnapshotError("pypistats response exceeded the size limit")
        return body
    except (OSError, http.client.HTTPException) as exc:
        raise RetryableSnapshotError("pypistats request failed") from exc
    finally:
        connection.close()


def _valid_date(raw_date: object) -> str | None:
    """Return one canonical ISO date or ``None`` for malformed input."""
    if not isinstance(raw_date, str) or raw_date != raw_date.strip():
        return None
    try:
        parsed = date.fromisoformat(raw_date)
    except ValueError:
        return None
    return raw_date if parsed.isoformat() == raw_date else None


def _valid_count(raw_count: object) -> int | None:
    """Return one non-negative integer download count or ``None``."""
    if isinstance(raw_count, bool) or not isinstance(raw_count, int):
        return None
    return raw_count if raw_count >= 0 else None


def validate_overall(payload: object, package: str) -> DownloadRows:
    """Reduce an exact overall-downloads JSON object to sparse daily rows.

    Parameters
    ----------
    payload : object
        Decoded object with exactly data, package and type fields. Every data row
        declares category, canonical ISO date and a non-negative integer count.
    package : str
        Exact expected package identity; no normalization is inferred.

    Returns
    -------
    DownloadRows
        Date-to-category mapping. Each day has with_mirrors; missing
        without_mirrors stays absent instead of becoming a fabricated zero.

    Raises
    ------
    DownloadSnapshotError
        Identity, type, keys, date, integer count, duplicate or mirror ordering
        is refused. Boolean counts are invalid; without cannot exceed with.

    Notes
    -----
    Counts are upstream declarations, not independent measurement or attestation.
    The caller supplies input; no files or network requests occur here.
    """
    if not isinstance(payload, dict):
        raise DownloadSnapshotError("pypistats response must be a JSON object")
    if set(payload) != PAYLOAD_KEYS:
        raise DownloadSnapshotError("pypistats response has unexpected top-level keys")
    if payload.get("package") != package:
        raise DownloadSnapshotError("pypistats response package identity mismatch")
    if payload.get("type") != PYPISTATS_TYPE:
        raise DownloadSnapshotError("pypistats response type mismatch")
    raw_rows = payload.get("data")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise DownloadSnapshotError("pypistats response data must be a non-empty list")

    rows: DownloadRows = {}
    seen: set[tuple[str, str]] = set()
    for index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, dict) or set(raw_row) != ROW_KEYS:
            raise DownloadSnapshotError(f"pypistats row {index} has an invalid object schema")
        category = raw_row.get("category")
        row_date = _valid_date(raw_row.get("date"))
        downloads = _valid_count(raw_row.get("downloads"))
        if category not in CATEGORIES:
            raise DownloadSnapshotError(f"pypistats row {index} has invalid category")
        if row_date is None:
            raise DownloadSnapshotError(f"pypistats row {index} has invalid date")
        if downloads is None:
            raise DownloadSnapshotError(f"pypistats row {index} has invalid downloads")
        key = (row_date, cast(str, category))
        if key in seen:
            raise DownloadSnapshotError(f"pypistats response duplicates {row_date}/{category}")
        seen.add(key)
        rows.setdefault(row_date, {})[cast(str, category)] = downloads

    for row_date, values in rows.items():
        if "with_mirrors" not in values:
            raise DownloadSnapshotError(f"pypistats response is missing with_mirrors for {row_date}")
        without_mirrors = values.get("without_mirrors")
        if without_mirrors is not None and without_mirrors > values["with_mirrors"]:
            raise DownloadSnapshotError(f"pypistats response violates mirror-count ordering for {row_date}")
    return rows


def fetch_overall(package: str, fetch: Fetch = _http_get) -> DownloadRows:
    """Fetch and decode one unambiguous overall-series snapshot.

    Parameters
    ----------
    package : str
        Exact identity expected in the decoded response.
    fetch : callable, optional
        Package-to-bytes transport; defaults to the fixed HTTPS host.

    Returns
    -------
    DownloadRows
        Strictly validated sparse daily count mapping.

    Raises
    ------
    DownloadSnapshotError
        JSON bytes, duplicate members or snapshot fields are refused.
    RetryableSnapshotError
        The selected transport reports a transient failure.

    Notes
    -----
    Injected transports own their timing and faults. This function does not retry,
    write files or authenticate externally supplied counts.
    """
    response = fetch(package)
    try:
        decoded: object = json.loads(response, object_pairs_hook=_reject_duplicate_object_keys)
    except DuplicateJSONObjectKeyError as exc:
        raise DownloadSnapshotError("pypistats returned invalid JSON: duplicate JSON object name") from exc
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise DownloadSnapshotError("pypistats returned invalid JSON") from exc
    return validate_overall(decoded, package)


def fetch_overall_with_retry(
    package: str,
    fetch: Fetch = _http_get,
    sleep: Sleep = time.sleep,
) -> DownloadRows | None:
    """Try a snapshot four times, soft-skipping exhausted transient failures.

    Parameters
    ----------
    package : str
        Exact expected package identity.
    fetch : callable, optional
        Transport invoked once per attempt.
    sleep : callable, optional
        Delay callback receiving 15, 30 and 60 seconds between transient failures.

    Returns
    -------
    DownloadRows or None
        First valid snapshot, or None after four transient failures.

    Raises
    ------
    DownloadSnapshotError
        A permanent response or snapshot validation refusal occurs.

    Notes
    -----
    Only RetryableSnapshotError is retried. The nominal 225-second arithmetic
    combines four 30-second timeouts and 105 seconds of sleep; it is not a hard
    wall deadline. Socket operations and supplied callbacks can take longer.
    """
    for delay in (*RETRY_DELAYS, None):
        try:
            return fetch_overall(package, fetch)
        except RetryableSnapshotError:
            if delay is not None:
                sleep(delay)
    return None

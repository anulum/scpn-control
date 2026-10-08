# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Lifecycle scalar and UTF-8 JSON parsing

"""Lifecycle scalar and UTF-8 JSON parsing."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

from tools.report_lifecycle_types import LifecycleRegistryError

_FILENAME_TIMESTAMP_RE: Final[re.Pattern[str]] = re.compile(r"(20[0-9]{6}T[0-9]{6})Z?")


_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")


_GIT_SHA_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{40}")


def _require_exact_keys(value: dict[str, object], required: set[str], *, context: str) -> None:
    """Require the complete schema field set, including no unknown fields.

    Parameters
    ----------
    value : dict of str to object
        Parsed JSON object being checked.
    required : set of str
        Exact permitted field names.
    context : str
        Field location used in the authored refusal.

    Raises
    ------
    LifecycleRegistryError
        Missing or unexpected fields are present.
    """
    actual = set(value)
    if actual != required:
        raise LifecycleRegistryError(
            f"{context} fields drift: missing={sorted(required - actual)} unknown={sorted(actual - required)}"
        )


def _require_object(value: object, context: str) -> dict[str, object]:
    """Require a JSON object without coercing a different container.

    Parameters
    ----------
    value : object
        Value produced by the JSON decoder.
    context : str
        Field location for the refusal.

    Returns
    -------
    dict of str to object
        Original object, with JSON-decoder string keys.

    Raises
    ------
    LifecycleRegistryError
        Value is not a dictionary.
    """
    if not isinstance(value, dict):
        raise LifecycleRegistryError(f"{context} must be an object")
    return cast(dict[str, object], value)


def _require_list(value: object, context: str) -> list[object]:
    """Require a JSON array without converting another value.

    Parameters
    ----------
    value : object
        Decoded field value.
    context : str
        Field location for the refusal.

    Returns
    -------
    list of object
        Original array; its elements are validated by the caller.

    Raises
    ------
    LifecycleRegistryError
        Value is not a list.
    """
    if not isinstance(value, list):
        raise LifecycleRegistryError(f"{context} must be an array")
    return value


def _require_string(value: object, context: str) -> str:
    """Require nonblank text while preserving its original spelling.

    Parameters
    ----------
    value : object
        Decoded field value.
    context : str
        Field location for the refusal.

    Returns
    -------
    str
        Unmodified string, including meaningful surrounding whitespace.

    Raises
    ------
    LifecycleRegistryError
        Value is not a string or contains only whitespace.
    """
    if not isinstance(value, str) or not value.strip():
        raise LifecycleRegistryError(f"{context} must be a non-empty string")
    return value


def _require_boolean(value: object, context: str) -> bool:
    """Require a literal JSON boolean rather than truthy text or numbers.

    Parameters
    ----------
    value : object
        Decoded admission or policy field.
    context : str
        Field location for the refusal.

    Returns
    -------
    bool
        Original boolean declaration.

    Raises
    ------
    LifecycleRegistryError
        Value is not a boolean.
    """
    if not isinstance(value, bool):
        raise LifecycleRegistryError(f"{context} must be a boolean")
    return value


def _require_integer(value: object, context: str, *, minimum: int) -> int:
    """Require an integer declaration at or above its field-specific minimum.

    Parameters
    ----------
    value : object
        Decoded count or freshness-policy value.
    context : str
        Field location for the refusal.
    minimum : int
        Inclusive lower bound; booleans are refused independently.

    Returns
    -------
    int
        Validated integer without coercion.

    Raises
    ------
    LifecycleRegistryError
        Value is boolean, noninteger or below the minimum.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LifecycleRegistryError(f"{context} must be an integer >= {minimum}")
    return value


def parse_datetime(value: str) -> datetime:
    """Parse an ISO-8601 or compact UTC timestamp.

    Parameters
    ----------
    value : str
        ISO-8601 text or ``YYYYMMDDTHHMMSS`` with an optional trailing ``Z``.
        Surrounding whitespace is removed. Naive timestamps denote UTC.

    Returns
    -------
    datetime.datetime
        Timezone-aware timestamp converted to UTC.

    Raises
    ------
    ValueError
        Timestamp is blank, malformed or outside the datetime range.
    """
    stripped = value.strip()
    if not stripped:
        raise ValueError("timestamp must be non-empty")
    if _FILENAME_TIMESTAMP_RE.fullmatch(stripped):
        parsed = datetime.strptime(stripped.removesuffix("Z"), "%Y%m%dT%H%M%S")
        return parsed.replace(tzinfo=UTC)
    normalized = stripped.replace("Z", "+00:00")
    return _normalize_datetime(datetime.fromisoformat(normalized))


def _read_json_object(path: Path) -> dict[str, object]:
    """Read one UTF-8 JSON object without rewriting its bytes.

    Parameters
    ----------
    path : pathlib.Path
        Registry, report or refresh artifact to read.

    Returns
    -------
    dict of str to object
        Decoded object; nested schema validation belongs to its consumer.

    Raises
    ------
    LifecycleRegistryError
        Top-level JSON value is not an object.
    OSError, ValueError
        File cannot be read or decoded as UTF-8 JSON.
    """
    with path.open(encoding="utf-8") as handle:
        payload: object = json.load(handle)
    if not isinstance(payload, dict):
        raise LifecycleRegistryError(f"{path} must contain a JSON object")
    return cast(dict[str, object], payload)


def _normalize_datetime(value: datetime) -> datetime:
    """Interpret a naive timestamp as UTC or convert an aware timestamp.

    Parameters
    ----------
    value : datetime.datetime
        Inventory or evidence timestamp.

    Returns
    -------
    datetime.datetime
        Equivalent timezone-aware UTC timestamp.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _validate_max_age_days(value: int) -> None:
    """Require an exact nonnegative count of days for either public loader.

    Parameters
    ----------
    value : int
        Advisory age window. There is no upper bound; the registry's recorded
        21-day policy is checked separately and is never rewritten here.

    Raises
    ------
    LifecycleRegistryError
        Value is negative or is not an exact built-in integer, including bool.
    """
    if type(value) is not int or value < 0:
        raise LifecycleRegistryError("max_age_days must be a non-negative integer")

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable Transport Latency Evidence Validation
"""Declared local differentiable latency fields contracts; no audit replay."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ONE_STEP_CLAIM_STATUS = "local audited gradient-admission latency only; not a real-time control-loop guarantee"
ROLLOUT_CLAIM_STATUS = "local audited rollout source-gradient latency only; not a real-time control-loop guarantee"
READINESS_BLOCKED_CLAIM_STATUS = "bounded differentiable transport readiness only; full-fidelity claim remains blocked"
READINESS_ADMITTED_CLAIM_STATUS = "full-fidelity differentiable transport claim admitted"
CHANNEL_ORDER = ["electron_temperature", "ion_temperature", "electron_density", "impurity_density"]
BLOCKED_CLAIM_STATUSES = {
    "no latency claim; JAX gradient backend unavailable in this environment",
}


def _load_json(path: Path) -> dict[str, Any]:
    """Read one caller-relative UTF-8 JSON object, rejecting duplicate keys.

    Parameters
    ----------
    path : pathlib.Path
        Report file; no directory creation or report mutation occurs.

    Returns
    -------
    dict[str, Any]
        Decoded object with unknown fields left uninterpreted.

    Raises
    ------
    OSError, UnicodeError, ValueError, RecursionError
        Read/decoding/parsing, duplicate keys, non-object root or excessive
        nesting fails. Standard json nonfinite tokens remain decoded.

    Notes
    -----
    No raw-byte digest, source authentication or locked snapshot is established.
    """
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
    if not isinstance(payload, dict):
        raise ValueError("report root must be a JSON object")
    return payload


def _is_sha256_hex(value: object) -> bool:
    """Check only the syntax of a declared SHA-256 digest.

    Parameters
    ----------
    value : object
        Candidate declaration; no file or referenced artifact is opened.

    Returns
    -------
    bool
        True for a string of exactly64 hexadecimal characters, case insensitive.
        This does not bind a report, audit, campaign or proof to actual bytes.
    """
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Construct an object while refusing repeated keys at every JSON depth.

    Parameters
    ----------
    pairs : list[tuple[str, Any]]
        Ordered pairs supplied by json.load's object_pairs_hook.

    Returns
    -------
    dict[str, Any]
        Fresh mapping preserving decoded pair order.

    Raises
    ------
    ValueError
        A key repeats; the diagnostic names that key.
    """
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _require_value(
    path: Path, payload: dict[str, Any], field: str, expected: object, errors: list[dict[str, object]]
) -> None:
    """Append a field diagnostic when a fixed declaration differs or coerces integers.

    Parameters
    ----------
    path : pathlib.Path
        Report path used only in the diagnostic.
    payload : dict[str, Any]
        Report or metadata mapping.
    field : str
        Required field key.
    expected : object
        Fixed expected value. Integer fields require nonboolean integer input.
    errors : list[dict[str, object]]
        Caller-owned ordered diagnostics, mutated in place.

    Returns
    -------
    None
        Missing, different or float/boolean integer declarations append an error.
        String fields retain exact equality, with no trimming or case folding.
    """
    value = payload.get(field)
    integer_type_error = type(expected) is int and (isinstance(value, bool) or not isinstance(value, int))
    if integer_type_error or value != expected:
        errors.append({"path": str(path), "field": field, "error": f"field must be {expected!r}"})


def _positive_int(value: object) -> bool:
    """Require a positive Python integer counter without accepting booleans.

    Parameters
    ----------
    value : object
        Decoded counter.

    Returns
    -------
    bool
        True only for nonboolean int values greater than zero.
    """
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _non_negative_int(value: object) -> bool:
    """Require a nonnegative Python integer without accepting boolean indices.

    Parameters
    ----------
    value : object
        Decoded counter or audit-index component.

    Returns
    -------
    bool
        True only for nonboolean int values at least zero.
    """
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _finite_positive(value: object) -> float | None:
    """Accept positive finite audit epsilon, tolerance or timestamp declarations.

    Parameters
    ----------
    value : object
        Decoded scalar; booleans are refused.

    Returns
    -------
    float or None
        Positive finite conversion, otherwise None.
    """
    numeric = _finite_number(value)
    if numeric is None or numeric <= 0.0:
        return None
    return numeric


def _finite_non_negative(value: object) -> float | None:
    """Accept nonnegative finite latency, loss or absolute-error declarations.

    Parameters
    ----------
    value : object
        Decoded scalar; zero is valid and booleans are refused.

    Returns
    -------
    float or None
        Nonnegative finite conversion, otherwise None.
    """
    numeric = _finite_number(value)
    if numeric is None or numeric < 0.0:
        return None
    return numeric


def _finite_number(value: object) -> float | None:
    """Convert int/float declarations only when finite and representable.

    Parameters
    ----------
    value : object
        Decoded numeric field; booleans and nonnumeric types are refused.

    Returns
    -------
    float or None
        Finite conversion or None for invalid types, nonfinite values and
        integer conversion overflow. No clipping or string conversion occurs.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        numeric = float(value)
    except OverflowError:
        return None
    return numeric if math.isfinite(numeric) else None

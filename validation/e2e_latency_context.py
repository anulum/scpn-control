# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — E2E Latency Evidence Validation
"""Validate declared UTC and local benchmark context without replaying a run."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

from validation.e2e_latency_payload import E2E_LATENCY_CLAIM_BOUNDARY


def _valid_utc_timestamp(value: object) -> bool:
    """Accept only nonblank ISO timestamps carrying an explicit UTC offset.

    Parameters
    ----------
    value : object
        Recorded generation timestamp; Z and zero numeric offsets are accepted.

    Returns
    -------
    bool
        True when datetime.fromisoformat parses a timezone-aware zero-offset
        instant. Naive dates/times and nonzero offsets are refused.

    Notes
    -----
    No freshness, clock synchronisation or authenticity check is performed.
    """
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        timestamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return False
    return timestamp.tzinfo is not None and timestamp.utcoffset() == timedelta(0)


def _validate_loadavg(payload: dict[str, Any], key: str, errors: list[str]) -> None:
    """Append one diagnostic for a malformed three-value host-load declaration.

    Parameters
    ----------
    payload : dict[str, Any]
        Benchmark context mapping; no mutation is performed.
    key : str
        Load-average list field to inspect.
    errors : list[str]
        Caller-owned diagnostic list to append to on the first invalid value.

    Returns
    -------
    None
        Lists of exactly three finite int/float values pass, excluding booleans.

    Raises
    ------
    OverflowError
        A declared integer cannot be represented as float.

    Notes
    -----
    Negative values retain the legacy reader behaviour. No host observation or
    load threshold is inferred from declarations.
    """
    value = payload.get(key)
    if not isinstance(value, list) or len(value) != 3:
        errors.append(f"context.{key} must contain three finite load-average values")
        return
    for item in value:
        if not isinstance(item, int | float) or isinstance(item, bool) or not math.isfinite(float(item)):
            errors.append(f"context.{key} must contain three finite load-average values")
            return


def _validate_benchmark_context(payload: dict[str, Any], errors: list[str]) -> None:
    """Check required local-claim and benchmark-context declarations in order.

    Parameters
    ----------
    payload : dict[str, Any]
        Parsed latency report; unknown fields are left uninterpreted.
    errors : list[str]
        Mutable caller-owned list receiving authored refusal messages.

    Returns
    -------
    None
        Append errors for command, explicit UTC, local boundary/class/flag,
        affinity, isolation, two load lists, governor presence and heavy jobs.

    Raises
    ------
    OverflowError
        A host-load integer cannot be converted to float.

    Notes
    -----
    Command matching uses a substring, CPU IDs need only be nonnegative ints
    and governor needs only be present, including null. Duplicated IDs and any
    nonblank job/isolation labels retain legacy behaviour. Nothing is executed;
    hardware identity, actual isolation and trustworthy clocks are not verified.
    """
    command = payload.get("command")
    if not isinstance(command, str) or "benchmarks/e2e_control_latency.py" not in command:
        errors.append("command must record the E2E benchmark invocation")

    if not _valid_utc_timestamp(payload.get("generated_utc")):
        errors.append("generated_utc must record an ISO-8601 UTC timestamp")

    if payload.get("evidence_class") != "local_regression":
        errors.append("evidence_class must be local_regression")
    if payload.get("production_claim_allowed") is not False:
        errors.append("production_claim_allowed must be false for local E2E latency reports")
    if payload.get("claim_boundary") != E2E_LATENCY_CLAIM_BOUNDARY:
        errors.append("claim_boundary must preserve the canonical local-evidence boundary")

    context = payload.get("context")
    if not isinstance(context, dict):
        errors.append("context must record benchmark host-load and isolation metadata")
        context = {}
    affinity = context.get("cpu_affinity")
    if (
        not isinstance(affinity, list)
        or not affinity
        or not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in affinity)
    ):
        errors.append("context.cpu_affinity must record at least one CPU")
    isolation_method = context.get("isolation_method")
    if not isinstance(isolation_method, str) or not isolation_method.strip():
        errors.append("context.isolation_method must record the benchmark isolation method")
    _validate_loadavg(context, "loadavg_start", errors)
    _validate_loadavg(context, "loadavg_end", errors)
    if "governor" not in context:
        errors.append("context.governor must record CPU governor state or null when unavailable")
    heavy_jobs = context.get("heavy_jobs_running")
    if not isinstance(heavy_jobs, str) or not heavy_jobs.strip():
        errors.append("context.heavy_jobs_running must record whether concurrent heavy jobs were observed")

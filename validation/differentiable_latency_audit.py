# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable Transport Latency Evidence Validation
"""Declared local differentiable latency audit contracts; no audit replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from validation.differentiable_latency_fields import (
    _finite_non_negative,
    _finite_positive,
    _non_negative_int,
    _positive_int,
)


def _validate_audit(path: Path, payload: dict[str, Any], kind: str, errors: list[dict[str, object]]) -> None:
    """Check declared sampled-gradient audit fields without recomputing gradients.

    Parameters
    ----------
    path : pathlib.Path
        Report path for diagnostics.
    payload : dict[str, Any]
        Parent report containing audit, radial width and optional rollout length.
    kind : str
        one_step checks chi and source errors; rollout checks only source error.
    errors : list[dict[str, object]]
        Ordered diagnostics appended in place.

    Returns
    -------
    None
        Require passed=True, positive finite epsilon/tolerance, nonnegative
        loss/errors, errors within declared tolerance and valid sampled indices.

    Notes
    -----
    Errors/loss inherit the producer's model units; no calibration, reference
    comparison, audit digest or campaign-tolerance binding is performed here.
    """
    audit = payload.get("audit")
    if not isinstance(audit, dict):
        errors.append({"path": str(path), "field": "audit", "error": "audit must be an object"})
        return
    if audit.get("passed") is not True:
        errors.append(
            {"path": str(path), "field": "audit.passed", "error": "audit must pass for admitted latency evidence"}
        )
    tolerance = _finite_positive(audit.get("tolerance"))
    epsilon = _finite_positive(audit.get("epsilon"))
    loss = _finite_non_negative(audit.get("loss"))
    if tolerance is None:
        errors.append(
            {"path": str(path), "field": "audit.tolerance", "error": "audit tolerance must be positive and finite"}
        )
    if epsilon is None:
        errors.append(
            {"path": str(path), "field": "audit.epsilon", "error": "audit epsilon must be positive and finite"}
        )
    if loss is None:
        errors.append({"path": str(path), "field": "audit.loss", "error": "audit loss must be finite and non-negative"})
    source_error = _finite_non_negative(audit.get("source_max_abs_error"))
    if source_error is None:
        errors.append(
            {
                "path": str(path),
                "field": "audit.source_max_abs_error",
                "error": "source audit error must be finite and non-negative",
            }
        )
    elif tolerance is not None and source_error > tolerance:
        errors.append(
            {"path": str(path), "field": "audit.source_max_abs_error", "error": "source audit error exceeds tolerance"}
        )
    if kind == "one_step":
        chi_error = _finite_non_negative(audit.get("chi_max_abs_error"))
        if chi_error is None:
            errors.append(
                {
                    "path": str(path),
                    "field": "audit.chi_max_abs_error",
                    "error": "chi audit error must be finite and non-negative",
                }
            )
        elif tolerance is not None and chi_error > tolerance:
            errors.append(
                {"path": str(path), "field": "audit.chi_max_abs_error", "error": "chi audit error exceeds tolerance"}
            )
    _validate_indices(path, audit.get("checked_indices"), kind, payload, errors)


def _validate_indices(
    path: Path,
    indices: object,
    kind: str,
    payload: dict[str, Any],
    errors: list[dict[str, object]],
) -> None:
    """Check sampled audit-index widths, uniqueness and declared array domains.

    Parameters
    ----------
    path : pathlib.Path
        Report path for diagnostics.
    indices : object
        Expected nonempty list of index lists.
    kind : str
        one_step uses [channel,rho]; rollout uses [step,channel,rho].
    payload : dict[str, Any]
        n_rho and n_steps domain declarations; invalid counters imply zero size.
    errors : list[dict[str, object]]
        Ordered diagnostics, appended on the first invalid index.

    Returns
    -------
    None
        Refuse malformed/noninteger/boolean/negative/duplicate/out-of-domain
        indices. Channels are0..3. No complete-domain sampling is required.
    """
    if not isinstance(indices, list) or not indices:
        errors.append(
            {"path": str(path), "field": "audit.checked_indices", "error": "checked_indices must be a non-empty list"}
        )
        return
    width = 3 if kind == "rollout" else 2
    n_rho = int(payload.get("n_rho", 0)) if _positive_int(payload.get("n_rho")) else 0
    n_steps = int(payload.get("n_steps", 0)) if _positive_int(payload.get("n_steps")) else 0
    seen: set[tuple[int, ...]] = set()
    for raw in indices:
        if not isinstance(raw, list) or len(raw) != width or not all(_non_negative_int(value) for value in raw):
            errors.append(
                {
                    "path": str(path),
                    "field": "audit.checked_indices",
                    "error": f"indices must be {width}-element non-negative integer lists",
                }
            )
            return
        item = tuple(int(value) for value in raw)
        if item in seen:
            errors.append(
                {"path": str(path), "field": "audit.checked_indices", "error": "checked_indices must be unique"}
            )
            return
        seen.add(item)
        if kind == "one_step":
            channel, rho_idx = item
            if channel >= 4 or rho_idx >= n_rho:
                errors.append(
                    {"path": str(path), "field": "audit.checked_indices", "error": "one-step audit index out of domain"}
                )
                return
        else:
            step, channel, rho_idx = item
            if step >= n_steps or channel >= 4 or rho_idx >= n_rho:
                errors.append(
                    {"path": str(path), "field": "audit.checked_indices", "error": "rollout audit index out of domain"}
                )
                return

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable Transport Latency Evidence Validation
"""Declared local differentiable latency readiness contracts; no audit replay."""

from __future__ import annotations

import json
from pathlib import Path

from validation.differentiable_latency_fields import (
    CHANNEL_ORDER,
    READINESS_ADMITTED_CLAIM_STATUS,
    READINESS_BLOCKED_CLAIM_STATUS,
    _is_sha256_hex,
    _load_json,
    _positive_int,
    _require_value,
)
from validation.differentiable_latency_reports import _validate_blocked_report


def _validate_readiness_report(path: Path, errors: list[dict[str, object]]) -> dict[str, object]:
    """Check readiness declarations and clear readiness on any local field error.

    Parameters
    ----------
    path : pathlib.Path
        Caller-relative readiness JSON path.
    errors : list[dict[str, object]]
        Shared ordered diagnostic list appended in place.

    Returns
    -------
    dict[str, object]
        path/kind/status/full_fidelity_ready/blocked_reasons. Invalid fields
        yield fail/False. Valid blocked declarations yield blocked/False;
        valid declared readiness yields pass/True only with admitted external
        reference and non-null external/controller digest declarations.

    Notes
    -----
    SHA fields are checked for syntax only; reports/audits/campaign/proof/
    reference bytes are not fetched or rebound. Valid readiness is a declaration,
    not scientific promotion, external-reference or theorem authentication.
    """
    errors_before = len(errors)
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append({"path": str(path), "field": "json", "error": str(exc)})
        return {"path": str(path), "kind": "full_fidelity_readiness", "status": "fail"}

    if payload.get("status") == "blocked":
        entry = _validate_blocked_report(path, payload, "full_fidelity_readiness", errors)
        entry["full_fidelity_ready"] = False
        entry["blocked_reasons"] = ["jax_backend"]
        return entry

    _require_value(path, payload, "schema_version", 1, errors)
    _require_value(path, payload, "backend", "jax", errors)
    if not _positive_int(payload.get("n_rho")) or int(payload.get("n_rho", 0)) < 3:
        errors.append({"path": str(path), "field": "readiness.n_rho", "error": "n_rho must be an integer >= 3"})
    if not _positive_int(payload.get("rollout_steps")):
        errors.append(
            {"path": str(path), "field": "readiness.rollout_steps", "error": "rollout_steps must be positive"}
        )
    for field in (
        "campaign_sha256",
        "gradient_latency_report_sha256",
        "gradient_audit_sha256",
        "rollout_latency_report_sha256",
        "rollout_audit_sha256",
    ):
        if not _is_sha256_hex(payload.get(field)):
            errors.append({"path": str(path), "field": field, "error": "field must be a SHA-256 hex digest"})
    for field in ("external_reference_artifact_sha256", "controller_formal_artifact_sha256"):
        value = payload.get(field)
        if value is not None and not _is_sha256_hex(value):
            errors.append({"path": str(path), "field": field, "error": "field must be null or a SHA-256 hex digest"})

    channel_order = payload.get("channel_order")
    if channel_order != CHANNEL_ORDER:
        errors.append({"path": str(path), "field": "readiness.channel_order", "error": "unexpected channel order"})
    if payload.get("equilibrium_coupled") is not True:
        errors.append(
            {"path": str(path), "field": "readiness.equilibrium_coupled", "error": "equilibrium coupling required"}
        )
    if not isinstance(payload.get("external_reference_admitted"), bool):
        errors.append(
            {"path": str(path), "field": "readiness.external_reference_admitted", "error": "field must be boolean"}
        )
    full_fidelity_ready = payload.get("full_fidelity_claim_admissible")
    blocked_reasons = payload.get("blocked_reasons")
    if not isinstance(full_fidelity_ready, bool):
        errors.append(
            {"path": str(path), "field": "readiness.full_fidelity_claim_admissible", "error": "field must be boolean"}
        )
        full_fidelity_ready = False
    if not isinstance(blocked_reasons, list) or not all(isinstance(reason, str) for reason in blocked_reasons):
        errors.append(
            {"path": str(path), "field": "readiness.blocked_reasons", "error": "blocked_reasons must be string list"}
        )
        blocked_reasons = []
    if full_fidelity_ready and blocked_reasons:
        errors.append(
            {
                "path": str(path),
                "field": "readiness.blocked_reasons",
                "error": "full-fidelity readiness cannot have blocked reasons",
            }
        )
    if not full_fidelity_ready and not blocked_reasons:
        errors.append(
            {
                "path": str(path),
                "field": "readiness.blocked_reasons",
                "error": "blocked readiness must explain blocked reasons",
            }
        )
    if full_fidelity_ready:
        if payload.get("external_reference_admitted") is not True:
            errors.append(
                {
                    "path": str(path),
                    "field": "readiness.external_reference_admitted",
                    "error": "full-fidelity readiness requires declared external reference admission",
                }
            )
        for field in ("external_reference_artifact_sha256", "controller_formal_artifact_sha256"):
            if payload.get(field) is None:
                errors.append(
                    {
                        "path": str(path),
                        "field": field,
                        "error": "full-fidelity readiness requires reference and controller digest declarations",
                    }
                )
    expected_claim_status = READINESS_ADMITTED_CLAIM_STATUS if full_fidelity_ready else READINESS_BLOCKED_CLAIM_STATUS
    _require_value(path, payload, "claim_status", expected_claim_status, errors)
    invalid = len(errors) > errors_before
    return {
        "path": str(path),
        "kind": "full_fidelity_readiness",
        "status": "fail" if invalid else ("pass" if full_fidelity_ready else "blocked"),
        "full_fidelity_ready": bool(full_fidelity_ready) and not invalid,
        "blocked_reasons": blocked_reasons,
    }

#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK geometry report contracts


"""Retain reference headers, report schema, unit labels, comparison bounds and consistency digests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

_REQUIRED_HEADER_FIELDS = (
    "spdx_license_id",
    "commercial_license",
    "concepts_copyright",
    "code_copyright",
    "orcid",
    "contact",
    "file",
)

_REPORT_SCHEMA = "scpn-control.gk-geometry-reference.v2"

_UNITS = {
    "theta": "rad",
    "R": "m",
    "Z": "m",
    "jacobian": "m",
    "g_rr": "dimensionless",
    "g_rt": "m-1",
    "g_tt": "m-2",
    "B_toroidal": "T",
    "b_dot_grad_theta": "m-1",
}

_FULL_EQUILIBRIUM_BLOCKED_REASON = (
    "Requires independent Miller-geometry implementation or external equilibrium-code evidence."
)

_ABS_TOLERANCE = 1.0e-11

_REL_TOLERANCE = 1.0e-10


def _validate_header(path: Path, payload: dict[str, Any], errors: list[dict[str, object]]) -> None:
    """Require all seven nonblank canonical header declarations and input schema1.0 without author authentication."""
    for field in _REQUIRED_HEADER_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(
                {"path": str(path), "field": field, "error": "reference file requires canonical header metadata"}
            )
    if payload.get("schema_version") != "1.0":
        errors.append({"path": str(path), "field": "schema_version", "error": "schema_version must be '1.0'"})


def _new_report(path: Path) -> dict[str, Any]:
    """Initialize original v2 bounded-local/full-equilibrium-false report with fixed tolerances and dimensionally correct metric units."""
    return {
        "schema_version": _REPORT_SCHEMA,
        "status": "pass",
        "reference_path": _portable_path(path),
        "reference_file_sha256": None,
        "payload_sha256": None,
        "tolerances": {"absolute": _ABS_TOLERANCE, "relative": _REL_TOLERANCE},
        "units": dict(_UNITS),
        "public_claims": {
            "bounded_local_miller_geometry_reference": False,
            "full_equilibrium_reconstruction": False,
            "full_equilibrium_blocked_reason": _FULL_EQUILIBRIUM_BLOCKED_REASON,
        },
        "cases": 0,
        "entries": [],
        "errors": [],
    }


def _json_sha256(payload: object) -> str:
    """Hash original compact sorted ASCII-escaped JSON, without producer authentication."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _portable_path(path: Path) -> str:
    """Render resolved repository-relative paths when possible; retain raw paths for outside or unresolved references."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except (OSError, ValueError, RuntimeError):
        return str(path)


def _finalise_report(report: dict[str, Any]) -> dict[str, Any]:
    """Bind original report consistency digest and set bounded-local admission only from passing comparison status."""
    report["public_claims"]["bounded_local_miller_geometry_reference"] = report["status"] == "pass"
    payload = dict(report)
    payload["payload_sha256"] = None
    report["payload_sha256"] = _json_sha256(payload)
    return report

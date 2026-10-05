#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable Transport Latency Evidence Validation
"""Validate local persisted declarations, preserving original helper aliases.

Per-entry status/counts follow field validity; readiness errors clear the
readiness declaration. No measurement, digest authentication or promotion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.differentiable_latency_audit import _validate_audit as _validate_audit
from validation.differentiable_latency_audit import _validate_indices as _validate_indices
from validation.differentiable_latency_context import _validate_latency_order as _validate_latency_order
from validation.differentiable_latency_context import _validate_runtime_metadata as _validate_runtime_metadata
from validation.differentiable_latency_fields import BLOCKED_CLAIM_STATUSES as BLOCKED_CLAIM_STATUSES
from validation.differentiable_latency_fields import CHANNEL_ORDER as CHANNEL_ORDER
from validation.differentiable_latency_fields import ONE_STEP_CLAIM_STATUS, ROLLOUT_CLAIM_STATUS
from validation.differentiable_latency_fields import READINESS_ADMITTED_CLAIM_STATUS as READINESS_ADMITTED_CLAIM_STATUS
from validation.differentiable_latency_fields import READINESS_BLOCKED_CLAIM_STATUS as READINESS_BLOCKED_CLAIM_STATUS
from validation.differentiable_latency_fields import _finite_non_negative as _finite_non_negative
from validation.differentiable_latency_fields import _finite_number as _finite_number
from validation.differentiable_latency_fields import _finite_positive as _finite_positive
from validation.differentiable_latency_fields import _is_sha256_hex as _is_sha256_hex
from validation.differentiable_latency_fields import _load_json as _load_json
from validation.differentiable_latency_fields import _non_negative_int as _non_negative_int
from validation.differentiable_latency_fields import _positive_int as _positive_int
from validation.differentiable_latency_fields import _reject_duplicate_json_keys as _reject_duplicate_json_keys
from validation.differentiable_latency_fields import _require_value as _require_value
from validation.differentiable_latency_readiness import _validate_readiness_report
from validation.differentiable_latency_reports import _validate_blocked_report as _validate_blocked_report
from validation.differentiable_latency_reports import _validate_report


def validate_differentiable_transport_latency(
    one_step_report: str | Path,
    rollout_report: str | Path | None = None,
    *,
    readiness_report: str | Path | None = None,
    require_admitted: bool = False,
) -> dict[str, Any]:
    """Check persisted local audited-latency and optional readiness declarations.

    Parameters
    ----------
    one_step_report : str or pathlib.Path
        Required caller-relative one-step JSON object.
    rollout_report : str or pathlib.Path or None
        Optional rollout report; None omits it.
    readiness_report : str or pathlib.Path or None
        Optional readiness report; None sets readiness_entry=None/ready=False.
    require_admitted : bool, default False
        Refuse valid blocked latency reports. This does not require readiness
        or full_fidelity_ready=True; local admission differs from promotion.

    Returns
    -------
    dict[str, Any]
        Aggregate pass/fail, paths/policy, valid pass/blocked report counts,
        readiness declaration and ordered entries/errors. Invalid entries are
        fail and never counted as admitted. aggregate readiness remains False on errors.

    Raises
    ------
    RecursionError, TypeError
        Native parsing/path errors outside the documented read refusals propagate.

    Notes
    -----
    Read each supplied file separately without a locked snapshot. OSError,
    UnicodeError and JSON/ValueError are caught per report as json diagnostics.
    Unknown fields remain uninterpreted. No reports are written; no JAX audit,
    benchmark, digest binding, hardware qualification or promotion is executed.
    """
    one_step_path = Path(one_step_report)
    rollout_path = Path(rollout_report) if rollout_report is not None else None
    readiness_path = Path(readiness_report) if readiness_report is not None else None
    errors: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []

    entries.append(
        _validate_report(
            one_step_path,
            kind="one_step",
            claim_status=ONE_STEP_CLAIM_STATUS,
            require_admitted=require_admitted,
            errors=errors,
        )
    )
    if rollout_path is not None:
        entries.append(
            _validate_report(
                rollout_path,
                kind="rollout",
                claim_status=ROLLOUT_CLAIM_STATUS,
                require_admitted=require_admitted,
                errors=errors,
            )
        )

    admitted = [entry for entry in entries if entry.get("status") == "pass"]
    blocked = [entry for entry in entries if entry.get("status") == "blocked"]
    readiness_entry: dict[str, object] | None = None
    if readiness_path is not None:
        readiness_entry = _validate_readiness_report(readiness_path, errors)
    if require_admitted and blocked:
        for entry in blocked:
            errors.append(
                {
                    "path": entry["path"],
                    "field": "status",
                    "error": "admitted differentiable transport latency evidence is required",
                }
            )

    return {
        "status": "pass" if not errors else "fail",
        "one_step_report": str(one_step_path),
        "rollout_report": str(rollout_path) if rollout_path is not None else None,
        "readiness_report": str(readiness_path) if readiness_path is not None else None,
        "require_admitted": require_admitted,
        "admitted_reports": len(admitted),
        "blocked_reports": len(blocked),
        "readiness_entry": readiness_entry,
        "full_fidelity_ready": bool(readiness_entry and readiness_entry.get("full_fidelity_ready") is True)
        and not errors,
        "entries": entries,
        "errors": errors,
    }


def main() -> None:
    """Print standalone declaration validation with optional local-admission policy.

    Parameters
    ----------
    None
        Read sys.argv. --one-step-report, --rollout-report and --readiness-report
        default to repository validation/reports; explicit paths use caller cwd.
        --require-admitted refuses blocked latency, --json-out selects JSON.

    Returns
    -------
    None
        Print aggregate JSON or status/counts and error lines, without writing.

    Raises
    ------
    SystemExit
        0 for aggregate pass/help,1 for fail,2 for argparse syntax.
        Uncaught native API errors cause script failure.

    Notes
    -----
    A pass with full_fidelity_ready=False is valid local latency admission.
    Neither a pass nor a declared readiness authenticates scientific promotion.
    """
    parser = argparse.ArgumentParser(description="Validate differentiable transport gradient-latency evidence")
    parser.add_argument(
        "--one-step-report",
        type=Path,
        default=ROOT / "validation" / "reports" / "differentiable_transport_latency.json",
    )
    parser.add_argument(
        "--rollout-report",
        type=Path,
        default=ROOT / "validation" / "reports" / "differentiable_transport_rollout_latency.json",
    )
    parser.add_argument(
        "--readiness-report",
        type=Path,
        default=ROOT / "validation" / "reports" / "differentiable_transport_full_fidelity_readiness.json",
    )
    parser.add_argument("--require-admitted", action="store_true")
    parser.add_argument("--json-out", action="store_true")
    args = parser.parse_args()

    report = validate_differentiable_transport_latency(
        args.one_step_report,
        args.rollout_report,
        readiness_report=args.readiness_report,
        require_admitted=args.require_admitted,
    )
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "Differentiable transport latency evidence: "
            f"{report['status']} admitted={report['admitted_reports']} blocked={report['blocked_reports']}"
        )
        for error in report["errors"]:
            print(f"ERROR {error['path']}:{error['field']}: {error['error']}")
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()

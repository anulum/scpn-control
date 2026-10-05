# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — E2E Latency Evidence Validation
"""Check persisted local latency declarations and an inclusive microsecond budget.

The payload/context helpers retain compatibility aliases here. A report pass
checks fields and a recomputable checksum; it does not qualify hardware or
replay timings. This standard-library command never writes report artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.e2e_latency_context import (
    _valid_utc_timestamp as _valid_utc_timestamp,
)
from validation.e2e_latency_context import (
    _validate_benchmark_context,
)
from validation.e2e_latency_context import (
    _validate_loadavg as _validate_loadavg,
)
from validation.e2e_latency_payload import (
    _UNQUALIFIED_VALUES as _UNQUALIFIED_VALUES,
)
from validation.e2e_latency_payload import (
    E2E_LATENCY_CLAIM_BOUNDARY,
    E2E_LATENCY_SCHEMA_VERSION,
    _finite_positive_number,
    _load_json,
    _payload_digest,
    _qualified_string,
    _validate_percentiles,
)
from validation.e2e_latency_payload import (
    build_e2e_latency_evidence_payload as build_e2e_latency_evidence_payload,
)


@dataclass(frozen=True)
class LatencyEvidenceReport:
    """Immutable outcome of declaration checks, without real-time certification.

    Attributes
    ----------
    status : str
        pass when no checks fail, otherwise fail.
    errors : tuple[str, ...]
        Ordered authored field diagnostics.
    p95_us : float or None
        Converted positive finite E2E p95 in microseconds, even on failed reports.
    target_hardware_id, target_hardware_class, rt_kernel : str or None
        Trimmed declared labels, or None for unqualified placeholders.

    Notes
    -----
    Direct construction does not validate fields. A validator pass authenticates
    no hardware, original samples, operator approval or end-to-end deadline.
    """

    status: str
    errors: tuple[str, ...]
    p95_us: float | None
    target_hardware_id: str | None
    target_hardware_class: str | None
    rt_kernel: str | None


def validate_e2e_latency_evidence(
    path: str | Path,
    *,
    require_target_hardware: bool = True,
    max_e2e_p95_us: float | None = None,
) -> LatencyEvidenceReport:
    """Check a persisted local report and an optional inclusive p95 budget.

    Parameters
    ----------
    path : str or pathlib.Path
        UTF-8 JSON object read from the caller's working directory.
    require_target_hardware : bool, default True
        Require nonplaceholder id/class/rt_kernel labels. False still requires
        the target_hardware mapping and preserves the local claim boundary.
    max_e2e_p95_us : float or None, default None
        Finite nonnegative microsecond upper bound; equality is accepted.
        None omits only this comparison. Booleans and nonnumeric budgets fail.

    Returns
    -------
    LatencyEvidenceReport
        Ordered field refusals and observed p95/labels. Self-digest, counters,
        percentile positivity/order, overhead, local boundary and context are
        checked without changing the report.

    Raises
    ------
    ValueError
        Budget is invalid, JSON parsing fails or the root is not an object.
        Budget validation happens before report I/O.
    OSError, UnicodeError
        The report cannot be read as UTF-8.
    OverflowError, TypeError, RecursionError
        Legacy report numeric conversion or canonical JSON serialisation fails.

    Notes
    -----
    The digest covers parsed canonical fields, not raw file bytes. Duplicate
    JSON keys keep the last value; unknown fields are retained in that digest.
    Timestamp must be explicitly UTC but need not be recent. Labels/context
    are declarations, not hardware or isolation verification. No timing replay,
    source signature, operator qualification or production admission occurs.
    """
    if max_e2e_p95_us is not None:
        if isinstance(max_e2e_p95_us, bool) or not isinstance(max_e2e_p95_us, int | float):
            raise ValueError("max_e2e_p95_us must be a finite non-negative number")
        try:
            budget = float(max_e2e_p95_us)
        except OverflowError as exc:
            raise ValueError("max_e2e_p95_us must be a finite non-negative number") from exc
        if not math.isfinite(budget) or budget < 0.0:
            raise ValueError("max_e2e_p95_us must be a finite non-negative number")
    payload = _load_json(path)
    errors: list[str] = []
    if payload.get("schema_version") != E2E_LATENCY_SCHEMA_VERSION:
        errors.append(f"schema_version must be {E2E_LATENCY_SCHEMA_VERSION!r}")
    declared_digest = payload.get("payload_sha256")
    if not isinstance(declared_digest, str) or len(declared_digest) != 64:
        errors.append("payload_sha256 must be a SHA-256 hex digest")
    elif _payload_digest(payload) != declared_digest.lower():
        errors.append("payload_sha256 does not match latency evidence payload")

    iterations = payload.get("iterations")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        errors.append("iterations must be a positive integer")
    warmup = payload.get("warmup")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        errors.append("warmup must be a non-negative integer")

    target = payload.get("target_hardware")
    if not isinstance(target, dict):
        target = {}
        errors.append("target_hardware must be an object")

    hardware_id = _qualified_string(target.get("id"))
    hardware_class = _qualified_string(target.get("class"))
    rt_kernel = _qualified_string(target.get("rt_kernel"))
    if require_target_hardware:
        if hardware_id is None:
            errors.append("target_hardware.id must identify the measured hardware")
        if hardware_class is None:
            errors.append("target_hardware.class must identify the hardware class")
        if rt_kernel is None:
            errors.append("target_hardware.rt_kernel must identify scheduler or RT-kernel evidence")

    kernel_values = _validate_percentiles(payload, "kernel_only_us", errors)
    e2e_values = _validate_percentiles(payload, "e2e_us", errors)
    p95_us = e2e_values["p95"]
    if p95_us is None:
        pass
    elif max_e2e_p95_us is not None and p95_us > max_e2e_p95_us:
        errors.append(f"e2e_us.p95 exceeds admission threshold {max_e2e_p95_us}")

    overhead = _finite_positive_number(payload.get("e2e_overhead_factor"))
    if overhead is None:
        errors.append("e2e_overhead_factor must be a positive finite number")
    elif kernel_values["p50"] is not None and e2e_values["p50"] is not None:
        expected = e2e_values["p50"] / max(kernel_values["p50"], 0.1)
        if not math.isclose(overhead, round(expected, 1), rel_tol=0.0, abs_tol=0.1):
            errors.append("e2e_overhead_factor must match p50 e2e/kernel ratio")

    claim_status = payload.get("claim_status")
    if claim_status != E2E_LATENCY_CLAIM_BOUNDARY:
        errors.append("claim_status must preserve the canonical local-evidence boundary")

    _validate_benchmark_context(payload, errors)

    return LatencyEvidenceReport(
        status="pass" if not errors else "fail",
        errors=tuple(errors),
        p95_us=p95_us,
        target_hardware_id=hardware_id,
        target_hardware_class=hardware_class,
        rt_kernel=rt_kernel,
    )


def main() -> None:
    """Read CLI arguments and print local-report diagnostics to standard output.

    Parameters
    ----------
    None
        Arguments come from sys.argv: report, --allow-local-unqualified,
        --max-e2e-p95-us (microseconds) and --json-out.

    Returns
    -------
    None
        No file is written; JSON output retains the existing six result fields.

    Raises
    ------
    SystemExit
        0 for a valid report or help, 1 for reported field refusals and 2 for
        argparse syntax errors. Native API/I/O exceptions remain uncaught and
        cause script failure; invalid budgets raise ValueError before I/O.

    Notes
    -----
    A pass is bounded metadata validation, not a hardware real-time guarantee.
    """
    parser = argparse.ArgumentParser(description="Validate E2E control-latency evidence")
    parser.add_argument("report", type=Path)
    parser.add_argument("--allow-local-unqualified", action="store_true")
    parser.add_argument("--max-e2e-p95-us", type=float, default=None)
    parser.add_argument("--json-out", action="store_true")
    args = parser.parse_args()

    result = validate_e2e_latency_evidence(
        args.report,
        require_target_hardware=not args.allow_local_unqualified,
        max_e2e_p95_us=args.max_e2e_p95_us,
    )
    payload = {
        "status": result.status,
        "errors": list(result.errors),
        "p95_us": result.p95_us,
        "target_hardware_id": result.target_hardware_id,
        "target_hardware_class": result.target_hardware_class,
        "rt_kernel": result.rt_kernel,
    }
    if args.json_out:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"E2E latency evidence: {result.status}")
        for error in result.errors:
            print(f"ERROR {error}")
    raise SystemExit(0 if result.status == "pass" else 1)


if __name__ == "__main__":
    main()

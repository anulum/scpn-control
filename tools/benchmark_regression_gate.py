#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — benchmark regression gate for Python/Rust latency parity paths.

"""File-based benchmark gate with compatibility exports for its pure record APIs.

The gate compares declared reports and baselines under explicit ratio policy.
It performs no new benchmark, source authentication, or scientific admission.
Use gate for complete local domain checks, compare only with prevalidated records.
Evidence-only reports rejection with exit zero; file and output errors still fail.
"""

from __future__ import annotations

import argparse
import errno
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scpn_control.benchmark_records import require_recorded_campaign
from tools.benchmark_gate_policy import (
    BASELINE_SCHEMA as BASELINE_SCHEMA,
)
from tools.benchmark_gate_policy import (
    LOWER_IS_BETTER_TOKENS as LOWER_IS_BETTER_TOKENS,
)
from tools.benchmark_gate_policy import (
    REPORT_SCHEMA as REPORT_SCHEMA,
)
from tools.benchmark_gate_policy import (
    _coerce_float as _coerce_float,
)
from tools.benchmark_gate_policy import (
    _metric_block_errors as _metric_block_errors,
)
from tools.benchmark_gate_policy import (
    _payload_digest as _payload_digest,
)
from tools.benchmark_gate_policy import (
    canonical_metrics_digest as canonical_metrics_digest,
)
from tools.benchmark_gate_policy import (
    metric_direction as metric_direction,
)
from tools.benchmark_gate_policy import (
    parse_thresholds as parse_thresholds,
)
from tools.benchmark_gate_policy import (
    resolve_threshold as resolve_threshold,
)
from tools.benchmark_gate_policy import (
    validate_report as validate_report,
)
from tools.benchmark_gate_policy import (
    verify_baseline_integrity as verify_baseline_integrity,
)
from tools.benchmark_gate_verdict import (
    VERDICT_SCHEMA as VERDICT_SCHEMA,
)
from tools.benchmark_gate_verdict import (
    Finding as Finding,
)
from tools.benchmark_gate_verdict import (
    compare as compare,
)
from tools.benchmark_gate_verdict import (
    gate as gate,
)
from tools.benchmark_gate_verdict import (
    hardware_mismatch as hardware_mismatch,
)
from validation.report_output_paths import checked_report_destination

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    """Read UTF-8 JSON and require a top-level object.

    IO, decoding, JSON syntax and object-shape failures propagate to the
    command's fixed refusal path. This preserves standard JSON decoding;
    duplicate-key or source authentication is not supplied by this loader.
    """
    data: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("metric document must be a JSON object")
    return data


def load_thresholds_file(path: Path) -> dict[str, dict[str, float]]:
    """Read binary TOML and validate its dimensionless ratio policy.

    Parameters
    ----------
    path : Path
        Caller-selected path; relative spelling follows cwd.

    Returns
    -------
    dict
        Parsed positive finite ratios with a required default table.

    Raises
    ------
    OSError, ValueError
        File, TOML or policy inspection fails; no policy is substituted.
    """
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    return parse_thresholds(raw)


def main(argv: list[str] | None = None) -> int:
    """Compare selected files and optionally write a protected JSON verdict.

    Parameters
    ----------
    argv : list of str or None
        argparse tokens, or None for process arguments. Report/baseline paths
        are required; policy defaults to the source-tree threshold file.

    Returns
    -------
    int
        0 for admission, or for a generated rejection in evidence-only mode.
        1 for rejected strict verdicts and supported read/write/custody failures.
        Evidence-only never converts an input or output error into success.

    Raises
    ------
    SystemExit
        argparse help or argument refusal.
    AttributeError, TypeError
        Malformed decoded provenance beyond the validated metric map contract.

    Notes
    -----
    Selected report, baseline and policy aliases cannot be output destinations,
    including resolved and existing hard-link identities. Unrelated output may
    be replaced with sorted UTF-8 JSON and a trailing newline. Sequential checks
    provide no lock, coherent snapshot or atomic replacement. A failed write
    may leave partial output. Persistent destinations require recorded-campaign
    custody. No benchmark, host authentication or physical admission is supplied.
    """
    parser = argparse.ArgumentParser(description="Benchmark regression gate.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "regression_thresholds.toml",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--generated-utc", default="")
    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help="Report rejected verdicts with exit zero (input loading errors still fail; for generic CI "
        "runners whose CPU differs from the declared-hardware baseline).",
    )
    args = parser.parse_args(argv)
    if args.json_out is not None:
        try:
            # A destination whose symbolic links never resolve is refused here.
            # Path resolution raised for one before Python 3.13 and returns it
            # unresolved since, so the loop is asked for explicitly.
            try:
                args.json_out.stat()
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise
            require_recorded_campaign(args.json_out, repository_root=REPO_ROOT)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"benchmark gate FAILED: output custody error: {exc}", file=sys.stderr)
            return 1

    # Missing-evidence fail-closed: an absent report or baseline is a failure,
    # not a skip.
    for label, path in (("report", args.report), ("baseline", args.baseline)):
        if not path.is_file():
            print(f"benchmark gate FAILED: {label} not found at {path}", file=sys.stderr)
            return 1

    try:
        report = _load_json(args.report)
        baseline = _load_json(args.baseline)
    except (OSError, ValueError, RecursionError):
        print("benchmark gate FAILED: cannot read JSON metric documents", file=sys.stderr)
        return 1
    try:
        thresholds = load_thresholds_file(args.thresholds)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"benchmark gate FAILED: threshold policy error: {exc}", file=sys.stderr)
        return 1

    verdict = gate(report, baseline, thresholds, generated_utc=args.generated_utc)

    if args.json_out is not None:
        try:
            target = checked_report_destination(args.json_out, inputs=[args.report, args.baseline, args.thresholds])
            encoded = json.dumps(verdict, indent=2, sort_keys=True, allow_nan=False) + "\n"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(encoded, encoding="utf-8")
        except (OSError, ValueError, RuntimeError):
            print("benchmark gate FAILED: cannot write JSON verdict", file=sys.stderr)
            return 1

    if not verdict["passed"]:
        stream = sys.stdout if args.evidence_only else sys.stderr
        header = "benchmark gate findings (evidence-only):" if args.evidence_only else "benchmark gate FAILED:"
        print(header, file=stream)
        for finding in verdict["findings"]:
            scope = "/".join(p for p in (finding["benchmark"], finding["language"], finding["metric"]) if p)
            print(f"  - [{finding['kind']}] {scope}: {finding['detail']}", file=stream)
        if not args.evidence_only:
            return 1
        return 0
    compared = sum(len(m) for b in baseline.get("benchmarks", {}).values() for m in b.get("languages", {}).values())
    print(f"benchmark gate passed: {compared} baseline metric(s) within policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

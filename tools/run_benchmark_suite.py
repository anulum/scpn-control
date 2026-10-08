#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — polyglot benchmark suite runner.
"""Run the registered Python/Rust comparison benchmarks and emit a gate report.

This produces the canonical `scpn-control.benchmark-regression.v1` document that
`tools/benchmark_regression_gate.py` consumes: per-benchmark, per-language
latency percentiles (p50/p95/p99) and throughput, plus the run provenance the
gate records (CPU model, Rust release profile, commit digest, affinity, load,
peak RSS, and whether the Rust backend was available).

Measurement is delegated to the existing, validated benchmark harnesses under
`benchmarks/` so the suite does not re-implement timing; the runner only
normalises their per-language statistics into the gate schema and stamps
provenance. Baseline updates are deliberately outside this command; only the
digest-verifying explicit promotion tool may change a regression baseline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
RUST_CARGO = REPO_ROOT / "scpn-control-rs" / "Cargo.toml"


def _ensure_repo_on_path(search_path: list[str] | None = None) -> None:
    """Insert the source tree and harness root once for standalone invocation.

    Parameters
    ----------
    search_path : list of str or None, optional
        List to update, or sys.path when omitted. Existing entries retain order.

    Notes
    -----
    This changes only import lookup. It does not install packages or authenticate
    loaded sources; scientific dependencies must already be available.
    """
    target = sys.path if search_path is None else search_path
    for path in (REPO_ROOT, REPO_ROOT / "src"):
        if str(path) not in target:
            target.insert(0, str(path))


_ensure_repo_on_path()

import scpn_control
from scpn_control.benchmark_records import CAMPAIGN_ENV, require_recorded_campaign
from tools.benchmark_suite_metrics import (
    AdapterResult,
    BenchmarkBlock,
    SuiteReport,
    SuiteReportBody,
    check_suite_settings,
    checked_adapter_result,
    checked_mapping,
)
from tools.benchmark_suite_metrics import (
    BenchmarkSuiteRefusal as BenchmarkSuiteRefusal,
)
from tools.benchmark_suite_metrics import (
    _language_metrics as _language_metrics,
)
from tools.benchmark_suite_metrics import (
    _payload_digest as _payload_digest,
)
from tools.inventory_file_output import InventoryOutputError, publish_guarded_outputs
from validation.report_output_paths import checked_report_destination


def _protected_suite_inputs() -> tuple[Path, ...]:
    """List producer, harness, manifest and baseline files protected from publication.

    Returns
    -------
    tuple of pathlib.Path
        Existing and reserved selected input spellings. Native input aliases are
        checked before timing and again before complete byte publication.

    Notes
    -----
    These sequential checks require cooperating writers; they are not a sandbox.
    """
    return (
        Path(__file__),
        Path(__file__).with_name("benchmark_suite_metrics.py"),
        Path(__file__).with_name("benchmark_suite_provenance.py"),
        RUST_CARGO,
        REPO_ROOT / "benchmarks/bench_capacitor_bank_energy.py",
        REPO_ROOT / "benchmarks/baselines/capacitor_bank.json",
        REPO_ROOT / "benchmarks/regression_thresholds.toml",
        Path(scpn_control.__file__).parent / "control/capacitor_bank_state.py",
        REPO_ROOT / "scpn-control-rs/crates/control-control/src/capacitor_bank.rs",
        REPO_ROOT / "scpn-control-rs/Cargo.lock",
    )


import platform

from tools.benchmark_suite_provenance import (
    _affinity as _affinity,
)
from tools.benchmark_suite_provenance import (
    _cpu_model as _cpu_model,
)
from tools.benchmark_suite_provenance import (
    _git_commit as _git_commit,
)
from tools.benchmark_suite_provenance import (
    _loadavg as _loadavg,
)
from tools.benchmark_suite_provenance import (
    _peak_rss_mb as _peak_rss_mb,
)
from tools.benchmark_suite_provenance import (
    _rust_release_profile as _rust_release_profile,
)

REPORT_SCHEMA = "scpn-control.benchmark-regression.v1"


def _load_control_benchmark_module(module_file_name: str) -> ModuleType:
    """Load an owning CONTROL harness by its source-tree file path.

    Parameters
    ----------
    module_file_name : str
        Harness filename under this source root's benchmarks directory.

    Returns
    -------
    types.ModuleType
        Loaded or process-cached harness, independent of a sibling benchmarks package.

    Raises
    ------
    ImportError
        Import machinery cannot create a file specification.
    OSError, Exception
        Native module loading or module initialization fails.

    Notes
    -----
    Callers select maintained harness names; the process cache is not an immutable
    source snapshot or authentication mechanism.
    """
    harness_path = Path(__file__).resolve().parent.parent / "benchmarks" / module_file_name
    module_name = f"_scpn_control_bench_{harness_path.stem}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, harness_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load CONTROL benchmark harness at {harness_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _capacitor_bank_discharge(steps: int, warmup: int) -> AdapterResult:
    """Delegate actual capacitor timing and normalize the consumed statistics.

    Parameters
    ----------
    steps : int
        Number of measured samples.
    warmup : int
        Number of unrecorded samples before timing.

    Returns
    -------
    dict of str to object
        Per-language metrics, the declared three-energy parity and Rust availability.

    Raises
    ------
    BenchmarkSuiteRefusal
        Consumed harness maps or latency statistics are malformed.
    Exception
        Native harness execution fails.

    Notes
    -----
    The existing kernel uses200discharge steps at1e-7seconds. Timing, RLC dynamics
    and parity arithmetic stay in the harness; declarations do not grant admission.
    """
    bench = _load_control_benchmark_module("bench_capacitor_bank_energy.py")
    _measure = bench._measure

    observed: object = _measure(steps=steps, warmup=warmup, discharge_steps=200, dt_s=1.0e-7)
    measured = checked_mapping(observed, "harness result")
    languages_raw = checked_mapping(measured.get("languages"), "harness languages")
    python = checked_mapping(languages_raw.get("python"), "Python measurement")
    languages = {"python": _language_metrics(checked_mapping(python.get("stats"), "Python statistics"))}
    rust = languages_raw.get("rust")
    if rust is not None:
        block = checked_mapping(rust, "Rust measurement")
        languages["rust"] = _language_metrics(checked_mapping(block.get("stats"), "Rust statistics"))
    return checked_adapter_result(
        {
            "languages": languages,
            "cross_language_parity": languages_raw.get("cross_language_parity"),
            "rust_available": rust is not None,
        }
    )


# name -> callable(steps, warmup) -> {"languages": {...}, ...}
BENCHMARKS: dict[str, Callable[[int, int], object]] = {
    "capacitor_bank_discharge": _capacitor_bank_discharge,
}


def run_suite(*, names: list[str], steps: int, warmup: int, evidence_class: str, generated_utc: str) -> SuiteReport:
    """Measure selected adapters and build a finite digest-bound V1 report.

    Parameters
    ----------
    names : list of str
        Nonempty distinct registered benchmark names, in execution order.
    steps : int
        Positive integer measured sample count, excluding bool.
    warmup : int
        Nonnegative integer unrecorded sample count, excluding bool.
    evidence_class : str
        Nonblank caller declaration, not an independently qualified claim level.
    generated_utc : str
        Caller-supplied timezone-aware UTC receipt string.

    Returns
    -------
    dict of str to object
        V1 report with checked consumed metrics, declared host context, optional
        campaign and SHA256. Production claims remain false.

    Raises
    ------
    BenchmarkSuiteRefusal
        Settings or consumed adapter declarations are inconsistent or nonfinite.
    Exception
        Measurement, provenance capture or finite serialization fails.

    Notes
    -----
    This API writes no report or baseline. Real timing varies with execution
    conditions. A digest authenticates neither source, host nor measured physics.
    """
    check_suite_settings(names, steps, warmup, evidence_class, generated_utc, registered=BENCHMARKS)
    load_start = _loadavg()
    benchmarks: dict[str, BenchmarkBlock] = {}
    rust_seen = False
    for name in names:
        result = checked_adapter_result(BENCHMARKS[name](steps, warmup))
        rust_seen = rust_seen or result["rust_available"]
        benchmarks[name] = {
            "languages": result["languages"],
            "cross_language_parity": result.get("cross_language_parity"),
        }
    report: SuiteReportBody = {
        "schema_version": REPORT_SCHEMA,
        "campaign_id": os.environ.get(CAMPAIGN_ENV),
        "generated_utc": generated_utc,
        "evidence_class": evidence_class,
        "production_claim_allowed": False,
        "provenance": {
            "cpu_model": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "rust_release_profile": _rust_release_profile(),
            "rust_backend": "present" if rust_seen else "absent",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "commit": _git_commit(),
            "cpu_affinity": _affinity(),
            "loadavg_start": load_start,
            "loadavg_end": _loadavg(),
            "peak_rss_mb": _peak_rss_mb(),
        },
        "settings": {"steps": steps, "warmup": warmup},
        "benchmarks": benchmarks,
    }
    return SuiteReport(**report, payload_sha256=_payload_digest(report))


def main(argv: list[str] | None = None) -> int:
    """Execute a selected suite and publish one protected complete JSON report.

    Parameters
    ----------
    argv : list of str or None, optional
        Process arguments when omitted; default400samples and40warmup iterations.

    Returns
    -------
    int
        Zero after a completed suite/publication; one after authored refusal or
        a caught native settings, measurement or output failure.

    Raises
    ------
    SystemExit
        Argument parsing requested help or refused malformed options/unknown names.

    Notes
    -----
    No output path prints finite sorted JSON. Persistent paths retain recorded-
        campaign custody; input aliases refuse before timing. Publication uses sibling
    staging/fsync/replacement with handled-failure recovery, not crash atomicity or
    hostile concurrency protection. Baseline promotion remains separate.
    """
    parser = argparse.ArgumentParser(description="Run the polyglot benchmark suite.")
    parser.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--evidence-class", default="local_regression")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    unknown = [b for b in args.benchmarks if b not in BENCHMARKS]
    if unknown:
        parser.error(f"unknown benchmark(s): {', '.join(unknown)}")
    generated_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        check_suite_settings(
            args.benchmarks, args.steps, args.warmup, args.evidence_class, generated_utc, registered=BENCHMARKS
        )
        if args.json_out is not None:
            checked_report_destination(args.json_out, inputs=_protected_suite_inputs())
            require_recorded_campaign(args.json_out, repository_root=REPO_ROOT)
        report = run_suite(
            names=args.benchmarks,
            steps=args.steps,
            warmup=args.warmup,
            evidence_class=args.evidence_class,
            generated_utc=generated_utc,
        )
        payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        if args.json_out is not None:
            publish_guarded_outputs(((args.json_out, payload),), protected_files=_protected_suite_inputs())
        else:
            print(payload.decode("utf-8"), end="")
    except (BenchmarkSuiteRefusal, InventoryOutputError) as exc:
        print(f"benchmark suite FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception:
        print("benchmark suite FAILED: inputs, measurement or output could not be processed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

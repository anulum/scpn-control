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
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from scpn_control.benchmark_records import CAMPAIGN_ENV, require_recorded_campaign

try:
    import resource
except ModuleNotFoundError:  # pragma: no cover - resource is Unix-only (absent on Windows).
    resource = None  # type: ignore[assignment]

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on Python 3.10 CI.
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
RUST_CARGO = REPO_ROOT / "scpn-control-rs" / "Cargo.toml"


def _ensure_repo_on_path(search_path: list[str] | None = None) -> None:
    """Put the repo root and `src` on the import path for standalone runs.

    Makes the in-repo `benchmarks` harnesses and the `scpn_control` package
    importable when this runner is invoked directly rather than via pytest.
    """
    target = sys.path if search_path is None else search_path
    for path in (REPO_ROOT, REPO_ROOT / "src"):
        if str(path) not in target:
            target.insert(0, str(path))


_ensure_repo_on_path()

REPORT_SCHEMA = "scpn-control.benchmark-regression.v1"


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _affinity() -> list[int] | None:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return None


def _loadavg() -> list[float] | None:
    try:
        return list(os.getloadavg())
    except (OSError, AttributeError):  # getloadavg is absent on Windows.
        return None


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _rust_release_profile() -> dict[str, Any]:
    """Read the workspace [profile.release] so flags are recorded, not guessed."""
    try:
        with RUST_CARGO.open("rb") as handle:
            data = tomllib.load(handle)
        profile: dict[str, Any] = data.get("profile", {}).get("release", {})
        return profile
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _peak_rss_mb() -> float:
    if resource is None:  # pragma: no cover - Windows has no resource module.
        return 0.0
    # ru_maxrss is kibibytes on Linux.
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def _language_metrics(stats: dict[str, Any]) -> dict[str, float]:
    """Map a harness stats block to the gate's metric names."""
    mean_us = float(stats["mean_us"])
    throughput = 1.0e6 / mean_us if mean_us > 0.0 else 0.0
    return {
        "p50_us": float(stats["median_us"]),
        "p95_us": float(stats["p95_us"]),
        "p99_us": float(stats["p99_us"]),
        "throughput_ops_s": throughput,
    }


def _load_control_benchmark_module(module_file_name: str) -> ModuleType:
    """Load a CONTROL ``benchmarks/*.py`` harness by file path.

    Dual-home environments often put SCPN-FUSION-CORE on ``sys.path``. FUSION
    ships a regular ``benchmarks`` package (``__init__.py``), which shadows the
    CONTROL harness namespace and breaks ``import benchmarks.*``. Path-based
    loading keeps this runner bound to the CONTROL tree.
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


def _capacitor_bank_discharge(steps: int, warmup: int) -> dict[str, Any]:
    """Polyglot capacitor-bank discharge benchmark via the existing harness."""
    bench = _load_control_benchmark_module("bench_capacitor_bank_energy.py")
    _measure = bench._measure

    measured = _measure(steps=steps, warmup=warmup, discharge_steps=200, dt_s=1.0e-7)
    languages_raw = measured["languages"]
    languages: dict[str, Any] = {"python": _language_metrics(languages_raw["python"]["stats"])}
    rust = languages_raw.get("rust")
    parity = languages_raw.get("cross_language_parity")
    if rust is not None:
        languages["rust"] = _language_metrics(rust["stats"])
    return {
        "languages": languages,
        "cross_language_parity": parity,
        "rust_available": rust is not None,
    }


# name -> callable(steps, warmup) -> {"languages": {...}, ...}
BENCHMARKS: dict[str, Callable[[int, int], dict[str, Any]]] = {
    "capacitor_bank_discharge": _capacitor_bank_discharge,
}


def _payload_digest(payload: dict[str, Any]) -> str:
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialised).hexdigest()


def run_suite(*, names: list[str], steps: int, warmup: int, evidence_class: str, generated_utc: str) -> dict[str, Any]:
    """Run selected benchmark adapters and assemble a digest-bound report."""
    load_start = _loadavg()
    benchmarks: dict[str, Any] = {}
    rust_seen = False
    for name in names:
        result = BENCHMARKS[name](steps, warmup)
        rust_seen = rust_seen or result.get("rust_available", False)
        benchmarks[name] = {
            "languages": result["languages"],
            "cross_language_parity": result.get("cross_language_parity"),
        }
    report = {
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
    report["payload_sha256"] = _payload_digest(report)
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the command-line benchmark suite and optionally write its report."""
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
    if args.json_out is not None:
        require_recorded_campaign(args.json_out, repository_root=REPO_ROOT)

    report = run_suite(
        names=args.benchmarks,
        steps=args.steps,
        warmup=args.warmup,
        evidence_class=args.evidence_class,
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json_out is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

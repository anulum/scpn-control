#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — libFuzzer campaign orchestrator and fail-closed triage gate.
"""Run the scpn-control-rs libFuzzer targets and emit a triage evidence report.

This is the nightly/self-hosted fuzzing driver for the Rust parser, numeric
adapter, vector-kernel, and FFI surfaces under ``scpn-control-rs/fuzz``. It is
not a normal commit blocker: a fast build check belongs in CI, while the
time-boxed fuzzing run belongs in a scheduled workflow.

The driver:

* copies the tracked seed corpus into the working corpus for each target,
* records the Rust toolchain, cargo-fuzz version, target triple, sanitiser
  configuration, per-target executed-unit counts, and run duration,
* hashes the seed corpus so evidence is bound to the exact seeds that ran,
* collects any libFuzzer crash/leak/timeout reproducer artefacts, and
* fails closed: the campaign is only admitted when every requested target ran
  and no target crashed, timed out, leaked, or produced an artefact.

The pure-logic helpers (`seed_corpus_manifest`, `parse_libfuzzer_stats`,
`collect_crash_artifacts`, `triage`, `assemble_report`) carry no subprocess or
filesystem-execution side effects so they can be unit-tested directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUST_ROOT = REPO_ROOT / "scpn-control-rs"
FUZZ_ROOT = RUST_ROOT / "fuzz"
SEEDS_ROOT = FUZZ_ROOT / "seeds"
CORPUS_ROOT = FUZZ_ROOT / "corpus"
ARTIFACTS_ROOT = FUZZ_ROOT / "artifacts"

SCHEMA_VERSION = "scpn-control.fuzz-campaign-evidence.v1"

# Each target maps to the untrusted/high-volume Rust surface it exercises.
FUZZ_TARGETS: dict[str, str] = {
    "config_json": "parser: reactor-configuration JSON deserialisation",
    "vmec_import": "parser: VMEC-like fixed-boundary text import",
    "bout_stability": "parser: BOUT++ linear-stability output",
    "capacitor_bank": "safety-critical numeric adapter / PyO3 FFI: series-RLC discharge ledger",
    "kuramoto_kernel": "vector kernel / PyO3 FFI: Kuramoto-Sakaguchi phase step",
}

# libFuzzer reproducer artefacts. Any of these prefixes under artifacts/<target>
# is a triage failure: the surface admitted an input that crashed, leaked,
# timed out, or hit an out-of-memory / slow-unit guard.
ARTEFACT_PREFIXES = ("crash-", "leak-", "timeout-", "oom-", "slow-unit-")


@dataclass(frozen=True)
class TargetRun:
    """Outcome of a single libFuzzer target run."""

    name: str
    surface: str
    duration_s: float
    exit_code: int
    executed_units: int
    average_exec_per_sec: int
    peak_rss_mb: int
    artefacts: list[str] = field(default_factory=list)

    @property
    def crashed(self) -> bool:
        return self.exit_code != 0 or bool(self.artefacts)


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def seed_corpus_manifest(seeds_root: Path, targets: list[str]) -> dict[str, Any]:
    """Hash the tracked seed corpus for each target.

    The aggregate digest is the SHA-256 of the sorted ``name:filehash`` lines so
    that adding, removing, or mutating any seed changes the bound evidence.
    """
    manifest: dict[str, Any] = {}
    for target in targets:
        target_dir = seeds_root / target
        files: dict[str, str] = {}
        total_bytes = 0
        if target_dir.is_dir():
            for seed in sorted(target_dir.iterdir()):
                if seed.is_file():
                    files[seed.name] = sha256_file(seed)
                    total_bytes += seed.stat().st_size
        aggregate_src = "\n".join(f"{name}:{h}" for name, h in sorted(files.items()))
        aggregate = hashlib.sha256(aggregate_src.encode()).hexdigest()
        manifest[target] = {
            "seed_count": len(files),
            "total_bytes": total_bytes,
            "files": files,
            "aggregate_sha256": aggregate,
        }
    return manifest


def parse_libfuzzer_stats(stdout: str) -> dict[str, int]:
    """Extract the ``stat::`` summary printed by ``-print_final_stats=1``."""
    wanted = {
        "stat::number_of_executed_units": "executed_units",
        "stat::average_exec_per_sec": "average_exec_per_sec",
        "stat::peak_rss_mb": "peak_rss_mb",
    }
    out: dict[str, int] = {key: 0 for key in wanted.values()}
    for line in stdout.splitlines():
        line = line.strip()
        for prefix, key in wanted.items():
            if line.startswith(prefix):
                token = line.split(":")[-1].strip()
                try:
                    out[key] = int(token)
                except ValueError:
                    out[key] = 0
    return out


def collect_crash_artifacts(artifacts_root: Path, target: str) -> list[str]:
    """List libFuzzer reproducer artefacts for a target, if any."""
    target_dir = artifacts_root / target
    if not target_dir.is_dir():
        return []
    found: list[str] = []
    for item in sorted(target_dir.iterdir()):
        if item.is_file() and item.name.startswith(ARTEFACT_PREFIXES):
            found.append(item.name)
    return found


def triage(runs: list[TargetRun], requested: list[str]) -> tuple[bool, list[str]]:
    """Fail closed: admit the campaign only with complete, crash-free evidence."""
    failures: list[str] = []
    ran = {run.name for run in runs}
    for target in requested:
        if target not in ran:
            failures.append(f"{target}: no run evidence recorded (missing-evidence fail-closed)")
    for run in runs:
        if run.exit_code != 0:
            failures.append(f"{run.name}: non-zero libFuzzer exit code {run.exit_code}")
        if run.artefacts:
            failures.append(f"{run.name}: reproducer artefacts present: {', '.join(run.artefacts)}")
    return (not failures, failures)


def assemble_report(
    *,
    runs: list[TargetRun],
    requested: list[str],
    seeds: dict[str, Any],
    toolchain: dict[str, str],
    target_triple: str,
    sanitizer: str,
    max_total_time_s: int,
    evidence_class: str,
    generated_utc: str,
) -> dict[str, Any]:
    """Assemble the campaign evidence document and embedded triage verdict."""
    passed, failures = triage(runs, requested)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": generated_utc,
        "evidence_class": evidence_class,
        "production_claim_allowed": False,
        "toolchain": toolchain,
        "target_triple": target_triple,
        "sanitizer": sanitizer,
        "settings": {
            "max_total_time_s": max_total_time_s,
            "requested_targets": list(requested),
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "seeds": seeds,
        "targets": [asdict(run) for run in runs],
        "triage": {
            "passed": passed,
            "failures": failures,
        },
    }
    payload["payload_sha256"] = _payload_digest(payload)
    return payload


def _payload_digest(payload: dict[str, Any]) -> str:
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialised).hexdigest()


# ── Subprocess / filesystem side of the driver (not imported by unit tests) ──


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)


def toolchain_metadata() -> tuple[dict[str, str], str]:
    """Record the nightly Rust toolchain, cargo-fuzz version, and target triple."""
    rustc = _run(["rustup", "run", "nightly", "rustc", "--version"]).stdout.strip()
    cargo_fuzz = _run(["cargo", "fuzz", "--version"]).stdout.strip()
    host = _run(["rustup", "run", "nightly", "rustc", "-vV"]).stdout
    triple = ""
    for line in host.splitlines():
        if line.startswith("host:"):
            triple = line.split(":", 1)[1].strip()
    toolchain = {"rustc": rustc, "cargo_fuzz": cargo_fuzz}
    return toolchain, triple or "unknown"


def _seed_corpus(target: str) -> None:
    corpus_dir = CORPUS_ROOT / target
    corpus_dir.mkdir(parents=True, exist_ok=True)
    seed_dir = SEEDS_ROOT / target
    if seed_dir.is_dir():
        for seed in seed_dir.iterdir():
            if seed.is_file():
                dest = corpus_dir / seed.name
                if not dest.exists():
                    dest.write_bytes(seed.read_bytes())


def run_target(target: str, max_total_time_s: int, rss_limit_mb: int) -> TargetRun:
    """Build (implicitly) and run one fuzz target for a bounded wall-clock time."""
    _seed_corpus(target)
    cmd = [
        "cargo",
        "+nightly",
        "fuzz",
        "run",
        target,
        str(CORPUS_ROOT / target),
        "--",
        f"-max_total_time={max_total_time_s}",
        f"-rss_limit_mb={rss_limit_mb}",
        "-print_final_stats=1",
    ]
    start = time.perf_counter()
    proc = _run(cmd, cwd=RUST_ROOT)
    duration = time.perf_counter() - start
    stats = parse_libfuzzer_stats(proc.stdout + "\n" + proc.stderr)
    artefacts = collect_crash_artifacts(ARTIFACTS_ROOT, target)
    return TargetRun(
        name=target,
        surface=FUZZ_TARGETS.get(target, "unknown"),
        duration_s=round(duration, 3),
        exit_code=proc.returncode,
        executed_units=stats["executed_units"],
        average_exec_per_sec=stats["average_exec_per_sec"],
        peak_rss_mb=stats["peak_rss_mb"],
        artefacts=artefacts,
    )


def build_all() -> int:
    """Build every fuzz target so a compile break fails fast and cheaply."""
    proc = subprocess.run(["cargo", "+nightly", "fuzz", "build"], cwd=RUST_ROOT, check=False)
    return proc.returncode


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Commercial license available -->",
        "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
        "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
        "<!-- ORCID: 0009-0009-3560-0851 -->",
        "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
        "<!-- SCPN Control — libFuzzer campaign evidence report. -->",
        "",
        "# libFuzzer Campaign Evidence",
        "",
        f"- Generated UTC: `{report['generated_utc']}`",
        f"- Evidence class: `{report['evidence_class']}`",
        f"- Toolchain: `{report['toolchain'].get('rustc', '')}`",
        f"- cargo-fuzz: `{report['toolchain'].get('cargo_fuzz', '')}`",
        f"- Target triple: `{report['target_triple']}`",
        f"- Sanitiser: `{report['sanitizer']}`",
        f"- Max total time per target (s): `{report['settings']['max_total_time_s']}`",
        f"- Triage passed: `{report['triage']['passed']}`",
        "",
        "## Per-target outcomes",
        "",
        "| Target | Surface | Executed units | exec/s | Peak RSS MB | Duration s | Exit | Artefacts |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report["targets"]:
        lines.append(
            f"| `{run['name']}` | {run['surface']} | {run['executed_units']} | "
            f"{run['average_exec_per_sec']} | {run['peak_rss_mb']} | {run['duration_s']} | "
            f"{run['exit_code']} | {len(run['artefacts'])} |"
        )
    lines.append("")
    if report["triage"]["failures"]:
        lines.append("## Triage failures")
        lines.append("")
        for failure in report["triage"]["failures"]:
            lines.append(f"- {failure}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the scpn-control-rs libFuzzer campaign.")
    parser.add_argument(
        "--targets",
        nargs="*",
        default=list(FUZZ_TARGETS),
        help="Subset of fuzz targets to run (default: all).",
    )
    parser.add_argument("--max-total-time", type=int, default=300)
    parser.add_argument("--rss-limit-mb", type=int, default=4096)
    parser.add_argument("--evidence-class", default="nightly_regression")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only build the targets (fast CI smoke check); do not fuzz.",
    )
    args = parser.parse_args(argv)

    unknown = [t for t in args.targets if t not in FUZZ_TARGETS]
    if unknown:
        parser.error(f"unknown fuzz target(s): {', '.join(unknown)}")

    build_rc = build_all()
    if build_rc != 0:
        print("fuzz build failed", file=sys.stderr)
        return build_rc
    if args.build_only:
        print("fuzz targets built successfully (build-only)")
        return 0

    toolchain, triple = toolchain_metadata()
    runs = [run_target(t, args.max_total_time, args.rss_limit_mb) for t in args.targets]
    report = assemble_report(
        runs=runs,
        requested=args.targets,
        seeds=seed_corpus_manifest(SEEDS_ROOT, args.targets),
        toolchain=toolchain,
        target_triple=triple,
        sanitizer="AddressSanitizer (cargo-fuzz default)",
        max_total_time_s=args.max_total_time,
        evidence_class=args.evidence_class,
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(_markdown(report), encoding="utf-8")
    if args.json_out is None and args.markdown_out is None:
        print(json.dumps(report, indent=2, sort_keys=True))

    if not report["triage"]["passed"]:
        print("FUZZ TRIAGE FAILED:", file=sys.stderr)
        for failure in report["triage"]["failures"]:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"fuzz triage passed: {len(runs)} target(s) clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

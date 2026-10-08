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

The seed and artefact helpers read the filesystem; they do not run children.
Triage checks submitted run records, not their authenticity or the executed
binary. The seed manifest hashes tracked inputs, not the evolving working
corpus. A successful verdict supplies no scientific or production admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
FUZZ_TOOLCHAIN = "nightly-2026-08-18"

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
    """Record one native target outcome for campaign triage.

    Parameters
    ----------
    name, surface
        Target identifier and its exact FUZZ_TARGETS description.
    duration_s
        Elapsed seconds including any implicit compilation. Triage requires a
        positive finite value.
    exit_code
        Native process status; zero is required for admission.
    executed_units
        Parsed native iteration count; triage requires a positive integer.
    average_exec_per_sec, peak_rss_mb
        Nonnegative integer summary values. Missing or malformed summaries
        parse to zero; these two fields alone do not prove execution.
    artefacts
        Observed reproducer filenames. Any entry refuses admission.

    Notes
    -----
    Construction stores supplied values without runtime validation. Triage
    validates the record. A record is not proof of a binary or source hash.
    """

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
        """Return whether the process failed or a reproducer was recorded.

        Zero executed units and malformed statistics are checked by triage;
        they do not by themselves change this process-outcome property.
        """
        return self.exit_code != 0 or bool(self.artefacts)


def sha256_file(path: Path) -> str:
    """Hash the complete bytes of one file.

    Parameters
    ----------
    path
        File opened through Path.read_bytes, following normal filesystem links.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256.

    Raises
    ------
    OSError
        Reading the file fails. No path or size admission is performed.
    """
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def seed_corpus_manifest(seeds_root: Path, targets: list[str]) -> dict[str, Any]:
    """Describe and hash the supplied tracked seed directories.

    Parameters
    ----------
    seeds_root
        Parent of the per-target seed directories.
    targets
        Directory names to inspect. Callers must supply registered targets;
        this helper does not restrict names or authenticate Git membership.

    Returns
    -------
    dict[str, Any]
        Per-target file digests, seed_count, total_bytes and aggregate_sha256.
        The aggregate hashes sorted name:digest lines. Missing directories
        produce zero counts and the empty-input digest. Links follow ordinary
        Path semantics. The evolving working corpus is not included.

    Raises
    ------
    OSError
        Directory inspection, byte reading or stat fails.
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
    """Read the three supported native final-summary counters.

    Parameters
    ----------
    stdout
        Combined standard output and standard error text.

    Returns
    -------
    dict[str, int]
        executed_units, average_exec_per_sec and peak_rss_mb. Missing or
        malformed values become zero; the last matching line wins. Parsing
        preserves signed integers. Triage separately checks admissible values;
        a parsed summary does not prove source identity or native success.
    """
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
    """List immediate reproducer filenames for one target.

    Parameters
    ----------
    artifacts_root
        Parent of the per-target artefact directories.
    target
        Target directory name, validated by the owning run entrypoint.

    Returns
    -------
    list[str]
        Sorted regular-file names beginning with ARTEFACT_PREFIXES. A missing
        directory gives an empty list. Nested files and ordinary corpus names
        are excluded; contents are not hashed and links follow Path semantics.

    Raises
    ------
    OSError
        Filesystem inspection fails.
    """
    target_dir = artifacts_root / target
    if not target_dir.is_dir():
        return []
    found: list[str] = []
    for item in sorted(target_dir.iterdir()):
        if item.is_file() and item.name.startswith(ARTEFACT_PREFIXES):
            found.append(item.name)
    return found


def triage(runs: list[TargetRun], requested: list[str]) -> tuple[bool, list[str]]:
    """Validate a nonempty campaign's submitted native run records.

    Parameters
    ----------
    runs
        TargetRun records to examine; names must occur exactly once, match
        requested targets and carry their registered surface descriptions.
    requested
        Nonempty, unique registered target identifiers.

    Returns
    -------
    tuple[bool, list[str]]
        Passed flag and accumulated failures in inspection order. Every target
        needs a run with positive executed_units, positive finite duration_s,
        nonnegative integer rate/RSS, zero exit and no reproducer artefacts.
        Empty, duplicate, unknown, extra or incomplete evidence fails closed.
        Counts exclude booleans. No subprocess is run and metadata, binaries,
        seed contents, source identity and sanitiser status are not certified.
    """
    failures: list[str] = []
    if not requested:
        failures.append("no targets requested (empty-campaign fail-closed)")
    if len(set(requested)) != len(requested):
        failures.append("duplicate requested targets")
    for target in requested:
        if target not in FUZZ_TARGETS:
            failures.append(f"{target}: unknown requested target")
    ran = {run.name for run in runs}
    if len(ran) != len(runs):
        failures.append("duplicate target run evidence")
    for target in requested:
        if target not in ran:
            failures.append(f"{target}: no run evidence recorded (missing-evidence fail-closed)")
    for run in runs:
        if run.name not in requested:
            failures.append(f"{run.name}: unrequested target run evidence")
        if run.name not in FUZZ_TARGETS or run.surface != FUZZ_TARGETS[run.name]:
            failures.append(f"{run.name}: unknown or mismatched target surface")
        if type(run.executed_units) is not int or run.executed_units <= 0:
            failures.append(f"{run.name}: no positive executed-unit evidence")
        if not _valid_duration(run.duration_s):
            failures.append(f"{run.name}: run duration must be positive and finite")
        if any(type(value) is not int or value < 0 for value in (run.average_exec_per_sec, run.peak_rss_mb)):
            failures.append(f"{run.name}: invalid nonnegative run statistics")
        if run.exit_code != 0:
            failures.append(f"{run.name}: non-zero libFuzzer exit code {run.exit_code}")
        if run.artefacts:
            failures.append(f"{run.name}: reproducer artefacts present: {', '.join(run.artefacts)}")
    return (not failures, failures)


def _valid_duration(value: float) -> bool:
    """Check an elapsed-second value without accepting booleans or overflow."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value > 0.0
    except OverflowError:
        return False


def _validate_limits(max_total_time_s: int, rss_limit_mb: int) -> None:
    """Reject noninteger, boolean or nonpositive native resource limits."""
    for name, value in (("max_total_time_s", max_total_time_s), ("rss_limit_mb", rss_limit_mb)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


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
    """Build a JSON-compatible report from caller-supplied evidence.

    Parameters
    ----------
    runs, requested
        Records and target selection passed to triage.
    seeds
        Seed metadata, normally returned by seed_corpus_manifest.
    toolchain
        Supplied Rust and cargo-fuzz version strings.
    target_triple, sanitizer
        Reported compilation host and sanitiser description.
    max_total_time_s
        Reported libFuzzer time limit in seconds, excluding compilation.
    evidence_class, generated_utc
        Caller-supplied label and UTC timestamp text.

    Returns
    -------
    dict[str, Any]
        Version-one report, submitted targets, triage verdict and digest of
        sorted compact JSON before payload_sha256 is added. Host platform and
        Python version are observed locally; other provenance is carried
        without authentication. production_claim_allowed is always false.

    Notes
    -----
    This function does not rerun targets, validate source bindings or sanitiser
    settings, or inspect seed files. Its digest binds the submitted payload;
    it is not a producer signature or a correctness certificate.
    """
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
    """Hash sorted compact JSON without adding or removing payload fields."""
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialised).hexdigest()


# ── Subprocess / filesystem side of the driver (not imported by unit tests) ──


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Wait for a native argv without a shell and return its captured text.

    The child inherits this environment. No Python timeout is supplied; native
    libFuzzer limits do not bound compilation or a child that ignores them.
    Launch and cwd errors propagate as OSError.
    """
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)


def toolchain_metadata() -> tuple[dict[str, str], str]:
    """Read versions and the host triple from three native commands.

    Returns
    -------
    tuple[dict[str, str], str]
        Pinned nightly rustc --version, cargo fuzz --version and rustc -vV
        observations from successful commands with nonempty output and one
        nonempty host line. This does not qualify the pinned hosted toolchain.

    Raises
    ------
    OSError
        Starting a version command fails. No build is requested.
    RuntimeError
        A version command fails, produces empty output or omits its host.
    """
    rustc = _toolchain_output(["rustup", "run", FUZZ_TOOLCHAIN, "rustc", "--version"], "Rust compiler version")
    cargo_fuzz = _toolchain_output(["cargo", f"+{FUZZ_TOOLCHAIN}", "fuzz", "--version"], "cargo-fuzz version")
    host = _toolchain_output(["rustup", "run", FUZZ_TOOLCHAIN, "rustc", "-vV"], "Rust compiler host")
    triples = [line.split(":", 1)[1].strip() for line in host.splitlines() if line.startswith("host:")]
    if len(triples) != 1 or not triples[0]:
        raise RuntimeError("Rust compiler host metadata is missing or ambiguous")
    toolchain = {"rustc": rustc, "cargo_fuzz": cargo_fuzz}
    return toolchain, triples[0]


def _toolchain_output(cmd: list[str], observation: str) -> str:
    """Require successful, nonempty output from one real version command.

    Parameters
    ----------
    cmd
        Native argv passed unchanged to _run, without requesting a build.
    observation
        Authored name of the observation used in a failure message.

    Returns
    -------
    str
        Stripped standard output from a zero-status native command.

    Raises
    ------
    OSError
        Native launch fails.
    RuntimeError
        The native status is nonzero or standard output is empty.
    """
    result = _run(cmd)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"{observation} command failed or produced no output")
    return result.stdout.strip()


def _seed_corpus(target: str) -> None:
    """Copy absent tracked seed names into the existing working corpus.

    Existing destination names and additional evolved corpus inputs are kept.
    Their bytes are not authenticated against the tracked seed manifest.
    Filesystem errors propagate; this helper does not execute the fuzzer.
    """
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
    """Run one registered target with positive native resource limits.

    Parameters
    ----------
    target
        Identifier in FUZZ_TARGETS.
    max_total_time_s, rss_limit_mb
        Positive integer libFuzzer limits, in seconds and megabytes. Booleans
        and nonintegers are refused before creating a corpus or native child.

    Returns
    -------
    TargetRun
        Native status, elapsed seconds, parsed summary and currently observed
        reproducer names. The pinned nightly fuzz run can implicitly compile; the
        libFuzzer limit does not include build time. Existing corpus and
        artefacts are reused. Missing statistics parse to zero and triage
        refuses a zero executed-unit count.

    Raises
    ------
    ValueError
        The target or resource limits are invalid.
    OSError
        Corpus inspection, seed copying or native launch fails.

    Notes
    -----
    This executes Rust harnesses directly, not their Python FFI entrypoints.
    No binary/source provenance or scientific correctness is established.
    """
    if target not in FUZZ_TARGETS:
        raise ValueError("unknown fuzz target")
    _validate_limits(max_total_time_s, rss_limit_mb)
    _seed_corpus(target)
    cmd = [
        "cargo",
        f"+{FUZZ_TOOLCHAIN}",
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
    """Invoke cargo fuzz build with the pinned nightly for every harness.

    Returns
    -------
    int
        Native cargo status. A successful build does not execute a campaign.

    Raises
    ------
    OSError
        Cargo cannot be started or RUST_ROOT cannot be entered.

    Notes
    -----
    Compilation is not bounded by the per-target libFuzzer time setting.
    """
    proc = subprocess.run(["cargo", f"+{FUZZ_TOOLCHAIN}", "fuzz", "build"], cwd=RUST_ROOT, check=False)
    return proc.returncode


def _markdown(report: dict[str, Any]) -> str:
    """Render the supplied report metadata, target rows and triage failures.

    Values are not escaped for Markdown. The report must have the structure
    produced by assemble_report; malformed structures raise native key/type
    errors. This formats in memory and does not write files.
    """
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
    """Build selected campaign harnesses, execute them and report triage.

    Parameters
    ----------
    argv
        CLI arguments; None reads sys.argv. Targets must be nonempty, unique
        and registered, with positive time and RSS limits. Invalid selection
        refuses before build or corpus work, including in build-only mode.

    Returns
    -------
    int
        Zero for a successful build-only request or an admitted run report;
        the native build status on build failure; one for failed triage or
        unavailable campaign toolchain metadata, which refuses before build.
        Build-only runs no targets and emits no campaign evidence.

    Raises
    ------
    SystemExit
        Argument parsing displays help or refuses invalid selection with two.
    OSError
        Native launch or filesystem/report writing fails.

    Notes
    -----
    Roots resolve from this script, while output paths resolve from the caller
    cwd. Explicit outputs create parents and overwrite existing paths; output
    aliases and protected destinations are not validated here. Version-only
    observations and report hashes do not authenticate executed sources.
    The default campaign is 300 seconds for each of five targets, plus build
    time. This command is separate from the fast Python preflight wrapper.
    """
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
    if not args.targets:
        parser.error("at least one fuzz target is required")
    if len(set(args.targets)) != len(args.targets):
        parser.error("duplicate fuzz targets are not allowed")
    if args.max_total_time <= 0:
        parser.error("--max-total-time must be a positive integer")
    if args.rss_limit_mb <= 0:
        parser.error("--rss-limit-mb must be a positive integer")

    toolchain: dict[str, str] = {}
    triple = ""
    if not args.build_only:
        try:
            toolchain, triple = toolchain_metadata()
        except (OSError, RuntimeError):
            print("fuzz toolchain metadata failed", file=sys.stderr)
            return 1

    build_rc = build_all()
    if build_rc != 0:
        print("fuzz build failed", file=sys.stderr)
        return build_rc
    if args.build_only:
        print("fuzz targets built successfully (build-only)")
        return 0

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

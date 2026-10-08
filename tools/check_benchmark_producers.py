#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Benchmark producer custody audit
"""Fail when a benchmark producer lacks an explicit output-custody class."""

from __future__ import annotations

import argparse
import ast
import sys
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "benchmarks" / "producer_registry.toml"
SCHEMA = "scpn-control.benchmark-producer-registry.v1"
CATEGORIES = (
    "recorded_guard",
    "append_stream",
    "temporary_scratch",
    "stdout_or_build_product",
    "custody_infrastructure",
)


def _python_calls_recorded_guard(source: str) -> bool:
    """Parse Python text and find a call with the guard's literal name or attribute.

    Alias/import resolution, reachability, arguments and runtime custody are not
    checked. A commented/imported name alone is insufficient; syntax errors
    propagate without executing any producer.
    """
    tree = ast.parse(source)
    return any(
        isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == "require_recorded_campaign"
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == "require_recorded_campaign"
        )
        for node in ast.walk(tree)
    )


def _discover(repository_root: Path) -> set[str]:
    """Collect existing files in the maintained Python/Rust glob and explicit-path scope.

    Python benchmark/scripts/tools/validation owners and selected Rust benches,
    examples and the transport binary are included regardless of Git tracking.
    The auditor itself is excluded. Missing subdirectories add no paths; normal
    Path.is_file semantics follow file symlinks. This is lexical inventory, not
    a producer execution or a complete dynamic consumer/export discovery.
    """
    paths: set[Path] = set()
    paths.update((repository_root / "benchmarks").glob("*.py"))
    paths.update((repository_root / "scripts").glob("*benchmark*.py"))
    paths.update((repository_root / "tools").glob("*benchmark*.py"))
    paths.discard(repository_root / "tools" / "check_benchmark_producers.py")
    paths.update((repository_root / "validation").glob("benchmark_*.py"))
    paths.update(
        repository_root / "validation" / name
        for name in (
            "code_to_code_benchmark.py",
            "control_benchmark_suite.py",
            "free_boundary_tracking_acceptance.py",
            "scpn_pid_mpc_benchmark.py",
        )
    )
    rust_root = repository_root / "scpn-control-rs"
    paths.update((rust_root / "benches").glob("bench_*.rs"))
    paths.update((rust_root / "crates" / "control-control" / "examples").glob("bench_*.rs"))
    paths.add(rust_root / "crates" / "control-python" / "src" / "bin" / "transport_bench.rs")
    return {path.relative_to(repository_root).as_posix() for path in paths if path.is_file()}


def _documentation_command_findings(repository_root: Path, guarded_paths: set[str]) -> list[str]:
    """Scan sorted public Markdown lines for direct commands naming guarded producers.

    README and docs Markdown are inspected except changelog.md and internal path
    components. Literal Python/cargo/CMD prefixes and runner payload markers are
    recognised without parsing shell, Markdown fences or wrapper execution.
    Missing documents are skipped; read/decode errors propagate. Findings use
    repository-relative paths and one-based lines, ordered by document/line/path.
    """
    documents = [repository_root / "README.md"]
    documents.extend(
        path
        for path in sorted((repository_root / "docs").rglob("*.md"))
        if path.name != "changelog.md" and "internal" not in path.parts
    )
    allowed_payload_markers = (
        "-- python ",
        "-- .venv/bin/python ",
        "-- cargo ",
        '"--", "python"',
        '"--", "cargo"',
    )
    findings: list[str] = []
    for document in documents:
        if not document.is_file():
            continue
        for line_number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), start=1):
            for producer_path in sorted(guarded_paths):
                if producer_path not in line:
                    continue
                prefix = line.split(producer_path, 1)[0]
                looks_executable = "python" in prefix or "cargo run" in prefix or "CMD" in prefix
                if looks_executable and not any(marker in prefix for marker in allowed_payload_markers):
                    relative_document = document.relative_to(repository_root).as_posix()
                    findings.append(
                        f"public benchmark command bypasses recorded runner: "
                        f"{relative_document}:{line_number}: {producer_path}"
                    )
    return findings


def audit_registry(registry_path: Path = REGISTRY, repository_root: Path = REPO_ROOT) -> list[str]:
    """Read a TOML registry and existing repository, returning ordered lexical-custody findings.

    Registry and root paths are caller-relative; the root resolves to an existing
    directory or raises FileNotFoundError/ValueError. Schema/category/duplicate
    errors and inventory differences remain findings, including registry entries
    absent from discovery. Only discovered registered sources are inspected.
    Python guards use AST call-name presence; Rust, append, scratch and custody
    classes use literal source markers. These checks do not establish live guard
    execution, immutable output, numerical performance or scientific admission.
    IO, UTF-8, TOML and Python syntax errors propagate. No producer is executed or
    file mutated; multi-file reads are not a coherent concurrent snapshot.
    """
    repository_root = repository_root.resolve(strict=True)
    if not repository_root.is_dir():
        raise ValueError("benchmark producer repository root must be a directory")
    with registry_path.open("rb") as handle:
        raw: dict[str, Any] = tomllib.load(handle)
    findings: list[str] = []
    if raw.get("schema_version") != SCHEMA:
        findings.append(f"schema_version must be {SCHEMA}")

    ownership: dict[str, str] = {}
    for category in CATEGORIES:
        entries = raw.get(category)
        if not isinstance(entries, list) or any(not isinstance(entry, str) for entry in entries):
            findings.append(f"{category} must be an array of paths")
            continue
        for entry in entries:
            if entry in ownership:
                findings.append(f"{entry} appears in both {ownership[entry]} and {category}")
            ownership[entry] = category

    discovered = _discover(repository_root)
    for relative_path in sorted(discovered - ownership.keys()):
        findings.append(f"unclassified benchmark producer: {relative_path}")
    for relative_path in sorted(ownership.keys() - discovered):
        findings.append(f"registry path is not a discovered benchmark producer: {relative_path}")

    for relative_path, category in sorted(ownership.items()):
        if relative_path not in discovered:
            continue
        path = repository_root / relative_path
        source = path.read_text(encoding="utf-8")
        if category == "recorded_guard":
            guarded = (
                'env::var("SCPN_BENCHMARK_CAMPAIGN_ID")' in source
                and "persistent output requires tools/run_recorded_benchmark.py" in source
                if path.suffix == ".rs"
                else _python_calls_recorded_guard(source)
            )
            if not guarded:
                findings.append(f"recorded producer lacks campaign guard: {relative_path}")
        elif category == "append_stream":
            if '.open("a"' not in source and 'open(path, "a"' not in source:
                findings.append(f"append-stream producer does not visibly append: {relative_path}")
        elif category == "temporary_scratch":
            if "tempfile" not in source and "RESULTS_FILE" not in source:
                findings.append(f"temporary producer has no explicit scratch destination: {relative_path}")
        elif category == "custody_infrastructure":
            if "benchmark" not in source.lower():
                findings.append(f"custody infrastructure lacks benchmark contract text: {relative_path}")
    guarded_paths = {path for path, category in ownership.items() if category == "recorded_guard"}
    findings.extend(_documentation_command_findings(repository_root, guarded_paths))
    return findings


def main(argv: list[str] | None = None) -> int:
    """Return zero for classified selected inventory, one for findings or supported inspection errors.

    --repo selects the repository; absent --registry selects its benchmarks TOML.
    Explicit registry paths are caller-relative. Expected IO/decode/TOML/syntax/
    root errors use authored stderr, not tracebacks. Parser help/usage retain
    exits zero/two. Reported count is actual lexical discovery, not runtime or
    scientific qualification; this CLI never runs benchmarks or writes files.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--registry", type=Path)
    args = parser.parse_args(argv)
    try:
        args.repo.stat()
        findings = audit_registry(args.registry or args.repo / "benchmarks/producer_registry.toml", args.repo)
    except (OSError, UnicodeError, ValueError, SyntaxError, RuntimeError) as exc:
        print(f"benchmark producer registry FAILED: {exc}", file=sys.stderr)
        return 1
    if findings:
        print("benchmark producer registry FAILED:", file=sys.stderr)
        for finding in findings:
            print(f"  - {finding}", file=sys.stderr)
        return 1
    print(f"benchmark producer registry passed: {len(_discover(args.repo))} producers classified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

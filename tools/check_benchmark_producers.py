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
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 CI uses tomli.
    import tomli as tomllib

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
    """Return whether Python source invokes the persistent-output guard."""
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
    paths: set[Path] = set()
    paths.update((repository_root / "benchmarks").glob("*.py"))
    paths.update((repository_root / "scripts").glob("*benchmark*.py"))
    paths.update((repository_root / "tools").glob("*benchmark*.py"))
    paths.discard(repository_root / "tools" / "check_benchmark_producers.py")
    paths.update((repository_root / "validation").glob("benchmark_*.py"))
    paths.update(
        repository_root / "validation" / name
        for name in ("code_to_code_benchmark.py", "control_benchmark_suite.py", "scpn_pid_mpc_benchmark.py")
    )
    rust_root = repository_root / "scpn-control-rs"
    paths.update((rust_root / "benches").glob("bench_*.rs"))
    paths.update((rust_root / "crates" / "control-control" / "examples").glob("bench_*.rs"))
    paths.add(rust_root / "crates" / "control-python" / "src" / "bin" / "transport_bench.rs")
    return {path.relative_to(repository_root).as_posix() for path in paths if path.is_file()}


def _documentation_command_findings(repository_root: Path, guarded_paths: set[str]) -> list[str]:
    """Find public Python benchmark commands that bypass immutable custody."""
    documents = [repository_root / "README.md"]
    documents.extend(
        path
        for path in (repository_root / "docs").rglob("*.md")
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
            for producer_path in guarded_paths:
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
    """Return deterministic findings for producer inventory or custody drift."""
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
        path = repository_root / relative_path
        if not path.is_file():
            continue
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
    """Audit the registry and print a concise producer-custody verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    args = parser.parse_args(argv)
    try:
        findings = audit_registry(args.registry)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"benchmark producer registry FAILED: {exc}", file=sys.stderr)
        return 1
    if findings:
        print("benchmark producer registry FAILED:", file=sys.stderr)
        for finding in findings:
            print(f"  - {finding}", file=sys.stderr)
        return 1
    print(f"benchmark producer registry passed: {len(_discover(REPO_ROOT))} producers classified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

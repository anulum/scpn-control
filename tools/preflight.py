#!/usr/bin/env python3
# SCPN Control — Local CI preflight
# Mirrors every CI gate so failures are caught before push.
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0

from __future__ import annotations

import os
import subprocess  # noqa: S404
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parent.parent
RUST_DIR = ROOT / "scpn-control-rs"
_PY = sys.executable

GATES: list[tuple[str, list[str], Path | None]] = [
    ("ruff check", [_PY, "-m", "ruff", "check", "src/scpn_control/"], None),
    ("ruff format", [_PY, "-m", "ruff", "format", "--check", "src/scpn_control/", "tests/"], None),
    ("version-sync", [_PY, "tools/check_version_sync.py"], None),
    ("mypy", [_PY, "-m", "mypy"], None),
    ("module-linkage", [_PY, "tools/check_test_module_linkage.py"], None),
    ("pytest", [_PY, "-m", "pytest", "tests/", "-x", "--tb=short", "-q"], None),
    ("bandit", [_PY, "-m", "bandit", "-r", "src/scpn_control/", "-c", "pyproject.toml", "-ll"], None),
    ("cargo fmt", ["cargo", "fmt", "--check"], RUST_DIR),
    ("cargo clippy", ["cargo", "clippy", "--", "-D", "warnings"], RUST_DIR),
    ("cargo test", ["cargo", "test", "--workspace"], RUST_DIR),
]

# CI enforces 85% on Linux; Windows hits different platform branches,
# so local threshold is 80% to avoid false negatives.
LOCAL_COV_THRESHOLD = "80"

COVERAGE_PYTEST: tuple[str, list[str], Path | None] = (
    "pytest (coverage)",
    [_PY, "-m", "pytest", "tests/", "-x", "--tb=short", "-q",
     "--cov=scpn_control", f"--cov-fail-under={LOCAL_COV_THRESHOLD}"],
    None,
)

RUST_GATES = {"cargo fmt", "cargo clippy", "cargo test"}
TEST_GATES = {"pytest", "cargo test"}


def run_gate(name: str, cmd: list[str], cwd: Path | None) -> bool:
    t0 = time.monotonic()
    result = subprocess.run(  # noqa: S603
        cmd,
        cwd=cwd or ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.monotonic() - t0
    if result.returncode == 0:
        print(f"  PASS  {name} ({elapsed:.1f}s)")
        return True
    print(f"  FAIL  {name} ({elapsed:.1f}s)")
    tail = 10
    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    if out:
        for line in out.splitlines()[-tail:]:
            print(f"        {line}")
    if err:
        for line in err.splitlines()[-tail:]:
            print(f"        {line}")
    return False


def main() -> int:
    coverage = "--coverage" in sys.argv
    skip_tests = "--no-tests" in sys.argv
    skip_rust = "--no-rust" in sys.argv

    gates = list(GATES)
    if skip_tests:
        gates = [(n, c, d) for n, c, d in gates if n not in TEST_GATES]
    if skip_rust:
        gates = [(n, c, d) for n, c, d in gates if n not in RUST_GATES]
    if coverage:
        gates = [(n, c, d) if n != "pytest" else COVERAGE_PYTEST for n, c, d in gates]

    flags = []
    if coverage:
        flags.append("coverage")
    if skip_rust:
        flags.append("no-rust")
    if skip_tests:
        flags.append("no-tests")
    suffix = f" [{', '.join(flags)}]" if flags else ""
    print(f"preflight: {len(gates)} gates{suffix}")
    print()

    t_start = time.monotonic()
    failed: list[str] = []

    for name, cmd, cwd in gates:
        if not run_gate(name, cmd, cwd):
            failed.append(name)
            break

    elapsed = time.monotonic() - t_start
    print()
    if failed:
        print(f"BLOCKED: {', '.join(failed)} ({elapsed:.1f}s)")
        return 1
    print(f"ALL CLEAR: ready to push ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

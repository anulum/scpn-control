#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Project: SCPN Control
# Description: Local CI preflight gates.

from __future__ import annotations

import json
import os
import re
import subprocess  # noqa: S404
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parent.parent
RUST_DIR = ROOT / "scpn-control-rs"
_PY = sys.executable

# CI matrix minimum Python version — deps in COMPAT_REQUIREMENTS must support this.
CI_MIN_PYTHON = (3, 10)

# Requirements files used by ALL Python versions in the CI matrix (including 3.10).
# These must have version pins compatible with CI_MIN_PYTHON.
COMPAT_REQUIREMENTS = [
    "requirements/ci-deps.txt",
    "requirements/ci-test.txt",
]

# All requirements files used in CI (for hash-pin validation).
CI_REQUIREMENTS = [
    "requirements/ci-deps.txt",
    "requirements/ci-test.txt",
    "requirements/ci-lint.txt",
    "requirements/ci-viz.txt",
    "requirements/ci-notebook.txt",
    "requirements/ci-build.txt",
    "requirements/ci-security.txt",
    "requirements/ci-audit.txt",
]

GATES: list[tuple[str, list[str], Path | None]] = [
    ("ruff check", [_PY, "-m", "ruff", "check", "src/scpn_control/"], None),
    ("ruff format", [_PY, "-m", "ruff", "format", "--check", "src/scpn_control/", "tests/"], None),
    ("version-sync", [_PY, "tools/check_version_sync.py"], None),
    ("mypy", [_PY, "-m", "mypy"], None),
    ("test-quality-policy", [_PY, "tools/check_test_quality_policy.py"], None),
    ("generated-traceability", [_PY, "tools/check_generated_traceability.py"], None),
    ("release-evidence", [_PY, "-m", "scpn_control.cli", "validate-release-evidence"], None),
    ("benchmark-regression", [_PY, "validation/validate_benchmark_regression_gates.py"], None),
    ("module-linkage", [_PY, "tools/check_test_module_linkage.py"], None),
    ("pytest", [_PY, "-m", "pytest", "tests/", "-x", "--tb=short", "-q"], None),
    ("bandit", [_PY, "-m", "bandit", "-r", "src/scpn_control/", "-c", "pyproject.toml", "-ll"], None),
    ("cargo fmt", ["cargo", "fmt", "--check"], RUST_DIR),
    ("cargo clippy", ["cargo", "clippy", "--", "-D", "warnings"], RUST_DIR),
    ("cargo test", ["cargo", "test", "--workspace"], RUST_DIR),
]

LOCAL_COV_THRESHOLD = "93"

COVERAGE_PYTEST: tuple[str, list[str], Path | None] = (
    "pytest (coverage)",
    [
        _PY,
        "-m",
        "pytest",
        "tests/",
        "-x",
        "--tb=short",
        "-q",
        "--cov=scpn_control",
        f"--cov-fail-under={LOCAL_COV_THRESHOLD}",
    ],
    None,
)

RUST_GATES = {"cargo fmt", "cargo clippy", "cargo test"}
TEST_GATES = {"pytest", "cargo test"}


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    source_paths = [str(ROOT / "src"), str(ROOT)]
    existing = env.get("PYTHONPATH")
    if existing:
        source_paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(source_paths)
    return env


def run_gate(name: str, cmd: list[str], cwd: Path | None) -> bool:
    if name == "release-evidence":
        return run_release_evidence_gate()

    t0 = time.monotonic()
    result = subprocess.run(  # noqa: S603
        cmd,
        cwd=cwd or ROOT,
        env=_subprocess_env(),
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


def _print_command_failure(result: subprocess.CompletedProcess[str]) -> None:
    tail = 10
    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    if out:
        for line in out.splitlines()[-tail:]:
            print(f"        {line}")
    if err:
        for line in err.splitlines()[-tail:]:
            print(f"        {line}")


def run_release_evidence_gate() -> bool:
    """Generate and admit the top-level release evidence report."""

    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="scpn-control-release-evidence-") as tmp:
        tmp_path = Path(tmp)
        report_path = tmp_path / "release_evidence_report.json"
        admission_path = tmp_path / "release_evidence_admission.json"
        with report_path.open("w", encoding="utf-8") as report_handle:
            generate = subprocess.run(  # noqa: S603
                [_PY, "-m", "scpn_control.cli", "validate", "--json-out"],
                cwd=ROOT,
                env=_subprocess_env(),
                stdout=report_handle,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="replace",
            )
        if generate.returncode != 0:
            elapsed = time.monotonic() - t0
            print(f"  FAIL  release-evidence ({elapsed:.1f}s)")
            _print_command_failure(generate)
            return False
        with admission_path.open("w", encoding="utf-8") as admission_handle:
            admit = subprocess.run(  # noqa: S603
                [
                    _PY,
                    "-m",
                    "scpn_control.cli",
                    "validate-release-evidence",
                    str(report_path),
                    "--json-out",
                ],
                cwd=ROOT,
                env=_subprocess_env(),
                stdout=admission_handle,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="replace",
            )
        elapsed = time.monotonic() - t0
        if admit.returncode == 0:
            print(f"  PASS  release-evidence ({elapsed:.1f}s)")
            return True
        print(f"  FAIL  release-evidence ({elapsed:.1f}s)")
        _print_command_failure(admit)
        return False


def _requirements_changed() -> bool:
    """True if any requirements/ file is in the staged or unstaged diff."""
    result = subprocess.run(  # noqa: S603
        ["git", "diff", "--name-only", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    staged = subprocess.run(  # noqa: S603
        ["git", "diff", "--cached", "--name-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    changed = (result.stdout or "") + (staged.stdout or "")
    return any("requirements/" in line for line in changed.splitlines())


def _parse_pinned_packages(req_path: Path) -> list[tuple[str, str]]:
    """Extract (name, version) from pinned requirements file."""
    pkgs = []
    for line in req_path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([a-zA-Z0-9_-]+)==([0-9][0-9a-zA-Z.]*)", line.strip())
        if m:
            pkgs.append((m.group(1), m.group(2)))
    return pkgs


def run_python_compat_check() -> bool:
    """Verify pinned deps in ci-deps.txt support CI_MIN_PYTHON."""
    t0 = time.monotonic()
    errors: list[str] = []

    for req_file in COMPAT_REQUIREMENTS:
        req_path = ROOT / req_file
        if not req_path.exists():
            continue
        for pkg, ver in _parse_pinned_packages(req_path):
            try:
                url = f"https://pypi.org/pypi/{pkg}/{ver}/json"
                with urlopen(url, timeout=10) as resp:  # noqa: S310
                    data = json.loads(resp.read())
                requires_python = data.get("info", {}).get("requires_python", "")
                if not requires_python:
                    continue
                m_req = re.search(r">=\s*(\d+)\.(\d+)", requires_python)
                if m_req:
                    req_major, req_minor = int(m_req.group(1)), int(m_req.group(2))
                    if (req_major, req_minor) > CI_MIN_PYTHON:
                        errors.append(
                            f"{pkg}=={ver} requires Python >={req_major}.{req_minor}"
                            f", CI matrix includes {CI_MIN_PYTHON[0]}.{CI_MIN_PYTHON[1]}"
                            f" ({req_file})"
                        )
            except Exception:
                pass

    elapsed = time.monotonic() - t0
    if errors:
        print(f"  FAIL  python-compat ({elapsed:.1f}s)")
        for e in errors:
            print(f"        {e}")
        return False
    print(f"  PASS  python-compat ({elapsed:.1f}s)")
    return True


def run_hash_pin_check() -> bool:
    """Validate CI requirements via pip --require-hashes --dry-run."""
    t0 = time.monotonic()
    errors: list[str] = []

    for req_file in CI_REQUIREMENTS:
        req_path = ROOT / req_file
        if not req_path.exists():
            continue
        result = subprocess.run(  # noqa: S603
            [_PY, "-m", "pip", "install", "--require-hashes", "--dry-run", "-r", str(req_path)],
            cwd=ROOT,
            env=_subprocess_env(),
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            last_line = stderr.splitlines()[-1] if stderr else "unknown error"
            errors.append(f"{req_file}: {last_line}")

    elapsed = time.monotonic() - t0
    if errors:
        print(f"  FAIL  hash-pin check ({elapsed:.1f}s)")
        for e in errors:
            print(f"        {e}")
        return False
    print(f"  PASS  hash-pin check ({elapsed:.1f}s)")
    return True


def main() -> int:
    coverage = "--coverage" in sys.argv
    skip_tests = "--no-tests" in sys.argv
    skip_rust = "--no-rust" in sys.argv
    force_hashes = "--hashes" in sys.argv

    gates = list(GATES)
    if skip_tests:
        gates = [(n, c, d) for n, c, d in gates if n not in TEST_GATES]
    if skip_rust:
        gates = [(n, c, d) for n, c, d in gates if n not in RUST_GATES]
    if coverage:
        gates = [(n, c, d) if n != "pytest" else COVERAGE_PYTEST for n, c, d in gates]

    # Decide whether to run requirements validation.
    # Always run python-compat (fast, ~5s PyPI queries).
    # Run hash-pin check only if requirements changed or --hashes forced.
    do_hashes = force_hashes or _requirements_changed()

    flags = []
    if coverage:
        flags.append("coverage")
    if skip_rust:
        flags.append("no-rust")
    if skip_tests:
        flags.append("no-tests")
    suffix = f" [{', '.join(flags)}]" if flags else ""

    extra_count = 2 if do_hashes else 1
    print(f"preflight: {len(gates) + extra_count} gates{suffix}")
    if do_hashes:
        print("  (requirements/ changed → hash-pin validation enabled)")
    print()

    t_start = time.monotonic()
    failed: list[str] = []

    # Gate 0: Python version compatibility (fast, always runs)
    if not run_python_compat_check():
        failed.append("python-compat")

    # Gate 1: Hash-pin dry-run (slow, only when requirements changed)
    if do_hashes and not failed:
        if not run_hash_pin_check():
            failed.append("hash-pin check")

    if failed:
        elapsed = time.monotonic() - t_start
        print()
        print(f"BLOCKED: {', '.join(failed)} ({elapsed:.1f}s)")
        return 1

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

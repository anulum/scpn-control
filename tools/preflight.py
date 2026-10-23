# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Local CI preflight gates.

"""Run the repository's ordered local verification gates."""

from __future__ import annotations

import os
import subprocess  # noqa: S404
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    cast(Any, sys.stdout).reconfigure(encoding="utf-8", errors="replace")
    cast(Any, sys.stderr).reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from tools.python_dependency_compatibility import CI_PYTHON_VERSIONS, python_compatibility_errors
from tools.python_dependency_metadata import PYPI_METADATA_URL, DependencyMetadataError

RUST_DIR = ROOT / "scpn-control-rs"
_PY = sys.executable

# Requirements files used by every Python version in the CI matrix.
# Their declarations must support every CI_PYTHON_VERSIONS minor floor.
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
    "requirements/ci-imas.txt",
]

GATES: list[tuple[str, list[str], Path | None]] = [
    ("changed-python-static", [_PY, "tools/changed_python_checks.py"], None),
    ("changed-studio-static", [_PY, "tools/studio_source_checks.py"], None),
    ("changed-shell-static", [_PY, "tools/shell_source_checks.py"], None),
    ("changed-cpp-static", [_PY, "tools/cpp_source_checks.py"], None),
    ("changed-rust-static", [_PY, "tools/rust_source_checks.py"], None),
    ("ruff check", [_PY, "-m", "ruff", "check", "src/scpn_control/"], None),
    (
        "ruff check debt scope",
        [_PY, "-m", "ruff", "check", "--extend-ignore", "D", "tests/", "tools/", "validation/"],
        None,
    ),
    (
        "ruff format",
        [_PY, "-m", "ruff", "format", "--check", "src/scpn_control/", "tests/", "tools/", "validation/"],
        None,
    ),
    ("docstring-debt-ratchet", [_PY, "tools/check_docstring_debt.py"], None),
    # Native documentation is enforced for the remediated scalar benchmark owner.
    (
        "scalar-benchmark-docs",
        [_PY, "-m", "ruff", "check", "validation/control_benchmark_suite.py", "tests/test_control_benchmark_suite.py"],
        None,
    ),
    (
        "scalar-benchmark-format",
        [_PY, "-m", "ruff", "format", "--check", "validation/control_benchmark_suite.py"],
        None,
    ),
    ("python-lint-contract", [_PY, "tools/check_python_lint_contract.py"], None),
    ("ci-workflow-modularity", [_PY, "tools/check_ci_workflow_modularity.py"], None),
    ("rust-toolchain-contract", [_PY, "tools/check_rust_toolchain_contract.py"], None),
    ("version-sync", [_PY, "tools/check_version_sync.py"], None),
    ("changelog-sync", [_PY, "tools/check_changelog_sync.py"], None),
    ("source-header-policy", [_PY, "tools/check_source_headers.py"], None),
    ("competitive-evidence", [_PY, "tools/check_competitive_evidence.py"], None),
    ("joss-submission", [_PY, "tools/check_joss_submission.py"], None),
    ("mypy", [_PY, "-m", "mypy"], None),
    ("test-quality-policy", [_PY, "tools/check_test_quality_policy.py"], None),
    ("docstring-quality", [_PY, "tools/run_docstring_gate.py"], None),
    ("api-contracts", [_PY, "tools/check_api_contracts.py"], None),
    ("native-api-reference", [_PY, "tools/generate_native_api_reference.py", "--check"], None),
    ("public-surface-hygiene", [_PY, "tools/check_public_surface_hygiene.py"], None),
    ("docs-internal-private", [_PY, "tools/check_docs_internal_private.py"], None),
    ("document-links", [_PY, "tools/document_link_audit.py"], None),
    ("studio-deploy-key", [_PY, "tools/check_studio_deploy_key.py"], None),
    ("studio-offline-sealing", [_PY, "tools/check_studio_offline_sealing.py"], None),
    ("studio-web-manifest", [_PY, "tools/sync_studio_web_manifest.py", "--check"], None),
    ("studio-api-docs", ["pnpm", "docs:api"], ROOT / "studio-web"),
    ("generated-traceability", [_PY, "tools/check_generated_traceability.py"], None),
    ("public-claim-ledger", [_PY, "tools/public_claim_ledger.py", "--check"], None),
    ("release-evidence", [_PY, "-m", "scpn_control.cli", "validate-release-evidence"], None),
    ("benchmark-regression", [_PY, "validation/validate_benchmark_regression_gates.py"], None),
    ("module-linkage", [_PY, "tools/check_test_module_linkage.py"], None),
    ("runtime-wiring", [_PY, "tools/check_runtime_wiring.py"], None),
    ("coverage-pragmas", [_PY, "tools/check_coverage_pragmas.py"], None),
    ("coverage-exception-ledger", [_PY, "tools/coverage_exception_ledger.py", "--check"], None),
    ("native-coverage-matrix", [_PY, "tools/native_coverage_matrix.py"], None),
    (
        "studio-source-contracts",
        [_PY, "-m", "pytest", "native_tests/test_studio_source_checks.py", "-q"],
        None,
    ),
    ("shell-source-contracts", [_PY, "-m", "pytest", "native_tests/test_shell_source_checks.py", "-q"], None),
    (
        "cpp-source-contracts",
        [
            _PY,
            "-m",
            "pytest",
            "native_tests/test_cpp_source_checks.py",
            "native_tests/test_cpp_source_protocols.py",
            "-q",
        ],
        None,
    ),
    (
        "rust-source-contracts",
        [
            _PY,
            "-m",
            "pytest",
            "native_tests/test_rust_source_ast.py",
            "native_tests/test_rust_source_enrollment.py",
            "native_tests/test_rust_source_checks.py",
            "native_tests/test_rust_source_profile.py",
            "-q",
        ],
        None,
    ),
    ("pytest", [_PY, "-m", "pytest", "tests/", "-x", "--tb=short", "-q"], None),
    ("bandit", [_PY, "-m", "bandit", "-r", "src/scpn_control/", "-c", "pyproject.toml", "-ll"], None),
    ("cargo fmt", ["cargo", "fmt", "--all", "--check"], RUST_DIR),
    (
        "cargo clippy",
        ["cargo", "clippy", "--workspace", "--all-targets", "--all-features", "--locked", "--", "-D", "warnings"],
        RUST_DIR,
    ),
    ("cargo doc", ["cargo", "doc", "--workspace", "--all-features", "--no-deps"], RUST_DIR),
    ("cargo test", ["cargo", "test", "--workspace", "--exclude", "scpn-control-rs"], RUST_DIR),
]

LOCAL_COV_THRESHOLD = "100"

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

RUST_GATES = {"cargo fmt", "cargo clippy", "cargo doc", "cargo test"}
TEST_GATES = {
    "pytest",
    "cargo test",
    "studio-source-contracts",
    "shell-source-contracts",
    "cpp-source-contracts",
    "rust-source-contracts",
}


def _subprocess_env() -> dict[str, str]:
    """Retain caller settings while prepending the checkout's Python import roots."""
    env = os.environ.copy()
    source_paths = [str(ROOT / "src"), str(ROOT)]
    existing = env.get("PYTHONPATH")
    if existing:
        source_paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(source_paths)
    return env


def run_gate(name: str, cmd: list[str], cwd: Path | None) -> bool:
    """Run one named gate and report its bounded output.

    Parameters
    ----------
    name:
        Human-readable gate identifier.
    cmd:
        Argument vector executed without a shell.
    cwd:
        Working directory override, or the repository root when ``None``.

    Returns
    -------
    bool
        ``True`` when the process exits successfully.
    """
    if name == "release-evidence":
        return run_release_evidence_gate()

    t0 = time.monotonic()
    env = _subprocess_env()
    if name == "cargo doc":
        env["RUSTDOCFLAGS"] = "-D warnings -D missing_docs"
    result = subprocess.run(  # noqa: S603
        cmd,
        cwd=cwd or ROOT,
        env=env,
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
    """Print at most ten trailing lines from each captured command output stream."""
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
    """Return whether a requirements file is staged or modified."""
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


def run_python_compat_check(root: Path = ROOT, *, metadata_base_url: str = PYPI_METADATA_URL) -> bool:
    """Check every common exact pin against all declared Python minor floors.

    Parameters
    ----------
    root : pathlib.Path
        Repository containing both required common CI lockfiles.
    metadata_base_url : str
        Explicit JSON metadata origin; defaults to certificate-verified PyPI.
        A custom origin supplies declarations, not upstream authenticity proof.

    Returns
    -------
    bool
        True only after a complete, compatible measurement. Missing inputs,
        invalid declarations and metadata failures print authored refusal and
        return False before any later preflight gate can run.
    """
    t0 = time.monotonic()
    try:
        errors = python_compatibility_errors(
            [root / path for path in COMPAT_REQUIREMENTS],
            CI_PYTHON_VERSIONS,
            metadata_base_url=metadata_base_url,
        )
    except DependencyMetadataError as error:
        print(f"  FAIL  python-compat ({time.monotonic() - t0:.1f}s)")
        print(f"        {error}")
        return False
    elapsed = time.monotonic() - t0
    if errors:
        print(f"  FAIL  python-compat ({elapsed:.1f}s)")
        for e in errors:
            print(f"        {e}")
        return False
    print(f"  PASS  python-compat ({elapsed:.1f}s)")
    return True


def run_hash_pin_check(root: Path = ROOT) -> bool:
    """Validate every required hash lock through the interpreter's pip dry run.

    Parameters
    ----------
    root : pathlib.Path
        Repository containing all CI_REQUIREMENTS files. Missing files fail
        the check instead of silently reducing its declared scope.

    Returns
    -------
    bool
        True when every required file is present and every native hash-pinned
        dry run succeeds. No package is installed by this operation.
    """
    t0 = time.monotonic()
    errors: list[str] = []

    for req_file in CI_REQUIREMENTS:
        req_path = root / req_file
        if not req_path.exists():
            errors.append(f"{req_file}: required hash lock is missing")
            continue
        result = subprocess.run(  # noqa: S603
            [_PY, "-m", "pip", "install", "--require-hashes", "--dry-run", "-r", str(req_path)],
            cwd=root,
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
    """Run selected preflight gates and return zero only when all pass."""
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

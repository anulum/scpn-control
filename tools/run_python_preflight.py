# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Fast Python preflight gate.
"""Run the selected Python checks in order with an unconditional docstring gate."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_checks(
    *,
    skip_version_metadata: bool,
    skip_notebook_quality: bool,
    skip_threshold_smoke: bool,
    skip_mypy: bool,
) -> list[tuple[str, list[str]]]:
    """Construct the ordered child commands for the existing preflight profile.

    Parameters
    ----------
    skip_version_metadata
        Omit the project metadata test invocation.
    skip_notebook_quality
        Omit the inherited Golden notebook test invocation.
    skip_threshold_smoke
        Omit the two inherited Task 5/6 smoke test node IDs.
    skip_mypy
        Omit the configured mypy and strict-debt wrapper.

    Returns
    -------
    list[tuple[str, list[str]]]
        Human-readable check names and argument vectors using this process's
        Python executable. The docstring gate is always last, so even selecting
        all four skip flags leaves one check. Paths are relative to REPO_ROOT;
        constructing the list does not validate or execute the child targets.
    """
    checks: list[tuple[str, list[str]]] = []
    if not skip_version_metadata:
        checks.append(
            (
                "Version metadata consistency",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_project_metadata.py",
                    "-q",
                ],
            )
        )
    if not skip_notebook_quality:
        checks.append(
            (
                "Golden notebook quality gate",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_neuro_symbolic_control_demo_notebook.py",
                    "-q",
                ],
            )
        )
    if not skip_threshold_smoke:
        checks.append(
            (
                "Task 5/6 threshold smoke",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_task5_disruption_mitigation_integration.py::test_task5_campaign_passes_thresholds_smoke",
                    "tests/test_task6_heating_neutronics_realism.py::test_task6_campaign_passes_thresholds_smoke",
                    "-q",
                ],
            )
        )
    if not skip_mypy:
        checks.append(("mypy strict", [sys.executable, "tools/run_mypy_strict.py"]))
    checks.append(("docstring coverage", [sys.executable, "tools/run_docstring_gate.py"]))
    return checks


def _run_check(name: str, cmd: list[str]) -> int:
    """Print one command and wait for its real child in the owning repository.

    Parameters
    ----------
    name
        Check label printed before launch.
    cmd
        Argument vector passed directly to subprocess.call without a shell.

    Returns
    -------
    int
        The child's return code, including a negative signal number on POSIX.

    Raises
    ------
    OSError
        The child cannot be started or the repository working directory cannot
        be entered. Child output inherits this process's stdout and stderr.
    """
    rendered = " ".join(shlex.quote(part) for part in cmd)
    print(f"[preflight] {name}: {rendered}")
    return subprocess.call(cmd, cwd=REPO_ROOT)


def main(argv: list[str] | None = None) -> int:
    """Execute selected checks sequentially and propagate the first failure.

    Parameters
    ----------
    argv
        Optional command arguments; None parses sys.argv[1:]. Four independent
        skip flags omit metadata, notebook, threshold or mypy checks. The
        docstring gate has no skip flag.

    Returns
    -------
    int
        Zero when every selected child succeeds, otherwise the first failing
        child's return code. Later checks do not run after a failure. Success
        describes the selected checks and supplies no scientific admission.

    Raises
    ------
    SystemExit
        Argument parsing prints help with status zero or rejects invalid
        arguments with status two.
    OSError
        Starting a child fails; the launch error propagates to the caller.

    Notes
    -----
    Child commands use this interpreter and run from the script's repository,
    independently of the caller's working directory. The inherited Golden
    notebook and Task 5/6 test files are absent from the CONTROL extraction;
    selecting those checks must fail until their complete owning profiles are
    reconciled. Skip flags never make a partial selection a default-profile
    qualification. This wrapper does not invoke tools/preflight.py or all tests.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run fast local/CI Python preflight checks "
            "(version metadata, Golden notebook gate, threshold smokes, mypy strict, "
            "and an unconditional docstring gate)."
        )
    )
    parser.add_argument(
        "--skip-version-metadata",
        action="store_true",
        help="Skip tests/test_project_metadata.py",
    )
    parser.add_argument(
        "--skip-notebook-quality",
        action="store_true",
        help="Skip tests/test_neuro_symbolic_control_demo_notebook.py",
    )
    parser.add_argument(
        "--skip-threshold-smoke",
        action="store_true",
        help=(
            "Skip Task 5/6 threshold smoke tests "
            "(tests/test_task5_disruption_mitigation_integration.py and "
            "tests/test_task6_heating_neutronics_realism.py)."
        ),
    )
    parser.add_argument(
        "--skip-mypy",
        action="store_true",
        help="Skip tools/run_mypy_strict.py",
    )
    args = parser.parse_args(argv)

    checks = _build_checks(
        skip_version_metadata=args.skip_version_metadata,
        skip_notebook_quality=args.skip_notebook_quality,
        skip_threshold_smoke=args.skip_threshold_smoke,
        skip_mypy=args.skip_mypy,
    )
    for name, cmd in checks:
        rc = _run_check(name, cmd)
        if rc != 0:
            print(
                f"[preflight] FAILED at '{name}' with exit code {rc}.",
                file=sys.stderr,
            )
            return rc

    print("[preflight] All selected checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

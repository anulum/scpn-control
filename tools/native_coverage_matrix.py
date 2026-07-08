#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native coverage matrix guard.

"""Validate the Python/Rust coverage-combine contract."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Sequence

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOW: Final = REPO_ROOT / ".github" / "workflows" / "ci.yml"
DEFAULT_PYPROJECT: Final = REPO_ROOT / "pyproject.toml"
DEFAULT_DOCS: Final = (
    REPO_ROOT / "docs" / "validation.md",
    REPO_ROOT / "docs" / "development.md",
)


@dataclass(frozen=True)
class NativeCoverageFinding:
    """A missing native coverage matrix contract."""

    path: str
    check: str
    detail: str


@dataclass(frozen=True)
class NativeCoverageMatrix:
    """Validation result for the native coverage matrix."""

    findings: tuple[NativeCoverageFinding, ...]

    @property
    def passed(self) -> bool:
        """Return whether every required matrix contract is present."""
        return not self.findings

    def to_jsonable(self) -> dict[str, object]:
        """Return a stable JSON representation."""
        return {
            "schema_version": "scpn-control.native-coverage-matrix.v1",
            "passed": self.passed,
            "findings": [asdict(finding) for finding in self.findings],
        }


def _relative(path: Path) -> str:
    """Return ``path`` relative to the repository root when possible."""
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _contains_all(text: str, required: Sequence[str]) -> bool:
    """Return whether every required fragment appears in ``text``."""
    return all(fragment in text for fragment in required)


def _add_if_missing(
    findings: list[NativeCoverageFinding],
    *,
    path: Path,
    check: str,
    detail: str,
    ok: bool,
) -> None:
    """Append a finding when ``ok`` is false."""
    if not ok:
        findings.append(NativeCoverageFinding(path=_relative(path), check=check, detail=detail))


def validate_native_coverage_matrix(
    workflow_path: Path = DEFAULT_WORKFLOW,
    pyproject_path: Path = DEFAULT_PYPROJECT,
    docs_paths: Sequence[Path] = DEFAULT_DOCS,
) -> NativeCoverageMatrix:
    """Validate that native-dependent coverage is collected and combined.

    Parameters
    ----------
    workflow_path
        CI workflow file to inspect.
    pyproject_path
        Project metadata file carrying the coverage threshold.
    docs_paths
        Public documentation files that must describe the workflow.

    Returns
    -------
    NativeCoverageMatrix
        Findings for missing workflow, coverage, or documentation contracts.
    """
    findings: list[NativeCoverageFinding] = []
    workflow = workflow_path.read_text(encoding="utf-8")
    pyproject = pyproject_path.read_text(encoding="utf-8")
    docs = "\n".join(path.read_text(encoding="utf-8") for path in docs_paths)

    _add_if_missing(
        findings,
        path=workflow_path,
        check="python coverage data artifact",
        detail="Python coverage job must upload artifacts/coverage/python/.coverage.python.",
        ok=_contains_all(workflow, ("coverage-data-python", "artifacts/coverage/python/.coverage.python")),
    )
    _add_if_missing(
        findings,
        path=workflow_path,
        check="rust-present coverage data artifact",
        detail="Rust interop job must run parity tests with COVERAGE_FILE=.coverage.rust and upload the data file.",
        ok=_contains_all(
            workflow,
            (
                "COVERAGE_FILE=.coverage.rust",
                "tests/test_rust_python_parity.py tests/test_rust_compat_wrapper.py tests/test_snn_pyo3_bridge.py",
                "coverage-data-rust",
                "artifacts/coverage/rust/.coverage.rust",
            ),
        ),
    )
    _add_if_missing(
        findings,
        path=workflow_path,
        check="combined coverage job",
        detail="CI must combine Python and Rust-present coverage data and gate the merged report.",
        ok=_contains_all(
            workflow,
            (
                "native-coverage-combine:",
                "needs: [python-tests, rust-python-interop]",
                "python -m coverage combine --keep artifacts/coverage/python artifacts/coverage/rust",
                "python -m coverage report --fail-under=100",
                "coverage-report-combined",
            ),
        ),
    )
    _add_if_missing(
        findings,
        path=pyproject_path,
        check="coverage threshold",
        detail="pyproject.toml must keep the coverage gate at fail_under = 100.",
        ok="fail_under = 100" in pyproject,
    )
    _add_if_missing(
        findings,
        path=docs_paths[0],
        check="public docs",
        detail="Validation/development docs must show the coverage-combine workflow and guard entrypoint.",
        ok=_contains_all(
            docs,
            (
                "scpn-native-coverage-matrix",
                "coverage-data-python",
                "coverage-data-rust",
                "coverage combine --keep artifacts/coverage/python artifacts/coverage/rust",
                "scpn-control.native-coverage-matrix.v1",
            ),
        ),
    )
    return NativeCoverageMatrix(findings=tuple(findings))


def main(argv: list[str] | None = None) -> int:
    """Run the native coverage matrix guard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", default=str(DEFAULT_WORKFLOW), help="CI workflow file to inspect")
    parser.add_argument("--pyproject", default=str(DEFAULT_PYPROJECT), help="pyproject.toml path")
    parser.add_argument(
        "--docs",
        nargs="*",
        default=[str(path) for path in DEFAULT_DOCS],
        help="public documentation files that describe the native coverage matrix",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    docs_paths = tuple(Path(path) for path in args.docs)
    matrix = validate_native_coverage_matrix(Path(args.workflow), Path(args.pyproject), docs_paths)
    if args.json:
        print(json.dumps(matrix.to_jsonable(), indent=2, sort_keys=True))
        return 0 if matrix.passed else 1

    if matrix.passed:
        print("Native coverage matrix guard passed.")
        return 0

    print(f"Native coverage matrix guard FAILED: {len(matrix.findings)} finding(s).")
    for finding in matrix.findings:
        print(f"  - {finding.path}: {finding.check}: {finding.detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

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
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.native_coverage_contracts._workflow import _threshold100, _workflow_checks

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOW: Final = REPO_ROOT / "tools" / "ci_workflow_policy.json"
DEFAULT_PYPROJECT: Final = REPO_ROOT / "pyproject.toml"
DEFAULT_DOCS: Final = (
    REPO_ROOT / "docs" / "validation.md",
    REPO_ROOT / "docs" / "development.md",
)


@dataclass(frozen=True)
class NativeCoverageFinding:
    """One declaration or readable-input failure.

    Attributes
    ----------
    path : str
        Repository-relative or selected external input path.
    check : str
        Stable declaration category for callers.
    detail : str
        Authored diagnostic; caught interpreter exception text is not exposed.
    """

    path: str
    check: str
    detail: str


@dataclass(frozen=True)
class NativeCoverageMatrix:
    """Unchecked immutable collection of local declaration findings.

    Empty findings yield a passing declaration verdict. This data object does
    not authenticate a producer, prove hosted execution or measure coverage.
    """

    findings: tuple[NativeCoverageFinding, ...]

    @property
    def passed(self) -> bool:
        """Return whether the collection contains no declaration findings."""
        return not self.findings

    def to_jsonable(self) -> dict[str, object]:
        """Return a fresh v1 JSON-shaped dictionary without IO or self-sealing."""
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
    workflow_path: Path | None = None,
    pyproject_path: Path = DEFAULT_PYPROJECT,
    docs_paths: Sequence[Path] = DEFAULT_DOCS,
) -> NativeCoverageMatrix:
    """Inspect native coverage collection/combination declarations without running CI.

    Parameters
    ----------
    workflow_path
        Optional monolithic YAML workflow or distributed JSON policy. A policy's
        paths resolve from its parent directory's parent (the repository root).
        The live default reads the physical reusable workflows and coordinator.
    pyproject_path
        Project metadata file carrying the coverage threshold.
    docs_paths
        Public documentation files that must describe the workflow.

    Returns
    -------
    NativeCoverageMatrix
        Findings for malformed inputs, executable-step/dependency/artifact gaps,
        a parsed threshold other than numeric 100, or absent public prose.
        No files are written and no shell or hosted workflow is executed.
    """
    findings: list[NativeCoverageFinding] = []
    contract_path = DEFAULT_WORKFLOW if workflow_path is None else workflow_path
    distributed = contract_path.suffix.lower() == ".json"
    verdicts = (False, False, False)
    try:
        workflow = "" if distributed else contract_path.read_text(encoding="utf-8")
        verdicts = _workflow_checks(workflow, policy_path=contract_path if distributed else None)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError):
        findings.append(
            NativeCoverageFinding(
                _relative(contract_path),
                "workflow input",
                "Workflow source or ownership declarations are unreadable or malformed.",
            )
        )
    checks = (
        (
            "python coverage data artifact",
            "Python collection, saved coverage data and pinned upload must belong to one enabled producer job.",
        ),
        (
            "rust-present coverage data artifact",
            "Rust-present collection must execute all required test owners and save/upload its coverage data.",
        ),
        (
            "combined coverage job",
            "Enabled dependency/download/guard/merge/XML/100-percent gate/upload steps must be ordered and wired.",
        ),
    )
    for (check, detail), passed in zip(checks, verdicts, strict=True):
        _add_if_missing(findings, path=contract_path, check=check, detail=detail, ok=passed)
    threshold = False
    try:
        threshold = _threshold100(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        findings.append(
            NativeCoverageFinding(
                _relative(pyproject_path),
                "threshold input",
                "Coverage threshold input is unreadable or malformed TOML.",
            )
        )
    _add_if_missing(
        findings,
        path=pyproject_path,
        check="coverage threshold",
        detail="Parsed tool.coverage.report.fail_under must be numeric 100.",
        ok=threshold,
    )
    docs = ""
    for path in docs_paths:
        try:
            docs += path.read_text(encoding="utf-8") + "\n"
        except (OSError, UnicodeError):
            findings.append(
                NativeCoverageFinding(
                    _relative(path), "documentation input", "Documentation input is unreadable UTF-8."
                )
            )
    _add_if_missing(
        findings,
        path=docs_paths[0] if docs_paths else DEFAULT_DOCS[0],
        check="public docs",
        detail="Public docs must name the direct guard, artifacts, combine command and v1 schema.",
        ok=_contains_all(
            docs,
            (
                "python tools/native_coverage_matrix.py",
                "coverage-data-python",
                "coverage-data-rust",
                "coverage combine --keep artifacts/coverage/python artifacts/coverage/rust",
                "scpn-control.native-coverage-matrix.v1",
            ),
        ),
    )
    return NativeCoverageMatrix(findings=tuple(findings))


def main(argv: list[str] | None = None) -> int:
    """Print v1 JSON or authored diagnostics; return 0 for declarations or 1 for findings.

    argparse retains help exit 0 and malformed-argument exit 2. Selected input
    IO, YAML/TOML and shell-declaration failures become findings; empty docs
    fail the public-documentation check. No input is rewritten.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, help="optional monolithic YAML workflow or distributed JSON policy")
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
    matrix = validate_native_coverage_matrix(args.workflow, Path(args.pyproject), docs_paths)
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

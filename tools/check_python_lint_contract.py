# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Python lint-scope consistency gate.
"""Check literal Python lint declarations in four local repository surfaces.

Make, static-governance CI, local preflight and pre-commit must contain the
required fragments and omit the forbidden broad lint commands. The pre-commit
declaration also binds this checker to its own executable entry and file filter,
including the actual static-governance workflow.

This is a UTF-8 text inspection, not a YAML/Make/Python parser or an execution
check. Comments and inactive strings can satisfy a fragment; a pass does not
prove lint ran, tool versions agree or every definition has native documentation.
The separate docstring gates and actual lint commands enforce those contracts.
No tools, hooks, workflows or generated ledgers are run or rewritten by the API.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
TYPO_LEDGER_EXCLUSION: Final = "        exclude: ^tools/coverage_exception_ledger\\.json$\n"
LINT_HOOK_FILES: Final = (
    r"^(Makefile|pyproject\.toml|\.pre-commit-config\.yaml|"
    r"\.github/workflows/ci(?:-static-governance)?\.yml|"
    r"tools/(preflight|check_python_lint_contract)\.py|tests/test_python_lint_contract\.py)$"
)
LINT_HOOK_SECTION: Final = (
    "      - id: check-python-lint-contract\n"
    "        name: Python lint scope contract\n"
    "        entry: python tools/check_python_lint_contract.py\n"
    "        language: system\n"
    "        pass_filenames: false\n"
    f"        files: {LINT_HOOK_FILES}\n"
)


@dataclass(frozen=True)
class SurfaceContract:
    """Literal text requirements for one repository-relative surface.

    Attributes
    ----------
    path
        File spelling joined to the caller's repository root, without a
        containment or symlink check.
    required
        Fragments that must each occur at least once, including their whitespace.
    forbidden
        Fragments whose presence always yields an error; empty by default.

    Notes
    -----
    Construction stores the supplied declarations without runtime validation.
    Frozen fields prevent reassignment, not validation of file contents.
    """

    path: Path
    required: tuple[str, ...]
    forbidden: tuple[str, ...] = ()


CONTRACTS: Final = (
    SurfaceContract(
        Path("Makefile"),
        (
            "\truff check src/scpn_control/\n",
            "\truff check --extend-ignore D tests/ tools/ validation/\n",
            "\truff format --check src/ tests/ tools/ validation/\n",
            "\tpython tools/check_docstring_debt.py\n",
        ),
        ("\truff check src/ tests/\n",),
    ),
    SurfaceContract(
        Path(".github/workflows/ci-static-governance.yml"),
        (
            "run: ruff check src/scpn_control/",
            "run: ruff check --extend-ignore D tests/ tools/ validation/",
            "run: ruff format --check src/scpn_control/ tests/ tools/ validation/",
            "run: python tools/check_docstring_debt.py",
        ),
        ("run: ruff check src/ tests/",),
    ),
    SurfaceContract(
        Path("tools/preflight.py"),
        (
            '("ruff check", [_PY, "-m", "ruff", "check", "src/scpn_control/"], None)',
            '[_PY, "-m", "ruff", "check", "--extend-ignore", "D", "tests/", "tools/", "validation/"]',
            '[_PY, "-m", "ruff", "format", "--check", "src/scpn_control/", "tests/", "tools/", "validation/"]',
            '("docstring-debt-ratchet", [_PY, "tools/check_docstring_debt.py"], None)',
        ),
        ('"check", "src/", "tests/"',),
    ),
    SurfaceContract(
        Path(".pre-commit-config.yaml"),
        (
            "      - id: ruff\n        args: [--fix, --exit-non-zero-on-fix]\n        files: ^src/\n",
            "        args: [--fix, --exit-non-zero-on-fix, --extend-ignore, D]\n        files: ^(tests/|tools/|validation/)\n",
            "      - id: ruff-format\n        files: ^(src/|tests/|tools/|validation/)\n",
            "      - id: typos\n" + TYPO_LEDGER_EXCLUSION,
            LINT_HOOK_SECTION,
        ),
    ),
)


def lint_contract_errors(repo: Path) -> list[str]:
    """Return ordered declaration errors from the four policy-owned text files.

    Parameters
    ----------
    repo
        Root joined to the declared relative paths. Relative roots use the
        caller's working directory; symlinks are followed. No Git state is read.

    Returns
    -------
    list[str]
        Fresh list ordered by surface, then required and forbidden fragments.
        Missing or non-file surfaces produce one error and are not read.
        Expected read/UTF-8 failures produce fixed errors and inspection proceeds
        to the next surface. Empty means only that all literal fragments match.

    Notes
    -----
    UTF-8 text uses Python's universal newline handling. Matching is case and
    whitespace sensitive after newline normalisation. No parsing, linter run,
    mutation, cache, locking or coherent concurrent-file snapshot is provided.
    Fragment diagnostics quote authored policy constants, not exception text.

    Examples
    --------
    Inspect the actual repository declarations:

    >>> lint_contract_errors(ROOT)
    []
    """
    errors: list[str] = []
    for contract in CONTRACTS:
        surface = repo / contract.path
        if not surface.is_file():
            errors.append(f"missing contract surface: {contract.path.as_posix()}")
            continue
        try:
            text = surface.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            errors.append(f"could not read UTF-8 contract surface: {contract.path.as_posix()}")
            continue
        for fragment in contract.required:
            if fragment not in text:
                errors.append(f"{contract.path.as_posix()}: missing required fragment {fragment!r}")
        for fragment in contract.forbidden:
            if fragment in text:
                errors.append(f"{contract.path.as_posix()}: forbidden broad lint scope {fragment!r}")
    return errors


def main(argv: list[str] | None = None) -> int:
    """Inspect lint declarations through the stdlib command-line entrypoint.

    Parameters
    ----------
    argv
        Arguments without the executable name, or None for process arguments.
        --repo selects a root relative to the caller; default ROOT is determined
        from this script. The selected root is resolved before inspection.

    Returns
    -------
    int
        0 when literal declarations match, 1 for contract/read errors, or 2 when
        the root cannot be resolved. Authored diagnostics go to stdout only.

    Raises
    ------
    SystemExit
        ArgumentParser uses exit 0 for help and 2 for invalid arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        repo = args.repo.resolve()
    except (OSError, RuntimeError, ValueError):
        print("FAIL: could not resolve Python lint contract repository")
        return 2
    errors = lint_contract_errors(repo)
    if errors:
        print("FAIL: Python lint scope drift detected")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("PASS: required Python lint declarations match repository fragments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Python lint-scope consistency gate.
"""Keep repository lint and generated-artifact scopes consistent across gates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
TYPO_LEDGER_EXCLUSION: Final = "        exclude: ^tools/coverage_exception_ledger\\.json$\n"


@dataclass(frozen=True)
class SurfaceContract:
    """Required and forbidden command fragments for one repository surface."""

    path: Path
    required: tuple[str, ...]
    forbidden: tuple[str, ...] = ()


CONTRACTS: Final = (
    SurfaceContract(
        Path("Makefile"),
        ("\truff check src/scpn_control/\n", "\truff format --check src/ tests/\n"),
        ("\truff check src/ tests/\n",),
    ),
    SurfaceContract(
        Path(".github/workflows/ci-static-governance.yml"),
        ("run: ruff check src/scpn_control/", "run: ruff format --check src/scpn_control/ tests/"),
        ("run: ruff check src/ tests/",),
    ),
    SurfaceContract(
        Path("tools/preflight.py"),
        (
            '("ruff check", [_PY, "-m", "ruff", "check", "src/scpn_control/"], None)',
            '("ruff format", [_PY, "-m", "ruff", "format", "--check", "src/scpn_control/", "tests/"], None)',
        ),
        ('"check", "src/", "tests/"',),
    ),
    SurfaceContract(
        Path(".pre-commit-config.yaml"),
        (
            "      - id: ruff\n        args: [--fix, --exit-non-zero-on-fix]\n        files: ^src/\n",
            "      - id: ruff-format\n        files: ^(src/|tests/)\n",
            "      - id: typos\n" + TYPO_LEDGER_EXCLUSION,
        ),
    ),
)


def lint_contract_errors(repo: Path) -> list[str]:
    """Return lint-scope contract errors found below ``repo``."""
    errors: list[str] = []
    for contract in CONTRACTS:
        surface = repo / contract.path
        if not surface.is_file():
            errors.append(f"missing contract surface: {contract.path.as_posix()}")
            continue
        text = surface.read_text(encoding="utf-8")
        for fragment in contract.required:
            if fragment not in text:
                errors.append(f"{contract.path.as_posix()}: missing required fragment {fragment!r}")
        for fragment in contract.forbidden:
            if fragment in text:
                errors.append(f"{contract.path.as_posix()}: forbidden broad lint scope {fragment!r}")
    return errors


def main(argv: list[str] | None = None) -> int:
    """Validate the repository lint contract and return a process status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    errors = lint_contract_errors(args.repo.resolve())
    if errors:
        print("FAIL: Python lint scope drift detected")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("PASS: Python source lint and test format scopes agree across repository gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

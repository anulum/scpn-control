# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Public-API docstring coverage gate with per-module debt accounting
"""Ratchet missing public-API docstrings downward, module by module.

``tools/check_docs_coverage.py`` already guarantees that every module carries a
module-level docstring and appears in ``docs/api.md``. This gate covers the next
layer: every public **class**, **method**, and **function** must carry a
docstring. The codebase migrates to that contract incrementally, so the gate is
a ratchet rather than a hard pass/fail.

Detection is delegated to ruff's pydocstyle rules so the gate reuses a
maintained, well-tested implementation rather than re-deriving public-API
discovery from the AST:

* ``D101`` — undocumented public class
* ``D102`` — undocumented public method
* ``D103`` — undocumented public function
* ``D104`` — undocumented public package (``__init__.py``)
* ``D106`` — undocumented public nested class

``D105`` (magic methods) and ``D107`` (``__init__``) are deliberately excluded:
the NumPy docstring convention documents ``__init__`` in the class docstring and
does not require docstrings on dunder methods, so requiring them would invite
filler.

The committed ledger ``tools/docstring_debt.json`` records the remaining count
per module. On every run the current count must stay equal or fall, and no
module may exceed its recorded count — mirroring ``run_mypy_strict.py``. The
ledger is rewritten only with ``--update-baseline``, which refuses to raise the
recorded total unless ``--allow-baseline-increase`` is also given.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_TARGET = "src/scpn_control/"
LEDGER_PATH = REPO_ROOT / "tools" / "docstring_debt.json"
LEDGER_SCHEMA = "scpn-control.docstring-debt.v1"
COVERAGE_RULES = ("D101", "D102", "D103", "D104", "D106")


def file_to_module(path: str) -> str:
    """Return the dotted import path for a source file.

    Parameters
    ----------
    path
        A source path as emitted by ruff. Absolute paths and ``src``-relative
        paths are both accepted; everything up to and including ``src/`` is
        stripped.

    Returns
    -------
    str
        The import path, for example ``scpn_control.core.elm_model``.
    """

    normalised = path.replace("\\", "/")
    marker = "src/"
    index = normalised.rfind(marker)
    if index != -1:
        normalised = normalised[index + len(marker) :]
    return normalised.removesuffix(".py").replace("/", ".")


def parse_module_counts(diagnostics: list[dict[str, object]]) -> dict[str, int]:
    """Count missing-docstring diagnostics per module.

    Parameters
    ----------
    diagnostics
        The decoded ruff JSON diagnostics array. Each entry must carry a
        ``filename`` string.

    Returns
    -------
    dict[str, int]
        Mapping of dotted module import path to the number of diagnostics
        attributed to it. Modules with no diagnostics are omitted.
    """

    counts: dict[str, int] = {}
    for diagnostic in diagnostics:
        filename = diagnostic.get("filename")
        if not isinstance(filename, str):
            raise ValueError("ruff diagnostic missing a string 'filename' field")
        module = file_to_module(filename)
        counts[module] = counts.get(module, 0) + 1
    return counts


def run_ruff_probe() -> list[dict[str, object]]:
    """Run ruff over the coverage rules and return its JSON diagnostics.

    Returns
    -------
    list[dict[str, object]]
        The decoded diagnostics array. An empty list means full coverage.

    Raises
    ------
    RuntimeError
        If ruff emits output that is not a JSON array, which signals that the
        invocation itself did not run as expected.
    """

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--select",
            ",".join(COVERAGE_RULES),
            "--output-format",
            "json",
            SOURCE_TARGET,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    payload = result.stdout.strip()
    if not payload:
        # ruff prints nothing to stdout only on internal failure; surface stderr.
        raise RuntimeError(f"ruff produced no JSON output:\n{result.stderr}")
    try:
        diagnostics = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ruff output was not valid JSON: {exc}\n{result.stderr}") from exc
    if not isinstance(diagnostics, list):
        raise RuntimeError("ruff JSON output was not an array of diagnostics")
    return diagnostics


@dataclass(frozen=True)
class DocstringDebtLedger:
    """Committed snapshot of remaining missing-docstring debt.

    Attributes
    ----------
    total
        Total count of missing public-API docstrings across the package.
    per_module
        Mapping of dotted module import path to its recorded missing count.
    """

    total: int
    per_module: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-serialisable representation of the ledger."""

        return {
            "schema": LEDGER_SCHEMA,
            "rules": list(COVERAGE_RULES),
            "total": self.total,
            "per_module": dict(sorted(self.per_module.items())),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> DocstringDebtLedger:
        """Build a ledger from its JSON representation.

        Raises
        ------
        ValueError
            If the schema marker is absent or unrecognised, or the payload
            fields have the wrong type.
        """

        if payload.get("schema") != LEDGER_SCHEMA:
            raise ValueError(f"unexpected ledger schema: {payload.get('schema')!r}")
        per_module = payload.get("per_module", {})
        if not isinstance(per_module, dict):
            raise ValueError("ledger per_module must be an object")
        total = payload.get("total")
        if not isinstance(total, int):
            raise ValueError("ledger total must be an integer")
        return cls(
            total=total,
            per_module={str(k): int(v) for k, v in per_module.items()},
        )


@dataclass(frozen=True)
class RatchetResult:
    """Outcome of comparing current docstring debt against the ledger.

    Attributes
    ----------
    regressions
        Modules whose current missing count exceeds the recorded count, mapped
        to ``(recorded, current)`` pairs.
    total_delta
        ``current_total - ledger_total``; negative means debt fell.
    improvements
        Modules whose current count is strictly below the recorded count, mapped
        to ``(recorded, current)`` pairs.
    """

    regressions: dict[str, tuple[int, int]]
    total_delta: int
    improvements: dict[str, tuple[int, int]]

    @property
    def ok(self) -> bool:
        """Return whether the ratchet holds (no regressions, total not risen)."""

        return not self.regressions and self.total_delta <= 0


def evaluate_ratchet(
    current_total: int,
    current_modules: dict[str, int],
    ledger: DocstringDebtLedger,
) -> RatchetResult:
    """Compare current docstring debt against the committed ledger.

    Parameters
    ----------
    current_total
        Total missing-docstring count from this run.
    current_modules
        Per-module missing-docstring counts from this run.
    ledger
        The committed baseline ledger.

    Returns
    -------
    RatchetResult
        The regressions, improvements, and total delta versus the baseline.
    """

    regressions: dict[str, tuple[int, int]] = {}
    improvements: dict[str, tuple[int, int]] = {}
    for module, current in current_modules.items():
        recorded = ledger.per_module.get(module, 0)
        if current > recorded:
            regressions[module] = (recorded, current)
    for module, recorded in ledger.per_module.items():
        current = current_modules.get(module, 0)
        if current < recorded:
            improvements[module] = (recorded, current)
    return RatchetResult(
        regressions=regressions,
        total_delta=current_total - ledger.total,
        improvements=improvements,
    )


def load_ledger(path: Path = LEDGER_PATH) -> DocstringDebtLedger:
    """Load the committed docstring-debt ledger from disk."""

    return DocstringDebtLedger.from_dict(json.loads(path.read_text()))


def write_ledger(ledger: DocstringDebtLedger, path: Path = LEDGER_PATH) -> None:
    """Persist the ledger as formatted JSON with a trailing newline."""

    path.write_text(json.dumps(ledger.to_dict(), indent=2) + "\n")


def _report_debt(total: int, modules: dict[str, int], *, top: int = 10) -> None:
    """Print the current docstring debt and the worst-offending modules."""

    print(f"[docstrings] remaining missing public-API docstrings: {total} across {len(modules)} modules")
    worst = sorted(modules.items(), key=lambda item: (-item[1], item[0]))[:top]
    for module, count in worst:
        print(f"[docstrings]   {count:>4}  {module}")


def main(argv: list[str] | None = None) -> int:
    """Run the docstring-coverage ratchet.

    Parameters
    ----------
    argv
        Optional argument vector; defaults to ``sys.argv`` when ``None``.

    Returns
    -------
    int
        ``0`` on success, non-zero when the ratchet detects a regression or the
        ruff probe could not be parsed.
    """

    parser = argparse.ArgumentParser(description="Public-API docstring coverage gate with per-module debt accounting.")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Rewrite tools/docstring_debt.json from the current ruff probe.",
    )
    parser.add_argument(
        "--allow-baseline-increase",
        action="store_true",
        help="Permit --update-baseline to record a higher total than the existing ledger.",
    )
    args = parser.parse_args(argv)

    print(f"[docstrings] running probe: ruff check --select {','.join(COVERAGE_RULES)} {SOURCE_TARGET}")
    try:
        diagnostics = run_ruff_probe()
    except RuntimeError as exc:
        print(f"[docstrings] FAILED: {exc}", file=sys.stderr)
        return 2
    current_modules = parse_module_counts(diagnostics)
    current_total = sum(current_modules.values())

    if args.update_baseline:
        new_ledger = DocstringDebtLedger(total=current_total, per_module=current_modules)
        if LEDGER_PATH.exists() and not args.allow_baseline_increase:
            existing = load_ledger(LEDGER_PATH)
            if current_total > existing.total:
                print(
                    f"[docstrings] REFUSED: missing-docstring debt {current_total} exceeds recorded "
                    f"{existing.total}; pass --allow-baseline-increase to accept new debt.",
                    file=sys.stderr,
                )
                return 3
        write_ledger(new_ledger, LEDGER_PATH)
        print(f"[docstrings] baseline updated: total {current_total} missing docstrings recorded.")
        return 0

    ledger = load_ledger(LEDGER_PATH)
    result = evaluate_ratchet(current_total, current_modules, ledger)
    _report_debt(current_total, current_modules)

    if result.improvements and result.total_delta < 0:
        print(
            f"[docstrings] debt fell by {-result.total_delta} since the baseline; "
            "run --update-baseline to tighten the ratchet."
        )

    if not result.ok:
        if result.regressions:
            print("[docstrings] FAILED: missing-docstring debt regressed in these modules:", file=sys.stderr)
            for module, (recorded, current) in sorted(result.regressions.items()):
                print(f"[docstrings]   {module}: {recorded} -> {current}", file=sys.stderr)
        if result.total_delta > 0:
            print(
                f"[docstrings] FAILED: total missing-docstring debt rose by {result.total_delta} "
                f"(baseline {ledger.total}, current {current_total}).",
                file=sys.stderr,
            )
        return 1

    print(f"[docstrings] OK: missing-docstring debt within baseline ({current_total} <= {ledger.total}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

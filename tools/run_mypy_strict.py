# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Strict mypy gate with per-module debt accounting
"""Enforce the configured mypy gate and ratchet strict-typing debt downward.

The repository migrates to ``[tool.mypy] strict = true`` module by module. Two
contracts are checked here:

1. **Hard gate** — ``python -m mypy`` with the repository configuration (which
   applies the per-module strict overlays in ``pyproject.toml``) must report no
   errors. Any failure here is a release blocker.
2. **Debt ratchet** — ``python -m mypy --strict src/scpn_control/`` is run as an
   advisory probe over the whole package. Its per-module error counts are
   compared against the committed ledger ``tools/mypy_strict_debt.json``. The
   total may only stay equal or fall, and no individual module may exceed its
   recorded count. This prevents silent regressions in modules that have already
   been tightened while keeping the migration measurable.

The ledger is updated explicitly with ``--update-baseline`` (which refuses to
raise the recorded total unless ``--allow-baseline-increase`` is also given, so
accepting new debt is always a deliberate, reviewable act).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_TARGET = "src/scpn_control/"
LEDGER_PATH = REPO_ROOT / "tools" / "mypy_strict_debt.json"
LEDGER_SCHEMA = "scpn-control.mypy-strict-debt.v1"

_ERROR_LINE = re.compile(r"^(?P<path>[^:\n]+\.py):\d+: error:", re.MULTILINE)
_SUMMARY = re.compile(r"^Found (?P<count>\d+) errors? in \d+ files?", re.MULTILINE)
_SUCCESS = re.compile(r"^Success: no issues found", re.MULTILINE)


def file_to_module(path: str) -> str:
    """Return the dotted import path for a ``src``-relative source file.

    Parameters
    ----------
    path
        A source path as emitted by mypy, for example
        ``src/scpn_control/core/current_drive.py``.

    Returns
    -------
    str
        The import path, for example ``scpn_control.core.current_drive``.
    """

    normalised = path.replace("\\", "/")
    if normalised.startswith("src/"):
        normalised = normalised[len("src/") :]
    return normalised.removesuffix(".py").replace("/", ".")


def parse_module_error_counts(output: str) -> dict[str, int]:
    """Count strict-probe errors per module from mypy stdout.

    Parameters
    ----------
    output
        Combined stdout/stderr text from a ``mypy --strict`` run.

    Returns
    -------
    dict[str, int]
        Mapping of dotted module import path to the number of error lines
        attributed to it. Modules with no errors are omitted.
    """

    counts: dict[str, int] = {}
    for match in _ERROR_LINE.finditer(output):
        module = file_to_module(match.group("path"))
        counts[module] = counts.get(module, 0) + 1
    return counts


def parse_total_errors(output: str) -> int:
    """Return the total error count reported by mypy.

    Parameters
    ----------
    output
        Combined stdout/stderr text from a mypy run.

    Returns
    -------
    int
        The integer from the ``Found N errors`` summary, ``0`` when mypy printed
        a success summary, otherwise the number of parsed per-line errors as a
        fallback.

    Raises
    ------
    ValueError
        If the output contains neither a recognised summary nor any error line,
        which signals that the mypy invocation itself did not run as expected.
    """

    summary = _SUMMARY.search(output)
    if summary is not None:
        return int(summary.group("count"))
    if _SUCCESS.search(output) is not None:
        return 0
    line_errors = len(_ERROR_LINE.findall(output))
    if line_errors == 0:
        raise ValueError("mypy output had no recognisable summary or error lines")
    return line_errors


@dataclass(frozen=True)
class StrictDebtLedger:
    """Committed snapshot of remaining strict-typing debt.

    Attributes
    ----------
    mypy_version
        The mypy version that produced the recorded counts, for provenance.
    total
        Total ``--strict`` error count across the package.
    per_module
        Mapping of dotted module import path to its recorded error count.
    """

    mypy_version: str
    total: int
    per_module: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-serialisable representation of the ledger."""

        return {
            "schema": LEDGER_SCHEMA,
            "mypy_version": self.mypy_version,
            "total": self.total,
            "per_module": dict(sorted(self.per_module.items())),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> StrictDebtLedger:
        """Build a ledger from its JSON representation.

        Raises
        ------
        ValueError
            If the schema marker is absent or unrecognised.
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
            mypy_version=str(payload.get("mypy_version", "")),
            total=total,
            per_module={str(k): int(v) for k, v in per_module.items()},
        )


@dataclass(frozen=True)
class RatchetResult:
    """Outcome of comparing current strict debt against the ledger.

    Attributes
    ----------
    regressions
        Modules whose current error count exceeds the recorded count, mapped to
        ``(recorded, current)`` pairs.
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
    ledger: StrictDebtLedger,
) -> RatchetResult:
    """Compare current strict debt against the committed ledger.

    Parameters
    ----------
    current_total
        Total strict error count from this run.
    current_modules
        Per-module strict error counts from this run.
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


def _mypy_version() -> str:
    """Return the mypy version string, or ``"unknown"`` if it cannot be read."""

    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--version"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "unknown"


def run_configured_mypy() -> tuple[int, str]:
    """Run the repository-configured mypy gate.

    Returns
    -------
    tuple[int, str]
        The process return code and its combined stdout/stderr.
    """

    result = subprocess.run(
        [sys.executable, "-m", "mypy"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout + result.stderr


def run_strict_probe() -> str:
    """Run the advisory whole-package ``mypy --strict`` probe.

    Returns
    -------
    str
        The combined stdout/stderr of the probe. The return code is ignored
        because debt is expected during the migration; only the parsed counts
        matter.
    """

    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", SOURCE_TARGET],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


def load_ledger(path: Path = LEDGER_PATH) -> StrictDebtLedger:
    """Load the committed strict-debt ledger from disk."""

    return StrictDebtLedger.from_dict(json.loads(path.read_text()))


def write_ledger(ledger: StrictDebtLedger, path: Path = LEDGER_PATH) -> None:
    """Persist the ledger as formatted JSON with a trailing newline."""

    path.write_text(json.dumps(ledger.to_dict(), indent=2) + "\n")


def _report_debt(total: int, modules: dict[str, int], *, top: int = 10) -> None:
    """Print the current strict debt and the worst-offending modules."""

    print(f"[mypy-strict] remaining strict debt: {total} errors across {len(modules)} modules")
    worst = sorted(modules.items(), key=lambda item: (-item[1], item[0]))[:top]
    for module, count in worst:
        print(f"[mypy-strict]   {count:>4}  {module}")


def main(argv: list[str] | None = None) -> int:
    """Run the configured gate and the strict-debt ratchet.

    Parameters
    ----------
    argv
        Optional argument vector; defaults to ``sys.argv`` when ``None``.

    Returns
    -------
    int
        ``0`` on success, non-zero when the configured gate fails or the strict
        debt ratchet detects a regression.
    """

    parser = argparse.ArgumentParser(description="Strict mypy gate with per-module debt accounting.")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Rewrite tools/mypy_strict_debt.json from the current strict probe.",
    )
    parser.add_argument(
        "--allow-baseline-increase",
        action="store_true",
        help="Permit --update-baseline to record a higher total than the existing ledger.",
    )
    parser.add_argument(
        "--skip-configured-gate",
        action="store_true",
        help="Skip the hard `python -m mypy` gate and only run the debt ratchet.",
    )
    args = parser.parse_args(argv)

    if not args.skip_configured_gate:
        print("[mypy-strict] running configured gate: python -m mypy")
        rc, output = run_configured_mypy()
        if rc != 0:
            sys.stdout.write(output)
            print("[mypy-strict] FAILED: configured mypy gate reported errors.", file=sys.stderr)
            return rc
        print("[mypy-strict] configured gate clean.")

    print("[mypy-strict] running strict probe: python -m mypy --strict " + SOURCE_TARGET)
    probe = run_strict_probe()
    try:
        current_total = parse_total_errors(probe)
    except ValueError as exc:
        sys.stdout.write(probe)
        print(f"[mypy-strict] FAILED: {exc}", file=sys.stderr)
        return 2
    current_modules = parse_module_error_counts(probe)

    if args.update_baseline:
        new_ledger = StrictDebtLedger(
            mypy_version=_mypy_version(),
            total=current_total,
            per_module=current_modules,
        )
        if LEDGER_PATH.exists() and not args.allow_baseline_increase:
            existing = load_ledger(LEDGER_PATH)
            if current_total > existing.total:
                print(
                    f"[mypy-strict] REFUSED: strict debt {current_total} exceeds recorded "
                    f"{existing.total}; pass --allow-baseline-increase to accept new debt.",
                    file=sys.stderr,
                )
                return 3
        write_ledger(new_ledger, LEDGER_PATH)
        print(f"[mypy-strict] baseline updated: total {current_total} errors recorded.")
        return 0

    ledger = load_ledger(LEDGER_PATH)
    result = evaluate_ratchet(current_total, current_modules, ledger)
    _report_debt(current_total, current_modules)

    if result.improvements and result.total_delta < 0:
        print(
            f"[mypy-strict] debt fell by {-result.total_delta} since the baseline; "
            "run --update-baseline to tighten the ratchet."
        )

    if not result.ok:
        if result.regressions:
            print("[mypy-strict] FAILED: strict debt regressed in these modules:", file=sys.stderr)
            for module, (recorded, current) in sorted(result.regressions.items()):
                print(f"[mypy-strict]   {module}: {recorded} -> {current}", file=sys.stderr)
        if result.total_delta > 0:
            print(
                f"[mypy-strict] FAILED: total strict debt rose by {result.total_delta} "
                f"(baseline {ledger.total}, current {current_total}).",
                file=sys.stderr,
            )
        return 1

    print(f"[mypy-strict] OK: strict debt within baseline ({current_total} <= {ledger.total}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

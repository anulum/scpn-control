# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Strict mypy gate with per-module debt accounting
"""Enforce the configured mypy gate and ratchet strict-typing debt downward.

The repository already enables ``[tool.mypy] strict = true``. Two contracts are
checked here:

1. **Hard gate** — ``python -m mypy`` with the repository configuration (which
   checks source and validation targets with ``pyproject.toml``) must report no
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

        Counts are nonnegative integers, excluding booleans, and their sum must
        equal ``total``. Module labels are nonempty strings. An omitted version
        remains the legacy empty string; a present version must be a string.
        Unknown fields are ignored. Direct construction remains unchecked.

        Raises
        ------
        ValueError
            If the schema, counts, labels, version or count sum is invalid.
        """
        if payload.get("schema") != LEDGER_SCHEMA:
            raise ValueError("unexpected strict-debt ledger schema")
        per_module = payload.get("per_module", {})
        if not isinstance(per_module, dict):
            raise ValueError("ledger per_module must be an object")
        total = payload.get("total")
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise ValueError("ledger total must be a nonnegative integer")
        version = payload.get("mypy_version", "")
        if not isinstance(version, str):
            raise ValueError("ledger mypy_version must be a string")
        counts: dict[str, int] = {}
        for module, count in per_module.items():
            if not isinstance(module, str) or not module.strip():
                raise ValueError("ledger module labels must be nonempty strings")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("ledger module counts must be nonnegative integers")
            counts[module] = count
        if sum(counts.values()) != total:
            raise ValueError("ledger total must equal the sum of module counts")
        return cls(
            mypy_version=version,
            total=total,
            per_module=counts,
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
    """Return stripped native stdout, or ``unknown`` for empty stdout; launch errors propagate.

    This optional provenance observation does not validate the version format or
    native exit status. A ledger version string is not a toolchain certificate.
    """
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
        The combined stdout/stderr of the probe. Codes zero and one represent a
        clean check or reported typing errors; other codes refuse the probe.

    Raises
    ------
    RuntimeError
        If mypy exits outside its clean/error statuses.
    OSError
        If the native Python process cannot be launched.
    """
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", SOURCE_TARGET],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError("strict mypy probe did not complete")
    return result.stdout + result.stderr


def _unique_ledger_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate keys in each persisted JSON object before counts are decoded."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("strict-debt ledger contains duplicate object keys")
        result[key] = value
    return result


def load_ledger(path: Path = LEDGER_PATH) -> StrictDebtLedger:
    """Read UTF-8 JSON and validate its object root, unique keys and ledger counts.

    File/decode errors propagate. Invalid JSON or ledger values raise ValueError;
    direct dictionary construction cannot recover duplicate-key provenance.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_ledger_object)
    except ValueError:
        raise ValueError("strict-debt ledger JSON could not be decoded") from None
    if not isinstance(payload, dict):
        raise ValueError("strict-debt ledger must be an object")
    return StrictDebtLedger.from_dict(payload)


def write_ledger(ledger: StrictDebtLedger, path: Path = LEDGER_PATH) -> None:
    """Write formatted UTF-8 JSON without creating parents or validating direct constructors.

    The write is not atomic. The caller owns path selection and concurrent-file
    coordination; IO/decode/serialisation errors propagate.
    """
    path.write_text(json.dumps(ledger.to_dict(), indent=2) + "\n", encoding="utf-8")


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
        Optional argument vector; defaults to ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    int
        Zero for accepted debt/update, one for debt regression, two for a
        probe/count/ledger failure and three for a refused update increase.
        Configured-check failures retain the native process status. Argument
        parsing may raise SystemExit before any native check.
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
        try:
            rc, output = run_configured_mypy()
        except OSError:
            print("[mypy-strict] FAILED: configured mypy process could not be launched.", file=sys.stderr)
            return 2
        if rc != 0:
            sys.stdout.write(output)
            print("[mypy-strict] FAILED: configured mypy gate reported errors.", file=sys.stderr)
            return rc
        print("[mypy-strict] configured gate clean.")

    print("[mypy-strict] running strict probe: python -m mypy --strict " + SOURCE_TARGET)
    try:
        probe = run_strict_probe()
    except (OSError, RuntimeError):
        print("[mypy-strict] FAILED: strict mypy probe did not complete.", file=sys.stderr)
        return 2
    try:
        current_total = parse_total_errors(probe)
    except ValueError:
        print("[mypy-strict] FAILED: strict mypy output could not be counted.", file=sys.stderr)
        return 2
    current_modules = parse_module_error_counts(probe)
    if current_total != sum(current_modules.values()):
        print("[mypy-strict] FAILED: strict mypy total disagrees with module counts.", file=sys.stderr)
        return 2

    if args.update_baseline:
        if LEDGER_PATH.exists():
            try:
                existing = load_ledger(LEDGER_PATH)
            except (OSError, UnicodeError, ValueError):
                print("[mypy-strict] FAILED: strict-debt ledger could not be read.", file=sys.stderr)
                return 2
            if current_total > existing.total and not args.allow_baseline_increase:
                print(
                    f"[mypy-strict] REFUSED: strict debt {current_total} exceeds recorded "
                    f"{existing.total}; pass --allow-baseline-increase to accept new debt.",
                    file=sys.stderr,
                )
                return 3
        try:
            new_ledger = StrictDebtLedger(mypy_version=_mypy_version(), total=current_total, per_module=current_modules)
            write_ledger(new_ledger, LEDGER_PATH)
        except (OSError, UnicodeError, ValueError):
            print("[mypy-strict] FAILED: strict-debt ledger could not be written.", file=sys.stderr)
            return 2
        print(f"[mypy-strict] baseline updated: total {current_total} errors recorded.")
        return 0

    try:
        ledger = load_ledger(LEDGER_PATH)
    except (OSError, UnicodeError, ValueError):
        print("[mypy-strict] FAILED: strict-debt ledger could not be read.", file=sys.stderr)
        return 2
    result = evaluate_ratchet(current_total, current_modules, ledger)
    _report_debt(current_total, current_modules)

    if result.improvements and result.total_delta < 0:
        print(
            f"[mypy-strict] debt fell by {-result.total_delta} since the baseline; "
            "run --update-baseline to tighten the ratchet."
        )

    if not result.ok:
        # Coherent ledger/current sums make a total increase imply a module increase.
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

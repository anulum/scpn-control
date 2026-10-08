# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — docstring debt ratchet.
"""Hold the test, tool and validation docstring debt at its recorded ceiling.

Division of labour with the two neighbouring gates:

* ``ruff check src/scpn_control/`` enforces the full ``D`` rule set on the
  package source, where there is no debt.
* ``tools/run_docstring_gate.py`` owns the public-API coverage floor inside
  ``src/scpn_control/`` with a per-module ledger.
* This gate owns ``tests/``, ``tools/`` and ``validation/``. Those surfaces are
  admitted to the lint gate with ``--extend-ignore D`` so their non-docstring
  rules are enforced immediately, and the whole ``D`` backlog is held here
  instead of blocking the gate on it.

The count measures native docstring presence and style, not semantic accuracy
or executable examples. A new undocumented owner can be offset by improvements
elsewhere until the ceiling is explicitly lowered; the total is not a per-owner
admission decision. Invalid Python and unreadable source refuse the measurement.
The repository configuration still determines which documentation rules apply.

The CLI resolves ``--repo`` against the working directory and uses that root's
``tools/docstring_debt_ceiling.json``. Without it, the owning repository is used.
Exit zero means a completed measurement within the ceiling, one means a debt
regression or refused increase, and two means an invalid measurement or ledger.
``--update`` writes only after a valid, non-increasing comparison; it does not
modify source files or certify their runtime behaviour.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
LEDGER: Final = Path(__file__).resolve().parent / "docstring_debt_ceiling.json"
SCOPE: Final = ("tests", "tools", "validation")


def measure(repo: Path) -> dict[str, int]:
    """Count current ``D`` findings per rule code across the debt scope.

    The scope is linted with the repository's own ruff configuration, so the
    project ``select`` and ``ignore`` lists both apply. Passing ``--select D``
    instead would bypass ``ignore`` and count rules the project has retired.

    Parameters
    ----------
    repo:
        Repository root containing ``pyproject.toml`` and the debt scope.

    Returns
    -------
    dict[str, int]
        Finding count keyed by ruff rule code, restricted to ``D`` rules.

    Raises
    ------
    RuntimeError
        If the debt scope is missing or holds no Python files, or if ruff
        cannot be executed, exits abnormally, or emits unparsable output. An
        absent scope must fail loudly: reporting zero debt for a directory that
        has been renamed away would silently retire the ratchet.

    Examples
    --------
    Measure the maintained repository without changing its ledger. A passing
    count does not prove semantic contracts or executable documentation.

    >>> counts = measure(ROOT)
    >>> sum(counts.values()) <= read_ceiling(LEDGER)
    True
    """
    repo = repo.resolve()
    missing = [part for part in SCOPE if not (repo / part).is_dir()]
    if missing:
        raise RuntimeError(f"debt scope missing under {repo}: {', '.join(missing)}")
    empty = [part for part in SCOPE if not any((repo / part).rglob("*.py"))]
    if empty:
        raise RuntimeError(f"debt scope holds no Python files: {', '.join(empty)}")

    argv = [sys.executable, "-m", "ruff", "check", "--output-format", "json", "--no-cache"]
    argv += [str(repo / part) for part in SCOPE]
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, cwd=repo, check=False)
    except OSError as exc:
        raise RuntimeError(f"could not run ruff: {exc}") from exc
    if completed.returncode not in (0, 1):
        raise RuntimeError(f"ruff failed with exit {completed.returncode}: {completed.stderr.strip()}")
    if completed.stdout.strip() == "":
        raise RuntimeError(f"ruff produced no output; stderr: {completed.stderr.strip()}")
    try:
        findings = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"could not parse ruff output: {exc}") from exc
    if not isinstance(findings, list):
        raise RuntimeError("ruff output must be an array of diagnostics")
    counts: dict[str, int] = {}
    for finding in findings:
        if not isinstance(finding, dict) or not isinstance(finding.get("code"), str):
            raise RuntimeError("ruff diagnostic must contain a string rule code")
        code = finding["code"]
        if code in ("invalid-syntax", "E902", "E999"):
            raise RuntimeError(f"ruff could not inspect source: {finding.get('filename')}: {code}")
        if re.fullmatch(r"D[0-9]{3}", code):
            counts[code] = counts.get(code, 0) + 1
    return counts


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject ambiguous duplicate keys at every depth of the ledger JSON."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate ledger key: {key}")
        result[key] = value
    return result


def read_ceiling(ledger: Path) -> int:
    """Read a nonnegative integer ceiling bound to the exact debt scope.

    Parameters
    ----------
    ledger
        UTF-8 JSON path. ``total`` must be a nonnegative integer, excluding
        booleans, numeric strings and floating-point values. ``scope`` must
        equal the ordered tests/tools/validation scope. Optional ``by_code``
        counts must be nonnegative integers for D-rule codes and sum to total.

    Returns
    -------
    int
        The validated ceiling; omitted rule counts do not imply zero debt.

    Raises
    ------
    RuntimeError
        The file is missing, unreadable, ambiguous or violates the ledger
        contract. No default ceiling or numeric coercion is applied.
    """
    try:
        payload = json.loads(ledger.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        if not isinstance(payload, dict):
            raise ValueError("ledger must be an object")
        total = payload.get("total")
        if type(total) is not int or total < 0:
            raise ValueError("total must be a nonnegative integer")
        if payload.get("scope") != list(SCOPE):
            raise ValueError("scope must match tests/tools/validation")
        if "by_code" in payload:
            counts = payload["by_code"]
            if not isinstance(counts, dict) or any(
                not re.fullmatch(r"D[0-9]{3}", key) or type(value) is not int or value < 0
                for key, value in counts.items()
            ):
                raise ValueError("by_code must contain nonnegative integer D-rule counts")
            if sum(counts.values()) != total:
                raise ValueError("by_code sum must equal total")
        return total
    except FileNotFoundError as exc:
        raise RuntimeError(f"missing docstring debt ledger: {ledger}") from exc
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(f"malformed docstring debt ledger {ledger}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    """Compare measured debt against the ceiling and report the verdict.

    Parameters
    ----------
    argv:
        Command-line arguments; ``None`` reads :data:`sys.argv`.

    Returns
    -------
    int
        Zero for a completed measurement within the ceiling, one for a debt
        regression or refused increase, two for root/measurement/ledger failure.
        Argparse exits with two on unsupported arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="Lower the ceiling to the measured value. Refuses to raise it.",
    )
    parser.add_argument("--repo", type=Path, default=ROOT, help="Repository root whose scope and ledger are checked.")
    args = parser.parse_args(argv)
    try:
        try:
            args.repo.stat()
        except FileNotFoundError:
            pass
        repo = args.repo.resolve()
        ledger = repo / "tools" / LEDGER.name
        counts = measure(repo)
        ceiling = read_ceiling(ledger)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    total = sum(counts.values())

    if args.update:
        if total > ceiling:
            print(f"FAIL: refusing to raise the ceiling from {ceiling} to {total}")
            return 1
        payload = {
            "total": total,
            "scope": list(SCOPE),
            "by_code": dict(sorted(counts.items())),
            "note": "Measured ceiling. Lower it by documenting code; never raise it.",
        }
        try:
            ledger.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"FAIL: could not write docstring debt ledger {ledger}: {exc}", file=sys.stderr)
            return 2
        print(f"PASS: ceiling lowered from {ceiling} to {total}")
        return 0

    if total > ceiling:
        print(f"FAIL: docstring debt rose from {ceiling} to {total} (+{total - ceiling})")
        for code, count in sorted(counts.items()):
            print(f"  {code}: {count}")
        print("Document the new code, or run with --update only after lowering it.")
        return 1

    if total < ceiling:
        print(f"PASS: docstring debt fell from {ceiling} to {total}; run --update to lock it in")
        return 0

    print(f"PASS: docstring debt holds at {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

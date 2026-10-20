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
or executable examples. Mandatory per-file/rule bounds prevent one repair from
offsetting another owner's regression. Unrecorded files and rules have a zero
bound. Historical debt is admitted only for unchanged fingerprinted source;
a changed file must clear every Ruff-selected documentation finding. Invalid
Python and unreadable source refuse the measurement. The root
configuration and Ruff distribution version must match the recorded profile;
configuration discovery and source enrollment remain separate qualifications.

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

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.docstring_debt_ledger import (
    SCOPE as SCOPE,
)
from tools.docstring_debt_ledger import (
    SOURCE_FINGERPRINT_PROTOCOL,
    measure_owner_sources,
    measurement_profile,
    owner_path,
    read_debt_ceiling,
    read_owner_ledger,
)

ROOT: Final = Path(__file__).resolve().parents[1]
LEDGER: Final = Path(__file__).resolve().parent / "docstring_debt_ceiling.json"


def measure_owners(repo: Path) -> dict[str, dict[str, int]]:
    """Count current Ruff-selected ``D`` findings by repository file and rule.

    The scope is linted with the repository's own ruff configuration, so the
    project ``select`` and ``ignore`` lists both apply. Passing ``--select D``
    instead would bypass ``ignore`` and count rules the project has retired.

    Parameters
    ----------
    repo:
        Repository root containing ``pyproject.toml`` and the debt scope.

    Returns
    -------
    dict[str, dict[str, int]]
        Nonzero finding counts keyed by lexical relative owner and D-rule code.
        Owners without diagnostics are omitted, not counted as undocumented.

    Raises
    ------
    RuntimeError
        If the debt scope is missing or holds no Python files, or if ruff
        cannot be executed, exits abnormally, or emits unparsable output. An
        absent scope must fail loudly: reporting zero debt for a directory that
        has been renamed away would silently retire the ratchet.

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
        completed = subprocess.run(argv, capture_output=True, text=True, cwd=repo, check=False, timeout=60)
    except (OSError, UnicodeError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not run ruff: {exc}") from exc
    if completed.returncode not in (0, 1):
        raise RuntimeError(f"ruff failed with exit {completed.returncode}: {completed.stderr.strip()}")
    if completed.stdout.strip() == "":
        raise RuntimeError(f"ruff produced no output; stderr: {completed.stderr.strip()}")
    try:
        findings = json.loads(completed.stdout)
    except (ValueError, RecursionError):
        raise RuntimeError("could not parse ruff output as JSON") from None
    if not isinstance(findings, list):
        raise RuntimeError("ruff output must be an array of diagnostics")
    counts: dict[str, dict[str, int]] = {}
    for finding in findings:
        if not isinstance(finding, dict) or not isinstance(finding.get("code"), str):
            raise RuntimeError("ruff diagnostic must contain a string rule code")
        code = finding["code"]
        if code in ("invalid-syntax", "E902", "E999"):
            raise RuntimeError(f"ruff could not inspect source: {finding.get('filename')}: {code}")
        if re.fullmatch(r"D[0-9]{3}", code):
            filename = finding.get("filename")
            if not isinstance(filename, str):
                raise RuntimeError("ruff documentation diagnostic requires a string filename")
            try:
                owner = owner_path(Path(filename).relative_to(repo).as_posix())
            except ValueError:
                raise RuntimeError("ruff documentation diagnostic is outside the declared owner scope") from None
            owner_counts = counts.setdefault(owner, {})
            owner_counts[code] = owner_counts.get(code, 0) + 1
    return counts


def _rule_counts(owners: dict[str, dict[str, int]]) -> dict[str, int]:
    """Aggregate already-measured file counts without changing their owner bounds."""
    result: dict[str, int] = {}
    for counts in owners.values():
        for code, count in counts.items():
            result[code] = result.get(code, 0) + count
    return result


def measure(repo: Path) -> dict[str, int]:
    """Retain the public aggregate measurement API over the same native owner probe.

    Parameters
    ----------
    repo : pathlib.Path
        Caller-relative or absolute root containing all three debt directories.

    Returns
    -------
    dict of str to int
        Ruff-selected documentation counts by rule, with zero-count rules omitted.

    Raises
    ------
    RuntimeError
        The scope, native Ruff process or diagnostic protocol cannot be inspected.

    Examples
    --------
    Measurement alone is not owner admission or semantic qualification:

    >>> counts = measure(ROOT)
    >>> sum(counts.values()) <= read_ceiling(LEDGER)
    True
    """
    return _rule_counts(measure_owners(repo))


def read_ceiling(ledger: Path) -> int:
    """Read a nonnegative integer ceiling bound to the exact debt scope.

    Parameters
    ----------
    ledger
        UTF-8 JSON path. ``total`` must be a nonnegative integer, excluding
        booleans, numeric strings and floating-point values. ``scope`` must
        equal the ordered tests/tools/validation scope. Optional ``by_code``
        counts must be nonnegative integers for D-rule codes and sum to total.
        Present owner/profile fields are validated; their absence remains
        readable here but cannot clear mandatory CLI owner enforcement.

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
    return read_debt_ceiling(ledger)


def main(argv: list[str] | None = None) -> int:
    """Compare measured debt against the ceiling and report the verdict.

    Parameters
    ----------
    argv:
        Command-line arguments; ``None`` reads :data:`sys.argv`.

    Returns
    -------
    int
        Zero for completed aggregate/owner comparisons within their bounds, one
        for a debt regression, two for root/measurement/ledger/profile failure.
        Argparse exits with two on unsupported arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="Lower aggregate and owner/rule bounds to measured counts; refuse any regression.",
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
        owners = measure_owners(repo)
        counts = _rule_counts(owners)
        bounds = read_owner_ledger(ledger)
        ceiling = bounds.total
        if bounds.profile != measurement_profile(repo):
            raise RuntimeError("docstring debt profile changed; qualify comparable owner bounds before adoption")
        sources = measure_owner_sources(repo, owners)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    total = sum(counts.values())

    regressions = [
        f"{path}: {code} {bounds.per_file.get(path, {}).get(code, 0)} -> {count}"
        for path, values in sorted(owners.items())
        for code, count in sorted(values.items())
        if count > bounds.per_file.get(path, {}).get(code, 0)
    ]
    if regressions and total <= ceiling:
        print("FAIL: docstring debt rose within an owner/rule; repairs cannot offset regressions")
        print("\n".join(regressions))
        return 1

    changed_owners = [
        path
        for path, fingerprint in sources.items()
        if path in bounds.per_file and fingerprint != bounds.source_fingerprints[path]
    ]
    if changed_owners and total <= ceiling:
        print("FAIL: changed owners retain documentation debt; complete the native file profile")
        print("\n".join(sorted(changed_owners)))
        return 1

    if args.update:
        if total > ceiling:
            print(f"FAIL: refusing to raise the ceiling from {ceiling} to {total}")
            return 1
        payload = {
            "total": total,
            "scope": list(SCOPE),
            "by_code": dict(sorted(counts.items())),
            "per_file": {path: dict(sorted(values.items())) for path, values in sorted(owners.items())},
            "profile": bounds.profile,
            "source_fingerprints": sources,
            "source_fingerprint_protocol": SOURCE_FINGERPRINT_PROTOCOL,
            "origin_revision": bounds.origin_revision,
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

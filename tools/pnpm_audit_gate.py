# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — pnpm lock audit with expiring development exceptions.
"""Fail on every advisory in a pnpm lock unless an expiring exception covers it.

``pnpm audit --json`` reads ``pnpm-lock.yaml`` without installing anything and
lists each advisory with its GitHub id, the vulnerable package and its findings:
the installed version, whether that install is development-only, and the
dependency paths (``.>direct>...>package``) that reach it.

A lock passes only when every advisory it carries is named, for that package, by
an exception that has not expired, and every finding of it is a development-only
install whose every path starts at a ``devDependencies`` entry of the lock
directory's ``package.json``. Any other advisory, an expired exception, a
production install path, or an unreadable or contradictory audit fails, as plain
``pnpm audit`` would. An exception lasts at most seven days and covers
development dependencies only; renewing it needs a new grant.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

MAX_EXCEPTION_DAYS = 7
AUDIT_COMMAND = ("pnpm", "audit", "--json")
_FIELDS = frozenset({"advisory", "package", "scope", "granted", "expires", "reason"})
_GHSA = re.compile(r"GHSA(?:-[a-z0-9]{4}){3}")
_PRODUCTION_GROUPS = ("dependencies", "optionalDependencies", "peerDependencies")

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


class AuditGateError(ValueError):
    """The audit cannot be judged: a malformed input or an invalid exception."""


@dataclass(frozen=True)
class AdvisoryException:
    """One granted, expiring exception for one advisory in one package."""

    advisory: str
    package: str
    granted: date
    expires: date
    reason: str


def parse_exception(item: object) -> AdvisoryException:
    """Validate one exception entry: development scope, one to seven days."""
    if not isinstance(item, dict) or set(item) != _FIELDS:
        raise AuditGateError(f"an exception needs exactly the fields {sorted(_FIELDS)}")
    advisory, package, reason = (str(item[key]).strip() for key in ("advisory", "package", "reason"))
    if not _GHSA.fullmatch(advisory):
        raise AuditGateError(f"{advisory!r} is not a GitHub advisory id")
    if not package or not reason:
        raise AuditGateError(f"{advisory}: an exception names its package and its reason")
    if item["scope"] != "dev":
        raise AuditGateError(f"{advisory}: an exception covers development dependencies only")
    granted = date.fromisoformat(str(item["granted"]))
    expires = date.fromisoformat(str(item["expires"]))
    span = (expires - granted).days
    if not 1 <= span <= MAX_EXCEPTION_DAYS:
        raise AuditGateError(f"{advisory}: an exception lasts 1 to {MAX_EXCEPTION_DAYS} days, not {span}")
    return AdvisoryException(advisory, package, granted, expires, reason)


def load_exceptions(path: Path) -> tuple[AdvisoryException, ...]:
    """Read the exceptions file: ``{"schema": 1, "exceptions": [...]}``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != 1 or not isinstance(data.get("exceptions"), list):
        raise AuditGateError("the exceptions file must be {schema: 1, exceptions: [...]}")
    return tuple(parse_exception(item) for item in data["exceptions"])


def run_audit(lock_dir: Path, runner: Runner = subprocess.run) -> dict[str, Any]:
    """Run pnpm's audit of the committed lock and return its JSON report."""
    completed = runner(
        list(AUDIT_COMMAND),
        cwd=lock_dir,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise AuditGateError(f"pnpm audit printed no JSON report (exit {completed.returncode})") from exc
    if not isinstance(report, dict) or "error" in report:
        raise AuditGateError(f"pnpm audit failed (exit {completed.returncode}): {str(report)[:300]}")
    advisories = report.get("advisories")
    if not isinstance(advisories, dict) or not isinstance(report.get("metadata"), dict):
        raise AuditGateError("pnpm audit did not return an advisories report")
    if (completed.returncode, bool(advisories)) not in {(0, False), (1, True)}:
        raise AuditGateError(f"pnpm audit exit {completed.returncode} disagrees with its report")
    return report


def development_roots(manifest: Mapping[str, Any]) -> frozenset[str]:
    """Return the direct dependencies ``package.json`` lists for development only."""
    production: set[str] = set()
    for group in _PRODUCTION_GROUPS:
        entries = manifest.get(group) or {}
        if not isinstance(entries, dict):
            raise AuditGateError(f"package.json {group} is not a mapping")
        production.update(entries)
    development = manifest.get("devDependencies") or {}
    if not isinstance(development, dict):
        raise AuditGateError("package.json devDependencies is not a mapping")
    return frozenset(set(development) - production)


def _finding_failures(label: str, finding: Mapping[str, Any], dev_roots: frozenset[str]) -> list[str]:
    """Return the failures of one finding: development-only install and development roots."""
    failures: list[str] = []
    version = finding.get("version")
    if finding.get("dev") is not True:
        failures.append(f"{label}@{version}: pnpm does not report a development-only install")
    paths = finding.get("paths") or []
    if not paths:
        failures.append(f"{label}@{version}: the report names no dependency path")
    for path in paths:
        hops = str(path).split(">")
        if len(hops) < 2 or hops[0] != "." or hops[1] not in dev_roots:
            failures.append(f"{label}@{version}: {path} does not start at a development dependency")
    return failures


def judge(
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    exceptions: Sequence[AdvisoryException],
    today: date,
) -> tuple[list[str], set[tuple[str, str]]]:
    """Return the failures and the accepted ``(advisory, package)`` pairs for one lock's report."""
    dev_roots = development_roots(manifest)
    failures = [
        f"{item.package} {item.advisory}: the exception expired on {item.expires}"
        for item in exceptions
        if today >= item.expires
    ]
    active = {(item.advisory, item.package) for item in exceptions if today < item.expires}
    used: set[tuple[str, str]] = set()
    for key, advisory in sorted(report["advisories"].items()):
        if not isinstance(advisory, dict):
            failures.append(f"advisory {key}: unreadable entry")
            continue
        ghsa = str(advisory.get("github_advisory_id", ""))
        package = str(advisory.get("module_name", ""))
        label = f"{package} {ghsa or key}"
        if (ghsa, package) not in active:
            failures.append(f"{label} ({advisory.get('severity')}) has no active exception")
            continue
        findings = advisory.get("findings") or []
        if not findings:
            failures.append(f"{label}: the report names no installed version")
        for finding in findings:
            failures.extend(_finding_failures(label, finding, dev_roots))
        used.add((ghsa, package))
    return failures, used


def _check_lock(
    lock_dir: Path,
    exceptions: Sequence[AdvisoryException],
    today: date,
    runner: Runner,
) -> tuple[list[str], set[tuple[str, str]]]:
    """Judge one lock directory; report unusable inputs as failures."""
    try:
        if not (lock_dir / "pnpm-lock.yaml").is_file():
            raise AuditGateError(f"{lock_dir} has no pnpm-lock.yaml")
        manifest = json.loads((lock_dir / "package.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise AuditGateError(f"{lock_dir}/package.json is not an object")
        return judge(run_audit(lock_dir, runner), manifest, exceptions, today)
    except (OSError, ValueError) as exc:
        return [str(exc)], set()


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Runner = subprocess.run,
    today: date | None = None,
) -> int:
    """Audit each named lock directory.

    Exit 1 on any failure and 2 on an unusable exceptions file. Each accepted
    advisory is printed with its package and the earliest expiry among that
    pair's active exceptions, the day from which the gate fails again.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--exceptions", type=Path, required=True)
    parser.add_argument("lock_dirs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    day = today or datetime.now(UTC).date()
    try:
        exceptions = load_exceptions(args.exceptions)
    except (OSError, ValueError) as exc:
        print(f"pnpm audit gate: {exc}", file=sys.stderr)
        return 2
    expiry: dict[tuple[str, str], date] = {}
    for item in exceptions:
        key = (item.advisory, item.package)
        if day < item.expires:
            expiry[key] = min(item.expires, expiry.get(key, item.expires))
    status = 0
    for lock_dir in args.lock_dirs:
        failures, used = _check_lock(lock_dir, exceptions, day, runner)
        for advisory, package in sorted(used):
            print(
                f"{lock_dir}: {package} {advisory} accepted in development dependencies only;"
                f" the exception expires on {expiry[(advisory, package)]}"
            )
        for failure in failures:
            print(f"{lock_dir}: {failure}", file=sys.stderr)
        status = 1 if failures else status
        if not failures and not used:
            print(f"{lock_dir}: found 0 vulnerabilities")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

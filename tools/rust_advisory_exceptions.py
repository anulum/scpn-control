# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Rust advisory exceptions with expiry and build-only scope.
"""Bind the RustSec advisories ignored by cargo-audit and cargo-deny to expiring grants.

``cargo audit`` reads ``[advisories] ignore`` from ``.cargo/audit.toml`` and
``cargo deny`` reads it from ``deny.toml``. Neither file can say when an ignore
ends or why it is acceptable. This check makes both lists follow one exceptions
file. Each exception names one RustSec advisory and one crate, has scope
``build``, a reason, and lasts one to seven days.

The check fails when an exception has expired, when either ignore list differs
from the active exceptions, when a ``deny.toml`` entry has no reason, or when an
excepted crate can be reached from a workspace member without passing a build
dependency edge (``cargo metadata``). A crate reached only through build scripts
runs on the build host, never in a shipped library.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

MAX_EXCEPTION_DAYS = 7
METADATA_COMMAND = ("cargo", "metadata", "--format-version", "1", "--locked")
_FIELDS = frozenset({"advisory", "package", "scope", "granted", "expires", "reason"})
_RUSTSEC = re.compile(r"RUSTSEC-\d{4}-\d{4}")

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


class ExceptionsError(ValueError):
    """The exceptions or the audit configuration cannot be judged."""


@dataclass(frozen=True)
class AdvisoryException:
    """One granted, expiring exception for one RustSec advisory in one crate."""

    advisory: str
    package: str
    granted: date
    expires: date
    reason: str


def parse_exception(item: object) -> AdvisoryException:
    """Validate one exception entry: build scope, one to seven days."""
    if not isinstance(item, dict) or set(item) != _FIELDS:
        raise ExceptionsError(f"an exception needs exactly the fields {sorted(_FIELDS)}")
    advisory, package, reason = (str(item[key]).strip() for key in ("advisory", "package", "reason"))
    if not _RUSTSEC.fullmatch(advisory):
        raise ExceptionsError(f"{advisory!r} is not a RustSec advisory id")
    if not package or not reason:
        raise ExceptionsError(f"{advisory}: an exception names its crate and its reason")
    if item["scope"] != "build":
        raise ExceptionsError(f"{advisory}: an exception covers build dependencies only")
    granted = date.fromisoformat(str(item["granted"]))
    expires = date.fromisoformat(str(item["expires"]))
    span = (expires - granted).days
    if not 1 <= span <= MAX_EXCEPTION_DAYS:
        raise ExceptionsError(f"{advisory}: an exception lasts 1 to {MAX_EXCEPTION_DAYS} days, not {span}")
    return AdvisoryException(advisory, package, granted, expires, reason)


def load_exceptions(path: Path) -> tuple[AdvisoryException, ...]:
    """Read the exceptions file: ``{"schema": 1, "exceptions": [...]}``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != 1 or not isinstance(data.get("exceptions"), list):
        raise ExceptionsError("the exceptions file must be {schema: 1, exceptions: [...]}")
    return tuple(parse_exception(item) for item in data["exceptions"])


def deny_ignores(path: Path) -> list[str]:
    """Return the advisory ids that ``deny.toml`` ignores; every entry needs a reason."""
    advisories = tomllib.loads(path.read_text(encoding="utf-8")).get("advisories", {})
    ids: list[str] = []
    for entry in advisories.get("ignore", []):
        if not isinstance(entry, dict) or not str(entry.get("id", "")).strip():
            raise ExceptionsError(f"{path}: every ignore entry is a table with an id")
        if not str(entry.get("reason", "")).strip():
            raise ExceptionsError(f"{path}: {entry['id']} is ignored without a reason")
        ids.append(str(entry["id"]))
    return ids


def audit_ignores(path: Path) -> list[str]:
    """Return the advisory ids that a cargo-audit ``audit.toml`` ignores."""
    advisories = tomllib.loads(path.read_text(encoding="utf-8")).get("advisories", {})
    ignore = advisories.get("ignore", [])
    if not isinstance(ignore, list) or not all(isinstance(item, str) for item in ignore):
        raise ExceptionsError(f"{path}: [advisories] ignore must be a list of ids")
    return list(ignore)


def cargo_metadata(workspace: Path, runner: Runner = subprocess.run) -> dict[str, Any]:
    """Run ``cargo metadata`` for one workspace and return its resolved graph."""
    completed = runner(list(METADATA_COMMAND), cwd=workspace, capture_output=True, text=True, check=False, timeout=600)
    if completed.returncode != 0:
        raise ExceptionsError(f"{workspace}: cargo metadata failed: {completed.stderr.strip()[:300]}")
    try:
        metadata = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ExceptionsError(f"{workspace}: cargo metadata printed no JSON") from exc
    if not isinstance(metadata, dict) or not isinstance(metadata.get("resolve"), dict):
        raise ExceptionsError(f"{workspace}: cargo metadata has no resolved graph")
    return metadata


def _crate_name(package_id: str, names: Mapping[str, str]) -> str:
    """Return the crate name of a package id."""
    return names.get(package_id, package_id)


def non_build_reach(metadata: Mapping[str, Any]) -> set[str]:
    """Return the crate names reachable from workspace members without a build edge."""
    names = {str(item["id"]): str(item["name"]) for item in metadata.get("packages", [])}
    nodes = {str(node["id"]): node for node in metadata["resolve"].get("nodes", [])}
    stack = [str(member) for member in metadata.get("workspace_members", [])]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        for dep in nodes.get(current, {}).get("deps", []):
            kinds = [kind.get("kind") for kind in dep.get("dep_kinds", [])] or [None]
            if any(kind != "build" for kind in kinds):
                stack.append(str(dep["pkg"]))
    return {_crate_name(package_id, names) for package_id in seen}


def check(
    exceptions: Sequence[AdvisoryException],
    deny_ids: Iterable[str],
    audit_ids: Iterable[str],
    metadata: Sequence[Mapping[str, Any]],
    today: date,
) -> list[str]:
    """Return every failure of the exceptions against the configuration and graphs."""
    failures = [
        f"{item.advisory}: the exception expired on {item.expires}" for item in exceptions if today >= item.expires
    ]
    active = sorted({item.advisory for item in exceptions if today < item.expires})
    for label, ids in (("deny.toml", list(deny_ids)), (".cargo/audit.toml", list(audit_ids))):
        if sorted(ids) != active:
            failures.append(f"{label} ignores {sorted(ids)}, the active exceptions are {active}")
    for graph in metadata:
        reach = non_build_reach(graph)
        for item in exceptions:
            if item.package in reach:
                failures.append(f"{item.advisory}: {item.package} is reachable outside build dependencies")
    return failures


def main(argv: Sequence[str] | None = None, *, runner: Runner = subprocess.run, today: date | None = None) -> int:
    """Check the exceptions; exit 1 on any failure and 2 on unusable inputs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--exceptions", type=Path, required=True)
    parser.add_argument("--deny", type=Path, required=True)
    parser.add_argument("--audit-config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    day = today or datetime.now(UTC).date()
    try:
        exceptions = load_exceptions(args.exceptions)
        deny_ids = deny_ignores(args.deny)
        audit_ids = audit_ignores(args.audit_config)
        graphs = [cargo_metadata(workspace, runner) for workspace in args.workspace]
    except (OSError, ValueError) as exc:
        print(f"rust advisory exceptions: {exc}", file=sys.stderr)
        return 2
    failures = check(exceptions, deny_ids, audit_ids, graphs, day)
    for failure in failures:
        print(f"rust advisory exceptions: {failure}", file=sys.stderr)
    if failures:
        return 1
    for item in exceptions:
        print(f"{item.advisory} ({item.package}) accepted for build dependencies only; it expires on {item.expires}")
    if not exceptions:
        print("no Rust advisory exceptions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

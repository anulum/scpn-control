# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Semantic source-header policy gate.

"""Validate tracked text headers and reviewed format exemptions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict

ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_POLICY: Final = ROOT / "tools/source_header_policy.toml"
EXPECTED: Final = (
    "SPDX-License-Identifier: AGPL-3.0-or-later",
    "Commercial license available",
    "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
    "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
    "ORCID: 0009-0009-3560-0851",
    "Contact: www.anulum.li | protoscience@anulum.li",
)
SLASH_SUFFIXES: Final = frozenset({".cpp", ".h", ".js", ".mjs", ".rs", ".ts", ".tsx"})


@dataclass(frozen=True)
class Finding:
    """One tracked file rejected by the header policy."""

    path: str
    category: str
    detail: str


@dataclass(frozen=True)
class Exemption:
    """One reviewed family of files that cannot carry the source header."""

    category: str
    reason: str
    suffixes: frozenset[str]
    names: frozenset[str]


@dataclass(frozen=True)
class Policy:
    """Parsed source-header scope and exemption contract."""

    schema: str
    enforced_suffixes: frozenset[str]
    enforced_names: frozenset[str]
    exemptions: tuple[Exemption, ...]


class FindingPayload(TypedDict):
    """Stable JSON representation of one source-header finding."""

    path: str
    category: str
    detail: str


class AuditResult(TypedDict):
    """Stable machine-readable result returned by :func:`audit`."""

    schema: str
    policy_schema: str
    source_head: str
    passed: bool
    classifications: dict[str, int]
    exemptions: dict[str, int]
    findings: list[FindingPayload]


def load_policy(path: Path) -> Policy:
    """Load and validate the policy TOML."""
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    schema = raw.get("schema")
    if schema != "scpn-control.source-header-policy.v1":
        raise ValueError(f"unsupported source-header policy schema: {schema!r}")
    enforced = raw.get("enforced")
    if not isinstance(enforced, dict):
        raise ValueError("policy requires an [enforced] table")
    exemptions: list[Exemption] = []
    for entry in raw.get("exemptions", []):
        if not isinstance(entry, dict):
            raise ValueError("every exemption must be a TOML table")
        category = entry.get("category")
        reason = entry.get("reason")
        if not isinstance(category, str) or not category.strip():
            raise ValueError("every exemption requires a category")
        if not isinstance(reason, str) or len(reason.strip()) < 20:
            raise ValueError(f"exemption {category!r} requires a specific reason")
        exemptions.append(
            Exemption(
                category=category,
                reason=reason,
                suffixes=frozenset(str(value).casefold() for value in entry.get("suffixes", [])),
                names=frozenset(str(value) for value in entry.get("names", [])),
            )
        )
    policy = Policy(
        schema=schema,
        enforced_suffixes=frozenset(str(value).casefold() for value in enforced.get("suffixes", [])),
        enforced_names=frozenset(str(value) for value in enforced.get("names", [])),
        exemptions=tuple(exemptions),
    )
    _validate_disjoint(policy)
    return policy


def _validate_disjoint(policy: Policy) -> None:
    seen_suffixes = set(policy.enforced_suffixes)
    seen_names = set(policy.enforced_names)
    for exemption in policy.exemptions:
        overlap_suffixes = seen_suffixes.intersection(exemption.suffixes)
        overlap_names = seen_names.intersection(exemption.names)
        if overlap_suffixes or overlap_names:
            raise ValueError(
                f"overlapping source-header policy entries in {exemption.category}: "
                f"suffixes={sorted(overlap_suffixes)}, names={sorted(overlap_names)}"
            )
        seen_suffixes.update(exemption.suffixes)
        seen_names.update(exemption.names)


def tracked_paths(root: Path) -> list[Path]:
    """Return every tracked path without consulting ignored private state."""
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return sorted(Path(raw.decode("utf-8")) for raw in completed.stdout.split(b"\0") if raw)


def classify(path: Path, policy: Policy) -> tuple[str, str]:
    """Classify a tracked path as enforced, exempt, or unclassified."""
    suffix = path.suffix.casefold()
    if suffix in policy.enforced_suffixes or path.name in policy.enforced_names:
        return "enforced", ""
    for exemption in policy.exemptions:
        if suffix in exemption.suffixes or path.name in exemption.names:
            return "exempt", exemption.category
    return "unclassified", ""


def expected_header(path: Path, purpose: str) -> list[str]:
    """Render the exact format-native header for one tracked path."""
    content = [*EXPECTED, f"SCPN Control — {purpose}"]
    suffix = path.suffix.casefold()
    if suffix == ".lean":
        return ["/-", *content, "-/"]
    if suffix == ".html":
        return [f"<!-- {line} -->" for line in content]
    marker = "//" if suffix in SLASH_SUFFIXES else "#"
    return [f"{marker} {line}" for line in content]


def header_finding(root: Path, path: Path) -> Finding | None:
    """Return the semantic header defect for one enforced file, if any."""
    try:
        lines = (root / path).read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return Finding(str(path), "non_utf8_enforced", "enforced file is not UTF-8 text")
    offset = 1 if lines and lines[0].startswith("#!") else 0
    suffix = path.suffix.casefold()
    if suffix == ".lean":
        actual = lines[offset : offset + 9]
        expected = expected_header(path, "<purpose>")
        prefix = expected[:-2]
        valid = (
            len(actual) == 9
            and actual[:7] == prefix
            and actual[7].startswith("SCPN Control — ")
            and bool(actual[7].removeprefix("SCPN Control — ").strip())
            and actual[8] == "-/"
        )
    else:
        actual = lines[offset : offset + 7]
        expected = expected_header(path, "<purpose>")
        purpose_prefix = "<!-- SCPN Control — " if suffix == ".html" else expected[6].removesuffix("<purpose>")
        purpose_suffix = " -->" if suffix == ".html" else ""
        valid = (
            len(actual) == 7
            and actual[:6] == expected[:6]
            and actual[6].startswith(purpose_prefix)
            and actual[6].endswith(purpose_suffix)
            and bool(actual[6].removeprefix(purpose_prefix).removesuffix(purpose_suffix).strip())
        )
    if valid:
        return None
    return Finding(str(path), "header_mismatch", "expected exact seven-line semantics")


def audit(root: Path, policy_path: Path) -> AuditResult:
    """Audit all tracked paths against the source-header policy."""
    root = root.resolve()
    policy = load_policy(policy_path)
    findings: list[Finding] = []
    classifications: Counter[str] = Counter()
    exemptions: Counter[str] = Counter()
    for path in tracked_paths(root):
        disposition, category = classify(path, policy)
        classifications[disposition] += 1
        if disposition == "unclassified":
            findings.append(Finding(str(path), disposition, "no policy entry"))
        elif disposition == "exempt":
            exemptions[category] += 1
        else:
            finding = header_finding(root, path)
            if finding is not None:
                findings.append(finding)
    return {
        "schema": "scpn-control.source-header-audit.v1",
        "policy_schema": policy.schema,
        "source_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "passed": not findings,
        "classifications": dict(sorted(classifications.items())),
        "exemptions": dict(sorted(exemptions.items())),
        "findings": [
            {"path": finding.path, "category": finding.category, "detail": finding.detail} for finding in findings
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """Run the source-header audit and return a shell-compatible status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    try:
        result = audit(args.root, args.policy)
    except (OSError, subprocess.CalledProcessError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"source-header policy error: {exc}", file=sys.stderr)
        return 2
    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["passed"]:
        print("Source-header policy passed")
    else:
        for finding in result["findings"]:
            print(f"{finding['path']}: {finding['category']}: {finding['detail']}", file=sys.stderr)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Deterministic coverage exception and variant ledger.

"""Generate and validate ownership for every coverage exclusion and test skip."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.ci_workflow_inventory import read_ci_workflow_source

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "tools/coverage_exception_policy.toml"
OUTPUT_PATH = ROOT / "tools/coverage_exception_ledger.json"


@dataclass(frozen=True)
class ExceptionEntry:
    """One owned coverage exception or conditional test outcome."""

    id: str
    kind: str
    path: str
    line: int
    owner: str
    condition: str
    reason: str
    classification: str
    external_dependency: str
    execution_lane: str
    status: str
    last_review: str
    removal_condition: str


def _call_name(node: ast.expr) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _source_text(source: str, node: ast.AST | None) -> str:
    if node is None:
        return ""
    return (ast.get_source_segment(source, node) or "").strip()


def _literal_or_source(source: str, node: ast.AST | None) -> str:
    if node is None:
        return "unspecified by call site"
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        value = None
    if isinstance(value, str) and value.strip():
        return value.strip()
    expression = _source_text(source, node)
    return f"dynamic expression: {expression}" if expression else "unspecified by call site"


def _owner(path: Path) -> str:
    relative = path.relative_to(ROOT)
    if relative.parts[0] == "src" and len(relative.parts) >= 4:
        return f"python/{relative.parts[2]}"
    if relative.parts[0] in {"tests", "validation"}:
        return f"{relative.parts[0]}/{path.stem.removeprefix('test_')}"
    return "repository/coverage-policy"


def _classify(text: str, policy: dict[str, Any], *, allow_default: bool) -> dict[str, str] | None:
    rules = policy["rules"] if allow_default else policy["rules"][:-1]
    for rule in rules:
        if re.search(rule["pattern"], text, re.IGNORECASE):
            return {key: str(value) for key, value in rule.items() if key != "pattern"}
    if allow_default:
        raise ValueError(f"coverage exception was not classified: {text}")
    return None


def _entry(
    *,
    kind: str,
    path: Path,
    line: int,
    condition: str,
    reason: str,
    policy: dict[str, Any],
) -> ExceptionEntry:
    relative = path.relative_to(ROOT).as_posix()
    semantic_text = f"{condition} {reason}"
    rule = _classify(semantic_text, policy, allow_default=False)
    if rule is None:
        rule = _classify(f"{relative} {semantic_text}", policy, allow_default=True)
    assert rule is not None
    removal_condition = rule["removal_condition"]
    if kind == "pytest-xfail":
        removal_condition = (
            "Remove immediately when the diagnosed limitation in this entry's reason is fixed and the strict xfail "
            "becomes an unexpected pass."
        )
    stable_id = hashlib.sha256(f"{kind}\0{relative}\0{line}\0{condition}\0{reason}".encode()).hexdigest()[:16]
    return ExceptionEntry(
        id=f"covexc-{stable_id}",
        kind=kind,
        path=relative,
        line=line,
        owner=_owner(path),
        condition=condition,
        reason=reason,
        classification=rule["id"],
        external_dependency=rule["external_dependency"],
        execution_lane=rule["execution_lane"],
        status=rule["status"],
        last_review=str(policy["last_review"]),
        removal_condition=removal_condition,
    )


def _pragma_entries(policy: dict[str, Any]) -> list[ExceptionEntry]:
    entries: list[ExceptionEntry] = []
    pattern = re.compile(r"pragma:\s*no cover(?P<tail>.*)$")
    for path in sorted((ROOT / "src/scpn_control").rglob("*.py")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = pattern.search(line)
            if match is None:
                continue
            reason = match.group("tail").strip().lstrip("-:;.,#) ]–—").strip()
            if not reason:
                raise ValueError(f"unreasoned coverage pragma: {path.relative_to(ROOT)}:{line_number}")
            entries.append(
                _entry(
                    kind="pragma-no-cover",
                    path=path,
                    line=line_number,
                    condition=line.split("#", 1)[0].strip(),
                    reason=reason,
                    policy=policy,
                )
            )
    return entries


def _pytest_entries(policy: dict[str, Any]) -> list[ExceptionEntry]:
    entries: list[ExceptionEntry] = []
    call_kinds = {
        "pytest.mark.skipif": "pytest-skipif",
        "pytest.skip": "pytest-runtime-skip",
        "pytest.mark.xfail": "pytest-xfail",
    }
    for root_name in ("tests", "validation"):
        for path in sorted((ROOT / root_name).rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _call_name(node.func)
                kind = call_kinds.get(name)
                if kind is None:
                    continue
                reason_node = next((keyword.value for keyword in node.keywords if keyword.arg == "reason"), None)
                if reason_node is None:
                    reason_index = 1 if kind == "pytest-skipif" else 0
                    reason_node = node.args[reason_index] if len(node.args) > reason_index else None
                condition_node = node.args[0] if kind in {"pytest-skipif", "pytest-xfail"} and node.args else None
                entries.append(
                    _entry(
                        kind=kind,
                        path=path,
                        line=node.lineno,
                        condition=_source_text(source, condition_node) or "runtime call",
                        reason=_literal_or_source(source, reason_node),
                        policy=policy,
                    )
                )
    return entries


def _coverage_pattern_entries(policy: dict[str, Any]) -> list[ExceptionEntry]:
    path = ROOT / "pyproject.toml"
    source = path.read_text(encoding="utf-8")
    project = tomllib.loads(source)
    patterns = project["tool"]["coverage"]["report"]["exclude_lines"]
    lines = source.splitlines()
    entries: list[ExceptionEntry] = []
    for pattern in patterns:
        line = next(index for index, text in enumerate(lines, 1) if json.dumps(pattern)[1:-1] in text)
        entries.append(
            _entry(
                kind="coverage-exclude-pattern",
                path=path,
                line=line,
                condition=pattern,
                reason="configured coverage.py exclusion pattern",
                policy=policy,
            )
        )
    return entries


def build_ledger() -> dict[str, Any]:
    """Return the current deterministic exception ledger."""
    policy = tomllib.loads(POLICY_PATH.read_text(encoding="utf-8"))
    workflow = read_ci_workflow_source()
    for rule in policy["rules"]:
        if rule["workflow_evidence"] and rule["workflow_evidence"] not in workflow:
            raise ValueError(f"workflow evidence missing for policy rule {rule['id']}: {rule['workflow_evidence']}")
    entries = _pragma_entries(policy) + _coverage_pattern_entries(policy) + _pytest_entries(policy)
    entries.sort(key=lambda item: (item.kind, item.path, item.line, item.id))
    rust_test_owners = {
        item.path for item in entries if item.classification == "rust" and item.kind.startswith("pytest-")
    }
    missing_rust_owners = sorted(path for path in rust_test_owners if path not in workflow)
    if missing_rust_owners:
        raise ValueError(f"Rust-present CI lane omits conditional test owners: {missing_rust_owners}")
    counts: dict[str, int] = {}
    for item in entries:
        counts[item.kind] = counts.get(item.kind, 0) + 1
    serialised = [asdict(item) for item in entries]
    digest = hashlib.sha256(json.dumps(serialised, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema": "scpn-control.coverage-exception-ledger.v1",
        "generated_from": [
            "src/scpn_control/**/*.py",
            "tests/**/*.py",
            "validation/**/*.py",
            "pyproject.toml",
            "tools/ci_workflow_policy.json",
            ".github/workflows/ci-*.yml",
            "tools/coverage_exception_policy.toml",
        ],
        "last_review": policy["last_review"],
        "entry_count": len(entries),
        "entry_sha256": digest,
        "counts": dict(sorted(counts.items())),
        "entries": serialised,
    }


def main(argv: list[str] | None = None) -> int:
    """Generate the ledger or fail when the committed ledger is stale."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args(argv)
    ledger = build_ledger()
    if args.print_summary:
        print(json.dumps({key: ledger[key] for key in ("entry_count", "entry_sha256", "counts")}, indent=2))
        return 0
    policy = tomllib.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if ledger["entry_count"] != policy["expected_total"] or ledger["entry_sha256"] != policy["expected_sha256"]:
        print("coverage exception inventory changed without review")
        print(json.dumps({key: ledger[key] for key in ("entry_count", "entry_sha256", "counts")}, indent=2))
        return 1
    rendered = json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUTPUT_PATH.is_file() or OUTPUT_PATH.read_text(encoding="utf-8") != rendered:
            print(f"stale coverage exception ledger: {OUTPUT_PATH.relative_to(ROOT)}")
            return 1
        print(f"coverage exception ledger current: {ledger['entry_count']} owned entries")
        return 0
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUTPUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

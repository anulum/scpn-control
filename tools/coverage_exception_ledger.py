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
from datetime import date
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
    """Immutable ownership declaration for one lexical or AST exception.

    Parameters
    ----------
    id, kind, path : str
        Stable digest label, exception family and repository-relative POSIX path.
    line : int
        One-based physical source line; movement changes the stable identifier.
    owner, condition, reason : str
        Derived owner label, source expression and literal or dynamic rationale.
    classification, external_dependency, execution_lane, status : str
        First matching policy rule and its declared dependency/lane/status.
    last_review, removal_condition : str
        ISO calendar review label and stated retirement criterion.

    Attributes
    ----------
    id, kind, path, owner, condition, reason : str
        Stored identity and source descriptions.
    line : int
        Stored source position, with no clock or physical unit.
    classification, external_dependency, execution_lane, status : str
        Declared policy descriptions; no runtime or coverage attestation.
    last_review, removal_condition : str
        Stored review label and retirement criterion.

    Notes
    -----
    Native dataclass fields are frozen. Direct construction preserves values
    without validation; unexpected or missing constructor arguments raise
    TypeError. Collection uses validated policy metadata, not this constructor.
    """

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


def _review_date(value: object, label: str) -> None:
    """Require an exact ISO calendar date in an ownership declaration.

    Parameters
    ----------
    value : object
        Parsed TOML value; native TOML date objects are not string labels.
    label : str
        Field location used in the authored error.

    Raises
    ------
    ValueError
        The value is not a YYYY-MM-DD string or is not a real calendar date.
    """
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise ValueError(f"{label} must be a YYYY-MM-DD string")
    date.fromisoformat(value)


def _load_policy() -> dict[str, Any]:
    """Read and validate the native coverage-ownership policy.

    Returns
    -------
    dict[str, Any]
        Schema-v1 TOML object, preserving valid field values and rule order.
        Unknown keys are retained. The last rule is the fallback candidate.

    Raises
    ------
    ValueError
        Schema, inventory seal, review date, rule array, required string field,
        duplicate identifier or status is invalid.
    re.error
        A rule pattern is not a compilable regular expression.
    tomllib.TOMLDecodeError
        Policy text is not valid TOML.
    OSError
        The policy cannot be read.
    UnicodeDecodeError
        Policy bytes are not UTF-8.

    Notes
    -----
    Counts must be nonnegative integers, excluding booleans. Digests require
    64 lowercase hexadecimal characters. Nonempty string metadata and valid
    calendar dates prevent coercion from fabricating ownership labels. Empty
    workflow evidence explicitly declares no textual CI binding. Pattern and
    workflow declarations do not prove rationale quality or executed CI.
    """
    policy = tomllib.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if policy.get("schema") != "scpn-control.coverage-exception-policy.v1":
        raise ValueError("unsupported coverage exception policy schema")
    count = policy.get("expected_total")
    if type(count) is not int or count < 0:
        raise ValueError("expected_total must be a nonnegative integer")
    digest = policy.get("expected_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("expected_sha256 must be a lowercase SHA-256 string")
    _review_date(policy.get("last_review"), "last_review")
    rules = policy.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("rules must be a nonempty array of tables")
    identifiers: set[str] = set()
    required = ("id", "pattern", "external_dependency", "execution_lane", "status", "removal_condition")
    statuses = {
        "separate-process-evidence",
        "variant-ci-lane",
        "explicit-environment-blocker",
        "external-blocked",
        "reasoned-control-flow",
    }
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"rules[{index}] must be a table")
        for key in required:
            value = rule.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"rules[{index}].{key} must be a nonempty string")
        if not isinstance(rule.get("workflow_evidence"), str):
            raise ValueError(f"rules[{index}].workflow_evidence must be a string")
        if rule["id"] in identifiers:
            raise ValueError(f"duplicate coverage policy rule: {rule['id']}")
        identifiers.add(rule["id"])
        if rule["status"] not in statuses:
            raise ValueError(f"unsupported coverage policy status: {rule['status']}")
        re.compile(rule["pattern"], re.IGNORECASE)
        if "last_review" in rule:
            _review_date(rule["last_review"], f"rules[{index}].last_review")
    return policy


def _call_name(node: ast.expr) -> str:
    """Read an AST dotted call name without resolving imports or executing it.

    Parameters
    ----------
    node : ast.expr
        Callable expression from a parsed Python source file.

    Returns
    -------
    str
        Name/attribute suffix assembled from the AST. An unnamed expression
        contributes no base identifier. Aliases are not resolved.
    """
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _source_text(source: str, node: ast.AST | None) -> str:
    """Extract stripped native source text for an AST node.

    Parameters
    ----------
    source : str
        Original decoded source; positions refer to these exact lines.
    node : ast.AST or None
        Parsed node or an absent call argument.

    Returns
    -------
    str
        Stripped source segment, or empty text for absent/unpositioned nodes.
        Extraction does not evaluate the expression.
    """
    if node is None:
        return ""
    return (ast.get_source_segment(source, node) or "").strip()


def _literal_or_source(source: str, node: ast.AST | None) -> str:
    """Describe a skip rationale without evaluating dynamic Python code.

    Parameters
    ----------
    source : str
        Original decoded source.
    node : ast.AST or None
        Reason expression selected from keyword or positional arguments.

    Returns
    -------
    str
        Stripped nonempty literal string, otherwise a dynamic-expression
        description; an absent/unpositioned argument is explicitly unspecified.

    Notes
    -----
    Only ast.literal_eval is attempted. Non-string and blank literals remain
    source descriptions rather than inferred rationale quality.
    """
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
    """Derive a declared owner from a path inside the script's repository.

    Parameters
    ----------
    path : Path
        Physical source path within ROOT.

    Returns
    -------
    str
        python/subpackage for normal package paths, tests/stem or
        validation/stem for those trees, otherwise repository/coverage-policy.

    Raises
    ------
    ValueError
        The path is outside ROOT.
    """
    relative = path.relative_to(ROOT)
    if relative.parts[0] == "src" and len(relative.parts) >= 4:
        return f"python/{relative.parts[2]}"
    if relative.parts[0] in {"tests", "validation"}:
        return f"{relative.parts[0]}/{path.stem.removeprefix('test_')}"
    return "repository/coverage-policy"


def _classify(text: str, policy: dict[str, Any], *, allow_default: bool) -> dict[str, str] | None:
    """Select the first matching declaration in policy order.

    Parameters
    ----------
    text : str
        Condition/reason text, optionally prefixed with the relative path.
    policy : dict[str, Any]
        Validated TOML policy containing ordered rules.
    allow_default : bool
        Include the last fallback candidate when true; omit it when false.

    Returns
    -------
    dict[str, str] or None
        Matched metadata with the pattern removed, or None when no nonfallback
        rule matches and fallback is disabled.

    Raises
    ------
    ValueError
        No rule matches while fallback is enabled.

    Notes
    -----
    Matching is case-insensitive regular-expression search. A lane label is
    textual policy evidence, not proof that any CI job executed that owner.
    """
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
    """Bind one observed exception to declared ownership and a stable ID.

    Parameters
    ----------
    kind : str
        Pragma, exclusion-pattern or selected pytest-call family.
    path : Path
        Native path within ROOT.
    line : int
        One-based source location.
    condition, reason : str
        Already extracted condition and rationale descriptions.
    policy : dict[str, Any]
        Validated policy. Semantic text is classified before path-prefixed text.

    Returns
    -------
    ExceptionEntry
        Frozen declaration. ID uses the first 16 hexadecimal SHA-256 digits of
        NUL-separated kind/path/line/condition/reason UTF-8 text.

    Raises
    ------
    ValueError
        The path lies outside ROOT or no policy rule matches.

    Notes
    -----
    Strictness is not inferred from an xfail call: every recognised xfail gets
    an unexpected-pass retirement criterion. The rule date overrides the
    policy date. Neither the ID nor entry authenticates source/coverage data.
    """
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
        last_review=str(rule.get("last_review", policy["last_review"])),
        removal_condition=removal_condition,
    )


def _pragma_entries(policy: dict[str, Any]) -> list[ExceptionEntry]:
    """Scan package-source lines for exact lexical no-cover spellings.

    Parameters
    ----------
    policy : dict[str, Any]
        Validated ownership declarations.

    Returns
    -------
    list[ExceptionEntry]
        Sorted-path, physical-line occurrences under src/scpn_control. Matching
        is case-sensitive and lexical, including strings/docstrings. Reasons
        lose leading separators before ownership classification.

    Raises
    ------
    ValueError
        A matched line has no rationale after separators or classification fails.
    OSError
        An enumerated source cannot be read.
    UnicodeDecodeError
        A source is not UTF-8.

    Notes
    -----
    This inventory does not use Python tokenization or simulate coverage.py's
    exclusion application. Directory traversal uses native Path.rglob.
    """
    entries: list[ExceptionEntry] = []
    pattern = re.compile(r"pragma:\s*no cover(?P<tail>.*)$")
    for path in sorted((ROOT / "src/scpn_control").rglob("*.py")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = pattern.search(line)
            if match is None:
                continue
            reason = match.group("tail").strip().lstrip("-:;.,#) ]–—").strip()
            if not reason:
                raise ValueError(f"unreasoned coverage pragma: {path.relative_to(ROOT).as_posix()}:{line_number}")
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
    """Inventory three exact pytest dotted call names from parsed source.

    Parameters
    ----------
    policy : dict[str, Any]
        Validated ownership declarations.

    Returns
    -------
    list[ExceptionEntry]
        Calls to pytest.mark.skipif, pytest.skip and pytest.mark.xfail under
        tests and validation. Keyword reason wins over positional extraction.
        Conditions retain source spelling; absent arguments are labelled.

    Raises
    ------
    SyntaxError
        An enumerated Python file cannot be parsed.
    OSError
        A source cannot be read.
    UnicodeDecodeError
        A source is not UTF-8.
    ValueError
        Ownership classification fails.

    Notes
    -----
    Aliases, mark.skip, importorskip and dynamically obtained callables are
    outside this three-name inventory. Decorators and runtime calls are scanned
    without execution. xfail strictness and dependency availability are unchecked.
    """
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
    """Read configured coverage exclusion strings and physical source lines.

    Parameters
    ----------
    policy : dict[str, Any]
        Validated ownership declarations.

    Returns
    -------
    list[ExceptionEntry]
        One entry per tool.coverage.report.exclude_lines item. Location is the
        first line containing its JSON-escaped string spelling, even on a
        duplicate item or earlier comment.

    Raises
    ------
    KeyError
        The expected TOML tables or exclude_lines key are absent.
    StopIteration
        A pattern's JSON-escaped spelling is absent from source lines.
    tomllib.TOMLDecodeError
        pyproject.toml is invalid.
    OSError
        Project text cannot be read.
    UnicodeDecodeError
        Project bytes are not UTF-8.
    ValueError
        Ownership classification fails.
    TypeError
        A malformed project exclusion declaration cannot be iterated/encoded.

    Notes
    -----
    Patterns are inventory declarations; they are not compiled or applied to
    coverage data here. TOML literal/basic-string spelling can affect line lookup.
    """
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
    """Build a deterministic inventory from current repository text.

    Returns
    -------
    dict[str, Any]
        Schema scpn-control.coverage-exception-ledger.v1 with generated_from,
        last_review, entry_count, entry_sha256, sorted counts and sorted entry
        dictionaries. Entries sort by kind/path/line/id; the full-entry digest
        hashes compact, sorted-key JSON encoded as UTF-8.

    Raises
    ------
    ValueError
        Ownership policy is invalid, required workflow substrings are absent,
        a Rust-conditional test owner is absent from workflow text, or an
        observed exception has no rationale/classification.
    re.error
        A rule has an invalid regular expression.
    SyntaxError
        Enumerated test/validation source is invalid Python.
    KeyError, StopIteration
        Project exclusion declarations are missing or cannot be located.
    OSError
        A required policy/source/workflow cannot be read.
    UnicodeDecodeError
        Required text is not UTF-8.

    Notes
    -----
    Paths resolve from the imported script's physical repository, independent
    of caller cwd. Reads are sequential with no atomic snapshot, timestamp or
    clock. Workflow evidence is substring membership in the maintained CI view,
    not enabled-job parsing or execution proof. No source/test code is executed.
    This function does not compare the declared count/digest or write output;
    main performs admission after building, except in summary mode. An inventory
    is not a coverage report, scientific/facility admission or variant execution.
    """
    policy = _load_policy()
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
    """Generate the sealed JSON ledger or check its exact rendered bytes.

    Parameters
    ----------
    argv : list[str] or None, optional
        Native argparse arguments, or process arguments when None. Supports
        --check and --print-summary; paths are fixed to the script repository.

    Returns
    -------
    int
        Zero for informational summary, a current checked ledger or a successful
        write. One for an unreviewed count/digest or stale/missing checked output.

    Raises
    ------
    SystemExit
        Argparse help exits zero; an unknown/malformed option exits two.
    ValueError, SyntaxError, KeyError, StopIteration
        Native build/policy inspection fails; these errors propagate to callers.
    OSError, UnicodeDecodeError
        Input reads or direct UTF-8 output writes fail.

    Notes
    -----
    Summary takes precedence over --check, prints count/digest/counts, does not
    read or write output and does not require the inventory seal to match.
    Other modes reload policy and refuse seal drift before inspecting/writing
    output. Check compares exact indent-two sorted-key JSON with a final newline.
    Default writing creates the output parent and replaces the file directly,
    without atomic replacement or rollback on partial IO. Paths are not confined
    against symlink aliases. Concurrent reads/writes have no snapshot guarantee.
    The CLI leaves inspection exceptions native; uncaught failures normally exit
    one with stderr traceback. A successful seal records inventory ownership,
    not executed coverage, dependency availability or readiness.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args(argv)
    ledger = build_ledger()
    if args.print_summary:
        print(json.dumps({key: ledger[key] for key in ("entry_count", "entry_sha256", "counts")}, indent=2))
        return 0
    policy = _load_policy()
    if ledger["entry_count"] != policy["expected_total"] or ledger["entry_sha256"] != policy["expected_sha256"]:
        print("coverage exception inventory changed without review")
        print(json.dumps({key: ledger[key] for key in ("entry_count", "entry_sha256", "counts")}, indent=2))
        return 1
    rendered = json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUTPUT_PATH.is_file() or OUTPUT_PATH.read_text(encoding="utf-8") != rendered:
            print(f"stale coverage exception ledger: {OUTPUT_PATH.relative_to(ROOT).as_posix()}")
            return 1
        print(f"coverage exception ledger current: {ledger['entry_count']} owned entries")
        return 0
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUTPUT_PATH.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

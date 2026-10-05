# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Semantic source-header policy gate.

"""Validate tracked text headers and reviewed format exemptions.

The TOML policy classifies Git-index paths. Current worktree bytes are read
without rewriting sources; Git HEAD is a separate post-scan observation.
Native comment syntax preserves the same seven semantic identity fields.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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
    """Immutable diagnostic for one rejected tracked path.

    Parameters
    ----------
    path : str
        Native relative path spelling reported by the audit.
    category : str
        ``unclassified``, ``header_mismatch`` or ``non_utf8_enforced``.
    detail : str
        Authored explanation of the classification or header failure.

    Attributes
    ----------
    path : str
        Diagnostic path; construction performs no filesystem access.
    category : str
        Category label, without constructor-level membership validation.
    detail : str
        Explanation, without constructor-level text validation.

    Notes
    -----
    Fields are frozen. Native dataclass initialisation rejects missing or
    unexpected arguments but does not coerce or validate supplied values.
    """

    path: str
    category: str
    detail: str


@dataclass(frozen=True)
class Exemption:
    """Immutable declaration for one reviewed exemption family.

    Parameters
    ----------
    category : str
        Exact category label used in audit counts.
    reason : str
        Review rationale; the loader requires at least 20 stripped characters.
    suffixes : frozenset[str]
        Suffixes; the loader case-folds them before matching.
    names : frozenset[str]
        Case-sensitive basenames, matched anywhere in the repository.
    paths : frozenset[str], optional
        Exact relative POSIX paths. The default is empty.

    Attributes
    ----------
    category, reason : str
        Stored declaration labels and review rationale.
    suffixes, names, paths : frozenset[str]
        Stored match sets, without constructor-level validation.

    Notes
    -----
    Construction freezes fields and preserves supplied values. Validation,
    normalisation and overlap checks occur in ``load_policy``, not here.
    """

    category: str
    reason: str
    suffixes: frozenset[str]
    names: frozenset[str]
    paths: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Policy:
    """Immutable header scope returned by the validated TOML loader.

    Parameters
    ----------
    schema : str
        Expected policy schema ``scpn-control.source-header-policy.v1``.
    enforced_suffixes : frozenset[str]
        Case-folded suffixes that require exact native headers.
    enforced_names : frozenset[str]
        Case-sensitive basenames that require headers.
    exemptions : tuple[Exemption, ...]
        Ordered reviewed format and exact-path declarations.

    Attributes
    ----------
    schema : str
        Stored schema label.
    enforced_suffixes, enforced_names : frozenset[str]
        Stored enforced match sets.
    exemptions : tuple[Exemption, ...]
        Stored declarations in policy order.

    Notes
    -----
    Direct dataclass construction freezes fields but performs no validation.
    Use ``load_policy`` for schema, type, overlap and path checks.
    """

    schema: str
    enforced_suffixes: frozenset[str]
    enforced_names: frozenset[str]
    exemptions: tuple[Exemption, ...]


class FindingPayload(TypedDict):
    """Dictionary representation of one source-header finding.

    Attributes
    ----------
    path : str
        Relative native path spelling.
    category : str
        Classification or header-failure label.
    detail : str
        Authored diagnostic text.

    Notes
    -----
    ``TypedDict`` specifies the static key contract; runtime construction is
    ordinary dictionary construction without value validation.
    """

    path: str
    category: str
    detail: str


class AuditResult(TypedDict):
    """Dictionary result returned by the tracked-file audit.

    Attributes
    ----------
    schema : str
        ``scpn-control.source-header-audit.v1``.
    policy_schema : str
        Validated source-header policy schema.
    source_head : str
        Git HEAD obtained after the tracked-file scan.
    passed : bool
        True precisely when the findings list is empty.
    classifications : dict[str, int]
        Observed ``enforced``, ``exempt`` and ``unclassified`` counts, with sorted
        keys; zero-count categories are omitted.
    exemptions : dict[str, int]
        Observed category counts for exempt paths, with sorted keys.
    findings : list[FindingPayload]
        Findings in sorted tracked-path order.

    Notes
    -----
    These integer counts have no physical units or time axis. ``TypedDict``
    adds no runtime validation, snapshot guarantee or scientific admission.
    """

    schema: str
    policy_schema: str
    source_head: str
    passed: bool
    classifications: dict[str, int]
    exemptions: dict[str, int]
    findings: list[FindingPayload]


def load_policy(path: Path) -> Policy:
    """Read and validate a source-header policy in native TOML.

    Parameters
    ----------
    path : Path
        Policy file; a relative path is resolved by the caller's working directory.

    Returns
    -------
    Policy
        Schema-v1 declarations with suffixes case-folded and duplicate array
        members deduplicated. Names, exact paths, labels and reasons retain their
        spelling. Omitted collection fields and exemptions default to empty.

    Raises
    ------
    ValueError
        The schema or enforced table is invalid; exemptions are not an array of
        tables; collection members are not strings; category/reason checks fail;
        declarations overlap; or an exact path is noncanonical.
    tomllib.TOMLDecodeError
        The file is not valid TOML.
    OSError
        The policy cannot be opened or read.
    UnicodeDecodeError
        TOML bytes are not UTF-8.

    Notes
    -----
    Enforced/exemption suffixes and names must be disjoint; exact paths cannot
    repeat, contain traversal/backslashes, use absolute or normalised spellings,
    or overlap any declared suffix/basename. Unknown keys are ignored. Empty
    match sets are valid. This checks declaration syntax, not rationale quality,
    license correctness, source content or filesystem containment.
    """
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    schema = raw.get("schema")
    if schema != "scpn-control.source-header-policy.v1":
        raise ValueError(f"unsupported source-header policy schema: {schema!r}")
    enforced = raw.get("enforced")
    if not isinstance(enforced, dict):
        raise ValueError("policy requires an [enforced] table")
    entries = raw.get("exemptions", [])
    if not isinstance(entries, list):
        raise ValueError("exemptions must be a TOML array of tables")
    exemptions: list[Exemption] = []
    for entry in entries:
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
                suffixes=frozenset(
                    value.casefold() for value in _string_array(entry, "suffixes", f"exemption {category!r}")
                ),
                names=_string_array(entry, "names", f"exemption {category!r}"),
                paths=_string_array(entry, "paths", f"exemption {category!r}"),
            )
        )
    policy = Policy(
        schema=schema,
        enforced_suffixes=frozenset(value.casefold() for value in _string_array(enforced, "suffixes", "enforced")),
        enforced_names=_string_array(enforced, "names", "enforced"),
        exemptions=tuple(exemptions),
    )
    _validate_disjoint(policy)
    return policy


def _string_array(table: dict[str, object], key: str, owner: str) -> frozenset[str]:
    """Read one optional TOML string array without scalar or member coercion.

    Parameters
    ----------
    table : dict[str, object]
        Parsed enforced or exemption table.
    key : str
        Collection field; an omitted field denotes an empty array.
    owner : str
        Authored table label used in the validation diagnostic.

    Returns
    -------
    frozenset[str]
        Exact string members with duplicates removed.

    Raises
    ------
    ValueError
        The field is not an array or contains a non-string member.
    """
    values = table.get(key, [])
    if not isinstance(values, list):
        raise ValueError(f"{owner}.{key} must be a TOML array of strings")
    result: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"{owner}.{key} must be a TOML array of strings")
        result.add(value)
    return frozenset(result)


def _validate_disjoint(policy: Policy) -> None:
    """Reject overlapping declarations and noncanonical exact paths.

    Parameters
    ----------
    policy : Policy
        Parsed declarations; suffixes are already case-folded by the loader.

    Raises
    ------
    ValueError
        Match sets overlap or an exact path is not canonical relative POSIX.

    Notes
    -----
    Sets are temporary; the frozen policy and filesystem are unchanged.
    """
    seen_suffixes = set(policy.enforced_suffixes)
    seen_names = set(policy.enforced_names)
    seen_paths: set[str] = set()
    for exemption in policy.exemptions:
        overlap_suffixes = seen_suffixes.intersection(exemption.suffixes)
        overlap_names = seen_names.intersection(exemption.names)
        if overlap_suffixes or overlap_names or seen_paths.intersection(exemption.paths):
            raise ValueError(
                f"overlapping source-header policy entries in {exemption.category}: "
                f"suffixes={sorted(overlap_suffixes)}, names={sorted(overlap_names)}"
            )
        seen_suffixes.update(exemption.suffixes)
        seen_names.update(exemption.names)
        seen_paths.update(exemption.paths)
    for value in seen_paths:
        path = PurePosixPath(value)
        if not value or path.is_absolute() or ".." in path.parts or value != path.as_posix() or "\\" in value:
            raise ValueError(f"exemption path must be an exact relative POSIX path: {value!r}")
        if path.suffix.casefold() in seen_suffixes or path.name in seen_names:
            raise ValueError(f"overlapping source-header path exemption: {value!r}")


def tracked_paths(root: Path) -> list[Path]:
    """List tracked Git paths without opening their contents.

    Parameters
    ----------
    root : Path
        Working directory for ``git ls-files -z``.

    Returns
    -------
    list[Path]
        Sorted decoded index paths. Ignored and untracked files are absent.

    Raises
    ------
    subprocess.CalledProcessError
        Git rejects the repository or command.
    OSError
        Git cannot start or the working directory is unavailable.
    UnicodeDecodeError
        A tracked filename is not UTF-8.

    Notes
    -----
    Git may discover an enclosing repository. Names come from its index, not a
    filesystem traversal, HEAD tree or atomic content snapshot. No source is
    modified.
    """
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return sorted(Path(raw.decode("utf-8")) for raw in completed.stdout.split(b"\0") if raw)


def classify(path: Path, policy: Policy) -> tuple[str, str]:
    """Match a path against enforced and reviewed exemption declarations.

    Parameters
    ----------
    path : Path
        Supplied path spelling; suffix matching is case-folded, names and exact
        ``as_posix`` paths are case-sensitive. No root resolution is performed.
    policy : Policy
        Loaded declarations or a caller-constructed policy.

    Returns
    -------
    tuple[str, str]
        ``("enforced", "")``, ``("exempt", category)`` or
        ``("unclassified", "")``. Enforced matches take precedence; exemption
        families are checked in their stored order.

    Notes
    -----
    This function performs no I/O or policy revalidation. It does not prove
    that a path exists, is relative, lies within a repository or has a header.
    """
    suffix = path.suffix.casefold()
    if suffix in policy.enforced_suffixes or path.name in policy.enforced_names:
        return "enforced", ""
    for exemption in policy.exemptions:
        if suffix in exemption.suffixes or path.name in exemption.names or path.as_posix() in exemption.paths:
            return "exempt", exemption.category
    return "unclassified", ""


def expected_header(path: Path, purpose: str) -> list[str]:
    """Render the exact native syntax for seven semantic header fields.

    Parameters
    ----------
    path : Path
        Suffix selects syntax: Lean block, HTML comments, slash comments for
        ``.cpp/.h/.js/.mjs/.rs/.ts/.tsx`` and hash comments otherwise.
    purpose : str
        Text following ``SCPN Control — ``; rendering does not validate it.

    Returns
    -------
    list[str]
        Seven physical lines, or nine for Lean including ``/-`` and ``-/``.
        The first six semantic fields have exact repository identity text.

    Notes
    -----
    No file is read or written. A blank or multiline purpose can render a
    header rejected by ``header_finding``; this renderer is not admission.
    """
    content = [*EXPECTED, f"SCPN Control — {purpose}"]
    suffix = path.suffix.casefold()
    if suffix == ".lean":
        return ["/-", *content, "-/"]
    if suffix == ".html":
        return [f"<!-- {line} -->" for line in content]
    marker = "//" if suffix in SLASH_SUFFIXES else "#"
    return [f"{marker} {line}" for line in content]


def header_finding(root: Path, path: Path) -> Finding | None:
    """Check current UTF-8 file bytes for exact native header semantics.

    Parameters
    ----------
    root : Path
        Base joined with ``path``; absolute paths retain native Path behaviour.
    path : Path
        File path and reported diagnostic spelling. The function does not check
        whether the file is tracked or belongs to enforced scope.

    Returns
    -------
    Finding or None
        None for a valid header. Otherwise ``header_mismatch`` or
        ``non_utf8_enforced`` with an authored diagnostic.

    Raises
    ------
    OSError
        The file cannot be read, is absent or is a directory.

    Notes
    -----
    One leading shebang is accepted. ``splitlines`` accepts native newline
    spellings; the six identity lines are exact. The final purpose must have
    the native prefix/closing syntax and nonblank text. Later file content is
    ignored. This read follows native path/symlink behaviour without containment,
    legal-content review, mutation or atomic snapshot guarantees.
    """
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
    """Audit current disk headers for all paths in the Git index.

    Parameters
    ----------
    root : Path
        Resolved working directory for Git enumeration and subsequent reads.
    policy_path : Path
        TOML file; relative paths remain relative to the process working directory,
        independently of ``root``.

    Returns
    -------
    AuditResult
        Schema-v1 counts, findings and Git HEAD. An empty tracked repository can
        pass; absent count categories are omitted. Exempt files are not read.

    Raises
    ------
    ValueError
        Policy validation fails.
    tomllib.TOMLDecodeError
        The policy TOML is malformed.
    OSError
        Required policy/source reads or Git startup fail.
    subprocess.CalledProcessError
        Git enumeration or HEAD lookup fails.
    UnicodeDecodeError
        Policy bytes or tracked filenames are not UTF-8. Non-UTF-8 enforced
        file content is instead represented as a finding.

    Notes
    -----
    Policy validation precedes Git access. Sorted paths come from the current
    index; content comes from the worktree; HEAD is queried after the scan.
    These are separate observations, not a commit-content/atomic custody proof.
    The audit writes no source or report and admits no scientific evidence.
    """
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
    """Run the tracked-source audit with shell-compatible output and status.

    Parameters
    ----------
    argv : list[str] or None, optional
        Native argparse arguments; None reads process arguments. ``--root`` and
        ``--policy`` default to the script repository and its policy file.
        Explicit relative options use the process working directory.

    Returns
    -------
    int
        Zero for no findings; one for findings; two for caught policy, filesystem,
        decoding or Git errors. ``--json`` emits the complete result for zero/one.
        Text mode prints success to stdout or each finding to stderr. Caught
        errors print ``source-header policy error:`` to stderr without JSON.

    Raises
    ------
    SystemExit
        Argparse help exits zero and invalid arguments exit two before auditing.

    Notes
    -----
    Output does not mutate the policy, repository, Git index or reports. A zero
    status establishes only this local tracked-source header result.
    """
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

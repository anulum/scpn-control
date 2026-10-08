# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Competitive evidence manifest and public-page gate.

"""Validate the dated competitive evidence registry and its public rendering.

This reads local JSON, TOML and UTF-8 Markdown without fetching cited sources,
resolving commit SHAs or running a quantitative comparison. Version/date/field
checks and literal page-presence checks establish declaration consistency, not
the truth of a competitor assessment. Defaults are script-relative; explicit
paths are relative to caller cwd. Inputs are read sequentially without locking.
The CLI writes only stdout/stderr and refuses input failures with fixed text.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from datetime import date
from pathlib import Path
from typing import Any, Final, TypedDict, cast
from urllib.parse import urlsplit

ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST: Final = ROOT / "docs/_data/competitive_evidence.json"
DEFAULT_PAGE: Final = ROOT / "docs/competitive_analysis.md"
DEFAULT_PROJECT: Final = ROOT / "pyproject.toml"
SCHEMA: Final = "scpn-control.competitive-evidence.v1"
ALLOWED_KINDS: Final = frozenset({"repository-snapshot", "release", "paper"})
PRIMARY_SOURCE_HOSTS: Final = frozenset({"doi.org", "github.com", "www.omfit.io"})
SYSTEM_FIELDS: Final = frozenset(
    {
        "id",
        "name",
        "artifact_kind",
        "assessed_version",
        "artifact_date",
        "commit_sha",
        "primary_role",
        "documented_strengths",
        "evidence_boundary",
        "source_urls",
    }
)
COMPARISON_FIELDS: Final = frozenset(
    {
        "system_a",
        "system_b",
        "problem_id",
        "inputs_digest",
        "precision",
        "tolerances",
        "convergence_criteria",
        "warmup_and_compilation",
        "sample_definition",
        "hardware_and_load",
        "isolation",
        "failure_accounting",
        "result_artifact",
    }
)
FULL_SHA: Final = re.compile(r"[0-9a-f]{40}")
SYSTEM_ID: Final = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
VERSION: Final = re.compile(r"v[0-9]+\.[0-9]+\.[0-9]+")
FORBIDDEN_TONE: Final = re.compile(
    r"\b(?:superior|inferior|best[- ]in[- ]class|beats?|crush(?:es|ed)?|"
    r"no competitor|uniquely dominant|obsolete)\b",
    re.IGNORECASE,
)
PRIVATE_MARKER: Final = re.compile(
    r"(?:docs/internal/|\.coordination/|competitor-ahead|TIER 0)",
    re.IGNORECASE,
)


class AuditResult(TypedDict):
    """Declaration findings, selected paths and counts from the loaded registry.

    ``passed`` means ``errors`` is empty. Counts describe supplied lists, even
    when findings exist; they are not independently qualified evidence totals.
    Paths retain their supplied spelling. This result is not signed or a proof
    that cited sources, comparison results or page claims are correct.
    """

    schema: str
    passed: bool
    manifest: str
    page: str
    project_version: str
    system_count: int
    quantitative_comparison_count: int
    errors: list[str]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while refusing duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_manifest(path: Path) -> dict[str, Any]:
    """Decode a UTF-8 JSON object, refusing duplicate keys at any nesting depth.

    Return the mutable decoded mapping without semantic validation or a digest.
    Filesystem and decoding errors propagate. ``ValueError`` covers duplicate
    keys and a non-object root; duplicate-key errors include the supplied key.
    """
    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(payload, dict):
        raise ValueError("competitive evidence manifest must be a JSON object")
    return cast(dict[str, Any], payload)


def load_project_version(path: Path) -> str:
    """Read an ASCII major.minor.patch version for the named SCPN package.

    TOML must identify ``project.name = "scpn-control"`` and provide three
    decimal components without a leading ``v``. Return the original spelling;
    this does not apply packaging-version normalisation or fetch a release.
    Filesystem/TOML errors propagate; invalid identity/version raises
    ``ValueError`` with an authored message.
    """
    with path.open("rb") as stream:
        payload = tomllib.load(stream)
    project = payload.get("project")
    if not isinstance(project, dict) or project.get("name") != "scpn-control":
        raise ValueError("project metadata must identify scpn-control")
    version = project.get("version")
    if not isinstance(version, str) or VERSION.fullmatch(f"v{version}") is None:
        raise ValueError("project metadata requires an exact major.minor.patch version")
    return version


def _parse_date(value: object, label: str, errors: list[str]) -> date | None:
    """Parse the standard-library ISO date forms or append the labelled finding."""
    if not isinstance(value, str):
        errors.append(f"{label} must be an ISO date")
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        errors.append(f"{label} must be an ISO date")
        return None


def _meaningful(value: object, minimum: int = 20) -> bool:
    """Check trimmed string length without interpreting its evidential meaning."""
    return isinstance(value, str) and len(value.strip()) >= minimum


def validate_manifest(payload: dict[str, Any], *, today: date | None = None) -> list[str]:
    """Collect schema, age, version, entry and comparison-declaration findings.

    Parameters
    ----------
    payload : dict
        Decoded registry, inspected without mutation. System/comparison fields
        must match their declared sets; other top-level fields are not rejected.
    today : date, optional
        Evaluation day. Defaults to the local calendar date; evidence ages from
        zero through 120 days are accepted. Artifact dates are parsed only.

    Returns
    -------
    list of str
        Findings in check order. Source SHAs are syntax-checked; admitted HTTPS
        hosts and path markers are checked without fetching or SHA resolution.
        A complete quantitative row checks field shape and text length, without
        comparing results, participants or tolerances. No ranking is established.

    Notes
    -----
    Malformed URL strings can raise ``ValueError`` from the URL parser rather
    than becoming a finding. The CLI maps such failures to status two.
    """
    errors: list[str] = []
    current_date = today or date.today()
    if payload.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")

    evidence_date = _parse_date(payload.get("as_of"), "as_of", errors)
    if evidence_date is not None:
        age = (current_date - evidence_date).days
        if age < 0:
            errors.append("as_of cannot be in the future")
        elif age > 120:
            errors.append("competitive evidence is older than 120 days")

    project_version = payload.get("project_version")
    if not isinstance(project_version, str) or VERSION.fullmatch(f"v{project_version}") is None:
        errors.append("project_version must be an exact major.minor.patch release")

    policy = payload.get("comparison_policy")
    if not isinstance(policy, dict):
        errors.append("comparison_policy must be an object")
    else:
        if policy.get("absence_state") != "not assessed":
            errors.append("comparison_policy.absence_state must be 'not assessed'")
        for field in ("numeric_admission", "scope"):
            if not _meaningful(policy.get(field)):
                errors.append(f"comparison_policy.{field} requires a specific statement")

    systems = payload.get("systems")
    if not isinstance(systems, list) or len(systems) < 8:
        errors.append("systems must contain at least eight assessed entries")
        systems = []
    seen_ids: set[str] = set()
    for index, raw_system in enumerate(systems):
        label = f"systems[{index}]"
        if not isinstance(raw_system, dict):
            errors.append(f"{label} must be an object")
            continue
        system = cast(dict[str, Any], raw_system)
        if frozenset(system) != SYSTEM_FIELDS:
            errors.append(f"{label} fields must exactly match schema")
        system_id = system.get("id")
        if not isinstance(system_id, str) or SYSTEM_ID.fullmatch(system_id) is None:
            errors.append(f"{label}.id must be a stable lowercase identifier")
        elif system_id in seen_ids:
            errors.append(f"duplicate system id: {system_id}")
        else:
            seen_ids.add(system_id)
        name = system.get("name")
        if not _meaningful(name, 2):
            errors.append(f"{label}.name is required")
        kind = system.get("artifact_kind")
        if not isinstance(kind, str) or kind not in ALLOWED_KINDS:
            errors.append(f"{label}.artifact_kind is unsupported")
        assessed_version = system.get("assessed_version")
        if not _meaningful(assessed_version, 4):
            errors.append(f"{label}.assessed_version is required")
        _parse_date(system.get("artifact_date"), f"{label}.artifact_date", errors)
        commit_sha = system.get("commit_sha")
        if kind in ("release", "repository-snapshot") and (
            not isinstance(commit_sha, str) or FULL_SHA.fullmatch(commit_sha) is None
        ):
            errors.append(f"{label}.commit_sha must bind source evidence")
        if kind == "paper" and commit_sha is not None:
            errors.append(f"{label}.commit_sha must be null for paper evidence")
        if not _meaningful(system.get("primary_role")):
            errors.append(f"{label}.primary_role requires a specific statement")
        if not _meaningful(system.get("evidence_boundary")):
            errors.append(f"{label}.evidence_boundary requires a specific statement")
        strengths = system.get("documented_strengths")
        if (
            not isinstance(strengths, list)
            or len(strengths) < 2
            or not all(_meaningful(item, 12) for item in strengths)
        ):
            errors.append(f"{label}.documented_strengths requires at least two specific statements")
        urls = system.get("source_urls")
        if not isinstance(urls, list) or not urls:
            errors.append(f"{label}.source_urls requires primary sources")
        else:
            for url in urls:
                if not isinstance(url, str) or not url.startswith("https://"):
                    errors.append(f"{label}.source_urls must use HTTPS")
                elif urlsplit(url).hostname not in PRIMARY_SOURCE_HOSTS:
                    errors.append(f"{label}.source_urls must use an admitted primary-source host")
                elif kind in ("release", "repository-snapshot") and (
                    "/latest" in url or "/main/" in url or "/master/" in url
                ):
                    errors.append(f"{label}.source_urls must bind the assessed release")

    if "scpn-control" not in seen_ids:
        errors.append("systems must include scpn-control")

    comparisons = payload.get("quantitative_comparisons")
    if not isinstance(comparisons, list):
        errors.append("quantitative_comparisons must be a list")
    else:
        for index, raw_comparison in enumerate(comparisons):
            label = f"quantitative_comparisons[{index}]"
            if not isinstance(raw_comparison, dict):
                errors.append(f"{label} must be an object")
                continue
            comparison = cast(dict[str, Any], raw_comparison)
            if frozenset(comparison) != COMPARISON_FIELDS:
                errors.append(f"{label} fields must exactly match the matched protocol")
                continue
            if not all(_meaningful(value, 2) for value in comparison.values()):
                errors.append(f"{label} cannot contain empty protocol fields")
    return errors


def validate_page(page: str, payload: dict[str, Any]) -> list[str]:
    """Collect literal page-presence, vocabulary and disclosure findings.

    Expected dates, versions, system names and URLs are tested as substrings,
    including text in comments or code blocks. The fixed ranking/private-marker
    regexes do not constitute semantic review. Empty comparisons require the
    declared empty-set sentence. No Markdown rendering, URL fetching or input
    mutation occurs; malformed registry types are handled by manifest validation.
    """
    errors: list[str] = []
    if str(payload.get("as_of")) not in page:
        errors.append("public page must display the manifest evidence date")
    if str(payload.get("project_version")) not in page:
        errors.append("public page must display the assessed SCPN version")
    if "docs/_data/competitive_evidence.json" not in page:
        errors.append("public page must link the machine-readable registry")
    systems = payload.get("systems", [])
    if isinstance(systems, list):
        for raw_system in systems:
            if not isinstance(raw_system, dict):
                continue
            system = cast(dict[str, Any], raw_system)
            for field in ("name", "assessed_version"):
                value = system.get(field)
                if isinstance(value, str) and value not in page:
                    errors.append(f"public page is missing {field} for {system.get('id')}")
            urls = system.get("source_urls", [])
            if isinstance(urls, list):
                for url in urls:
                    if isinstance(url, str) and url not in page:
                        errors.append(f"public page is missing primary source {url}")
    tone = FORBIDDEN_TONE.search(page)
    if tone is not None:
        errors.append(f"public page uses ranking language: {tone.group(0)}")
    private = PRIVATE_MARKER.search(page)
    if private is not None:
        errors.append(f"public page exposes private planning marker: {private.group(0)}")
    comparisons = payload.get("quantitative_comparisons")
    if comparisons == [] and "quantitative comparison set is currently empty" not in page:
        errors.append("public page must disclose that no cross-project numeric comparison is admitted")
    return errors


def audit(
    manifest_path: Path,
    page_path: Path,
    project_path: Path = DEFAULT_PROJECT,
    *,
    today: date | None = None,
) -> AuditResult:
    """Inspect the selected registry, metadata and page as one declaration set.

    Read the registry, collect its findings, read package metadata, compare its
    exact version, then inspect the page. Relative paths resolve from caller cwd;
    replacing manifest/page paths does not change the script-relative default
    project metadata path. Files are observed sequentially, without mutation or
    a snapshot guarantee. Native input failures propagate to API callers.

    Return an ``AuditResult`` containing both finding sets and supplied-list
    counts. ``passed`` establishes local declaration consistency only.

    Examples
    --------
    >>> registry = load_manifest(DEFAULT_MANIFEST)
    >>> result = audit(DEFAULT_MANIFEST, DEFAULT_PAGE)
    >>> result["quantitative_comparison_count"] == len(registry["quantitative_comparisons"])
    True
    """
    payload = load_manifest(manifest_path)
    errors = validate_manifest(payload, today=today)
    project_version = load_project_version(project_path)
    if payload.get("project_version") != project_version:
        errors.append("manifest project_version must match pyproject.toml")
    errors.extend(validate_page(page_path.read_text(encoding="utf-8"), payload))
    return {
        "schema": "scpn-control.competitive-evidence-audit.v1",
        "passed": not errors,
        "manifest": str(manifest_path),
        "page": str(page_path),
        "project_version": project_version,
        "system_count": len(payload.get("systems", [])) if isinstance(payload.get("systems"), list) else 0,
        "quantitative_comparison_count": len(payload.get("quantitative_comparisons", []))
        if isinstance(payload.get("quantitative_comparisons"), list)
        else 0,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    """Audit selected files and return zero/pass, one/findings or two/input failure.

    ``--json`` emits ``AuditResult`` on stdout for zero/one; text mode sends
    findings to stderr. Caught filesystem, decoding, TOML/JSON and value errors
    produce fixed stderr text and no JSON. No report file or remote request is
    created. Argparse help/usage follows its native ``SystemExit`` behaviour.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--page", type=Path, default=DEFAULT_PAGE)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    try:
        result = audit(args.manifest, args.page, args.project)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, tomllib.TOMLDecodeError, ValueError):
        print("competitive evidence error: inputs could not be inspected", file=sys.stderr)
        return 2
    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["passed"]:
        print("Competitive evidence passed")
    else:
        for error in result["errors"]:
            print(f"competitive evidence: {error}", file=sys.stderr)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

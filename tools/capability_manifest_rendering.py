# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Capability inventory output.
"""Render and compare local inventory snapshots and publish selected files.

This owner formats a supplied inventory, compares UTF-8 snapshot text with universal newline normalisation,
and replaces the designated README block. Filesystem write failures may leave
partial outputs; rendering and marker validation must finish before writes.
The fragments describe source inventory, not implementation execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.capability_manifest_inventory import ManifestError, _load_config, build_manifest

README_START = "<!-- capability-snapshot:start -->"
README_END = "<!-- capability-snapshot:end -->"


def render_json(manifest: dict[str, Any]) -> str:
    """Serialise a supplied mapping as deterministic indented Unicode JSON text.

    Sort keys, preserve non-ASCII characters and append one LF. NaN/Infinity
    raise ValueError; unsupported JSON values raise TypeError. This serialiser
    does not validate inventory counts/schema or execute listed capabilities.
    """
    return json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"


def render_markdown(manifest: dict[str, Any]) -> str:
    """Render selected declared metadata/counts as the README-safe inventory table.

    Requires the counts/project fields produced by build_manifest. Preserve
    row order and include the fixed generator/check commands and declared source/Rust-root
    description. The fragment is Markdown text, not a native feature renderer,
    execution result or scientific-readiness report. Missing/unsupported field
    shapes propagate errors; no files are read or written.
    """
    counts = manifest["counts"]
    project = manifest["project"]
    rows = [
        ("Package version", project["version"]),
        ("Python requirement", project["python_requires"]),
        ("Project scripts", counts["project_script_count"]),
        ("Public API exports", counts["public_api_export_count"]),
        ("Python control/physics modules", counts["source_module_count"]),
        ("Python public classes", counts["public_class_count"]),
        ("Rust source files", counts["rust_source_file_count"]),
        ("Rust PyO3 exports", counts["pyo3_export_count"]),
        ("Validation scripts", counts["validation_script_count"]),
        ("Optional extras", len(project["optional_extras"])),
        ("Python test files", counts["python_test_file_count"]),
        ("Public documentation pages", counts["public_markdown_count"]),
        ("GitHub Actions workflows", counts["workflow_count"]),
    ]
    lines = [
        "**Capability Inventory**",
        "",
        "| Surface | Count |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {label} | {value} |" for label, value in rows)
    source_roots = ", ".join(f"`{path}`" for path in manifest["python"]["source_roots"])
    lines.extend(
        [
            "",
            f"**Evidence roots:** {source_roots}, `{manifest['rust']['workspace_root']}`, "
            "`validation`, `tests`, `docs`, and `.github/workflows`.",
            "",
            "Refresh with `python tools/capability_manifest.py`; enforce with "
            "`python tools/capability_manifest.py --check`.",
            "",
        ]
    )
    return "\n".join(lines)


def _readme_bounds(readme: str) -> tuple[int, int]:
    """Locate exactly one ordered README marker pair or raise ValueError.

    Return indices at each marker's beginning. Missing, repeated or reversed
    markers cannot identify an unambiguous replaceable inventory fragment.
    """
    if readme.count(README_START) != 1 or readme.count(README_END) != 1:
        raise ValueError("README.md is missing or has invalid capability snapshot markers")
    start = readme.index(README_START)
    end = readme.index(README_END)
    if end < start:
        raise ValueError("README.md is missing or has invalid capability snapshot markers")
    return start, end


def extract_readme_block(readme: str) -> str:
    """Return the fragment inside one ordered README marker pair.

    Remove at most one leading LF after the start marker and retain the rest
    exactly as supplied. Missing/duplicate/reversed markers raise ValueError.
    Callers that read files with read_text apply universal newline conversion;
    the extractor itself neither parses Markdown nor normalises other text.
    """
    start, end = _readme_bounds(readme)
    start += len(README_START)
    block = readme[start:end]
    if block.startswith("\n"):
        block = block[1:]
    return block


def _replace_readme_block(readme: str, markdown: str) -> str:
    """Replace one ordered marker fragment while preserving all surrounding text.

    Insert one newline after the start marker and retain both markers.
    ManifestError wraps missing/ambiguous/reversed marker failures, before
    any publisher writes its prepared output strings.
    """
    try:
        start, end = _readme_bounds(readme)
    except ValueError as exc:
        raise ManifestError(str(exc)) from exc
    end += len(README_END)
    return f"{readme[:start]}{README_START}\n{markdown}{README_END}{readme[end:]}"


def write_outputs(repo_root: Path | str | None = None) -> None:
    """Build/render the selected inventory and publish configured local snapshots.

    Parameters
    ----------
    repo_root
        Root as Path/string; None uses cwd. Resolve the root and read its
        configuration. Publish JSON/Markdown and replace the configured README
        marker fragment. Existing output files are overwritten.

    Raises
    ------
    ManifestError
        Inventory inspection/invariants or README markers fail. Build all
        output text and validate the marker pair before creating directories
        or writing outputs, so those preparation failures preserve existing
        files. Filesystem/decode/configuration/serialisation errors propagate.

    Notes
    -----
    Writes are sequential and not a three-file transaction; an OS write error
    can leave partial outputs. UTF-8 text writes use platform newline handling.
    This operation does not commit, publish externally or admit capabilities.
    """
    repo_root = Path.cwd() if repo_root is None else Path(repo_root)
    repo_root = repo_root.resolve()
    config = _load_config(repo_root)
    manifest = build_manifest(repo_root)
    markdown = render_markdown(manifest)
    json_path = repo_root / config["paths"]["json_output"]
    markdown_path = repo_root / config["paths"]["markdown_output"]
    readme_path = repo_root / config["paths"]["readme"]

    # Prepare every rendered value and validate README before mutating files.
    json_text = render_json(manifest)
    readme_text = _replace_readme_block(readme_path.read_text(encoding="utf-8"), markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_text, encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    readme_path.write_text(readme_text, encoding="utf-8")


def check_outputs(repo_root: Path | str | None = None) -> list[str]:
    """Compare actual snapshots against a fresh configured source inventory.

    Parameters
    ----------
    repo_root
        Root as Path/string; None uses cwd. Resolve it and read config/source
        plus JSON/Markdown/README output text without rewriting any file.

    Returns
    -------
    list[str]
        Missing/stale generated files followed by invalid/stale README block
        diagnostics, in deterministic order. Empty means normalised UTF-8 text
        matches current rendering; read_text permits LF/CRLF equivalence, not
        arbitrary whitespace or JSON reformatting. No runtime readiness follows.

    Raises
    ------
    ManifestError
        Source inventory cannot be built/validated. File/decode/configuration
        errors propagate; a missing/unreadable README is a read failure.
    """
    repo_root = Path.cwd() if repo_root is None else Path(repo_root)
    repo_root = repo_root.resolve()
    config = _load_config(repo_root)
    manifest = build_manifest(repo_root)
    markdown = render_markdown(manifest)
    expected = {
        config["paths"]["json_output"]: render_json(manifest),
        config["paths"]["markdown_output"]: markdown,
    }
    failures: list[str] = []
    for rel_path, expected_text in expected.items():
        path = repo_root / rel_path
        if not path.exists():
            failures.append(f"{rel_path} is missing")
            continue
        if path.read_text(encoding="utf-8") != expected_text:
            failures.append(f"{rel_path} is stale")
    readme_path = repo_root / config["paths"]["readme"]
    readme = readme_path.read_text(encoding="utf-8")
    try:
        embedded = extract_readme_block(readme)
    except ValueError:
        failures.append("README.md is missing or has invalid capability snapshot markers")
    else:
        if embedded != markdown:
            failures.append("README.md capability snapshot is stale")
    return failures

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Static capability inventory.
"""Discover configured source declarations without executing their implementations.

This owner reads TOML metadata, Python AST declarations and literal __all__,
Rust wrapper text, and configured file trees. Labels/counts describe present
source text, including working-tree files; they do not admit runtime capability,
scientific accuracy, benchmark performance, test success or publication.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("tools/capability_manifest.toml")


class ManifestError(RuntimeError):
    """Raised when a capability inventory cannot be inspected or validated."""


def _relative(repo_root: Path, path: Path) -> str:
    """Return a resolved path relative to the resolved repository root, using POSIX separators.

    Symlinks resolve before containment; a path outside the root raises
    ValueError. Resolution/filesystem errors propagate.
    """
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _read_toml(path: Path) -> dict[str, Any]:
    """Parse one binary-opened TOML carrier without changing it.

    Filesystem and TOMLDecodeError failures propagate. Values retain TOML
    types; this reader performs no project/configuration schema admission.
    """
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _load_config(repo_root: Path) -> dict[str, Any]:
    """Read tools/capability_manifest.toml beneath the supplied repository root.

    Missing or malformed configuration propagates its read/parse failure.
    The configured path declarations select the inventory and output scope.
    """
    return _read_toml(repo_root / CONFIG_PATH)


def _load_pyproject(repo_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Read the configured project TOML path under the selected root.

    Requires config.paths.pyproject; missing keys and read/parse errors
    propagate. No installed-distribution or remote-version query is made.
    """
    return _read_toml(repo_root / config["paths"]["pyproject"])


def _iter_existing_files(repo_root: Path, roots: list[str], suffix: str) -> list[Path]:
    """List matching regular files recursively beneath declared roots in path order.

    Missing roots are omitted, so this is presence discovery, not a required
    component-availability gate. Overlapping roots retain repeated entries.
    The suffix is matched by rglob; callers decide whether __init__ is a
    source module and whether an entry belongs to the public inventory.
    """
    files: list[Path] = []
    for root in roots:
        path = repo_root / root
        if not path.exists():
            continue
        files.extend(file for file in path.rglob(f"*{suffix}") if file.is_file())
    return sorted(files)


def _python_classes(path: Path) -> list[str]:
    """List public class names from a Python source AST, including nested declarations.

    Names beginning with underscore are omitted; no source module is executed.
    Invalid syntax raises ManifestError with the owner path instead of turning
    failed inspection into an empty class list. File/decode errors propagate.
    Names are unqualified labels; build_manifest deduplicates labels globally.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        raise ManifestError(f"could not inspect Python source {path}: {exc}") from exc
    return sorted(
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
    )


def _public_exports(package_init: Path) -> list[str]:
    """Read the first literal __all__ assignment in the package module body.

    A literal list/tuple of nonempty strings is required when the assignment
    exists. Missing __all__ returns an empty list, refused by the manifest
    validator. Dynamic/non-string export carriers raise ManifestError;
    syntax/read failures propagate. Returned sorted labels are declarations,
    not proof that those names can be imported at runtime.
    """
    tree = ast.parse(package_init.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError) as exc:
                    raise ManifestError(f"__all__ must be a literal string sequence: {package_init}") from exc
                if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) or not item for item in value):
                    raise ManifestError(f"__all__ must be a literal string sequence: {package_init}")
                return sorted(value)
    return []


def _rust_pyo3_exports(path: Path) -> list[str]:
    """List sorted unique PyO3 names found by text patterns in one wrapper source.

    Missing wrappers return an empty list, refused by manifest validation.
    Match pyfunction/pyclass declarations and registration macros. This is
    lexical discovery, including matching comments/text; it does not compile
    Rust, load an extension or verify cross-language numerical equivalence.
    Read and UTF-8 errors propagate.
    """
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    exports: set[str] = set()
    exports.update(re.findall(r"#\[pyfunction\]\s*fn\s+([A-Za-z_][A-Za-z0-9_]*)", text))
    exports.update(re.findall(r"#\[pyclass\]\s*(?:#\[[^\]]+\]\s*)*(?:pub\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)", text))
    exports.update(re.findall(r"wrap_pyfunction!\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,", text))
    exports.update(re.findall(r"add_class::<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>", text))
    return sorted(exports)


def _docs_markdown(repo_root: Path, config: dict[str, Any]) -> list[str]:
    """List sorted Markdown paths excluding configured whole path components.

    Scan docs_root recursively and exclude any path containing a component
    from exclude_doc_parts. Return resolved root-relative POSIX paths. A
    missing docs root produces an empty inventory; read/resolve errors or
    paths escaping the root propagate. Markdown is not rendered or admitted.
    """
    docs_root = repo_root / config["paths"]["docs_root"]
    excluded = set(config.get("exclude_doc_parts", []))
    files = []
    for path in docs_root.rglob("*.md"):
        rel_parts = set(path.relative_to(repo_root).parts)
        if rel_parts & excluded:
            continue
        files.append(_relative(repo_root, path))
    return sorted(files)


def _project_metadata(pyproject: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Project selected declared package fields from TOML and configuration.

    Include label/name/package/version/Python requirement, sorted optional
    extra names and sorted project scripts. Required keys must exist. Script
    names/targets are stringified here and then lexically checked by the
    manifest validator; dependencies or entry points are not imported.
    """
    project = pyproject["project"]
    optional_deps = project.get("optional-dependencies", {})
    scripts = project.get("scripts", {})
    return {
        "label": config["project_label"],
        "name": project["name"],
        "package": config["package_name"],
        "version": project["version"],
        "python_requires": project["requires-python"],
        "optional_extras": sorted(optional_deps),
        "scripts": dict(sorted((str(name), str(target)) for name, target in scripts.items())),
    }


def build_manifest(repo_root: Path | str | None = None) -> dict[str, Any]:
    """Discover configured working-tree source labels and return a checked inventory.

    Parameters
    ----------
    repo_root
        Repository root as Path/string; None uses the caller's cwd. Resolve
        paths and read configuration there, without importing scanned sources.

    Returns
    -------
    dict[str, Any]
        Configured schema/project metadata, sorted source/file/export labels
        and their counts. Python modules exclude __init__.py; public class
        names include nested declarations and are globally deduplicated by
        unqualified name. Validation/tests/workflows/docs and Rust are file
        presence/text inventories. No execution, test pass, facility validity
        or release availability is certified.

    Raises
    ------
    ManifestError
        Inspection syntax/export data or internal count/required-label checks
        fail. File/decoding/TOML errors and unsupported configuration shapes
        propagate. Missing optional scan roots remain omitted by discovery.

    Examples
    --------
    Inspect the real caller's checkout without running scanned implementations:

    >>> snapshot = build_manifest()
    >>> snapshot["counts"]["project_script_count"] == len(snapshot["project"]["scripts"])
    True
    """
    repo_root = Path.cwd() if repo_root is None else Path(repo_root)
    repo_root = repo_root.resolve()
    config = _load_config(repo_root)
    pyproject = _load_pyproject(repo_root, config)
    paths = config["paths"]

    source_files = _iter_existing_files(repo_root, list(paths["source_roots"]), ".py")
    source_modules = [_relative(repo_root, path) for path in source_files if path.name != "__init__.py"]
    public_classes = sorted({class_name for path in source_files for class_name in _python_classes(path)})
    public_exports = _public_exports(repo_root / paths["package_root"] / "__init__.py")

    rust_files = [_relative(repo_root, path) for path in _iter_existing_files(repo_root, [paths["rust_root"]], ".rs")]
    pyo3_exports = _rust_pyo3_exports(repo_root / paths["rust_wrappers"])

    validation_scripts = [
        _relative(repo_root, path)
        for path in _iter_existing_files(repo_root, [paths["validation_root"]], ".py")
        if path.name != "__init__.py"
    ]
    test_files = [
        _relative(repo_root, path)
        for path in _iter_existing_files(repo_root, [paths["tests_root"]], ".py")
        if "__pycache__" not in path.parts
    ]
    workflows = [
        _relative(repo_root, path) for path in _iter_existing_files(repo_root, [paths["workflows_root"]], ".yml")
    ]
    docs = _docs_markdown(repo_root, config)
    project_metadata = _project_metadata(pyproject, config)

    manifest = {
        "$schema": config["schema_version"],
        "spdx_license_identifier": "AGPL-3.0-or-later",
        "copyright": {
            "concepts": "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
            "code": "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
            "orcid": "0009-0009-3560-0851",
            "contact": "www.anulum.li | protoscience@anulum.li",
        },
        "project": project_metadata,
        "python": {
            "source_roots": list(paths["source_roots"]),
            "source_modules": source_modules,
            "public_classes": public_classes,
            "public_api_exports": public_exports,
        },
        "rust": {
            "workspace_root": paths["rust_root"],
            "pyo3_wrapper": paths["rust_wrappers"],
            "source_files": rust_files,
            "pyo3_exports": pyo3_exports,
        },
        "validation": {"scripts": validation_scripts},
        "tests": {"python_files": test_files},
        "docs": {"public_markdown": docs},
        "ci": {"workflows": workflows},
        "counts": {
            "source_module_count": len(source_modules),
            "project_script_count": len(project_metadata["scripts"]),
            "public_class_count": len(public_classes),
            "public_api_export_count": len(public_exports),
            "rust_source_file_count": len(rust_files),
            "pyo3_export_count": len(pyo3_exports),
            "validation_script_count": len(validation_scripts),
            "python_test_file_count": len(test_files),
            "public_markdown_count": len(docs),
            "workflow_count": len(workflows),
        },
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Check selected inventory invariants without executing declared capabilities.

    Parameters
    ----------
    manifest
        Nested mapping with project/python/rust/validation/tests/docs/ci
        sections and counts as produced by build_manifest.

    Raises
    ------
    ManifestError
        A declared count is not a nonnegative nonboolean integer equal to its
        list/mapping length; required public/PyO3 exports or project scripts
        are empty; script targets lack dotted-identifier module:callable shape;
        public Markdown contains an internal path component.
    KeyError, TypeError
        The supplied mapping omits required sections or has unsupported shapes.

    Notes
    -----
    This is selected consistency validation, not a complete JSON schema,
    importability check, compiler run or empirical capability certification.
    """
    checks = {
        "source_module_count": len(manifest["python"]["source_modules"]),
        "project_script_count": len(manifest["project"]["scripts"]),
        "public_class_count": len(manifest["python"]["public_classes"]),
        "public_api_export_count": len(manifest["python"]["public_api_exports"]),
        "rust_source_file_count": len(manifest["rust"]["source_files"]),
        "pyo3_export_count": len(manifest["rust"]["pyo3_exports"]),
        "validation_script_count": len(manifest["validation"]["scripts"]),
        "python_test_file_count": len(manifest["tests"]["python_files"]),
        "public_markdown_count": len(manifest["docs"]["public_markdown"]),
        "workflow_count": len(manifest["ci"]["workflows"]),
    }
    for count_name, expected in checks.items():
        actual = manifest["counts"].get(count_name)
        if type(actual) is not int or actual < 0 or actual != expected:
            raise ManifestError(f"{count_name} drift: expected {expected}, found {actual}")
    if not manifest["python"]["public_api_exports"]:
        raise ManifestError("public_api_exports must not be empty")
    scripts = manifest["project"]["scripts"]
    if not scripts:
        raise ManifestError("project scripts must not be empty")
    invalid_scripts = [
        name
        for name, target in scripts.items()
        if not isinstance(name, str)
        or not name.strip()
        or not isinstance(target, str)
        or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", target)
    ]
    if invalid_scripts:
        raise ManifestError(f"project scripts must map to import targets: {', '.join(sorted(invalid_scripts))}")
    if not manifest["rust"]["pyo3_exports"]:
        raise ManifestError("pyo3_exports must not be empty")
    if any("/internal/" in f"/{path}/" for path in manifest["docs"]["public_markdown"]):
        raise ManifestError("public markdown inventory must not include docs/internal")

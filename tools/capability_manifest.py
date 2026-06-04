# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Capability manifest generator.

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI.
    import tomli as tomllib


README_START = "<!-- capability-snapshot:start -->"
README_END = "<!-- capability-snapshot:end -->"
CONFIG_PATH = Path("tools/capability_manifest.toml")


class ManifestError(RuntimeError):
    """Raised when the capability manifest is internally inconsistent."""


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _load_config(repo_root: Path) -> dict[str, Any]:
    return _read_toml(repo_root / CONFIG_PATH)


def _load_pyproject(repo_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    return _read_toml(repo_root / config["paths"]["pyproject"])


def _iter_existing_files(repo_root: Path, roots: list[str], suffix: str) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        path = repo_root / root
        if not path.exists():
            continue
        files.extend(file for file in path.rglob(f"*{suffix}") if file.is_file())
    return sorted(files)


def _python_classes(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    return sorted(
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
    )


def _public_exports(package_init: Path) -> list[str]:
    tree = ast.parse(package_init.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                value = ast.literal_eval(node.value)
                return sorted(str(item) for item in value)
    return []


def _rust_pyo3_exports(path: Path) -> list[str]:
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
        if actual != expected:
            raise ManifestError(f"{count_name} drift: expected {expected}, found {actual}")
    if not manifest["python"]["public_api_exports"]:
        raise ManifestError("public_api_exports must not be empty")
    scripts = manifest["project"]["scripts"]
    if not scripts:
        raise ManifestError("project scripts must not be empty")
    invalid_scripts = [
        name
        for name, target in scripts.items()
        if not isinstance(name, str) or not name.strip() or not isinstance(target, str) or ":" not in target
    ]
    if invalid_scripts:
        raise ManifestError(f"project scripts must map to import targets: {', '.join(sorted(invalid_scripts))}")
    if not manifest["rust"]["pyo3_exports"]:
        raise ManifestError("pyo3_exports must not be empty")
    if any("/internal/" in f"/{path}/" for path in manifest["docs"]["public_markdown"]):
        raise ManifestError("public markdown inventory must not include docs/internal")


def render_json(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def render_markdown(manifest: dict[str, Any]) -> str:
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
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Commercial license available -->",
        "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
        "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
        "<!-- ORCID: 0009-0009-3560-0851 -->",
        "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
        "<!-- SCPN Control — Generated capability snapshot. -->",
        "",
        "**Capability Inventory**",
        "",
        "| Surface | Count |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {label} | {value} |" for label, value in rows)
    lines.extend(
        [
            "",
            "**Evidence roots:** `src/scpn_control/{core,control,phase,scpn}`, `scpn-control-rs/crates`, "
            "`validation`, `tests`, `docs`, and `.github/workflows`.",
            "",
            "Refresh with `python tools/capability_manifest.py`; enforce with "
            "`python tools/capability_manifest.py --check`.",
            "",
        ]
    )
    return "\n".join(lines)


def extract_readme_block(readme: str) -> str:
    start = readme.index(README_START) + len(README_START)
    end = readme.index(README_END)
    block = readme[start:end]
    if block.startswith("\n"):
        block = block[1:]
    return block


def _replace_readme_block(readme: str, markdown: str) -> str:
    if README_START not in readme or README_END not in readme:
        raise ManifestError("README.md is missing capability snapshot markers")
    start = readme.index(README_START)
    end = readme.index(README_END) + len(README_END)
    return f"{readme[:start]}{README_START}\n{markdown}{README_END}{readme[end:]}"


def write_outputs(repo_root: Path | str | None = None) -> None:
    repo_root = Path.cwd() if repo_root is None else Path(repo_root)
    repo_root = repo_root.resolve()
    config = _load_config(repo_root)
    manifest = build_manifest(repo_root)
    markdown = render_markdown(manifest)
    json_path = repo_root / config["paths"]["json_output"]
    markdown_path = repo_root / config["paths"]["markdown_output"]
    readme_path = repo_root / config["paths"]["readme"]

    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(render_json(manifest), encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    readme_path.write_text(_replace_readme_block(readme_path.read_text(encoding="utf-8"), markdown), encoding="utf-8")


def check_outputs(repo_root: Path | str | None = None) -> list[str]:
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
        failures.append("README.md is missing capability snapshot markers")
    else:
        if embedded != markdown:
            failures.append("README.md capability snapshot is stale")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or check the SCPN-CONTROL capability manifest.")
    parser.add_argument("--check", action="store_true", help="fail if generated manifest surfaces are stale")
    args = parser.parse_args(argv)

    if args.check:
        failures = check_outputs(Path.cwd())
        if failures:
            for failure in failures:
                print(failure, file=sys.stderr)
            return 1
        return 0
    write_outputs(Path.cwd())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

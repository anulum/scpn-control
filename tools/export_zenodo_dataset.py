#!/usr/bin/env python3
"""Bundle scpn-control source + validation data into a Zenodo-ready zip.

Usage:
    python tools/export_zenodo_dataset.py [--output scpn-control-0.3.0.zip]

Upload the resulting zip manually at https://zenodo.org/deposit/new
with metadata from .zenodo.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent

INCLUDE_PATTERNS = [
    "src/**/*.py",
    "tests/**/*.py",
    "examples/**/*.ipynb",
    "examples/**/*.py",
    "validation/**/*.json",
    "validation/**/*.csv",
    "validation/**/*.py",
    "docs/**/*.md",
    "scpn-control-rs/Cargo.toml",
    "scpn-control-rs/Cargo.lock",
    "scpn-control-rs/crates/**/*.rs",
    "scpn-control-rs/crates/**/*.toml",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "LICENSE-MIT",
    "LICENSE-APACHE",
    "CITATION.cff",
    "CHANGELOG.md",
    ".zenodo.json",
    "mkdocs.yml",
]

EXCLUDE_DIRS = {".venv", "__pycache__", ".mypy_cache", ".ruff_cache", "target", "node_modules", "site", "dist"}


def collect_files() -> list[pathlib.Path]:
    files: set[pathlib.Path] = set()
    for pattern in INCLUDE_PATTERNS:
        files.update(ROOT.glob(pattern))
    return sorted(f for f in files if not any(part in EXCLUDE_DIRS for part in f.parts))


def main():
    meta = json.loads((ROOT / ".zenodo.json").read_text())
    version = meta["version"]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=f"scpn-control-{version}.zip")
    args = parser.parse_args()

    files = collect_files()
    prefix = f"scpn-control-{version}"

    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            arcname = f"{prefix}/{f.relative_to(ROOT)}"
            zf.write(f, arcname)

    print(f"Created {args.output} ({len(files)} files)")


if __name__ == "__main__":
    main()

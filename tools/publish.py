#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — PyPI Publish Script
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Build, verify, and publish scpn-control to PyPI or TestPyPI.

Usage::

    # Dry-run: build + twine check only
    python tools/publish.py --dry-run

    # Publish to TestPyPI
    python tools/publish.py --target testpypi

    # Publish to PyPI (requires --confirm)
    python tools/publish.py --target pypi --confirm

    # Bump version first, then publish
    python tools/publish.py --bump minor --target testpypi

Prerequisites::

    pip install build twine

For CI-based publishing, use the GitHub Actions workflow instead::

    .github/workflows/publish-pypi.yml
    # Triggered by git tag: git tag v0.2.0 && git push --tags
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
DIST = ROOT / "dist"


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=ROOT, check=check)


def read_version() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise SystemExit("Cannot parse version from pyproject.toml")
    return match.group(1)


def bump_version(part: str) -> str:
    """Bump major/minor/patch in pyproject.toml, return new version."""
    old = read_version()
    parts = old.split(".")
    if len(parts) != 3:
        raise SystemExit(f"Version {old!r} is not semver (major.minor.patch)")

    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    if part == "major":
        major, minor, patch = major + 1, 0, 0
    elif part == "minor":
        major, minor, patch = major, minor + 1, 0
    elif part == "patch":
        major, minor, patch = major, minor, patch + 1
    else:
        raise SystemExit(f"Invalid bump part: {part!r} (use major/minor/patch)")

    new = f"{major}.{minor}.{patch}"
    text = PYPROJECT.read_text(encoding="utf-8")
    text = text.replace(f'version = "{old}"', f'version = "{new}"', 1)
    PYPROJECT.write_text(text, encoding="utf-8")
    print(f"  Version bumped: {old} -> {new}")
    return new


def clean_dist():
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir()


def build():
    _run([sys.executable, "-m", "build"])


def check():
    _run([sys.executable, "-m", "twine", "check", "dist/*"])


def upload(target: str):
    cmd = [sys.executable, "-m", "twine", "upload"]
    if target == "testpypi":
        cmd += ["--repository", "testpypi"]
    cmd.append("dist/*")
    _run(cmd)


def run_tests():
    env_flag = "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
    _run([
        sys.executable, "-m", "pytest",
        "-p", "hypothesis.extra.pytestplugin",
        "tests/", "-x", "-q", "--tb=short",
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Build and publish scpn-control to PyPI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--target", choices=["pypi", "testpypi"], default="testpypi",
        help="Upload target (default: testpypi)",
    )
    parser.add_argument(
        "--bump", choices=["major", "minor", "patch"],
        help="Bump version before publishing",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build and check only, do not upload",
    )
    parser.add_argument(
        "--skip-tests", action="store_true",
        help="Skip pytest before building",
    )
    parser.add_argument(
        "--confirm", action="store_true",
        help="Required for PyPI uploads (safety gate)",
    )
    args = parser.parse_args()

    if args.target == "pypi" and not args.dry_run and not args.confirm:
        raise SystemExit(
            "PyPI upload requires --confirm flag. "
            "Use --dry-run to preview, or --target testpypi for testing."
        )

    print("=" * 60)
    print("  scpn-control publish pipeline")
    print("=" * 60)

    # Step 1: version bump
    if args.bump:
        print(f"\n[1/5] Bumping version ({args.bump})...")
        version = bump_version(args.bump)
    else:
        version = read_version()
        print(f"\n[1/5] Current version: {version}")

    # Step 2: tests
    if not args.skip_tests:
        print("\n[2/5] Running tests...")
        run_tests()
    else:
        print("\n[2/5] Skipping tests (--skip-tests)")

    # Step 3: clean + build
    print("\n[3/5] Building sdist + wheel...")
    clean_dist()
    build()

    # Step 4: twine check
    print("\n[4/5] Checking package metadata...")
    check()

    # Step 5: upload
    if args.dry_run:
        print(f"\n[5/5] Dry run — skipping upload to {args.target}")
        print(f"  Artifacts in: {DIST}")
        for f in sorted(DIST.glob("*")):
            print(f"    {f.name}  ({f.stat().st_size / 1024:.0f} KB)")
    else:
        print(f"\n[5/5] Uploading to {args.target}...")
        upload(args.target)

    print("\n" + "=" * 60)
    print(f"  Done. Version {version} {'built' if args.dry_run else 'published'}.")
    if not args.dry_run and args.target == "testpypi":
        print(f"  Install: pip install -i https://test.pypi.org/simple/ scpn-control=={version}")
    elif not args.dry_run:
        print(f"  Install: pip install scpn-control=={version}")
    print("=" * 60)


if __name__ == "__main__":
    main()

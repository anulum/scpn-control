#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Publish.

"""
Build, verify, and publish scpn-control to PyPI or TestPyPI.

Usage::

    # Dry-run: tests + build + twine check; no upload
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
import shutil
import subprocess  # nosec B404
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
DIST = ROOT / "dist"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.publish_metadata import PublishMetadataError, bump_project_version, read_project_version


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[bytes]:
    """Run the exact argument vector from ROOT, inheriting output and process status."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=ROOT, check=check)  # nosec B603


def read_version() -> str:
    """Read the actual project.version string from complete TOML metadata.

    Returns
    -------
    str
        Nonempty version string in PYPROJECT; other version fields are ignored.

    Raises
    ------
    SystemExit
        Authored refusal for unreadable, malformed or missing project metadata.
        Reading does not require the three-component bump grammar.
    """
    try:
        return read_project_version(PYPROJECT)
    except PublishMetadataError as exc:
        raise SystemExit(str(exc)) from exc


def bump_version(part: str) -> str:
    """Atomically update only the project.version token in PYPROJECT.

    Parameters
    ----------
    part : str
        Exactly major, minor or patch; the original version must be plain
        major.minor.patch without leading zeroes or a suffix.

    Returns
    -------
    str
        Newly written version, printed only after successful publication.

    Raises
    ------
    SystemExit
        Authored metadata, semantic-version, changed-input or publication refusal.

    Notes
    -----
    Other metadata tokens/comments and other release files are untouched.
    Operators must reconcile those files separately before a release. Writers
    coordinate concurrent changes; per-file atomicity is not a crash transaction.
    """
    try:
        old, new = bump_project_version(
            PYPROJECT, part, protected_files=(Path(__file__), Path(__file__).with_name("publish_metadata.py"))
        )
    except PublishMetadataError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"  Version bumped: {old} -> {new}")
    return new


def clean_dist() -> None:
    """Replace DIST while refusing source namespaces, symlinks and non-directories.

    Raises
    ------
    SystemExit
        Selected cleanup aliases repository source or is not a real directory.
    OSError
        Native deletion or creation fails. No transactional rollback is claimed.

    Notes
    -----
    ROOT and DIST are configured module paths. Callers own the selected
    distribution data and coordinate other writers before destructive cleanup.
    """
    if DIST.is_symlink() or (DIST.exists() and not DIST.is_dir()):
        raise SystemExit("Distribution cleanup requires a real directory")
    selected = DIST.resolve()
    protected = (ROOT, PYPROJECT, ROOT / "src", ROOT / "tools", ROOT / "tests", ROOT / "docs", ROOT / "scpn-control-rs")
    if any(path.resolve().is_relative_to(selected) for path in protected):
        raise SystemExit("Distribution cleanup must not remove repository sources")
    if any(selected.is_relative_to(path.resolve()) for path in protected[2:]):
        raise SystemExit("Distribution cleanup must not remove repository sources")
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir()


def build() -> None:
    """Invoke the existing checkout release builder with the current interpreter.

    Raises
    ------
    OSError, subprocess.SubprocessError
        Launch or build failure. The selected backend executes project code;
        this function does not grant publication or scientific admission.
    """
    _run([sys.executable, "tools/build_release_artifacts.py"])


def check() -> None:
    """Run Twine metadata checks on the current exact distribution path set.

    Raises
    ------
    SystemExit
        The selected distribution directory or artifact set is inadmissible.
    OSError, subprocess.SubprocessError
        Twine launch or validation fails; no upload occurs in this function.
    """
    _run([sys.executable, "-m", "twine", "check", *distribution_paths()])


def distribution_paths() -> list[str]:
    """Select sorted nonempty regular wheel/sdist paths without shell globbing.

    Returns
    -------
    list of str
        Exact native paths ending in .whl or .tar.gz, including spaces as a
        single argument. Selection does not authenticate or inspect archives.

    Raises
    ------
    SystemExit
        Missing/symlink directory, no artifacts, or a matching empty, linked
        or nonregular node. Twine owns later archive/metadata validation.
    OSError
        Native iteration or size inspection fails.
    """
    if DIST.is_symlink() or not DIST.is_dir():
        raise SystemExit("Distribution artifacts require a real directory")
    candidates = [path for path in sorted(DIST.iterdir()) if path.suffix == ".whl" or path.name.endswith(".tar.gz")]
    if any(path.is_symlink() or not path.is_file() or path.stat().st_size == 0 for path in candidates):
        raise SystemExit("Distribution artifacts must be nonempty regular files")
    artifacts = [str(path) for path in candidates]
    if not artifacts:
        raise SystemExit("No distribution artifacts found")
    return artifacts


def upload(target: str) -> None:
    """Invoke Twine upload for one explicitly selected supported index.

    Parameters
    ----------
    target : str
        Exactly pypi or testpypi. An unknown target refuses before file/process
        access. This direct operator API does not implement CLI --confirm.

    Raises
    ------
    SystemExit
        Unknown index or inadmissible distribution paths.
    OSError, subprocess.SubprocessError
        Actual upload launch or completion fails. Invoking this API requires
        separate operator publication authority and configured credentials.
    """
    if target not in ("pypi", "testpypi"):
        raise SystemExit("Invalid upload target (use pypi/testpypi)")
    cmd = [sys.executable, "-m", "twine", "upload"]
    if target == "testpypi":
        cmd += ["--repository", "testpypi"]
    cmd.extend(distribution_paths())
    _run(cmd)


def run_tests() -> None:
    """Invoke the existing full publish-time pytest gate from ROOT.

    Raises
    ------
    OSError, subprocess.SubprocessError
        Test process launch or completion fails. A nonzero test status stops
        the caller's release workflow before building or uploading.
    """
    _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "hypothesis.extra.pytestplugin",
            "tests/",
            "-x",
            "-q",
            "--tb=short",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    """Execute the existing local release workflow through its ordinary CLI.

    Parameters
    ----------
    argv : list of str or None, optional
        Tokens excluding the executable; None reads sys.argv. TestPyPI remains
        the default. PyPI upload requires --confirm unless --dry-run is used.

    Raises
    ------
    SystemExit
        Argument, confirmation or authored metadata refusal; native workflow
        faults use a fixed refusal sentence. Successful execution returns None.

    Notes
    -----
    Dry-run omits upload, but still tests/builds/checks, replaces DIST and may
    change PYPROJECT with --bump. --skip-tests omits only pytest. A metadata
    bump is not rolled back after a later workflow failure. Actual publication
    requires separate operator authority; no remote outcome is inferred here.
    """
    parser = argparse.ArgumentParser(
        description="Build and publish scpn-control to PyPI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--target",
        choices=["pypi", "testpypi"],
        default="testpypi",
        help="Upload target (default: testpypi)",
    )
    parser.add_argument(
        "--bump",
        choices=["major", "minor", "patch"],
        help="Bump version before publishing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and check only, do not upload",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest before building",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required for PyPI uploads (safety gate)",
    )
    args = parser.parse_args(argv)

    if args.target == "pypi" and not args.dry_run and not args.confirm:
        raise SystemExit(
            "PyPI upload requires --confirm flag. Use --dry-run to preview, or --target testpypi for testing."
        )

    try:
        _pipeline(args)
    except (OSError, UnicodeError, ValueError, subprocess.SubprocessError) as exc:
        raise SystemExit("Publish pipeline refused: a local workflow operation failed") from exc


def _pipeline(args: argparse.Namespace) -> None:
    """Run parsed operator actions in order; upload is omitted for dry-run."""
    print("=" * 60)
    print("  scpn-control publish pipeline")
    print("=" * 60)

    if args.bump:
        print(f"\n[1/5] Bumping version ({args.bump})...")
        version = bump_version(args.bump)
    else:
        version = read_version()
        print(f"\n[1/5] Current version: {version}")

    if not args.skip_tests:
        print("\n[2/5] Running tests...")
        run_tests()
    else:
        print("\n[2/5] Skipping tests (--skip-tests)")

    print("\n[3/5] Building sdist + wheel...")
    clean_dist()
    build()

    print("\n[4/5] Checking package metadata...")
    check()

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

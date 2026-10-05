# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Universal Python dependency advisory inventory.

"""Export every locked registry package for a universal ``pip-audit`` scan.

The advisory inventory retains every name/version pair, including optional
dependencies and requirements guarded by another Python version or platform.
It does not resolve, install, update, or assert artifact integrity. The local
project has no upstream advisory identity and is excluded explicitly. Pinned
source archives require checksum-bound advisory metadata; unknown non-registry
dependencies are refused rather than silently skipped.
"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from collections.abc import Sequence
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def locked_packages(root: Path) -> list[tuple[str, str]]:
    """Collect all upstream package identities from the committed Python locks.

    Parameters
    ----------
    root : pathlib.Path
        Repository containing ``pyproject.toml``, ``uv.lock`` and the hash-pinned
        ``requirements/ci-*.txt`` files and ``tools/dependency_advisory_sources.json``.
        Environment markers are not evaluated.

    Returns
    -------
    list of tuple of str
        Sorted, unique canonical names and versions. Different locked versions
        of the same package remain distinct advisory targets.

    Raises
    ------
    ValueError
        An upstream dependency lacks an exact registry version, a requirements
        declaration is unsupported, or no CI requirements locks exist.
    OSError, KeyError, TypeError
        A source file is missing, unreadable, or lacks the expected structure.
    """
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project_name = canonicalize_name(project["project"]["name"])
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    source_metadata = json.loads((root / "tools/dependency_advisory_sources.json").read_text(encoding="utf-8"))
    packages: set[tuple[str, str]] = set()
    for package in lock["package"]:
        name = canonicalize_name(package["name"])
        source = package["source"]
        if source == {"editable": "."} and name == project_name:
            continue
        if set(source) != {"registry"} or not source["registry"]:
            raise ValueError(f"uv.lock: {name} is not an upstream registry dependency")
        packages.add((name, str(Version(package["version"]))))

    requirements = sorted((root / "requirements").glob("ci-*.txt"))
    if not requirements:
        raise ValueError("no requirements/ci-*.txt locks found")
    for path in requirements:
        logical_lines = path.read_text(encoding="utf-8").replace("\\\n", "").splitlines()
        for line in logical_lines:
            declaration = line.strip()
            if not declaration or declaration.startswith("#"):
                continue
            if declaration.startswith(("--index-url https://", "--extra-index-url https://")):
                continue
            requirement = Requirement(re.sub(r"\s+--hash=\S+", "", declaration))
            if requirement.url:
                metadata = source_metadata.get(requirement.url)
                hashes = re.findall(r"\s+--hash=sha256:([a-f0-9]{64})(?:\s|$)", declaration)
                if metadata is None:
                    raise ValueError(f"{path.name}: {requirement.name} direct source has no verified advisory metadata")
                if (
                    canonicalize_name(metadata["name"]) != canonicalize_name(requirement.name)
                    or metadata["sha256"] not in hashes
                ):
                    raise ValueError(f"{path.name}: {requirement.name} direct source advisory binding differs")
                packages.add((canonicalize_name(requirement.name), str(Version(metadata["version"]))))
                continue
            specifiers = list(requirement.specifier)
            if len(specifiers) != 1 or specifiers[0].operator != "==":
                raise ValueError(f"{path.name}: {requirement.name} must have one exact version")
            packages.add((canonicalize_name(requirement.name), str(Version(specifiers[0].version))))
    if not packages:
        raise ValueError("locked registry package inventory is empty")
    return sorted(packages)


def write_audit_lock(root: Path, output: Path) -> int:
    """Write the universal advisory inventory in ``pip-audit`` pylock format.

    Parameters
    ----------
    root : pathlib.Path
        Repository passed to :func:`locked_packages` without modifying its locks.
    output : pathlib.Path
        New output file in an existing directory. Existing files are refused.

    Returns
    -------
    int
        Number of distinct upstream package name/version pairs written.

    Raises
    ------
    OSError, ValueError, KeyError, TypeError
        Source collection fails or the output cannot be created exclusively.

    Notes
    -----
    This is an advisory inventory consumed by ``pip-audit --locked``, not an
    installation lock: it contains no artifact URLs, hashes or dependency graph.
    Local build suffixes such as PyTorch's ``+cpu`` use the upstream public
    version for advisory lookup; the complete locked version remains in each
    package's tool metadata. The scan does not certify vendor build changes.
    """
    packages = locked_packages(root)
    lines = ['lock-version = "1.0"', 'created-by = "scpn-control dependency advisory inventory"']
    for name, version in packages:
        lines.extend(
            [
                "",
                "[[packages]]",
                f"name = {json.dumps(name)}",
                f"version = {json.dumps(Version(version).public)}",
                f"tool.scpn-control.locked-version = {json.dumps(version)}",
            ]
        )
    with output.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return len(packages)


def main(argv: Sequence[str] | None = None) -> int:
    """Export the repository inventory through the source-tree CLI.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments; ``None`` reads the actual process command line. ``--output``
        is required; ``--repo-root`` defaults to the script's repository.

    Returns
    -------
    int
        Zero after writing a nonempty inventory. Argparse raises exit status two
        for usage, source-contract and output failures; help exits with zero.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        count = write_audit_lock(arguments.repo_root, arguments.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Exported {count} locked upstream package versions to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

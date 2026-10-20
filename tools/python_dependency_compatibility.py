# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Fail-closed locked dependency Python compatibility.
"""Check declared Python constraints for exact pins and explicit target versions.

This does not resolve or install requirements, validate hashes or certify a
package's behaviour. Unsupported lock syntax or unavailable metadata refuses
the entire measurement; neither can be counted as compatible.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.python_dependency_metadata import PYPI_METADATA_URL, DependencyMetadataError, read_python_constraint

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CI_PYTHON_VERSIONS = ("3.11", "3.12", "3.13")


@dataclass(frozen=True)
class _Pin:
    """Retain a normalised exact release and its optional Python-only marker."""

    name: str
    version: str
    marker: Marker | None


def _pins(path: Path) -> list[_Pin]:
    """Read a nonempty UTF-8 lock without ignoring unsupported declarations."""
    try:
        text = path.read_text(encoding="utf-8").replace("\\\n", "")
    except (OSError, UnicodeError):
        raise DependencyMetadataError("Could not read a required compatibility lock.") from None
    pins: list[_Pin] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        declaration = re.sub(r"\s+--hash=sha256:[a-f0-9]{64}(?=\s|$)", "", line)
        try:
            requirement = Requirement(declaration)
            specs = list(requirement.specifier)
            if requirement.url is not None or len(specs) != 1 or specs[0].operator != "==":
                raise ValueError
            version = str(Version(specs[0].version))
        except ValueError:
            raise DependencyMetadataError("Compatibility locks require exact registry name==version pins.") from None
        marker = requirement.marker
        if marker is not None:
            unquoted = re.sub(r"'[^']*'|\"[^\"]*\"", "", str(marker))
            tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*", unquoted))
            if tokens - {"python_version", "python_full_version", "and", "or", "not", "in"}:
                raise DependencyMetadataError("Non-Python markers require a separately qualified platform profile.")
        pins.append(_Pin(canonicalize_name(requirement.name), version, marker))
    if not pins:
        raise DependencyMetadataError("A compatibility lock must contain at least one exact pin.")
    return pins


def python_compatibility_errors(
    requirements: Sequence[Path],
    python_versions: Sequence[str] = CI_PYTHON_VERSIONS,
    *,
    metadata_base_url: str = PYPI_METADATA_URL,
    timeout_s: float = 10.0,
) -> list[str]:
    """Compare exact lock releases against every applicable Python target.

    Parameters
    ----------
    requirements : sequence of pathlib.Path
        Nonempty list of nonempty hash-lock or exact-pin files. Paths are read
        from the caller's working directory. Includes, URLs, ranges, options
        and non-Python environment markers refuse this Python-only profile.
    python_versions : sequence of str
        Nonempty unique Python versions, including prerelease/dev suffixes.
        Epochs, post/local versions and releases with more than three numeric
        components refuse. A minor spelling such as ``3.11`` means its zero
        patch floor, not the latest runner patch.
    metadata_base_url : str
        Explicit release JSON origin; default PyPI. Custom origins are input
        declarations, not independently authenticated upstream evidence.
    timeout_s : float
        Finite positive socket timeout passed to the bounded metadata reader.

    Returns
    -------
    list of str
        Incompatible release/target diagnostics, ordered by release and target.
        Empty means all selected declared constraints admit applicable targets.
        Releases excluded by a supported Python marker are not fetched. An
        entirely excluded release inventory refuses rather than passing.

    Raises
    ------
    DependencyMetadataError
        A target, lock, request, identity or constraint is unavailable or
        invalid. No partial success or automatic baseline update is returned.

    Notes
    -----
    This is metadata compatibility only. Hash-verified installations on each
    supported OS/interpreter remain separate gates. Explicit null or empty
    ``requires_python`` declares no restriction; it does not prove execution.
    """
    if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise DependencyMetadataError("The metadata timeout must be finite and positive.")
    try:
        targets = []
        for value in python_versions:
            target = Version(value)
            if target.epoch or target.post is not None or target.local is not None or len(target.release) > 3:
                raise ValueError
            suffix = str(target)[len(target.base_version) :]
            targets.append(Version(f"{target.major}.{target.minor}.{target.micro}{suffix}"))
    except ValueError:
        raise DependencyMetadataError("Python targets must be valid versions.") from None
    if not requirements or not targets or len(set(targets)) != len(targets):
        raise DependencyMetadataError("Compatibility needs locks and unique Python targets.")
    pins = [pin for path in requirements for pin in _pins(path)]
    releases: dict[tuple[str, str], set[Version]] = {}
    for pin in pins:
        for target in targets:
            environment = {
                "python_version": f"{target.major}.{target.minor}",
                "python_full_version": str(target),
            }
            if pin.marker is None or pin.marker.evaluate(environment):
                releases.setdefault((pin.name, pin.version), set()).add(target)
    if not releases:
        raise DependencyMetadataError("No locked release applies to the requested Python targets.")
    errors: list[str] = []
    for (name, version), applicable in sorted(releases.items()):
        constraint = read_python_constraint(name, version, base_url=metadata_base_url, timeout_s=timeout_s)
        try:
            specifier = SpecifierSet(constraint or "")
        except ValueError:
            raise DependencyMetadataError("Release metadata declares an invalid Python constraint.") from None
        for target in sorted(applicable):
            if not specifier.contains(target, prereleases=True):
                errors.append(f"{name}=={version} requires Python {constraint}; target {target} is incompatible")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Run only the compatibility gate through its standalone operator CLI.

    Parameters
    ----------
    argv : sequence of str or None
        Argparse arguments. Repeat ``--requirements`` and ``--python-version``
        for exact inputs; defaults are the two common CI locks and three
        declared Python minor floors. No other preflight gates are launched.

    Returns
    -------
    int
        Zero for admitted declarations, one for incompatibility, two for a
        refused measurement. Argparse retains zero/two help/usage exits.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, action="append")
    parser.add_argument("--python-version", action="append")
    parser.add_argument("--metadata-base-url", default=PYPI_METADATA_URL)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    args = parser.parse_args(argv)
    requirements = args.requirements or [
        REPOSITORY_ROOT / f"requirements/{name}.txt" for name in ("ci-deps", "ci-test")
    ]
    try:
        errors = python_compatibility_errors(
            requirements,
            args.python_version or CI_PYTHON_VERSIONS,
            metadata_base_url=args.metadata_base_url,
            timeout_s=args.timeout_s,
        )
    except DependencyMetadataError as error:
        print(f"Python compatibility inspection refused: {error}")
        return 2
    if errors:
        print("Python compatibility rejected:\n" + "\n".join(errors))
        return 1
    print("Python compatibility declarations admit every applicable target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

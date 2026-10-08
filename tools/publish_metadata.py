# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Project version parsing and checked TOML mutation.
"""Read project.version and replace only its original TOML string token.

Complete TOML parsing owns field identity. Candidate token replacements must
preserve the parsed document except for project.version, including unrelated
tables and versions. Publication is atomic per file with handled-failure
recovery; callers still coordinate concurrent writers.
"""

from __future__ import annotations

import math
import re
import tomllib
from collections.abc import Iterator
from pathlib import Path

from tools.inventory_file_output import publish_guarded_outputs


class PublishMetadataError(ValueError):
    """Report an authored metadata refusal without exposing native error text."""


def _load(path: Path) -> tuple[bytes, str, dict[str, object], dict[str, object], str]:
    """Read strict UTF-8 TOML and identify the actual nonempty project version."""
    try:
        original = path.read_bytes()
        text = original.decode("utf-8")
        parsed: dict[str, object] = tomllib.loads(text)
    except (OSError, UnicodeError, ValueError) as exc:
        raise PublishMetadataError("Cannot parse version from pyproject.toml") from exc
    project = parsed.get("project")
    if not isinstance(project, dict):
        raise PublishMetadataError("Cannot parse version from pyproject.toml")
    version = project.get("version")
    if not isinstance(version, str) or not version.strip():
        raise PublishMetadataError("Cannot parse version from pyproject.toml")
    return original, text, parsed, project, version


def _string_tokens(text: str) -> Iterator[tuple[int, int]]:
    """Locate quoted string tokens in already validated TOML, omitting comments."""
    position = 0
    while position < len(text):
        character = text[position]
        if character == "#":
            end = text.find("\n", position)
            position = len(text) if end < 0 else end + 1
        elif character in ("'", '"'):
            start = position
            multiline = text.startswith(character * 3, position)
            position += 3 if multiline else 1
            while True:
                if character == '"' and text[position] == "\\":
                    position += 2
                elif text[position] == character:
                    end = position + 1
                    if multiline:
                        while end < len(text) and text[end] == character:
                            end += 1
                        if end - position < 3:
                            position = end
                            continue
                    position = end
                    yield start, end
                    break
                else:
                    position += 1
        else:
            position += 1


def _comparable(value: object) -> object:
    """Normalise parsed NaN values to a tuple unavailable in TOML's value grammar."""
    if isinstance(value, dict):
        return {key: _comparable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_comparable(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return ("toml.nan",)
    return value


def read_project_version(path: Path) -> str:
    """Read the nonempty string declared by the complete TOML project table.

    Parameters
    ----------
    path : pathlib.Path
        UTF-8 project metadata file. Relative paths follow the caller's cwd.

    Returns
    -------
    str
        The exact project.version string; other version fields are ignored.
        Reading does not require the three-component bump grammar.

    Raises
    ------
    PublishMetadataError
        Unreadable or malformed metadata, including duplicate TOML fields,
        or an absent, non-string or blank project.version. Nothing is written.
    """
    return _load(path)[4]


def bump_project_version(path: Path, part: str, *, protected_files: tuple[Path, ...] = ()) -> tuple[str, str]:
    """Replace project.version while preserving every other TOML value and token.

    Parameters
    ----------
    path : pathlib.Path
        Existing UTF-8 metadata file. Symlink and nonregular output refusal
        follows the guarded publisher. Other release carriers are untouched.
    part : str
        Exactly major, minor or patch. The original version must contain three
        nonnegative decimal components without leading zeroes or a suffix.
    protected_files : tuple of pathlib.Path, optional
        Source files whose path or inode must not alias this metadata output.

    Returns
    -------
    tuple of str
        Original and updated versions after successful atomic publication.

    Raises
    ------
    PublishMetadataError
        Invalid metadata, bump part or semantic version, unavailable unique
        string-token edit, changed input or handled publication refusal.

    Notes
    -----
    Whitespace, comments and unrelated token bytes remain exact. Inline,
    dotted and quoted project declarations are resolved by reparsing rather
    than guessed table positions. This is not a crash transaction or a lock;
    callers coordinate concurrent writes. The version's decimal components
    remain subject to the interpreter's integer conversion limit.
    """
    if part not in ("major", "minor", "patch"):
        raise PublishMetadataError("Invalid bump part (use major/minor/patch)")
    original, text, parsed, project, old = _load(path)
    if re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", old) is None:
        raise PublishMetadataError("Version is not semver (major.minor.patch)")
    try:
        major, minor, patch = map(int, old.split("."))
        if part == "major":
            major, minor, patch = major + 1, 0, 0
        elif part == "minor":
            major, minor, patch = major, minor + 1, 0
        else:
            patch += 1
        new = f"{major}.{minor}.{patch}"
    except ValueError as exc:
        raise PublishMetadataError("Version components exceed the integer conversion limit") from exc
    expected = dict(parsed)
    expected["project"] = {**project, "version": new}
    candidates: list[str] = []
    for start, end in _string_tokens(text):
        token = text[start:end]
        if tomllib.loads("value=" + token)["value"] != old:
            continue
        changed = text[:start] + '"' + new + '"' + text[end:]
        try:
            observed = tomllib.loads(changed)
        except tomllib.TOMLDecodeError:
            continue
        if _comparable(expected) == _comparable(observed):
            candidates.append(changed)
    try:
        (replacement,) = candidates
        if path.read_bytes() != original:
            raise PublishMetadataError("Project metadata changed before version publication")
        publish_guarded_outputs(((path, replacement.encode("utf-8")),), protected_files=protected_files)
    except PublishMetadataError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise PublishMetadataError("Cannot publish the updated project version") from exc
    return old, new

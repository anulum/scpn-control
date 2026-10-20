# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Per-owner documentation debt ledger contracts.
"""Read non-increasing documentation bounds without inferring source validity.

The aggregate ceiling remains a separate upper bound. Owner bounds apply per
repository-relative file and Ruff rule, so one repair cannot pay for another
owner's regression. Profiles bind the Ruff distribution version and complete
root pyproject bytes; they do not authenticate a reviewer or measure runtime
behaviour. Source enrollment outside the configured Ruff scope is separate.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath

SCOPE = ("tests", "tools", "validation")
SOURCE_FINGERPRINT_PROTOCOL = "sha256-utf8-universal-newlines-v1"


@dataclass(frozen=True)
class OwnerDebtLedger:
    """Validated aggregate and per-file/rule bounds with a comparable profile.

    Attributes
    ----------
    total : int
        Historical aggregate ceiling; owner counts may sum to less than it.
    per_file : dict of str to dict of str to int
        Relative file/rule ceilings. Missing owners or rules have a zero bound.
    profile : dict of str to str
        Exact Ruff version and SHA-256 of the root pyproject configuration.
    source_fingerprints : dict of str to str
        Original UTF-8 owner text SHA-256 after universal newline normalisation.
    origin_revision : str
        Optional original baseline commit, not a current source attestation.

    Notes
    -----
    Only the loader validates values. Direct construction is unchecked, and
    frozen fields do not deeply freeze the supplied dictionaries.
    """

    total: int
    per_file: dict[str, dict[str, int]]
    profile: dict[str, str]
    source_fingerprints: dict[str, str]
    origin_revision: str = ""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Refuse duplicate ledger fields without displaying their input spelling."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate ledger key")
        result[key] = value
    return result


def _counts(value: object) -> dict[str, int]:
    """Validate nonnegative integer documentation-rule counts without coercion."""
    if not isinstance(value, dict):
        raise ValueError("rule counts must be an object")
    result: dict[str, int] = {}
    for code, count in value.items():
        if not re.fullmatch(r"D[0-9]{3}", code):
            raise ValueError("rule counts require D-rule keys")
        if type(count) is not int or count < 0:
            raise ValueError("rule counts must be nonnegative integers")
        result[code] = count
    return result


def owner_path(value: str) -> str:
    """Validate a canonical scope-relative Python owner without resolving files.

    Parameters
    ----------
    value : str
        Slash-separated Python file spelling beneath tests, tools or validation.

    Returns
    -------
    str
        Unchanged key, preserving Unicode and case.

    Raises
    ------
    ValueError
        The key is not canonical or is outside the declared lexical scope.

    Notes
    -----
    Absolute paths, aliases through dots, repeated separators, backslashes and
    control characters refuse. This checks a lexical key, not file existence,
    filesystem containment through symlinks, or source authenticity.
    """
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or ".." in path.parts
        or "\\" in value
        or any(ord(character) < 32 for character in value)
        or not path.parts
        or path.parts[0] not in SCOPE
        or path.suffix != ".py"
    ):
        raise ValueError("owner paths must be canonical scope-relative Python files")
    return value


def _owner_fields(payload: dict[str, object], total: int) -> OwnerDebtLedger:
    """Decode owner bounds and their exact measurement profile from one payload."""
    owners = payload.get("per_file")
    if not isinstance(owners, dict):
        raise ValueError("per_file owner bounds are required")
    per_file: dict[str, dict[str, int]] = {}
    for path, counts in owners.items():
        per_file[owner_path(path)] = _counts(counts)
    if sum(sum(counts.values()) for counts in per_file.values()) > total:
        raise ValueError("per_file counts must not exceed the aggregate ceiling")
    if payload.get("source_fingerprint_protocol") != SOURCE_FINGERPRINT_PROTOCOL:
        raise ValueError("source fingerprint protocol must match the supported comparison")
    fingerprints = payload.get("source_fingerprints")
    if not isinstance(fingerprints, dict) or set(fingerprints) != set(per_file):
        raise ValueError("source fingerprints must match every recorded owner")
    source_fingerprints: dict[str, str] = {}
    for path, digest in fingerprints.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("source fingerprints require SHA-256 digests")
        source_fingerprints[path] = digest
    profile = payload.get("profile")
    if not isinstance(profile, dict) or set(profile) != {"ruff_version", "configuration_sha256"}:
        raise ValueError("owner bounds require an exact Ruff/configuration profile")
    ruff_version, digest = profile["ruff_version"], profile["configuration_sha256"]
    if not isinstance(ruff_version, str) or not ruff_version.strip() or ruff_version != ruff_version.strip():
        raise ValueError("Ruff profile version must be nonempty text")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("configuration profile requires a SHA-256 digest")
    origin = payload.get("origin_revision", "")
    if not isinstance(origin, str) or (origin and not re.fullmatch(r"[0-9a-f]{40}", origin)):
        raise ValueError("origin revision must be an original commit SHA")
    return OwnerDebtLedger(
        total, per_file, {"ruff_version": ruff_version, "configuration_sha256": digest}, source_fingerprints, origin
    )


def _read_payload(path: Path) -> tuple[dict[str, object], int]:
    """Read unique UTF-8 JSON and validate the legacy aggregate count contract."""
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except FileNotFoundError:
        raise RuntimeError("missing docstring debt ledger") from None
    except (OSError, ValueError, RecursionError):
        raise RuntimeError("malformed docstring debt ledger: could not read unique UTF-8 JSON") from None
    try:
        if not isinstance(payload, dict):
            raise ValueError("ledger must be an object")
        total = payload.get("total")
        if type(total) is not int or total < 0:
            raise ValueError("total must be a nonnegative integer")
        if payload.get("scope") != list(SCOPE):
            raise ValueError("scope must match tests/tools/validation")
        if "by_code" in payload and sum(_counts(payload["by_code"]).values()) != total:
            raise ValueError("by_code sum must equal total")
        return payload, total
    except ValueError as error:
        raise RuntimeError(f"malformed docstring debt ledger: {error}") from None


def read_debt_ceiling(path: Path) -> int:
    """Validate aggregate JSON, including owner fields when present, and return its ceiling.

    Parameters
    ----------
    path : pathlib.Path
        UTF-8 JSON carrier; relative paths follow caller cwd and symlinks are followed.

    Returns
    -------
    int
        Recorded nonnegative aggregate bound, not the measured current debt.

    Raises
    ------
    RuntimeError
        File, JSON, aggregate or present owner/profile fields cannot be validated.

    Notes
    -----
    Legacy aggregate-only files remain readable through this API. They cannot
    pass the CLI's mandatory owner-ledger loader. No values are rewritten.
    """
    payload, total = _read_payload(path)
    if any(field in payload for field in ("per_file", "profile", "source_fingerprints", "source_fingerprint_protocol")):
        try:
            _owner_fields(payload, total)
        except ValueError as error:
            raise RuntimeError(f"malformed docstring debt ledger: {error}") from None
    return total


def read_owner_ledger(path: Path) -> OwnerDebtLedger:
    """Load mandatory owner/profile bounds; aggregate-only input cannot clear enforcement.

    Parameters
    ----------
    path : pathlib.Path
        Caller-selected UTF-8 JSON ledger; relative paths use caller cwd.

    Returns
    -------
    OwnerDebtLedger
        Validated counts/profile in newly decoded dictionaries. Owner counts
        may sum below the preserved aggregate ceiling.

    Raises
    ------
    RuntimeError
        A carrier, aggregate, owner key/count or profile is missing or invalid.

    Notes
    -----
    The optional origin revision is declaration metadata, not a fetched or
    authenticated Git object. Loading does not compare current source or tools.
    """
    payload, total = _read_payload(path)
    try:
        return _owner_fields(payload, total)
    except ValueError as error:
        raise RuntimeError(f"malformed docstring debt ledger: {error}") from None


def measurement_profile(repo: Path) -> dict[str, str]:
    """Bind the current interpreter's Ruff distribution and root configuration bytes.

    Parameters
    ----------
    repo : pathlib.Path
        Root containing pyproject.toml; relative paths follow caller cwd.

    Returns
    -------
    dict of str to str
        Ruff distribution version and SHA-256 of complete root configuration bytes.

    Raises
    ------
    RuntimeError
        The actual distribution metadata or configuration file is unavailable.

    Notes
    -----
    This does not inspect nested configuration discovery, tool binaries or
    external ignore files; source-enrollment qualification remains separate.
    """
    try:
        return {
            "ruff_version": version("ruff"),
            "configuration_sha256": hashlib.sha256((repo / "pyproject.toml").read_bytes()).hexdigest(),
        }
    except (PackageNotFoundError, OSError):
        raise RuntimeError("could not bind the Ruff/configuration measurement profile") from None


def measure_owner_sources(repo: Path, owners: Iterable[str]) -> dict[str, str]:
    """Fingerprint measured owner source rather than allowing defect exchanges.

    Parameters
    ----------
    repo : pathlib.Path
        Source root; relative paths follow caller cwd and native symlinks are followed.
    owners : iterable of str
        Canonical scope-relative Python paths selected by the native measurement.

    Returns
    -------
    dict of str to str
        SHA-256 of UTF-8 text after universal CRLF/CR/LF newline normalisation.
        Other whitespace, comments, definitions and docstrings remain significant.

    Raises
    ------
    RuntimeError
        A key or actual source file cannot be inspected. No missing-source hash
        or fallback allowance is manufactured.

    Notes
    -----
    This reads sequential files without locking; it does not authenticate
    baseline provenance or resolve filesystem aliases beyond lexical keys.
    """
    try:
        return {
            owner_path(path): hashlib.sha256((repo / path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
            for path in owners
        }
    except (ValueError, OSError):
        raise RuntimeError("could not fingerprint the measured documentation owners") from None

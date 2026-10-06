# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Report destination checks against selected input paths

"""Keep standalone metadata report destinations distinct from their selected inputs."""

from __future__ import annotations

import errno
from collections.abc import Sequence
from pathlib import Path


def refuse_link_loop(path: Path) -> None:
    """Raise for a path whose symbolic links never resolve.

    ``Path.resolve`` raised ``RuntimeError`` for a link loop before Python 3.13
    and returns the path unresolved since. The loop is asked for explicitly, so
    callers get the same refusal on every supported interpreter.

    Parameters
    ----------
    path : Path
        Path to inspect; a missing path is not a loop and is left to the caller.

    Raises
    ------
    RuntimeError
        A symbolic link on the path refers back to itself.
    """
    try:
        path.stat()
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise RuntimeError(f"Symlink loop from {str(path)!r}") from exc


def _resolve_or_refuse_loop(path: Path) -> Path:
    """Refuse a link loop, then resolve the path."""
    refuse_link_loop(path)
    return path.resolve()


def checked_report_destination(destination: str | Path, *, inputs: Sequence[str | Path]) -> Path:
    """Return the caller's path after refusing aliases of any selected input.

    Parameters
    ----------
    destination
        Output spelling; relative paths use the current working directory.
    inputs
        Explicit protected files or directories. Missing names are protected
        by their resolved spelling. Existing hard links are compared by inode.

    Returns
    -------
    Path
        Original destination spelling. This function creates no directory and
        writes no bytes; an unrelated existing destination may be replaced.

    Raises
    ------
    ValueError
        Destination aliases an input, or a path contains an invalid value.
    OSError, RuntimeError
        Filesystem inspection or symlink resolution fails.

    Notes
    -----
    Symlinks and parent symlinks are resolved before comparison. Only listed
    inputs are protected. Checks are sequential, with no lock, atomic write,
    coherent snapshot or protection against concurrent pathname replacement.
    """
    target = Path(destination)
    resolved_target = _resolve_or_refuse_loop(target)
    for value in inputs:
        source = Path(value)
        if resolved_target == _resolve_or_refuse_loop(source):
            raise ValueError("report output aliases a selected input")
        if target.exists() and source.exists() and target.samefile(source):
            raise ValueError("report output aliases a selected input")
    return target


def manifest_report_inputs(root: str | Path) -> list[Path]:
    """List the manifest report's selected local input paths without hashing bytes.

    Parameters
    ----------
    root : str or Path
        Directory spelling; relative paths follow the working directory.

    Returns
    -------
    list of Path
        Root, discovered manifests/specifications/required artifacts and
        resolvable local manifest artifact paths. Duplicates are retained.

    Raises
    ------
    OSError, ValueError, RuntimeError
        Directory discovery or path inspection fails.

    Notes
    -----
    Include the root, discovered manifests, acquisition specifications and
    required DIII-D artifacts. For each metadata-valid manifest also include its
    resolvable local artifact URIs through the defining evidence-root resolver.
    Invalid metadata still protects its manifest file; unreadable/unresolvable
    artifact names supply no extra paths. No remote artifact is fetched.

    This is an additional sequential discovery for destination checking. It
    establishes no validation outcome, byte integrity or acquisition provenance.
    Discovery/path errors propagate. No concurrent filesystem snapshot exists.
    """
    from validation import validate_data_manifests as owner

    directory = Path(root)
    manifests = owner.iter_manifest_paths(directory)
    paths = [
        directory,
        *manifests,
        *owner.iter_acquisition_spec_paths(directory),
        *owner.iter_diiid_artifact_paths(directory),
    ]
    for path in manifests:
        try:
            manifest = owner.load_real_data_manifest(path, verify_artifact=False)
        except (OSError, ValueError, RuntimeError):
            continue
        for uri in owner._covered_artifact_uris(manifest):
            resolved = owner._resolve_manifest_uri(uri, path, directory)
            if resolved is not None:
                paths.append(resolved)
    return paths

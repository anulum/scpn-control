# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Guarded file publication and recovery.
"""Publish serialised files while preserving inputs and handled-failure recovery."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from pathlib import Path
from uuid import uuid4


class InventoryOutputError(ValueError):
    """Refuse an unsafe publication or retain an incomplete recovery.

    Parameters
    ----------
    message : str
        Deliberately authored refusal, without native exception text.
    recovery_paths : tuple of pathlib.Path
        Retained staging/predecessor paths when recovery could not finish.
        Inspect these before attempting another publication.
    """

    __module__ = "tools.report_inventory_output"

    def __init__(self, message: str, *, recovery_paths: tuple[Path, ...] = ()) -> None:
        super().__init__(message)
        self.recovery_paths = recovery_paths


def _digest(path: Path) -> str:
    """Hash a regular output without loading a predecessor into memory.

    Parameters
    ----------
    path : pathlib.Path
        Published output whose bytes must still match this writer's payload.

    Returns
    -------
    str
        SHA-256 digest computed with one-megabyte read chunks.

    Raises
    ------
    OSError
        Output cannot be opened or read during recovery.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_destinations(
    destinations: tuple[Path, ...],
    protected_files: tuple[Path, ...],
    protected_roots: tuple[Path, ...],
    exclusive_files: tuple[Path, ...] = (),
) -> None:
    """Reject input namespaces, inode aliases and overlapping destinations.

    Parameters
    ----------
    destinations : tuple of pathlib.Path
        Requested output paths, validated together before any creation.
    protected_files : tuple of pathlib.Path
        Registry and indexed report/refresh files, including missing local ones.
    protected_roots : tuple of pathlib.Path
        Caller-selected read-only namespaces, resolved before containment checks.
    exclusive_files : tuple of pathlib.Path, optional
        Requested destinations that must not already have an occupant.

    Raises
    ------
    InventoryOutputError
        A destination is unsafe, nonregular or aliases another output.
    OSError, ValueError
        Filesystem identity or path resolution cannot be inspected.

    Notes
    -----
    These checks require callers to coordinate concurrent namespace changes;
    they do not implement a filesystem sandbox against hostile writers.
    """
    if any(path not in destinations for path in exclusive_files):
        raise InventoryOutputError("Exclusive outputs must belong to the requested destinations")
    for index, destination in enumerate(destinations):
        if destination in exclusive_files and destination.exists():
            raise InventoryOutputError("Exclusive output destinations must not already exist")
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise InventoryOutputError("Inventory outputs must be regular files or new file paths")
        resolved = destination.resolve()
        for root in protected_roots:
            if root.exists() and any(parent.exists() and parent.samefile(root) for parent in resolved.parents):
                raise InventoryOutputError("Inventory outputs must remain outside report and refresh namespaces")
            if resolved.is_relative_to(root.resolve()):
                raise InventoryOutputError("Inventory outputs must remain outside report and refresh namespaces")
        for source in protected_files:
            if resolved == source.resolve() or (
                destination.exists() and source.exists() and destination.samefile(source)
            ):
                raise InventoryOutputError("Inventory outputs must not replace source evidence or its registry")
        for previous in destinations[:index]:
            if resolved == previous.resolve() or (
                destination.exists() and previous.exists() and destination.samefile(previous)
            ):
                raise InventoryOutputError("Inventory outputs must use distinct files")


def _stage(destination: Path, payload: bytes) -> Path:
    """Write a complete sibling file and preserve an existing output's mode.

    Parameters
    ----------
    destination : pathlib.Path
        Validated output path; missing parents are created.
    payload : bytes
        Complete serialised output, flushed and fsynced before return.

    Returns
    -------
    pathlib.Path
        Exclusively created sibling with a random temporary name.

    Raises
    ------
    OSError
        Creation, write, flush or mode preservation fails. Partial staging is
        removed when possible before the original failure is propagated.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name[:32]}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists():
            temporary.chmod(stat.S_IMODE(destination.stat().st_mode))
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _backup(destination: Path) -> Path | None:
    """Preserve an existing output independently of atomic replacement.

    Parameters
    ----------
    destination : pathlib.Path
        Existing regular output or a currently absent destination.

    Returns
    -------
    pathlib.Path or None
        Fsynced sibling copy with predecessor metadata, or null when absent.

    Raises
    ------
    OSError
        Copying or metadata preservation fails; partial backup is removed when
        possible. No output replacement has happened at this stage.
    """
    if not destination.exists():
        return None
    backup = destination.with_name(f".{destination.name[:32]}.{uuid4().hex}.backup")
    try:
        with destination.open("rb") as source, backup.open("xb") as target:
            shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
        shutil.copystat(destination, backup)
        return backup
    except BaseException:
        backup.unlink(missing_ok=True)
        raise


def _publish(
    outputs: tuple[tuple[Path, bytes], ...],
    protected_files: tuple[Path, ...],
    protected_roots: tuple[Path, ...],
    exclusive_files: tuple[Path, ...] = (),
) -> None:
    """Stage all outputs and recover prior bytes after failed publication.

    Parameters
    ----------
    outputs : tuple of (pathlib.Path, bytes)
        Validated distinct destinations and complete serialised payloads.
    protected_files, protected_roots : tuple of pathlib.Path
        Consumed input files and reserved namespaces checked again after parent
        creation, before publication.
    exclusive_files : tuple of pathlib.Path, optional
        Immutable destinations published exclusively from complete siblings.

    Raises
    ------
    InventoryOutputError
        Recovery cannot safely finish; existing staging/backups are retained
        on the exception instead of overwriting a changed output.
    BaseException
        Staging/publication failed and predecessor recovery completed. The
        original interruption is propagated, including cancellation.

    Notes
    -----
    All staging and backups precede replacement. Recovery proceeds in reverse
    publication order and compares each output's bytes with this writer's
    payload. This is handled-failure recovery, not a crash-atomic transaction.
    """
    staged: list[Path] = []
    backups: list[Path | None] = []
    published: list[int] = []
    recovered = True
    try:
        for destination, payload in outputs:
            staged.append(_stage(destination, payload))
            backups.append(_backup(destination))
        _validate_destinations(tuple(path for path, _ in outputs), protected_files, protected_roots, exclusive_files)
        for index, (destination, _) in enumerate(outputs):
            for previous, _ in outputs[:index]:
                if destination.exists() and previous.exists() and destination.samefile(previous):
                    raise InventoryOutputError("Inventory outputs must use distinct files")
            if destination in exclusive_files:
                os.link(staged[index], destination)
            else:
                os.replace(staged[index], destination)
            published.append(index)
    except BaseException:
        try:
            for index in reversed(published):
                destination, payload = outputs[index]
                if (
                    destination.is_symlink()
                    or not destination.is_file()
                    or _digest(destination) != hashlib.sha256(payload).hexdigest()
                ):
                    raise InventoryOutputError("Inventory recovery found an output changed by another writer")
                backup = backups[index]
                if backup is None:
                    destination.unlink()
                else:
                    os.replace(backup, destination)
        except BaseException as recovery_error:
            recovered = False
            retained = tuple(path for path in (*staged, *backups) if path is not None and path.exists())
            raise InventoryOutputError(
                "Inventory output recovery is incomplete; retained files require inspection",
                recovery_paths=retained,
            ) from recovery_error
        raise
    finally:
        if recovered:
            for path in (*staged, *backups):
                if path is not None:
                    path.unlink(missing_ok=True)


def publish_guarded_outputs(
    outputs: tuple[tuple[Path, bytes], ...],
    *,
    protected_files: tuple[Path, ...],
    protected_roots: tuple[Path, ...] = (),
    exclusive_files: tuple[Path, ...] = (),
) -> None:
    """Publish complete byte payloads while preserving caller-selected inputs.

    Parameters
    ----------
    outputs : tuple of (pathlib.Path, bytes)
        Complete payloads and distinct caller-relative destinations.
    protected_files : tuple of pathlib.Path
        Read inputs and reserved paths that publication must not replace.
    protected_roots : tuple of pathlib.Path, optional
        Read-only namespaces, including paths that do not yet exist.

    exclusive_files : tuple of pathlib.Path, optional
        Requested destinations requiring atomic exclusive name creation from
        staged sibling bytes. Existing or concurrently created names cannot
        be overwritten. Unsupported hard-link filesystems fail closed.

    Raises
    ------
    InventoryOutputError
        Unsafe output, input alias, or incomplete handled-failure recovery.
        Recovery paths are retained on the exception for inspection.
    OSError, ValueError, TypeError
        Filesystem inspection or publication fails after successful recovery.

    Notes
    -----
    Replacements are atomic per file. Handled failures restore predecessors
    only while published bytes remain unchanged. Callers must coordinate
    concurrent writers; this is not a crash transaction or hostile sandbox.
    InventoryOutputError retains its canonical report_inventory_output import
    and pickle address for existing consumers.
    """
    outputs = tuple((path.absolute(), payload) for path, payload in outputs)
    exclusive_files = tuple(path.absolute() for path in exclusive_files)
    _validate_destinations(tuple(path for path, _ in outputs), protected_files, protected_roots, exclusive_files)
    _publish(outputs, protected_files, protected_roots, exclusive_files)

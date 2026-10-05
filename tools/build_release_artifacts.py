# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Reproducible Python release-artifact builder.

"""Build source-tree distributions and inspect their archive declarations.

The frontend executes the selected project's build backend. Source distributions
receive sorted members, canonical ownership, a caller-selected unsigned gzip
epoch and an exclusively created temporary file before replacement. Archive
names must be portable POSIX paths; private/build-only members are refused.
Wheel checks inspect the literal SPDX expression and shipped console modules.
These checks do not execute console callables, authenticate artifacts, verify
every wheel RECORD/CRC, or guarantee reproducibility of an arbitrary backend.
"""

from __future__ import annotations

import argparse
import configparser
import copy
import gzip
import hashlib
import io
import os
import shutil
import subprocess  # nosec B404
import sys
import tarfile
import tempfile
import zipfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = ROOT / "dist"
BLOCKED_PARTS = frozenset({".coordination", ".git", "papers", "site"})


@dataclass(frozen=True)
class ArtifactSummary:
    """Store an archive's inspected filename, entry count and byte identity.

    Parameters
    ----------
    path : pathlib.Path
        Inspected path, preserving the caller's spelling.
    entries : int
        Number of archive entries, including directories and duplicate names.
    sha256 : str
        SHA-256 of the compressed archive bytes. This is not an authenticated
        signature. Direct dataclass construction does not validate fields.
    """

    path: Path
    entries: int
    sha256: str


def _validate_member_name(name: str) -> None:
    """Refuse traversal, Windows aliases and private/build-only components.

    Parameters
    ----------
    name : str
        Archive member spelling; portable archives use forward slashes.

    Raises
    ------
    ValueError
        The path is absolute, traverses parents, uses a Windows drive or
        backslash, or contains a blocked component. No extraction occurs.
    """
    member = PurePosixPath(name)
    parts = member.parts
    if member.is_absolute() or ".." in parts or "\\" in name or PureWindowsPath(name).drive:
        raise ValueError(f"unsafe archive member path: {name}")
    if BLOCKED_PARTS.intersection(parts) or any(
        left == "docs" and right == "internal" for left, right in zip(parts, parts[1:])
    ):
        raise ValueError(f"private or build-only archive member: {name}")


def _normalised_info(source: tarfile.TarInfo, epoch: int) -> tarfile.TarInfo:
    """Copy one header with fixed epoch/ownership and no PAX overrides.

    Parameters
    ----------
    source : tarfile.TarInfo
        Original header; payload, name, mode and type remain unchanged.
    epoch : int
        Validated unsigned seconds since the Unix epoch.

    Returns
    -------
    tarfile.TarInfo
        Independent header; the input object is not mutated.
    """
    info = copy.copy(source)
    info.mtime = epoch
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.pax_headers = {}
    return info


def _validate_epoch(epoch: int) -> None:
    """Require the integer epoch representable by the gzip header.

    Parameters
    ----------
    epoch : int
        Unsigned 32-bit Unix seconds. Boolean values are refused.

    Raises
    ------
    ValueError
        The value is not an integer in the inclusive range 0 through 2**32-1.
    """
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 0 <= epoch <= 0xFFFFFFFF:
        raise ValueError("source date epoch must be an integer between 0 and 4294967295")


def normalise_sdist(path: Path, epoch: int) -> None:
    """Replace one sdist with canonical headers and a deterministic gzip stream.

    Parameters
    ----------
    path : pathlib.Path
        Existing gzip tar archive. The caller owns directory coordination;
        ordinary symlink/read/replace behavior follows the native filesystem.
    epoch : int
        Unsigned 32-bit Unix seconds excluding booleans.

    Raises
    ------
    ValueError
        Epoch/member declarations are invalid or a member is not a file/directory.
    OSError, tarfile.TarError
        Reading, writing or replacement fails. Input inspection completes before
        rewriting; the exclusively created temporary file is cleaned afterward.

    Notes
    -----
    Payloads are buffered in memory. Modes and duplicate names are retained.
    Replacement uses a closed, exclusive sibling temporary file; an unrelated
    legacy fixed-name temporary file is preserved. No fsync/directory lock or
    hostile-directory protection is provided.
    """
    _validate_epoch(epoch)
    records: list[tuple[tarfile.TarInfo, bytes | None]] = []
    with tarfile.open(path, "r:gz") as source:
        for member in source:
            _validate_member_name(member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"unsupported sdist member type: {member.name}")
            stream = source.extractfile(member) if member.isfile() else None
            payload = stream.read() if stream is not None else None
            records.append((_normalised_info(member, epoch), payload))

    with ExitStack() as cleanup:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=".scpn-sdist-", suffix=".tmp", delete=False
        ) as raw:
            temporary = Path(raw.name)
            cleanup.callback(temporary.unlink, missing_ok=True)
            with (
                gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed,
                tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as target,
            ):
                for member, payload in sorted(records, key=lambda record: record[0].name):
                    target.addfile(member, io.BytesIO(payload) if payload is not None else None)
        os.replace(temporary, path)


def _wheel_entry_points(archive: zipfile.ZipFile, names: list[str]) -> dict[str, str]:
    """Parse the sole wheel entry-point declaration with ConfigParser.

    Parameters
    ----------
    archive : zipfile.ZipFile
        Open wheel whose UTF-8 declaration is read without interpolation.
    names : list of str
        Actual archive names. Exactly one entry_points.txt is required.

    Returns
    -------
    dict of str to str
        Console-script declarations, or an empty mapping for no console section.
        ConfigParser's ordinary option normalization applies.

    Raises
    ------
    ValueError, configparser.Error, UnicodeError, OSError
        Cardinality, syntax, encoding or archive reads fail.
    """
    candidates = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
    if len(candidates) != 1:
        raise ValueError(f"wheel must contain exactly one entry_points.txt, found {len(candidates)}")
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(archive.read(candidates[0]).decode("utf-8"))
    return dict(parser.items("console_scripts")) if parser.has_section("console_scripts") else {}


def _validate_wheel_targets(archive: zipfile.ZipFile, names: list[str]) -> None:
    """Require each console declaration's module to be shipped in the wheel.

    Parameters
    ----------
    archive : zipfile.ZipFile
        Open wheel with a singular entry-point declaration.
    names : list of str
        Actual member names. Module.py or module/__init__.py establishes presence.

    Raises
    ------
    ValueError
        A declared console module is absent. Callable existence, syntax,
        importability and runtime execution are not inspected.
    """
    available = set(names)
    for command, target in _wheel_entry_points(archive, names).items():
        module = target.partition(":")[0].strip()
        module_path = module.replace(".", "/")
        if f"{module_path}.py" not in available and f"{module_path}/__init__.py" not in available:
            raise ValueError(f"console script {command!r} targets missing wheel module {module!r}")


def _validate_license_expression(archive: zipfile.ZipFile, names: list[str]) -> None:
    """Require one wheel METADATA containing the exact canonical SPDX line.

    Parameters
    ----------
    archive : zipfile.ZipFile
        Open wheel; the metadata is decoded as UTF-8.
    names : list of str
        Actual member names. No complete metadata-schema parser is used.

    Raises
    ------
    ValueError, UnicodeError, OSError
        Metadata cardinality, canonical expression, decoding or reading fails.
    """
    candidates = [name for name in names if name.endswith(".dist-info/METADATA")]
    if len(candidates) != 1:
        raise ValueError(f"wheel must contain exactly one METADATA file, found {len(candidates)}")
    metadata = archive.read(candidates[0]).decode("utf-8")
    if "License-Expression: AGPL-3.0-or-later" not in metadata.splitlines():
        raise ValueError("wheel metadata lacks the canonical SPDX License-Expression")


def validate_artifact(path: Path) -> ArtifactSummary:
    """Inspect archive names, wheel declarations and compressed byte identity.

    Parameters
    ----------
    path : pathlib.Path
        A .whl or .tar.gz file. Relative paths use the caller's working directory.
        Tar archives admit regular files/directories; wheels require one literal
        license metadata and one entry-point declaration.

    Returns
    -------
    ArtifactSummary
        Caller-spelled path, entry count and SHA-256 after inspection.

    Raises
    ------
    ValueError, OSError, tarfile.TarError, zipfile.BadZipFile
        Format, member, metadata, console presence or native archive reads fail.
    configparser.Error, UnicodeError
        Wheel declaration parsing or UTF-8 decoding fails.

    Notes
    -----
    Empty tar inventories and duplicate names are not a package-schema check.
    This does not extract/import payloads, check all CRC/RECORD entries, verify
    signatures, validate console callables or admit a release for publication.
    """
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            for name in names:
                _validate_member_name(name)
            _validate_license_expression(archive, names)
            _validate_wheel_targets(archive, names)
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            for member in members:
                _validate_member_name(member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"unsupported sdist member type: {member.name}")
    else:
        raise ValueError(f"unsupported release artifact: {path.name}")
    return ArtifactSummary(
        path=path,
        entries=len(names),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _source_date_epoch(explicit: int | None) -> int:
    """Select explicit, environment or actual source-commit seconds in order.

    Parameters
    ----------
    explicit : int or None
        Value returned unchanged when provided. Otherwise SOURCE_DATE_EPOCH is
        parsed, then PATH-resolved Git reads HEAD from this script's repository.

    Returns
    -------
    int
        Selected seconds; the caller subsequently validates the gzip range.

    Raises
    ------
    ValueError, RuntimeError, subprocess.CalledProcessError, OSError
        Environment/Git output is invalid, Git is absent or the native probe fails.
        No build starts while selection fails.
    """
    if explicit is not None:
        return explicit
    configured = os.environ.get("SOURCE_DATE_EPOCH")
    if configured is not None:
        try:
            return int(configured)
        except ValueError as error:
            raise ValueError("SOURCE_DATE_EPOCH must be an integer") from error
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to derive SOURCE_DATE_EPOCH")
    # The executable is PATH-resolved once and every argument is constant.
    result = subprocess.run(  # nosec B603
        [git, "log", "-1", "--format=%ct", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def build_release_artifacts(
    outdir: Path,
    *,
    epoch: int,
    sdist_only: bool = False,
) -> list[ArtifactSummary]:
    """Invoke the actual build frontend and inspect the requested artifact count.

    Parameters
    ----------
    outdir : pathlib.Path
        Resolved output directory, created if absent. Existing .whl/.tar.gz files
        refuse before a backend runs; other files are preserved.
    epoch : int
        Unsigned 32-bit Unix seconds excluding booleans, forwarded in the inherited
        environment as SOURCE_DATE_EPOCH.
    sdist_only : bool
        Request one source distribution; otherwise the frontend's default pair.

    Returns
    -------
    list of ArtifactSummary
        Filename-sorted summaries after sdist normalization and inspection.

    Raises
    ------
    ValueError, OSError, subprocess.CalledProcessError
        Epoch/stale/count/member/metadata checks or the native build fail. Produced
        artifacts are retained on failure; publication is never attempted.

    Notes
    -----
    The child uses sys.executable, an argument vector and this script's repository
    cwd. The backend may install isolated build requirements and execute project
    build hooks. No timeout, coherent source snapshot, output-directory lock or
    cross-backend/platform reproducibility guarantee is provided.
    """
    _validate_epoch(epoch)
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    stale = sorted((*outdir.glob("*.whl"), *outdir.glob("*.tar.gz")))
    if stale:
        raise ValueError(f"output directory contains stale distributions: {stale}")

    command = [sys.executable, "-m", "build"]
    if sdist_only:
        command.append("--sdist")
    command.extend(("--outdir", str(outdir)))
    environment = os.environ.copy()
    environment["SOURCE_DATE_EPOCH"] = str(epoch)
    # The command is an argument vector assembled from fixed tokens and one path.
    subprocess.run(command, cwd=ROOT, env=environment, check=True)  # nosec B603

    artifacts = sorted((*outdir.glob("*.whl"), *outdir.glob("*.tar.gz")))
    expected = 1 if sdist_only else 2
    if len(artifacts) != expected:
        raise ValueError(f"expected {expected} distribution artifact(s), found {len(artifacts)}")
    for artifact in artifacts:
        if artifact.name.endswith(".tar.gz"):
            normalise_sdist(artifact, epoch)
    return [validate_artifact(artifact) for artifact in artifacts]


def main(argv: list[str] | None = None) -> int:
    """Build through the source-tree CLI and print tab-separated byte summaries.

    Parameters
    ----------
    argv : list of str or None
        Argparse arguments; None reads the process argv. Defaults use the script
        repository's dist directory and selected commit/environment epoch.

    Returns
    -------
    int
        Zero after a successful build. Help exits zero; usage/epoch-range errors
        exit two. Native build/selection/archive exceptions propagate.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--source-date-epoch", type=int)
    parser.add_argument("--sdist-only", action="store_true")
    arguments = parser.parse_args(argv)
    epoch = _source_date_epoch(arguments.source_date_epoch)
    if not 0 <= epoch <= 0xFFFFFFFF:
        parser.error("source date epoch must be between 0 and 4294967295")
    for summary in build_release_artifacts(
        arguments.outdir,
        epoch=epoch,
        sdist_only=arguments.sdist_only,
    ):
        print(f"{summary.path.name}\tentries={summary.entries}\tsha256={summary.sha256}\tsource_date_epoch={epoch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

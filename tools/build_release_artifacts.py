# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Reproducible Python release-artifact builder.

"""Build and validate deterministic SCPN Control Python distributions."""

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
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = ROOT / "dist"
BLOCKED_PARTS = frozenset({".coordination", ".git", "papers", "site"})


@dataclass(frozen=True)
class ArtifactSummary:
    """Describe one validated release artifact."""

    path: Path
    entries: int
    sha256: str


def _validate_member_name(name: str) -> None:
    member = PurePosixPath(name)
    parts = member.parts
    if member.is_absolute() or ".." in parts:
        raise ValueError(f"unsafe archive member path: {name}")
    if BLOCKED_PARTS.intersection(parts) or any(
        left == "docs" and right == "internal" for left, right in zip(parts, parts[1:])
    ):
        raise ValueError(f"private or build-only archive member: {name}")


def _normalised_info(source: tarfile.TarInfo, epoch: int) -> tarfile.TarInfo:
    info = copy.copy(source)
    info.mtime = epoch
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.pax_headers = {}
    return info


def normalise_sdist(path: Path, epoch: int) -> None:
    """Rewrite an sdist with stable ordering, ownership, timestamps, and gzip metadata."""
    records: list[tuple[tarfile.TarInfo, bytes | None]] = []
    with tarfile.open(path, "r:gz") as source:
        for member in source:
            _validate_member_name(member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"unsupported sdist member type: {member.name}")
            stream = source.extractfile(member) if member.isfile() else None
            payload = stream.read() if stream is not None else None
            records.append((_normalised_info(member, epoch), payload))

    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with (
            temporary.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed,
            tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as target,
        ):
            for member, payload in sorted(records, key=lambda record: record[0].name):
                target.addfile(member, io.BytesIO(payload) if payload is not None else None)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _wheel_entry_points(archive: zipfile.ZipFile, names: list[str]) -> dict[str, str]:
    candidates = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
    if len(candidates) != 1:
        raise ValueError(f"wheel must contain exactly one entry_points.txt, found {len(candidates)}")
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(archive.read(candidates[0]).decode("utf-8"))
    return dict(parser.items("console_scripts")) if parser.has_section("console_scripts") else {}


def _validate_wheel_targets(archive: zipfile.ZipFile, names: list[str]) -> None:
    available = set(names)
    for command, target in _wheel_entry_points(archive, names).items():
        module = target.partition(":")[0].strip()
        module_path = module.replace(".", "/")
        if f"{module_path}.py" not in available and f"{module_path}/__init__.py" not in available:
            raise ValueError(f"console script {command!r} targets missing wheel module {module!r}")


def _validate_license_expression(archive: zipfile.ZipFile, names: list[str]) -> None:
    candidates = [name for name in names if name.endswith(".dist-info/METADATA")]
    if len(candidates) != 1:
        raise ValueError(f"wheel must contain exactly one METADATA file, found {len(candidates)}")
    metadata = archive.read(candidates[0]).decode("utf-8")
    if "\nLicense-Expression: AGPL-3.0-or-later\n" not in f"\n{metadata}":
        raise ValueError("wheel metadata lacks the canonical SPDX License-Expression")


def validate_artifact(path: Path) -> ArtifactSummary:
    """Validate archive boundaries, metadata, and installed console-script targets."""
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
            for name in names:
                _validate_member_name(name)
    else:
        raise ValueError(f"unsupported release artifact: {path.name}")
    return ArtifactSummary(
        path=path,
        entries=len(names),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _source_date_epoch(explicit: int | None) -> int:
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
    """Build, normalise, and validate the requested distribution set."""
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
    """Build release artifacts and print their deterministic identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--source-date-epoch", type=int)
    parser.add_argument("--sdist-only", action="store_true")
    arguments = parser.parse_args(argv)
    epoch = _source_date_epoch(arguments.source_date_epoch)
    if epoch < 0:
        parser.error("source date epoch must be non-negative")
    for summary in build_release_artifacts(
        arguments.outdir,
        epoch=epoch,
        sdist_only=arguments.sdist_only,
    ):
        print(f"{summary.path.name}\tentries={summary.entries}\tsha256={summary.sha256}\tsource_date_epoch={epoch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

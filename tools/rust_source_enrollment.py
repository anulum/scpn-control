# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual Cargo target and compiler dependency enrollment.
"""Bind physical Rust sources to actual Cargo artifacts and native compiler dep-info."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

NativeCommand = Callable[[Path, Sequence[str], float], subprocess.CompletedProcess[str]]
_TARGET_KINDS = {"lib", "bin", "example", "test", "bench", "cdylib", "staticlib", "dylib", "proc-macro"}


@dataclass(frozen=True)
class CargoWorkspace:
    """Retain exact workspace roots, member identities and declared physical targets.

    Attributes
    ----------
    root : pathlib.Path
        Canonical native Cargo workspace root within the supplied repository.
    manifest : pathlib.Path
        Workspace manifest used by subsequent native commands.
    members : frozenset of str
        Actual native package IDs; dependency artifacts do not replace members.
    targets : frozenset of pathlib.Path
        Declared target roots, including external bench/example/test paths.
    """

    root: Path
    manifest: Path
    members: frozenset[str]
    targets: frozenset[Path]


def _object(value: object) -> dict[str, object]:
    """Require complete string-keyed native JSON objects."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise RuntimeError("Cargo requires complete native JSON objects.")
    return {key: item for key, item in value.items()}


def _sequence(value: object) -> list[object]:
    """Require an actual JSON array without coercing absent protocol fields."""
    if not isinstance(value, list):
        raise RuntimeError("Cargo requires complete native JSON arrays.")
    return list(value)


def repository_path(repo: Path, value: object) -> Path:
    """Require an existing regular canonical repository path, retaining legitimate Cargo parent segments.

    Parameters
    ----------
    repo : pathlib.Path
        Canonical Git root used for source ownership.
    value : object
        Absolute native Cargo path; literal parent segments may identify benches.

    Returns
    -------
    pathlib.Path
        Canonical physical file or directory within repository custody.

    Raises
    ------
    RuntimeError
        The native path is absent, relative, symlinked or outside the repository.
    """
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise RuntimeError("Cargo source paths must be absolute native paths.")
    path = Path(value)
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise RuntimeError("Cargo source paths cannot use symlink custody.")
    canonical = path.resolve()
    if not canonical.is_relative_to(repo) or not canonical.exists():
        raise RuntimeError("Cargo source paths require existing repository custody.")
    return canonical


def _metadata_workspace(repo: Path, data: dict[str, object]) -> CargoWorkspace:
    """Require complete unique member identities and physical targets from actual metadata."""
    root = repository_path(repo, data.get("workspace_root"))
    members = _sequence(data.get("workspace_members"))
    if not members or any(not isinstance(member, str) or not member for member in members):
        raise RuntimeError("Cargo metadata requires nonempty native member IDs.")
    if len(set(members)) != len(members):
        raise RuntimeError("Cargo native member IDs must be unique.")
    identities: set[str] = set()
    targets: set[Path] = set()
    for value in _sequence(data.get("packages")):
        package = _object(value)
        identity = package.get("id")
        if not isinstance(identity, str) or identity not in members or package.get("edition") != "2021":
            raise RuntimeError("Cargo package identity or edition requires profile qualification.")
        repository_path(repo, package.get("manifest_path"))
        if identity in identities:
            raise RuntimeError("Cargo package IDs must be unique.")
        identities.add(identity)
        native_targets = _sequence(package.get("targets"))
        if not native_targets:
            raise RuntimeError("Cargo member packages must declare native targets.")
        for target_value in native_targets:
            target = _object(target_value)
            kinds = _sequence(target.get("kind"))
            if not kinds or any(not isinstance(kind, str) or kind not in _TARGET_KINDS for kind in kinds):
                raise RuntimeError("Cargo target kind, including build scripts, requires profile qualification.")
            targets.add(repository_path(repo, target.get("src_path")))
    if identities != set(members) or not targets:
        raise RuntimeError("Cargo metadata must enroll every member and physical target.")
    return CargoWorkspace(root, root / "Cargo.toml", frozenset(identities), frozenset(targets))


def read_workspace(repo: Path, manifest: Path, run: NativeCommand, timeout_s: float) -> CargoWorkspace:
    """Read actual locked Cargo metadata, enrolling every native workspace target.

    Parameters
    ----------
    repo : pathlib.Path
        Canonical repository root.
    manifest : pathlib.Path
        Physical manifest selected from the changed source/carrier context.
    run : callable
        Bounded actual native command runner with a qualified environment.
    timeout_s : float
        Per-command bound already validated by the public source checker.

    Returns
    -------
    CargoWorkspace
        Exact nonempty member and target sets from format-version1 metadata.

    Raises
    ------
    RuntimeError
        Metadata, the 2021 edition, local member ownership or a supported target
        is unavailable. Workspace build scripts require separate qualification.
    """
    result = run(
        manifest.parent,
        [
            "cargo",
            "+1.98.0",
            "metadata",
            "--manifest-path",
            str(manifest),
            "--format-version",
            "1",
            "--no-deps",
            "--locked",
            "--offline",
        ],
        timeout_s,
    )
    if result.returncode or any(line.startswith("warning:") for line in result.stderr.splitlines()):
        raise RuntimeError(f"Cargo metadata is unavailable or unqualified:\n{result.stderr}")
    data = _object(json.loads(result.stdout))
    return _metadata_workspace(repo, data)


def _dep_words(text: str) -> list[str]:
    """Decode Rust dep-info space escapes; hashes and dollars are native literal filename data."""
    words: list[str] = []
    word = ""
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 1
            if index == len(text):
                raise RuntimeError("Rust dep-info has a truncated filename escape.")
            word += text[index]
        elif char.isspace():
            if word:
                words.append(word)
                word = ""
        else:
            word += char
        index += 1
    if word:
        words.append(word)
    return words


def _dependencies(path: Path, workspace: CargoWorkspace) -> set[Path]:
    """Read the first complete native rule and require the actual qualified Clippy argument binding."""
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise RuntimeError("Rust compiler dep-info requires regular artifact custody.")
    text = path.read_bytes().decode("utf-8")
    if not text.endswith("\n") or "# env-dep:CLIPPY_ARGS=-D__CLIPPY_HACKERY__warnings__CLIPPY_HACKERY__\n" not in text:
        raise RuntimeError("Rust dep-info requires complete qualified Clippy argument binding.")
    rule = text.replace("\\\n", "").splitlines()[0]
    target, separator, dependencies = rule.partition(": ")
    if not target or not separator:
        raise RuntimeError("Rust dep-info requires a complete compiler dependency rule.")
    words = _dep_words(dependencies)
    if not words:
        raise RuntimeError("Rust dep-info has no physical compiler inputs.")
    return {(workspace.root / word).resolve() for word in words}


def _artifact_inputs(repo: Path, workspace: CargoWorkspace, artifact: dict[str, object]) -> set[Path]:
    """Bind one actual member artifact to its declared root and qualified compiler dep-info."""
    target = _object(artifact.get("target"))
    source = repository_path(repo, target.get("src_path"))
    if source not in workspace.targets:
        raise RuntimeError("Cargo artifact target is outside declared workspace enrollment.")
    observed: set[Path] = set()
    for filename in _sequence(artifact.get("filenames")):
        if not isinstance(filename, str) or not Path(filename).is_absolute():
            raise RuntimeError("Cargo artifact filenames must be absolute native paths.")
        output = Path(filename)
        stems = {output.stem, output.stem.removeprefix("lib")}
        for stem in stems:
            depfile = output.with_name(stem + ".d")
            if depfile.is_file():
                inputs = _dependencies(depfile, workspace)
                if source not in inputs:
                    raise RuntimeError("Cargo compiler dep-info omits its declared target root.")
                observed.update(inputs)
    if not observed:
        raise RuntimeError("Cargo artifact requires actual native compiler dep-info.")
    return observed


def cargo_build_result(
    repo: Path, workspace: CargoWorkspace, result: subprocess.CompletedProcess[str]
) -> tuple[list[str], set[Path]]:
    """Read genuine native compiler findings and physical member input enrollment.

    Parameters
    ----------
    repo : pathlib.Path
        Canonical repository source root.
    workspace : CargoWorkspace
        Already measured native metadata context.
    result : subprocess.CompletedProcess of str
        Actual completed Cargo Clippy JSON command with warnings denied.

    Returns
    -------
    tuple of list of str and set of pathlib.Path
        Native findings and actual compiled member inputs. A successful footer,
        zero findings and complete source enrollment are separate requirements.

    Raises
    ------
    RuntimeError
        Native build results, qualified dep-info or compiler measurement is
        incomplete. Missing offline dependencies are not source-quality passes.
    """
    messages = [_object(json.loads(line)) for line in result.stdout.splitlines()]
    findings: list[str] = []
    inputs: set[Path] = set()
    footers = []
    for message in messages:
        reason = message.get("reason")
        if reason == "compiler-message":
            diagnostic = _object(message.get("message"))
            if diagnostic.get("level") in {"warning", "error", "failure-note"}:
                findings.append(str(diagnostic.get("rendered") or diagnostic["message"]))
        elif reason == "compiler-artifact" and message.get("package_id") in workspace.members:
            inputs.update(_artifact_inputs(repo, workspace, message))
        elif reason == "build-finished":
            footers.append(message.get("success"))
        elif reason not in {"compiler-artifact", "build-script-executed"}:
            raise RuntimeError("Cargo build protocol contains an unqualified message reason.")
    if result.returncode and findings:
        return findings, inputs
    if result.returncode or footers != [True] or not inputs:
        raise RuntimeError(f"Cargo requires a completed native build and compiler enrollment:\n{result.stderr}")
    findings.extend(line for line in result.stderr.splitlines() if line.startswith("warning:"))
    return findings, inputs

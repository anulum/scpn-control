# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native changed-source Rust checks and actual Cargo enrollment.
"""Check full changed Rust sources through actual native parsers and Cargo targets."""

from __future__ import annotations

import argparse
import functools
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.docstring_debt_ledger import read_owner_ledger
from tools.rust_source_ast import rust_ast_errors
from tools.rust_source_enrollment import CargoWorkspace, NativeCommand, cargo_build_result, read_workspace
from tools.rust_source_profile import _CONFIG_NAMES, _PARSER, _native, _profile, _snapshot
from tools.source_quality_scope import SelectedFile, _git, _paths, select_changed_files

ROOT = Path(__file__).resolve().parents[1]


def _carrier(path: Path) -> bool:
    """Identify native manifest, lock, toolchain and configuration inputs."""
    return path.name in {"Cargo.toml", "Cargo.lock", "rust-toolchain.toml"} or any(
        path.as_posix().endswith(name) for name in _CONFIG_NAMES
    )


def _deleted_inputs(repo: Path, base_revision: str, timeout_s: float) -> list[Path]:
    """Refuse hidden materialized inputs and keep real Rust/carrier deletions in qualification."""
    flags = _paths(_git(repo, ["ls-files", "-v", "-z"], timeout_s))
    for entry in flags:
        status, _, name = entry.partition(" ")
        path = repo / name
        if status != "H" and path.exists() and (path.suffix.casefold() == ".rs" or _carrier(path)):
            raise RuntimeError("Materialized Rust inputs cannot use hidden Git index flags.")
    raw = _git(
        repo,
        ["diff", "--no-ext-diff", "--no-textconv", "--name-only", "--diff-filter=D", "-z", base_revision, "--"],
        timeout_s,
    )
    return [
        repo / name for name in sorted(_paths(raw)) if Path(name).suffix.casefold() == ".rs" or _carrier(Path(name))
    ]


def _workspaces(repo: Path, paths: list[Path], run: NativeCommand, timeout_s: float) -> list[CargoWorkspace]:
    """Discover actual independent Cargo contexts, including the excluded fuzz workspace."""
    contexts: dict[Path, CargoWorkspace] = {}
    manifests: set[Path] = set()
    for path in paths:
        manifest = next(
            (
                parent / "Cargo.toml"
                for parent in (path.parent, *path.parent.parents)
                if parent.is_relative_to(repo) and (parent / "Cargo.toml").is_file()
            ),
            None,
        )
        if manifest is None:
            raise RuntimeError(f"{path}: Rust source/carrier requires a real Cargo manifest.")
        if manifest in manifests:
            continue
        manifests.add(manifest)
        workspace = read_workspace(repo, manifest, run, timeout_s)
        contexts[workspace.root] = workspace
    return [contexts[root] for root in sorted(contexts)]


def _source_errors(repo: Path, paths: list[Path], run: NativeCommand, target_dir: Path, timeout_s: float) -> list[str]:
    """Parse complete physical files, enforce native declarations and check pinned formatting."""
    errors: list[str] = []
    for path in paths:
        result = run(
            repo,
            [
                "rustc",
                f"+{_PARSER}",
                "--edition=2021",
                "--crate-type=lib",
                "--crate-name=scpn_source",
                "-Zunpretty=ast-tree",
                str(path),
            ],
            timeout_s,
        )
        if result.returncode:
            errors.append(f"{path}: native Rust AST parse failed:\n{result.stderr}")
        else:
            errors.extend(rust_ast_errors(result.stdout, path))
        result = run(
            repo,
            [
                "rustfmt",
                "+1.98.0",
                "--edition=2021",
                "--check",
                "--config-path",
                str(target_dir / ".quality-profile/rustfmt.toml"),
                str(path),
            ],
            timeout_s,
        )
        if result.returncode:
            errors.append(f"{path}: native rustfmt failed:\n{result.stdout}{result.stderr}")
    return errors


def _compile(
    repo: Path, contexts: list[CargoWorkspace], inputs: dict[Path, bytes], run: NativeCommand, timeout_s: float
) -> tuple[list[str], set[Path]]:
    """Measure actual all-target/all-feature member builds and reject unenrolled compiler source."""
    errors: list[str] = []
    observed: set[Path] = set()
    for workspace in contexts:
        if not workspace.targets <= inputs.keys():
            raise RuntimeError("Cargo targets require enrolled current Git source bytes.")
        result = run(
            workspace.root,
            [
                "cargo",
                "+1.98.0",
                "clippy",
                "--manifest-path",
                str(workspace.manifest),
                "--workspace",
                "--all-targets",
                "--all-features",
                "--locked",
                "--offline",
                "--target=x86_64-unknown-linux-gnu",
                "--message-format=json",
                "--",
                "-D",
                "warnings",
            ],
            timeout_s,
        )
        findings, compiled = cargo_build_result(repo, workspace, result)
        errors.extend(findings)
        if not findings and not workspace.targets <= compiled:
            raise RuntimeError("Cargo did not measure every declared workspace target.")
        rust_inputs = {path for path in compiled if path.suffix.casefold() == ".rs"}
        if not rust_inputs <= inputs.keys():
            raise RuntimeError("Cargo consumed Rust source outside the enrolled physical Git inputs.")
        observed.update(rust_inputs)
    return errors, observed


def _measure(
    repo: Path,
    base_revision: str,
    scope: tuple[SelectedFile, ...],
    rust: list[Path],
    carriers: list[Path],
    target_dir: Path,
    timeout_s: float,
) -> list[str]:
    """Require whole-file native results and stable final physical/compiler input enrollment."""
    run = functools.partial(_native, target_dir=target_dir)
    profile = _profile(repo, target_dir, run, timeout_s)
    inputs = _snapshot(repo, run, timeout_s)
    contexts = _workspaces(repo, rust + carriers, run, timeout_s)
    errors = _source_errors(repo, rust, run, target_dir, timeout_s)
    findings, observed = _compile(repo, contexts, inputs, run, timeout_s)
    errors.extend(findings)
    if not errors and not set(rust) <= observed:
        raise RuntimeError("Changed Rust source is not actually enrolled in native compilation.")
    if _snapshot(repo, run, timeout_s) != inputs or any(
        path.is_symlink() or path.parent.is_symlink() or path.read_bytes() != content
        for path, content in profile.items()
    ):
        raise RuntimeError("Rust source/configuration inputs changed during native checks.")
    if select_changed_files(repo, base_revision) != scope:
        raise RuntimeError("Rust source selection changed during native checks.")
    return errors


def changed_rust_errors(
    repo: Path, base_revision: str, *, target_dir: Path | None = None, timeout_s: float = 600.0
) -> list[str]:
    """Check changed whole Rust files and manifest-triggered native Cargo contexts.

    Parameters
    ----------
    repo : pathlib.Path
        Canonical Git repository root; selected source is never rewritten.
    base_revision : str
        Trusted original full ancestor revision, checked by the shared selector.
    target_dir : pathlib.Path or None
        Explicit absolute owned build allocation, required for a Rust cohort.
    timeout_s : float
        Finite positive bound per native command; Cargo uses one build job.
        The default leaves room for a cold workspace Clippy run, measured at
        129 s for the slowest command with one job.

    Returns
    -------
    list of str
        Actual native documentation, formatter or compiler/lint findings.

    Raises
    ------
    RuntimeError
        Qualified tools/profile, real compiler enrollment, source custody or
        stable final inputs are unavailable. Runtime and semantic gates remain.
    """
    if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise RuntimeError("The Rust-check timeout must be finite and positive.")
    repo = repo.resolve()
    scope = select_changed_files(repo, base_revision)
    rust = [repo / item.path for item in scope if item.profile == "rust"]
    carriers = [repo / item.path for item in scope if _carrier(Path(item.path))]
    deleted = _deleted_inputs(repo, base_revision, timeout_s)
    if not rust and not carriers and not deleted:
        return []
    if target_dir is None:
        raise RuntimeError("Rust checks require an explicit owned target directory.")
    try:
        return _measure(repo, base_revision, scope, rust, carriers + deleted, target_dir, timeout_s)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, RecursionError) as error:
        raise RuntimeError("Rust checks require readable sources and complete native protocols.") from error


def main(argv: Sequence[str] | None = None) -> int:
    """Run native checks using the original ledger revision and explicit build custody.

    Parameters
    ----------
    argv : sequence of str or None
        CLI arguments; --target-dir defaults only to SCPN_RUST_SOURCE_TARGET_DIR.

    Returns
    -------
    int
        Zero for this static profile, one for findings, two for refused input.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--base")
    parser.add_argument("--target-dir", type=Path, default=os.environ.get("SCPN_RUST_SOURCE_TARGET_DIR"))
    args = parser.parse_args(argv)
    try:
        base = args.base or read_owner_ledger(args.repo / "tools/docstring_debt_ceiling.json").origin_revision
        errors = changed_rust_errors(args.repo, base, target_dir=args.target_dir)
    except RuntimeError as error:
        print(f"Changed-Rust checks refused: {error}", file=sys.stderr)
        return 2
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Changed-Rust static profile passed; runtime, generated-source and other native gates remain separate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Qualified Rust tools and compiler input custody.
"""Qualify native Rust tools, fixed configuration and complete physical input custody."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from tools.rust_source_enrollment import NativeCommand

_PARSER = "nightly-2026-08-21"
_VERSIONS = [
    (["rustc", "+1.98.0", "--version"], "rustc 1.98.0 (88d9e12ae 2026-08-18)"),
    (["cargo", "+1.98.0", "--version"], "cargo 1.98.0 (797e8a9bc 2026-08-05)"),
    (["rustfmt", "+1.98.0", "--version"], "rustfmt 1.9.0-stable (88d9e12ae1 2026-08-18)"),
    (["cargo", "+1.98.0", "clippy", "--version"], "clippy 0.1.98 (88d9e12ae1 2026-08-18)"),
    (["rustc", f"+{_PARSER}", "--version"], "rustc 1.100.0-nightly (8925ea358 2026-08-20)"),
]
_TOOLCHAIN_SHA = "7244985ca7ace5c883a14b6a4bf19e5ffe1c79f72b6ddad0b40ce872db3847cd"
_CONFIG_NAMES = ("clippy.toml", ".clippy.toml", "rustfmt.toml", ".rustfmt.toml", ".cargo/config", ".cargo/config.toml")
_FORMAT = 'edition = "2021"\nstyle_edition = "2021"\n'


def _native(repo: Path, argv: Sequence[str], timeout_s: float, *, target_dir: Path) -> subprocess.CompletedProcess[str]:
    """Run one bounded Linux native command with controlled build/config environments."""
    env = os.environ.copy()
    for name in list(env):
        if name in {
            "RUSTFLAGS",
            "RUSTDOCFLAGS",
            "RUSTC_WRAPPER",
            "RUSTC_WORKSPACE_WRAPPER",
            "RUSTC_BOOTSTRAP",
            "CLIPPY_ARGS",
            "RUSTC",
            "RUSTDOC",
            "CARGO_BUILD_TARGET",
        } or name.endswith("_RUSTFLAGS"):
            env.pop(name, None)
    env.update(
        CARGO_TARGET_DIR=str(target_dir),
        CARGO_BUILD_JOBS="1",
        CARGO_INCREMENTAL="0",
        CARGO_NET_OFFLINE="true",
        CARGO_TERM_COLOR="never",
        CARGO_ENCODED_RUSTFLAGS="",
        CLIPPY_CONF_DIR=str(target_dir / ".quality-profile"),
        RUSTC=shutil.which("rustc") or "rustc",
        RUSTC_WRAPPER="",
        RUSTC_WORKSPACE_WRAPPER="",
    )

    try:
        return subprocess.run(
            ["/usr/bin/nice", "-n", "15", "/usr/bin/prlimit", "--as=2147483648:2147483648", "--", *argv],
            cwd=repo,
            input="",
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
            env=env,
        )
    except (OSError, UnicodeError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"Rust checks require completed native {argv[0]} execution.") from error


def _profile(repo: Path, target_dir: Path, run: NativeCommand, timeout_s: float) -> dict[Path, bytes]:
    """Require exact compiler/profile bytes and prepare explicit immutable format/lint configuration."""
    if any(path.is_symlink() for path in (target_dir, *target_dir.parents)):
        raise RuntimeError("Rust build allocation cannot have symlink custody.")
    if (
        sys.platform != "linux"
        or not target_dir.is_absolute()
        or target_dir.resolve() != target_dir
        or any(char in str(target_dir) for char in ":\n\r\\")
    ):
        raise RuntimeError("Rust checks require the qualified Linux platform and absolute build allocation.")
    if target_dir.is_relative_to(repo):
        result = run(repo, ["git", "check-ignore", "--quiet", "--", str(target_dir.relative_to(repo)) + "/"], timeout_s)
        if result.returncode:
            raise RuntimeError("Rust build output inside the repository must be Git ignored.")
    toolchain = repo / "rust-toolchain.toml"
    if hashlib.sha256(toolchain.read_bytes()).hexdigest() != _TOOLCHAIN_SHA:
        raise RuntimeError("Rust toolchain declarations require an explicitly qualified profile update.")
    directory = target_dir / ".quality-profile"
    if directory.is_symlink():
        raise RuntimeError("Rust native profile directory cannot have symlink custody.")
    directory.mkdir(parents=True, exist_ok=True)
    files = {directory / "rustfmt.toml": _FORMAT.encode(), directory / "clippy.toml": b""}
    for path, content in files.items():
        if path.is_symlink():
            raise RuntimeError("Rust native profile files cannot have symlink custody.")
        if path.exists() and path.read_bytes() != content:
            raise RuntimeError("Rust native profile configuration changed.")
        if not path.exists():
            path.write_bytes(content)
    for argv, expected in _VERSIONS:
        result = run(repo, argv, timeout_s)
        if result.returncode or result.stdout.strip() != expected:
            raise RuntimeError(f"Rust checks require the qualified native version: {expected}.")
    return {toolchain: toolchain.read_bytes(), **files}


def _snapshot(repo: Path, run: NativeCommand, timeout_s: float) -> dict[Path, bytes]:
    """Bind every current Git-owned Rust source/manifest/lock, refusing unreviewed native configuration."""
    result = run(
        repo, ["git", "--no-optional-locks", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], timeout_s
    )
    if result.returncode or (result.stdout and not result.stdout.endswith("\0")):
        raise RuntimeError("Rust inputs require a complete native Git source inventory.")
    paths = [repo / name for name in result.stdout.rstrip("\0").split("\0") if name]
    inputs: dict[Path, bytes] = {}
    for path in paths:
        if not path.is_file() or (path.suffix.casefold() != ".rs" and path.name not in {"Cargo.toml", "Cargo.lock"}):
            continue
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            raise RuntimeError("Rust compiler inputs require regular source custody.")
        inputs[path] = path.read_bytes()
        for parent in path.relative_to(repo).parents:
            directory = repo / parent
            if any((directory / name).exists() for name in _CONFIG_NAMES):
                raise RuntimeError("Rust native configuration requires explicit profile qualification.")
    return inputs

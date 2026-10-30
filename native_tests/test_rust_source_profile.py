# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Rust toolchain, build allocation and final input custody.
"""Exercise physical profile boundaries through the actual native source operator."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from test_tools.test_source_quality_scope import _write
from test_tools.test_source_quality_scope import git_repo as git_repo

from native_tests.test_rust_source_checks import _VALID
from native_tests.test_rust_source_checks import rust_repo as rust_repo
from tools.rust_source_checks import changed_rust_errors
from tools.rust_source_profile import _native


@pytest.mark.parametrize("fault", ["relative", "symlink", "unignored", "toolchain", "directory", "file", "config"])
def test_physical_profile_custody_refuses(rust_repo: tuple[Path, str, Path], fault: str) -> None:
    """Unqualified allocations and physical profile bytes cannot replace the recorded native tools."""
    repo, base, target = rust_repo
    if fault == "relative":
        target = Path("relative-build")
    elif fault == "symlink":
        actual = target.with_name(target.name + "-actual")
        actual.mkdir()
        target.symlink_to(actual, target_is_directory=True)
    elif fault == "unignored":
        target = repo / "unignored-build"
    elif fault == "toolchain":
        _write(repo, "rust-toolchain.toml", b"unqualified bytes\n")
    else:
        directory = target / ".quality-profile"
        directory.mkdir(parents=True)
        if fault == "directory":
            directory.rmdir()
            directory.symlink_to(repo, target_is_directory=True)
        elif fault == "file":
            (directory / "clippy.toml").symlink_to(repo / "rust-toolchain.toml")
        else:
            (directory / "rustfmt.toml").write_text('edition = "2018"\n')
    with pytest.raises(RuntimeError):
        changed_rust_errors(repo, base, target_dir=target)


def test_ignored_in_repository_allocation_and_controlled_environment(
    rust_repo: tuple[Path, str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuine ignored build passes despite inherited lint waivers and compiler-wrapper settings."""
    repo, base, _ = rust_repo
    target = repo / "owned-build"
    _write(repo, ".gitignore", b"/owned-build/\n")
    monkeypatch.setenv("RUSTFLAGS", "--cap-lints allow")
    monkeypatch.setenv("RUSTC_WRAPPER", "/not/a/compiler")
    assert changed_rust_errors(repo, base, target_dir=target) == []
    assert changed_rust_errors(repo, base, target_dir=target) == []


@pytest.mark.parametrize(
    "fault",
    [
        "inventory",
        "nul",
        "source-symlink",
        "profile",
        "selection",
        "target",
        "observed",
        "unenrolled",
        "protocol",
        "missing-target",
    ],
)
def test_damage_after_genuine_native_results_refuses(
    rust_repo: tuple[Path, str, Path], monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """Real native outputs remain the producer before one custody or completeness property is damaged."""
    repo, base, target = rust_repo
    inventory_calls = 0
    if fault == "missing-target":
        _write(repo, "new-root/examples/extra.rs", b"//! Describe a target.\n\n/// Execute the target.\nfn main() {}\n")

    def damage(
        root: Path, argv: Sequence[str], timeout_s: float, *, target_dir: Path
    ) -> subprocess.CompletedProcess[str]:
        """Preserve actual execution before injecting a negative fault at the public boundary."""
        nonlocal inventory_calls
        result = _native(root, argv, timeout_s, target_dir=target_dir)
        if "ls-files" in argv:
            inventory_calls += 1
            if fault == "inventory":
                result.returncode = 1
            elif fault == "nul":
                result.stdout = result.stdout.rstrip("\0")
            elif fault == "target":
                result.stdout = result.stdout.replace("new-root/src/lib.rs\0", "")
            elif fault == "unenrolled":
                _write(repo, ".gitignore", b"/new-root/src/child.rs\n")
                _write(repo, "new-root/src/child.rs", _VALID.encode())
                text = _VALID.replace("    1", "    child::value()") + "\n/// Group the child value.\nmod child;\n"
                _write(repo, "new-root/src/lib.rs", text.encode())
            elif fault == "source-symlink":
                original = repo / "new-root/src/lib.rs"
                backup = repo / "retained-source"
                original.rename(backup)
                original.symlink_to(backup)
        if "--message-format=json" in argv:
            if fault == "profile":
                (target / ".quality-profile/clippy.toml").write_text("unqualified\n")
            elif fault == "selection":
                _write(repo, "additional.txt", b"new selection\n")
            elif fault == "observed":
                result.stdout = result.stdout.replace(
                    '"reason":"compiler-artifact"', '"reason":"build-script-executed"'
                )
            elif fault == "missing-target":
                messages = [json.loads(line) for line in result.stdout.splitlines()]
                messages = [
                    item
                    for item in messages
                    if not str(item.get("target", {}).get("src_path", "")).endswith("/examples/extra.rs")
                ]
                result.stdout = "\n".join(json.dumps(item) for item in messages)
            elif fault == "protocol":
                result.stdout = "not JSON"
        return result

    monkeypatch.setattr("tools.rust_source_checks._native", damage)
    with pytest.raises(RuntimeError):
        changed_rust_errors(repo, base, target_dir=target)
    assert inventory_calls >= 1


def test_absent_build_allocation_and_manifest_refuse(rust_repo: tuple[Path, str, Path]) -> None:
    """Rust source selection needs both explicit build custody and an existing native manifest."""
    repo, base, target = rust_repo
    with pytest.raises(RuntimeError, match="explicit owned"):
        changed_rust_errors(repo, base)
    (repo / "new-root/Cargo.toml").unlink()
    (repo / "new-root/Cargo.lock").unlink()
    with pytest.raises(RuntimeError, match="real Cargo manifest"):
        changed_rust_errors(repo, base, target_dir=target)


def test_actual_child_resource_limits_and_timeout(rust_repo: tuple[Path, str, Path]) -> None:
    """The actual native child observes the declared address-space/nice bounds and finite timeout."""
    repo, _, target = rust_repo
    result = _native(
        repo,
        [
            sys.executable,
            "-B",
            "-c",
            "import os,resource; print(resource.getrlimit(resource.RLIMIT_AS)); print(os.nice(0))",
        ],
        30.0,
        target_dir=target,
    )
    assert result.returncode == 0
    limits, nice = result.stdout.splitlines()
    assert limits == "(4294967296, 4294967296)" and int(nice) >= 15
    with pytest.raises(RuntimeError, match="completed native"):
        _native(repo, [sys.executable, "-B", "-c", "import time; time.sleep(10)"], 0.001, target_dir=target)

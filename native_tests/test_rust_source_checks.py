# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual Rust source API, CLI, formatter and Cargo contracts.
"""Measure real native Rust targets without executing application or test bodies."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_tools.test_source_quality_scope import ROOT, _write
from test_tools.test_source_quality_scope import git_repo as git_repo

from tools.preflight import GATES, run_gate
from tools.rust_source_checks import changed_rust_errors, main

_VALID = "//! Describe a native source contract.\n\n/// Return a fixed value.\npub fn value() -> i32 {\n    1\n}\n"
_MANIFEST = '[package]\nname = "native_quality"\nversion = "0.1.0"\nedition = "2021"\n\n[workspace]\n'


def _sparse(repo: Path) -> None:
    """Mark only absent original index entries as unmaterialized, using actual Git sparse bits."""
    raw = subprocess.check_output(["git", "-C", str(repo), "ls-files", "-z"])
    missing = [name for name in raw.split(b"\0") if name and not (repo / name.decode()).exists()]
    subprocess.run(
        ["git", "-C", str(repo), "update-index", "--skip-worktree", "-z", "--stdin"],
        input=b"\0".join(missing) + b"\0",
        check=True,
        capture_output=True,
    )


def _lock(crate: Path) -> None:
    """Generate an actual dependency-free native lock rather than bypassing locked execution."""
    result = subprocess.run(
        ["cargo", "+1.98.0", "generate-lockfile", "--offline"],
        cwd=crate,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def rust_repo(git_repo: tuple[Path, str]) -> tuple[Path, str, Path]:
    """Supply one physical std-only crate in a new root against real original Git objects."""
    repo, base = git_repo
    _sparse(repo)
    subprocess.run(
        ["git", "-C", str(repo), "update-index", "--no-skip-worktree", "--", "rust-toolchain.toml"],
        check=True,
        capture_output=True,
    )
    _write(repo, "rust-toolchain.toml", (ROOT / "rust-toolchain.toml").read_bytes())
    _write(repo, "new-root/Cargo.toml", _MANIFEST.encode())
    _write(repo, "new-root/src/lib.rs", _VALID.encode())
    _lock(repo / "new-root")
    return repo, base, repo.parent / (repo.name + "-rust-target")


def _cli(repo: Path, base: str, target: Path) -> subprocess.CompletedProcess[str]:
    """Run the physical public command with optional original-process coverage."""
    argv = [sys.executable, "-B"]
    coverage_rc = os.environ.get("SCPN_RUST_SOURCE_COVERAGE_RC")
    if coverage_rc:
        argv += ["-m", "coverage", "run", "--parallel-mode", f"--rcfile={coverage_rc}"]
    return subprocess.run(
        [
            *argv,
            str(ROOT / "tools/rust_source_checks.py"),
            "--repo",
            str(repo),
            "--base",
            base,
            "--target-dir",
            str(target),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )


def test_actual_native_tools_and_public_cli_accept_a_new_crate(rust_repo: tuple[Path, str, Path]) -> None:
    """Real parser, formatter, Cargo metadata and compiler dep-info all bind the selected owner."""
    repo, base, target = rust_repo
    assert changed_rust_errors(repo, base, target_dir=target) == []
    result = _cli(repo, base, target)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "static profile passed" in result.stdout


@pytest.mark.parametrize(
    "content,finding",
    [
        (_VALID.replace("    1", "    true"), "mismatched types"),
        (_VALID.replace("    1", "    let unused = 1;\n    1"), "unused variable"),
        (_VALID.replace("    1", "  1"), "native rustfmt failed"),
        (_VALID.replace("/// Return a fixed value.\n", ""), "attached contract"),
        (_VALID.replace("pub fn", "#[allow(dead_code)]\npub fn"), "profile configuration"),
        (_VALID.replace("pub fn value()", "pub fn value("), "AST parse failed"),
        (_VALID + "\n#[doc(hidden)]\nfn missing() {}\n", "attached contract"),
        (_VALID + "\n#[cfg(test)]\nmod tests { #[test] fn missing() {} }\n", "attached contract"),
    ],
)
def test_actual_native_findings_reach_the_cli(rust_repo: tuple[Path, str, Path], content: str, finding: str) -> None:
    """Compilation, style, documentation and actual waiver defects cannot certify a static pass."""
    repo, base, target = rust_repo
    _write(repo, "new-root/src/lib.rs", content.encode())
    result = _cli(repo, base, target)
    assert result.returncode == 1 and finding in result.stderr, result.stdout + result.stderr


def test_physical_orphan_source_is_not_cargo_enrollment(rust_repo: tuple[Path, str, Path]) -> None:
    """A nearby Cargo manifest cannot certify a source absent from actual compiler dependencies."""
    repo, base, target = rust_repo
    _write(repo, "new-root/src/orphan.rs", _VALID.encode())
    result = _cli(repo, base, target)
    assert result.returncode == 2 and "not actually enrolled" in result.stderr


def test_actual_loaded_submodule_is_bound_by_native_compiler_depinfo(rust_repo: tuple[Path, str, Path]) -> None:
    """Both the declared crate root and physical module are measured from genuine native artifacts."""
    repo, base, target = rust_repo
    _write(repo, "new-root/src/child.rs", _VALID.encode())
    source = _VALID.replace(
        "/// Return a fixed value.", "/// Group the child contract.\nmod child;\n\n/// Return a fixed value."
    )
    source = source.replace("    1", "    child::value()")
    _write(repo, "new-root/src/lib.rs", source.encode())
    assert changed_rust_errors(repo, base, target_dir=target) == []


def test_feature_selected_source_uses_all_features(rust_repo: tuple[Path, str, Path]) -> None:
    """An actual optional module compiles only because the registered command enables its feature."""
    repo, base, target = rust_repo
    _write(repo, "new-root/Cargo.toml", (_MANIFEST + "\n[features]\nqualified = []\n").encode())
    _write(repo, "new-root/src/child.rs", _VALID.encode())
    source = _VALID.replace(
        "/// Return a fixed value.",
        '/// Expose the optional native contract.\n#[cfg(feature = "qualified")]\npub mod child;\n\n/// Return a fixed value.',
    )
    _write(repo, "new-root/src/lib.rs", source.encode())
    assert changed_rust_errors(repo, base, target_dir=target) == []


def test_inactive_physical_source_cannot_count_as_compiled(rust_repo: tuple[Path, str, Path]) -> None:
    """A false cfg cannot substitute syntax-only presence for actual target measurement."""
    repo, base, target = rust_repo
    _write(repo, "new-root/src/child.rs", _VALID.encode())
    source = _VALID.replace(
        "/// Return a fixed value.",
        "/// Expose an inactive contract.\n#[cfg(any())]\nmod child;\n\n/// Return a fixed value.",
    )
    _write(repo, "new-root/src/lib.rs", source.encode())
    with pytest.raises(RuntimeError, match="not actually enrolled"):
        changed_rust_errors(repo, base, target_dir=target)


def test_real_examples_benches_tests_and_bins_are_all_compiled(rust_repo: tuple[Path, str, Path]) -> None:
    """Actual all-target compilation must emit dependencies for every maintained target family."""
    repo, base, target = rust_repo
    manifest = _MANIFEST + '\n[[bench]]\nname = "qualified"\nharness = false\n'
    _write(repo, "new-root/Cargo.toml", manifest.encode())
    program = "//! Exercise the library contract.\n\n/// Check the library value.\nfn main() {\n    assert_eq!(native_quality::value(), 1);\n}\n"
    for name in ("examples/qualified.rs", "benches/qualified.rs", "src/bin/qualified.rs"):
        _write(repo, "new-root/" + name, program.encode())
    test = "//! Exercise the library contract.\n\n/// Check the library value.\n#[test]\nfn fixed_value() {\n    assert_eq!(native_quality::value(), 1);\n}\n"
    _write(repo, "new-root/tests/qualified.rs", test.encode())
    assert changed_rust_errors(repo, base, target_dir=target) == []
    _write(repo, "new-root/examples/qualified.rs", program.replace("native_quality::value()", "unavailable()").encode())
    assert changed_rust_errors(repo, base, target_dir=target)


def test_separate_excluded_workspace_has_its_own_native_build(rust_repo: tuple[Path, str, Path]) -> None:
    """A parent workspace's success cannot replace an independently excluded source context."""
    repo, base, target = rust_repo
    manifest = _MANIFEST.replace("[workspace]", '[workspace]\nexclude = ["fuzz"]')
    _write(repo, "new-root/Cargo.toml", manifest.encode())
    _write(repo, "new-root/fuzz/Cargo.toml", _MANIFEST.replace("native_quality", "independent_quality").encode())
    _write(repo, "new-root/fuzz/src/lib.rs", _VALID.encode())
    _lock(repo / "new-root/fuzz")
    assert changed_rust_errors(repo, base, target_dir=target) == []
    _write(repo, "new-root/fuzz/src/lib.rs", _VALID.replace("    1", "    true").encode())
    assert changed_rust_errors(repo, base, target_dir=target)


def test_build_scripts_require_qualification_before_execution(rust_repo: tuple[Path, str, Path]) -> None:
    """The actual metadata target reveals a new build script before its side effects can occur."""
    repo, base, target = rust_repo
    sentinel = repo / "executed"
    _write(
        repo,
        "new-root/build.rs",
        (
            '//! Define an operational build script.\n\n/// Write only if executed.\nfn main() { std::fs::write("'
            + str(sentinel)
            + '", "executed").unwrap(); }\n'
        ).encode(),
    )
    with pytest.raises(RuntimeError, match="build scripts"):
        changed_rust_errors(repo, base, target_dir=target)
    assert not sentinel.exists()


def test_application_and_test_bodies_are_not_executed(rust_repo: tuple[Path, str, Path]) -> None:
    """Native static compilation does not run actual application or unit-test side effects."""
    repo, base, target = rust_repo
    source = (
        _VALID
        + '\n/// Exercise a runtime-only failure.\n#[test]\nfn runtime_only() {\n    panic!("Executed a test body");\n}\n'
    )
    _write(repo, "new-root/src/lib.rs", source.encode())
    assert changed_rust_errors(repo, base, target_dir=target) == []


def test_registered_preflight_propagates_real_native_outcomes(rust_repo: tuple[Path, str, Path]) -> None:
    """Execute the actual registered public command with both valid and invalid physical sources."""
    repo, base, target = rust_repo
    name, command, cwd = next(gate for gate in GATES if gate[0] == "changed-rust-static")
    command = [*command, "--repo", str(repo), "--base", base, "--target-dir", str(target)]
    assert run_gate(name, command, cwd)
    _write(repo, "new-root/src/lib.rs", _VALID.replace("    1", "    true").encode())
    assert not run_gate(name, command, cwd)


def test_default_origin_and_absent_cohort_are_not_fallbacks(
    git_repo: tuple[Path, str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Absent Rust needs no toolchain/build dir; unavailable original history still refuses."""
    repo, base = git_repo
    _sparse(repo)
    assert changed_rust_errors(repo, base) == []
    assert main(["--repo", str(repo), "--base", base]) == 0
    assert "other native gates remain separate" in capsys.readouterr().out
    assert main(["--repo", str(repo)]) == 2


@pytest.mark.parametrize(
    "name", ["scpn-control-rs/crates/control-types/src/constants.rs", "scpn-control-rs/Cargo.lock"]
)
def test_real_rust_deletion_is_not_an_absent_cohort(git_repo: tuple[Path, str], name: str) -> None:
    """An actual original Rust deletion still triggers qualification and requires explicit build custody."""
    repo, base = git_repo
    _sparse(repo)
    subprocess.run(["git", "-C", str(repo), "update-index", "--no-skip-worktree", "--", name], check=True)
    with pytest.raises(RuntimeError, match="explicit owned"):
        changed_rust_errors(repo, base)


def test_manifest_only_changes_compile_actual_unchanged_source(rust_repo: tuple[Path, str, Path]) -> None:
    """A Cargo-only change compiles an exact original maintained Rust file through real native metadata."""
    repo, base, target = rust_repo
    name = "scpn-control-rs/crates/control-types/src/constants.rs"
    subprocess.run(["git", "-C", str(repo), "update-index", "--no-skip-worktree", "--", name], check=True)
    _write(repo, name, subprocess.check_output(["git", "-C", str(repo), "show", base + ":" + name]))
    (repo / "new-root/src/lib.rs").unlink()
    _write(repo, "new-root/Cargo.toml", (_MANIFEST + '\n[lib]\npath = "../' + name + '"\n').encode())
    assert changed_rust_errors(repo, base, target_dir=target) == []


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_hidden_materialized_native_carriers_refuse(rust_repo: tuple[Path, str, Path], flag: str) -> None:
    """Native Git index flags cannot hide a materialized toolchain or changed source input."""
    repo, base, target = rust_repo
    subprocess.run(["git", "-C", str(repo), "update-index", flag, "--", "rust-toolchain.toml"], check=True)
    with pytest.raises(RuntimeError, match="hidden Git index"):
        changed_rust_errors(repo, base, target_dir=target)

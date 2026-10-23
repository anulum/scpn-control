# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native Cargo protocol, compiler enrollment and profile refusals.
"""Keep actual Cargo processes while testing incomplete protocol and input custody."""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest
from test_tools.test_source_quality_scope import _write
from test_tools.test_source_quality_scope import git_repo as git_repo

from native_tests.test_rust_source_checks import _VALID
from native_tests.test_rust_source_checks import rust_repo as rust_repo
from tools.rust_source_checks import changed_rust_errors
from tools.rust_source_enrollment import cargo_build_result, read_workspace
from tools.rust_source_profile import _VERSIONS, _native


def _build(repo: Path, target: Path) -> subprocess.CompletedProcess[str]:
    """Execute the actual pinned Cargo command using its already prepared profile and cache."""
    return _native(
        repo / "new-root",
        [
            "cargo",
            "+1.98.0",
            "clippy",
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
        120.0,
        target_dir=target,
    )


def test_actual_metadata_and_dependency_inputs(rust_repo: tuple[Path, str, Path]) -> None:
    """Genuine fresh and cached artifacts retain the declared root and actual compiler inputs."""
    repo, base, target = rust_repo
    assert changed_rust_errors(repo, base, target_dir=target) == []
    workspace = read_workspace(repo, repo / "new-root/Cargo.toml", functools.partial(_native, target_dir=target), 30.0)
    result = _build(repo, target)
    findings, inputs = cargo_build_result(repo, workspace, result)
    assert findings == []
    assert workspace.targets == {repo / "new-root/src/lib.rs"}
    assert workspace.targets <= inputs
    assert any(json.loads(line).get("fresh") is True for line in result.stdout.splitlines())


@pytest.mark.parametrize(
    "fault",
    [
        "members",
        "duplicate",
        "edition",
        "targets",
        "build-script",
        "root",
        "array",
        "package",
        "object",
        "path",
        "symlink",
        "incomplete",
        "metadata-error",
    ],
)
def test_damage_after_actual_cargo_metadata_refuses(
    rust_repo: tuple[Path, str, Path], monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """Native metadata remains the original producer; only a negative protocol mutation is injected."""
    repo, base, target = rust_repo

    def corrupt(
        root: Path, argv: Sequence[str], timeout_s: float, *, target_dir: Path
    ) -> subprocess.CompletedProcess[str]:
        """Return the real command result after damaging one required metadata field."""
        result = _native(root, argv, timeout_s, target_dir=target_dir)
        if "metadata" in argv:
            data = json.loads(result.stdout)
            if fault == "members":
                data["workspace_members"] = []
            elif fault == "duplicate":
                data["workspace_members"].append(data["workspace_members"][0])
            elif fault == "edition":
                data["packages"][0]["edition"] = "2024"
            elif fault == "targets":
                data["packages"][0]["targets"] = []
            elif fault == "build-script":
                data["packages"][0]["targets"][0]["kind"] = ["custom-build"]
            elif fault == "root":
                data["workspace_root"] = str(repo.parent)
            elif fault == "array":
                data["packages"] = "incomplete"
            elif fault == "package":
                data["packages"].append(data["packages"][0])
            elif fault == "object":
                data["packages"][0] = "invalid-object"
            elif fault == "path":
                data["workspace_root"] = "relative"
            elif fault == "symlink":
                link = repo / "alias"
                link.symlink_to(repo / "new-root", target_is_directory=True)
                data["workspace_root"] = str(link)
            elif fault == "incomplete":
                data["packages"] = []
            elif fault == "metadata-error":
                result.returncode = 1
            result.stdout = json.dumps(data)
        return result

    monkeypatch.setattr("tools.rust_source_checks._native", corrupt)
    with pytest.raises(RuntimeError):
        changed_rust_errors(repo, base, target_dir=target)


@pytest.mark.parametrize(
    "fault",
    [
        "footer",
        "filename",
        "source",
        "reason",
        "binding",
        "dependencies",
        "relative-output",
        "dep-symlink",
        "rule",
        "empty",
        "escape",
        "continuation",
        "spacing",
        "stderr",
        "note",
        "dependency-artifact",
    ],
)
def test_damage_after_actual_compilation_cannot_certify_enrollment(
    rust_repo: tuple[Path, str, Path], fault: str
) -> None:
    """Actual completed compiler results cannot bypass missing artifact/argument/source custody."""
    repo, base, target = rust_repo
    assert changed_rust_errors(repo, base, target_dir=target) == []
    workspace = read_workspace(repo, repo / "new-root/Cargo.toml", functools.partial(_native, target_dir=target), 30.0)
    result = _build(repo, target)
    messages = [json.loads(line) for line in result.stdout.splitlines()]
    artifact = next(item for item in messages if item.get("reason") == "compiler-artifact")
    if fault == "footer":
        messages = [item for item in messages if item.get("reason") != "build-finished"]
    elif fault == "filename":
        artifact["filenames"] = []
    elif fault == "source":
        _write(repo, "orphan.rs", _VALID.encode())
        artifact["target"]["src_path"] = str(repo / "orphan.rs")
    elif fault == "reason":
        messages.append({"reason": "unknown-native-message"})
    elif fault == "relative-output":
        artifact["filenames"] = ["relative.rmeta"]
    elif fault == "stderr":
        result.stderr += "\nwarning: injected native diagnostic\n"
    elif fault == "note":
        messages.append({"reason": "compiler-message", "message": {"level": "note", "message": "native note"}})
    elif fault == "dependency-artifact":
        messages.append({"reason": "compiler-artifact", "package_id": "external-dependency"})
    else:
        for path in target.rglob("*.d"):
            text = path.read_text()
            if fault == "dep-symlink":
                original = path.with_suffix(".original")
                path.rename(original)
                path.symlink_to(original)
                continue
            if fault == "binding":
                text = text.replace("CLIPPY_ARGS=-D", "CLIPPY_ARGS=-A")
            elif fault == "dependencies":
                text = text.replace("src/lib.rs", "src/missing.rs")
            else:
                first, rest = text.split("\n", 1)
                if fault == "rule":
                    first = "invalid dependency rule"
                elif fault == "empty":
                    first = first.partition(": ")[0] + ": "
                elif fault == "escape":
                    first += " \\"
                elif fault == "continuation":
                    first = first.replace("src/lib.rs", "\\\nsrc/lib.rs")
                elif fault == "spacing":
                    first = first.replace(": ", ":  ", 1)
                text = first + ("\r\n" if fault == "escape" else "\n") + rest
            path.write_text(text)
    result.stdout = "\n".join(json.dumps(item) for item in messages) + "\n"
    if fault in {"continuation", "spacing", "stderr", "note", "dependency-artifact"}:
        findings, inputs = cargo_build_result(repo, workspace, result)
        assert workspace.targets <= inputs
        assert bool(findings) == (fault == "stderr")
    else:
        with pytest.raises(RuntimeError):
            cargo_build_result(repo, workspace, result)


@pytest.mark.parametrize("version", range(5))
def test_real_native_version_output_rejects_a_mismatched_profile(
    rust_repo: tuple[Path, str, Path], monkeypatch: pytest.MonkeyPatch, version: int
) -> None:
    """Keep real installed tools and change only the expected declaration as a negative control."""
    repo, base, target = rust_repo
    declarations = list(_VERSIONS)
    declarations[version] = (declarations[version][0], "unqualified-version")
    monkeypatch.setattr("tools.rust_source_profile._VERSIONS", declarations)
    with pytest.raises(RuntimeError, match="qualified native version"):
        changed_rust_errors(repo, base, target_dir=target)


@pytest.mark.parametrize("timeout", [True, 0.0, -1.0, float("inf"), float("nan")])
def test_unavailable_resource_bounds_refuse(rust_repo: tuple[Path, str, Path], timeout: float) -> None:
    """A nonfinite or nonpositive bound cannot disable native execution limits."""
    repo, base, target = rust_repo
    with pytest.raises(RuntimeError, match="finite and positive"):
        changed_rust_errors(repo, base, target_dir=target, timeout_s=timeout)


def test_real_absent_tools_and_native_timeout_refuse(
    rust_repo: tuple[Path, str, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Actual process absence/timeouts cannot become a completed empty source measurement."""
    repo, base, target = rust_repo
    with pytest.raises(RuntimeError, match="completed native|successful Git|qualified native version"):
        changed_rust_errors(repo, base, target_dir=target, timeout_s=1e-12)
    tools = tmp_path.parent / (tmp_path.name + "-git-only")
    tools.mkdir()
    git = shutil.which("git")
    assert git is not None
    (tools / "git").symlink_to(git)
    monkeypatch.setenv("PATH", str(tools))
    with pytest.raises(RuntimeError, match="completed native|successful Git|qualified native version"):
        changed_rust_errors(repo, base, target_dir=target)


@pytest.mark.parametrize("config", ["rustfmt.toml", ".clippy.toml", ".cargo/config.toml"])
def test_actual_per_directory_configuration_requires_review(rust_repo: tuple[Path, str, Path], config: str) -> None:
    """An ignored configuration cannot weaken formatting, compiler flags or lint coverage."""
    repo, base, target = rust_repo
    _write(repo, ".gitignore", ("/new-root/" + config + "\n").encode())
    _write(repo, "new-root/" + config, b"# Deliberate unqualified profile configuration.\n")
    with pytest.raises(RuntimeError, match="configuration requires"):
        changed_rust_errors(repo, base, target_dir=target)


def test_actual_profile_and_source_changes_refuse_after_native_results(
    rust_repo: tuple[Path, str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve completed compiler outcomes before physically changing their source binding."""
    repo, base, target = rust_repo

    def mutate(
        root: Path, argv: Sequence[str], timeout_s: float, *, target_dir: Path
    ) -> subprocess.CompletedProcess[str]:
        """Return the genuine native build result, then change the measured file."""
        result = _native(root, argv, timeout_s, target_dir=target_dir)
        if "--message-format=json" in argv:
            _write(repo, "new-root/src/lib.rs", _VALID.replace("    1", "    2").encode())
        return result

    monkeypatch.setattr("tools.rust_source_checks._native", mutate)
    with pytest.raises(RuntimeError, match="changed during"):
        changed_rust_errors(repo, base, target_dir=target)


def test_actual_native_build_inputs_support_escaped_source_names(rust_repo: tuple[Path, str, Path]) -> None:
    """Native Cargo, the Rust parser and dep-info must preserve spaces, hashes, dollars and Unicode."""
    repo, base, target = rust_repo
    path = repo / "new-root/src/lib.rs"
    path.rename(path.with_name("space #$ ž.rs"))
    manifest = (repo / "new-root/Cargo.toml").read_text() + '\n[lib]\npath = "src/space #$ ž.rs"\n'
    _write(repo, "new-root/Cargo.toml", manifest.encode())
    assert changed_rust_errors(repo, base, target_dir=target) == []

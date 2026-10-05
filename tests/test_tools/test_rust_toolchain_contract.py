# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Rust toolchain contract tests
"""Regression tests for exact local and hosted Rust toolchain pins."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pytest import CaptureFixture

from tools.check_rust_toolchain_contract import (
    EXPECTED_WORKFLOWS,
    NIGHTLY_TOOLCHAIN,
    ROOT,
    STABLE_TOOLCHAIN,
    _toolchain_steps,
    check_rust_toolchain_contract,
    main,
)

ACTION_SHA = "6" * 40


def _write_contract(root: Path) -> None:
    """Write all local policy inputs as actual minimal workflow job mappings.

    Parameters
    ----------
    root : pathlib.Path
        Existing physical fixture directory.
    """
    (root / "rust-toolchain.toml").write_text(
        '[toolchain]\nchannel = "1.98.0"\ncomponents = ["clippy", "rustfmt"]\nprofile = "minimal"\n',
        encoding="utf-8",
    )
    for relative_path, (toolchain, count, components) in EXPECTED_WORKFLOWS.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        components_line = f"\n          components: {components}" if components is not None else ""
        steps = "\n".join(
            f"      - uses: dtolnay/rust-toolchain@{ACTION_SHA}\n        with:\n"
            f"          toolchain: {toolchain}{components_line}"
            for _ in range(count)
        )
        path.write_text(f"jobs:\n  check:\n    steps:\n{steps}\n", encoding="utf-8")


def test_live_repository_contract_passes() -> None:
    """The checked-in stable and nightly pins form one exact contract."""
    assert check_rust_toolchain_contract(ROOT) == []


def test_toolchain_step_parser_stops_at_next_step(tmp_path: Path) -> None:
    """A later step cannot accidentally supply an omitted toolchain value."""
    workflow = tmp_path / "workflow.yml"
    workflow.write_text(
        f"jobs:\n  check:\n    steps:\n      - uses: dtolnay/rust-toolchain@{ACTION_SHA}\n      - run: echo next\n        toolchain: fake\n",
        encoding="utf-8",
    )
    assert _toolchain_steps(workflow) == [(ACTION_SHA, None, None)]

    workflow.write_text(
        f"jobs:\n  check:\n    steps:\n      - uses: dtolnay/rust-toolchain@{ACTION_SHA}\n", encoding="utf-8"
    )
    assert _toolchain_steps(workflow) == [(ACTION_SHA, None, None)]


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        ('channel = "stable"', "rust-toolchain.toml drift"),
        ('components = ["rustfmt", "clippy"]', "rust-toolchain.toml drift"),
        ('profile = "default"', "rust-toolchain.toml drift"),
    ],
)
def test_contract_rejects_toolchain_table_drift(tmp_path: Path, replacement: str, match: str) -> None:
    """The local toolchain table is exact, including component order."""
    _write_contract(tmp_path)
    path = tmp_path / "rust-toolchain.toml"
    lines = path.read_text(encoding="utf-8").splitlines()
    key = replacement.split(" =", maxsplit=1)[0]
    path.write_text("\n".join(replacement if line.startswith(key) else line for line in lines) + "\n", encoding="utf-8")
    assert any(match in error for error in check_rust_toolchain_contract(tmp_path))


def test_contract_rejects_missing_or_extra_toolchain_tables(tmp_path: Path) -> None:
    """Malformed and expanded toolchain files fail closed."""
    _write_contract(tmp_path)
    path = tmp_path / "rust-toolchain.toml"
    path.write_text('[other]\nvalue = "x"\n', encoding="utf-8")
    assert check_rust_toolchain_contract(tmp_path) == ["rust-toolchain.toml requires a [toolchain] table"]

    _write_contract(tmp_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('[other]\nvalue = "x"\n')
    assert "may contain only" in " ".join(check_rust_toolchain_contract(tmp_path))


def test_contract_rejects_invalid_toml_and_missing_file(tmp_path: Path) -> None:
    """Unreadable or invalid canonical toolchain input fails closed."""
    assert "cannot read" in check_rust_toolchain_contract(tmp_path)[0]
    (tmp_path / "rust-toolchain.toml").write_text("[", encoding="utf-8")
    assert "cannot read" in check_rust_toolchain_contract(tmp_path)[0]


def test_contract_rejects_workflow_count_pin_and_version_drift(tmp_path: Path) -> None:
    """Every expected action must exist with a full SHA and exact channel."""
    _write_contract(tmp_path)
    stable_path = tmp_path / ".github/workflows/ci-native-polyglot.yml"
    text = stable_path.read_text(encoding="utf-8")
    text = text.replace(ACTION_SHA, "main", 1).replace(f"toolchain: {STABLE_TOOLCHAIN}", "toolchain: stable", 1)
    stable_path.write_text(text, encoding="utf-8")
    fuzz_path = tmp_path / ".github/workflows/fuzz-nightly.yml"
    fuzz_path.write_text("jobs: {}\n", encoding="utf-8")

    errors = check_rust_toolchain_contract(tmp_path)
    assert any("not pinned to a full SHA" in error for error in errors)
    assert any("expected toolchain 1.98.0, got stable" in error for error in errors)
    assert any("expected 2 Rust toolchain steps, found 0" in error for error in errors)


def test_contract_rejects_stable_component_drift(tmp_path: Path) -> None:
    """Stable hosted jobs must install the canonical components up front."""
    _write_contract(tmp_path)
    path = tmp_path / ".github/workflows/ci-native-polyglot.yml"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("components: rustfmt, clippy", "components: rustfmt", 1), encoding="utf-8")
    assert any(
        "expected components rustfmt, clippy, got rustfmt" in error for error in check_rust_toolchain_contract(tmp_path)
    )


def test_contract_reports_missing_workflow(tmp_path: Path) -> None:
    """A missing expected workflow is a named fail-closed error."""
    _write_contract(tmp_path)
    (tmp_path / ".github/workflows/pre-commit.yml").unlink()
    assert any(
        "cannot read .github/workflows/pre-commit.yml" in error for error in check_rust_toolchain_contract(tmp_path)
    )


def test_main_reports_pass_and_failure(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """The CLI exposes stable success and non-zero failure results."""
    _write_contract(tmp_path)
    assert main(["--root", str(tmp_path)]) == 0
    assert STABLE_TOOLCHAIN in capsys.readouterr().out

    (tmp_path / "rust-toolchain.toml").unlink()
    assert main(["--root", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "FAIL:" in output
    assert NIGHTLY_TOOLCHAIN not in output


@pytest.fixture
def physical_contract(tmp_path: Path) -> Path:
    """Create real inputs and a byte-identical standalone policy script.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-owned physical directory.

    Returns
    -------
    pathlib.Path
        Actual root usable through the public API and script-relative CLI.
    """
    root = tmp_path / "rust-policy"
    (root / "tools").mkdir(parents=True)
    _write_contract(root)
    source = ROOT / "tools/check_rust_toolchain_contract.py"
    target = root / "tools/check_rust_toolchain_contract.py"
    shutil.copyfile(source, target)
    assert hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(source.read_bytes()).digest()
    return root


def _physical_cli(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the real script from an unrelated cwd, optionally measuring its source.

    Parameters
    ----------
    root : pathlib.Path
        Physical repository root containing the copied script.
    *args : str
        Public argparse arguments.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Actual process exit and UTF-8 output; no globals/providers are replaced.
    """
    command = [sys.executable]
    config = os.environ.get("SCPN_RUST_POLICY_COVERAGE_RC")
    if config:
        command += ["-m", "coverage", "run", "--rcfile=" + config, "--parallel-mode"]
    command += [str(root / "tools/check_rust_toolchain_contract.py"), *args]
    return subprocess.run(command, cwd=root.parent, capture_output=True, text=True, encoding="utf-8", timeout=30)


@pytest.mark.parametrize(
    ("declaration", "accepted"),
    [
        (
            f"- name: Rust\n        uses: dtolnay/rust-toolchain@{ACTION_SHA}\n        with: {{toolchain: 1.98.0, components: 'rustfmt, clippy'}}",
            True,
        ),
        (
            f"- uses: dtolnay/rust-toolchain@{ACTION_SHA}\n        env: {{toolchain: 1.98.0, components: 'rustfmt, clippy'}}",
            False,
        ),
        (
            f"- run: |\n          - uses: dtolnay/rust-toolchain@{ACTION_SHA}\n            with:\n              toolchain: 1.98.0\n              components: rustfmt, clippy",
            False,
        ),
        (
            f"- uses: dtolnay/rust-toolchain@{ACTION_SHA}\n      - run: echo next\n        with: {{toolchain: 1.98.0, components: 'rustfmt, clippy'}}",
            False,
        ),
        (f"- uses: dtolnay/rust-toolchain@{ACTION_SHA}\n        with: {{components: 'rustfmt, clippy'}}", False),
        (f"- uses: dtolnay/rust-toolchain@{ACTION_SHA}\n        with: {{toolchain: 1.98.0}}", False),
        (f"- uses: dtolnay/rust-toolchain@{ACTION_SHA}", False),
    ],
)
def test_real_policy_action_input_binding(physical_contract: Path, declaration: str, accepted: bool) -> None:
    """Bind Rust counts and settings to actual step mappings, never surrounding text.

    Parameters
    ----------
    physical_contract : pathlib.Path
        Real healthy repository fixture.
    declaration : str
        Replacement job step declarations, including actual run/env text.
    accepted : bool
        Expected declaration disposition.
    """
    path = physical_contract / ".github/workflows/ci-rust-benchmark.yml"
    path.write_text("jobs:\n  check:\n    steps:\n      " + declaration + "\n", encoding="utf-8")
    assert (check_rust_toolchain_contract(physical_contract) == []) is accepted
    child = _physical_cli(physical_contract)
    assert child.returncode == int(not accepted)
    assert bool("contract passed" in child.stdout) is accepted and not child.stderr


@pytest.mark.parametrize(
    "text",
    [
        "",
        "[]",
        "!custom {}",
        "{}",
        "jobs: []",
        "jobs: !custom {}",
        "jobs: {check: []}",
        "jobs: {check: !custom {}}",
        "jobs: {check: {steps: {}}}",
        "jobs: {check: {steps: !custom []}}",
        "jobs: {check: {steps: [scalar]}}",
        "jobs: {check: {steps: [{uses: []}]}}",
        "jobs: {check: {steps: [{uses: !!int 1}]}}",
        f"jobs: {{check: {{steps: [{{uses: 'dtolnay/rust-toolchain@{ACTION_SHA}', with: []}}]}}}}",
        f"jobs: {{check: {{steps: [{{uses: 'dtolnay/rust-toolchain@{ACTION_SHA}', with: {{toolchain: []}}}}]}}}}",
        f"jobs: {{check: {{steps: [{{uses: 'dtolnay/rust-toolchain@{ACTION_SHA}', with: {{toolchain: 1.98.0, components: []}}}}]}}}}",
        "jobs: {}\njobs: {}",
        "jobs: {check: {}, check: {}}",
        "jobs: {<<: {}}",
        "? [jobs]\n: {}",
        "!!int 1: {}",
        "jobs: *missing",
        "jobs: [",
        "jobs: {}\n---\njobs: {}",
    ],
)
def test_real_policy_malformed_workflows(physical_contract: Path, text: str) -> None:
    """Refuse real malformed, duplicate, merged, or wrongly typed relevant YAML.

    Parameters
    ----------
    physical_contract : pathlib.Path
        Real healthy repository fixture.
    text : str
        Malformed replacement workflow bytes after UTF-8 encoding.
    """
    path = physical_contract / ".github/workflows/ci-rust-benchmark.yml"
    path.write_text(text, encoding="utf-8")
    errors = check_rust_toolchain_contract(physical_contract)
    assert any("cannot parse .github/workflows/ci-rust-benchmark.yml:" in error for error in errors)
    child = _physical_cli(physical_contract)
    assert child.returncode == 1 and not child.stderr and "contract passed" not in child.stdout


def test_real_policy_aliases_extra_jobs_and_tags(physical_contract: Path) -> None:
    """Support ordinary aliases/named jobs without constructing an unrelated tag.

    Parameters
    ----------
    physical_contract : pathlib.Path
        Real healthy repository fixture.
    """
    sentinel = physical_contract / "constructor-executed"
    expression = f"__import__('pathlib').Path({str(sentinel)!r}).touch()"
    path = physical_contract / ".github/workflows/ci-rust-benchmark.yml"
    path.write_text(
        "on: workflow_call\nextra: !!python/object/apply:builtins.eval\n  - "
        + repr(expression)
        + "\nsettings: &settings {toolchain: 1.98.0, components: 'rustfmt, clippy'}\n"
        + f"jobs:\n  external: {{uses: other/reusable@{ACTION_SHA}}}\n  check:\n    steps:\n"
        + f"      - uses: actions/checkout@{ACTION_SHA}\n"
        + f"      - name: Rust\n        uses: dtolnay/rust-toolchain@{ACTION_SHA}\n        with: *settings\n",
        encoding="utf-8",
    )
    assert check_rust_toolchain_contract(physical_contract) == []
    assert _physical_cli(physical_contract).returncode == 0 and not sentinel.exists()
    result = check_rust_toolchain_contract(physical_contract)
    result.append("caller annotation")
    assert check_rust_toolchain_contract(physical_contract) == []


@pytest.mark.parametrize("input_name", ["rust-toolchain.toml", ".github/workflows/ci-rust-benchmark.yml"])
@pytest.mark.parametrize("fault", ["directory", "invalid_utf8"])
def test_real_policy_native_input_failures(physical_contract: Path, input_name: str, fault: str) -> None:
    """Preserve native UTF-8 errors and authored read findings on actual files.

    Parameters
    ----------
    physical_contract : pathlib.Path
        Real healthy repository fixture.
    input_name : str
        Local or workflow input receiving the fault.
    fault : str
        Directory read or invalid UTF-8 byte content.
    """
    path = physical_contract / input_name
    if fault == "directory":
        path.unlink()
        path.mkdir()
        assert any("cannot read" in error for error in check_rust_toolchain_contract(physical_contract))
        child = _physical_cli(physical_contract)
        assert child.returncode == 1 and not child.stderr
    else:
        path.write_bytes(b"\xffinvalid UTF-8")
        with pytest.raises(UnicodeError):
            check_rust_toolchain_contract(physical_contract)
        child = _physical_cli(physical_contract)
        assert child.returncode != 0 and "UnicodeDecodeError" in child.stderr
    assert "contract passed" not in child.stdout


def test_real_policy_public_cli_and_table_domains(physical_contract: Path) -> None:
    """Exercise real argparse exits, relative roots, TOML domains and native signatures.

    Parameters
    ----------
    physical_contract : pathlib.Path
        Real healthy repository fixture.
    """
    assert _physical_cli(physical_contract, "--root", physical_contract.name).returncode == 0
    assert _physical_cli(physical_contract, "--help").returncode == 0
    assert _physical_cli(physical_contract, "--unknown").returncode == 2
    assert _physical_cli(physical_contract, "--root").returncode == 2
    path = physical_contract / "rust-toolchain.toml"
    for text in [
        "[",
        "[other]\nx=1\n",
        "toolchain = true\n",
        "[toolchain]\nchannel = 1\n",
        path.read_text() + "\n[other]\nx=1\n",
    ]:
        path.write_text(text, encoding="utf-8")
        assert check_rust_toolchain_contract(physical_contract)
        assert _physical_cli(physical_contract).returncode == 1
    path.unlink()
    assert _physical_cli(physical_contract).returncode == 1
    with pytest.raises(TypeError):
        sys.modules[check_rust_toolchain_contract.__module__].check_rust_toolchain_contract(unexpected_argument=True)
    with pytest.raises(TypeError):
        sys.modules[main.__module__].main(unexpected_argument=True)

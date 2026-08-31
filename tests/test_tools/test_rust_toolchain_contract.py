# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Rust toolchain contract tests
"""Regression tests for exact local and hosted Rust toolchain pins."""

from __future__ import annotations

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
        f"steps:\n  - uses: dtolnay/rust-toolchain@{ACTION_SHA}\n  - run: echo next\n    toolchain: fake\n",
        encoding="utf-8",
    )
    assert _toolchain_steps(workflow) == [(ACTION_SHA, None, None)]

    workflow.write_text(f"steps:\n  - uses: dtolnay/rust-toolchain@{ACTION_SHA}\n", encoding="utf-8")
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

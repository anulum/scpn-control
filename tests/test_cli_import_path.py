# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Import-path contract of the command-line interface.
"""The CLI puts a source checkout on the import path when a command runs, never on import.

Importing ``scpn_control.cli`` must leave the caller's import path alone. A
command run from a source checkout puts the checkout first, once, so the
validation commands can import ``validation.*``. A module root without a
``validation`` package, as in an installed wheel where the root is the
interpreter's library directory, is never added. Every check strips the
checkout first: with it present, "insert if absent" would leave the path
unchanged either way.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

import scpn_control.cli as cli

ROOT = Path(__file__).resolve().parents[1]


def _without_checkout() -> list[str]:
    """Return the current import path without the checkout root."""
    return [entry for entry in sys.path if Path(entry or ".").resolve() != ROOT]


def test_importing_the_cli_leaves_the_import_path_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Executing the CLI module adds nothing to a path without the checkout."""
    stripped = _without_checkout()
    monkeypatch.setattr(sys, "path", list(stripped))
    spec = importlib.util.spec_from_file_location("scpn_control.cli", ROOT / "src" / "scpn_control" / "cli.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._REPO_ROOT == ROOT
    assert sys.path == stripped


def test_a_command_run_from_the_checkout_puts_it_first_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two command runs from the checkout insert it once, ahead of the caller's entries."""
    stripped = _without_checkout()
    monkeypatch.setattr(sys, "path", list(stripped))
    runner = CliRunner()
    assert runner.invoke(cli.main, ["info", "--json-out"]).exit_code == 0
    assert sys.path == [str(ROOT), *stripped]
    assert runner.invoke(cli.main, ["info", "--json-out"]).exit_code == 0
    assert sys.path == [str(ROOT), *stripped]


def test_a_root_without_a_validation_package_is_never_added(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A module root without ``validation/__init__.py`` stays off the path; with the package it is added."""
    stripped = _without_checkout()
    monkeypatch.setattr(sys, "path", list(stripped))
    monkeypatch.setattr(cli, "_REPO_ROOT", tmp_path)
    runner = CliRunner()
    assert runner.invoke(cli.main, ["info", "--json-out"]).exit_code == 0
    assert sys.path == stripped
    (tmp_path / "validation").mkdir()
    (tmp_path / "validation" / "__init__.py").write_text("", encoding="utf-8")
    assert runner.invoke(cli.main, ["info", "--json-out"]).exit_code == 0
    assert sys.path == [str(tmp_path), *stripped]

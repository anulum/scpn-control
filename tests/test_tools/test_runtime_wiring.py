# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Runtime-wiring checker tests.

"""Tests for the source-module runtime wiring guard."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "check_runtime_wiring.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_runtime_wiring", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def wiring_tool() -> ModuleType:
    """Load the wiring checker from the checked-out repository."""
    return _load_tool()


def test_relative_imports_resolve_from_package_init(wiring_tool: Any) -> None:
    """Package ``__init__`` relative imports must stay inside that package."""
    resolved = wiring_tool._resolve_relative(
        "scpn_control.studio",
        1,
        "adapters",
        is_package=True,
    )

    assert resolved == "scpn_control.studio.adapters"


def test_relative_imports_resolve_from_normal_module(wiring_tool: Any) -> None:
    """Normal module relative imports resolve from the containing package."""
    same_package = wiring_tool._resolve_relative(
        "scpn_control.control.nmpc_controller",
        1,
        "realtime_efit",
        is_package=False,
    )
    parent_package = wiring_tool._resolve_relative(
        "scpn_control.control.nmpc_controller",
        2,
        "core",
        is_package=False,
    )

    assert same_package == "scpn_control.control.realtime_efit"
    assert parent_package == "scpn_control.core"


def test_live_source_tree_has_no_orphan_modules(wiring_tool: Any) -> None:
    """Every source module is referenced from a package, test, tool, or pipeline file."""
    orphans, total = wiring_tool.find_orphans()

    assert total >= 150
    assert orphans == []


def test_main_fails_closed_when_orphans_are_reported(
    wiring_tool: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI must fail if a future source module loses repository wiring."""
    monkeypatch.setattr(wiring_tool, "find_orphans", lambda: (["scpn_control.detached"], 162))

    assert wiring_tool.main([]) == 1
    assert "scpn_control.detached" in capsys.readouterr().out

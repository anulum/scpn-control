# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Coverage pragma checker tests.

"""Tests for the coverage-pragma reason gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "check_coverage_pragmas.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_coverage_pragmas", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def pragma_tool() -> ModuleType:
    """Load the checker from the checked-out repository."""
    return _load_tool()


def test_reason_parser_accepts_hyphen_colon_and_dash_reasons(pragma_tool: Any) -> None:
    """Reasoned coverage pragmas may use common separator styles."""
    assert pragma_tool._is_reasoned(" - optional dependency path") is True
    assert pragma_tool._is_reasoned(": defensive guard") is True
    assert pragma_tool._is_reasoned(" — unreachable numerical branch") is True
    assert pragma_tool._is_reasoned("") is False
    assert pragma_tool._is_reasoned(")") is False


def test_checker_reports_bare_pragmas(tmp_path: Path, pragma_tool: Any) -> None:
    """Bare pragmas are returned with path, line, and source text."""
    source = tmp_path / "example.py"
    source.write_text(
        "\n".join(
            [
                "def optional_path():  # pragma: no cover - optional dependency path",
                "    return 1",
                "def bare_path():  # pragma: no cover",
                "    return 2",
            ]
        ),
        encoding="utf-8",
    )

    violations = pragma_tool.find_unreasoned_pragmas([source])

    assert len(violations) == 1
    assert violations[0].line == 3
    assert violations[0].text == "def bare_path():  # pragma: no cover"


def test_live_source_tree_has_no_unreasoned_pragmas(pragma_tool: Any) -> None:
    """Every source coverage exclusion must carry a one-line reason."""
    violations = pragma_tool.find_unreasoned_pragmas([REPO_ROOT / "src" / "scpn_control"])

    assert violations == []


def test_main_fails_closed_when_unreasoned_pragmas_are_reported(
    pragma_tool: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI must fail if a future bare pragma enters the source tree."""
    violation = pragma_tool.CoveragePragmaViolation(
        path="src/scpn_control/example.py",
        line=10,
        text="if optional:  # pragma: no cover",
    )
    monkeypatch.setattr(pragma_tool, "find_unreasoned_pragmas", lambda _paths: [violation])

    assert pragma_tool.main([]) == 1
    assert "src/scpn_control/example.py:10" in capsys.readouterr().out

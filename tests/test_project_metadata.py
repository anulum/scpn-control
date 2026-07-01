# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Project Metadata Tests
"""Regression tests for repository metadata and packaging configuration."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Protocol, cast


class _TomlModule(Protocol):
    def loads(self, data: str, /) -> dict[str, Any]:
        """Parse TOML text into a metadata dictionary."""


def _toml_module() -> _TomlModule:
    """Return the stdlib TOML parser, or the Python 3.10 backport."""
    module_name = "tomllib" if sys.version_info >= (3, 11) else "tomli"
    return cast("_TomlModule", importlib.import_module(module_name))


ROOT = Path(__file__).resolve().parents[1]
TOML = _toml_module()


def _load_pyproject() -> dict[str, Any]:
    """Return the parsed project metadata from the repository pyproject."""
    return TOML.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_facility_optional_extra_declares_mdsplus_thin_client() -> None:
    """Require the facility extra to expose the MDSplus thin client."""
    pyproject = _load_pyproject()

    project = cast("dict[str, Any]", pyproject["project"])
    extras = cast("dict[str, list[str]]", project["optional-dependencies"])
    assert "facility" in extras
    assert "mdsthin>=1.6.3" in extras["facility"]
    assert "mdsthin>=1.6.3" in extras["all"]


def test_mypy_optional_overrides_do_not_hide_removed_first_party_modules() -> None:
    """Keep optional mypy imports aligned with live repository modules."""
    pyproject = _load_pyproject()
    tool_config = cast("dict[str, Any]", pyproject["tool"])
    mypy_config = cast("dict[str, Any]", tool_config["mypy"])
    overrides = cast("list[dict[str, object]]", mypy_config["overrides"])

    override_modules: set[str] = set()
    for override in overrides:
        configured_modules = override.get("module")
        if isinstance(configured_modules, str):
            override_modules.add(configured_modules)
        elif isinstance(configured_modules, list):
            override_modules.update(module for module in configured_modules if isinstance(module, str))
        else:
            msg = "mypy override entries must declare module as a string or string list"
            raise AssertionError(msg)

    assert "director_module" not in override_modules

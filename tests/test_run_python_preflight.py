# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Python preflight wrapper tests.
"""Regression tests for the fast Python preflight command list."""

from __future__ import annotations

from tools.run_python_preflight import _build_checks


def _commands_by_name(*, skip_version_metadata: bool = False) -> dict[str, list[str]]:
    """Return preflight command lists keyed by check name."""

    return dict(
        _build_checks(
            skip_version_metadata=skip_version_metadata,
            skip_notebook_quality=True,
            skip_threshold_smoke=True,
            skip_mypy=True,
        )
    )


def test_preflight_uses_live_project_metadata_test() -> None:
    """The version metadata gate points at the live project metadata test."""

    commands = _commands_by_name()
    command = commands["Version metadata consistency"]

    assert "tests/test_project_metadata.py" in command
    assert "tests/test_version_metadata.py" not in command


def test_preflight_skip_version_metadata_removes_project_metadata_gate() -> None:
    """The skip flag removes the metadata test from the command list."""

    commands = _commands_by_name(skip_version_metadata=True)

    assert "Version metadata consistency" not in commands

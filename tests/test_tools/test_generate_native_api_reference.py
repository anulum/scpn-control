# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native API reference generation contracts.

"""Tests for the generated native C and Lean API reference."""

from __future__ import annotations

from tools import generate_native_api_reference as generator


def test_render_includes_all_versioned_legacy_and_lean_declarations() -> None:
    """The generated reference covers both ABI generations and Lean scope."""
    rendered = generator.render()

    assert rendered.count("### `scpn_solver_") == 5
    assert "### `create_solver`" in rendered
    assert "### `destroy_solver`" in rendered
    assert "### `pulsed_fsm_eventually_returns_to_idle`" in rendered
    assert "not evidence of continuous plant or plasma safety" in rendered


def test_checked_reference_is_current() -> None:
    """The checked-in Markdown is byte-current with normative sources."""
    assert generator.main(["--check"]) == 0

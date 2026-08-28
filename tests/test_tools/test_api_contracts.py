# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Sotek. All rights reserved.
# © Code 2020–2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Project: SCPN Control
# Description: Cross-language API inventory and contract tests.
"""Tests for deterministic cross-language API ownership classification."""

from __future__ import annotations

from tools import check_api_contracts


def test_api_inventory_has_disjoint_complete_python_classification() -> None:
    """Every Python candidate belongs to exactly one ownership class."""
    inventory = check_api_contracts.build_inventory()
    python = inventory["python"]

    assert sum(python["classifications"].values()) == python["candidate_count"]
    assert python["stable_export_count"] == 44
    assert len(inventory["c"]["symbols"]) == 10
    assert len(inventory["lean"]["symbols"]) == 9


def test_committed_api_contract_registry_is_current() -> None:
    """The committed registry matches the live cross-language inventory."""
    assert check_api_contracts.main([]) == 0
